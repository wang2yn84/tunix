# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI entry point for agentic GRPO training recipes.

Handles both single-turn (e.g. DeepScaleR) and multi-turn (e.g. DeepSWE)
agentic GRPO workflows.  Everything that differs between recipes is expressed
in the YAML config; this file adds no recipe-specific logic.

Usage::

    # DeepScaleR (single-turn math reasoning)
    python -m tunix.cli.agentic_grpo_main \\
        examples/deepscaler/configs/deepseek_r1_distill_qwen_1.5b.yaml

    # DeepSWE (multi-turn software engineering)
    python -m tunix.cli.agentic_grpo_main \\
        examples/deepswe/configs/qwen3_32b.yaml

    # Override individual keys
    python -m tunix.cli.agentic_grpo_main \\
        examples/deepscaler/configs/deepseek_r1_distill_qwen_1.5b.yaml \\
        agentic_grpo_config.num_generations=4 \\
        rl_training_config.max_steps=100
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from tunix.cli import config as config_lib
from tunix.cli import grpo_main
from tunix.cli.utils import data as data_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib
from tunix.rl.rollout import base_rollout


_PATHWAYS_BNS = flags.DEFINE_string(
    "pathways_bns", None, "BNS address of the Pathways server."
)


class AgenticGrpoPipeline(grpo_main.GrpoPipeline):
  """Extends GrpoPipeline with agentic GRPO capabilities.

  Supports:
  - Single-turn and multi-turn agentic rollouts via GRPOLearner.
  - Auto device-split meshes (rollout on first half, trainer on second half).
  - Multi-turn KV cache sizing.
  - vLLM max_batched_tokens computed as (max_num_seqs * kv_cache_size) // 4.
  - Configurable chat parser (default or qwen).
  - Dynamic agent/env class loading from dotted Python paths.
  - Optional Kubernetes environment setup.
  """

  # ------------------------------------------------------------------
  # Mesh
  # ------------------------------------------------------------------

  def create_role_to_mesh(self):
    split_cfg = self.config.get("split_mesh_config") or {}
    if not split_cfg.get("enabled", False):
      return super().create_role_to_mesh()

    devices = jax.devices()
    split = split_cfg.get("split", len(devices) // 2)

    num_kv_heads = split_cfg.get("rollout_num_kv_heads", split)
    rollout_tp = int(np.gcd(split, num_kv_heads))
    rollout_fsdp = split // rollout_tp
    rollout_devices = np.array(devices[:split]).reshape(rollout_fsdp, rollout_tp)
    rollout_mesh = jax.sharding.Mesh(rollout_devices, axis_names=("fsdp", "tp"))

    agentic_cfg = self.config.get("agentic_grpo_config", {})
    num_generations = agentic_cfg.get("num_generations", 2)
    rl_train_cfg = self.config.get("rl_training_config", {})
    train_micro_bs = rl_train_cfg.get("train_micro_batch_size", 1)
    train_fsdp = int(np.gcd(split, train_micro_bs * num_generations))
    train_tp = split // train_fsdp
    train_devices = np.array(devices[split:]).reshape(train_fsdp, train_tp)
    trainer_mesh = jax.sharding.Mesh(train_devices, axis_names=("fsdp", "tp"))

    logging.info("Split mesh — trainer: %s  rollout: %s", trainer_mesh, rollout_mesh)
    return {
        rl_cluster_lib.Role.ACTOR: trainer_mesh,
        rl_cluster_lib.Role.REFERENCE: trainer_mesh,
        rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
    }

  # ------------------------------------------------------------------
  # Rollout config  (extends parent, adds multi-turn KV cache + vllm)
  # ------------------------------------------------------------------

  def create_rollout_config(self) -> base_rollout.RolloutConfig:
    rollout_cfg = self.config["rollout_config"]
    engine = self.config.get("rollout_engine", "vanilla")
    agentic_cfg = self.config.get("agentic_grpo_config", {})

    max_prompt = rollout_cfg.get("max_prompt_length", 2048)
    max_response = rollout_cfg.get("total_generation_steps", 8192)

    # KV cache: multi-turn uses context_ratio * max_turns to account for
    # accumulated conversation history; single-turn adds a small buffer.
    max_turns = agentic_cfg.get("max_turns", 1)
    context_ratio = agentic_cfg.get("context_ratio", 1)
    if max_turns > 1:
      kv_cache_size = max_prompt + max_response * context_ratio * max_turns
    else:
      kv_cache_size = max_prompt + max_response + 256

    logging.info("kv_cache_size: %d", kv_cache_size)

    # Base fields (same as parent, but kv_cache_size computed above)
    base = dict(
        max_prompt_length=max_prompt,
        kv_cache_size=kv_cache_size,
        temperature=rollout_cfg.get("temperature", 0.8),
        top_p=rollout_cfg.get("top_p"),
        top_k=rollout_cfg.get("top_k"),
        return_logprobs=True,
        max_tokens_to_generate=max_response,
    )

    # Engine-specific extras
    if engine == "sglang_jax":
      sglang_cfg = self.config.get("sglang_jax_config", {})
      model_version = self.config.get("actor_model_config", {}).get(
          "model_id", ""
      )
      extra = dict(
          rollout_sglang_jax_model_version=sglang_cfg.get(
              "model_version", model_version
          ),
          rollout_sglang_jax_mem_fraction_static=sglang_cfg.get(
              "mem_fraction_static", 0.8
          ),
          rollout_sglang_jax_init_with_random_weights=sglang_cfg.get(
              "init_with_random_weights", True
          ),
          rollout_sglang_jax_disable_radix_cache=sglang_cfg.get(
              "disable_radix_cache", True
          ),
          rollout_sglang_jax_enable_deterministic_sampling=sglang_cfg.get(
              "enable_deterministic_sampling", False
          ),
          rollout_sglang_jax_chunked_prefill_size=sglang_cfg.get(
              "chunked_prefill_size", 2048
          ),
          rollout_sglang_jax_max_running_requests=sglang_cfg.get(
              "max_running_requests", agentic_cfg.get("max_concurrency", 768)
          ),
          rollout_sglang_jax_page_size=sglang_cfg.get("page_size", 128),
          rollout_sglang_jax_use_sort_for_toppk_minp=sglang_cfg.get(
              "use_sort_for_toppk_minp", False
          ),
      )
    elif engine == "vllm":
      vllm_cfg = self.config.get("vllm_config", {})
      role_to_mesh = self.create_role_to_mesh()
      rollout_mesh = role_to_mesh[rl_cluster_lib.Role.ROLLOUT]
      rollout_shape = rollout_mesh.devices.shape
      max_num_seqs = rollout_cfg.get(
          "rollout_vllm_max_num_seqs",
          vllm_cfg.get("max_num_seqs", 768),
      )
      # max_batched_tokens = (max_num_seqs * kv_cache_size) // 4
      max_batched_tokens = rollout_cfg.get(
          "rollout_vllm_max_num_batched_tokens",
          vllm_cfg.get(
              "max_num_batched_tokens",
              (max_num_seqs * kv_cache_size) // 4,
          ),
      )
      model_version = self.config.get("actor_model_config", {}).get(
          "model_id", ""
      )
      os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
      extra = dict(
          rollout_vllm_model_version=vllm_cfg.get("model_version", model_version),
          rollout_vllm_hbm_utilization=vllm_cfg.get("hbm_utilization", 0.4),
          rollout_vllm_tpu_backend_type=vllm_cfg.get("tpu_backend_type", "jax"),
          rollout_vllm_server_mode=vllm_cfg.get("server_mode", True),
          rollout_vllm_async_scheduling=vllm_cfg.get("async_scheduling", True),
          tensor_parallel_size=(
              rollout_shape[1] if len(rollout_shape) > 1 else 1
          ),
          data_parallel_size=rollout_shape[0],
          rollout_vllm_max_num_seqs=max_num_seqs,
          rollout_vllm_max_num_batched_tokens=max_batched_tokens,
          rollout_vllm_kwargs=vllm_cfg.get(
              "kwargs",
              {
                  "kv_cache_metrics": True,
                  "disable_log_stats": False,
                  "enable_prefix_caching": True,
              },
          ),
      )
    else:
      extra = {}

    valid_fields = {f.name for f in dataclasses.fields(base_rollout.RolloutConfig)}
    filtered = {k: v for k, v in {**base, **extra}.items() if k in valid_fields}
    return base_rollout.RolloutConfig(**filtered)

  # ------------------------------------------------------------------
  # GRPOConfig
  # ------------------------------------------------------------------

  def _create_grpo_config(self) -> GRPOConfig:
    cfg = dict(self.config.get("agentic_grpo_config", {}))

    # episode_timeout: computed from per_turn_timeout_secs * max_turns
    # when not set explicitly.
    if "episode_timeout" not in cfg:
      per_turn = cfg.pop("per_turn_timeout_secs", None)
      max_turns = cfg.get("max_turns", 1)
      if per_turn is not None:
        cfg["episode_timeout"] = per_turn * max_turns

    # max_response_length maps from rollout_config.total_generation_steps
    if "max_response_length" not in cfg:
      cfg["max_response_length"] = self.config["rollout_config"].get(
          "total_generation_steps", 8192
      )

    # Strip keys that are not GRPOConfig fields
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    # Also strip agentic-specific helper keys not in the dataclass
    cfg.pop("max_turns", None)
    cfg.pop("context_ratio", None)
    filtered = {k: v for k, v in cfg.items() if k in valid}
    return GRPOConfig(**filtered)

  # ------------------------------------------------------------------
  # Chat parser
  # ------------------------------------------------------------------

  def _create_chat_parser(self, tokenizer: Any) -> Any:
    parser_cfg = self.config.get("chat_parser_config", {})
    parser_type = parser_cfg.get("type", "default")
    if parser_type == "qwen":
      return chat_parser_lib.QwenChatTemplateParser(tokenizer)
    return chat_parser_lib.DefaultChatTemplateParser(tokenizer)

  # ------------------------------------------------------------------
  # Dynamic class loading
  # ------------------------------------------------------------------

  def _load_class_from_path(self, dotted_path: str) -> type:
    """Load a Python class from a dotted module path (e.g. 'a.b.c.MyClass')."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

  # ------------------------------------------------------------------
  # Dataset
  # ------------------------------------------------------------------

  def _load_raw_dataset(self) -> Any:
    """Load a raw grain.MapDataset from the configured data_module.

    The data_module must expose a ``create_dataset(**data_config)`` function
    that returns a ``grain.MapDataset``.  An optional ``batch_fn`` attribute
    on the module is used as ``custom_batch_fn`` in post_init_dataset.
    """
    module_path = self.config["data_module"]
    module = importlib.import_module(module_path)
    data_config = dict(self.config.get("data_config", {}))
    return module.create_dataset(**data_config), getattr(module, "batch_fn", None)

  # ------------------------------------------------------------------
  # Kubernetes
  # ------------------------------------------------------------------

  def _setup_kubernetes(self) -> None:
    k8s_cfg = self.config.get("kubernetes_config") or {}
    if not k8s_cfg:
      return
    node_selector_key = k8s_cfg.get(
        "node_selector_key", "cloud.google.com/gke-nodepool"
    )
    node_selector_val = k8s_cfg.get("node_selector_val", "deepswe-cpu-pool")
    kubeconfig = k8s_cfg.get("kubeconfig", "~/.kube/config")
    os.environ["KUBECONFIG"] = kubeconfig
    os.environ["NODE_SELECTOR_KEY"] = node_selector_key
    os.environ["NODE_SELECTOR_VAL"] = node_selector_val
    logging.info(
        "Kubernetes node selector: %s=%s", node_selector_key, node_selector_val
    )
    try:
      from kubernetes import client as k8s_client_lib  # pylint: disable=g-import-not-at-top
      from kubernetes import config as k8s_config_lib  # pylint: disable=g-import-not-at-top
      k8s_config_lib.load_kube_config()
      k8s_client_lib.CoreV1Api()
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Kubernetes config loading failed: %s", e)

  # ------------------------------------------------------------------
  # Main runner
  # ------------------------------------------------------------------

  def run_agentic_grpo_trainer(self) -> None:
    """Execute the full agentic GRPO training pipeline."""
    self._setup_kubernetes()

    from tunix.cli.utils import model as model_lib  # pylint: disable=g-import-not-at-top

    tokenizer = model_lib.create_tokenizer(
        self.config["tokenizer_config"],
        self.config["tokenizer_config"]["tokenizer_path"],
    )
    chat_parser = self._create_chat_parser(tokenizer)

    # Dataset
    raw_dataset, custom_batch_fn = self._load_raw_dataset()
    max_prompt_length = self.config["rollout_config"].get("max_prompt_length")
    batch_size = self.config.get("batch_size", 1)
    num_batches = self.config.get("num_batches")
    num_epochs = self.config.get("num_train_epochs", 1)
    train_fraction = self.config.get("train_fraction", 1.0)
    prompt_key = self.config.get("prompt_key", "prompts")

    dataset, _ = data_lib.post_init_dataset(
        raw_dataset,
        tokenizer,
        batch_size=batch_size,
        num_batches=num_batches,
        max_prompt_length=max_prompt_length,
        fraction=train_fraction,
        num_epochs=num_epochs,
        prompt_key=prompt_key,
        custom_batch_fn=custom_batch_fn,
    )

    # RLCluster (reuses GrpoPipeline.create_rl_cluster)
    self.compute_params(raw_dataset)
    rl_cluster = self.create_rl_cluster(tokenizer)

    # Agent / env classes
    agent_class_path = self.config.get("agent_class_path")
    agent_class = (
        self._load_class_from_path(agent_class_path)
        if agent_class_path
        else None
    )
    agent_kwargs = dict(self.config.get("agent_kwargs") or {})

    env_class_path = self.config.get("env_class_path")
    env_class = (
        self._load_class_from_path(env_class_path) if env_class_path else None
    )
    env_kwargs = dict(self.config.get("env_kwargs") or {})

    # GRPOConfig
    algo_config = self._create_grpo_config()

    # Reward functions (None when env provides rewards, e.g. DeepSWE)
    reward_fns = self.obtain_reward_fn() if self.config.get("reward_functions") else None

    learner_kwargs: dict[str, Any] = dict(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=reward_fns,
        chat_parser=chat_parser,
    )
    if agent_class is not None:
      learner_kwargs["agent_class"] = agent_class
      learner_kwargs["agent_kwargs"] = agent_kwargs
    if env_class is not None:
      learner_kwargs["env_class"] = env_class
      learner_kwargs["env_kwargs"] = env_kwargs

    learner = GRPOLearner(**learner_kwargs)
    logging.info("Starting agentic GRPO training...")
    learner.train(dataset)


def _setup_jax_pathways(pathways_bns: str) -> None:
  flags.FLAGS.pathways_ifrt = True
  jax.config.update("jax_xla_backend", "pathways")
  jax.config.update("jax_backend_target", pathways_bns)


def main(argv, **kwargs):
  if _PATHWAYS_BNS.value:
    _setup_jax_pathways(_PATHWAYS_BNS.value)
  pipeline = AgenticGrpoPipeline(argv, **kwargs)
  logging.info(
      "--- Launching Agentic GRPO pipeline ---\n%r\n---", pipeline.config
  )
  pipeline.run_agentic_grpo_trainer()


if __name__ == "__main__":
  app.run(main)
