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

"""CLI entry point for the DeepScaleR agentic GRPO recipe.

DeepScaleR trains a reasoning model (e.g. DeepSeek-R1-Distill-Qwen-1.5B) on
math problems using Group Relative Policy Optimisation with an agentic
single-turn framework.

Usage::

    # Minimal: use the bundled default config
    python -m tunix.cli.deepscaler_main \\
        examples/deepscaler/configs/deepseek_r1_distill_qwen_1.5b.yaml

    # Override individual keys
    python -m tunix.cli.deepscaler_main \\
        examples/deepscaler/configs/deepseek_r1_distill_qwen_1.5b.yaml \\
        grpo_config.num_generations=4 \\
        rl_training_config.max_steps=100

    # Point at a different config and override the model path
    python -m tunix.cli.deepscaler_main my_config.yaml \\
        model_config.model_path=gs://my-bucket/DeepSeek-R1-Distill-Qwen-7B

Reference: https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2
"""

from __future__ import annotations

import dataclasses
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
import optax
from orbax import checkpoint as ocp
import pandas as pd

from tunix.cli import config as config_lib
from tunix.cli.utils import data as data_lib
from tunix.cli.utils import model as model_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.utils import math_rewards


_PATHWAYS_BNS = flags.DEFINE_string(
    "pathways_bns", None, "BNS address of the Pathways server."
)


class DeepScalerPipeline(config_lib.HyperParameters):
  """Runs the DeepScaleR agentic GRPO training recipe from a YAML config."""

  # ------------------------------------------------------------------
  # Mesh helpers
  # ------------------------------------------------------------------

  def _create_split_meshes(self):
    """Build trainer and rollout meshes from config, splitting devices."""
    trainer_mesh_cfg = self.config.get("trainer_mesh", {})
    rollout_mesh_cfg = self.config.get("rollout_mesh", {})

    def _parse_shape(shape_str: str) -> tuple[int, ...]:
      return tuple(int(x) for x in shape_str.strip("()").split(",") if x.strip())

    def _parse_axes(axes_str: str) -> tuple[str, ...]:
      cleaned = axes_str.strip("()").replace("'", "").replace('"', "")
      return tuple(x.strip() for x in cleaned.split(",") if x.strip())

    trainer_shape = _parse_shape(
        trainer_mesh_cfg.get("shape", "(4,1)")
    )
    trainer_axes = _parse_axes(
        trainer_mesh_cfg.get("axis_names", "('fsdp','tp')")
    )
    rollout_shape = _parse_shape(
        rollout_mesh_cfg.get("shape", "(4,1)")
    )
    rollout_axes = _parse_axes(
        rollout_mesh_cfg.get("axis_names", "('fsdp','tp')")
    )

    trainer_n = math.prod(trainer_shape)
    rollout_n = math.prod(rollout_shape)
    total_devices = jax.device_count()

    if trainer_n + rollout_n > total_devices:
      raise ValueError(
          f"trainer ({trainer_n}) + rollout ({rollout_n}) devices exceed "
          f"available {total_devices} devices."
      )

    rollout_engine = self.config.get("rollout_engine", "vanilla")
    if rollout_engine in ("sglang_jax", "vllm"):
      rollout_devices = jax._src.mesh_utils.create_device_mesh(  # pylint: disable=protected-access
          rollout_shape, jax.devices()[:rollout_n]
      )
      rollout_mesh = jax.sharding.Mesh(
          rollout_devices,
          axis_names=rollout_axes,
          axis_types=(jax.sharding.AxisType.Auto,) * len(rollout_shape),
      )
      trainer_devices = jax._src.mesh_utils.create_device_mesh(  # pylint: disable=protected-access
          trainer_shape, jax.devices()[-trainer_n:]
      )
      trainer_mesh = jax.sharding.Mesh(
          trainer_devices,
          axis_names=trainer_axes,
          axis_types=(jax.sharding.AxisType.Auto,) * len(trainer_shape),
      )
    else:
      # Vanilla: use all devices for a single mesh
      all_devices = jax._src.mesh_utils.create_device_mesh(trainer_shape)  # pylint: disable=protected-access
      trainer_mesh = jax.sharding.Mesh(all_devices, axis_names=trainer_axes)
      rollout_mesh = trainer_mesh

    return trainer_mesh, rollout_mesh

  # ------------------------------------------------------------------
  # Dataset
  # ------------------------------------------------------------------

  def _create_datasets(self, tokenizer):
    """Load DeepScaleR train set and AIME eval set from configured paths."""
    data_cfg = self.config.get("data_config", {})
    train_path = data_cfg.get("train_data_path")
    eval_path = data_cfg.get("eval_data_path")
    shuffle = data_cfg.get("shuffle", False)
    seed = self.config.get("seed", 42)

    import datasets as datasets_lib
    import grain
    import fsspec

    Dataset = datasets_lib.Dataset

    def preprocess_fn(example, index):
      del index
      return {
          "question": example["problem"],
          "ground_truth": example["answer"],
          "data_source": "math",
      }

    with fsspec.open(train_path) as train_f:
      train_df = pd.read_json(train_f)
    with fsspec.open(eval_path, "rb") as eval_f:
      eval_df = pd.read_parquet(eval_f)

    train_ds = Dataset.from_pandas(train_df).map(preprocess_fn, with_indices=True)
    eval_ds = Dataset.from_pandas(eval_df).map(preprocess_fn, with_indices=True)

    if shuffle:
      train_ds = train_ds.shuffle(seed)
      eval_ds = eval_ds.shuffle(seed)

    instruction = (
        "Let's think step by step, and put your final answer within \\boxed{}."
    )

    def process_item(item):
      question = item.get("question", item.get("problem", ""))
      answer = item.get("answer", item.get("ground_truth", ""))
      return {
          "prompts": f"{question} {instruction}",
          "question": question,
          "answer": answer,
      }

    train_ds = grain.MapDataset.source(train_ds).map(process_item)
    eval_ds = grain.MapDataset.source(eval_ds).map(process_item)
    return train_ds, eval_ds

  # ------------------------------------------------------------------
  # Rollout config
  # ------------------------------------------------------------------

  def _create_rollout_config(self, rollout_mesh: jax.sharding.Mesh) -> base_rollout.RolloutConfig:
    rc = self.config.get("rollout_config", {})
    engine = self.config.get("rollout_engine", "vanilla")
    max_prompt = rc.get("max_prompt_length", 2048)
    max_response = rc.get("max_response_length", 8192)
    max_concurrency = self.config.get("grpo_config", {}).get("max_concurrency", 768)

    base = dict(
        max_prompt_length=max_prompt,
        kv_cache_size=max_prompt + max_response + 256,
        temperature=rc.get("temperature", 0.8),
        top_p=rc.get("top_p", 0.95),
        top_k=rc.get("top_k", None),
        return_logprobs=True,
        max_tokens_to_generate=max_response,
    )

    model_version = self.config.get("model_config", {}).get(
        "model_version", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    )

    if engine == "sglang_jax":
      sglang_cfg = self.config.get("sglang_jax_config", {})
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
              "max_running_requests", max_concurrency
          ),
          rollout_sglang_jax_page_size=sglang_cfg.get("page_size", 128),
          rollout_sglang_jax_use_sort_for_toppk_minp=sglang_cfg.get(
              "use_sort_for_toppk_minp", False
          ),
      )
    elif engine == "vllm":
      vllm_cfg = self.config.get("vllm_config", {})
      rollout_shape = rollout_mesh.devices.shape
      max_num_seqs = vllm_cfg.get("max_num_seqs", 768)
      max_batched_tokens = vllm_cfg.get(
          "max_num_batched_tokens", max_num_seqs * 10 * 1024 // 8
      )
      extra = dict(
          rollout_vllm_model_version=vllm_cfg.get("model_version", model_version),
          rollout_vllm_hbm_utilization=vllm_cfg.get("hbm_utilization", 0.4),
          rollout_vllm_tpu_backend_type=vllm_cfg.get("tpu_backend_type", "jax"),
          rollout_vllm_server_mode=vllm_cfg.get("server_mode", True),
          rollout_vllm_async_scheduling=vllm_cfg.get("async_scheduling", True),
          tensor_parallel_size=rollout_shape[1] if len(rollout_shape) > 1 else 1,
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
  # Optimizer
  # ------------------------------------------------------------------

  def _create_optimizer(self, max_steps: int) -> optax.GradientTransformation:
    opt_cfg = self.config.get("optimizer_config", {})
    lr = opt_cfg.get("learning_rate", 1e-6)
    b1 = opt_cfg.get("b1", 0.9)
    b2 = opt_cfg.get("b2", 0.99)
    wd = opt_cfg.get("weight_decay", 0.01)
    max_grad_norm = opt_cfg.get("max_grad_norm", 1.0)
    warmup_ratio = opt_cfg.get("warmup_ratio", 0.1)
    warmup_steps = opt_cfg.get("warmup_steps", int(warmup_ratio * max_steps))

    optimizer = optax.adamw(
        learning_rate=optax.schedules.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=max_steps,
            end_value=0.0,
        ),
        b1=b1,
        b2=b2,
        weight_decay=wd,
    )
    if max_grad_norm:
      optimizer = optax.chain(
          optax.clip_by_global_norm(max_norm=max_grad_norm),
          optimizer,
      )
    return optimizer

  # ------------------------------------------------------------------
  # Main runner
  # ------------------------------------------------------------------

  def run_deepscaler(self):
    """Execute the full DeepScaleR training pipeline."""
    from tunix.rl import utils as rl_utils  # pylint: disable=g-import-not-at-top

    trainer_mesh, rollout_mesh = self._create_split_meshes()
    logging.info("Trainer mesh: %s", trainer_mesh)
    logging.info("Rollout mesh: %s", rollout_mesh)

    # ---- Tokenizer ----
    model_cfg = self.config["model_config"]
    tokenizer = model_lib.create_tokenizer(
        self.config["tokenizer_config"],
        self.config["tokenizer_config"]["tokenizer_path"],
    )
    chat_parser = chat_parser_lib.DefaultChatTemplateParser(tokenizer)

    # ---- Dataset ----
    data_cfg = self.config.get("data_config", {})
    batch_size = self.config.get("batch_size", 128)
    num_batches = self.config.get("num_batches", 312)
    num_test_batches = self.config.get("num_test_batches", 50)
    num_epochs = self.config.get("num_epochs", 3)
    train_fraction = self.config.get("train_fraction", 1.0)
    max_prompt_length = self.config.get("rollout_config", {}).get(
        "max_prompt_length", 2048
    )
    seed = self.config.get("seed", 42)

    train_ds_raw, eval_ds_raw = self._create_datasets(tokenizer)
    train_dataset, _ = data_lib.post_init_dataset(
        train_ds_raw,
        tokenizer,
        batch_size=batch_size,
        num_batches=num_batches,
        max_prompt_length=max_prompt_length,
        fraction=train_fraction,
        num_epochs=num_epochs,
    )
    eval_dataset, _ = data_lib.post_init_dataset(
        eval_ds_raw,
        tokenizer,
        batch_size=batch_size,
        num_batches=num_test_batches,
        max_prompt_length=max_prompt_length,
    )

    # ---- Models ----
    logging.info("Loading reference model from %s ...", model_cfg.get("model_path"))
    qwen2_ref, _ = model_lib.create_model(
        model_cfg,
        self.config["tokenizer_config"],
        trainer_mesh,
    )

    train_with_lora = model_cfg.get("train_with_lora", False)
    if train_with_lora:
      qwen2_actor = model_lib.apply_lora_to_model(
          qwen2_ref,
          trainer_mesh,
          model_cfg["lora_config"],
      )
    else:
      graph_def, params = nnx.split(qwen2_ref)
      qwen2_actor = nnx.merge(graph_def, jax.tree.map(jnp.copy, params))

    # ---- Training config ----
    rl_train_cfg = self.config.get("rl_training_config", {})
    grpo_cfg = self.config.get("grpo_config", {})
    max_response_length = self.config.get("rollout_config", {}).get(
        "max_response_length", 8192
    )
    max_steps = rl_train_cfg.get("max_steps", num_batches * num_epochs)
    eval_every_n_steps = rl_train_cfg.get("eval_every_n_steps", 1000)
    mini_batch_size = rl_train_cfg.get("mini_batch_size", batch_size)
    train_micro_batch_size = rl_train_cfg.get("train_micro_batch_size", 2)
    ckpt_dir = rl_train_cfg.get("checkpoint_root_directory", "/tmp/tunix/deepscaler")
    save_interval_steps = rl_train_cfg.get("save_interval_steps", 500)
    max_to_keep = rl_train_cfg.get("max_to_keep", 4)

    metrics_log_dir = rl_train_cfg.get("metrics_log_dir", "/tmp/tensorboard/deepscaler")
    metrics_flush_steps = rl_train_cfg.get("metrics_flush_every_n_steps", 20)
    metrics_logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir=metrics_log_dir,
        flush_every_n_steps=metrics_flush_steps,
    )

    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=save_interval_steps,
        max_to_keep=max_to_keep,
    )

    optimizer = self._create_optimizer(max_steps)

    # ---- Rollout config ----
    rollout_engine = self.config.get("rollout_engine", "vanilla")
    rollout_engine_config = self._create_rollout_config(rollout_mesh)

    # ---- Cluster config ----
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: trainer_mesh,
            rl_cluster_lib.Role.REFERENCE: trainer_mesh,
            rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
        },
        rollout_engine=rollout_engine,
        offload_to_cpu=self.config.get("offload_to_cpu", False),
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=eval_every_n_steps,
            max_steps=max_steps,
            mini_batch_size=mini_batch_size,
            train_micro_batch_size=train_micro_batch_size,
            metrics_logging_options=metrics_logging_options,
            checkpoint_root_directory=ckpt_dir,
            checkpointing_options=checkpointing_options,
        ),
        rollout_config=rollout_engine_config,
    )

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=qwen2_actor,
        reference=qwen2_ref,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    # ---- GRPO algo config ----
    algo_config = GRPOConfig(
        num_generations=grpo_cfg.get("num_generations", 8),
        num_iterations=grpo_cfg.get("num_iterations", 1),
        max_response_length=max_response_length,
        beta=grpo_cfg.get("beta", 0.0),
        epsilon=grpo_cfg.get("epsilon", 0.2),
        epsilon_high=grpo_cfg.get("epsilon_high", 0.28),
        system_prompt=grpo_cfg.get("system_prompt", ""),
        max_concurrency=grpo_cfg.get("max_concurrency", 768),
        off_policy_steps=grpo_cfg.get("off_policy_steps", 0),
        loss_agg_mode=grpo_cfg.get("loss_agg_mode", "token-mean"),
        kl_loss_mode=grpo_cfg.get("kl_loss_mode", "low_var_kl"),
    )

    # ---- Metric fn ----
    def metric_fn(prompts, completions, rewards, advantages, **kwargs):  # pylint: disable=unused-argument
      del prompts, completions, advantages, kwargs
      solve_all = (rewards > 0.1).all()
      solve_none = (rewards == 0).all()
      solve_ratio = (rewards > 0.1).mean()
      return {
          "rewards/solve_all": (1 if solve_all else 0, np.mean),
          "rewards/solve_none": (1 if solve_none else 0, np.mean),
          "rewards/solve_partial": (
              1 if (not solve_all and not solve_none) else 0,
              np.mean,
          ),
          "rewards/solve_ratio": (solve_ratio, np.mean),
      }

    # ---- Trainer ----
    trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=[math_rewards.math_reward],
        algo_config=algo_config,
        chat_parser=chat_parser,
        metric_fns=[metric_fn],
    )
    trainer.train(train_dataset)


def _setup_jax_pathways(pathways_bns: str):
  flags.FLAGS.pathways_ifrt = True
  jax.config.update("jax_xla_backend", "pathways")
  jax.config.update("jax_backend_target", pathways_bns)


def main(argv, **kwargs):
  if _PATHWAYS_BNS.value:
    _setup_jax_pathways(_PATHWAYS_BNS.value)
  pipeline = DeepScalerPipeline(argv, **kwargs)
  logging.info(
      "--- Launching DeepScaleR pipeline ---\n%r\n---", pipeline.config
  )
  pipeline.run_deepscaler()


if __name__ == "__main__":
  app.run(main)
