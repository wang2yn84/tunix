#!/bin/bash
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
#
# DeepSWE training launcher — all knobs exposed as CLI flags.
# Defaults match examples/deepswe/configs/qwen3_32b.yaml.
# Uncomment or edit any line to override the YAML default.
#
# Prerequisites:
#   - Kubernetes kubeconfig at ~/.kube/config with a node pool matching
#     kubernetes_config.node_selector_val (default: "deepswe-cpu-pool")
#   - Qwen3-32B weights accessible via model_config.model_path (or HF download)
#
# Usage:
#   bash examples/deepswe/run_deepswe.sh
#
# Run from the tunix repo root.

set -euo pipefail

CONFIG="examples/deepswe/configs/qwen3_32b.yaml"

python -m tunix.cli.grpo_main "${CONFIG}" \
  \
  `# ── Model ────────────────────────────────────────────────────────────` \
  `# model_config.model_id="Qwen/Qwen3-32B"` \
  `# model_config.model_source="huggingface"` \
  `# model_config.model_path="/path/to/local/Qwen3-32B"` \
  `# model_config.rng_seed=42` \
  \
  `# ── Data ─────────────────────────────────────────────────────────────` \
  `# data_config.dataset_name="R2E-Gym/R2E-Gym-V1"` \
  `# data_config.dataset_split="train"` \
  `# data_config.cache_dir="/tmp/dataset_cache"` \
  `# data_config.shuffle=true` \
  `# data_config.seed=42` \
  \
  `# ── Training loop ────────────────────────────────────────────────────` \
  `# batch_size=1` \
  `# num_batches=20` \
  `# num_train_epochs=1` \
  `# train_fraction=1.0` \
  \
  `# ── Rollout engine (vanilla | vllm | sglang_jax) ─────────────────────` \
  `# rollout_engine="vllm"` \
  \
  `# ── Rollout config ───────────────────────────────────────────────────` \
  `# rollout_config.max_prompt_length=4096` \
  `# rollout_config.total_generation_steps=8192` \
  `# rollout_config.temperature=1.0` \
  `# rollout_config.top_p=null` \
  `# rollout_config.top_k=null` \
  \
  `# ── vLLM (used when rollout_engine=vllm) ─────────────────────────────` \
  `# vllm_config.hbm_utilization=0.4` \
  `# vllm_config.max_num_seqs=2` \
  \
  `# ── SGLang-JAX (used when rollout_engine=sglang_jax) ─────────────────` \
  `# sglang_jax_config.mem_fraction_static=0.9` \
  `# sglang_jax_config.init_with_random_weights=true` \
  `# sglang_jax_config.disable_radix_cache=false` \
  `# sglang_jax_config.chunked_prefill_size=2048` \
  `# sglang_jax_config.page_size=128` \
  \
  `# ── Kubernetes ───────────────────────────────────────────────────────` \
  `# kubernetes_config.node_selector_val="deepswe-cpu-pool"` \
  `# kubernetes_config.kubeconfig="~/.kube/config"` \
  \
  `# ── Agentic / multi-turn ─────────────────────────────────────────────` \
  `# agentic_grpo_config.max_turns=20` \
  `# agentic_grpo_config.per_turn_timeout_secs=300` \
  `# agentic_grpo_config.context_ratio=2` \
  `# agentic_grpo_config.max_concurrency=1` \
  \
  `# ── GRPO algorithm ───────────────────────────────────────────────────` \
  `# agentic_grpo_config.num_generations=2` \
  `# agentic_grpo_config.num_iterations=1` \
  `# agentic_grpo_config.beta=0.001` \
  `# agentic_grpo_config.epsilon=0.2` \
  `# agentic_grpo_config.epsilon_high=0.28` \
  `# agentic_grpo_config.off_policy_steps=0` \
  \
  `# ── Optimizer ────────────────────────────────────────────────────────` \
  `# rl_training_config.actor_optimizer_config.learning_rate=1e-6` \
  `# rl_training_config.actor_optimizer_config.b1=0.9` \
  `# rl_training_config.actor_optimizer_config.b2=0.99` \
  `# rl_training_config.actor_optimizer_config.weight_decay=0.1` \
  `# rl_training_config.actor_optimizer_config.max_grad_norm=0.1` \
  `# rl_training_config.actor_optimizer_config.warmup_ratio=0.1` \
  \
  `# ── RL training ──────────────────────────────────────────────────────` \
  `# rl_training_config.max_steps=10` \
  `# rl_training_config.eval_every_n_steps=10` \
  `# rl_training_config.mini_batch_size=1` \
  `# rl_training_config.train_micro_batch_size=1` \
  `# rl_training_config.rollout_micro_batch_size=1` \
  `# rl_training_config.compute_logps_micro_batch_size=1` \
  `# rl_training_config.checkpoint_root_directory="/tmp/tunix/checkpoints/deepswe"` \
  `# rl_training_config.checkpointing_options.save_interval_steps=500` \
  `# rl_training_config.checkpointing_options.max_to_keep=4` \
  `# rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/deepswe"` \
  `# rl_training_config.metrics_logging_options.flush_every_n_steps=2` \
  \
  "$@"
