#!/bin/bash
# smoke_ppo_05b.sh — verl 0.6.1 PPO 冒烟（NPU 单卡/多卡 · 0.5B · GSM8K 规则奖励 RLVR）
# ✅ 实测通过版（含多卡通信兼容与 NPU Dynamo 禁用）
# 适配：支持动态检测学号挂载路径，或通过环境变量 HOME_DIR / STUDENT_ID 指定

# 自动解析 HOME_DIR（默认识别父级 /data/home/<学号> 目录）
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(cd "$CURRENT_DIR/.." && pwd)
if [[ "$PARENT_DIR" =~ ^/data/home/[^/]+$ ]]; then
    DEFAULT_HOME="$PARENT_DIR"
else
    DEFAULT_HOME="/data/home/<你的学号>"
fi
HOME_DIR="${HOME_DIR:-$DEFAULT_HOME}"

# 自动激活持久虚拟环境（避免新终端遗漏激活导致 ModuleNotFoundError: No module named 'verl'）
if [ -f "$HOME_DIR/project/envs/verl_env/bin/activate" ]; then
    source "$HOME_DIR/project/envs/verl_env/bin/activate"
fi

MODEL=$HOME_DIR/Qwen2.5-0.5B-Instruct
DATA=$HOME_DIR/project/data/gsm8k
REWARD=$HOME_DIR/project/rewards/smoke_gsm8k.py

# NPU 注意：不要 export VLLM_ATTENTION_BACKEND=XFORMERS（NPU 无 xformers）
# vllm-ascend 0.9.x 默认引擎与 verl 0.6.1 兼容，无需额外变量
export TORCHDYNAMO_DISABLE=1        # 全局禁用 dynamo/compile（NPU 上编译会挂，实测）

python3 -m verl.trainer.main_ppo \
    trainer.device=npu \
    algorithm.adv_estimator=gae \
    data.train_files="['$DATA/train_300.parquet']" \
    data.val_files="['$DATA/test.parquet']" \
    data.train_batch_size=64 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="$MODEL" \
    critic.ppo_micro_batch_size_per_gpu=8 \
    reward_model.enable=False \
    custom_reward_function.path="$REWARD" \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=True \
    trainer.experiment_name='smoke_05b_npu' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=99 \
    trainer.total_epochs=1 "$@"
