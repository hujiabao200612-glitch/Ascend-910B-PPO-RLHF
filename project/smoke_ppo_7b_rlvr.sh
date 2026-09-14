#!/bin/bash
# smoke_ppo_7b_rlvr.sh — D3 阶段：7B × 8 卡 PPO 代码 RLVR 20 步极速闭环验证
# 适用硬件：8 × 华为昇腾 Ascend 910B3 (64GB)
# 核心指标：验证“数据加载 -> 8卡 Rollout -> 沙箱真跑打分 -> Critic GAE -> 梯度更新 -> Checkpoint 导出”全链路

# 1. 自动解析集群 HOME_DIR 与虚拟环境
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(cd "$CURRENT_DIR/.." && pwd)
if [[ "$PARENT_DIR" =~ ^/data/home/[^/]+$ ]]; then
    DEFAULT_HOME="$PARENT_DIR"
else
    DEFAULT_HOME="/data/home/<你的学号>"
fi
HOME_DIR="${HOME_DIR:-$DEFAULT_HOME}"

# 自动激活持久虚拟环境
if [ -f "$HOME_DIR/project/envs/verl_env/bin/activate" ]; then
    source "$HOME_DIR/project/envs/verl_env/bin/activate"
fi

# 2. 自动打 vllm-ascend 8 卡通信补丁（幂等）
if [ -f "$HOME_DIR/project/patch_vllm_ascend.py" ]; then
    python3 "$HOME_DIR/project/patch_vllm_ascend.py"
fi

# 3. 路径配置
MODEL="${HOME_DIR}/Qwen2.5-7B-Instruct"
DATA="${HOME_DIR}/project/data/rlvr"
REWARD="${HOME_DIR}/project/rewards/code_rlvr.py"
CKPT_DIR="${HOME_DIR}/project/checkpoints/d3_smoke_7b_rlvr"

mkdir -p "$CKPT_DIR"

# 4. 华为昇腾 NPU 专用环境变量配置
export TORCHDYNAMO_DISABLE=1          # 禁用 dynamo 编译，避免 NPU 算子挂死
export VLLM_ASCEND_ENABLE_NZ=0        # 禁用非连续内存优化，保障长序列稳定性
export HCCL_OP_EXPANSION_MODE="AIV"

echo "================================================================="
echo "[D3 Smoke] 启动 7B × 8 卡 PPO RLVR 20-step 闭环训练"
echo "  - 模型路径: $MODEL"
echo "  - 训练数据: $DATA/train_smoke.parquet (128题)"
echo "  - 评测数据: $DATA/test.parquet (200题)"
echo "  - 奖励函数: $REWARD (pytest 沙箱驱动)"
echo "  - 检查点目录: $CKPT_DIR"
echo "================================================================="

# 5. 执行 PPO 训练
python3 -m verl.trainer.main_ppo \
    trainer.device=npu \
    algorithm.adv_estimator=gae \
    data.train_files="['$DATA/train_smoke.parquet']" \
    data.val_files="['$DATA/test.parquet']" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=768 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="$MODEL" \
    critic.ppo_micro_batch_size_per_gpu=4 \
    reward_model.enable=False \
    custom_reward_function.path="$REWARD" \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.experiment_name='d3_smoke_7b_rlvr' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=20 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"
