#!/bin/bash
# run_ppo_7b_ablation_sparse.sh — 奖励函数消融实验 2：7B × 8 卡 PPO 纯稀疏二进制奖励训练 (Ablation Exp 2)
# 特性：启用 code_rlvr_sparse.py ({0.0, 1.0} 纯稀疏二值奖励)，检验阿里 Qwen 团队 Proxy Hacking 假说

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(cd "$CURRENT_DIR/.." && pwd)
if [[ "$PARENT_DIR" =~ ^/data/home/[^/]+$ ]]; then
    DEFAULT_HOME="$PARENT_DIR"
else
    DEFAULT_HOME="/data/home/1120250939"
fi
HOME_DIR="${HOME_DIR:-$DEFAULT_HOME}"

# 1. 自动激活持久虚拟环境
if [ -f "$HOME_DIR/project/envs/verl_env/bin/activate" ]; then
    source "$HOME_DIR/project/envs/verl_env/bin/activate"
fi

# 2. 自动打并发沙箱补丁（幂等）
if [ -f "$HOME_DIR/project/patch_vllm_ascend.py" ]; then
    python3 "$HOME_DIR/project/patch_vllm_ascend.py"
fi

# 3. 路径配置（物理隔离）
MODEL="${HOME_DIR}/Qwen2.5-7B-Instruct"
DATA="${HOME_DIR}/project/data/rlvr"
REWARD="${HOME_DIR}/project/rewards/code_rlvr_sparse.py"
CKPT_DIR="${HOME_DIR}/project/checkpoints/d4_ablation_7b_sparse"

mkdir -p "$CKPT_DIR"
mkdir -p "${HOME_DIR}/project/logs"

# 4. 华为昇腾 NPU 专用环境变量与 64 线程沙箱控制
export TORCHDYNAMO_DISABLE=1
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_OP_EXPANSION_MODE="AIV"
export VERL_REWARD_WORKERS=64

echo "================================================================="
echo "[Ablation Exp 2] 启动 7B × 8 卡 PPO 纯稀疏二进制奖励 ({0.0, 1.0}) 消融训练"
echo "  - 模型路径: $MODEL"
echo "  - 训练数据: $DATA/train_full.parquet (4,518 题)"
echo "  - 评测数据: $DATA/test.parquet (200 题)"
echo "  - 奖励函数: $REWARD (全对 +1.0, 任何未全通/语法错/空输出一律 0.0)"
echo "  - 检查点目录: $CKPT_DIR"
echo "  - 训练步数: 50 步 (与 Exp 1/GRPO 消融步数对齐，耗时约 40 分钟)"
echo "================================================================="

# 5. 执行 PPO 训练
python3 -m verl.trainer.main_ppo \
    trainer.device=npu \
    algorithm.adv_estimator=gae \
    data.train_files="['$DATA/train_full.parquet']" \
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
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
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
    trainer.experiment_name='d4_ablation_7b_sparse' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=50 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"
