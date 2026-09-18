#!/bin/bash
# run_ppo_7b_nextgen.sh — 7B × 8 卡 PPO Next-Gen 全量训练 (71 Steps, 4,518 题)
# 核心架构：融合四大顶会创新成果
#   1. VeRPO (arXiv:2601.03525): 消除基数偏差，超线性密度校准攻坚边界用例；
#   2. TIPS (ICLR 2026, arXiv:2510.04652): PBRS 连续平滑势能，消灭断崖跳变与 Reward Hacking；
#   3. DHRCL (arXiv:2607.26457): 三阶段课程学习 (0-28步语法防崩 -> 28-53步功能攻坚 -> 53-71步防注水精简)；
#   4. MAPO (arXiv:2502.19340): veRL 原生混合优势估计，稳定长代码价值回传。

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

# 2. 自动打并发沙箱与环境补丁（幂等）
if [ -f "$HOME_DIR/project/patch_vllm_ascend.py" ]; then
    python3 "$HOME_DIR/project/patch_vllm_ascend.py"
fi

# 3. 路径配置（物理隔离）
MODEL="${HOME_DIR}/Qwen2.5-7B-Instruct"
DATA="${HOME_DIR}/project/data/rlvr"
REWARD="${HOME_DIR}/project/rewards/code_rlvr_nextgen.py"
CKPT_DIR="${HOME_DIR}/project/checkpoints/d4_full_7b_nextgen"

mkdir -p "$CKPT_DIR"
mkdir -p "${HOME_DIR}/project/logs"

# 4. 华为昇腾 NPU 专用环境变量与 64 线程沙箱控制
export TORCHDYNAMO_DISABLE=1
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_OP_EXPANSION_MODE="AIV"
export VERL_REWARD_WORKERS=64
export TRL_EXPERIMENTAL_SILENCE=1

# 5. DHRCL 课程退火总步数同步
export RLVR_TOTAL_STEPS=71

echo "================================================================="
echo "[PPO Next-Gen Full 71-Step] 启动 7B × 8 卡下一代融合奖励主训练"
echo "  - 模型基座: $MODEL"
echo "  - 训练数据: $DATA/train_full.parquet (4,518 题, 100% 全覆盖)"
echo "  - 奖励函数: $REWARD (VeRPO + TIPS-PBRS + DHRCL 课程)"
echo "  - 优势估计: PPO GAE / MAPO (veRL 原生支持)"
echo "  - 检查点目录: $CKPT_DIR"
echo "  - 训练总步数: 71 步 (1 Full Epoch, batch_size=64, 预估耗时 ~55 分钟)"
echo "================================================================="

# 6. 执行 PPO 71 步训练
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
    trainer.experiment_name='d4_full_7b_nextgen' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=71 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.default_local_dir="$CKPT_DIR" \
    "$@"
