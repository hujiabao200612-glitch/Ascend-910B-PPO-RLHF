#!/bin/bash
# run_ppo_7b_verpo.sh — 7B × 8 卡 PPO-VeRPO 纯粹消融训练 (71 Steps, 4,518 题)
# 核心机理：纯粹 VeRPO 超线性密度校准 (gamma=1.6)，消除测试用例基数偏差，攻坚极端边缘断言

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(cd "$CURRENT_DIR/.." && pwd)
DEFAULT_HOME="/data/home/1120250939"
HOME_DIR="${HOME_DIR:-$DEFAULT_HOME}"

# 1. 激活虚拟环境与环境补丁
source "$HOME_DIR/project/envs/verl_env/bin/activate"
python3 "$HOME_DIR/project/patch_vllm_ascend.py"

# 2. 路径配置
MODEL="${HOME_DIR}/Qwen2.5-7B-Instruct"
DATA="${HOME_DIR}/project/data/rlvr"
REWARD="rewards/code_rlvr_verpo.py"
CKPT_DIR="${HOME_DIR}/project/checkpoints/d4_full_7b_verpo"
TARGET_HF="${HOME_DIR}/project/checkpoints/d4_full_7b_verpo_hf"
EVAL_PREFIX="${HOME_DIR}/project/eval_results/eval_full_verpo_"

mkdir -p "$CKPT_DIR"
mkdir -p "${HOME_DIR}/project/logs"
mkdir -p "${HOME_DIR}/project/eval_results"

# 3. 华为昇腾 NPU 专用环境变量与 64 线程沙箱控制
export TORCHDYNAMO_DISABLE=1
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_OP_EXPANSION_MODE="AIV"
export VERL_REWARD_WORKERS=64
export TRL_EXPERIMENTAL_SILENCE=1

echo "================================================================="
echo "  🚀 [VeRPO 消融实验] 启动 7B × 8 卡 VeRPO 71 步训练与评测"
echo "  - 模型基座: $MODEL"
echo "  - 奖励函数: $REWARD (纯粹 VeRPO 消除基数偏差)"
echo "  - 检查点目录: $CKPT_DIR"
echo "  - 目标 HF:    $TARGET_HF"
echo "================================================================="

# 4. 清理残留
ray stop --force > /dev/null 2>&1 || true
pkill -9 -f "verl" > /dev/null 2>&1 || true
pkill -9 -f "ray" > /dev/null 2>&1 || true
sleep 4

rm -rf "$CKPT_DIR"
mkdir -p "$CKPT_DIR"

# 5. 执行 PPO 71 步全量训练
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
    custom_reward_function.path="${HOME_DIR}/project/${REWARD}" \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.experiment_name='d4_full_7b_verpo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_training_steps=71 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.default_local_dir="$CKPT_DIR"

# 6. 释放训练显存
ray stop --force > /dev/null 2>&1 || true
pkill -9 -f "verl" > /dev/null 2>&1 || true
sleep 4

# 7. 合并导出 HuggingFace 格式
echo ">>> [2/3] 合并导出 HuggingFace 权重 ..."
CKPT_SRC="${CKPT_DIR}/global_step_71/actor"
[ ! -d "$CKPT_SRC" ] && CKPT_SRC="${CKPT_DIR}/global_step_70/actor"

rm -rf "$TARGET_HF"
mkdir -p "$TARGET_HF"
python3 -c "
import os, sys
try:
    import verl.model_merger as mm
    cmd = f'{sys.executable} -m verl.model_merger merge --backend fsdp --local_dir \"$CKPT_SRC\" --target_dir \"$TARGET_HF\"'
    os.system(cmd)
except Exception:
    cmd_fallback = f'{sys.executable} -m verl.utils.checkpoint.fsdp_to_hf --fsdp_checkpoint_dir \"$CKPT_SRC\" --target_dir \"$TARGET_HF\"'
    os.system(cmd_fallback)
"
cp -n "$MODEL"/tokenizer* "$TARGET_HF"/ 2>/dev/null || true
cp -n "$MODEL"/vocab* "$TARGET_HF"/ 2>/dev/null || true
cp -n "$MODEL"/merges* "$TARGET_HF"/ 2>/dev/null || true
cp -n "$MODEL"/chat_template* "$TARGET_HF"/ 2>/dev/null || true

# 8. 执行三大基准全量 791 题评测
echo ">>> [3/3] 启动 791 题全量三大基准评测 ..."
python3 eval_test_set.py --model "$TARGET_HF" --output "${EVAL_PREFIX}kodcode.json"
python3 eval_humaneval.py --model "$TARGET_HF" --output "${EVAL_PREFIX}humaneval.json"
python3 eval_mbpp.py --model "$TARGET_HF" --output "${EVAL_PREFIX}mbpp.json"

echo "================================================================="
echo "  🎉 VeRPO (71步) 训练、导出与 791 题评测全部圆满大功告成！"
echo "================================================================="
