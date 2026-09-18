#!/bin/bash
# run_all_4_ablations_71.sh — 4 组奖励消融实验 71 步 (1.0 Epoch) 全自动一键无人值守流水线
# 覆盖实验：
#   1. Exp 1: Neg-Penalty (强负惩罚 -1.0 vs +1.0)
#   2. Exp 2: Sparse-RLVR (纯稀疏二进制奖励 0.0 vs +1.0)
#   3. Exp 3: Discrete-Bins (全离散三档硬阶梯 -1.0 / 0.0 / +1.0)
#   4. Exp 4: Len-Efficiency (正确性与精简度双目标 +1.5 / +1.0 / -1.0)
# 全流程：每组自动执行 71 步训练 -> 释放显存 -> FSDP合并为HF -> 791题全量评测 -> 自动打印大榜汇总

set -e

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(cd "$CURRENT_DIR/.." && pwd)
if [[ "$PARENT_DIR" =~ ^/data/home/[^/]+$ ]]; then
    DEFAULT_HOME="$PARENT_DIR"
else
    DEFAULT_HOME="/data/home/1120250939"
fi
HOME_DIR="${HOME_DIR:-$DEFAULT_HOME}"

cd "$HOME_DIR/project"

# 1. 激活环境
if [ -f "$HOME_DIR/project/envs/verl_env/bin/activate" ]; then
    source "$HOME_DIR/project/envs/verl_env/bin/activate"
fi

# 2. 补丁检查
if [ -f "$HOME_DIR/project/patch_vllm_ascend.py" ]; then
    python3 "$HOME_DIR/project/patch_vllm_ascend.py"
fi

# 3. 环境变量
export TORCHDYNAMO_DISABLE=1
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_OP_EXPANSION_MODE="AIV"
export VERL_REWARD_WORKERS=64
export TRL_EXPERIMENTAL_SILENCE=1

MODEL="${HOME_DIR}/Qwen2.5-7B-Instruct"
DATA="${HOME_DIR}/project/data/rlvr"

mkdir -p eval_results
mkdir -p logs

function clean_npu() {
    echo ">>> [系统清理] 正在终止残留 Ray/NPU 进程并重置显存 ..."
    ray stop --force > /dev/null 2>&1 || true
    pkill -9 -f "verl" > /dev/null 2>&1 || true
    pkill -9 -f "ray" > /dev/null 2>&1 || true
    sleep 4
}

function run_single_ablation() {
    local EXP_ID="$1"
    local EXP_TAG="$2"
    local REWARD_FILE="$3"
    local EXP_NAME="d4_full_7b_${EXP_TAG}"
    local CKPT_DIR="${HOME_DIR}/project/checkpoints/${EXP_NAME}"
    local TARGET_HF="${HOME_DIR}/project/checkpoints/${EXP_NAME}_hf"
    local EVAL_PREFIX="${HOME_DIR}/project/eval_results/eval_full_${EXP_TAG}_"

    echo ""
    echo "================================================================="
    echo "  [流水线 ${EXP_ID}/4 启动] 实验: ${EXP_TAG} (71 Steps, 1.0 Epoch)"
    echo "  - 奖励函数: ${REWARD_FILE}"
    echo "  - 检查点目录: ${CKPT_DIR}"
    echo "  - 目标 HF 权重: ${TARGET_HF}"
    echo "================================================================="

    # 1. 清理
    clean_npu

    # 2. 71 步训练
    echo ">>> [1/3] 开始 71 步 PPO 训练 ..."
    mkdir -p "$CKPT_DIR"
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
        custom_reward_function.path="${HOME_DIR}/project/${REWARD_FILE}" \
        custom_reward_function.name=compute_score \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.val_before_train=False \
        trainer.experiment_name="${EXP_NAME}" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.total_training_steps=71 \
        trainer.save_freq=10 \
        trainer.test_freq=-1 \
        trainer.default_local_dir="$CKPT_DIR"

    # 3. 释放训练显存
    clean_npu

    # 4. FSDP 权重合并为 HF
    echo ">>> [2/3] 开始 FSDP 权重合并导出为原生 HuggingFace 格式 ..."
    local CKPT_SRC="${CKPT_DIR}/global_step_71/actor"
    if [ ! -d "$CKPT_SRC" ]; then
        if [ -d "${CKPT_DIR}/global_step_70/actor" ]; then
            CKPT_SRC="${CKPT_DIR}/global_step_70/actor"
        fi
    fi

    mkdir -p "$TARGET_HF"
    python3 -c "
import os, sys
ckpt_src = '$CKPT_SRC'
target_hf = '$TARGET_HF'
try:
    import verl.model_merger as mm
    cmd = f'{sys.executable} -m verl.model_merger merge --backend fsdp --local_dir {ckpt_src} --target_dir {target_hf}'
    os.system(cmd)
except Exception:
    cmd_fallback = f'{sys.executable} -m verl.utils.checkpoint.fsdp_to_hf --fsdp_checkpoint_dir {ckpt_src} --target_dir {target_hf}'
    os.system(cmd_fallback)
"
    cp -n "$MODEL"/tokenizer* "$TARGET_HF"/ 2>/dev/null || true
    cp -n "$MODEL"/vocab* "$TARGET_HF"/ 2>/dev/null || true
    cp -n "$MODEL"/merges* "$TARGET_HF"/ 2>/dev/null || true
    cp -n "$MODEL"/chat_template* "$TARGET_HF"/ 2>/dev/null || true

    # 5. 三大基准 791 题评测
    echo ">>> [3/3] 启动三大基准 791 题全量评测 ..."
    python3 eval_test_set.py --model "$TARGET_HF" --output "${EVAL_PREFIX}kodcode.json"
    python3 eval_humaneval.py --model "$TARGET_HF" --output "${EVAL_PREFIX}humaneval.json"
    python3 eval_mbpp.py --model "$TARGET_HF" --output "${EVAL_PREFIX}mbpp.json"

    echo ">>> [完成] 实验 ${EXP_TAG} (71 步) 训练、导出与 791 题评测全部通过！"
    clean_npu
}

START_TIME=$(date +%s)
echo "================================================================="
echo "  🚀 启动 4 组消融实验 71 步全自动无人值守总控流水线"
echo "  - 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  - 预计总耗时: 约 3.5 ~ 4.0 小时"
echo "================================================================="

# 执行第 1 组：强负惩罚
run_single_ablation 1 "neg_penalty" "rewards/code_rlvr_neg_penalty.py"

# 执行第 2 组：纯稀疏
run_single_ablation 2 "sparse" "rewards/code_rlvr_sparse.py"

# 执行第 3 组：全离散硬阶梯
run_single_ablation 3 "discrete_bins" "rewards/code_rlvr_discrete_bins.py"

# 执行第 4 组：长度双目标
run_single_ablation 4 "len_efficiency" "rewards/code_rlvr_len_efficiency.py"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "================================================================="
echo "  🎉🎉🎉 全部 4 组消融实验 71 步训练与评测圆满大功告成！"
echo "  - 总耗时: $((ELAPSED / 60)) 分 $((ELAPSED % 60)) 秒"
echo "================================================================="

# 打印最终总结大榜
python3 -c "
import json, glob, os

exps = [
    ('Exp 1: Neg-Penalty 71步', 'eval_results/eval_full_neg_penalty_'),
    ('Exp 2: Sparse-RLVR 71步', 'eval_results/eval_full_sparse_'),
    ('Exp 3: Discrete-Bins 71步', 'eval_results/eval_full_discrete_bins_'),
    ('Exp 4: Len-Efficiency 71步', 'eval_results/eval_full_len_efficiency_'),
]

print(f'\n{\"模型版本\":<28} | {\"KodCode (200)\":<14} | {\"HumanEval (164)\":<16} | {\"MBPP (427)\":<12} | {\"791题总通过率\":<12}')
print('-' * 95)

for name, prefix in exps:
    kod_f = prefix + 'kodcode.json'
    he_f = prefix + 'humaneval.json'
    mbpp_f = prefix + 'mbpp.json'
    
    kod_p, he_p, mbpp_p = 'N/A', 'N/A', 'N/A'
    kod_c, he_c, mbpp_c = 0, 0, 0
    
    if os.path.exists(kod_f):
        try:
            d = json.load(open(kod_f))
            kod_p = f\"{d.get('pass@1', 0)*100:.2f}% ({d.get('passed_tasks', 0)}/200)\"
            kod_c = d.get('passed_tasks', 0)
        except: pass
        
    if os.path.exists(he_f):
        try:
            d = json.load(open(he_f))
            he_p = f\"{d.get('pass@1', 0)*100:.2f}% ({round(d.get('pass@1', 0)*164)}/164)\"
            he_c = round(d.get('pass@1', 0)*164)
        except: pass
        
    if os.path.exists(mbpp_f):
        try:
            d = json.load(open(mbpp_f))
            mbpp_p = f\"{d.get('pass@1', 0)*100:.2f}% ({d.get('passed_tasks', 0)}/427)\"
            mbpp_c = d.get('passed_tasks', 0)
        except: pass
        
    total_c = kod_c + he_c + mbpp_c
    total_p = f\"{total_c/791*100:.2f}% ({total_c}/791)\"
    print(f'{name:<28} | {kod_p:<14} | {he_p:<16} | {mbpp_p:<12} | {total_p:<12}')
print('=' * 95)
"
