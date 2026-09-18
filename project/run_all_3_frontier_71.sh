#!/bin/bash
# run_all_3_frontier_71.sh — 三大前沿稠密奖励模型 (VeRPO, TIPS, DHRCL) 71 步全自动无人值守连续训练与评测总控流水线
# 
# 覆盖模型（三大独立前沿消融）：
#   1. Model 1: VeRPO (纯超线性测试用例密度校准，arXiv:2601.03525)
#   2. Model 2: TIPS (纯势能差分奖励塑形 PBRS 双曲正切平滑，ICLR 2026)
#   3. Model 3: DHRCL (纯三阶段课程退火与时序调度，arXiv:2607.26457)
#
# 全自动流水线（完全免人工接管）：
#   依次执行：71 步 PPO 训练 -> 释放 8 卡 NPU 显存 -> FSDP 合并导出 HF -> 三大基准 791 题全量评测 -> 自动生成终极对比大榜

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

# 1. 激活虚拟环境
if [ -f "$HOME_DIR/project/envs/verl_env/bin/activate" ]; then
    source "$HOME_DIR/project/envs/verl_env/bin/activate"
fi

# 2. 检查并应用 vLLM-Ascend 补丁
if [ -f "$HOME_DIR/project/patch_vllm_ascend.py" ]; then
    python3 "$HOME_DIR/project/patch_vllm_ascend.py"
fi

# 3. 环境变量配置
export TORCHDYNAMO_DISABLE=1
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_OP_EXPANSION_MODE="AIV"
export VERL_REWARD_WORKERS=64
export TRL_EXPERIMENTAL_SILENCE=1

MODEL="${HOME_DIR}/Qwen2.5-7B-Instruct"
DATA="${HOME_DIR}/project/data/rlvr"

mkdir -p eval_results
mkdir -p logs
mkdir -p checkpoints

function clean_npu() {
    echo ">>> [系统重置] 正在终止残留 Ray/NPU 进程并重置显存空间 ..."
    ray stop --force > /dev/null 2>&1 || true
    pkill -9 -f "verl" > /dev/null 2>&1 || true
    pkill -9 -f "ray" > /dev/null 2>&1 || true
    rm -rf /tmp/rlvr_dhrcl_calls 2>/dev/null || true
    sleep 4
}

function run_single_frontier_ablation() {
    local MODEL_NUM="$1"
    local EXP_TAG="$2"
    local DISPLAY_NAME="$3"
    local REWARD_FILE="$4"
    local EXP_NAME="d4_full_7b_${EXP_TAG}"
    local CKPT_DIR="${HOME_DIR}/project/checkpoints/${EXP_NAME}"
    local TARGET_HF="${HOME_DIR}/project/checkpoints/${EXP_NAME}_hf"
    local EVAL_PREFIX="${HOME_DIR}/project/eval_results/eval_full_${EXP_TAG}_"

    echo ""
    echo "================================================================="
    echo "  🚀 [无人值守流水线 ${MODEL_NUM}/3 启动] ${DISPLAY_NAME}"
    echo "  - 奖励函数: ${REWARD_FILE}"
    echo "  - 检查点目录: ${CKPT_DIR}"
    echo "  - 目标 HF 权重: ${TARGET_HF}"
    echo "  - 启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================="

    # 1. 彻底清空显存与残留进程
    clean_npu

    # 2. 71 步全量 PPO 训练
    echo ">>> [1/3] 开始 71 步 PPO 训练 (1.0 Epoch, 4,518 题) ..."
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

    # 3. 训练完毕，立即释放 8 卡 NPU 训练显存
    clean_npu

    # 4. FSDP 权重合并导出为原生 HuggingFace 格式
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

    # 5. 三大基准 791 题全量自动化评测
    echo ">>> [3/3] 启动三大基准 791 题全量评测 (KodCode 200 + HumanEval 164 + MBPP 427) ..."
    python3 eval_test_set.py --model "$TARGET_HF" --output "${EVAL_PREFIX}kodcode.json"
    python3 eval_humaneval.py --model "$TARGET_HF" --output "${EVAL_PREFIX}humaneval.json"
    python3 eval_mbpp.py --model "$TARGET_HF" --output "${EVAL_PREFIX}mbpp.json"

    echo ">>> [完成] ${DISPLAY_NAME} 训练、HF导出与 791 题评测全部通过！"
    clean_npu
}

TOTAL_START=$(date +%s)
echo "================================================================="
echo "  🚀 启动三大前沿稠密奖励 (VeRPO / TIPS / DHRCL) 全自动无人值守批处理"
echo "  - 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  - 运行机制: 3 个模型依次串行执行，中途零人工干预"
echo "  - 预计总耗时: 约 2.5 ~ 2.8 小时"
echo "================================================================="

# 第 1 棒：VeRPO 纯超线性密度校准 (71 步)
run_single_frontier_ablation 1 "verpo" "VeRPO 71步 (超线性密度校准 arXiv:2601.03525)" "rewards/code_rlvr_verpo.py"

# 第 2 棒：TIPS 纯势能奖励塑形 PBRS (71 步)
run_single_frontier_ablation 2 "tips" "TIPS 71步 (连续势能奖励塑形 ICLR 2026)" "rewards/code_rlvr_tips.py"

# 第 3 棒：DHRCL 纯三阶段课程退火 (71 步)
run_single_frontier_ablation 3 "dhrcl" "DHRCL 71步 (三阶段课程退火调度 arXiv:2607.26457)" "rewards/code_rlvr_dhrcl.py"

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "================================================================="
echo "  🎉🎉🎉 三大前沿稠密奖励模型训练与 791 题评测全部圆满大功告成！"
echo "  - 总耗时: $((TOTAL_ELAPSED / 3600)) 小时 $(((TOTAL_ELAPSED % 3600) / 60)) 分 $((TOTAL_ELAPSED % 60)) 秒"
echo "================================================================="

# 打印最终总结大榜（汇总三大前沿成果，并与历史最佳 Exp 4、Exp 1、Base 深度对比）
python3 -c "
import json, glob, os

exps = [
    ('Base (Qwen2.5-7B-Instruct)', None, '74.50%', '79.27%', '69.56%', '72.82%', '74.44%'),
    ('Exp 1: Neg-Penalty 71步', 'eval_results/eval_full_neg_penalty_', '77.00%', '84.76%', '74.00%', '77.00%', '78.59%'),
    ('Exp 2: Sparse-RLVR 71步', 'eval_results/eval_full_sparse_', '78.50%', '80.49%', '75.41%', '77.24%', '78.13%'),
    ('Exp 4: Len-Efficiency 71步 🏆', 'eval_results/eval_full_len_efficiency_', '81.00%', '84.76%', '75.88%', '79.01%', '80.55%'),
    ('---------------------------', None, '------', '------', '------', '------', '------'),
    ('Frontier 1: VeRPO 71步', 'eval_results/eval_full_verpo_', None, None, None, None, None),
    ('Frontier 2: TIPS 71步', 'eval_results/eval_full_tips_', None, None, None, None, None),
    ('Frontier 3: DHRCL 71步', 'eval_results/eval_full_dhrcl_', None, None, None, None, None),
]

print(f'\n{\"模型架构 / 实验版本\":<32} | {\"KodCode (200)\":<14} | {\"HumanEval (164)\":<16} | {\"MBPP (427)\":<12} | {\"791题全量通过率\":<14} | {\"Macro Avg\":<10}')
print('=' * 115)

for item in exps:
    name, prefix = item[0], item[1]
    if prefix is None:
        if name.startswith('-'):
            print('-' * 115)
        else:
            print(f'{name:<32} | {item[2]:<14} | {item[3]:<16} | {item[4]:<12} | {item[5]:<14} | {item[6]:<10}')
        continue
        
    kod_f = prefix + 'kodcode.json'
    he_f = prefix + 'humaneval.json'
    mbpp_f = prefix + 'mbpp.json'
    
    kod_acc, he_acc, mbpp_acc = 0.0, 0.0, 0.0
    kod_p, he_p, mbpp_p = '未开始/中断', '未开始/中断', '未开始/中断'
    kod_c, he_c, mbpp_c = 0, 0, 0
    
    if os.path.exists(kod_f):
        try:
            d = json.load(open(kod_f))
            kod_acc = d.get('pass@1', 0.0)
            kod_c = d.get('passed_tasks', 0)
            kod_p = f\"{kod_acc*100:.2f}% ({kod_c}/200)\"
        except: pass
        
    if os.path.exists(he_f):
        try:
            d = json.load(open(he_f))
            he_acc = d.get('pass@1', 0.0)
            he_c = round(he_acc * 164)
            he_p = f\"{he_acc*100:.2f}% ({he_c}/164)\"
        except: pass
        
    if os.path.exists(mbpp_f):
        try:
            d = json.load(open(mbpp_f))
            mbpp_acc = d.get('pass@1', 0.0)
            mbpp_c = d.get('passed_tasks', 0)
            mbpp_p = f\"{mbpp_acc*100:.2f}% ({mbpp_c}/427)\"
        except: pass
        
    total_c = kod_c + he_c + mbpp_c
    total_p = f\"{total_c/791*100:.2f}% ({total_c}/791)\" if total_c > 0 else 'N/A'
    macro_p = f\"{(kod_acc + he_acc + mbpp_acc)/3*100:.2f}%\" if (kod_acc > 0 or he_acc > 0 or mbpp_acc > 0) else 'N/A'
    
    print(f'{name:<32} | {kod_p:<14} | {he_p:<16} | {mbpp_p:<12} | {total_p:<14} | {macro_p:<10}')

print('=' * 115)
"
