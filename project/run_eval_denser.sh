#!/bin/bash
# run_eval_denser.sh — PPO-DenseR (Exp 5) 权重合并与三大权威基准全自动评测脚本
# 评测矩阵：三大基准 791 题（KodCode 200 题 + OpenAI HumanEval 164 题 + Google MBPP 427 题）

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$CURRENT_DIR"

# 1. 激活环境
if [ -f "envs/verl_env/bin/activate" ]; then
    source "envs/verl_env/bin/activate"
fi

export TORCHDYNAMO_DISABLE=1
export VLLM_ASCEND_ENABLE_NZ=0
export HCCL_OP_EXPANSION_MODE="AIV"
export VERL_REWARD_WORKERS=64

CKPT_SRC="checkpoints/d4_ablation_7b_denser/global_step_50/actor"
TARGET_HF="checkpoints/d4_ablation_7b_denser_hf"
BASE_MODEL="/data/home/1120250939/Qwen2.5-7B-Instruct"

mkdir -p "$TARGET_HF"
mkdir -p "eval_results"

echo "================================================================="
echo "[PPO-DenseR Exp 5] 第一步：将 FSDP 分片权重合并导出为原生 HuggingFace 格式"
echo "  - 输入分片目录: $CKPT_SRC"
echo "  - 输出模型目录: $TARGET_HF"
echo "================================================================="

# 优先调用 verl.model_merger，带自动回退
python3 -c "
import os, sys
ckpt_src = '$CKPT_SRC'
target_hf = '$TARGET_HF'

try:
    import verl.model_merger as mm
    cmd = f'{sys.executable} -m verl.model_merger merge --backend fsdp --local_dir {ckpt_src} --target_dir {target_hf}'
    print('>>> 执行命令:', cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError('model_merger exit non-zero')
except Exception as e:
    print('>>> 尝试使用 legacy / fsdp_to_hf 合并器...')
    cmd_fallback = f'{sys.executable} -m verl.utils.checkpoint.fsdp_to_hf --fsdp_checkpoint_dir {ckpt_src} --target_dir {target_hf}'
    os.system(cmd_fallback)
"

# 补充 Tokenizer 与配置文件（若合并器未自动复制）
cp -n "$BASE_MODEL"/tokenizer* "$TARGET_HF"/ 2>/dev/null || true
cp -n "$BASE_MODEL"/vocab* "$TARGET_HF"/ 2>/dev/null || true
cp -n "$BASE_MODEL"/merges* "$TARGET_HF"/ 2>/dev/null || true
cp -n "$BASE_MODEL"/chat_template* "$TARGET_HF"/ 2>/dev/null || true

echo -e "\n================================================================="
echo "[PPO-DenseR Exp 5] 第二步：评测 KodCode 独立测试集 (Held-Out 200 题)"
echo "================================================================="
python3 eval_test_set.py \
    --model "$TARGET_HF" \
    --output eval_results/eval_denser_kodcode.json

echo -e "\n================================================================="
echo "[PPO-DenseR Exp 5] 第三步：评测 OpenAI HumanEval (164 题算法基准)"
echo "================================================================="
python3 eval_humaneval.py \
    --model "$TARGET_HF" \
    --output eval_results/eval_denser_humaneval.json

echo -e "\n================================================================="
echo "[PPO-DenseR Exp 5] 第四步：评测 Google MBPP Sanitized (427 题函数基准)"
echo "================================================================="
python3 eval_mbpp.py \
    --model "$TARGET_HF" \
    --output eval_results/eval_denser_mbpp.json

echo -e "\n================================================================="
echo "🎉 PPO-DenseR (Exp 5) 三大权威基准全量实测全部完成！"
echo "  - KodCode 详情:   eval_results/eval_denser_kodcode.json"
echo "  - HumanEval 详情: eval_results/eval_denser_humaneval.json"
echo "  - MBPP 详情:      eval_results/eval_denser_mbpp.json"
echo "================================================================="
