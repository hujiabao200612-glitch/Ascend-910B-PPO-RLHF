# -*- coding: utf-8 -*-
"""eval_test_set.py — 独立评测集 (test.parquet, 200题) 全自动 Pass@1 评测脚本

特性：
1. 采用 vLLM 高性能离线推理，200 道题在昇腾 910B 上约 15~20 秒完成生成。
2. 自动注入 Qwen 对话模板 (<|im_start|>user...)，与 PPO 训练期分布严格对齐。
3. 挂载 64 线程 CPU 沙箱并发评测，5 秒内完成全部 200 题 pytest 真实打分。
4. 输出清晰的 Pass@1 满分率、平均分与详细报告 JSON。
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# 导入奖励沙箱判分内核
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PIPE_DIR = os.path.join(CUR_DIR, "pipeline")
REWARD_DIR = os.path.join(CUR_DIR, "rewards")
for p in [CUR_DIR, PIPE_DIR, REWARD_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from code_rlvr import compute_score


def parse_args():
    parser = argparse.ArgumentParser(description="独立评测集 Pass@1 自动化打分器")
    parser.add_argument("--model", type=str, required=True, help="待评测模型路径 (HuggingFace 格式目录)")
    parser.add_argument("--data", type=str, default="data/rlvr/test.parquet", help="评测集 test.parquet 路径")
    parser.add_argument("--output", type=str, default=None, help="详细结果 JSON 导出路径")
    parser.add_argument("--max-tokens", type=int, default=768, help="最大生成 Token 长度")
    parser.add_argument("--workers", type=int, default=64, help="沙箱并发线程数")
    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = os.path.abspath(args.model)
    data_file = os.path.abspath(args.data)
    model_name = os.path.basename(os.path.normpath(model_dir))

    if not os.path.exists(model_dir):
        print(f"[ERROR] 找不到待评测模型目录: {model_dir}")
        sys.exit(1)
    if not os.path.exists(data_file):
        print(f"[ERROR] 找不到测试集文件: {data_file}")
        sys.exit(1)

    out_file = args.output or f"eval_result_{model_name}.json"

    print("=================================================================")
    print(f"[Eval Engine] 启动 200 题独立测试集 Pass@1 评测流水线")
    print(f"  - 评测模型: {model_dir}")
    print(f"  - 测试数据: {data_file}")
    print(f"  - 并发线程: {args.workers} (CPU 沙箱)")
    print("=================================================================")

    # 1. 加载测试集
    df = pd.read_parquet(data_file)
    n_samples = len(df)
    print(f"
>>> [1/3] 成功加载测试集: {n_samples} 道代码题")

    # 2. 构造严格对齐的 Prompt
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    raw_prompts = []
    tests = []
    extra_infos = []

    for _, row in df.iterrows():
        p = row["prompt"]
        if isinstance(p, (list, tuple)):
            dialog = list(p)
        else:
            dialog = [{"role": "user", "content": str(p)}]

        formatted = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
        raw_prompts.append(formatted)

        # 提取测试代码
        rm = row["reward_model"]
        test_code = rm.get("ground_truth", "") if isinstance(rm, dict) else str(rm)
        tests.append(test_code)
        extra_infos.append(row.get("extra_info", {}))

    # 3. vLLM 批量贪心推理 (Greedy Decoding for Pass@1)
    print(f"
>>> [2/3] 启动 vLLM 批量推理生成 ({n_samples} 题) ...")
    t0_gen = time.time()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        top_p=1.0,
    )

    llm = LLM(
        model=model_dir,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        trust_remote_code=True,
    )

    outputs = llm.generate(raw_prompts, sampling_params)
    generated_texts = [o.outputs[0].text for o in outputs]
    gen_duration = time.time() - t0_gen
    print(f"    vLLM 推理完成！总耗时: {gen_duration:.2f}s (平均 {gen_duration/n_samples*1000:.1f}ms/题)")

    # 4. 多线程 CPU 沙箱并发跑 pytest
    print(f"
>>> [3/3] 启动 {args.workers} 线程沙箱并发判分 ...")
    t0_eval = time.time()

    eval_items = []
    for i in range(n_samples):
        eval_items.append((
            i,
            generated_texts[i],
            tests[i],
            extra_infos[i],
        ))

    def _eval_worker(item):
        idx, resp, test, extra = item
        score = compute_score(
            data_source="kodcode",
            solution_str=resp,
            ground_truth=test,
            extra_info=extra,
        )
        return idx, score

    with ThreadPoolExecutor(max_workers=min(args.workers, n_samples)) as executor:
        scored_pairs = list(executor.map(_eval_worker, eval_items))

    eval_duration = time.time() - t0_eval
    print(f"    沙箱判分完成！总耗时: {eval_duration:.2f}s (平均 {eval_duration/n_samples:.2f}s/题)")

    # 5. 统计核心量化指标
    scores = [pair[1] for pair in sorted(scored_pairs, key=lambda x: x[0])]

    perfect_count = sum(1 for s in scores if s >= 1.0)
    partial_count = sum(1 for s in scores if 0.1 < s < 1.0)
    format_only_count = sum(1 for s in scores if 0.05 < s <= 0.1)
    zero_count = sum(1 for s in scores if s <= 0.05)

    pass_at_1 = (perfect_count / n_samples) * 100.0
    mean_reward = sum(scores) / n_samples
    avg_len = sum(len(tokenizer.encode(t)) for t in generated_texts) / n_samples

    print("
=================================================================")
    print(f"                    评测成绩核算 (Pass@1 Report)                  ")
    print("=================================================================")
    print(f"  模型名称:               {model_name}")
    print(f"  测试题目总数:           {n_samples}")
    print(f"  Pass@1 满分通过率:      {pass_at_1:.2f}% ({perfect_count}/{n_samples} 题)")
    print(f"  平均奖励得分 (Score):   {mean_reward:.4f} / 1.0000")
    print(f"  部分用例通过题数:       {partial_count} 题 ({(partial_count/n_samples)*100:.1f}%)")
    print(f"  仅格式正确但用例全挂:   {format_only_count} 题")
    print(f"  无代码块或严重报错:     {zero_count} 题")
    print(f"  平均生成代码长度:       {avg_len:.1f} Tokens")
    print(f"  总评测耗时:             {gen_duration + eval_duration:.2f} 秒")
    print("=================================================================")

    # 保存详细报告
    records = []
    for i in range(n_samples):
        records.append({
            "idx": i,
            "prompt": raw_prompts[i],
            "response": generated_texts[i],
            "test": tests[i],
            "score": scores[i],
            "status": "PASS" if scores[i] >= 1.0 else ("PARTIAL" if scores[i] > 0.1 else "FAIL")
        })

    report = {
        "model": model_dir,
        "n_samples": n_samples,
        "pass_at_1": pass_at_1,
        "mean_reward": mean_reward,
        "perfect_count": perfect_count,
        "partial_count": partial_count,
        "zero_count": zero_count,
        "avg_len": avg_len,
        "details": records,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"
[OK] 完整评测明细报告已写入: {out_file}
")


if __name__ == "__main__":
    main()
