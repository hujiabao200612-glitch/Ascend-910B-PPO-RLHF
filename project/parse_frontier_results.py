# -*- coding: utf-8 -*-
"""parse_frontier_results.py — 提取并格式化三大前沿模型真实评测成绩与对比大榜

支持双重数据源容灾：
1. 优先读取 eval_results/ 下的评测明细文件 (kodcode.json, humaneval.json_results.jsonl, mbpp.json)
2. 自动从全量运行日志 train_3_frontier.log 中提取真实控制台输出
"""

import json
import os
import re
import sys
from glob import glob

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(PROJ_DIR, "train_3_frontier.log")
EVAL_DIR = os.path.join(PROJ_DIR, "eval_results")

# 预先录入的基线对照数据
BASELINES = [
    {
        "name": "Base (Qwen2.5-7B-Instruct)",
        "kod": "74.50% (149/200)", "kod_acc": 74.50, "kod_c": 149,
        "he": "79.27% (130/164)", "he_acc": 79.27, "he_c": 130,
        "mbpp": "69.56% (297/427)", "mbpp_acc": 69.56, "mbpp_c": 297,
        "total": "72.82% (576/791)", "total_c": 576,
        "macro": "74.44%",
        "avg_len": "382.1",
    },
    {
        "name": "Exp 1: Neg-Penalty 71步",
        "kod": "77.00% (154/200)", "kod_acc": 77.00, "kod_c": 154,
        "he": "84.76% (139/164) 🏆", "he_acc": 84.76, "he_c": 139,
        "mbpp": "74.00% (316/427)", "mbpp_acc": 74.00, "mbpp_c": 316,
        "total": "77.00% (609/791)", "total_c": 609,
        "macro": "78.59%",
        "avg_len": "208.5",
    },
    {
        "name": "Exp 2: Sparse-RLVR 71步",
        "kod": "78.50% (157/200)", "kod_acc": 78.50, "kod_c": 157,
        "he": "80.49% (132/164)", "he_acc": 80.49, "he_c": 132,
        "mbpp": "75.41% (322/427)", "mbpp_acc": 75.41, "mbpp_c": 322,
        "total": "77.24% (611/791)", "total_c": 611,
        "macro": "78.13%",
        "avg_len": "219.0",
    },
    {
        "name": "Exp 4: Len-Efficiency 71步 🥇",
        "kod": "81.00% (162/200) ⚡", "kod_acc": 81.00, "kod_c": 162,
        "he": "84.76% (139/164) 🏆", "he_acc": 84.76, "he_c": 139,
        "mbpp": "75.88% (324/427) 🏆", "mbpp_acc": 75.88, "mbpp_c": 324,
        "total": "79.01% (625/791) 🥇", "total_c": 625,
        "macro": "80.55% 🥇",
        "avg_len": "162.4",
    },
]

FRONTIER_MODELS = [
    ("Frontier 1: VeRPO 71步 (超线性校准)", "verpo"),
    ("Frontier 2: TIPS 71步 (连续势能PBRS)", "tips"),
    ("Frontier 3: DHRCL 71步 (三阶段退火)", "dhrcl"),
]


def extract_kodcode(tag: str, log_content: str):
    json_path = os.path.join(EVAL_DIR, f"eval_full_{tag}_kodcode.json")
    acc, count, total, avg_len = None, None, 200, None
    if os.path.exists(json_path):
        try:
            d = json.load(open(json_path, "r", encoding="utf-8"))
            acc = d.get("pass_at_1", d.get("pass@1", None))
            if acc is not None and acc <= 1.0:
                acc = acc * 100.0
            count = d.get("perfect_count", d.get("passed_tasks", None))
            total = d.get("n_samples", 200)
            avg_len = d.get("avg_len", None)
        except Exception:
            pass

    if acc is None and log_content:
        pattern = rf"{tag}.*?Pass@1 满分通过率:\s*([\d\.]+)%\s*\((\d+)/(\d+)"
        m = re.search(pattern, log_content, re.DOTALL)
        if m:
            acc = float(m.group(1))
            count = int(m.group(2))
            total = int(m.group(3))

    return acc, count, total, avg_len


def extract_humaneval(tag: str, log_content: str):
    res_path = os.path.join(EVAL_DIR, f"eval_full_{tag}_humaneval.json_results.jsonl")
    acc, count, total = None, None, 164
    if os.path.exists(res_path):
        try:
            lines = open(res_path, "r", encoding="utf-8").readlines()
            succ = sum(1 for line in lines if json.loads(line).get("passed", False))
            count = succ
            total = len(lines) if len(lines) > 0 else 164
            acc = (float(count) / float(total)) * 100.0
        except Exception:
            pass

    if acc is None and log_content:
        # 寻找对应的 pass@1
        pattern = rf"eval_full_{tag}_humaneval\.json.*?\{{'pass@1':\s*([\d\.]+)\}}"
        m = re.search(pattern, log_content, re.DOTALL)
        if m:
            acc = float(m.group(1)) * 100.0
            count = int(round(float(m.group(1)) * 164))

    return acc, count, total


def extract_mbpp(tag: str, log_content: str):
    acc, count, total = None, None, 427
    if log_content:
        pattern = rf"eval_full_{tag}_mbpp\.json.*?Google MBPP.*?Pass@1 满分通过率:\s*([\d\.]+)%\s*\((\d+)/(\d+)"
        m = re.search(pattern, log_content, re.DOTALL)
        if m:
            acc = float(m.group(1))
            count = int(m.group(2))
            total = int(m.group(3))

    # 如果日志里没正则到，尝试直接用 mbpp.json 和 compute_score 判分
    if acc is None:
        mbpp_json = os.path.join(EVAL_DIR, f"eval_full_{tag}_mbpp.json")
        if os.path.exists(mbpp_json):
            try:
                cur_dir = os.path.abspath(PROJ_DIR)
                sys.path.insert(0, os.path.join(cur_dir, 'pipeline'))
                sys.path.insert(0, os.path.join(cur_dir, 'rewards'))
                from code_rlvr import compute_score
                from concurrent.futures import ThreadPoolExecutor

                lines = open(mbpp_json, "r", encoding="utf-8").readlines()
                items = [json.loads(line) for line in lines]
                with ThreadPoolExecutor(max_workers=64) as ex:
                    passed = list(ex.map(lambda it: compute_score(data_source='kodcode', solution_str=it['response'], ground_truth=it['test']) >= 1.0, items))
                count = sum(passed)
                total = len(passed)
                acc = (float(count) / float(total)) * 100.0
            except Exception:
                pass

    return acc, count, total


def main():
    log_content = ""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                log_content = f.read()
        except Exception:
            pass

    print("=" * 118)
    print("  🏆 华为昇腾 910B (8-NPU) 71 步全量实验与前沿三大稠密模型终极大榜对比")
    print("=" * 118)
    print(f"{'模型架构 / 实验版本':<35} | {'KodCode (200)':<16} | {'HumanEval (164)':<16} | {'MBPP (427)':<15} | {'791题总通过率':<16} | {'Macro Avg':<10}")
    print("-" * 118)

    # 1. 打印基线历史最佳成绩
    for b in BASELINES:
        print(f"{b['name']:<35} | {b['kod']:<16} | {b['he']:<16} | {b['mbpp']:<15} | {b['total']:<16} | {b['macro']:<10}")

    print("-" * 118)

    # 2. 解析提取前沿三大模型
    for display_name, tag in FRONTIER_MODELS:
        k_acc, k_c, k_tot, avg_len = extract_kodcode(tag, log_content)
        h_acc, h_c, h_tot = extract_humaneval(tag, log_content)
        m_acc, m_c, m_tot = extract_mbpp(tag, log_content)

        kod_str = f"{k_acc:.2f}% ({k_c}/{k_tot})" if k_acc is not None else "待提取"
        he_str = f"{h_acc:.2f}% ({h_c}/{h_tot})" if h_acc is not None else "待提取"
        mbpp_str = f"{m_acc:.2f}% ({m_c}/{m_tot})" if m_acc is not None else "待提取"

        if k_c is not None and h_c is not None and m_c is not None:
            tot_c = k_c + h_c + m_c
            tot_str = f"{tot_c/791*100:.2f}% ({tot_c}/791)"
            macro_str = f"{(k_acc + h_acc + m_acc)/3:.2f}%"
        else:
            tot_str = "计算中"
            macro_str = "计算中"

        print(f"{display_name:<35} | {kod_str:<16} | {he_str:<16} | {mbpp_str:<15} | {tot_str:<16} | {macro_str:<10}")

    print("=" * 118)


if __name__ == "__main__":
    main()
