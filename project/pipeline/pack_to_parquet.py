# -*- coding: utf-8 -*-
"""pack_to_parquet.py —— 数据集打包与格式转换（D3 任务 1）

功能：
    将 D2 阶段产出的黄金数据池与独立评测集（JSONL）转换为 verl 0.6.1 标准 Parquet 格式：
    1. data/rl_pool.jsonl (2,938 题)  --> data/rlvr/train_full.parquet (全量训练集)
    2. data/rl_pool.jsonl[:smoke_size] --> data/rlvr/train_smoke.parquet (极速验证集，默认 128 题)
    3. data/heldout.jsonl (200 题)    --> data/rlvr/test.parquet (独立未污染评测集)

Parquet 字段规范（与 verl RLHFDataset 及 NaiveRewardManager 严格对齐）：
    - data_source:  "kodcode"
    - ability:      "coding"
    - prompt:       [{"role": "user", "content": prompt_str}] (兼容 return_raw_chat=True)
    - reward_model: {"style": "rule", "ground_truth": test_code}
    - extra_info:   {"question_id": id, "test": test_code, "solution": solution_code}
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List
import pandas as pd


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """读取 JSONL 文件并返回字典列表。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到数据集文件: {file_path}")
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(data)
            except json.JSONDecodeError as exc:
                print(f"[WARN] 第 {line_no} 行解析 JSON 失败: {exc}")
    return records


def format_for_verl(records: List[Dict[str, Any]], split_name: str = "train") -> List[Dict[str, Any]]:
    """将题目字典转换为 verl 所需的 schema 字典。"""
    formatted_data = []
    for idx, item in enumerate(records):
        prompt_text = item.get("prompt", "")
        test_code = item.get("test", "")
        solution_code = item.get("solution", "")
        qid = item.get("question_id", f"{split_name}_{idx:06d}")

        # prompt 结构：符合 OpenAI / HuggingFace 对话规范，供 return_raw_chat 使用
        prompt_dialog = [{"role": "user", "content": prompt_text}]

        row = {
            "data_source": "kodcode",
            "ability": "coding",
            "prompt": prompt_dialog,
            "reward_model": {
                "style": "rule",
                "ground_truth": test_code,
            },
            "extra_info": {
                "question_id": qid,
                "test": test_code,
                "solution": solution_code,
                "split": split_name,
                "index": idx,
            },
        }
        formatted_data.append(row)
    return formatted_data


def save_as_parquet(data: List[Dict[str, Any]], output_path: str) -> None:
    """转换并写入 Parquet 文件，进行完整性校验。"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_parquet(output_path, engine="pyarrow", index=False)

    # 立即回读验证
    verify_df = pd.read_parquet(output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [OK] 成功导出: {output_path}")
    print(f"       样本总数: {len(verify_df)} 条 | 大小: {file_size_mb:.2f} MB")
    print(f"       列名列表: {list(verify_df.columns)}")


def main():
    parser = argparse.ArgumentParser(description="将 RLVR JSONL 数据集转换为 verl Parquet 格式")
    parser.add_argument("--pool", type=str, default="data/rl_pool.jsonl", help="黄金训练池 JSONL 路径")
    parser.add_argument("--heldout", type=str, default="data/heldout.jsonl", help="隔离考试集 JSONL 路径")
    parser.add_argument("--out-dir", type=str, default="data/rlvr", help="Parquet 输出目标目录")
    parser.add_argument("--smoke-size", type=int, default=128, help="D3 冒烟验证集抽样条数 (默认 128)")
    args = parser.parse_args()

    print("=================================================================")
    print("[D3-Task1] 开始打包 RLVR 数据集为 verl Parquet 标准格式")
    print("=================================================================")

    # 1. 转换训练集
    print(f"\n>>> [1/3] 读取训练黄金池: {args.pool} ...")
    pool_records = load_jsonl(args.pool)
    print(f"    共读取到 {len(pool_records)} 条训练样本。")

    train_full_formatted = format_for_verl(pool_records, split_name="train_full")
    train_full_path = os.path.join(args.out_dir, "train_full.parquet")
    save_as_parquet(train_full_formatted, train_full_path)

    # 2. 裁剪冒烟集 (用于 D3 极速闭环验证)
    smoke_count = min(args.smoke_size, len(pool_records))
    print(f"\n>>> [2/3] 制作 D3 专属冒烟验证集 (前 {smoke_count} 条) ...")
    smoke_records = pool_records[:smoke_count]
    train_smoke_formatted = format_for_verl(smoke_records, split_name="train_smoke")
    train_smoke_path = os.path.join(args.out_dir, "train_smoke.parquet")
    save_as_parquet(train_smoke_formatted, train_smoke_path)

    # 3. 转换隔离评测集
    print(f"\n>>> [3/3] 读取隔离考试集: {args.heldout} ...")
    heldout_records = load_jsonl(args.heldout)
    print(f"    共读取到 {len(heldout_records)} 条独立评测样本。")

    heldout_formatted = format_for_verl(heldout_records, split_name="test")
    test_path = os.path.join(args.out_dir, "test.parquet")
    save_as_parquet(heldout_formatted, test_path)

    print("\n=================================================================")
    print("[OK] 全部 Parquet 打包完成！清单如下：")
    print(f"  1. 完整训练集: {train_full_path} ({len(pool_records)} 条)")
    print(f"  2. 极速验证集: {train_smoke_path} ({smoke_count} 条 - D3 优先使用)")
    print(f"  3. 隔离评测集: {test_path} ({len(heldout_records)} 条)")
    print("=================================================================")


if __name__ == "__main__":
    main()
