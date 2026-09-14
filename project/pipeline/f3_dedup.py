# -*- coding: utf-8 -*-
"""f3_dedup.py —— 筛 3：去重 + 截长 + 泄漏过滤（任务书 §5.6）。

为什么需要这一筛：step2_kept 里仍可能有三类不该进训练池的题——
  1) 完全重复的题面（同一 question 被不同子集/风格重复收录）；
  2) prompt 过长的题（超过模型上下文友好上限，训练噪声大）；
  3) 与公开基准高度相似的题（benchmark_similarity > 0.9，存在评测泄漏风险，会虚高 RLVR 指标）。
这一筛把三类题剔除，产出最终可用的候选池 step3_verified_pool.jsonl（目标 5~8k 条）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

try:
    import config
except ImportError:
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config


def question_md5(question) -> str:
    """对题面算 md5，作为去重键（§5.6）。

    为什么用 question 而不是 prompt：任务书 §5.6 明确按 question 去重；prompt 是
    question 套固定模板后的产物，question 相同则 prompt 必相同，去重口径一致。
    题面缺失时回落到空串（极端情况下多个缺题面题会聚到同一哈希，被去重保留第一条），
    不单独引入任务书未列的淘汰规则。
    """
    text = question if isinstance(question, str) else ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def classify(rec: dict, seen: set) -> str | None:
    """返回该记录的淘汰原因；返回 None 表示保留。

    检查顺序与 §5.6 行文顺序一致，首个命中即定原因：
      1) duplicate     —— question 的 md5 已在「已保留集合」中（先到先得：仅对真正保留过的
                           题记哈希，故「首个有效副本」会存活，后续副本才被去重）；
      2) too_long      —— len(prompt) > MAX_PROMPT_LEN（§5.6：>2000 字符弃用）；
      3) leakage_risk  —— benchmark_similarity 存在且 > 阈值（§5.6：>0.9 视为泄漏）。

    字段缺失/不可解析的情形（benchmark_similarity 为 None 或无法转 float、prompt 缺失）
    一律按「保留」处理——这与 §5.6「字段缺失为 None 时保留」一致；泄漏判定只针对可确认
    严格大于 0.9 的值，reason 名 leakage_risk 也仅在此情形生效（§5.6 括号说明）。
    """
    qmd5 = question_md5(rec.get("question"))
    if qmd5 in seen:
        return "duplicate"

    prompt = rec.get("prompt")
    if not isinstance(prompt, str):
        prompt = ""
    if len(prompt) > config.MAX_PROMPT_LEN:
        return "too_long"

    sim = rec.get("benchmark_similarity")
    if sim is not None:
        try:
            sim_f = float(sim)
        except (TypeError, ValueError):
            sim_f = None
        if sim_f is not None and sim_f > config.BENCHMARK_SIMILARITY_THRESHOLD:
            return "leakage_risk"

    return None


def run(input_path: str, output_path: str, rejects_path: str) -> None:
    """主流程：逐行流式读 step2_kept，按 classify 分流，写出 verified pool 与 rejects。

    为什么流式：原料 15,000 条（f2 后仍有数千~上万），整文件载入没必要也没必要占内存；
    逐行读、边判边写，内存恒定。幂等：重跑会重写整个 output / rejects，结果一致。
    """
    seen: set = set()
    stats = {
        "read": 0,
        "kept": 0,
        "bad_lines": 0,
        "duplicate": 0,
        "too_long": 0,
        "leakage_risk": 0,
    }

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout, \
         open(rejects_path, "w", encoding="utf-8") as frej:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["read"] += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # 坏行不静默丢弃：记一笔，但本筛不负责修复，下游行数对得上即可。
                stats["bad_lines"] += 1
                continue

            reason = classify(rec, seen)
            if reason is None:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                # 仅当真正保留时才登记哈希：保证「首个有效副本」存活，
                # 而因 too_long / leakage 被弃的题不会挡住后续同题有效副本。
                seen.add(question_md5(rec.get("question")))
                stats["kept"] += 1
            else:
                # 淘汰题带上 drop_reason 供人工复核（§6 验收闸门强调尺子刻度要人审）。
                rec["drop_reason"] = reason
                frej.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats[reason] += 1

    _print_stats(stats)


def _print_stats(stats: dict) -> None:
    """结尾打印统计：进/出/各淘汰原因计数（§7.5）。"""
    dropped = stats["duplicate"] + stats["too_long"] + stats["leakage_risk"]
    print(
        "[f3_dedup] 读入 %d / 保留 %d / 淘汰 %d / 重复 %d / 过长 %d / 泄漏 %d / 坏行 %d"
        % (
            stats["read"],
            stats["kept"],
            dropped,
            stats["duplicate"],
            stats["too_long"],
            stats["leakage_risk"],
            stats["bad_lines"],
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="筛 3：去重 + 截长 + 泄漏过滤（§5.6）")
    ap.add_argument("--input", default=config.STEP2_KEEP, help="step2_kept.jsonl 路径")
    ap.add_argument("--output", default=config.STEP3_VERIFIED_POOL, help="最终池输出路径")
    ap.add_argument(
        "--rejects",
        default=config.STEP3_REJECTS,
        help="被淘汰题输出（复核用；传空字符串 '--rejects \"\"' 可关闭）",
    )
    args = ap.parse_args()
    # 允许用 '--rejects ""' 关闭淘汰题落盘；关闭时写入 os.devnull，分流与统计照常。
    rejects_path = args.rejects if args.rejects else os.devnull
    run(args.input, args.output, rejects_path)


if __name__ == "__main__":
    main()
