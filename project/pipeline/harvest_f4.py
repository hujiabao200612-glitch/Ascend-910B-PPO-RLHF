# -*- coding: utf-8 -*-
"""harvest_f4.py —— 筛 4 提前收割与结算工具。

当 8 卡已处理题数达到 5,000+ 题、黄金池已达 2,000~2,500 条时，
可无需再等待后续几千道题，直接将当前各 worker 临时文件结算为最终的
data/rl_pool.jsonl (2k+) 与 data/heldout.jsonl (200题)。
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys


def harvest(
    tmp_pattern: str = "data/_tmp_f4_worker_*.jsonl",
    rl_pool_path: str = "data/rl_pool.jsonl",
    heldout_path: str = "data/heldout.jsonl",
    rejects_path: str = "data/step4_rejects.jsonl",
    pass_rate_min: float = 0.1,
    pass_rate_max: float = 0.9,
    n_heldout: int = 200,
    seed: int = 42,
    clean_tmp: bool = False,
):
    random.seed(seed)
    files = sorted(glob.glob(tmp_pattern))
    if not files:
        print(f"[ERROR] 未找到匹配的临时分片文件: {tmp_pattern}", file=sys.stderr)
        return

    print(f"[Harvest] 找到 {len(files)} 个 Worker 临时文件，正在读取...")
    all_results = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))

    total = len(all_results)
    print(f"[Harvest] 成功载入 {total} 条已评测题目数据，正在进行黄金难度分级...")

    golden_candidates = []
    too_easy_items = []
    too_hard_items = []

    for item in all_results:
        pr = item.get("sample_pass_rate", 0.0)
        if pr > pass_rate_max:
            item["filter_reason"] = "too_easy"
            too_easy_items.append(item)
        elif pr < pass_rate_min:
            item["filter_reason"] = "too_hard"
            too_hard_items.append(item)
        else:
            golden_candidates.append(item)

    print(f"[Harvest] 分级结果: 黄金题 {len(golden_candidates)} | 极易题 {len(too_easy_items)} | 极难题 {len(too_hard_items)}")

    if len(golden_candidates) < n_heldout:
        print(f"[ERROR] 黄金题数量 ({len(golden_candidates)}) 不足 {n_heldout} 条，无法满足 heldout 隔离要求！", file=sys.stderr)
        return

    # 随机隔离 heldout 200 题
    random.shuffle(golden_candidates)
    heldout_items = golden_candidates[:n_heldout]
    rl_pool_items = golden_candidates[n_heldout:]

    os.makedirs(os.path.dirname(os.path.abspath(rl_pool_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(heldout_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(rejects_path)), exist_ok=True)

    with open(heldout_path, "w", encoding="utf-8") as fheld:
        for item in heldout_items:
            fheld.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(rl_pool_path, "w", encoding="utf-8") as frl:
        for item in rl_pool_items:
            frl.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(rejects_path, "w", encoding="utf-8") as frej:
        for item in too_easy_items + too_hard_items:
            frej.write(json.dumps(item, ensure_ascii=False) + "\n")

    if clean_tmp:
        for fpath in files:
            try:
                os.remove(fpath)
            except OSError:
                pass
        print("[Harvest] 已清理临时分片文件。")

    print("\n" + "=" * 55)
    print("🎉 D2 数据集正式结算成功（提前收割完成）！")
    print(f"  - 总累计处理题目: {total} 道")
    print(f"  - 淘汰极易题目: {len(too_easy_items)} 道")
    print(f"  - 淘汰极难题目: {len(too_hard_items)} 道")
    print(f"  - 🌟 黄金题总数: {len(golden_candidates)} 道 ({len(golden_candidates)/total*100:.1f}%)")
    print(f"  =================================================")
    print(f"  ✅ 独立评测集 (heldout.jsonl): {len(heldout_items)} 道 -> {heldout_path}")
    print(f"  ✅ 强化学习训练池 (rl_pool.jsonl): {len(rl_pool_items)} 道 -> {rl_pool_path}")
    print(f"  ✅ 淘汰集 (step4_rejects.jsonl): {len(too_easy_items) + len(too_hard_items)} 道 -> {rejects_path}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    clean = "--clean" in sys.argv
    harvest(clean_tmp=clean)
