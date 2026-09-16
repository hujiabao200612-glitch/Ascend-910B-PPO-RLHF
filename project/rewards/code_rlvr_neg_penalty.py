# -*- coding: utf-8 -*-
"""code_rlvr_neg_penalty.py —— 奖励函数消融实验 1：强负惩罚驱动函数 (Ablation Exp 1)

学术动机：
    借鉴 CUDA Kernel 生成顶刊论文的 Robust Reward Scheduling 设计原则：
    对任何未全通、逻辑错误、语法错误或超时的候选输出施加硬性红线扣分 (-1.0)，
    将策略模型做对 (+1.0) 与做错 (-1.0) 的 Advantage 差异由原来的 1.0 翻倍扩大到 2.0，
    测试是否能够强力驱动 Critic 建立对缺陷代码的敏感惩罚，彻底抑制运行时的致命崩溃 (Fatal Errors)。

奖励分档：
    - 100% 用例全通 (PassAll)：+1.0
    - 任何用例失败、语法错误、超时或未生成代码块：-1.0
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_CUR_DIR)
_PIPE_DIR = os.path.join(_PROJ_DIR, "pipeline")

for _p in [_PROJ_DIR, _PIPE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from score import score_kernel
except ImportError:
    try:
        import pipeline.score as score_module
        score_kernel = score_module.score_kernel
    except Exception:
        score_kernel = None


def compute_score(*args, **kwargs) -> float:
    """verl 标准奖励入口函数（强负惩罚消融版本）。"""
    solution_str = kwargs.get("solution_str", None)
    ground_truth = kwargs.get("ground_truth", None)
    extra_info = kwargs.get("extra_info", None)
    data_source = kwargs.get("data_source", None)

    if solution_str is None and len(args) > 0:
        if len(args) == 1:
            solution_str = args[0]
        elif len(args) == 2:
            solution_str, ground_truth = args[0], args[1]
        elif len(args) >= 3:
            data_source, solution_str, ground_truth = args[0], args[1], args[2]
            if len(args) > 3:
                extra_info = args[3]

    if solution_str is None:
        solution_str = ""

    test_code = ""
    if isinstance(ground_truth, dict):
        test_code = ground_truth.get("ground_truth", ground_truth.get("test", ""))
    elif isinstance(ground_truth, str):
        test_code = ground_truth

    if not test_code and isinstance(extra_info, dict):
        test_code = extra_info.get("test", "")

    if not isinstance(test_code, str) or not test_code.strip():
        return -1.0

    if score_kernel is None:
        return -1.0

    try:
        passed, total, detail = score_kernel(
            response=str(solution_str),
            test=test_code,
            timeout_s=5,
        )
    except Exception:
        return -1.0

    mode = detail.get("mode", "")
    timed_out = detail.get("timed_out", False)

    # 1. 任何无代码块、测试解析失败或超时直接判 -1.0
    if mode in ("no_code", "no_tests", "error") or total <= 0 or timed_out:
        return -1.0

    # 2. 正确性二值判定：只有 100% 全过才给 +1.0，只要有一条没过坚决扣 -1.0
    acc = (passed / total) if total > 0 else 0.0
    if acc >= 1.0:
        return 1.0
    else:
        return -1.0


if __name__ == "__main__":
    print("=================================================================")
    print("[Self-Test] 正在核验 rewards/code_rlvr_neg_penalty.py ...")
    print("=================================================================")

    dummy_test = (
        "from solution import add\n"
        "def test_1(): assert add(1, 2) == 3\n"
        "def test_2(): assert add(0, 0) == 0\n"
    )

    perfect_resp = "```python\ndef add(a, b):\n    return a + b\n```"
    s1 = compute_score(solution_str=perfect_resp, ground_truth=dummy_test)
    print(f"  Case 1 (完美解法，预期 +1.0): score = {s1}")
    assert s1 == 1.0, f"Case 1 失败，实测 {s1}"

    wrong_resp = "```python\ndef add(a, b):\n    return a - b\n```"
    s2 = compute_score(solution_str=wrong_resp, ground_truth=dummy_test)
    print(f"  Case 2 (逻辑错误，预期 -1.0): score = {s2}")
    assert s2 == -1.0, f"Case 2 失败，实测 {s2}"

    no_code_resp = "我不会写代码"
    s3 = compute_score(solution_str=no_code_resp, ground_truth=dummy_test)
    print(f"  Case 3 (无代码块，预期 -1.0): score = {s3}")
    assert s3 == -1.0, f"Case 3 失败，实测 {s3}"

    dict_gt = {"ground_truth": dummy_test}
    s4 = compute_score(data_source="kodcode", solution_str=perfect_resp, ground_truth=dict_gt)
    print(f"  Case 4 (字典传参，预期 +1.0): score = {s4}")
    assert s4 == 1.0, f"Case 4 失败，实测 {s4}"

    print("\n[PASS] code_rlvr_neg_penalty.py 验证完全通过！")
