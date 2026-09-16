# -*- coding: utf-8 -*-
"""code_rlvr_sparse.py —— 奖励函数消融实验 2：纯稀疏二进制奖励驱动函数 (Ablation Exp 2)

学术动机：
    检验阿里通义千问 Qwen 团队在《The Verification Horizon: No Silver Bullet for Coding Agent Rewards》
    中提出的“过程分诱导代理欺骗 (Proxy Hacking)”假说。
    彻底剥离格式分 (0.1) 与语法执行分 (0.2)，去除部分用例的连续插值分：
    只有 100% 全部断言通过才给 +1.0，其余任何情况（包括格式工整、空函数、部分通过、超时）一律给 0.0。
    测试在无任何过程平滑分引导下，纯稀疏 RLVR 对策略探索与代码形态的纯粹影响。

奖励分档：
    - 100% 用例全通 (PassAll)：+1.0
    - 任何未全通、逻辑错、语法错、超时或无代码块：0.0
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
    """verl 标准奖励入口函数（纯稀疏二进制消融版本）。"""
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

    if not isinstance(test_code, str) or not test_code.strip() or score_kernel is None:
        return 0.0

    try:
        passed, total, detail = score_kernel(
            response=str(solution_str),
            test=test_code,
            timeout_s=5,
        )
    except Exception:
        return 0.0

    mode = detail.get("mode", "")
    timed_out = detail.get("timed_out", False)

    # 1. 任何未提取出代码、错误、超时或无断言，一律给 0.0
    if mode in ("no_code", "no_tests", "error") or total <= 0 or timed_out:
        return 0.0

    # 2. 纯稀疏二进制：100% 全过给 1.0，否则一律 0.0（不给任何过程修饰分）
    acc = (passed / total) if total > 0 else 0.0
    return 1.0 if acc >= 1.0 else 0.0


if __name__ == "__main__":
    print("=================================================================")
    print("[Self-Test] 正在核验 rewards/code_rlvr_sparse.py ...")
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
    print(f"  Case 2 (逻辑错误，预期 0.0): score = {s2}")
    assert s2 == 0.0, f"Case 2 失败，实测 {s2}"

    no_code_resp = "我不会写代码"
    s3 = compute_score(solution_str=no_code_resp, ground_truth=dummy_test)
    print(f"  Case 3 (无代码块，预期 0.0): score = {s3}")
    assert s3 == 0.0, f"Case 3 失败，实测 {s3}"

    dict_gt = {"ground_truth": dummy_test}
    s4 = compute_score(data_source="kodcode", solution_str=perfect_resp, ground_truth=dict_gt)
    print(f"  Case 4 (字典传参，预期 +1.0): score = {s4}")
    assert s4 == 1.0, f"Case 4 失败，实测 {s4}"

    print("\n[PASS] code_rlvr_sparse.py 纯稀疏二值核验全部通过！")
