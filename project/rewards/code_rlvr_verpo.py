# -*- coding: utf-8 -*-
"""code_rlvr_verpo.py —— 纯粹 VeRPO 奖励函数消融实验 (Ablation: Pure VeRPO)

学术出处：
    "VeRPO: Verifiable Dense Reward Policy Optimization for Code Generation" (Wang et al., 2026, arXiv:2601.03525)

核心机理：
    1. 消除基数偏差 (Anti-Cardinality Bias)：
       在标准线性通过率下，题目中的简单测试用例易被攻克，导致模型做对基础样例后在 80% 收益上“躺平”，
       缺乏攻克复杂边缘用例的边际动力（KodCode 出现 16% 部分分平台期）。
    2. 超线性密度校准 (Super-linear Density Calibration)：
       引入指数因子 gamma = 1.6。通过超线性变换 (passed / total) ** 1.6：
       - 低阶用例 (2/10 = 0.20) -> 校准后为 0.076，抑制靠猜测常数赚取低级分；
       - 高阶边缘用例 (9/10 = 0.90) -> 校准后为 0.845，最后 1 个难用例边际回报大幅攀升！
    3. 纯粹性设计：
       不添加任何长度惩罚、不添加时间步退火，100% 独立验证 VeRPO 难度校准机制的真实增益。
"""

from __future__ import annotations

import math
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
    """纯 VeRPO 奖励函数计算入口。"""
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
    returncode = detail.get("returncode", 0)
    is_conservative = detail.get("conservative", False)

    # 1. 灾难性硬底线防坠 (-1.0)
    if (
        mode in ("no_code", "no_tests", "error")
        or total <= 0
        or timed_out
        or returncode not in (0, 1)
        or is_conservative
        or passed <= 0
    ):
        return -1.0

    # 2. 纯粹 VeRPO 超线性密度校准档 (0 < passed < total)
    if passed < total:
        raw_ratio = float(passed) / float(total)
        # 严格采用 VeRPO 建议的 gamma = 1.6 密度幂律
        calibrated_ratio = math.pow(raw_ratio, 1.6)
        # 奖励区间映射至 [0.10, 0.70]
        score_val = 0.10 + 0.60 * calibrated_ratio
        return float(round(score_val, 4))

    # 3. 100% 全通达标档 (+1.0 标准满分，不引入长度偏置，纯净检验正确性)
    return 1.00


if __name__ == "__main__":
    print("=================================================================")
    print("[Self-Test] 正在对纯 VeRPO 奖励函数进行三层核心核验 ...")
    print("=================================================================")

    dummy_test = (
        "from solution import solve\n"
        "def test_1(): assert solve(2) == 4\n"
        "def test_2(): assert solve(3) == 6\n"
        "def test_3(): assert solve(0) == 0\n"
        "def test_4(): assert solve(-1) == -2\n"
        "def test_5(): assert solve(5) == 10\n"
    )

    # 用例 1: 完美全通 (应得纯净 +1.00)
    compact_resp = "```python\ndef solve(x):\n    return x * 2\n```"
    s1 = compute_score(solution_str=compact_resp, ground_truth=dummy_test)
    assert s1 == 1.00, f"Case 1 期望 1.00，实测 {s1}"
    print(f"  [PASS] Case 1 (全通纯净满分): score = {s1:.4f}")

    # 用例 2: 高比例部分通过 (5 过 4 = 0.80，VeRPO 赋权后应有坚实攻坚回报)
    high_resp = "```python\ndef solve(x):\n    return 0 if x == 0 else (4 if x == 2 else (6 if x == 3 else 10))\n```"
    s2 = compute_score(solution_str=high_resp, ground_truth=dummy_test)
    assert 0.45 <= s2 <= 0.60, f"Case 2 期望 [0.45, 0.60]，实测 {s2}"
    print(f"  [PASS] Case 2 (高阶边缘用例攻坚): score = {s2:.4f}")

    # 用例 3: 致命崩溃/无代码 (死守 -1.0)
    crash_resp = "无有效 Python 代码"
    s3 = compute_score(solution_str=crash_resp, ground_truth=dummy_test)
    assert s3 == -1.0, f"Case 3 期望 -1.0，实测 {s3}"
    print(f"  [PASS] Case 3 (崩溃严苛底线): score = {s3:.4f}")

    print("\n[OK] code_rlvr_verpo.py 自验通过！")
