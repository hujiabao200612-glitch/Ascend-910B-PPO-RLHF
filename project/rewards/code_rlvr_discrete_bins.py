# -*- coding: utf-8 -*-
"""code_rlvr_discrete_bins.py —— 奖励函数消融实验 3：全离散三档硬阶梯奖励驱动函数 (Ablation Exp 3)

学术动机：
    借鉴顶刊前沿成果（CUDA 算子生成之 Robust Reward Scheduling 理论），连续浮点奖励（如 0.325, 0.781）
    会给 Critic 价值网络引入高频噪声，导致价值拟合方差过大；且连续分数（如通过 8/10 题给 0.8 分）
    会向策略网络发送“差不多得了”的局部最优麻痹信号，削弱模型攻克最后 1~2 个边缘用例的决心。
    
    本实验设计全离散三档阶梯（Discrete Bins）：
    1. 严重故障档 (-1.0)：未提取出代码、语法错误、运行时崩溃、超时；
    2. 中间探索档 (0.0)：代码可正常编译执行无崩溃，但未 100% 通过全部测试断言（包括部分通过）；
    3. 卓越解决档 (+1.0)：100% 全部测试用例通过。

两级梯度势能：
    第一级跳跃（-1.0 -> 0.0）：强力迫使模型消除致命报错，学会编写安全规范的语法；
    第二级跳跃（0.0 -> +1.0）：以 1.0 的巨大 Advantage 驱动模型追求全部用例通过，消除浮点方差。
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
    """verl 标准奖励入口函数（全离散三档阶梯消融版本）。"""
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

    # 1. 严重故障档 (-1.0)：
    #    - 未抽取出代码块 (mode == "no_code")
    #    - 超时 (timed_out)
    #    - 语法错/导入异常导致 pytest 收集崩溃 (returncode not in (0, 1) 或 conservative 为 True)
    #    - 测试用例无效或沙箱异常 (mode in ("no_tests", "error") 或 total <= 0)
    if (
        mode in ("no_code", "no_tests", "error")
        or total <= 0
        or timed_out
        or returncode not in (0, 1)
        or is_conservative
    ):
        return -1.0

    # 2. 卓越解决档 (+1.0)：100% 全部断言通过
    if total > 0 and passed >= total:
        return 1.0

    # 3. 中间探索档 (0.0)：代码语法合法且正常执行完毕，但部分断言未通过
    return 0.0


if __name__ == "__main__":
    print("=================================================================")
    print("[Self-Test] 正在核验 rewards/code_rlvr_discrete_bins.py ...")
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
    print(f"  Case 2 (逻辑错误但无崩溃，预期 0.0): score = {s2}")
    assert s2 == 0.0, f"Case 2 失败，实测 {s2}"

    syntax_err_resp = "```python\ndef add(a, b\n    return a + b\n```"
    s3 = compute_score(solution_str=syntax_err_resp, ground_truth=dummy_test)
    print(f"  Case 3 (语法错误/崩溃，预期 -1.0): score = {s3}")
    assert s3 == -1.0, f"Case 3 失败，实测 {s3}"

    no_code_resp = "我不会写代码"
    s4 = compute_score(solution_str=no_code_resp, ground_truth=dummy_test)
    print(f"  Case 4 (无代码块，预期 -1.0): score = {s4}")
    assert s4 == -1.0, f"Case 4 失败，实测 {s4}"

    dict_gt = {"ground_truth": dummy_test}
    s5 = compute_score(data_source="kodcode", solution_str=perfect_resp, ground_truth=dict_gt)
    print(f"  Case 5 (字典传参，预期 +1.0): score = {s5}")
    assert s5 == 1.0, f"Case 5 失败，实测 {s5}"

    print("\n[PASS] code_rlvr_discrete_bins.py 全离散三档核验全部通过！")
