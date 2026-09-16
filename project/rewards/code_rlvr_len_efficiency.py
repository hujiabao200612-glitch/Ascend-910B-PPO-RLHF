# -*- coding: utf-8 -*-
"""code_rlvr_len_efficiency.py —— 奖励函数消融实验 4：正确性与代码精简度双目标驱动函数 (Ablation Exp 4)

学术动机：
    针对主训练中 PPO 生成代码体积偏长（平均 379.7 Tokens）的痛点，以及消融实验中 GRPO 凭借极简代码
    （192.4 Tokens）在 OpenAI HumanEval 上斩获 83.54% 历史纪录的发现。
    
    本实验设计双目标驱动阶梯奖励（Length/Efficiency-Aware）：
    1. 坚守正确性红线：未全通、报错、崩溃、超时一律处以 -1.0 硬惩罚；
    2. 卓越全通保底：100% 用例全通，但代码冗长 (> 250 Tokens)，给予 +1.0 标准分；
    3. 极简优雅爆发加成：100% 用例全通且代码精炼紧凑 (<= 250 Tokens)，给予 +1.5 巨额加成（+50% 超额 Advantage）！

科学假说：
    在保证绝对逻辑正确的前提下，+1.5 的强阶梯优势将驱使 PPO 策略主动消除过度工程（Over-engineering）、
    死代码与冗余类型声明，使 PPO 兼具 KodCode 的复杂防御性与 HumanEval 的极简高分能力，冲击 84%+。
"""

import os
import re
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


def _estimate_token_count(text: str) -> int:
    """轻量极速估算文本 Token 数量（与 BPE Tokenizer 保持 >95% 高度相关且零 IO 开销）。"""
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text))


def compute_score(*args, **kwargs) -> float:
    """verl 标准奖励入口函数（正确性 + 极简度双目标版本）。"""
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

    # 1. 严重故障或未全通红线档 (-1.0)：
    #    - 未抽取出有效代码、超时、语法错、崩溃、部分用例未过
    if (
        mode in ("no_code", "no_tests", "error")
        or total <= 0
        or timed_out
        or returncode not in (0, 1)
        or is_conservative
        or passed < total
    ):
        return -1.0

    # 2. 100% 用例全通分支：根据代码精炼度进行双目标分流
    token_len = _estimate_token_count(str(solution_str))

    # 紧凑精炼短代码：给予 +1.5 爆发加成；长代码保底 +1.0
    if token_len <= 250:
        return 1.5
    else:
        return 1.0


if __name__ == "__main__":
    print("=================================================================")
    print("[Self-Test] 正在核验 rewards/code_rlvr_len_efficiency.py ...")
    print("=================================================================")

    dummy_test = (
        "from solution import add\n"
        "def test_1(): assert add(1, 2) == 3\n"
        "def test_2(): assert add(0, 0) == 0\n"
    )

    short_perfect_resp = "```python\ndef add(a, b):\n    return a + b\n```"
    s1 = compute_score(solution_str=short_perfect_resp, ground_truth=dummy_test)
    print(f"  Case 1 (精炼满分解法 <=250 tok，预期 +1.5): score = {s1}")
    assert s1 == 1.5, f"Case 1 失败，实测 {s1}"

    # 构造超过 250 token 的长解法
    long_comment = "\n".join([f"    # comment line {i}: this is a verbose padding comment" for i in range(50)])
    long_perfect_resp = f"```python\ndef add(a, b):\n{long_comment}\n    return a + b\n```"
    s2 = compute_score(solution_str=long_perfect_resp, ground_truth=dummy_test)
    print(f"  Case 2 (冗长满分解法 >250 tok，预期 +1.0): score = {s2}")
    assert s2 == 1.0, f"Case 2 失败，实测 {s2}"

    wrong_resp = "```python\ndef add(a, b):\n    return a - b\n```"
    s3 = compute_score(solution_str=wrong_resp, ground_truth=dummy_test)
    print(f"  Case 3 (逻辑错误未全过，预期 -1.0): score = {s3}")
    assert s3 == -1.0, f"Case 3 失败，实测 {s3}"

    syntax_err_resp = "```python\ndef add(a, b\n    return a + b\n```"
    s4 = compute_score(solution_str=syntax_err_resp, ground_truth=dummy_test)
    print(f"  Case 4 (语法错误/崩溃，预期 -1.0): score = {s4}")
    assert s4 == -1.0, f"Case 4 失败，实测 {s4}"

    no_code_resp = "我不会写代码"
    s5 = compute_score(solution_str=no_code_resp, ground_truth=dummy_test)
    print(f"  Case 5 (无代码块，预期 -1.0): score = {s5}")
    assert s5 == -1.0, f"Case 5 失败，实测 {s5}"

    dict_gt = {"ground_truth": dummy_test}
    s6 = compute_score(data_source="kodcode", solution_str=short_perfect_resp, ground_truth=dict_gt)
    print(f"  Case 6 (字典传参，预期 +1.5): score = {s6}")
    assert s6 == 1.5, f"Case 6 失败，实测 {s6}"

    print("\n[PASS] code_rlvr_len_efficiency.py 双目标奖励核验全部通过！")
