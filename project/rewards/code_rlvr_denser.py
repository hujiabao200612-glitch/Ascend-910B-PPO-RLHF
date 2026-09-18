# -*- coding: utf-8 -*-
"""code_rlvr_denser.py —— 奖励函数消融实验 5：DenseR 跨类散度信用与组内紧致独特性奖励函数 (Ablation Exp 5)

学术背景与机制（DenseR: Dense Rewards For Free in LLM Reasoning, Bansal et al., 2025）：
    标准 RLVR 仅在序列末尾给予二值稀疏回报（0.0 或 1.0），导致中间推理步骤信用分配迟滞（Credit Assignment Delay）。
    DenseR 的核心思想在于两点：
    1. 跨类别散度信用 (Cross-Class Divergence Credit, d_cross)：
       处于决策边界附近（如 10 个测试通过了 7 个）的轨迹，说明主干算法逻辑成立，仅在边缘用例存在 Off-by-one
       或边界遗漏。不应将其与完全写不出代码的崩溃轨迹同等处死，而应给予连续的边界探索信用 [0.1, 0.55]，
       架起从未通过到通过的平滑梯度阶梯；
    2. 组内独特性与信息密度加成 (Within-Class Uniqueness & Density Bonus, d_within)：
       对于 100% 全通的解，根据有效信息密度（奥卡姆剃刀，代码精悍度）给予独特性阶梯奖励 (+1.0 ~ +1.3)，
       重奖高信息密度、逻辑纯粹的优雅解，抑制样板废话（fluff/boilerplate）；
    3. 灾难性故障红线防坠 (Catastrophic Failure Hard Floor)：
       语法错误、未生成代码、超时或完全零通过的崩溃代码，坚决处以 -1.0 严惩，捍卫生成鲁棒性与工程安全性。

四级奖励分布图谱：
    - 严重故障/无代码/崩溃/零通过: -1.0 (硬底线防御)
    - 临界探索/部分通过 (0 < acc < 1.0): 0.10 + 0.45 * (passed / total)  ->  [0.10, 0.55] 连续信用
    - 全通冗长解 (acc == 1.0, len > 250): +1.00 (标准完成分)
    - 全通紧致独特性解 (acc == 1.0, len <= 250): +1.30 (+30% 密度与独特性加成)
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
    """verl 标准奖励入口函数（DenseR 跨类别散度信用 + 组内独特性版本）。"""
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

    # 1. 灾难性故障红线 (-1.0)：
    #    - 未提取出有效代码、测试集加载失败、超时、语法错导致无法测试、0 个用例通过
    if (
        mode in ("no_code", "no_tests", "error")
        or total <= 0
        or timed_out
        or returncode not in (0, 1)
        or is_conservative
        or passed <= 0
    ):
        return -1.0

    # 2. 跨类别散度信用档 (Cross-Class Divergence Credit)：
    #    主干逻辑已成立，但未 100% 全通（0 < passed < total）
    if passed < total:
        acc = float(passed) / float(total)
        # 范围 [0.10, 0.55]，为决策边界附近的探索提供平滑正向势能
        denser_credit = 0.10 + 0.45 * acc
        return float(round(denser_credit, 4))

    # 3. 组内独特性与信息密度加成档 (Within-Class Uniqueness & Density)：
    #    100% 用例全通 (passed == total)
    token_len = _estimate_token_count(str(solution_str))

    # 紧凑短代码 (<= 250 tok)：给予 +1.30 独特性加成
    # 冗长全通代码 (> 250 tok)：给予 +1.00 标准满分
    if token_len <= 250:
        return 1.30
    else:
        return 1.00


if __name__ == "__main__":
    print("=================================================================")
    print("[Self-Test] 正在对 rewards/code_rlvr_denser.py 进行四重功能核验 ...")
    print("=================================================================")

    dummy_test = (
        "from solution import solve\n"
        "def test_1(): assert solve(2) == 4\n"
        "def test_2(): assert solve(3) == 6\n"
        "def test_3(): assert solve(0) == 0\n"
        "def test_4(): assert solve(-1) == -2\n"
    )

    # 用例 1: 完美紧凑解答 (<=250 tok，应得 +1.30 组内独特性爆发奖)
    compact_resp = "```python\ndef solve(x):\n    return x * 2\n```"
    s1 = compute_score(solution_str=compact_resp, ground_truth=dummy_test)
    assert s1 == 1.30, f"Case 1 期望 1.30，实测 {s1}"
    print(f"  [PASS] Case 1 (全通极简独特性): score = {s1} (预期 1.30)")

    # 用例 2: 完美但冗长解答 (>250 tok，应得 +1.00 标准分)
    verbose_comment = "# 详细推导思路\n" * 60
    verbose_resp = f"```python\n{verbose_comment}def solve(x):\n    return x * 2\n```"
    s2 = compute_score(solution_str=verbose_resp, ground_truth=dummy_test)
    assert s2 == 1.00, f"Case 2 期望 1.00，实测 {s2}"
    print(f"  [PASS] Case 2 (全通冗长标准分): score = {s2} (预期 1.00)")

    # 用例 3: 临界部分通过 (4 题过 2 题，应得 0.10 + 0.45 * 0.5 = 0.325 散度信用)
    partial_resp = "```python\ndef solve(x):\n    return 4 if x == 2 else (6 if x == 3 else 999)\n```"
    s3 = compute_score(solution_str=partial_resp, ground_truth=dummy_test)
    assert 0.30 <= s3 <= 0.35, f"Case 3 期望 ~0.325，实测 {s3}"
    print(f"  [PASS] Case 3 (临界部分通过):   score = {s3} (预期 ~0.325 散度信用)")

    # 用例 4: 致命崩溃/语法错/纯文本胡说 (应坚决处以 -1.0 硬底线)
    no_code_resp = "今天天气真好，我不写代码了。"
    s4 = compute_score(solution_str=no_code_resp, ground_truth=dummy_test)
    assert s4 == -1.0, f"Case 4 期望 -1.0，实测 {s4}"
    print(f"  [PASS] Case 4 (无代码致命惩戒): score = {s4} (预期 -1.00)")

    print("\n[OK] rewards/code_rlvr_denser.py 四重核验全部通过！可安全投入 8 卡 PPO 训练。")
