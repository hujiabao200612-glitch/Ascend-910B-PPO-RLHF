# -*- coding: utf-8 -*-
"""code_rlvr_dhrcl.py —— 纯粹 DHRCL 三阶段分层课程学习退火消融实验 (Ablation: Pure DHRCL Curriculum)

学术出处：
    "DHRCL: Dynamic Hierarchical Reinforcement Learning with Curriculum Annealing for Code Generation"
    (arXiv:2607.26457)

核心机理：
    1. 动态三阶段时序解耦 (Dynamic 3-Phase Curriculum)：
       在强化学习前期与后期，模型面临的探索与对齐矛盾截然不同：
       - Phase 1 语法与防崩探索期 (0% ~ 35% 进度，约 0 ~ 25 步)：
         模型初探代码空间，给予线性递减的语法保底回报 w_syntax ∈ [0.15 -> 0.0]，保护早期探索，
         测试用例权重设为轻度校准 (gamma = 1.2)，快速逃离 0 分冷启动陷阱；
       - Phase 2 逻辑与边界攻坚期 (35% ~ 70% 进度，约 25 ~ 50 步)：
         语法分坚决归零 (w_syntax = 0.0)，彻底杜绝样板废话混分；测试用例指数提升至 gamma = 1.6，
         大幅拉开高阶边缘用例边际收益，攻克复杂用例；
       - Phase 3 严苛全通精炼期 (70% ~ 100% 进度，约 50 ~ 71 步)：
         语法分归零，测试用例指数提高至 gamma = 2.0（非全通得分严重衰减），
         倒逼策略全力收敛至 100% 正确的不动点。
    2. 无锁多进程自适应步数感知 (Lock-free Progress Tracker)：
       支持通过环境变量 `RLVR_CURRENT_STEP` 显式注入，或通过 `/tmp/rlvr_dhrcl_calls` 目录
       自动统计调用批次（每 64 次评测对应 1 个 PPO 训练步），零外部依赖，100% 全自动自适应。
"""

from __future__ import annotations

import math
import os
import sys
import uuid
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

_STEP_DIR = "/tmp/rlvr_dhrcl_calls"


def _get_curriculum_progress(total_steps: int = 71, batch_size: int = 64) -> Tuple[float, float, float, str]:
    """读取并计算当前课程学习进度。
    
    返回:
        (progress, syntax_w, gamma, phase_name)
    """
    # 1. 优先读取外部显式指定步数
    if "RLVR_CURRENT_STEP" in os.environ:
        try:
            step = float(os.environ["RLVR_CURRENT_STEP"])
            progress = min(max(step / float(total_steps), 0.0), 1.0)
        except Exception:
            progress = 0.0
    else:
        # 2. 自动通过 tmpfs 计数器统计（多进程无锁快速探测）
        try:
            os.makedirs(_STEP_DIR, exist_ok=True)
            # 记录一次调用标记
            token_path = os.path.join(_STEP_DIR, f"{os.getpid()}_{uuid.uuid4().hex[:6]}")
            with open(token_path, "w") as f:
                f.write("1")
            call_count = len(os.listdir(_STEP_DIR))
            est_step = float(call_count) / float(batch_size)
            progress = min(max(est_step / float(total_steps), 0.0), 1.0)
        except Exception:
            progress = 0.0

    if progress < 0.35:
        # Phase 1: [0% ~ 35%) 语法保底线性退火 [0.15 -> 0.0], gamma = 1.2
        syntax_w = 0.15 * (1.0 - progress / 0.35)
        gamma = 1.2
        phase_name = "Phase 1: Syntax & Crash Exploration"
    elif progress < 0.70:
        # Phase 2: [35% ~ 70%) 语法归零，边界攻坚 gamma = 1.6
        syntax_w = 0.0
        gamma = 1.6
        phase_name = "Phase 2: Functional Hardening"
    else:
        # Phase 3: [70% ~ 100%] 严苛全通 gamma = 2.0
        syntax_w = 0.0
        gamma = 2.0
        phase_name = "Phase 3: Strict Convergence"

    return progress, syntax_w, gamma, phase_name


def compute_score(*args, **kwargs) -> float:
    """纯 DHRCL 三阶段课程退火奖励计算入口。"""
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

    # 2. 获取当前时序课程超参
    progress, syntax_w, gamma, phase_name = _get_curriculum_progress(total_steps=71, batch_size=64)

    # 3. 临界与部分通过档 (依据当前课程阶梯超线性退火)
    if passed < total:
        raw_ratio = float(passed) / float(total)
        calibrated_ratio = math.pow(raw_ratio, gamma)
        # 基础得分 [0.10, 0.70] + 早期语法保底分 w_syntax (最高 0.15)
        step_score = 0.10 + 0.60 * calibrated_ratio + syntax_w
        return float(round(step_score, 4))

    # 4. 100% 全通达标档 (+1.00 满分)
    return 1.00


if __name__ == "__main__":
    print("=================================================================")
    print("[Self-Test] 正在对纯 DHRCL 课程奖励函数进行全阶段核验 ...")
    print("=================================================================")

    dummy_test = (
        "from solution import solve\n"
        "def test_1(): assert solve(2) == 4\n"
        "def test_2(): assert solve(3) == 6\n"
        "def test_3(): assert solve(0) == 0\n"
        "def test_4(): assert solve(-1) == -2\n"
        "def test_5(): assert solve(5) == 10\n"
    )

    resp_partial = "```python\ndef solve(x):\n    return 0 if x == 0 else (4 if x == 2 else 6)\n```"

    # 模拟 Phase 1 (早期步数 = 5)
    os.environ["RLVR_CURRENT_STEP"] = "5"
    p1, sw1, g1, ph1 = _get_curriculum_progress(71)
    s1 = compute_score(solution_str=resp_partial, ground_truth=dummy_test)
    print(f"  [PASS] Phase 1 (Step 5, {ph1}): syntax_w={sw1:.3f}, gamma={g1:.1f}, partial_score={s1:.4f}")
    assert sw1 > 0.05

    # 模拟 Phase 2 (中期步数 = 35)
    os.environ["RLVR_CURRENT_STEP"] = "35"
    p2, sw2, g2, ph2 = _get_curriculum_progress(71)
    s2 = compute_score(solution_str=resp_partial, ground_truth=dummy_test)
    print(f"  [PASS] Phase 2 (Step 35, {ph2}): syntax_w={sw2:.3f}, gamma={g2:.1f}, partial_score={s2:.4f}")
    assert sw2 == 0.0 and g2 == 1.6

    # 模拟 Phase 3 (后期步数 = 65)
    os.environ["RLVR_CURRENT_STEP"] = "65"
    p3, sw3, g3, ph3 = _get_curriculum_progress(71)
    s3 = compute_score(solution_str=resp_partial, ground_truth=dummy_test)
    print(f"  [PASS] Phase 3 (Step 65, {ph3}): syntax_w={sw3:.3f}, gamma={g3:.1f}, partial_score={s3:.4f}")
    assert sw3 == 0.0 and g3 == 2.0
    assert s3 < s2, "Phase 3 对未全通代码应比 Phase 2 施加更严厉折减"

    print("\n[OK] code_rlvr_dhrcl.py 自验通过！")
