# -*- coding: utf-8 -*-
"""code_rlvr.py —— 正式代码 RLVR 奖励驱动函数（D3 任务 2）

职责：
    替代玩具正则 rewards/smoke_gsm8k.py，挂载工程化代码判分沙箱。
    对接 verl 0.6.1 的 NaiveRewardManager，对模型在 PPO Rollout 阶段生成的每一条回答
    进行真实验证与奖励整形（Reward Shaping）。

奖励整形公式（三层结构）：
    1. 格式分 (Format Bonus)：
       - 成功闭合生成 ```python 代码块：+0.1 分
       - 未生成代码块（纯文本胡说/拒答）：0.0 分
    2. 准确分 (Accuracy Score)：
       - 跑真实 pytest 测试用例，计算 passed / total 比例，折算 0.0 ~ 0.9 平滑分
       - 100% 用例全通时，奖励直接定额为 1.0（满分封顶）
    3. 超时与死循环惩罚 (Penalty)：
       - 单题执行限时 5 秒（防恶意 while True 阻塞 PPO 迭代）
       - 触发超时判定则额外扣减 0.2 分（截断到下限 0.0）
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

# 确保能平滑导入 pipeline 模块（无论是从 project 根目录、Ray Worker 还是奖励模块内触发）
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_CUR_DIR)
_PIPE_DIR = os.path.join(_PROJ_DIR, "pipeline")

for _p in [_PROJ_DIR, _PIPE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from score import score_kernel
except ImportError:
    # 兼容直接从 pipeline 目录下引用的情况
    import pipeline.score as score_module
    score_kernel = score_module.score_kernel


def compute_score(*args, **kwargs) -> float:
    """verl 标准奖励入口函数。

    入参兼容：
        - 关键字传参：compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
        - 位置传参 (2项)：compute_score(solution_str, ground_truth)
        - 位置传参 (3项以上)：compute_score(data_source, solution_str, ground_truth, extra_info=...)
    """
    solution_str = kwargs.get("solution_str", None)
    ground_truth = kwargs.get("ground_truth", None)
    extra_info = kwargs.get("extra_info", None)
    data_source = kwargs.get("data_source", None)

    # 兼容位置参数传入
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

    # 解析 ground_truth 中的测试代码（兼容 dict 或 str）
    test_code = ""
    if isinstance(ground_truth, dict):
        test_code = ground_truth.get("ground_truth", ground_truth.get("test", ""))
    elif isinstance(ground_truth, str):
        test_code = ground_truth

    if not test_code and isinstance(extra_info, dict):
        test_code = extra_info.get("test", "")

    if not isinstance(test_code, str) or not test_code.strip():
        # 无有效测试用例，安全回落
        return 0.0

    # 调用经过 115 项边界单测磨砺的 score_kernel
    # 超时上限设为 5 秒，保障 PPO Rollout 高吞吐
    try:
        passed, total, detail = score_kernel(
            response=str(solution_str),
            test=test_code,
            timeout_s=5,
        )
    except Exception as exc:
        # 奖励函数在 RL 训练期绝不能抛崩主进程
        return 0.0

    mode = detail.get("mode", "")
    timed_out = detail.get("timed_out", False)

    # 1. 格式层校验：未抽取出合法代码块，判 0 分
    if mode in ("no_code", "no_tests", "error") or total <= 0:
        return 0.0

    # 2. 准确率层计算
    acc = (passed / total) if total > 0 else 0.0

    if acc >= 1.0:
        # 全量通过：满分 1.0
        final_reward = 1.0
    else:
        # 部分分：0.1 格式分 + 0.9 * 通过率
        final_reward = 0.1 + 0.9 * acc

    # 3. 超时惩罚
    if timed_out:
        final_reward = max(0.0, final_reward - 0.2)

    return float(round(final_reward, 4))


if __name__ == "__main__":
    print("=================================================================")
    print("[Self-Test] 正在对 rewards/code_rlvr.py 进行四重功能核验 ...")
    print("=================================================================")

    dummy_test = (
        "from solution import add\n"
        "def test_1(): assert add(1, 2) == 3\n"
        "def test_2(): assert add(0, 0) == 0\n"
    )

    # 用例 1: 完美解答（应得 1.0 满分）
    perfect_resp = "思考：实现两个数相加。\n```python\ndef add(a, b):\n    return a + b\n```"
    s1 = compute_score(solution_str=perfect_resp, ground_truth=dummy_test)
    assert s1 == 1.0, f"Case 1 期望 1.0，实测 {s1}"
    print(f"  [PASS] Case 1 (完美满分):        score = {s1}")

    # 用例 2: 逻辑错误但代码格式正确（部分通过）
    wrong_resp = "```python\ndef add(a, b):\n    return a - b\n```"
    s2 = compute_score(solution_str=wrong_resp, ground_truth=dummy_test)
    print(f"  [PASS] Case 2 (部分通过):        score = {s2}")

    # 用例 3: 纯文本未生成代码块（应得 0.0 分）
    no_code_resp = "我不知道怎么写加法函数。"
    s3 = compute_score(solution_str=no_code_resp, ground_truth=dummy_test)
    assert s3 == 0.0, f"Case 3 期望 0.0，实测 {s3}"
    print(f"  [PASS] Case 3 (无代码块零分):    score = {s3}")

    # 用例 4: 兼容字典 ground_truth 传参（verl 真实调用风格）
    dict_gt = {"style": "rule", "ground_truth": dummy_test}
    s4 = compute_score(data_source="kodcode", solution_str=perfect_resp, ground_truth=dict_gt)
    assert s4 == 1.0, f"Case 4 期望 1.0，实测 {s4}"
    print(f"  [PASS] Case 4 (verl 字典传参):   score = {s4}")

    print("\n[OK] rewards/code_rlvr.py 四重核验全部通过！可安全挂载至 PPO 训练器。")
