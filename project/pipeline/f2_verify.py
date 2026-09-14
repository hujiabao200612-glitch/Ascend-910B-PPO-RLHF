# -*- coding: utf-8 -*-
"""f2_verify.py —— 筛 2：用官方解答过沙箱，淘汰跑不通的（任务书 §5.2）。

职责
    把筛 1 的产出 ``step1_templated.jsonl`` 每条拿官方 ``solution`` + ``test`` 过一遍
    沙箱，确认「这道题 + 它的标准答案 + 它的测试」三者是自洽、可跑通的：
      - 跑通（pytest 全过、未超时）→ 保留，写 ``step2_kept.jsonl``；
      - 跑不通（断言失败 / 超时 / 沙箱环境错 / 缺字段）→ 淘汰，写 ``step2_rejects.jsonl``。

为什么这一步必须存在
    原料是「GPT 生成 + 自洽过滤」过的候选，但 15,000 条里仍混着测试写错、答案写错、
    或答案与测试对不上的脏样本。训练池里塞进一道「标准答案本身就跑不通」的题，等于
    给模型喂了错误正例，RLVR 的奖励信号会直接被污染。筛 2 就是这道「题本身得是自洽
    的」硬闸，过不了的连训练池都进不去。

为什么复用 sandbox.py 而不是自己起子进程
    sandbox 是任务书 §5.2 指定的唯一执行件（超时 Kill 进程树、UTF-8、禁 shell=True、
    增强版断网都在里面），判分器 score.py 也用它。筛 2 与判分器是「同一把尺子量两次」：
    今天用官方答案量一遍确认题没问题，将来用模型答案量一遍打奖励。复用保证两处执行
    语义一致，不会出现「判分器能跑、筛 2 却跑挂」的分裂。

任务书未明确、按合理工程实践处理的假设（便于同事评审时逐条核对）
    A. 运行方式（关键）：把 ``solution`` 与 ``test`` **合并**进同一个 ``test_case.py``，
       同时仍保留独立的 ``solution.py``。原因——数据集里两种 test 习惯并存：
         * import 风格（如 Taco）：``test`` 里写 ``from solution import foo``，靠
           ``solution.py`` 被 import；
         * concat 风格（如 Codeforces / Docs）：``test`` 直接调用 ``solution`` 里定义的
           函数、却**不**写 import，必须把 solution 与 test 放同一模块才能解析到名字。
       合并 + 保留 solution.py 对两种风格都兼容（已用真实样本验证），是最稳的单一策略。
    B. 判过 = ``returncode == 0 且 not timed_out``。pytest 退出码：0=全过、1=有断言失败、
       2=被中断、3=内部错、4=用法错、5=没收集到测试，全是非零 → 一律判不过；沙箱环境错
       返回 returncode=-1 → 也判不过（环境类失败按「运行失败」处理，见 sandbox 假设 D）。
       没过的样本按原因细分为 reason：test_failed（答案/测试真有 bug）、missing_dependency
       （pytest 报 ModuleNotFoundError，沙箱没装 torch/numpy/pandas 等第三方库——属环境缺失，
       不是题错，应在装齐依赖的环境重跑）、no_tests（rc=5，测试文件没定义 test_*）、timed_out、
       sandbox_error、missing_fields。细分只为便于统计与人工复核（见 §7.5），不改变「没过即淘汰」。
    C. 不把 stdout/stderr 写进输出文件：15k 条若每条都塞日志会让文件膨胀数 GB，筛 2 只
       需要「过/没过 + 原因 + 耗时」做统计与复核，故 verify 里只留结构化小字段。
    D. 每条输入的字段（question/prompt/test/solution/...）原样保留，仅追加一个 ``verify``
       子对象（passed / returncode / timed_out / duration_s / reason），实现「统计写回」
       （§5.2）——下游 f3 既能直接消费，也能一眼看到每条是如何过/没过的。
    E. 逐条顺序跑（不并发）：任务书 §5.2 明确「本机 Windows 清洗可能要几小时」，并发会引入
       进程树/端口/磁盘竞争的额外复杂度与潜在 bug；本地慢但稳，服务器侧可换增强版跑。
    F. 解释器取 config.JUDGE_PYTHON_EXE，空则回落当前 python（即运行本脚本的解释器，
       本地激活 venv_rlvr 时它自带 pytest）；这样 pytest 一定能被找到，不依赖 PATH 上的
       随机 python。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

__all__ = ["verify_one", "run_filter2"]

# ---------------------------------------------------------------------------
# 常量：优先读 config.py，读不到就用兜底默认值（与 sandbox/extract/score 同一套写法）
# ---------------------------------------------------------------------------

try:  # config.py 是任务书 §4 规定的常量集中地
    import config as _config  # type: ignore
except Exception:  # pragma: no cover - 仅在 config.py 缺失/损坏时走到
    _config = None


def _config_get(names: List[str], default):
    """按候选名在 config 里取常量，取不到返回 default。

    为什么：六个脚本都从 config 取统一阈值/路径，config 缺失时不能让脚本直接崩，
    给安全兜底，保证本地与服务器两侧都能跑（§7.2）。
    """
    if _config is not None:
        for name in names:
            value = getattr(_config, name, None)
            if value is not None:
                return value
    return default


# 单题超时（秒）——任务书 §7.3：10s
DEFAULT_TIMEOUT_S: int = int(_config_get(("TIMEOUT_S", "SANDBOX_TIMEOUT_S"), 10))
# 沙箱里固定的两个文件名（§5.4：候选代码 + 测试同目录）
SOLUTION_FILENAME: str = str(_config_get(("SOLUTION_FILENAME",), "solution.py"))
TESTCASE_FILENAME: str = str(_config_get(("TESTCASE_FILENAME",), "test_case.py"))
# 判分/验证用的解释器；空则回落 sys.executable（§7.3）
JUDGE_PYTHON_EXE: str = str(_config_get(("JUDGE_PYTHON_EXE",), ""))
# I/O 路径（§4 / §7.3 集中管理），命令行 --input/--output/--rejects 可覆盖。
DEFAULT_INPUT: str = str(_config_get(("STEP1_TEMPLATED",), "data/step1_templated.jsonl"))
DEFAULT_KEEP: str = str(_config_get(("STEP2_KEEP",), "data/step2_kept.jsonl"))
DEFAULT_REJECTS: str = str(_config_get(("STEP2_REJECTS",), "data/step2_rejects.jsonl"))

# 复用沙箱主函数（不依赖 sandbox 的 CLI 层，直接调 run_in_sandbox 拿 RunResult）
from sandbox import run_in_sandbox  # noqa: E402  (放在常量之后，确保 _config 已就绪)


def _resolve_python_exe() -> str:
    """返回真正用来跑 pytest 的解释器路径。

    为什么单独成函数：JUDGE_PYTHON_EXE 为空时回落 sys.executable（运行本脚本的那个），
    本地激活 venv_rlvr 即自带 pytest；显式设了就用显式的，服务器侧换解释器只改 config
    一处（§7.3）。
    """
    if JUDGE_PYTHON_EXE:
        return JUDGE_PYTHON_EXE
    return sys.executable


def _build_sandbox_files(record: Dict) -> Optional[Dict[str, str]]:
    """构造喂给沙箱的文件字典（见假设 A）。

    返回 None 表示缺 solution / test（字段不完整，无法验证），由调用方判为淘汰。

    为什么合并成一份 test_case.py：数据集两种 test 习惯并存（import / concat），
    合并 + 保留 solution.py 对两者都兼容，单一策略覆盖全部 15k 条（已验证）。
    """
    solution = record.get("solution")
    test = record.get("test")
    if not isinstance(solution, str) or not solution.strip():
        return None
    if not isinstance(test, str) or not test.strip():
        return None
    return {
        SOLUTION_FILENAME: solution,
        TESTCASE_FILENAME: solution + "\n" + test,
    }


def verify_one(record: Dict, timeout_s: int, enhanced: bool) -> Dict:
    """用官方解答验证单条样本，返回结构化的 verify 结果（假设 B/C）。

    verify 字典字段：
        passed     : bool   是否跑通
        returncode : int    沙箱/pytest 退出码（-1=沙箱环境错）
        timed_out  : bool   是否超时
        duration_s : float  执行耗时
        reason     : str|None  没过的简短原因（过则 None）

    reason 取值（便于 §7.5 统计与人工复核）：
        "missing_fields" 缺 solution/test，无法验证；
        "sandbox_error"  解释器起不来等环境错（returncode=-1）；
        "timed_out"      超时；
        "test_failed"    非零退出（断言失败/没收集到测试/中断/内部错 …）。
    """
    files = _build_sandbox_files(record)
    if files is None:
        return {
            "passed": False,
            "returncode": -1,
            "timed_out": False,
            "duration_s": 0.0,
            "reason": "missing_fields",
        }

    entry = [_resolve_python_exe(), "-m", "pytest", "-q", TESTCASE_FILENAME]
    try:
        result = run_in_sandbox(files, entry, timeout_s=timeout_s, enhanced=enhanced)
    except Exception as exc:
        # 入参校验类异常（_validate_args 抛的 TypeError/ValueError）属于调用方 bug，
        # 但单条样本不应拖垮整批：记成沙箱错误淘汰，并保留异常类型便于排查。
        return {
            "passed": False,
            "returncode": -1,
            "timed_out": False,
            "duration_s": 0.0,
            "reason": "sandbox_error:%s" % type(exc).__name__,
        }

    if result.timed_out:
        reason = "timed_out"
    elif result.returncode == -1:
        reason = "sandbox_error"
    elif result.returncode != 0:
        # pytest 把集合/报错信息写到 stdout（不是 stderr），故两类输出都要看。
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        # 缺失第三方库（torch/numpy/pandas...）是「沙箱环境没装依赖」，不是题目本身错：
        # 在装齐依赖的服务器环境重跑能通过。单独归类，便于事后把这些题挑出来重验，
        # 而不是和「答案真写错」混在一起（假设 B 的工程延伸）。
        if "ModuleNotFoundError" in combined or "ImportError" in combined or "No module named" in combined:
            reason = "missing_dependency"
        elif result.returncode == 5:
            # pytest 退出码 5 = 没收集到任何测试（test 文件没定义 test_*，或 harness 坏了）
            reason = "no_tests"
        else:
            reason = "test_failed"
    else:
        reason = None

    return {
        "passed": reason is None,
        "returncode": result.returncode,
        "timed_out": bool(result.timed_out),
        "duration_s": result.duration_s,
        "reason": reason,
    }


def run_filter2(
    input_path: str,
    keep_path: str,
    rejects_path: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    enhanced: bool = False,
) -> Dict[str, int]:
    """执行筛 2 全流程：逐条验证 → 分写「保留 / 淘汰」两个 jsonl。

    返回统计字典供 _main 打印（§7.5：读入 / 保留 / 淘汰 / 超时 / 沙箱错 / 缺字段 / 坏行 /
    平均耗时）。每条输入原样保留并追加 verify 子对象（假设 D：统计写回）。
    """
    stats = {
        "read": 0,
        "kept": 0,
        "rejected": 0,
        "timed_out": 0,
        "sandbox_error": 0,
        "missing_fields": 0,
        "test_failed": 0,
        "missing_dependency": 0,
        "no_tests": 0,
        "bad_lines": 0,
        "duration_sum_s": 0.0,
    }

    with open(input_path, "r", encoding="utf-8") as fin, open(
        keep_path, "w", encoding="utf-8", newline="\n"
    ) as fkeep, open(rejects_path, "w", encoding="utf-8", newline="\n") as frej:
        for index, raw in enumerate(fin):
            raw = raw.strip()
            if not raw:
                continue
            stats["read"] += 1

            try:
                record = json.loads(raw)
                if not isinstance(record, dict):
                    raise ValueError("每行必须是 JSON 对象")
            except Exception:
                stats["bad_lines"] += 1
                stats["rejected"] += 1
                frej.write(
                    json.dumps(
                        {"input_index": index, "verify": {"passed": False, "reason": "bad_json"}},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            verify = verify_one(record, timeout_s=timeout_s, enhanced=enhanced)
            stats["duration_sum_s"] += verify["duration_s"]
            if verify["reason"]:
                stats[verify["reason"]] = stats.get(verify["reason"], 0) + 1

            out = dict(record)  # 原样保留全部字段（prompt/test/solution/...）
            out["verify"] = verify

            if verify["passed"]:
                stats["kept"] += 1
                fkeep.write(json.dumps(out, ensure_ascii=False) + "\n")
            else:
                stats["rejected"] += 1
                frej.write(json.dumps(out, ensure_ascii=False) + "\n")

    return stats


# ---------------------------------------------------------------------------
# 命令行入口（任务书 §6.6：所有脚本支持 python xxx.py --input/--output）
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="筛 2：用官方解答过沙箱，淘汰跑不通的、保留跑得通的（任务书 §5.2）。",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="输入 jsonl 路径（默认 %s）" % DEFAULT_INPUT)
    parser.add_argument(
        "--output", default=DEFAULT_KEEP, help="保留集输出 jsonl（默认 %s）" % DEFAULT_KEEP
    )
    parser.add_argument(
        "--rejects", default=DEFAULT_REJECTS, help="淘汰集输出 jsonl（默认 %s）" % DEFAULT_REJECTS
    )
    parser.add_argument(
        "--timeout", type=float, default=None, help="覆盖单条超时（秒）；不传用默认 %d" % DEFAULT_TIMEOUT_S
    )
    parser.add_argument(
        "--enhanced", action="store_true", help="启用增强版沙箱（仅 Linux 生效，Windows 自动降级）"
    )
    return parser


def _main(argv: Optional[List[str]] = None) -> int:
    """CLI：解析参数 → 跑筛 2 → 按 §7.5 打印统计。

    返回进程退出码（0 正常；本地慢但稳，15k 条可能要数小时，见假设 E）。
    """
    args = _build_argparser().parse_args(argv)
    timeout_s = args.timeout if args.timeout is not None else DEFAULT_TIMEOUT_S
    stats = run_filter2(
        args.input, args.output, args.rejects, timeout_s=timeout_s, enhanced=args.enhanced
    )
    avg = (stats["duration_sum_s"] / stats["read"]) if stats["read"] else 0.0
    print(
        "[f2_verify] 读入 %d / 保留 %d / 淘汰 %d / 超时 %d / 沙箱错 %d / 缺字段 %d / "
        "测试失败 %d / 缺依赖 %d / 无测试 %d / 坏行 %d / 平均耗时 %.3fs"
        % (
            stats["read"],
            stats["kept"],
            stats["rejected"],
            stats["timed_out"],
            stats["sandbox_error"],
            stats["missing_fields"],
            stats["test_failed"],
            stats["missing_dependency"],
            stats["no_tests"],
            stats["bad_lines"],
            avg,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
