# -*- coding: utf-8 -*-
"""score.py —— 判分器（任务书 §5.4 + §5.3 的 score() 薄壳）。

职责
    score_kernel()：判分器的「核心批改函数」。把一条候选样本（模型输出 response
        + 官方测试 test）交给 extract 抽出代码、再交给 sandbox 跑 pytest，
        解析 pytest 的 -v 输出，数出「通过 / 失败 / 错误」的用例数，返回
        (passed, total, detail)。它是筛 2（f2_verify）与 100 条边界单测共用的尺子。
    score()：判分器的「外层包装薄壳」。把 score_kernel 的客观结果（通过率、是否
        提取到代码、是否超时）换算成给 RL 模型的奖励分。这是 D3/D4 之前占位的
        **临时线性版本**，只实现「线性通过率 + 格式分 0.1 + 超时惩罚」，已加 TODO，
        将来要整段重写（见 score() 内注释）。

为什么把解析逻辑单独收在 _parse_pytest
    §5.4 明确说「部分分计数是唯一有难度的点」。把 pytest 输出解析从编排逻辑里
    拆出来，单测可以直接喂一段假 stdout 进来验证计数，不必真跑 pytest（快、稳、
    可复现），也是任务书 §6「100 条边界单测」能低成本覆盖运行类的关键。

任务书未明确、按合理工程实践处理的假设（便于同事评审时逐条核对）
    A. 文件名固定：候选代码写 solution.py、测试写 test_case.py（§5.4 原话），
       不支持改文件名；测试里 `from solution import xxx` 的约定由此保证。
    B. test 字段必须是「可直接跑的 pytest 源码」（§24/§5.4）。把 test_list
       （assert 字符串列表）拼成测试文件是 f2_verify 的职责，不放在 score_kernel。
    C. pytest 可用性按「解释器 + -m pytest --version」探测一次并缓存；缺失则降级
       为 `python test_case.py` 全有全无（passed ∈ {0, total}，§5.4 末段）。
    D. 计数三档（对应 §5.4 三句话）：
       ① 能从 stdout 匹配到 ≥1 条 `::test_x STATUS` 行 → 逐函数计（passed/失败/
          错误分别数，total=匹配行数=pytest 实际收集到的用例数，含 parametrize 展开）；
       ② 一条 `::test_x` 行都没有（典型：采集错误，如 test 里 import 了 solution
          没提供的符号，整文件 import 失败 → pytest 报 collection ERROR，不打印逐
          函数行）→ 保守 passed=0, total=源码里 `def test_` 的个数；
       ③ 源码里 `def test_` 都为 0 → 该题无效，passed=0, total=0（§5.4 末句）。
    E. SKIPPED / XFAIL / XPASS 这类非 PASSED 非 FAILED/ERROR 的状态，正常数据不会出现；
       暂定计入 total 但不算 passed（保守，不给部分分）。若 D3/D4 要改口径在此一处。
    F. 为什么主用逐函数行、不用末尾汇总行：汇总行会省略 0 计数类别、且采集错误时
       写成 `ERROR collecting ...`（是「采集失败」非「用例失败」），按它取 total 会错数；
       逐函数行 + `def test_` 源码兜底才能正确落回「函数数」。
    G. 子进程 stdout/stderr 只在 detail 里存尾部截断（默认 8000 字符），避免上万条
       样本把输出 jsonl 撑爆；判分客观性不受影响（计数只看 STATUS 关键字，不看正文）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

__all__ = ["score_kernel", "score"]

# ---------------------------------------------------------------------------
# 常量：优先读 config.py，读不到就用本文件下方的默认值（与 sandbox/extract 同一套写法）
# ---------------------------------------------------------------------------

try:  # config.py 是任务书 §4 规定的常量集中地
    import config as _config  # type: ignore
except Exception:  # pragma: no cover - 仅在 config.py 缺失/损坏时走到
    _config = None


def _config_get(names: List[str], default):
    """按候选名在 config 里取常量，取不到返回 default。"""
    if _config is not None:
        for name in names:
            value = getattr(_config, name, None)
            if value is not None:
                return value
    return default


DEFAULT_TIMEOUT_S: int = int(_config_get(("TIMEOUT_S",), 10))
JUDGE_PYTHON_EXE: str = str(_config_get(("JUDGE_PYTHON_EXE",), ""))
SOLUTION_FILENAME: str = str(_config_get(("SOLUTION_FILENAME",), "solution.py"))
TESTCASE_FILENAME: str = str(_config_get(("TESTCASE_FILENAME",), "test_case.py"))
FORMAT_BONUS: float = float(_config_get(("FORMAT_BONUS",), 0.1))
TIMEOUT_PENALTY_FACTOR: float = float(_config_get(("TIMEOUT_PENALTY_FACTOR",), 0.5))

# detail 里保留的 stdout/stderr 尾部长度上限（字符），见假设 G
_DETAIL_TAIL_CHARS = 8000


# pytest 缺失时的降级 runner（见 score_kernel 的 fallback 分支与 «假设 C» 末尾）。
# 为什么不用任务书字面的 `python test_case.py`：pytest 风格文件只有 `def test_()`
# 定义，裸跑 python 只会定义、从不执行断言，必然退出码 0 → 全假阳性通过，
# 与任务书要求的「全有全无」语义矛盾。这里注入一个最小 runner：导入 test_case、
# 依次调用每个 test_* 函数，任一抛异常则整体退出 1（全挂），否则退出 0（全过）。
_STANDALONE_RUNNER = (
    "import importlib.util, sys, traceback\n"
    "spec = importlib.util.spec_from_file_location('test_case', 'test_case.py')\n"
    "mod = importlib.util.module_from_spec(spec)\n"
    "spec.loader.exec_module(mod)\n"
    "funcs = [getattr(mod, n) for n in dir(mod)\n"
    "         if n.startswith('test_') and callable(getattr(mod, n))]\n"
    "failed = 0\n"
    "for fn in funcs:\n"
    "    try:\n"
    "        fn()\n"
    "    except Exception:\n"
    "        failed += 1\n"
    "        traceback.print_exc()\n"
    "sys.exit(1 if failed else 0)\n"
)
_STANDALONE_RUNNER_NAME = "_standalone_runner.py"


# ---------------------------------------------------------------------------
# pytest 输出解析（§5.4 核心难点，单独抽出便于单测）
# ---------------------------------------------------------------------------

# 逐函数结果行：``test_case.py::test_ok PASSED [ 33%]``
#   ::\s*(test_\S+?)      用例名：test_ 开头，最小匹配到第一个空白；\S+? 兼容
#                        parametrize 产生的 ``test_x[1-2]`` 这种带方括号名字
#   \s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b   状态词必须紧跟在空格后
# 注意：短汇总区 ``FAILED test_case.py::test_bad - ...`` 状态词在 ``::`` 之前，
# 顺序相反，本正则不会二次命中，不会重复计数（见文件头 «假设 F» 讨论）。
_TEST_STATUS_RE = re.compile(
    r"::\s*(test_\S+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b"
)
# 源码里数测试函数：``def test_`` / ``async def test_``（统计 total 的回落来源）
_DEF_TEST_RE = re.compile(r"^\s*(?:async\s+)?def\s+test_", re.MULTILINE)


def _count_test_functions(test_source: str) -> int:
    """数 test 源码里的测试函数个数（§5.4：拿不到 -v 行时的 total 回落来源）。"""
    if not test_source:
        return 0
    return len(_DEF_TEST_RE.findall(test_source))


def _parse_pytest(stdout: str, test_source: str) -> Tuple[int, int, Dict[str, object]]:
    """解析 pytest -v 输出，返回 (passed, total, 计数明细)。

    三档逻辑见文件头 «假设 D»：
        ① 有逐函数行 → 逐函数计，total=行数；
        ② 无逐函数行（采集错误/输出异常）→ 保守 passed=0, total=源码 def 数；
        ③ 源码也无 def → (0, 0) 由调用方判为无效题。
    """
    text = stdout or ""
    matches = _TEST_STATUS_RE.findall(text)
    passed = sum(1 for _, status in matches if status == "PASSED")
    failed = sum(1 for _, status in matches if status == "FAILED")
    error = sum(1 for _, status in matches if status == "ERROR")
    other = len(matches) - passed - failed - error  # SKIPPED / XFAIL / XPASS（假设 E）

    n_status = len(matches)
    if n_status > 0:
        # ① 正常路径：pytest 真的跑完了，按每行状态计数
        return passed, n_status, {
            "failed": failed,
            "error": error,
            "other": other,
            "conservative": False,
        }

    # ② 没有逐函数行：通常是采集错误（import 失败）或被 unexpected 截断。
    #    保守取 passed=0, total=源码里 def test_ 的个数（见假设 D/②）。
    n_def = _count_test_functions(test_source)
    return 0, n_def, {
        "failed": 0,
        "error": 0,
        "other": 0,
        "conservative": True,
    }


def _tail(text: str, n: int = _DETAIL_TAIL_CHARS) -> str:
    """取文本尾部 n 字符，用于 detail 里截断 stdout/stderr（假设 G）。"""
    if not text:
        return ""
    return text if len(text) <= n else text[-n:]


# ---------------------------------------------------------------------------
# pytest 可用性探测（§5.4 末段降级依据）
# ---------------------------------------------------------------------------

# 按解释器缓存探测结果，避免批量跑时每条都起一次子进程
_PYTEST_PROBE: Dict[str, bool] = {}


def _pytest_available(python_exe: str) -> bool:
    """探测该解释器能否跑 pytest；结果按 python_exe 缓存。

    为什么不直接 try import pytest：判分器跑在 venv 的解释器里，本文件可能从别的
    python 启动；必须探测「真正用来跑沙箱的那个解释器」有没有 pytest。
    """
    if python_exe in _PYTEST_PROBE:
        return _PYTEST_PROBE[python_exe]
    try:
        probe = subprocess.run(
            [python_exe, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT_S,
        )
        ok = probe.returncode == 0 and "pytest" in (probe.stdout + probe.stderr).lower()
    except Exception:
        ok = False
    _PYTEST_PROBE[python_exe] = ok
    return ok


def _resolve_python_exe(python_exe: Optional[str]) -> str:
    """决定跑沙箱用的解释器：入参 > config.JUDGE_PYTHON_EXE > 当前解释器。"""
    if python_exe:
        return python_exe
    if JUDGE_PYTHON_EXE:
        return JUDGE_PYTHON_EXE
    return sys.executable


# ---------------------------------------------------------------------------
# 核心批改函数
# ---------------------------------------------------------------------------


def score_kernel(
    response: str,
    test: str,
    python_exe: Optional[str] = None,
    timeout_s: Optional[int] = None,
    enhanced: bool = False,
) -> Tuple[int, int, Dict[str, object]]:
    """判分核心：抽代码 → 沙箱跑 pytest → 解析通过/失败数（任务书 §5.4）。

    参数
        response   : 模型生成的完整输出文本（extract 的输入）。
        test       : 官方测试源码（pytest 风格，内部 `from solution import xxx`）。
        python_exe : 跑沙箱的解释器；不传则用 config.JUDGE_PYTHON_EXE 或 sys.executable。
        timeout_s  : 单题超时（秒）；不传用 config.TIMEOUT_S。
        enhanced   : 是否启用增强版沙箱（仅 Linux，§5.2）。

    返回
        (passed, total, detail)
            passed : 通过用例数（int）
            total  : 测试函数/用例总数（int）；0 表示该题无效（§5.4 末句）
            detail : dict，含 mode / timed_out / returncode / duration_s /
                     failed / error / 是否保守计数 / stdout_tail / stderr_tail 等，
                     供筛 2 与人工排查使用。

    设计要点（对应 §5.4）
        * 无代码块（extract 返回 None）→ 直接 (0, 0, mode="no_code")，不进沙箱；
        * pytest 可用 → 跑 `python -m pytest test_case.py -v`，按 _parse_pytest 三档计数；
        * pytest 缺失 → 降级 `python test_case.py`，退出码 0 则全过、否则全挂
          （passed ∈ {0, total}，total=源码 def 数）。
    本函数不抛异常：任何异常都折算成 (0, total, mode="error") 记进 detail，
    保证判分器作为「全项目尺子」永不因为单条脏样本而中断整批。
    """
    # 延迟 import：避免循环依赖，也让 score.py 单独导入时不强制拉起 sandbox/extract
    import extract
    import sandbox

    detail: Dict[str, object] = {}

    # 边界：response / test 不是字符串或为空
    if not isinstance(response, str) or not isinstance(test, str):
        detail["mode"] = "error"
        detail["note"] = "response 与 test 必须都是 str"
        return 0, 0, detail
    if not test.strip():
        # test 源码为空 → 没有任何测试函数 → 该题无效（§5.4 ③）
        detail["mode"] = "no_tests"
        detail["note"] = "test 为空，无测试函数"
        return 0, 0, detail

    # ① 抽代码（§3 / §5.3）：无围栏代码块 → 无代码
    try:
        code = extract.extract_python(response)
    except Exception as exc:  # extract 本身不应抛，但判分器必须兜底
        detail["mode"] = "error"
        detail["note"] = "extract 异常：%s" % exc
        return 0, _count_test_functions(test), detail

    if code is None:
        detail["mode"] = "no_code"
        detail["note"] = "未提取到代码块"
        return 0, _count_test_functions(test), detail

    # 写文件 + 决定解释器/入口
    files = {SOLUTION_FILENAME: code, TESTCASE_FILENAME: test}
    py = _resolve_python_exe(python_exe)
    to = DEFAULT_TIMEOUT_S if timeout_s is None else int(timeout_s)

    has_pytest = _pytest_available(py)
    if has_pytest:
        # 加 -p no:cacheprovider：不写 .pytest_cache，临时目录清理更干净
        entry = [py, "-m", "pytest", TESTCASE_FILENAME, "-v", "-p", "no:cacheprovider"]
    else:
        # §5.4 末段降级：pytest 缺失 → 用独立 runner 真正执行 test_* 函数，全有全无。
        # 注：此处偏离任务书字面的 `python test_case.py`（对 pytest 风格文件是空操作，
        # 会假阳性全过），改用 _STANDALONE_RUNNER 才符合「全有全无」语义（见其注释）。
        files = dict(files)  # 不污染上面那份，避免影响 pytest 分支的复用
        files[_STANDALONE_RUNNER_NAME] = _STANDALONE_RUNNER
        entry = [py, _STANDALONE_RUNNER_NAME]

    try:
        result = sandbox.run_in_sandbox(files, entry, timeout_s=to, enhanced=enhanced)
    except Exception as exc:
        # sandbox 入参校验类异常（理论上 files/entry 都合法）；折算成保守失败
        detail["mode"] = "error"
        detail["note"] = "sandbox 异常：%s" % exc
        return 0, _count_test_functions(test), detail

    detail["timed_out"] = result.timed_out
    detail["returncode"] = result.returncode
    detail["duration_s"] = result.duration_s
    detail["pytest_available"] = has_pytest

    if has_pytest:
        passed, total, counts = _parse_pytest(result.stdout, test)
        detail["mode"] = "pytest"
        detail["failed"] = counts["failed"]
        detail["error"] = counts["error"]
        detail["other"] = counts["other"]
        detail["conservative"] = counts["conservative"]
    else:
        # 降级路径：退出码 0 → 全过，否则全挂
        detail["mode"] = "pytest_fallback"
        total = _count_test_functions(test)
        passed = total if result.returncode == 0 else 0

    detail["passed"] = passed
    detail["total"] = total
    detail["stdout_tail"] = _tail(result.stdout)
    detail["stderr_tail"] = _tail(result.stderr)
    return passed, total, detail


# ---------------------------------------------------------------------------
# 外层奖励包装（临时版，TODO：D3/D4 重写）
# ---------------------------------------------------------------------------


def score(
    passed: int,
    total: int,
    *,
    extracted: bool = True,
    timed_out: bool = False,
) -> float:
    """把客观判分结果换算成给 RL 模型的奖励分（任务书 §5.3 薄壳）。

    ⚠️ TODO(D3/D4)：这是占位用的**临时线性版本**，奖励公式将在 D3/D4 整段重写
    （计划改为 margin 归一化 / 长度惩罚 / 部分用例加权等，与 RLVR 的奖励塑形对齐）。
    现在只实现任务书字面要求的「线性通过率 + 格式分 0.1 + 超时惩罚」：
        reward = (passed / total) + (FORMAT_BONUS if extracted else 0)
        reward = min(reward, 1.0)            # 格式分可能把满分区顶过 1.0，夹住
        if timed_out: reward *= TIMEOUT_PENALTY_FACTOR
        reward = max(reward, 0.0)

    参数
        passed    : score_kernel 返回的通过用例数
        total     : 测试总数；<=0 时通过率按 0 处理
        extracted : 是否成功提取到代码（无代码块应为 False）
        timed_out : 沙箱是否超时（由 score_kernel 的 detail["timed_out"] 透传）
    """
    rate = (passed / total) if total and total > 0 else 0.0
    reward = rate + (FORMAT_BONUS if extracted else 0.0)
    reward = min(reward, 1.0)
    if timed_out:
        reward *= TIMEOUT_PENALTY_FACTOR
    return max(reward, 0.0)


# ---------------------------------------------------------------------------
# 命令行入口（任务书 §6.6：所有脚本支持 python xxx.py --input/--output；§7.5 统计）
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="判分器批处理：读 jsonl（每行含 response/test），逐条跑沙箱并写结果。",
    )
    parser.add_argument("--input", required=True, help="输入 jsonl 路径，每行一个样本对象")
    parser.add_argument("--output", required=True, help="输出 jsonl 路径")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="覆盖单条超时（秒）；不传用 config.TIMEOUT_S",
    )
    parser.add_argument("--enhanced", action="store_true", help="启用增强版沙箱（仅 Linux 生效）")
    return parser


def _main(argv: Optional[List[str]] = None) -> int:
    """CLI：读 jsonl → 对每行调 score_kernel + score → 写出并补字段。

    输入 jsonl 每行：{"id": ..., "response": "...", "test": "..."}（可带 timeout_s / enhanced）
    输出 jsonl 每行 = 原对象 + {passed, total, reward, mode, timed_out, duration_s,
        failed, error, conservative, stdout_tail, stderr_tail}
    结尾按 §7.5 打印统计：读入 / 写出 / 无代码 / 超时 / 平均奖励。
    """
    args = _build_argparser().parse_args(argv)

    total = written = no_code = timed_out = bad_lines = 0
    reward_sum = 0.0

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line_no, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                rec = json.loads(raw)
                if not isinstance(rec, dict):
                    raise ValueError("每行必须是 JSON 对象")
                response = rec.get("response", "")
                test = rec.get("test", "")
                timeout_s = args.timeout if args.timeout is not None else rec.get("timeout_s")
                enhanced = bool(args.enhanced or rec.get("enhanced", False))

                passed, ttl, detail = score_kernel(
                    response, test, timeout_s=timeout_s, enhanced=enhanced
                )
                reward = score(
                    passed,
                    ttl,
                    extracted=detail.get("mode") not in ("no_code", "error"),
                    timed_out=bool(detail.get("timed_out", False)),
                )
            except Exception as exc:
                bad_lines += 1
                out = {
                    "input_line": line_no,
                    "passed": 0,
                    "total": 0,
                    "reward": 0.0,
                    "mode": "error",
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
            else:
                out = dict(rec)
                out["passed"] = passed
                out["total"] = ttl
                out["reward"] = reward
                out["mode"] = detail.get("mode")
                out["timed_out"] = detail.get("timed_out")
                out["duration_s"] = detail.get("duration_s")
                out["failed"] = detail.get("failed")
                out["error"] = detail.get("error")
                out["conservative"] = detail.get("conservative")
                out["stdout_tail"] = detail.get("stdout_tail")
                out["stderr_tail"] = detail.get("stderr_tail")
                no_code += 1 if detail.get("mode") == "no_code" else 0
                timed_out += 1 if detail.get("timed_out") else 0
                reward_sum += reward

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    avg_reward = (reward_sum / written) if written else 0.0
    print(
        "[score] 读入 %d 行 / 写出 %d 行 / 无代码 %d / 超时 %d / 坏行 %d / 平均奖励 %.3f"
        % (total, written, no_code, timed_out, bad_lines, avg_reward)
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
