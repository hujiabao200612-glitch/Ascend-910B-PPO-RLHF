# -*- coding: utf-8 -*-
"""sandbox.py —— 代码执行沙箱（任务书 §5.2，全项目最核心的工程件）。

职责
    把「一组源文件 + 一条命令行入口」放进独立临时目录里执行，返回统一的
    RunResult(returncode, stdout, stderr, timed_out, duration_s)，
    供 score.py（判分器）与 f2_verify.py（筛 2）复用。

为什么是两层设计
    * 基础版（enhanced=False，Windows / Linux 都能跑）：只用进程隔离 + 超时，
      满足本地筛 2 与 100 条边界单测；本机清洗全程用它。
    * 增强版（enhanced=True，仅 Linux，服务器侧）：面向「模型自己生成的、
      内容不可信」的代码，追加 RLIMIT_AS（内存上限）、RLIMIT_CPU（CPU 上限），
      并尝试用 `unshare -n` 断网；两项不可用时只打 warning 降级，绝不报错。

任务书明确要求的两处细节
    1) `import resource` 只写在函数内的 try 块 —— Windows 没有该模块，
       写在文件顶层会让本地直接 import 失败（§5.2）；
    2) 禁止 shell=True —— 入口命令以参数列表传入，永不经过 shell 解析（§5.2）。

任务书未明确、按合理工程实践处理的假设（便于同事评审时逐条核对）
    A. 常量来源：config.py（§4 规定的常量集中地）已随本批次创建，这里优先读取；
       为兼容性仍保留「import 不到则回落任务书 §7.3 默认值（超时 10s、内存 2GB）」
       的兜底，路径异常时也不会崩。
    B. 超时杀进程用 Popen + communicate(timeout=) 而非 subprocess.run(timeout=)：
       语义等价，但 run() 抛出 TimeoutExpired 后拿不到进程句柄，无法完成
       §5.2 要求的「kill 进程树」。这里在超时后用 taskkill /T（Windows）或
       killpg（POSIX）杀掉整棵进程树，避免残留孤儿进程拖住后续上万条样本。
    C. 子进程统一 UTF-8 运行（PYTHONIOENCODING / PYTHONUTF8），否则 Windows 默认
       GBK 会让「print 大量输出」这类用例在父进程解码时炸掉；同时剔除 PYTHONPATH，
       避免子进程从沙箱外 import 到不该 import 的东西。
    D. 两类错误分开处理：入参格式错误（files 不是 dict、路径穿越、entry 为空、
       timeout <= 0）直接抛异常，属于调用方 bug；环境类失败（解释器不存在、
       进程起不来）返回 returncode=-1 的 RunResult，让流水线按「运行失败」而非
       「脚本崩溃」处理，便于统计淘汰原因。
    E. 代码原样写入、不做任何改写；stdout/stderr 不做截断（判分器要客观），
       代价是候选代码疯狂打印时父进程内存会被撑大，属已知限制。
    F. CLI（§6.6 要求所有脚本支持 --input/--output）的输入输出约定见 _main()。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence

__all__ = ["RunResult", "run_in_sandbox"]

# ---------------------------------------------------------------------------
# 常量：优先读 config.py，读不到就用任务书 §7.3 的默认值（见假设 A）
# ---------------------------------------------------------------------------

try:  # config.py 是任务书 §4 规定的常量集中地，分批生成阶段可能还不存在
    import config as _config  # type: ignore
except Exception:  # pragma: no cover - 仅在 config.py 缺失/损坏时走到
    _config = None


def _config_get(names: Sequence[str], default):
    """按候选名依次在 config 里取常量，取不到则返回 default。

    为什么要给多个候选名：任务书只规定了「常量集中在 config.py」，没有约定变量名，
    这里对常见命名保持宽容，避免因命名不一致而静默使用错误默认值。
    """
    if _config is not None:
        for name in names:
            value = getattr(_config, name, None)
            if value is not None:
                return value
    return default


# 单题超时（秒）——任务书 §7.3：10s
DEFAULT_TIMEOUT_S: int = int(_config_get(("TIMEOUT_S", "SANDBOX_TIMEOUT_S", "TIMEOUT"), 10))
# 增强版内存上限——任务书 §5.2 / §7.3：2GB
MEMORY_LIMIT_BYTES: int = int(
    _config_get(("MEMORY_LIMIT_BYTES", "MEMORY_LIMIT", "RLIMIT_AS_BYTES"), 2 * 1024 ** 3)
)
# RLIMIT_CPU 的硬限比软限多留几秒，让程序有机会打完日志再被 SIGKILL
CPU_LIMIT_GRACE_S: int = int(_config_get(("CPU_LIMIT_GRACE_S",), 2))
# 超时杀掉进程后，回收残留输出/僵尸进程的等待上限（秒）
REAP_TIMEOUT_S: int = int(_config_get(("REAP_TIMEOUT_S",), 5))
# 临时工作目录前缀，便于人工排查「有没有清理干净」
WORKDIR_PREFIX: str = str(_config_get(("SANDBOX_WORKDIR_PREFIX",), "rlvr_sandbox_"))


class RunResult(NamedTuple):
    """沙箱单次执行的结果（任务书 §5.2 规定的五元组）。

    用 NamedTuple 而不是 dataclass：既能 res.returncode 按属性访问，
    也能 `rc, out, err, to, dur = run_in_sandbox(...)` 直接解包，
    对判分器与筛 2 两种调用风格都友好，且与任务书写的元组顺序完全一致。
    """

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _warn(msg: str) -> None:
    """增强项不可用时的降级提示走 stderr。

    任务书 §5.2 要求「不可用则打 warning 降级，不报错」，所以这里只提示，
    绝不抛异常、也不影响 RunResult 的正常返回。
    """
    print("[sandbox][warn] %s" % msg, file=sys.stderr, flush=True)


def _validate_args(files: dict, entry: Sequence[str], timeout_s: float) -> None:
    """入参校验：把格式错误挡在子进程之前（见假设 D）。

    空数据 / 格式错误统一在这里变成清晰的异常信息，而不是让子进程莫名其妙地
    以 returncode=2 失败，方便流水线定位问题。
    """
    if not isinstance(files, dict):
        raise TypeError("files 必须是 dict[str, str]，实际得到 %r" % type(files).__name__)
    for name, content in files.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("files 的文件名必须是非空字符串，实际得到 %r" % (name,))
        if not isinstance(content, str):
            raise TypeError("files[%r] 的内容必须是 str，实际得到 %r" % (name, type(content).__name__))

    if isinstance(entry, (str, bytes)):
        # 传 "python foo.py" 这种字符串是最常见的误用：不做 shell 拆分（禁止 shell=True），
        # 直接报错，强迫调用方显式给出参数列表。
        raise ValueError("entry 必须是参数列表（如 ['python', '-m', 'pytest', '-v', 'test_case.py']），不能是字符串")
    if not isinstance(entry, (list, tuple)) or len(entry) == 0:
        raise ValueError("entry 必须是非空的参数列表，实际得到 %r" % (entry,))
    for token in entry:
        if not isinstance(token, str):
            raise TypeError("entry 的每个元素必须是 str，实际得到 %r" % (token,))

    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise TypeError("timeout_s 必须是数值，实际得到 %r" % (timeout_s,))
    if timeout_s <= 0:
        raise ValueError("timeout_s 必须 > 0（不给超时的沙箱等于没有沙箱），实际得到 %r" % (timeout_s,))


def _safe_target_path(workdir: str, name: str) -> str:
    """把相对文件名解析到工作目录内，拒绝绝对路径与目录穿越（../）。

    为什么必须做：files 的内容将来可能来自外部数据，若允许绝对路径或 ../，
    沙箱就成了「任意文件写入器」，隔离形同虚设。
    """
    normalized = name.replace("\\", "/")
    if os.path.isabs(name) or normalized.startswith("/") or normalized.startswith("~"):
        raise ValueError("files 只接受相对路径，拒绝 %r" % (name,))
    if os.path.isabs(normalized) or (len(normalized) > 1 and normalized[1] == ":"):
        # Windows 盘符形式（C:/x 或 C:x）在 ntpath.join 下会顶掉前缀，必须单独拦
        raise ValueError("files 只接受相对路径，拒绝 %r" % (name,))

    root = os.path.normpath(workdir)
    target = os.path.normpath(os.path.join(root, normalized))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError("文件 %r 解析后越出了沙箱工作目录（%s）" % (name, target))
    return target


def _materialize_files(workdir: str, files: Dict[str, str]) -> None:
    """把 files 字典落盘到工作目录；支持 `pkg/mod.py` 这类子目录路径。

    统一 UTF-8 + newline='\\n'：写出的文件在 Windows/Linux 上字节一致，
    保证「同一份数据在两台机器上跑出同样的结果」。
    """
    for name, content in files.items():
        target = _safe_target_path(workdir, name)
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)


def _build_child_env() -> Dict[str, str]:
    """构造子进程环境变量（见假设 C）。

    只做两件事：强制 UTF-8；剔除 PYTHONPATH / PYTHONSTARTUP 这类会把外部
    路径注入子进程的变量（隔离要求，同时避免宿主机环境意外影响判分）。
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # 不留 __pycache__，临时目录清理更干净
    for key in ("PYTHONPATH", "PYTHONSTARTUP"):
        env.pop(key, None)
    return env


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """杀掉整棵进程树（§5.2 的「超时则 kill 进程树」）。

    为什么不能只 proc.kill()：pytest / 候选代码可能再 fork 子进程，只杀父进程
    会留下孤儿进程继续吃 CPU；筛 2 要连跑上万条，残留进程会越滚越多。
    Windows 用 taskkill /F /T（/T = 连子进程），POSIX 用 killpg 杀整个进程组
    （配合 Popen(start_new_session=True) 使用）。
    """
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=REAP_TIMEOUT_S,
                # 这里不用 shell=True；taskkill 是系统命令，直接以参数列表调用
            )
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    try:
        proc.kill()  # 兜底：进程组杀失败时至少杀掉直接子进程
    except Exception:
        pass


def _make_preexec_fn(timeout_s: float) -> Optional[Callable[[], None]]:
    """构造增强版的 preexec_fn（内存 + CPU 上限）；不可用返回 None。

    关键点（任务书 §5.2 硬性要求）：`import resource` 必须在函数内的 try 里，
    因为 Windows 没有 resource 模块，顶层 import 会让本地直接崩。
    另外 preexec_fn 只在 fork 出来的子进程里执行，异常不能外抛（会变成
    SubprocessError），所以每一项限制都单独 try 包住。
    """
    if os.name == "nt" or not sys.platform.startswith("linux"):
        return None
    try:
        import resource  # 仅 Linux 存在
    except Exception:
        return None

    cpu_soft = max(1, int(timeout_s))
    cpu_hard = cpu_soft + CPU_LIMIT_GRACE_S

    def _apply_limits() -> None:  # pragma: no cover - 只在 Linux 子进程里执行
        try:
            resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_hard))
        except Exception:
            pass

    return _apply_limits


_UNSHARE_EXE: Optional[str] = None
_UNSHARE_PROBED: bool = False


def _unshare_exe() -> Optional[str]:
    """探测 `unshare -n` 是否真的可用，返回可执行文件路径或 None。

    为什么要探测而不是直接 which：unshare 需要 CAP_SYS_ADMIN 或放开 user
    namespace，容器里 often 存在二进制但执行会失败。这里先跑一次
    `unshare -n true` 试一下；结果缓存，避免上万条样本每条都探测。
    """
    global _UNSHARE_EXE, _UNSHARE_PROBED
    if not _UNSHARE_PROBED:
        _UNSHARE_PROBED = True
        exe = shutil.which("unshare")
        if exe:
            try:
                probe = subprocess.run(
                    [exe, "-n", "true"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=REAP_TIMEOUT_S,
                )
                _UNSHARE_EXE = exe if probe.returncode == 0 else None
            except Exception:
                _UNSHARE_EXE = None
    return _UNSHARE_EXE


def _run_entry(
    entry: List[str],
    workdir: str,
    timeout_s: float,
    preexec_fn: Optional[Callable[[], None]],
) -> RunResult:
    """在工作目录里跑一次入口命令，返回 RunResult。

    单独抽出来的原因：run_in_sandbox 负责「准备工作目录 + 保证清理」，
    这里只管「跑 + 超时处置」，职责清晰，异常路径也只有一处。
    """
    start = time.perf_counter()

    popen_kwargs = {
        "cwd": workdir,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": _build_child_env(),
        "shell": False,  # 任务书 §5.2 硬性要求：禁止 shell=True
        # 文本模式 + UTF-8 + errors='replace'：即使子进程吐出非法字节也不会
        # 让父进程抛 UnicodeDecodeError，最多丢几个乱码字符（客观性优先）
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        # 屏蔽子进程控制台窗口：筛 2 会连续起上万个进程，否则屏幕一直闪
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        # 独立进程组/会话：超时后才能用 killpg 一次性杀干净（见 _kill_process_tree）
        popen_kwargs["start_new_session"] = True
    if preexec_fn is not None:
        popen_kwargs["preexec_fn"] = preexec_fn

    try:
        proc = subprocess.Popen(entry, **popen_kwargs)
    except OSError as exc:
        # 环境类失败（解释器不存在、权限不足…）：返回结构化失败而不是抛异常（假设 D）
        duration = time.perf_counter() - start
        return RunResult(
            returncode=-1,
            stdout="",
            stderr="[sandbox] 无法启动子进程 %r：%s" % (entry, exc),
            timed_out=False,
            duration_s=round(duration, 6),
        )

    timed_out = False
    stdout, stderr = "", ""
    try:
        # communicate(timeout=) 会同时把两个管道读空，天然避免 PIPE 写满导致死锁
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        try:
            # 进程被杀后回收残留输出并 wait()；拿不到就退化为空串，不影响判定
            stdout, stderr = proc.communicate(timeout=REAP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            stdout, stderr = "", ""

    returncode = proc.returncode if proc.returncode is not None else -1
    duration = time.perf_counter() - start
    return RunResult(
        returncode=int(returncode),
        stdout=stdout or "",
        stderr=stderr or "",
        timed_out=bool(timed_out),
        duration_s=round(duration, 6),
    )


# ---------------------------------------------------------------------------
# 对外主函数
# ---------------------------------------------------------------------------


def run_in_sandbox(
    files: dict[str, str],
    entry: list[str],
    timeout_s: int = 10,
    enhanced: bool = False,
) -> RunResult:
    """在独立临时目录里执行 entry 命令，返回 RunResult（任务书 §5.2）。

    参数
        files     : {相对文件名: 文件内容}，会原样写入临时工作目录；
                    支持 `pkg/mod.py` 子目录；绝对路径/../ 一律拒绝（安全）。
        entry     : 命令行参数列表，例如 ["python", "-m", "pytest", "-v", "test_case.py"]；
                    以 cwd=工作目录 执行，所以相对路径都相对工作目录。
        timeout_s : 单次执行超时（秒），超时后 kill 整棵进程树并置 timed_out=True。
        enhanced  : True 时叠加内存/CPU 上限与断网（仅 Linux）；不可用时打 warning
                    降级为基础版，不影响返回值（§5.2）。

    返回
        RunResult(returncode, stdout, stderr, timed_out, duration_s)

    异常
        入参格式错误（TypeError / ValueError）会直接抛出，属于调用方 bug（假设 D）。

    幂等性：每次调用都新建并销毁工作目录，两次调用互不干扰（对应边界单测里
    「文件互不串扰」「临时目录清理」两个类别）。
    """
    _validate_args(files, entry, timeout_s)
    entry = [str(token) for token in entry]
    timeout_s = float(timeout_s)

    preexec_fn: Optional[Callable[[], None]] = None
    if enhanced:
        if not sys.platform.startswith("linux"):
            _warn("enhanced=True 仅支持 Linux（当前平台 %s），已降级为基础版" % sys.platform)
        else:
            preexec_fn = _make_preexec_fn(timeout_s)
            if preexec_fn is None:
                _warn("未能加载 resource 模块，内存/CPU 上限未生效，已降级为基础版")
            unshare = _unshare_exe()
            if unshare:
                # `unshare -n` 让子进程进入独立的网络命名空间；失败时下面的
                # Popen 会以 OSError 形式返回结构化失败，不会污染正常路径
                entry = [unshare, "-n", "--"] + entry
            else:
                _warn("unshare -n 不可用（缺少权限或未安装），网络隔离已跳过，降级运行")

    # 工作目录用完必须清理（§5.2）：mkdir + finally rmtree 比 TemporaryDirectory
    # 更可控 —— 子进程被强杀后偶尔会残留句柄，ignore_errors 保证主流程不被清理失败中断
    workdir = tempfile.mkdtemp(prefix=WORKDIR_PREFIX)
    try:
        _materialize_files(workdir, files)
        return _run_entry(entry, workdir, timeout_s, preexec_fn)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 命令行入口（任务书 §6.6：所有脚本支持 python xxx.py --input 路径 --output 路径）
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="沙箱批处理：读 jsonl 用例，逐条执行并写出结果（任务书 §6.6）。",
    )
    parser.add_argument("--input", required=True, help="输入 jsonl 路径，每行一个用例对象")
    parser.add_argument("--output", required=True, help="输出 jsonl 路径")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="覆盖单条用例的超时（秒）；不传则用用例里的 timeout_s 或默认 %d" % DEFAULT_TIMEOUT_S,
    )
    parser.add_argument("--enhanced", action="store_true", help="启用增强版沙箱（仅 Linux 生效）")
    return parser


def _main(argv: Optional[List[str]] = None) -> int:
    """CLI：把 run_in_sandbox 的入参序列化进 jsonl，跑完再序列化结果。

    输入 jsonl 每行一个用例对象（字段即函数入参，便于与判分器共用同一份接口）：
        {"id": "q001",
         "files": {"solution.py": "...", "test_case.py": "..."},
         "entry": ["python", "-m", "pytest", "-v", "test_case.py"],
         "timeout_s": 10, "enhanced": false}
    输出 jsonl 每行：
        {"id": ..., "returncode": ..., "stdout": ..., "stderr": ...,
         "timed_out": ..., "duration_s": ...}

    假设说明：任务书只规定「支持 --input/--output」，未规定 CLI 的具体数据格式；
    这里选择与函数接口一一对应的 jsonl（最省心智、可直接落盘复用）。
    结尾按 §7.5 打印统计（读入 / 写出 / 各异常计数 / 平均耗时）。
    """
    args = _build_argparser().parse_args(argv)

    total = written = bad_lines = timed_out_count = nonzero_count = 0
    total_duration = 0.0

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line_no, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                case = json.loads(raw)
                if not isinstance(case, dict):
                    raise ValueError("用例必须是 JSON 对象")
                files = case.get("files", {})
                entry = case.get("entry", [])
                timeout_s = args.timeout if args.timeout is not None else case.get("timeout_s", DEFAULT_TIMEOUT_S)
                enhanced = bool(args.enhanced or case.get("enhanced", False))
                result = run_in_sandbox(files, entry, timeout_s=timeout_s, enhanced=enhanced)
            except Exception as exc:
                # 坏行不静默吞掉：原样记账，统计里能看见
                bad_lines += 1
                record = {
                    "id": None,
                    "input_line": line_no,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
            else:
                record = {
                    "id": case.get("id"),
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "timed_out": result.timed_out,
                    "duration_s": result.duration_s,
                }
                total_duration += result.duration_s
                timed_out_count += 1 if result.timed_out else 0
                nonzero_count += 1 if result.returncode != 0 else 0

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    avg = (total_duration / written) if written else 0.0
    print(
        "[sandbox] 读入 %d 行 / 写出 %d 行 / 坏行 %d / 超时 %d / 非零返回 %d / 平均耗时 %.3fs"
        % (total, written, bad_lines, timed_out_count, nonzero_count, avg)
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
