# -*- coding: utf-8 -*-
# =============================================================================
# 设计目标：结果可读性极强，方便人工逐条核对「每一个 case 的期望分数」。
#   * 本文件既可被 pytest 发现（parametrize 出 100 个节点，pytest -v 逐条显示），
#     也可作为脚本直接运行：`venv_rlvr/Scripts/python.exe pipeline/tests/test_boundary_100.py`
#     脚本模式会打印一张「期望 vs 实际」对照表，每条 case 给出：
#         - 期望：passed / total / timed_out / extracted / mode  => 期望score
#         - 实际：passed / total / timed_out / extracted / mode  => 实际score
#         - 结果：PASS / FAIL
#   人工核对时，重点看「期望score」是否合理，以及实际是否与期望一致。
#
# 期望值来源（务必与实现对齐，便于评审）：
#   score = (passed/total if total>0 else 0) + (0.1 if extracted else 0)
#           -> 夹到 [0,1] -> 若 timed_out 再 ×0.5
#   extracted = (mode not in ("no_code","error"))
#   mode: no_code / no_tests / error / pytest / pytest_fallback
#   详见 pipeline/score.py、pipeline/extract.py、pipeline/sandbox.py。
#
# ┌────────────────────────────────────────────────────────────────────────┐
# │ 已复核：（待人工签名：姓名 / 日期）                                        │
# │ 说明：本文件 100 条 case 的「期望分数」由上面的公式推算，需评审人逐条核对  │
# │       签名后方算通过验收闸门（任务书 §6.2）。其中「0 道测试用例却得 0.1」    │
# │       等边界属当前实现的真实行为，已在 note 中标注，请重点确认是否符合预期。│
# └────────────────────────────────────────────────────────────────────────┘
# =============================================================================

import os
import sys
import glob
import json
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# 把 pipeline 目录加入 sys.path，使本文件能 import 同目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config          # noqa: E402
import extract         # noqa: E402
import sandbox         # noqa: E402
import score          # noqa: E402
import pytest          # noqa: E402


# ---------------------------------------------------------------------------
# 小工具：构造模型输出 / 测试源码
# ---------------------------------------------------------------------------
NL = "\n"


def fence(code: str, lang: str = "python") -> str:
    """用围栏包住代码块。"""
    return "```" + lang + NL + code + NL + "```"


def think(text: str) -> str:
    return "让我梳理一下思路：" + text + "\n\n"


def normal_response(with_import: bool = True, with_helper: bool = True) -> str:
    """正常情况的标准回答：有思考、有 import、有辅助函数，主逻辑 solve 恒等。"""
    lines = []
    if with_import:
        lines.append("import math")
    if with_helper:
        lines.append("")
        lines.append("def _aux(x):")
        lines.append("    return x * 2")
    lines.append("")
    lines.append("def solve(x):")
    lines.append("    return x")
    code = NL.join(lines)
    prose = ("我先想一下：题目要实现一个 solve。引入 math 做数值处理，"
             "写个辅助函数 _aux，主逻辑直接返回 x。")
    return think(prose) + fence(code)


def normal_tests(P: int, T: int) -> str:
    """构造 T 个测试，其中前 P 个期望正确、其余故意期望错误（=> 恰好 P 个通过）。

    仅用于「正常情况」：solution 是恒等函数（永远正确），失败完全由测试侧控制。
    """
    parts = ["from solution import solve", ""]
    for i in range(T):
        exp = i if i < P else i + 1
        parts.append("def test_p%d():" % i)
        parts.append("    assert solve(%d) == %d" % (i, exp))
        parts.append("")
    return NL.join(parts)


def correct_tests(T: int) -> str:
    """构造 T 个测试，全部断言 solve(i)==i（正确期望）。

    用于「运行类-部分对」：solution 带真实 bug（对 failing 输入返回错误值），
    失败只来自 bug 本身，故 passed = T - len(failing)。与 normal_tests 的
    「测试侧故意写错」不叠加，避免重复计数。
    """
    parts = ["from solution import solve", ""]
    for i in range(T):
        parts.append("def test_%d():" % i)
        parts.append("    assert solve(%d) == %d" % (i, i))
        parts.append("")
    return NL.join(parts)


def bug_code(failing) -> str:
    """带真实 bug 的 solve：对 failing 集合里的输入故意返回错误值。"""
    return ("def solve(x):" + NL
            + "    if x in %r:" % (list(failing),) + NL
            + "        return x + 100  # 故意错误" + NL
            + "    return x" + NL)


def bug_response(failing) -> str:
    return think("实现 solve，但对某些输入故意返回错误值。") + fence(bug_code(failing))


# 提取类常用的「正确代码」与「错误代码」基准串（expect_extract 直接比对）
EX_CODE_STR = "def solve(x):" + NL + "    return x"
EX_BUG_STR = "def solve(x):" + NL + "    return x + 1"


def extest(n: int = 3) -> str:
    """提取类配套测试：对 solve(i)==i 做 n 次断言；代码正确则全过。"""
    parts = ["from solution import solve", ""]
    for i in range(n):
        parts.append("def test_%d():" % i)
        parts.append("    assert solve(%d) == %d" % (i, i))
        parts.append("")
    return NL.join(parts)


EX_TEST = extest(3)


# ---------------------------------------------------------------------------
# 期望分公式（与 score.score 保持一致；用 config 常量，D3/D4 改常量即同步）
# ---------------------------------------------------------------------------
def formula(passed: int, total: int, extracted: bool, timed_out: bool) -> float:
    rate = (passed / total) if total and total > 0 else 0.0
    r = rate + (config.FORMAT_BONUS if extracted else 0.0)
    r = min(r, 1.0)
    if timed_out:
        r *= config.TIMEOUT_PENALTY_FACTOR
    return max(r, 0.0)


def _score_of(passed, total, detail) -> float:
    """按 score_kernel 的 detail 还原出给 RL 的奖励分（与 score.py CLI 同口径）。"""
    mode = detail.get("mode")
    extracted = mode not in ("no_code", "error")
    timed_out = bool(detail.get("timed_out", False))
    return score.score(passed, total, extracted=extracted, timed_out=timed_out)


# ---------------------------------------------------------------------------
# case 注册表（顺序即编号 N01..N100）
# ---------------------------------------------------------------------------
CASES = []


def add(cat, desc, response, test, expect, **kw):
    c = {
        "id": "N%02d" % (len(CASES) + 1),
        "cat": cat,
        "desc": desc,
        "response": response,
        "test": test,
        "expect": expect,
    }
    c.update(kw)
    CASES.append(c)
    return c


# ===== A. 正常情况（20）====================================================
_NORMAL = [
    (1, 1), (2, 2), (2, 3), (3, 4), (5, 5), (4, 5), (3, 6), (7, 7),
    (6, 8), (0, 9), (10, 10), (5, 10), (1, 3), (0, 2), (1, 4),
    (5, 6), (8, 8), (4, 7), (12, 12), (9, 12),
]
for _i, (P, T) in enumerate(_NORMAL):
    wi = (_i % 2 == 0)
    wh = (_i % 3 != 0)
    add("正常情况",
        "有思考+import+辅助函数，%d/%d 通过" % (P, T),
        normal_response(with_import=wi, with_helper=wh),
        normal_tests(P, T),
        {"passed": P, "total": T, "timed_out": False, "extracted": True, "mode": "pytest"})


# ===== B. 提取类（20）======================================================
def _ext_ok(desc, response, note=""):
    add("提取类", desc, response, EX_TEST,
        {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
        expect_extract=EX_CODE_STR, note=note)


def _ext_none(desc, response, note=""):
    add("提取类", desc, response, EX_TEST,
        {"passed": 0, "total": 3, "timed_out": False, "extracted": False, "mode": "no_code"},
        expect_extract=None, note=note)


# B01 多代码块取第一个（第一个正确）
add("提取类", "多代码块：取第一个（正确）代码块",
    fence(EX_CODE_STR) + NL + "下面这段是错的，不应被采用：" + NL + fence(EX_BUG_STR),
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B02 多代码块：python 块后接裸块，取第一个 python 块
add("提取类", "多代码块：python 块 + 裸块，取第一个",
    fence(EX_CODE_STR) + NL + "```" + NL + EX_BUG_STR + NL + "```",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B03 标签大写 Python
add("提取类", "语言标签大写 ```Python",
    "```Python" + NL + EX_CODE_STR + NL + "```",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B04 标签全大写 PYTHON
add("提取类", "语言标签全大写 ```PYTHON",
    "```PYTHON" + NL + EX_CODE_STR + NL + "```",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B05 标签 python3
add("提取类", "语言标签 ```python3",
    "```python3" + NL + EX_CODE_STR + NL + "```",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B06 裸 ```（无标签）
add("提取类", "无语言标签的裸 ``` 围栏",
    "```" + NL + EX_CODE_STR + NL + "```",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B07 先 text 块后 python 块（跳过说明块）
add("提取类", "先 ```text 说明块，再 ```python（跳过 text）",
    "```text" + NL + "下面是说明，不是代码" + NL + "```" + NL + "```python" + NL + EX_CODE_STR + NL + "```",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B08 CRLF 换行
_crlf = "```python\r\n" + EX_CODE_STR + "\r\n```"
add("提取类", "围栏内使用 CRLF 换行（应归一化为 \\n）",
    _crlf, EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B09 缩进的围栏（行首有空格）
add("提取类", "开/闭围栏行首有缩进（仍应识别）",
    "  ```python" + NL + EX_CODE_STR + NL + "  ```",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B10 代码内含行内 ``` 字面量（不应误判为闭围栏）
_inline = 'def solve(x):' + NL + '    s = "```"' + NL + '    return x'
add("提取类", "代码内含行内 ``` 字面量（闭围栏须独占一行）",
    "```python" + NL + _inline + NL + "```",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=_inline)

# B11 未闭合围栏（被截断）
_ext_none("未闭合围栏（无闭围栏 => 视为无代码）",
          "```python" + NL + EX_CODE_STR + NL,
          "任务书 §5.3 假设 C：只有开围栏无闭围栏 => 无代码")

# B12 开围栏行尾跟解释文字（任务书 §5.3 假设 E => 无代码）
_ext_none("开围栏行尾跟解释文字 ```python 以下是实现 => 无代码",
          "```python 以下是实现" + NL + EX_CODE_STR + NL + "```",
          "严格照 §5.3 容忍范围（仅大小写+标签缺失），行尾文字不接受")

# B13 完全没有代码块
_ext_none("纯文本、不含任何围栏代码块 => 无代码",
          "这道题我不会做，下面是我的思路文字，但没有代码块。",
          "无围栏 => extract 返回 None")

# B14 多代码块：第一个是错误的（验证「取第一个」即使更差）
add("提取类", "多代码块：第一个是错的（取第一个 => 全失败）",
    fence(EX_BUG_STR) + NL + "正确版本在后面但不会被取：" + NL + fence(EX_CODE_STR),
    EX_TEST,
    {"passed": 0, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_BUG_STR,
    note="验证「取第一个」语义，即使第一个质量更差")

# B15 标签混排大小写 PyThOn
add("提取类", "语言标签混排大小写 ```PyThOn",
    "```PyThOn" + NL + EX_CODE_STR + NL + "```",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B16 代码块内有首尾空行（应只去首尾换行）
add("提取类", "围栏内首尾有空行（strip 仅去换行）",
    "```python" + NL + NL + EX_CODE_STR + NL + NL + "```",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B17 闭围栏行尾有空格
add("提取类", "闭围栏行尾带空格 ```   ",
    "```python" + NL + EX_CODE_STR + NL + "```   ",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B18 回复只有代码块、没有任何其它文字
add("提取类", "回复仅含代码块、无前后文字",
    "```python" + NL + EX_CODE_STR + NL + "```",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B19 多个 text 块夹着 python 块（取 python）
add("提取类", "多个 ```text 块夹着 ```python（跳过说明）",
    "```text" + NL + "a" + NL + "```" + NL + "```text" + NL + "b" + NL + "```" + NL
    + "```python" + NL + EX_CODE_STR + NL + "```",
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=EX_CODE_STR)

# B20 提取带 import 的代码
_code_import = "import os" + NL + NL + "def solve(x):" + NL + "    return x"
add("提取类", "提取含 import 的代码块",
    think("下面给出实现。") + fence(_code_import),
    EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    expect_extract=_code_import)


# ===== C. 运行类（20）======================================================
# C01-C05 部分对（真实 bug）
_PARTIAL = [
    ({0}, 5, 4),          # x==0 错 => 4/5
    ({1, 2}, 5, 3),       # 1,2 错 => 3/5
    ({0, 1, 2}, 5, 2),    # 0,1,2 错 => 2/5
    ({3}, 4, 3),          # 3 错 => 3/4
    ({0, 1, 2, 3, 4}, 5, 0),  # 全错 => 0/5
]
for _i, (fail, T, P) in enumerate(_PARTIAL):
    add("运行类", "部分对：真实 bug（%d/%d 通过）" % (P, T),
        bug_response(fail), correct_tests(T),
        {"passed": P, "total": T, "timed_out": False, "extracted": True, "mode": "pytest"},
        note="解决方案对输入 %r 故意返回错误值；测试均断言正确期望，失败只来自 bug" % (list(fail),))

# C06-C08 死循环（超时杀进程）
_loop_mod = "while True:" + NL + "    pass" + NL  # 模块顶层死循环 => import 即挂起
_loop_fn = "def solve(x):" + NL + "    while True:" + NL + "        pass" + NL


def _loop_case(desc, code, test, T):
    return add("运行类", desc,
               fence(code), test,
               {"passed": 0, "total": T, "timed_out": True, "extracted": True, "mode": "pytest"},
               timeout_s=2,
               note="死循环被超时杀掉 => 保守计数(0,%d)，奖励×0.5 => 期望 0.05" % T)


_loop_case("死循环（模块顶层 while）：import 挂起超时",
           _loop_mod, "from solution import nope" + NL + "def test_a():" + NL + "    pass" + NL, 1)
_loop_case("死循环（被调用的函数内）：测试调用即挂起超时",
           _loop_fn, "from solution import solve" + NL + "def test_a():" + NL + "    assert solve(1) == 1" + NL, 1)
_loop_case("死循环（被调用函数）且测试 3 个用例：超时 total=3",
           _loop_fn, "from solution import solve" + NL
           + NL.join("def test_%d():" % i + NL + "    assert solve(%d) == %d" % (i, i)
                     for i in range(3)) + NL, 3)

# C09-C11 语法错误（采集失败 => 保守 0/defcount）
add("运行类", "语法错误：def 缺冒号 => 采集失败保守 0/1",
    fence("def solve(x)" + NL + "    return x" + NL),
    "from solution import solve" + NL + "def test_a():" + NL + "    assert solve(1) == 1" + NL,
    {"passed": 0, "total": 1, "timed_out": False, "extracted": True, "mode": "pytest"},
    note="import 失败 => 无 ::test 行 => 保守 0/1，奖励 0.1")

add("运行类", "语法错误：return 后缺操作数 => 采集失败保守 0/2",
    fence("def solve(x):" + NL + "    return x +" + NL),
    "from solution import solve" + NL
    + "def test_a():" + NL + "    assert solve(1) == 1" + NL
    + "def test_b():" + NL + "    assert solve(2) == 2" + NL,
    {"passed": 0, "total": 2, "timed_out": False, "extracted": True, "mode": "pytest"},
    note="`return x +` 是 SyntaxError => import 失败 => 无 ::test 行 => 保守 0/2，奖励 0.1")

add("运行类", "语法错误：括号不闭合 => 采集失败保守 0/1",
    fence("def solve(x):" + NL + "    return [1, 2" + NL),
    "from solution import solve" + NL + "def test_a():" + NL + "    assert solve(1) == 1" + NL,
    {"passed": 0, "total": 1, "timed_out": False, "extracted": True, "mode": "pytest"})

# C12-C20 抛出异常 / 运行错误
add("运行类", "抛出异常：solve 恒 raise => 全部 ERROR 0/3",
    fence("def solve(x):" + NL + "    raise RuntimeError('boom')" + NL),
    "from solution import solve" + NL
    + NL.join("def test_%d():" % i + NL + "    assert solve(%d) == %d" % (i, i)
              for i in range(3)) + NL,
    {"passed": 0, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    note="运行时异常 => ERROR 状态，不计入 passed")

add("运行类", "抛出异常：仅负数 raise，其余正常 2/3",
    fence("def solve(x):" + NL + "    if x < 0:" + NL + "        raise ValueError('neg')" + NL
          + "    return x" + NL),
    "from solution import solve" + NL
    + "def test_neg():" + NL + "    assert solve(-1) == -1" + NL
    + "def test_0():" + NL + "    assert solve(0) == 0" + NL
    + "def test_1():" + NL + "    assert solve(1) == 1" + NL,
    {"passed": 2, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    note="负数触发 ValueError(ERROR)，0/1 正常 => 2/3")

add("运行类", "NameError：引用未定义变量 => 全部 ERROR 0/2",
    fence("def solve(x):" + NL + "    return x + y" + NL),
    "from solution import solve" + NL
    + "def test_a():" + NL + "    assert solve(1) == 1" + NL
    + "def test_b():" + NL + "    assert solve(2) == 2" + NL,
    {"passed": 0, "total": 2, "timed_out": False, "extracted": True, "mode": "pytest"})

add("运行类", "除零错误：0/0 触发，另一条正常 1/2",
    fence("def solve(a, b):" + NL + "    return a // b" + NL),
    "from solution import solve" + NL
    + "def test_div0():" + NL + "    assert solve(1, 0) == 0" + NL
    + "def test_ok():" + NL + "    assert solve(4, 2) == 2" + NL,
    {"passed": 1, "total": 2, "timed_out": False, "extracted": True, "mode": "pytest"},
    note="1//0 触发 ZeroDivisionError(ERROR)，4//2 正常 => 1/2")

add("运行类", "索引越界：越界触发，另一条正常 1/2",
    fence("def solve():" + NL + "    return [1, 2]" + NL),
    "from solution import solve" + NL
    + "def test_ok():" + NL + "    assert solve()[0] == 1" + NL
    + "def test_oob():" + NL + "    assert solve()[5] == 6" + NL,
    {"passed": 1, "total": 2, "timed_out": False, "extracted": True, "mode": "pytest"})

add("运行类", "递归爆栈（无限递归）：快速 RecursionError 0/1（非超时）",
    fence("def solve(x):" + NL + "    return solve(x)" + NL),
    "from solution import solve" + NL + "def test_a():" + NL + "    assert solve(1) == 1" + NL,
    {"passed": 0, "total": 1, "timed_out": False, "extracted": True, "mode": "pytest"},
    note="递归上限很快触发 RecursionError，不是挂起超时")

add("运行类", "辅助函数抛异常：所有测试 ERROR 0/2",
    fence("def _h(x):" + NL + "    raise ValueError('x')" + NL
          + "def solve(x):" + NL + "    return _h(x)" + NL),
    "from solution import solve" + NL
    + "def test_a():" + NL + "    assert solve(1) == 1" + NL
    + "def test_b():" + NL + "    assert solve(2) == 2" + NL,
    {"passed": 0, "total": 2, "timed_out": False, "extracted": True, "mode": "pytest"})

add("运行类", "返回类型错误（str vs int）：TypeError 0/1",
    fence("def solve(x):" + NL + "    return 'str'" + NL),
    "from solution import solve" + NL + "def test_a():" + NL + "    assert solve(1) == 1" + NL,
    {"passed": 0, "total": 1, "timed_out": False, "extracted": True, "mode": "pytest"})

add("运行类", "测试调用参数个数错（solve 只需 1 个）：TypeError 2/3",
    fence("def solve(x):" + NL + "    return x" + NL),
    "from solution import solve" + NL
    + "def test_ok1():" + NL + "    assert solve(1) == 1" + NL
    + "def test_ok2():" + NL + "    assert solve(2) == 2" + NL
    + "def test_bad():" + NL + "    assert solve(1, 2) == 3" + NL,
    {"passed": 2, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    note="solve(1,2) 多参 => TypeError(ERROR)，另两条正常 => 2/3")


# ===== D. 沙箱类（20）======================================================
# D01-D04 超时杀进程（用「标记文件」可观测地验证进程树被杀死）
def _marker_case(desc, to):
    add("沙箱类", desc, None, None,
        {"passed": 0, "total": 1, "timed_out": True, "extracted": True, "mode": "pytest"},
        kind="marker", timeout_s=to,
        note="标记文件在超时后停止增长 => 进程树被 kill；奖励 ×0.5 => 期望 0.05")


_marker_case("超时杀进程（可观测）：模块级死循环写标记文件，超时后文件停写", 2)
_marker_case("超时杀进程（可观测）：被调函数死循环写标记文件，超时后停写", 2)
_marker_case("超时杀进程（可观测）：1.5s 超时也能杀掉", 1.5)
_marker_case("超时杀进程（可观测）：长死循环 2s 超时后停写", 2)

# D05-D06 简单超时（直接验证 run_in_sandbox 的 timed_out 与非零返回码）
add("沙箱类", "简单超时：score_kernel 跑模块级死循环 => timed_out",
    fence(_loop_mod), "from solution import nope" + NL + "def test_a():" + NL + "    pass" + NL,
    {"passed": 0, "total": 1, "timed_out": True, "extracted": True, "mode": "pytest"},
    timeout_s=2, kind="score_timeout_simple",
    note="run_in_sandbox 返回 timed_out=True 且 returncode!=0")

add("沙箱类", "简单超时：被调函数死循环 => timed_out",
    fence(_loop_fn), "from solution import solve" + NL + "def test_a():" + NL + "    assert solve(1) == 1" + NL,
    {"passed": 0, "total": 1, "timed_out": True, "extracted": True, "mode": "pytest"},
    timeout_s=2, kind="score_timeout_simple")

# D07-D09 多任务互不干扰（并发）
def _iso_subs(n, mixed=False):
    subs = []
    for i in range(n):
        if mixed:
            P = i % 3
            T = 3
        else:
            P = T = 3
        subs.append((normal_response(), normal_tests(P, T), P, T))
    return subs


add("沙箱类", "并发隔离：6 个全通过任务并行，各得自身结果",
    None, None, {}, kind="isolation", subs=_iso_subs(6),
    note="ThreadPool 并发跑，每个任务必须拿到自己的 passed/total，互不串扰")
add("沙箱类", "并发隔离：8 个任务并行（含部分失败），各得自身结果",
    None, None, {}, kind="isolation", subs=_iso_subs(8, mixed=True))
add("沙箱类", "并发隔离：10 个快速任务压力并行，各得自身结果",
    None, None, {}, kind="isolation", subs=_iso_subs(10))

# D10-D13 临时目录清理干净
def _cleanup_batch():
    batch = []
    for P, T in [(3, 3), (2, 4), (5, 5), (0, 2), (4, 6)]:
        batch.append((normal_response(), normal_tests(P, T)))
    # 再混入一个超时任务，验证超时后目录也被清理
    batch.append((fence(_loop_mod),
                  "from solution import nope" + NL + "def test_a():" + NL + "    pass" + NL))
    return batch


add("沙箱类", "临时目录清理：跑 5 个正常任务后无残留沙箱目录",
    None, None, {}, kind="cleanup", batch=_cleanup_batch(), note="前后 glob rlvr_sandbox_* 数量相等")
add("沙箱类", "临时目录清理：含超时任务后也无残留沙箱目录",
    None, None, {}, kind="cleanup", batch=_cleanup_batch(), timeout_s=2)
add("沙箱类", "临时目录清理：语法错误批次后无残留",
    None, None, {}, kind="cleanup",
    batch=[(fence("def solve(x)" + NL + "    return x" + NL),
            "from solution import solve" + NL + "def test_a():" + NL + "    assert solve(1)==1" + NL)])
add("沙箱类", "临时目录清理：混合（正常+语法错误+超时）后无残留",
    None, None, {}, kind="cleanup",
    batch=_cleanup_batch() + [(fence("def solve(x):" + NL + "    return x + 1" + NL),
                               "from solution import solve" + NL + "def test_a():" + NL + "    assert solve(1)==1" + NL)],
    timeout_s=2)

# D14-D15 直接调 run_in_sandbox（验证沙箱本身）
PY = score._resolve_python_exe(None)
add("沙箱类", "沙箱直跑：print 正常返回 0 且捕获输出",
    None, None, {}, kind="sandbox_direct",
    files={"a.py": "print('hello sandbox')" + NL},
    entry=[PY, "a.py"], expect_returncode=0, expect_contains="hello sandbox")
add("沙箱类", "沙箱直跑：简单运算输出 2",
    None, None, {}, kind="sandbox_direct",
    files={"s.py": "x = 1 + 1" + NL + "print(x)" + NL},
    entry=[PY, "s.py"], expect_returncode=0, expect_contains="2")

# D16 文件互不串扰（顺序跑两个不同 solution，不污染）
add("沙箱类", "文件互不串扰：顺序跑两个不同 solve，结果各自独立",
    None, None, {}, kind="seq_isolation",
    seq=[(fence(EX_CODE_STR), EX_TEST, 3, 3),
         (fence(EX_BUG_STR), EX_TEST, 0, 3)],
    note="先后写 solution.py 内容不同，不能互相残留")

# D17 子目录 pkg 也能正确落盘并运行
add("沙箱类", "子目录 pkg/ 落盘：import 子包正常",
    None, None, {}, kind="sandbox_direct",
    files={"pkg/__init__.py": NL,
           "pkg/mod.py": "def solve(x):" + NL + "    return x" + NL,
           "t.py": "from pkg.mod import solve" + NL + "print(solve(7))" + NL},
    entry=[PY, "t.py"], expect_returncode=0, expect_contains="7")

# D18-D19 入参校验（应抛异常，不静默）
add("沙箱类", "入参校验：entry 传字符串应抛 ValueError",
    None, None, {}, kind="sandbox_validate",
    files={"a.py": "print(1)" + NL}, entry="python a.py", expect_raises=ValueError)
add("沙箱类", "入参校验：files 内容非 str 应抛 TypeError",
    None, None, {}, kind="sandbox_validate",
    files={"a.py": 123}, entry=[PY, "a.py"], expect_raises=TypeError)

# D20 增强版在 Windows 上降级但不报错
add("沙箱类", "增强版沙箱：Windows 下降级为基础版仍可正常跑分",
    normal_response(), normal_tests(3, 3),
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    enhanced=True, kind="score_enhanced",
    note="enhanced=True 在 Windows 仅 warning 降级，返回正常结果")


# ===== E. 组装类（20）======================================================
# E01-E03 极端超时
add("组装类", "极端边界：超时 1s 死循环 => timed_out",
    fence(_loop_mod), "from solution import nope" + NL + "def test_a():" + NL + "    pass" + NL,
    {"passed": 0, "total": 1, "timed_out": True, "extracted": True, "mode": "pytest"},
    timeout_s=1, kind="score_timeout_simple")
add("组装类", "极端边界：超时 1s 死循环（极短超时）=> timed_out",
    fence(_loop_mod), "from solution import nope" + NL + "def test_a():" + NL + "    pass" + NL,
    {"passed": 0, "total": 1, "timed_out": True, "extracted": True, "mode": "pytest"},
    timeout_s=1, kind="score_timeout_simple",
    note="⚠ timeout_s 须为整数（int(0.5)=0 会被沙箱拒绝），故用 1s")
add("组装类", "极端边界：超时 2s 被调函数死循环 => timed_out",
    fence(_loop_fn), "from solution import solve" + NL + "def test_a():" + NL + "    assert solve(1) == 1" + NL,
    {"passed": 0, "total": 1, "timed_out": True, "extracted": True, "mode": "pytest"},
    timeout_s=2, kind="score_timeout_simple")

# E04-E06 0 道测试用例
add("组装类", "0 测试用例：test 为空串 => mode=no_tests，(0,0)，奖励 0.1",
    normal_response(), "",
    {"passed": 0, "total": 0, "timed_out": False, "extracted": True, "mode": "no_tests"},
    note="⚠ 当前实现：no_tests 的 extracted=True => 奖励 0.1（非 0），请确认是否符合预期")
add("组装类", "0 测试用例：test 仅 print 无 def => mode=pytest，(0,0)，奖励 0.1",
    normal_response(), "print('hi')" + NL,
    {"passed": 0, "total": 0, "timed_out": False, "extracted": True, "mode": "pytest"},
    note="⚠ 无 ::test 行 => 保守 (0,0)，extracted=True => 奖励 0.1")
add("组装类", "0 测试用例：test 仅赋值无 def => mode=pytest，(0,0)，奖励 0.1",
    normal_response(), "x = 1" + NL,
    {"passed": 0, "total": 0, "timed_out": False, "extracted": True, "mode": "pytest"})

# E07-E08 response/test 非字符串
add("组装类", "极端边界：response=None（非 str）=> mode=error，(0,0)，奖励 0",
    None, normal_tests(1, 1),
    {"passed": 0, "total": 0, "timed_out": False, "extracted": False, "mode": "error"},
    note="⚠ 实现中 response 或 test 任一非 str => (0,0)（不看 test 的 def 数）")
add("组装类", "极端边界：test=None（非 str）=> mode=error，(0,0)，奖励 0",
    normal_response(), None,
    {"passed": 0, "total": 0, "timed_out": False, "extracted": False, "mode": "error"})

# E09-E10 空/空白 response（提取为 None => no_code）
add("组装类", "极端边界：response=''（空串）=> 无代码 (0, defcount)，奖励 0",
    "", normal_tests(1, 1),
    {"passed": 0, "total": 1, "timed_out": False, "extracted": False, "mode": "no_code"},
    note="空串 => extract 返回 None => no_code")
add("组装类", "极端边界：response 仅空白 => 无代码 (0, defcount)，奖励 0",
    "   " + NL + "  ", normal_tests(2, 2),
    {"passed": 0, "total": 2, "timed_out": False, "extracted": False, "mode": "no_code"})

# E11 response=None 且设了 timeout（超时无关，仍 error）
add("组装类", "极端边界：response=None 且带 timeout => 仍 error (0,0)，奖励 0",
    None, normal_tests(1, 1),
    {"passed": 0, "total": 0, "timed_out": False, "extracted": False, "mode": "error"},
    timeout_s=5)

# E12 死循环 + 空 test（test 空优先短路，不进沙箱）
add("组装类", "极端边界：死循环 + 空 test => test 空短路为 no_tests（不超时）",
    fence(_loop_mod), "",
    {"passed": 0, "total": 0, "timed_out": False, "extracted": True, "mode": "no_tests"},
    timeout_s=5,
    note="⚠ test 为空在提取之前就返回 no_tests，所以即便代码会死循环也不超时")

# E13 参数化测试名（验证 parser 兼容 test_x[a-b]）
_para = ("import pytest" + NL + "from solution import solve" + NL + NL
         + "@pytest.mark.parametrize('a,b,exp', [(1,2,3),(4,5,9),(0,0,0)])" + NL
         + "def test_add(a, b, exp):" + NL + "    assert solve(a, b) == exp" + NL)
add("组装类", "极端边界：parametrize 展开名 test_add[1-2-3] 正确计数 3/3",
    fence("def solve(a, b):" + NL + "    return a + b" + NL), _para,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    note="验证 _parse_pytest 兼容带方括号的用例名")

# E14 SKIPPED（假设 E：计入 total 不算 passed）
_skip = ("import pytest" + NL + "from solution import solve" + NL + NL
         + "@pytest.mark.skip(reason='skip')" + NL + "def test_a():" + NL + "    assert solve(1) == 1" + NL
         + "@pytest.mark.skip(reason='s2')" + NL + "def test_b():" + NL + "    assert solve(2) == 2" + NL)
add("组装类", "极端边界：@pytest.mark.skip 全跳过 => 0/2，奖励 0.1",
    fence(EX_CODE_STR), _skip,
    {"passed": 0, "total": 2, "timed_out": False, "extracted": True, "mode": "pytest"},
    note="⚠ 假设 E：SKIPPED 计入 total 但不算 passed => 奖励 0.1")

# E15 XFAIL（假设 E：XFAIL 不算 passed）
_xfail = ("import pytest" + NL + "from solution import solve" + NL + NL
          + "@pytest.mark.xfail" + NL + "def test_a():" + NL + "    assert solve(1) == 2" + NL)
add("组装类", "极端边界：@pytest.mark.xfail 失败满足 => XFAIL 0/1，奖励 0.1",
    fence(EX_CODE_STR), _xfail,
    {"passed": 0, "total": 1, "timed_out": False, "extracted": True, "mode": "pytest"},
    note="⚠ 假设 E：XFAIL 计入 total 不算 passed => 奖励 0.1")

# E16 超长 response（>MAX_PROMPT_LEN）但代码正常
_long_prose = ("这是一段很长的题目背景描述。" * 200) + NL + NL
add("组装类", "极端边界：超长 response（>2000 字符）但代码正常 => 仍提取并 3/3",
    _long_prose + fence(EX_CODE_STR), EX_TEST,
    {"passed": 3, "total": 3, "timed_out": False, "extracted": True, "mode": "pytest"},
    note="MAX_PROMPT_LEN 只用于筛 1 过滤，判分器照常提取")

# E17 无代码但测试本身能通过（验证提取闸门：无代码 => 0）
add("组装类", "极端边界：无代码块但测试自包含可通过 => 仍记 no_code 奖励 0",
    "我没有写代码，只有文字。",
    "def test_a():" + NL + "    assert 1 == 1" + NL,
    {"passed": 0, "total": 1, "timed_out": False, "extracted": False, "mode": "no_code"},
    note="⚠ 提取闸门优先：无代码即 no_code，即使测试自包含通过也得 0")

# E18 多测试全过（class 风格 solution）
_class_code = "class Solver:" + NL + "    def solve(self, x):" + NL + "        return x" + NL
add("组装类", "极端边界：多测试全过（class 风格）=> 1.0",
    think("用类实现。") + fence(_class_code),
    "from solution import Solver" + NL
    + NL.join("def test_%d():" % i + NL + "    assert Solver().solve(%d) == %d" % (i, i)
              for i in range(4)) + NL,
    {"passed": 4, "total": 4, "timed_out": False, "extracted": True, "mode": "pytest"})

# E19 代码返回 None 导致全部失败
add("组装类", "极端边界：solve 返回 None => 全部断言失败 0/2",
    fence("def solve(x):" + NL + "    return None" + NL),
    "from solution import solve" + NL
    + "def test_a():" + NL + "    assert solve(1) == 1" + NL
    + "def test_b():" + NL + "    assert solve(2) == 2" + NL,
    {"passed": 0, "total": 2, "timed_out": False, "extracted": True, "mode": "pytest"})

# E20 死循环 + 0 def 测试（测试须 import solution 才会触发模块级死循环；超时后 total=0）
add("组装类", "极端边界：死循环 + 测试仅 import 无 def => 超时 (0,0)，奖励 0.05",
    fence(_loop_mod), "from solution import nope" + NL,
    {"passed": 0, "total": 0, "timed_out": True, "extracted": True, "mode": "pytest"},
    timeout_s=2,
    note="对比 E12：本例测试 import solution => 模块级死循环被触发并超时；无 def => 保守 (0,0)，×0.5 => 0.05")


# 守卫：必须恰好 100 条
assert len(CASES) == 100, "case 数量应为 100，实际 %d" % len(CASES)


# ---------------------------------------------------------------------------
# run_case：按 kind 分支执行一条 case，并做断言（失败抛 AssertionError）
# ---------------------------------------------------------------------------
def _check_score_fields(passed, total, detail, expect, response=None, extract_expect=None):
    """校验 score 类 case 的字段，并返回 (期望score, 实际score, 摘要字段)。"""
    mode = detail.get("mode")
    extracted = mode not in ("no_code", "error")
    timed_out = bool(detail.get("timed_out", False))
    actual_score = score.score(passed, total, extracted=extracted, timed_out=timed_out)

    if extract_expect is not None:
        got = extract.extract_python(response)
        if extract_expect is None:
            assert got is None, "extract 期望 None，实际 %r" % got
        else:
            assert got == extract_expect, "extract 期望 %r，实际 %r" % (extract_expect, got)

    assert passed == expect["passed"], "passed 期望 %s 实际 %s" % (expect["passed"], passed)
    assert total == expect["total"], "total 期望 %s 实际 %s" % (expect["total"], total)
    assert timed_out == expect["timed_out"], "timed_out 期望 %s 实际 %s" % (expect["timed_out"], timed_out)
    assert extracted == expect["extracted"], "extracted 期望 %s 实际 %s" % (expect["extracted"], extracted)
    if "mode" in expect:
        assert mode == expect["mode"], "mode 期望 %s 实际 %s" % (expect["mode"], mode)

    exp_score = formula(expect["passed"], expect["total"], expect["extracted"], expect["timed_out"])
    assert abs(actual_score - exp_score) < 1e-9, "score 期望 %.4f 实际 %.4f" % (exp_score, actual_score)
    return exp_score, actual_score, mode, extracted, timed_out


def run_case(case: dict) -> dict:
    """执行一条 case，返回供报告使用的摘要 dict；任何不符抛 AssertionError。"""
    kind = case.get("kind", "score")
    cid, cat, desc = case["id"], case["cat"], case["desc"]

    if kind in ("score", "score_timeout_simple", "score_enhanced"):
        passed, total, detail = score.score_kernel(
            case["response"], case["test"],
            timeout_s=case.get("timeout_s"), enhanced=case.get("enhanced", False))
        exp_score, actual_score, mode, extracted, timed_out = _check_score_fields(
            passed, total, detail, case["expect"], case["response"], case.get("expect_extract"))
        return {
            "id": cid, "cat": cat, "desc": desc,
            "expect_str": _fmt_expect(case["expect"], exp_score),
            "actual_str": "passed=%s total=%s timed_out=%s extracted=%s mode=%s score=%.4f"
                          % (passed, total, timed_out, extracted, mode, actual_score),
            "note": case.get("note", ""),
        }

    if kind == "marker":
        to = case.get("timeout_s", 2)
        marker = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
        mpath = marker.replace("\\", "/")
        code = ("import time" + NL
                + "f = open('%s', 'a')" % mpath + NL
                + "while True:" + NL + "    f.write('x'); f.flush(); time.sleep(0.05)" + NL)
        resp = fence(code)
        test = "from solution import nope" + NL + "def test_a():" + NL + "    pass" + NL
        passed, total, detail = score.score_kernel(resp, test, timeout_s=to)
        # 可观测验证：超时后标记文件应停止增长（进程树被 kill）
        size1 = os.path.getsize(marker)
        time.sleep(1.2)
        size2 = os.path.getsize(marker)
        try:
            os.remove(marker)
        except Exception:
            pass
        assert detail.get("timed_out") is True, "marker 期望超时，实际 timed_out=%s" % detail.get("timed_out")
        assert size2 == size1, "超时后进程仍在写文件(size %d->%d)，进程树未被杀掉" % (size1, size2)
        exp_score, actual_score, mode, extracted, timed_out = _check_score_fields(
            passed, total, detail, case["expect"], resp)
        return {
            "id": cid, "cat": cat, "desc": desc,
            "expect_str": _fmt_expect(case["expect"], exp_score) + " | 标记文件停写=True",
            "actual_str": "timed_out=%s 标记停写=%s mode=%s score=%.4f"
                          % (timed_out, size2 == size1, mode, actual_score),
            "note": case.get("note", ""),
        }

    if kind == "isolation":
        subs = case["subs"]
        results = [None] * len(subs)

        def _worker(idx):
            resp, test, P, T = subs[idx]
            passed, total, detail = score.score_kernel(resp, test)
            results[idx] = (passed, total, detail)

        with ThreadPoolExecutor(max_workers=len(subs)) as ex:
            list(ex.map(_worker, range(len(subs))))
        lines = []
        ok = True
        for idx, (resp, test, P, T) in enumerate(subs):
            passed, total, detail = results[idx]
            if passed != P or total != T:
                ok = False
                lines.append("  子任务%d 期望(%d,%d) 实际(%d,%d) [FAIL]" % (idx, P, T, passed, total))
            else:
                lines.append("  子任务%d (%d/%d) OK" % (idx, passed, total))
        assert ok, "并发隔离失败:\n" + NL.join(lines)
        return {
            "id": cid, "cat": cat, "desc": desc,
            "expect_str": "%d 个并发任务各得自身 passed/total" % len(subs),
            "actual_str": NL.join(lines),
            "note": case.get("note", ""),
        }

    if kind == "cleanup":
        batch = case["batch"]
        to = case.get("timeout_s")
        prefix = os.path.join(tempfile.gettempdir(), config.SANDBOX_WORKDIR_PREFIX + "*")
        before = len(glob.glob(prefix))
        for resp, test in batch:
            score.score_kernel(resp, test, timeout_s=to)
        after = len(glob.glob(prefix))
        assert after == before, "临时目录残留：前后数量 %d -> %d" % (before, after)
        return {
            "id": cid, "cat": cat, "desc": desc,
            "expect_str": "沙箱目录数量不变（前后均为 %d）" % before,
            "actual_str": "before=%d after=%d（无残留）" % (before, after),
            "note": case.get("note", ""),
        }

    if kind == "sandbox_direct":
        res = sandbox.run_in_sandbox(case["files"], case["entry"], timeout_s=case.get("timeout_s", 10))
        assert res.returncode == case["expect_returncode"], \
            "returncode 期望 %s 实际 %s" % (case["expect_returncode"], res.returncode)
        assert case["expect_contains"] in res.stdout, \
            "输出应包含 %r，实际 stdout=%r" % (case["expect_contains"], res.stdout[:200])
        return {
            "id": cid, "cat": cat, "desc": desc,
            "expect_str": "returncode=%s 且输出含 %r" % (case["expect_returncode"], case["expect_contains"]),
            "actual_str": "returncode=%s stdout含%r=True timed_out=%s"
                          % (res.returncode, case["expect_contains"], res.timed_out),
            "note": case.get("note", ""),
        }

    if kind == "seq_isolation":
        lines = []
        ok = True
        for idx, (resp, test, P, T) in enumerate(case["seq"]):
            passed, total, detail = score.score_kernel(resp, test)
            if passed != P or total != T:
                ok = False
                lines.append("  顺序%d 期望(%d,%d) 实际(%d,%d) [FAIL]" % (idx, P, T, passed, total))
            else:
                lines.append("  顺序%d (%d/%d) OK" % (idx, passed, total))
        assert ok, "顺序隔离失败:\n" + NL.join(lines)
        return {
            "id": cid, "cat": cat, "desc": desc,
            "expect_str": "顺序任务各得自身 passed/total",
            "actual_str": NL.join(lines),
            "note": case.get("note", ""),
        }

    if kind == "sandbox_validate":
        raised = False
        try:
            sandbox.run_in_sandbox(case["files"], case["entry"], timeout_s=case.get("timeout_s", 10))
        except case["expect_raises"]:
            raised = True
        assert raised, "未如期抛出 %s" % case["expect_raises"].__name__
        return {
            "id": cid, "cat": cat, "desc": desc,
            "expect_str": "应抛出 %s" % case["expect_raises"].__name__,
            "actual_str": "已抛出 %s" % case["expect_raises"].__name__,
            "note": case.get("note", ""),
        }

    raise AssertionError("未知 kind: %s" % kind)


def _fmt_expect(expect: dict, exp_score: float) -> str:
    return ("passed=%s total=%s timed_out=%s extracted=%s mode=%s => 期望score=%.4f"
            % (expect["passed"], expect["total"], expect["timed_out"],
               expect["extracted"], expect.get("mode", "?"), exp_score))


# ---------------------------------------------------------------------------
# pytest 入口：100 个节点
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"] + "_" + c["cat"])
def test_boundary_cases(case):
    run_case(case)


# ---------------------------------------------------------------------------
# 脚本入口：打印可读的「期望 vs 实际」对照表，方便人工逐条核对
# ---------------------------------------------------------------------------
def _main() -> int:
    print("=" * 78)
    print("100 条边界单测 · 人工核对报告（任务书 §6）")
    print("=" * 78)
    print("公式: score = (passed/total if total>0 else 0) + (0.1 if extracted else 0)")
    print("      -> 夹到 [0,1] -> 若 timed_out 再 ×0.5")
    print("      extracted = (mode not in ('no_code','error'))")
    print("=" * 78)
    passed_cnt = 0
    failed = []
    cur_cat = None
    for case in CASES:
        if case["cat"] != cur_cat:
            cur_cat = case["cat"]
            print("\n----- %s -----" % cur_cat)
        try:
            summary = run_case(case)
            passed_cnt += 1
            status = "PASS"
        except AssertionError as e:
            _exp = case["expect"]
            _es = formula(_exp["passed"], _exp["total"], _exp["extracted"], _exp["timed_out"])
            summary = {
                "id": case["id"], "desc": case["desc"],
                "expect_str": _fmt_expect(_exp, _es),
                "actual_str": "FAIL: " + str(e),
                "note": case.get("note", ""),
            }
            status = "FAIL"
            failed.append(case["id"])
        print("[%s] %s | %s" % (summary["id"], summary["desc"], status))
        print("    期望: " + summary["expect_str"])
        print("    实际: " + summary["actual_str"])
        if summary.get("note"):
            print("    注  : " + summary["note"])
    print("\n" + "=" * 78)
    print("合计 %d 条：通过 %d / 失败 %d" % (len(CASES), passed_cnt, len(CASES) - passed_cnt))
    if failed:
        print("失败 case: " + ", ".join(failed))
    print("=" * 78)
    print("┌──────────────────────────────────────────────────────────────────┐")
    print("│ 已复核：（待人工签名：姓名 / 日期）                                  │")
    print("└──────────────────────────────────────────────────────────────────┘")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_main())
