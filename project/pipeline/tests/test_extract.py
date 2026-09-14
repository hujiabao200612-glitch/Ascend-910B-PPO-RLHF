# -*- coding: utf-8 -*-
"""tests/test_extract.py —— extract.extract_python 的 15 条单元测试（任务书 §5.3）。

覆盖本次要求必须验证的四类情况：
    情况 1：正常带标记代码
    情况 2：两块代码块（取第一个）
    情况 3：完全没有代码块
    情况 4：标签大写 ` ```Python `

运行：
    venv_rlvr/Scripts/python.exe -m pytest pipeline/tests/test_extract.py -v
（在项目根目录 D:/学习资料/code_rlvr 下执行；Windows 路径用正斜杠避免转义告警）

注意（任务书 §6.2）：AI 生成的期望提取结果必须由人工逐条核对签名。
下面这行是留给评审人签字的占位，确认无误后填上 姓名 / 日期。
已复核：（待人工签名：姓名 / 日期）
"""

import os
import sys

# 让测试无论从哪个 cwd 启动都能 import 到 pipeline/ 下的 extract 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extract  # noqa: E402


# ===================== 情况 1：正常带标记代码 =====================

def test_1_basic_marked_code():
    """最基本：一个标准 ```python 块，应原样取出代码主体。"""
    assert extract.extract_python("```python\nprint(1)\n```") == "print(1)"


def test_2_with_comment():
    """代码块内带注释，注释应保留在结果里。"""
    resp = "前文\n```python\n# 注释\nx = 1\nprint(x)\n```\n后文"
    assert extract.extract_python(resp) == "# 注释\nx = 1\nprint(x)"


def test_3_with_import():
    """代码块内含 import，应正常提取（不裁剪、不改写）。"""
    resp = "```python\nimport math\nprint(math.pi)\n```"
    assert extract.extract_python(resp) == "import math\nprint(math.pi)"


def test_4_with_helper_function():
    """代码块内含辅助函数定义与调用，整体取出。"""
    resp = "```python\ndef add(a, b):\n    return a + b\nprint(add(1, 2))\n```"
    assert extract.extract_python(resp) == "def add(a, b):\n    return a + b\nprint(add(1, 2))"


def test_5_prose_around_block():
    """代码块前后都有解释文字，只取中间代码块、不受前后文影响。"""
    resp = "这是题解：\n```python\nprint('hi')\n```\n以上。\n"
    assert extract.extract_python(resp) == "print('hi')"


# ===================== 情况 2：两块代码块 =====================

def test_6_two_python_blocks_takes_first():
    """两个 ```python 块，应取第一个。"""
    resp = "```python\nA=1\n```\n中间说明\n```python\nB=2\n```"
    assert extract.extract_python(resp) == "A=1"


def test_7_python_then_bare_takes_first():
    """先出现带 python 标签块、后出现裸块，取第一个（python 块的 A=1）。"""
    resp = "```python\nA=1\n```\n```\nB=2\n```"
    assert extract.extract_python(resp) == "A=1"


def test_8_bare_then_python_takes_bare_first():
    """先出现裸块、后出现 python 块：裸块也接受（标签缺失容忍），取第一个裸块的 A=1。"""
    resp = "```\nA=1\n```\n```python\nB=2\n```"
    assert extract.extract_python(resp) == "A=1"


def test_9_three_blocks_takes_first():
    """三段代码链，始终取第一个可用块。"""
    resp = "```python\nA=1\n```\n```python\nB=2\n```\n```python\nC=3\n```"
    assert extract.extract_python(resp) == "A=1"


# ===================== 情况 3：完全没有代码块 =====================

def test_10_pure_prose_no_code():
    """纯文本、无任何围栏，应返回 None（视为无代码）。"""
    assert extract.extract_python("please solve this problem with care") is None


def test_11_inline_ticks_in_prose():
    """行内出现 ``` 但不是独立围栏行，不应被误判为代码块。"""
    assert extract.extract_python("use ``` backticks like this in prose") is None


def test_12_none_input():
    """输入为 None（边界：脏数据），必须返回 None 而不是抛异常。"""
    assert extract.extract_python(None) is None


def test_13_whitespace_input():
    """输入只有空白字符，应返回 None。"""
    assert extract.extract_python("   \n\t  \n") is None


# ===================== 情况 4：标签大写 ```Python =====================

def test_14_uppercase_python_tag():
    """语言标签首字母大写 ```Python，大小写不敏感，应正常提取。"""
    assert extract.extract_python("```Python\nx = 1\n```") == "x = 1"


def test_15_all_caps_python_tag():
    """语言标签全大写 ```PYTHON，同样应正常提取。"""
    assert extract.extract_python("```PYTHON\nx = 1\n```") == "x = 1"
