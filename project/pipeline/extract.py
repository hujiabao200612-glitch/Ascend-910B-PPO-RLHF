# -*- coding: utf-8 -*-
"""extract.py —— 从模型输出中提取 Python 代码（任务书 §3 + §5.3）。

规则（与冻结的 PROMPT_TEMPLATE 是双胞胎：改模板措辞 = 改这里，同定同冻）
    1) 从模型输出中提取 **第一个** ```python ... ``` 围栏代码块；
    2) 无围栏代码块 → 返回 None，判分器据此记为 passed=0, total=0（§3）。

§5.3 明确要求的容忍度
    * 语言标签大小写不敏感：```Python / ```PYTHON 都算；
    * 首行语言标签缺失也接受：裸 ``` 围栏同样视为代码块；
    * 多个代码块 → 取第一个。

代码 5.3 之外、任务书未明确、按合理工程实践处理的假设（评审时请重点看这里）
    A. 「第一个」的判定：按文档顺序扫描围栏块，取第一个「标签为空 或 标签以
       python 开头」的块。因此先出现的 ```text / ```bash 说明块会被跳过，
       不会把解释文字当成代码；同时也容忍 ```python3 / ```python3.11。
    B. 输入边界：response 不是 str（None / bytes / 数字…）或全为空白 → 返回 None，
       不抛异常 —— 判分器在 RL 打分时会拿到各种脏输入，这里必须永不崩。
    C. 只有开围栏、没有闭围栏（模型被 max_tokens 截断）→ 视为无代码。理由：
       不猜代码边界，避免把半截输出当代码送进沙箱，产生误导性的判分。
    D. 闭围栏必须独占一行（允许缩进与行尾空格）。因此代码里行内出现的 ```
       字面量（如 s = "```"）不会被误判成块结束 —— 这是 §6.1「块内含 ``` 字面量」
       一类的关键；反过来，把 ``` 单独写在一行（例如在三引号字符串里）会被误认为
       闭围栏，属已知限制。
    E. 开围栏行尾不允许跟解释文字（如 "```python 以下是实现"）→ 判为无代码。
       这是严格照 §5.3 的容忍范围（仅「大小写 + 标签缺失」）执行的结果；
       是否放宽属于接口决策，改动前请与判分器一起评估。
    F. 取出的代码原样返回，只去掉首尾空行并统一换行符为 \n，**不做自动去缩进**。
       理由：判分器是全项目的尺子，尺子的刻度要客观 —— 模型多缩进一层导致
       IndentationError，应当由运行结果客观反映，而不是由提取器悄悄修好。

任务书要求「15 条单元级测试覆盖以上变体」，测试用例由后续批次的
pipeline/tests/ 承担，本文件只交付提取逻辑本身。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Iterator, List, Optional, Tuple

__all__ = ["extract_python"]

# config.py 是任务书 §4 规定的常量集中地：存在则优先使用，缺失则回落默认值
# （与 sandbox.py 保持同一套「先 import config，失败回落」的写法，便于同事评审）
try:  # pragma: no cover - 仅在 config.py 缺失/损坏时走到
    import config as _config  # type: ignore
except Exception:
    _config = None


def _config_get(names, default):
    """按候选名在 config 里取常量，取不到返回 default。"""
    if _config is not None:
        for name in names:
            value = getattr(_config, name, None)
            if value is not None:
                return value
    return default


# CLI 默认从对象哪个字段取模型输出（§任务书 §5.3 CLI / 任务书 §6.6）
EXTRACT_DEFAULT_FIELD = str(_config_get(("EXTRACT_DEFAULT_FIELD",), "response"))

# 提取规则的正则（任务书 §5.3：正则匹配 ```python\n ... ```）。
# 逐段解释：
#   ^[ \t]*```            开围栏必须在行首（允许缩进）——行内出现的 ``` 不算，见假设 D
#   [ \t]*(?P<tag>[^\s`]*) 语言标签：可空（假设 A 的「标签缺失也接受」），不含空白与反引号
#   [ \t]*\r?\n           标签后必须换行（容忍 Windows 的 \r\n）
#   (?P<code>.*?)         代码主体，惰性匹配
#   ^[ \t]*```[ \t]*$     闭围栏必须独占一行（DOTALL 下 $ 仍按 MULTILINE 匹配行尾）
_FENCE_RE = re.compile(
    r"^[ \t]*```[ \t]*(?P<tag>[^\s`]*)"
    r"[ \t]*\r?\n"
    r"(?P<code>.*?)"
    r"^[ \t]*```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


def _iter_fenced_blocks(response: str) -> Iterator[Tuple[str, str]]:
    """按文档顺序产出所有「完整」围栏块 (语言标签, 代码主体)。

    为什么单独抽一个生成器：让 extract_python 的「取第一个可用块」逻辑保持
    一眼可读，同时把正则细节（含被截断块不匹配的行为）收在一处便于评审。
    """
    for match in _FENCE_RE.finditer(response):
        yield match.group("tag"), match.group("code")


def _is_python_tag(tag: str) -> bool:
    """判断语言标签是否可接受：空标签或形如 python / python3 / python3.11。"""
    return tag == "" or tag.lower().startswith("python")


def extract_python(response: str) -> Optional[str]:
    """从模型输出中提取第一个可用的 Python 代码块；失败返回 None（§3 / §5.3）。

    参数
        response: 模型生成的完整文本（可能很长、可能夹杂解释与多个代码块）。

    返回
        提取到的代码字符串（已去掉首尾空行、换行统一为 \\n）；
        无法提取时返回 None —— 判分器把它当「无代码」处理（passed=0, total=0）。

    本函数不抛异常（假设 B）：脏输入一律以 None 表示「没有拿到代码」。
    """
    # 边界：非字符串 / 空串 / 纯空白（假设 B）
    if not isinstance(response, str):
        return None
    if not response.strip():
        return None

    try:
        for tag, code in _iter_fenced_blocks(response):
            if _is_python_tag(tag.strip()):
                # 统一换行符（Windows 侧数据常见 \r\n），再去掉首尾空行；
                # 保留行内缩进，见假设 F
                return code.replace("\r\n", "\n").strip("\n")
    except Exception:
        # 正则理论上不会抛异常，但判分器绝不能因为提取器崩掉而丢样本
        return None
    return None


# ---------------------------------------------------------------------------
# 命令行入口（任务书 §6.6：所有脚本支持 python xxx.py --input 路径 --output 路径）
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="批量提取模型输出中的 Python 代码块（任务书 §6.6）。",
    )
    parser.add_argument("--input", required=True, help="输入 jsonl 路径（如判分器输入/模型输出样本）")
    parser.add_argument("--output", required=True, help="输出 jsonl 路径")
    parser.add_argument(
        "--field",
        default=EXTRACT_DEFAULT_FIELD,
        help="从输入对象的哪个字段取模型输出（默认 %s）" % EXTRACT_DEFAULT_FIELD,
    )
    return parser


def _main(argv: Optional[List[str]] = None) -> int:
    """CLI：逐行读 jsonl → 提取 → 在原对象上补 extracted_code / has_code 字段。

    输入 jsonl 每行一个 JSON 对象，模型输出默认放在 "response" 字段（可用 --field 改）。
    输出 jsonl 每行 = 原对象 + 两个字段：
        "extracted_code": str | null
        "has_code": bool
    行内容不是合法 JSON 时不丢弃：写出 {"input_index": n, "has_code": false,
    "extracted_code": null, "error": "..."} 记账（假设：任务书只要求 --input/--output，
    未规定 CLI 数据格式，这里选择与数据线一致的 jsonl 惯例）。

    结尾按 §7.5 打印统计：读入 / 有代码 / 无代码 / 坏行。
    """
    args = _build_argparser().parse_args(argv)

    total = with_code = without_code = bad_lines = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8", newline="\n"
    ) as fout:
        for index, raw in enumerate(fin):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                record = json.loads(raw)
                if not isinstance(record, dict):
                    raise ValueError("每行必须是 JSON 对象")
            except Exception as exc:
                bad_lines += 1
                out = {
                    "input_index": index,
                    "extracted_code": None,
                    "has_code": False,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                }
            else:
                code = extract_python(record.get(args.field))
                out = dict(record)  # 不破坏原始字段，便于上下游对账
                out["extracted_code"] = code
                out["has_code"] = code is not None
                if code is None:
                    without_code += 1
                else:
                    with_code += 1

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(
        "[extract] 读入 %d 行 / 有代码 %d / 无代码 %d / 坏行 %d"
        % (total, with_code, without_code, bad_lines)
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
