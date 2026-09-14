# -*- coding: utf-8 -*-
"""f1_template.py —— 筛 1：套模板 + 字段瘦身（任务书 §5.1 + §4 交付物清单）。

职责
    把原料 ``kodcode_candidates.jsonl``（15,000 条候选编程题）加工成
    ``step1_templated.jsonl``：
      1. 用冻结的 PROMPT_TEMPLATE 把每条的 ``question`` 填成统一 ``prompt``；
      2. 只保留下游要用的少量字段（字段瘦身），其余全部丢弃，减小数据体积、
         避免把 source 里的脏字段（gpt_pass_sequence / trials / metadata ...）
         泄漏进训练池。

为什么单独成脚本
    筛 1 是「纯格式转换、零淘汰」的一步（§5.1：「筛 1 不淘汰」），职责极单纯，
    单独文件方便同事评审、也方便在 f2_verify / f3_dedup 之前先单独验证模板套得对。

为什么保留 question 字段（与 §5.1 字面清单的小偏差，需评审确认）
    §5.1 的「保留字段」清单写的是 question_id/style/subset/test/solution/
    gpt_difficulty/benchmark_similarity，**未列 question**；但 §5.6 筛 3 明确要
    对 ``question`` 做 md5 去重，f3 需要原始 question 才能工作。因此本脚本在
    瘦身集里**额外保留 question**（与原清单其余字段一并保留），否则 f3 会断链。
    prompt 由 question 派生，二者本就同源，保留 question 不引入额外风险。
    若评审决定「f3 改为对 prompt 去重、删掉 question」，把 KEEP_FIELDS 里的
    "question" 一行删掉即可，无需改其他逻辑。

任务书未明确、按合理工程实践处理的假设（便于同事评审时逐条核对）
    A. 字段取值原样搬运，不重写、不清洗：test / solution / question 等代码型字段
       的「能不能跑」由筛 2（f2_verify）负责，筛 1 只做格式与瘦身，不做语义判断，
       保证职责单一、可幂等重跑。
    B. question 缺失 / 非字符串 → 不崩：按空串填模板，并在统计里记 missing_question
       条数（脏输入在 15k 量级数据里难免，筛 1 必须永不崩，符合全项目「永不抛异常」
       的工程文化，见 sandbox / extract 的同款处理）。
    C. 输出逐行 = 瘦身后的固定字段 + prompt，顺序稳定，便于 f2 / f3 与人工对账。
    D. 流式逐行读写：原料 jsonl 约 264MB，绝不一口气 load 进内存；用标准库 json 逐行
       解析即可，不依赖 pandas / pyarrow（§7.1 白名单允许但非必须，逐行更省内存）。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

__all__ = ["template_one", "run_filter1"]

# ---------------------------------------------------------------------------
# 常量：优先读 config.py，读不到就用本文件下方的默认值（与 sandbox/extract/score 同一套写法）
# ---------------------------------------------------------------------------

try:  # config.py 是任务书 §4 规定的常量集中地
    import config as _config  # type: ignore
except Exception:  # pragma: no cover - 仅在 config.py 缺失/损坏时走到
    _config = None


def _config_get(names: List[str], default):
    """按候选名在 config 里取常量，取不到返回 default。

    为什么：六个脚本都从 config 取统一阈值/模板，config 缺失时不能让脚本直接崩，
    给一个安全的兜底默认值，保证本地与服务器两侧都能跑（§7.2）。
    """
    if _config is not None:
        for name in names:
            value = getattr(_config, name, None)
            if value is not None:
                return value
    return default


# 冻结提示词模板（§3）：改动会影响判分器提取规则，禁止改一个字符。
PROMPT_TEMPLATE: str = str(_config_get(("PROMPT_TEMPLATE",), ""))
# 默认 I/O 路径（§4 / §7.3 集中管理），命令行 --input/--output 可覆盖。
DEFAULT_INPUT: str = str(_config_get(("CANDIDATES_JSONL",), "data/kodcode_candidates.jsonl"))
DEFAULT_OUTPUT: str = str(_config_get(("STEP1_TEMPLATED",), "data/step1_templated.jsonl"))

# 瘦身后保留的字段白名单（见文件头「为什么保留 question」说明，question 为必要补充）。
# 顺序即输出 jsonl 的字段顺序，稳定便于下游对账。
KEEP_FIELDS: List[str] = [
    "question_id",
    "question",
    "style",
    "subset",
    "test",
    "solution",
    "gpt_difficulty",
    "benchmark_similarity",
]


def template_one(record: Dict) -> str:
    """把单条候选样本填成统一 prompt（§5.1）。

    为什么单独成函数：① 模板填充逻辑是唯一有「规则」的地方，单测可直接喂一条假
    样本进来验证 prompt 长什么样，不必读 264MB 原料；② 与判分器的 extract 规则
    是「双胞胎」，提取要的 ```python 围栏由这里下发、由那里回收，集中在一处好对账。

    行为约定（与文件头假设 B 一致）：
        - record 必须是 dict，否则按「无 question」处理（空串填模板）；
        - question 缺失 / None / 非字符串 → 填空串，不抛异常；
        - 用 str.format(question=...) 传参，question 正文里即便含 { } 也不会被二次解析。
    """
    if not isinstance(record, dict):
        return PROMPT_TEMPLATE.format(question="")
    question = record.get("question")
    if question is None:
        question = ""
    elif not isinstance(question, str):
        question = str(question)
    return PROMPT_TEMPLATE.format(question=question)


def run_filter1(input_path: str, output_path: str) -> Dict[str, int]:
    """执行筛 1 全流程：逐行读原料 → 套模板 + 瘦身 → 写出。

    返回统计字典（读入 / 写出 / 缺 question 条数），供 _main 打印（§7.5 统计要求）。

    为什么返回统计而不是就地 print：把「业务逻辑」与「CLI 打印」解耦，单测可以调用
    本函数拿到计数做断言，不必真的解析 stdout；与 extract / score 的 _main 拆分一致。
    """
    stats = {"read": 0, "written": 0, "missing_question": 0, "bad_lines": 0}

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8", newline="\n"
    ) as fout:
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
                # 坏行不丢弃：写出带 input_index 的记账行，保证下游行数对得上、不静默丢数据。
                stats["bad_lines"] += 1
                out = {
                    "input_index": index,
                    "error": "bad_json_line",
                    "prompt": None,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["written"] += 1
                continue

            # 字段瘦身：只取白名单里的字段，其余全丢弃（§5.1 瘦身）。
            out: Dict = {}
            for key in KEEP_FIELDS:
                if key in record:
                    out[key] = record[key]
                else:
                    out[key] = None  # 字段缺失也占位，保持输出 schema 稳定

            if not isinstance(record.get("question"), str) or not record.get("question"):
                stats["missing_question"] += 1

            out["prompt"] = template_one(record)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["written"] += 1

    return stats


# ---------------------------------------------------------------------------
# 命令行入口（任务书 §6.6：所有脚本支持 python xxx.py --input 路径 --output 路径）
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="筛 1：套冻结模板生成统一 prompt，并对字段瘦身（任务书 §5.1）。",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="输入 jsonl 路径（默认 %s）" % DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="输出 jsonl 路径（默认 %s）" % DEFAULT_OUTPUT,
    )
    return parser


def _main(argv: Optional[List[str]] = None) -> int:
    """CLI：解析参数 → 跑筛 1 → 按 §7.5 打印统计（读入 / 写出 / 缺 question / 坏行）。

    返回进程退出码（0 正常）。
    """
    args = _build_argparser().parse_args(argv)
    stats = run_filter1(args.input, args.output)
    print(
        "[f1_template] 读入 %d 行 / 写出 %d 行 / 缺 question %d / 坏行 %d"
        % (stats["read"], stats["written"], stats["missing_question"], stats["bad_lines"])
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
