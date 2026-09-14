# 任务书：代码 RL 数据清洗流水线（筛 1~3 + 判分器）

> 本文档是给"人 + AI"协作的完整任务书。把整份文档喂给 AI，让它按规格生成代码；生成的代码必须通过 §6 的验收闸门才算完成。
> 角色：你是数据管线工程师。你的产出会被同事评审，评审标准见验收部分。

---

## 1. 背景（两句话）

我们在昇腾 910B3 上做代码任务的 PPO 强化学习（RLVR：可验证奖励）。训练池是 15,000 条候选编程题（`kodcode_candidates.jsonl`），需要一条本地清洗流水线把它加工成"每道题都有统一 prompt + 可执行单测"的 `rl_pool.jsonl`。

## 2. 数据格式（输入）

### 输入 1：`data/kodcode_candidates.jsonl`（15,000 行，每行一个 JSON 对象）

实际存在的字段（已核实）：`version, style, subset, question_id, question, solution, test, test_info, gpt_pass_sequence, gpt_pass_trial_num, gpt_difficulty, gpt_pass_percentage, trials, chosen_trial, metadata, benchmark_similarity, benchmark_instruction, benchmark_task_id, filter_reason`

流水线只关心这些字段：
| 字段 | 含义 |
|---|---|
| `question_id` | 唯一 ID |
| `question` | 题面（含函数签名/docstring 的英文编程题） |
| `solution` | 官方参考解答（筛 2 用它验证测试可跑） |
| `test` | pytest 风格单测代码，内部形如 `from solution import xxx` 后跟多个 `test_xxx()` 函数 |
| `style` / `subset` | 来源风格/子集（写入 meta） |
| `gpt_difficulty` | 官方难度标签 easy/medium/hard（写入 meta） |
| `benchmark_similarity` | 与公开基准的相似度浮点数（>0.9 的题要过滤，防评测泄漏） |

### 输入 2：`data/mbpp_sanitized.jsonl`（420 行）

字段：`source_file, task_id, prompt, code, test_imports, test_list(list[str]), split`
用途：冒烟测试判分器的小样本；`test_list` 是 assert 字符串列表，需拼接成测试文件（前面加 `code` 作为被测实现，或按 `from solution import` 约定组装——两种都支持，用参数切换）。

## 3. Prompt 模板（已冻结，禁止改动一个字符）

```python
PROMPT_TEMPLATE = """Solve the following Python problem. Think briefly, then put your complete
implementation in a ```python code block.

Problem:
{question}"""
```

配套提取规则（判分器输入接口）：从模型输出中提取**第一个** ` ```python ... ``` ` 围栏代码块；无围栏代码块 → 视为无代码（passed=0, total=0）。

## 4. 交付物（六个文件 + 一个测试文件）

```
pipeline/
├── config.py            # 所有常量集中在此：模板、超时、阈值、路径、内存上限
├── sandbox.py           # run_in_sandbox()：两层设计（见 §5.2）
├── extract.py           # extract_python()：按 §3 提取规则
├── score.py             # score_kernel()：调用前两者，返回 (passed, total, detail)；另含 score() 薄壳（线性通过率 + 格式分 0.1 + 超时惩罚，作为默认策略，后续会被替换——加 TODO 标注）
├── f1_template.py       # 筛 1：套模板 + 字段瘦身
├── f2_verify.py         # 筛 2：官方解答过沙箱，淘汰测试跑不通的题
├── f3_dedup.py          # 筛 3：去重 + 截长 + 泄漏过滤
└── tests/
    └── test_boundary_100.py   # 100 条边界单测（参数化）
```

## 5. 详细规格

### 5.1 筛 1（f1_template.py）
- 输入 `kodcode_candidates.jsonl` → 输出 `step1_templated.jsonl`；
- 每条：`question` 填入模板生成 `prompt`；保留 `question_id/style/subset/test/solution/gpt_difficulty/benchmark_similarity`，其余字段丢弃（瘦身）；
- 结尾打印统计：读入 N / 写出 N（筛 1 不淘汰）。

### 5.2 沙箱（sandbox.py）——全项目最核心的工程件
```python
def run_in_sandbox(files: dict[str, str], entry: list[str], timeout_s: int = 10,
                   enhanced: bool = False) -> RunResult
# RunResult: (returncode, stdout, stderr, timed_out: bool, duration_s: float)
```
- 基础版（默认，Windows 可用）：`tempfile` 建独立工作目录 → `files` 字典写入该目录 → `subprocess.run(entry, timeout=timeout_s, capture_output=True, cwd=工作目录)` → 超时则 kill 进程树并标记 `timed_out=True`；
- 增强版（`enhanced=True`，仅 Linux，服务器用）：在基础版上加 `preexec_fn`：`resource.setrlimit(RLIMIT_AS, 2GB)` + `RLIMIT_CPU`；网络隔离尝试 `unshare -n` 包一层（不可用则打 warning 降级，不报错）；
- **`import resource` 必须放进函数内的 try 块**（Windows 无此模块，文件顶层 import 会让本地直接崩）；
- 工作目录用完清理（或 `TemporaryDirectory` 上下文）；
- 禁止 `shell=True`。

### 5.3 提取（extract.py）
- 正则匹配 ` ```python\n ... ``` `（容忍大小写、容忍首行语言标签缺失）；
- 多个块 → 取第一个；无块 → `None`；
- 15 条单元级测试覆盖以上变体。

### 5.4 测试执行与部分分计数（score.py 核心难点）
- 被测流程：候选代码写入 `solution.py` + `test` 内容写入 `test_case.py`（同目录）→ 沙箱执行 `python -m pytest test_case.py -v`；
- 计数规则：pytest 全绿 → `passed = total = 测试函数数`；有失败/错误 → 解析 `-v` 输出按 `PASSED/FAILED/ERROR` 逐函数计数；解析失败（输出格式意外）→ 保守取 `passed=0, total=函数数`；
- 测试函数数：优先从 `-v` 行数统计；拿不到时从 test 源码 `def test_` 计数；
- pytest 缺失时降级为 `python test_case.py` 全有全无（passed ∈ {0, total}）；
- `total == 0`（无任何测试函数）→ 该题无效。

### 5.5 筛 2（f2_verify.py）
- 对每条：`solution` 作为代码 + `test` 作为测试 → 沙箱执行；
- **保留**：`passed == total` 且 `total > 0` 且未超时 且 duration ≤ timeout；
- **淘汰**（写入 `step2_rejects.jsonl`，带 `drop_reason` 字段，取值：no_code / compile_error / timeout / tests_fail / no_tests）；
- 结尾打印统计：总数 / 保留 / 各淘汰原因计数 / 平均耗时。

### 5.6 筛 3（f3_dedup.py）
- `question` 的 md5 去重；
- `len(prompt) > 2000` 字符弃用（reason: too_long）；
- `benchmark_similarity > 0.9` 或字段缺失为 None 时保留（reason: leakage_risk 仅对 >0.9 生效）；
- 输出 `step3_verified_pool.jsonl`（目标 5~8k 条）。

### 5.7 筛 4：不在本任务书范围（需服务器 + 7B 模型，另行处理）。score.py 中给它留好接口即可。

## 6. 验收闸门（全部通过才算交付）

1. **100 条边界单测**（tests/test_boundary_100.py，pytest 参数化）覆盖以下类别，每条 case 是（模型输出 response, 测试文件 test, 期望 (passed,total)）三元组：
   - 正常类（~15）：单代码块、带注释、带 import、含辅助函数、代码块前后有解释文字；
   - 提取类（~25）：无代码块、两个代码块（取第一）、```无语言标签、```Python 大写、嵌套三引号字符串、块内含 ``` 字面量、行内 ` ``` `；
   - 运行类（~30）：全对 / 全错 / 部分对（10 函数过 7 挂 3，验证部分计数）/ 死循环（验证超时杀进程）/ 语法错 / 缺 import / raise 异常 / print 大量输出；
   - 沙箱类（~15）：超时后进程真的被杀、Windows 与 Linux 行为一致（timeout 生效）、临时目录清理、文件互不串扰（两题先后跑互不影响）；
   - 组装类（~15）：score() 的线性通过率、格式分、超时惩罚、组合边界（0/0、超时但有部分分等）。
2. **期望值人工复核**：AI 生成的期望 (passed, total) 必须由人逐条核对签名（在文件头部写"已复核：姓名/日期"）；判分器是全项目的尺子，尺子的刻度不许由 AI 说了算；
3. 100 条单测在本地全绿；
4. 筛 2 对 100 行随机候选样本端到端跑通，输出统计正常（官方解答+官方测试，通过率通常 ≥80%）；
5. 同事**人工通读** `sandbox.py` 与 `extract.py` 全文（这两个文件的正确性无法完全靠单测保证）；
6. 所有脚本支持 `python xxx.py --input 路径 --output 路径` 命令行调用。

## 7. 硬性约束

1. 依赖白名单：标准库 + pytest + pandas + pyarrow。**禁止**任何网络请求、禁止 requests/huggingface_hub；
2. 兼容 Python 3.10 与 3.14（本地 3.14、服务器 3.10 都要能跑）：不用 3.11+ 特性；`import resource` 只能放函数内 try；
3. 所有常量集中在 `config.py`（超时 10s、内存 2GB、长度 2000、相似度阈值 0.9、模板）；
4. 输出统一 jsonl、UTF-8、`ensure_ascii=False`；
5. 每个脚本幂等可重跑，结尾打印统计（进/出/各淘汰原因计数）；
6. 不许改模板字符串；不许引入本任务书未列的淘汰规则；
7. 代码注释用中文，每个函数一段 docstring 说明"为什么"。

## 8. 给协作 AI 的提示（生成代码时告诉它的话）

- "先生成 sandbox.py + extract.py + 15 条提取类单测，我确认后再生成其余部分"——分批生成，不要一次性生成全部六个文件；
- "部分分计数（§5.4）是唯一有难度的点，优先设计并展示你的解析方案让我确认"；
- "Windows 的 subprocess 超时杀进程要用 `subprocess.run(..., timeout=t)` 外加超时后 `proc.kill()` 兜底，注意 `capture_output` 与死锁问题"；
- "不要引入任何第三方依赖，不要联网"。
