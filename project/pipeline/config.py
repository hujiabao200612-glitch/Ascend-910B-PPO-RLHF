# -*- coding: utf-8 -*-
"""config.py —— 所有常量集中在此（任务书 §4 / §7.3）。

为什么单独放一个文件：超时、阈值、路径、内存上限、模板这些数字散落在各脚本里，
改一处要翻五个文件，迟早出 bug。这里做成「单一事实来源」，sandbox / extract /
f1_f3 全部从这里 import，别的地方不允许出现魔法数字。

注意：任务书 §3 的 PROMPT_TEMPLATE 是冻结模板，禁止改动一个字符（与 extract.py 的
提取规则是双胞胎，改模板 = 改判分器，要同定同冻）。
"""

# ---------------------------------------------------------------------------
# 沙箱（sandbox.run_in_sandbox）——原 sandbox.py 的运行时参数
# ---------------------------------------------------------------------------
# 单题执行超时（秒）；任务书 §7.3：10s
TIMEOUT_S = 10
# 增强版内存上限；任务书 §5.2 / §7.3：2GB
MEMORY_LIMIT_BYTES = 2 * 1024 ** 3
# RLIMIT_CPU 硬限比软限（= TIMEOUT_S）多留几秒，让程序有机会打完日志再被 SIGKILL
CPU_LIMIT_GRACE_S = 2
# 超时杀进程后，回收残留输出 / 僵尸进程的等待上限（秒）
REAP_TIMEOUT_S = 5
# 临时工作目录前缀，便于人工排查「有没有清理干净」
SANDBOX_WORKDIR_PREFIX = "rlvr_sandbox_"

# ---------------------------------------------------------------------------
# 提取（extract.extract_python）——原 extract.py 的 CLI 参数
# ---------------------------------------------------------------------------
# CLI 默认从对象哪个字段取模型输出（--field 可覆盖）
EXTRACT_DEFAULT_FIELD = "response"

# ---------------------------------------------------------------------------
# 提示词模板（§3，冻结模板；改动会影响判分器提取规则，需与 D3/D4 一起评审）
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """Solve the following Python problem. Think briefly, then put your complete
implementation in a ```python code block.

Problem:
{question}"""

# ---------------------------------------------------------------------------
# 清洗阈值（筛 1~3）
# ---------------------------------------------------------------------------
# 筛 3：prompt 字符数超过此上限弃用（§5.6）
MAX_PROMPT_LEN = 2000
# 筛 3：benchmark_similarity > 此值视为泄漏风险、弃用（§5.6）
BENCHMARK_SIMILARITY_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# 判分（score.py）
# ---------------------------------------------------------------------------
# 判分器跑 pytest 用的解释器；留空（""）则回落 sys.executable（即运行 score.py 自身的
# 那个 python，本地清洗用 venv_rlvr/Scripts/python.exe 时它自带 pytest）。
# 服务器侧若要用别的解释器，在这里改一处即可（§7.3：常量集中）。
JUDGE_PYTHON_EXE = ""

# score() 临时版的奖励公式参数（§5.3：线性通过率 + 格式分 0.1 + 超时惩罚）。
# 注意：这是 D3/D4 之前占位的临时策略，后续会被重写，所以只放最朴素的两个旋钮。
FORMAT_BONUS = 0.1          # 成功提取出代码（has_code）额外奖励 0.1
TIMEOUT_PENALTY_FACTOR = 0.5  # 超时则奖励乘以该系数（临时惩罚，非清零）

# score_kernel 沙箱里写出的两个固定文件名（§5.4：候选代码 + 测试同目录）
SOLUTION_FILENAME = "solution.py"
TESTCASE_FILENAME = "test_case.py"

# ---------------------------------------------------------------------------
# 数据 / 输出路径（相对项目根目录 D:\学习资料\code_rlvr）
# ---------------------------------------------------------------------------
DATA_DIR = "data"
CANDIDATES_JSONL = "data/kodcode_candidates.jsonl"      # 筛 1~3 原料（15,000 条）
MBPP_JSONL = "data/mbpp_sanitized.jsonl"                # 冒烟小样本（420 条）
STEP1_TEMPLATED = "data/step1_templated.jsonl"          # 筛 1 输出（筛 2 输入）
STEP2_KEEP = "data/step2_kept.jsonl"                    # 筛 2 保留（官方解答跑通，喂筛 3）
STEP2_REJECTS = "data/step2_rejects.jsonl"              # 筛 2 淘汰（跑不通，供复核）
STEP3_VERIFIED_POOL = "data/step3_verified_pool.jsonl"  # 筛 3 输出（目标 5~8k）
STEP3_REJECTS = "data/step3_rejects.jsonl"              # 筛 3 淘汰（复核用；§5.6 未强制要求，属附加审查产物）
