# Ascend-910B-PPO-RLHF

> **基于华为昇腾 910B3（8 卡 NPU）集群的大模型 PPO 强化学习后训练（RLVR: 可验证奖励）实战套件**

本项目面向高性能昇腾智算集群（SCOW AI / K8s 容器架构），提供了在 **8×Ascend 910B3 (64GB)** 上基于 **verl 0.6.1 + vllm-ascend 0.9.1rc1** 运行 **Qwen2.5-7B-Instruct** / **0.5B** PPO 强化学习后训练的完整工程方案、踩坑修复与一键自动化运行脚本。

---

## 📌 核心特性与技术沉淀

1. **确定性技术栈搭配**：
   - 官方预打包底座镜像：`quay.io/openeuler/vllm-ascend:0.9.1rc1-torch_npu2.5.1-cann8.1.rc1-python3.10-oe2203lts`
   - 训练与推理栈：`verl 0.6.1` + `torch_npu 2.5.1` + `CANN 8.1.rc1` + `vllm-ascend 0.9.1rc1`。
2. **多卡通信组 Bug 独家修复 (`patch_vllm_ascend.py`)**：
   - **现象**：在多卡 verl rollout 初始化阶段报 `AssertionError: assert self.cpu_group is not None`；
   - **根因**：`vllm-ascend 0.9.1rc1` 按单引擎独占世界创建通信组（EP/ETP 组 ranks 仅为 `[0]`），与 verl 的全局通信域（world=N）冲突，导致 rank > 0 的 worker 无法加入通信组；
   - **修复**：内置幂等补丁，改用全局 world size 动态建立通信组，完美支持 8 卡分布式 rollout。
3. **平台持久化隔离设计**：
   - 针对 SCOW / K8s 平台「容器即焚、仅挂载目录持久保存」的特性，建立挂载目录持久虚拟环境（`envs/verl_env`），继承底层 PyTorch/NPU 驱动，免除每次作业重新编译安装的时间；
   - 脚本内置**学号路径自动检测**与**虚拟环境自动激活**，杜绝因未激活环境抛出的 `ModuleNotFoundError: No module named 'verl'`。
4. **RLVR（可验证规则奖励）流水线**：
   - 采用客观代码单元测试作为判定内核，零 RM 模型显存开销；
   - 支持 0.5B 单卡冒烟、8 卡拓扑验证与 7B 8 卡全量训练。
5. **三大权威基准 791 题全量实测验证**：
   - 覆盖 KodCode Held-Out（200 题）、OpenAI HumanEval（164 题）、Google MBPP Sanitized（427 题），达成跨基准 +4.12% 宏观净增益，GRPO 创 HumanEval 83.54% 纪录。

---

## 🏆 核心实测成果：三大权威基准（791 题）全景对比

所有数据均在 **8×华为昇腾 910B (64GB)** 真实物理算力环境下执行确定性评测（贪婪解码，64 线程 CPU 原生安全沙箱执行全量断言），详情参见完整技术报告 [`ASCEND_910B_RLVR_TECHNICAL_REPORT.md`](ASCEND_910B_RLVR_TECHNICAL_REPORT.md)：

| 模型版本 | KodCode (200 题) | HumanEval (164 题) | MBPP (427 题) | 全量 791 题宏观通过率 | 致命错误数 | 平均代码长度 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-7B-Instruct 基座** | 76.50% (153/200) | 78.66% (129/164) | 73.30% (313/427) | **76.15% (595/791)** | 24 次 | 382.1 tok |
| **PPO Step 71 (主线最终版)** | **82.00% (164/200)** | 82.93% (136/164) | **75.88% (324/427)** | **80.27% (624/791)** 🏆 | **12 次 (-50%)** | 379.7 tok |
| **GRPO Step 50 (消融最终版)** | 79.00% (158/200) | **83.54% (137/164)** ⚡ | **75.88% (324/427)** | **79.47% (619/791)** | 18 次 | **192.4 tok (-49%)** |
| **PPO-NegPenalty Step 50 (消融 1)** | 77.50% (155/200) | 78.66% (129/164) | 73.30% (313/427) | **76.24% (597/791)** | **仅 2 次 (-91.7%)** 🛡️ | **203.3 tok (-46.8%)** |
| **PPO-SparseRLVR Step 50 (消融 2)** | 74.00% (148/200) | **81.10% (133/164)** ⚡ | **75.41% (322/427)** ⚡ | **76.23% (603/791)** | **仅 4 次 (-83.3%)** 🛡️ | 359.5 tok (-5.3%) |
| **PPO-DiscreteBins Step 50 (消融 3)**| **76.00% (152/200)** ⚡ | **81.10% (133/164)** ⚡ | **75.64% (323/427)** ⚡ | **76.86% (608/791)** | **仅 4 次 (-83.3%)** 🛡️ | 362.4 tok (-5.2%) |
| **PPO-LenEfficiency Step 50 (消融 4)**| 75.50% (151/200) | **81.10% (133/164)** ⚡ | **76.11% (325/427)** 🏆 | **76.99% (609/791)** 🥇 | **仅 2 次 (-91.7%)** 🛡️ | **174.9 tok (-54.0%)** ⚡ |

* **PPO 领跑总榜**：全量 791 题通过率达 **80.27%**（+4.12%，净多解 29 题），在长逻辑工程题（KodCode）上斩获 82.00%，致命运行时崩溃腰斩 50%；
* **GRPO 极简突围**：单步耗时缩短 42%，显存节省 14GB，在纯算法基准 **OpenAI HumanEval** 上以 **83.54%** 超越 PPO，创全场最高分！

---

## 📦 必须额外下载与准备的外部资源清单（.gitignore 忽略项说明）

由于 GitHub 存在 **单文件 100MB 限制** 且代码仓库不适宜存储大模型权重与海量数据集，以下内容已在 `.gitignore` 中排除。**在新平台/新账户首次部署时，需按下方说明下载并上传至集群挂载点**：

| 资源类别 | 文件/目录名称 | 预估大小 | 官方/镜像下载源 | 集群内放置路径 | 说明 |
|---|---|---|---|---|---|
| **大模型权重** | `Qwen2.5-7B-Instruct/` | ~15.2 GB | [ModelScope](https://modelscope.cn/models/qwen/Qwen2.5-7B-Instruct) / [HF-Mirror](https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct) | `/data/home/<你的学号>/Qwen2.5-7B-Instruct` | 主训练基座（共 13 个文件，含 4 个 safetensors 分片） |
| **大模型权重** | `Qwen2.5-0.5B-Instruct/` | ~954 MB | [ModelScope](https://modelscope.cn/models/qwen/Qwen2.5-0.5B-Instruct) / [HF-Mirror](https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct) | `/data/home/<你的学号>/Qwen2.5-0.5B-Instruct` | 快速冒烟/拓扑验证基座（10 个文件） |
| **Web IDE** | `code-server-4.137.0-linux-arm64` | ~223 MB (.tar) | [GitHub Releases v4.137.0](https://github.com/coder/code-server/releases/download/v4.137.0/code-server-4.137.0-linux-arm64.tar.gz) | `/data/home/<你的学号>/project/code-server-4.137.0-linux-arm64` | 容器内网页 VSCode 服务（解压后使用） |
| **冒烟数据集** | `data/gsm8k/` | ~5 MB | 由 verl 内置工具在线生成 | `/data/home/<你的学号>/project/data/gsm8k/` | 包含 `train_300.parquet` 与 `test.parquet` |
| **代码题候选库** | `kodcode_candidates.jsonl` | ~264 MB | 本地抽取 / [KodCode-V1](https://huggingface.co/datasets/kodcode/kodcode-v1) | `/data/home/<你的学号>/project/data/` | D2 阶段数据清洗使用（可本地清洗后传 `verified_pool.jsonl`） |

### 1. 模型快速下载脚本（推荐在能高速联网的机器下载后上传）
```bash
# 推荐使用 ModelScope（国内镜像源极速下载，无 401 凭证问题）
pip install modelscope
modelscope download --model qwen/Qwen2.5-7B-Instruct --local_dir ./Qwen2.5-7B-Instruct
modelscope download --model qwen/Qwen2.5-0.5B-Instruct --local_dir ./Qwen2.5-0.5B-Instruct
```

### 2. code-server 下载、服务器内解压与移动就位
`code-server` 用于在容器中启动网页版 VSCode 交互界面。由于安装包体积（~223MB）超出 GitHub 限制，需要在服务器内自行解压并把文件夹移动至 `project/` 目录下：

1. **下载安装包**：
   - 官方发布页下载：[code-server-4.137.0-linux-arm64.tar.gz](https://github.com/coder/code-server/releases/download/v4.137.0/code-server-4.137.0-linux-arm64.tar.gz)（约 223MB，注意必须为 **linux-arm64** 架构）；
   - 上传至平台用户主目录 `/data/home/<你的学号>/`。
2. **在服务器内解压并移动文件夹**：
   ```bash
   cd /data/home/<你的学号>
   # 在服务器内解压
   tar -xzf code-server-4.137.0-linux-arm64.tar.gz
   # 将解压出的文件夹移动到 project/ 目录下
   mv code-server-4.137.0-linux-arm64 project/
   ```
3. **核验最终路径**：
   - 确保核心程序就位：
     `/data/home/<你的学号>/project/code-server-4.137.0-linux-arm64/bin/code-server`
   > **⚠️ 避坑提醒（平台网页解压陷阱）**：如果使用智算平台网页端「文件管理」自带的解压功能，平台可能会按文件名多建一层目录（如 `code-server-4.137.0-linux-arm64.tar/code-server-4.137.0-linux-arm64/`），此时务必手动将内层真正的 `code-server-4.137.0-linux-arm64` 文件夹 `mv` 移动到 `/data/home/<你的学号>/project/` 根目录下！

### 3. 冒烟数据集一键生成（免手动下载）
仓库已内置生成脚本 [`project/prepare_gsm8k.sh`](file:///e:/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/4_%E9%A1%B9%E7%9B%AE/RLHF/Ascend-910B-PPO-RLHF/project/prepare_gsm8k.sh)，在容器内虚拟环境建好后直接执行即可：
```bash
cd /data/home/<你的学号>/project
bash prepare_gsm8k.sh
# 自动生成 data/gsm8k/train.parquet, data/gsm8k/train_300.parquet, data/gsm8k/test.parquet
```

---

## 📂 仓库目录结构

```text
Ascend-910B-PPO-RLHF/
├── README.md                      # 项目说明与上手文档
├── 平台操作SOP.md                 # 实测部署 SOP（按步骤操作即可跑通）
├── 项目计划_910B_PPO后训练.md      # 项目方案、奖励函数与消融实验设计
├── 项目计划_910B_PPO后训练.pdf     # 完整项目计划 PDF 版
├── D1_开工清单.md                  # 环境选型摸底与实测基线
├── D2_代码任务书_给AI生成.md       # 数据清洗与沙箱判分器需求规格
├── D2_数据准备清单.md              # 训练池准备与流水线规划
├── D3_PPO调通与闭环验证清单.md      # PPO 20-step 冒烟验证与闭环设计
├── D4_主训练与消融评测清单.md      # 全量主训练、消融矩阵与评测交付
├── 北理工智算集群用户指南.docx       # 智算平台使用说明手册
└── project/                       # 核心执行脚本与代码
    ├── patch_vllm_ascend.py       # vllm-ascend 0.9.1rc1 多卡通信组修复补丁
    ├── prepare_gsm8k.sh           # GSM8K 冒烟数据集一键生成与裁剪脚本
    ├── smoke_ppo_05b.sh           # 0.5B PPO 冒烟测试脚本（单卡/多卡均可）
    ├── smoke_ppo_7b.sh            # 7B 8卡 PPO 训练验证脚本（微批自适应）
    ├── start_train.sh             # 无人值守正式训练入口（含保活与 code-server）
    ├── start_smoke.sh             # 自动化冒烟测试与保活
    ├── setup_env_msrl.sh          # MindSpeed-RL 备选环境安装脚本
    ├── mbpp_sanitized.jsonl       # 冒烟验证用小型测试集
    ├── rewards/
    │   └── smoke_gsm8k.py         # 兼容 verl 0.6.1 签名的规则打分 Wrapper
    └── pipeline/                  # 数据清洗漏斗与沙箱判分器（RLVR 核心基石）
        ├── config.py              # 全局配置中心（冻结模板、超时阈值、去重规则）
        ├── extract.py             # 模型输出代码提取器（正则截取 ```python 块）
        ├── sandbox.py             # 安全执行沙箱（进程树监控、超时强杀、无 shell 注入）
        ├── score.py               # 判分器与奖励函数（pytest 部分分捕获、降级 runner）
        ├── f1_template.py         # 筛 1：套用统一 Prompt 模板
        ├── f2_verify.py           # 筛 2：标准答案过沙箱自洽性校验
        ├── f3_dedup.py            # 筛 3：去重 + 截长 + 评测防泄漏过滤
        ├── f4_sample_filter.py    # 筛 4：7B 8 卡并行采样与 0.1~0.9 黄金难度池筛选
        └── tests/                 # 115 项自动化验收测试（100% 通过）
            ├── test_extract.py        # 15 条提取模块单元测试
            └── test_boundary_100.py   # 100 条全边界极限沙箱测试
```

---

## 🚀 极速上手流程（针对集群新作业）

### 第 1 步：注册与拉取镜像（网页端）
* 镜像来源：**远程镜像**
* 镜像地址：
  ```text
  quay.io/openeuler/vllm-ascend:0.9.1rc1-torch_npu2.5.1-cann8.1.rc1-python3.10-oe2203lts
  ```
* **注意**：用户名 / 密码**必须留空**（无需凭证，填写错误会导致 401 Unauthorized）。

### 第 2 步：上传模型与创建开发容器
* 上传下载好的模型至平台个人根目录 `/data/home/<你的学号>/`；
* 平台「应用」→「创建 VSCode」：
  * **镜像**：选择上一步拉取的镜像；
  * **算力**：开发/冒烟选 1 卡，正式训练选 8 卡（910B3）；
  * **挂载点**：勾选 `project` 目录以及模型目录（`Qwen2.5-7B-Instruct` / `Qwen2.5-0.5B-Instruct`）；
  * **运行命令**：
    ```bash
    chmod +x /data/home/<你的学号>/project/code-server-4.137.0-linux-arm64/bin/code-server && /data/home/<你的学号>/project/code-server-4.137.0-linux-arm64/bin/code-server --auth none
    ```

### 第 3 步：容器内初始化（仅首次需建 venv，后续跨容器永久复用）
进入容器终端：
```bash
# 1. 验证原生镜像驱动
python3 -c "import torch, torch_npu, vllm; print(torch.__version__, torch.npu.is_available(), vllm.__version__)"

# 2. 创建并激活持久虚拟环境
python3 -m venv --system-site-packages /data/home/<你的学号>/project/envs/verl_env
source /data/home/<你的学号>/project/envs/verl_env/bin/activate
pip install verl==0.6.1

# 3. 验证环境
python3 -c "import torch, torch_npu, vllm, verl; print('OK', torch.npu.is_available())"

# 4. 生成冒烟数据集
cd /data/home/<你的学号>/project
bash prepare_gsm8k.sh
```

> **提示**：建议在 `~/.bashrc` 中写入：
> ```bash
> echo 'source /data/home/<你的学号>/project/envs/verl_env/bin/activate' >> ~/.bashrc
> ```
> 这样每次新开终端都会自动进入正确的 Python 环境。

---

## 📊 数据清洗与沙箱判分（D2 Pipeline）

本仓库集成了完整的 RLVR 数据清洗漏斗与高并发安全沙箱，支持在本地 CPU 或服务器上完成全套过滤：

```bash
cd /data/home/<你的学号>/project

# 1. 运行沙箱与提取器回归单测（115 项全边界测试，预期 100% Passed）
python -m pytest pipeline/tests/ -v

# 2. 筛 1：模板填充与 Prompt 规范化
python pipeline/f1_template.py \
    --input data/kodcode_candidates.jsonl \
    --output data/step1_templated.jsonl

# 3. 筛 2：官方解答过沙箱自洽性校验（务必开启 16 核并发，实测 28 分钟 1.5 万条全量跑完）
# 实测基准：读入 15000 / 保留 13871 / 淘汰 1129 / 通过率 92.5% / 平均耗时 1.875s
nohup python pipeline/f2_verify.py \
    --input data/step1_templated.jsonl \
    --output data/step2_kept.jsonl \
    --rejects data/step2_rejects.jsonl \
    --workers 16 > f2_verify.log 2>&1 &

# 4. 筛 3：去重 + 截长(>2000字) + 评测防泄漏过滤(>0.9)（纯 CPU 计算，3 秒搞定）
# 实测基准：读入 13871 / 保留 10434 / 淘汰 3437 (过长 3407，泄漏 30)
python pipeline/f3_dedup.py \
    --input data/step2_kept.jsonl \
    --output data/step3_verified_pool.jsonl \
    --rejects data/step3_rejects.jsonl

# 5. 筛 4：7B 基座 8 卡并行预采样与难度分级（产出 rl_pool.jsonl 2~4k + heldout.jsonl 200）
# ① 推荐先跑 16 题快速冒烟（每卡 2 题，约 20~30 秒完成全链路核验）：
python pipeline/f4_sample_filter.py \
    --input data/step3_verified_pool.jsonl \
    --rl_pool data/test_rl_pool.jsonl \
    --heldout data/test_heldout.jsonl \
    --model /data/home/<你的学号>/Qwen2.5-7B-Instruct \
    --gpus 8 \
    --max_problems 16

# ② 正式全量 8 卡并行后台运行（10,434 题，8 卡总吞吐 ~1800 tok/s）：
nohup python pipeline/f4_sample_filter.py \
    --input data/step3_verified_pool.jsonl \
    --rl_pool data/rl_pool.jsonl \
    --heldout data/heldout.jsonl \
    --rejects data/step4_rejects.jsonl \
    --model /data/home/<你的学号>/Qwen2.5-7B-Instruct \
    --gpus 8 \
    --batch_size 50 \
    --gpu_memory_utilization 0.6 > f4_filter.log 2>&1 &

# ③ 实时监控运行状态与筛选进展命令（常用操作）：
# a. 查看实时日志滚动流
tail -n 25 f4_filter.log

# b. 查看各卡 GPU Worker 独立处理进度（行数累计）：
wc -l data/_tmp_f4_worker_*.jsonl

# c. 一键统计当前已处理题数及【极易 / 极难 / 🌟黄金难度池】分布：
python3 -c "
import json, glob
easy, hard, golden = 0, 0, 0
for f in glob.glob('data/_tmp_f4_worker_*.jsonl'):
    for line in open(f, encoding='utf-8'):
        d = json.loads(line)
        pr = d.get('sample_pass_rate', 0.0)
        if pr > 0.9: easy += 1
        elif pr < 0.1: hard += 1
        else: golden += 1
tot = easy + hard + golden
print(f'已处理: {tot} 题 | 极易(>0.9): {easy} | 极难(<0.1): {hard} | 🌟 黄金池(0.1~0.9): {golden} 条 ({golden/tot*100:.1f}%)')
"
```

### 💡 筛 4 耗时特征与算力归属分析（NPU vs CPU）

* **为什么 8 卡并行筛选仍需约 1 ~ 1.5 小时？**
  * **8 卡 NPU 生成极快**：8 张 910B 算力充沛，单卡吞吐达 `~1,800 tok/s`，每批 50 题（每题采样 8 次 = 400 个代码回复）的 NPU 生成仅需 **约 1 分 15 秒**；
  * **耗时主要在 CPU 沙箱单测**：每个 Worker 需将这 400 个模型生成的完整 Python 脚本逐一拉起 `pytest` 沙箱进行动态断言测试，CPU 单进程串行跑完 400 题单测需 **约 6 分钟**；
  * 因此单批次（50题/卡 × 8卡 = 400题）总耗时约 7~8 分钟。
* **数据筛选 vs 强化学习训练的算力使用区别**：
  * **D2 数据筛选阶段**：由于需要对数万次采样结果逐个跑单测，CPU 在执行 pytest 时成为主要耗时瓶颈，NPU 大部分时间处于等待沙箱返回状态；
* **实测筛选成果与提前收割（Final Dataset Settled）**：
  * **处理至 6,000 题时实测统计**：极易题（>0.9）134 题，极难题（<0.1）3,196 题，**🌟 黄金难度题（0.1~0.9）累计达 2,670 条（占比 44.5%）**！
  * **提前收割最佳实践**：因 2,670 条已完美契合 2k~4k 的 PPO 训练池区间，为避免容器作业到期被掐断，直接在终端执行以下自包含命令一键结算并打包：
    ```bash
    cd /data/home/<你的学号>/project

    # 1. 停掉后台筛选进程
    pkill -9 -f f4_sample_filter

    # 2. 一键切分并导出终版数据集
    python3 -c "
    import glob, json, os, random
    random.seed(42)
    files = sorted(glob.glob('data/_tmp_f4_worker_*.jsonl'))
    all_data = [json.loads(l) for f in files for l in open(f, encoding='utf-8') if l.strip()]

    golden, easy, hard = [], [], []
    for item in all_data:
        pr = item.get('sample_pass_rate', 0.0)
        if pr > 0.9: easy.append(item)
        elif pr < 0.1: hard.append(item)
        else: golden.append(item)

    random.shuffle(golden)
    heldout = golden[:200]
    rl_pool = golden[200:]

    with open('data/heldout.jsonl', 'w', encoding='utf-8') as f:
        for item in heldout: f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open('data/rl_pool.jsonl', 'w', encoding='utf-8') as f:
        for item in rl_pool: f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open('data/step4_rejects.jsonl', 'w', encoding='utf-8') as f:
        for item in (easy + hard): f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'🎉 结算完成！heldout: {len(heldout)} 条 | rl_pool: {len(rl_pool)} 条 | rejects: {len(easy+hard)} 条')
    "

    # 3. 将【筛选前】与【筛选后】全套数据集打包为单个压缩包（供本地下载或消融对比）
    tar -czvf rlvr_datasets.tar.gz \
        data/step3_verified_pool.jsonl \
        data/rl_pool.jsonl \
        data/heldout.jsonl \
        data/step4_rejects.jsonl
    ```
  * **最终产出清单**：
    * `data/heldout.jsonl`：**200 条** 严格隔离考试题（绝不进训练池）；
    * `data/rl_pool.jsonl`：**2,470 条** 黄金难度 PPO 训练池（通过率 10%~90%，完美契合 8 卡 3~4 小时训练规模）；
    * `data/step4_rejects.jsonl`：**3,330 条** 淘汰题目；
    * `rlvr_datasets.tar.gz`：全套数据集压缩包（在网页 VSCode 或 SCOW 文件管理中直接下载到本地电脑）。

---

## 🧪 8 卡集群冒烟与正式训练

进入 `project` 目录：
```bash
cd /data/home/<你的学号>/project

# 1. 打多卡通信补丁（容器层修改，幂等，已内置在各启动脚本中）
python3 patch_vllm_ascend.py

# 2. 测试 1：0.5B × 8 卡拓扑冒烟（~10 分钟）
bash smoke_ppo_05b.sh trainer.n_gpus_per_node=8 2>&1 | tee smoke_8c_05b.log

# 3. 测试 2：7B × 8 卡正式训练彩排（~30 分钟）
bash smoke_ppo_7b.sh trainer.n_gpus_per_node=8 2>&1 | tee smoke_8c_7b.log
```

**成功标志**：
1. 日志打印 `world size 8`，Ray 成功拉起 8 个 Actor/Critic worker；
2. 8 个 vLLM rollout 引擎初始化就绪；
3. `step:0` 出现且 reward 均值与方差开始滚动更新。

---

## 🎯 后续阶段规划与任务指引（D3 & D4）

项目全周期包含严密的阶段演进，D2 数据闭环完成后，后续工作被严格拆分为两份独立实施清单：

### 1. [D3 阶段：PPO 调通与 20-Step 闭环验证清单](./D3_PPO调通与闭环验证清单.md)
* **核心哲学**：**绝不上来就跑全量过夜长训！** 必须先在 8 卡 910B 上跑通 20 步的最小全链路闭环，确认显存不会打爆、奖励函数正常判分、梯度能回传。
* **四大关键任务**：
  1. **数据打包与格式转换（`pipeline/pack_to_parquet.py`）**：将 `rl_pool.jsonl` 转换为 verl 标准 Parquet 格式，拆分为 `train_smoke.parquet` (128题)、`train_full.parquet` 与 `test.parquet` (200题)；
  2. **正式代码 RLVR 奖励驱动（`rewards/code_rlvr.py`）**：挂载正式代码沙箱，支持格式分（0.1）+ 测试用例通过率连续分（0.0~0.9）；
  3. **7B × 8 卡 PPO 20-step 闭环验证**：执行 `smoke_ppo_7b_rlvr.sh`，核验单卡显存（<45GB）、单步吞吐（2~4分钟）、Reward 稳定上行、KL 散度受控（0.001~0.05）与 Checkpoint 成功保存；
  4. **KL 惩罚系数 $\beta$ 短跑摸底**：对比 $\beta=0.001$ 与 $\beta=0.0005$ 的探索自由度与策略熵，定稿 D4 超参。

### 2. [D4 阶段：全量主训练、消融矩阵与评测交付清单](./D4_主训练与消融评测清单.md)
* **核心哲学**：**主训练出顶梁柱模型，消融实验出科研与答辩深度！** 证明模型不是死记硬背，而是真正掌握了通用代码推理能力。
* **四大核心任务**：
  1. **全量 7B × 8 卡正式主训练（过夜长训）**：全量 2,000 ~ 4,000 条黄金题目，跑 1 ~ 2 Epoch（约 40 ~ 80 steps，3 ~ 4 小时），使用 `start_train.sh` 后台无人值守运行，每 10 步自动存权重；
  2. **三大核心消融实验（论文灵魂）**：
     - **消融 1（奖励机制）**：纯 0/1 稀疏奖励 vs 平滑连续部分分（主线方案）；
     - **消融 2（算法架构）**：PPO（带 Critic） vs GRPO（无 Critic 群组优势采样）；
     - **消融 3（数据纯度）**：未经筛选的原题池 vs D2 筛4黄金难度池，量化“四道大筛”对 RL 收敛的决定性贡献；
  3. **客观独立评测与基准对比**：在从未见过的 **Held-out 200 题** 与 **公共基准 MBPP** 上评测 Pass@1 / Pass@8，形成最终的模型能力跃升大对照表；
  4. **训练动力学与安全审计（防 Reward Hacking）**：导出 TensorBoard 论文级曲线；人工抽检 50 条高分代码，编写《防 Hacking 审计报告》，确认无作弊行为。

---

## ⚠️ 常见避坑指南（踩坑总结）

| # | 踩坑现象 | 根本原因 | 正确应对 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'verl'` | 新开终端未激活持久 venv，使用了镜像默认 Python | 运行 `source /data/home/<学号>/project/envs/verl_env/bin/activate` 或使用本仓库脚本（内置自动激活） |
| 2 | `AssertionError: assert self.cpu_group is not None` | vllm-ascend 0.9.1rc1 通信组只分配了 rank 0 | 执行 `python3 patch_vllm_ascend.py` 打入多 worker 通信组补丁 |
| 3 | 算子执行报 `ADD_TO_LAUNCHER_LIST` 失败 | 手动 source 了外置的 CANN `set_env.sh` 导致环境冲突 | **严禁手动 source 任何 CANN 环境变量**，镜像自带的 8.1.rc1 已经就绪 |
| 4 | pip 报错 `AssertionError: pyc_path` | dpc 网络文件系统与 pyc 预编译冲突 | 安装前执行 `export PIP_NO_COMPILE=1`，安装 verl 与 pytest |
| 5 | 断网 / 关闭网页终端导致训练中断 | 终端 session 被杀死 | 使用无人值守入口 `start_train.sh` 或后台运行 `nohup ... &` |
| 6 | GitHub clone 报 `HTTP/2 stream 1 was not closed cleanly` | 国内服务器直连 GitHub 网络抖动阻断 | 执行 `git config --global http.version HTTP/1.1` 或使用加速镜像 `https://ghfast.top/https://raw.githubusercontent.com/...` |
| 7 | vLLM 实例加载报错 `unexpected keyword argument 'tensor_model_parallel_size'` | 混淆了 verl 配置键名与 vLLM 原生 Python API 参数名 | 原生 vLLM 构造函数形参为 `tensor_parallel_size=1`（单卡亦可直接缺省） |
| 8 | 多卡采样时主进程永久卡死在 Queue 获取 | `multiprocessing.Queue` 跨进程传输上万条大字典填满系统 buffer | 采用 Worker 独立分块直写文件 `_tmp_f4_worker_{gpu_id}.jsonl`，主进程安全合并 |
| 9 | 昇腾 910B 推理报 HBM 碎片或 OOM 告警 | `gpu_memory_utilization` 设过高 (0.85+) 导致挤占算子及驱动显存 | 设为 `0.6` 即可（14.25GB 权重 + 24GB KV Cache，单卡 64GB 非常宽裕） |
| 10 | 筛 4 多卡采样耗时约 1 小时，怀疑卡死或 GPU 未加速 | 8 卡 NPU 生成极快（~1800 tok/s，每批只需 1 分钟），耗时瓶颈在 CPU 逐题串行跑 pytest 沙箱单测（~6 分钟） | 属正常现象，通过 `tail -n 25 f4_filter.log` 观察 Worker 推进；后续 D3/D4 强化学习训练完全由 8 卡 NPU 全速运算 |
| 11 | pkill 杀死多进程脚本后二次启动报 OOM | pkill 仅杀死 Python 父进程，残留的 `multiprocessing.spawn` 子进程仍霸占 8 张 NPU 显存（56GB/卡） | 运行 `npu-smi info` 核对；如有残留显存占用，用 `kill -9 <PIDs>` 或 `pkill -9 -f spawn_main` 彻底清空显存 |


---

## 👥 协作与使用说明
* 欢迎协作团队与班级同学基于此模板克隆并配置自己的训练任务；
* 运行前将 `<你的学号>` 替换为各自平台的主目录路径即可无缝复用。

