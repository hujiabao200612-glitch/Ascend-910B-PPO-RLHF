# 平台操作 SOP v3（队友部署版 · 按实测者实测通过路径编写）

> **你将得到的最终成果**：一个能在 910B3 上跑 PPO 强化学习训练的容器——verl 0.6.1 + torch_npu 2.5.1 + CANN 8.1.rc1 + vllm-ascend 0.9.1rc1，冒烟已全链路验证（4 步训练 + 验证生成 + 奖励计算全通）。
> **全程耗时**：镜像注册 5 分钟 + 上传 30~60 分钟（模型 15GB 是大头）+ 容器初始化 5 分钟 + 冒烟 15 分钟。
> **三个铁律**（都是踩过坑的）：① 注册镜像时**用户名密码必须留空**（填了学号会 401）；② **不要 source 任何 CANN set_env.sh**（镜像自带的 8.1.rc1 就是正确版本，叠加第二套会全算子报错）；③ 要保留的东西只写 `project` 挂载目录（容器其他位置随作业蒸发）。

---

## 第 1 步：注册镜像（每人 5 分钟，纯网页操作，无需下载）

平台 → 镜像 → 添加镜像，逐栏填写：

| 栏位 | 填什么 | 注意 |
|---|---|---|
| 镜像来源 | **远程镜像** | 不用下载任何东西 |
| 镜像地址 | `quay.io/openeuler/vllm-ascend:0.9.1rc1-torch_npu2.5.1-cann8.1.rc1-python3.10-oe2203lts` | 官方预配套组合，原样复制 |
| **用户名 / 密码** | **留空！** | 填了无效凭证会把匿名拉取搞成 401（实测踩坑） |
| 镜像名称 / 标签 | 随意，如 `vllm-ascend-091` / `v1` | 平台内部名字 |
| 类型 | 下拉里选通用/自定义类 | 别选纯"推理服务"类 |

提交后等平台从 quay.io 拉取（几分钟）。**状态可用的标志**：镜像列表里出现且无报错。

## 第 2 步：上传模型与数据（每人 30~60 分钟，与第 1 步并行）

平台 → 文件管理 → 上传以下内容（到你的用户目录下）：

| 文件 | 大小 | 来源 | 用途 |
|---|---|---|---|
| `Qwen2.5-7B-Instruct/`（整个文件夹，13 个文件） | 15.2GB | HF/镜像源下载 | 主基座模型 |
| `Qwen2.5-0.5B-Instruct/`（10 个文件） | 954MB | 同上 | 冒烟炮灰 |
| `mbpp_sanitized.jsonl` | 256KB | 本地 data_set | 冒烟数据 |

注意事项：
- **7B 是 13 个文件**（4 个 safetensors 分片 + index + config + tokenizer），缺一不可，传完核对字节数（分片应分别为 3945441440 / 3864726352 / 3864726424 / 3556377672）；
- 平台上传路径即你的用户目录：`/data/home/<你的学号>/...`（⚠️ 作业详情页显示的挂载路径可能显示别人的 ID 样式，**以 `mount | grep <你的学号>` 的实际输出为准**）；
- 谁的模型谁来传——每人用户目录独立，不共享（若管理员有共享数据集机制则可省）。

## 第 3 步：建作业（3 分钟）

应用 → 创建 VSCode，逐栏：

| 栏位 | 填什么 |
|---|---|
| 镜像源 | 本地镜像 → 选 `vllm-ascend-091:v1` |
| 加速卡 | 1（冒烟/开发）；正式训练另建 8 卡作业 |
| 最长运行时间 | 24h |
| 挂载点 | `project` + 两个模型文件夹（三个都勾） |
| **运行命令** | 见下（⚠️ **不能填裸的 `code-server`**——这个镜像里没有，会 `command not found` 秒挂） |

运行命令（整段复制）：

```
chmod +x /data/home/<你的学号>/project/code-server-4.137.0-linux-arm64/bin/code-server && /data/home/<你的学号>/project/code-server-4.137.0-linux-arm64/bin/code-server --auth none
```

（前提：`code-server-4.137.0-linux-arm64.tar.gz` 上传后在服务器内解压，并将解压得到的 `code-server-4.137.0-linux-arm64` 文件夹移动到 `/data/home/<你的学号>/project/` 目录下；`--auth none` 免密码；平台会自动追加 `--bind-addr` 参数，code-server 正好认。）

提交 → RUNNING → 点"连接" → VSCode 网页界面打开。

## 第 4 步：容器内初始化（每个新容器 ~5 分钟）

VSCode 里开终端（Ctrl+`），⚠️ **不要 source 任何 set_env.sh / env.sh**（镜像原生 CANN 环境已正确，叠加必坏——实测复现）：

```bash
# 1) 原生栈验证（应输出 2.5.1 True 0.9.1rc1）
python3 -c "import torch, torch_npu, vllm; print(torch.__version__, torch.npu.is_available(), vllm.__version__)"

# 2) 持久 venv（建在挂载目录，跨作业/跨容器永驻；--system-site-packages 复用镜像的 torch）
python3 -m venv --system-site-packages /data/home/<你的学号>/project/envs/verl_env
source /data/home/<你的学号>/project/envs/verl_env/bin/activate
# ⚠️ 网络存储盘装包必须禁用 pyc 编译，同时安装 verl 与沙箱判分必须的 pytest
export PIP_NO_COMPILE=1
pip install verl==0.6.1 pytest
python3 -c "import verl, pytest; print('verl & pytest OK')"
```

## 第 5 步：数据与奖励文件（1 分钟确认 + 缺了按 SOP 第 3 节补）

```bash
ls /data/home/<你的学号>/project/data/gsm8k/     # 应有 train.parquet / train_300.parquet / test.parquet
ls /data/home/<你的学号>/project/rewards/smoke_gsm8k.py 2>/dev/null || echo "需新建（内容见平台操作SOP 第 3 节）"
```

## 第 6 步：跑 PPO 冒烟（10~20 分钟）

```bash
cd /data/home/<你的学号>/project
bash smoke_ppo_05b.sh \
    custom_reward_function.path=/data/home/<你的学号>/project/rewards/smoke_gsm8k.py \
    2>&1 | tee smoke.log
```

**成功标志**（按出现顺序）：配置 dump → `All configuration checks passed successfully!` → Critic/Actor 加载进 NPU → 验证生成 → **`step:0` 出现且 reward 指标滚动** → 4 步跑完回提示符。

**实测基线**（0.5B 冒烟，供对照）：每步 23~27 秒；rewards/mean 0.0156（64 条里 1 条对）、max 1.0；response clip_ratio 0.64~0.69（2/3 撞 256 上限——正式训练提到 512+）；MFU 0.06~0.075。

**退出时的 TBE ERROR 噪音**：无害（CANN 编译子进程在主进程退出时的清理抱怨）。判别法：**结尾出现 = 无害；训练中段出现且进程消失 = 真故障**。

---

## 第 7 步：D2 代码任务数据清洗与难度分级（8 卡全流程实测通过）

如果进入正式的代码 RLVR 任务（取代 GSM8K 玩具数据），使用本套已经过全量 1.5 万条实跑验证的数据清洗管线：

```bash
cd /data/home/<你的学号>/project

# 1. 回归自检：运行沙箱与提取器回归单测（115 项全边界测试，100% 全部通过）
python -m pytest pipeline/tests/ -v

# 2. 筛 1：Prompt 规范化与字段瘦身（15,000 条全入库，约 10 秒）
python pipeline/f1_template.py \
    --input data/kodcode_candidates.jsonl \
    --output data/step1_templated.jsonl

# 3. 筛 2：官方解答过沙箱自洽性校验（务必开启 16 核并发，28 分钟全量跑完）
# 实测基准：读入 15000 / 保留 13871 / 淘汰 1129 / 通过率 92.5% / 平均耗时 1.875s
nohup python pipeline/f2_verify.py \
    --input data/step1_templated.jsonl \
    --output data/step2_kept.jsonl \
    --rejects data/step2_rejects.jsonl \
    --workers 16 > f2_verify.log 2>&1 &
# 实时看进度：tail -f f2_verify.log

# 4. 筛 3：去重 + 截长(>2000字符) + 泄漏过滤(>0.9)（纯 CPU 计算，3 秒搞定）
# 实测基准：读入 13871 / 保留 10434 / 淘汰 3437 (过长 3407，泄漏 30)
python pipeline/f3_dedup.py \
    --input data/step2_kept.jsonl \
    --output data/step3_verified_pool.jsonl \
    --rejects data/step3_rejects.jsonl

# 5. 筛 4：7B 基座 8 卡并行预采样与难度分级（产出最终 rl_pool.jsonl + heldout.jsonl）
# 推荐先用 16 题极速冒烟验证（每卡 2 题，约 20~30 秒测通）：
python pipeline/f4_sample_filter.py \
    --input data/step3_verified_pool.jsonl \
    --rl_pool data/test_rl_pool.jsonl \
    --heldout data/test_heldout.jsonl \
    --model ../Qwen2.5-7B-Instruct \
    --gpus 8 \
    --max_problems 16

# 冒烟通过后，启动全量 8 卡并行正式运行（10,434 题，8 卡总吞吐 ~1800 tok/s，约 40~60 分钟）：
nohup python pipeline/f4_sample_filter.py \
    --input data/step3_verified_pool.jsonl \
    --rl_pool data/rl_pool.jsonl \
    --heldout data/heldout.jsonl \
    --rejects data/step4_rejects.jsonl \
    --model ../Qwen2.5-7B-Instruct \
    --gpus 8 \
    --batch_size 50 \
    --gpu_memory_utilization 0.6 > f4_filter.log 2>&1 &
# 实时监控：tail -f f4_filter.log
```

产出文件：
- `data/rl_pool.jsonl`：黄金难度强化学习训练池（2,000 ~ 4,000 条，通过率 10%~90%）；
- `data/heldout.jsonl`：严格隔离的独立评测集（200 条，绝不进入训练池）。

---

## 日常循环：配置固化后，每个新作业只剩 3 分钟网页操作

第 1~6 步全部是**一次性**的：venv（含 verl）、脚本、数据、模型都在挂载目录里，跨作业永驻。新容器蒸发的是容器层的东西——而我们要的全在挂载里，所以**第 4 步（建 venv + 装 verl）以后永远不用重做**。之后每个新作业只有两种模式：

| 事项 | 模式 A：无人值守训练 ⭐ | 模式 B：交互开发 |
|---|---|---|
| 适用 | 过夜长训、正式训练 | 调试、跑单卡测试、看现场 |
| 运行命令填 | `bash /data/home/<你的学号>/project/start_train.sh` | 第 3 步的 code-server 命令 |
| 电脑能否关 | 随时关，作业照跑 | 关电脑 = 终端断（长命令改用 nohup，坑表 #11） |
| 看进度 | 点"连接"进 VSCode `tail -f project/train_*.log`（作业详情日志页也行） | 直接终端里看 |

**start_train.sh 的四个动作**（文件已随 project 分发）：① 平台追加的 `--bind-addr` 原样转交 code-server，"连接"按钮照常可用；② 激活持久 venv；③ 后台跑 `TRAIN_CMD`（**换训练内容只改脚本里这一行**），日志落盘 `project/train_月日_时分.log`；④ 训练结束后 `tail` 保活容器，随时连进去收 checkpoint——**收完记得手动停掉作业**（保活会一直算机时）。checkpoint 目录记得在训练脚本里开 `trainer.save_freq>0`（写挂载目录）。

**每个新作业 10 秒自检**（可选）：

```bash
python3 -c "import torch,torch_npu,vllm,verl;print('OK',torch.npu.is_available())"   # 输出 OK True 即环境完好
```

**两条纪律**：① **别换镜像**——venv 绑定镜像里的 torch/CANN，换镜像版本 = 重做第 4 步（5 分钟）；② **挂载点一个不能少勾**（`project` + 两个模型文件夹），少勾哪个容器里就缺哪个。

---

## 附 A：7B 单卡 A/B 验证（实测通过，2026-09-14）

用 `ab_rollout_7b.py`（project 目录内）在 1 卡 910B3 上验证 7B 推理引擎，**队友可跳过此步**（已验证过，结果如下）：

| 指标 | 实测 | 结论 |
|---|---|---|
| HF transformers 加载 7B | 51.7s | 网络盘读 15GB 可接受 |
| HF generate 16 条×256tok | 2.5s | 对照组（右 padding + 短答案早停，数字仅供参考） |
| vllm 引擎加载 | 33.2s | rollout 引擎就绪 |
| vllm 生成 16 条 | 1.2s，输出 316 tok/s | 输出为正确 Python 代码 |
| 显存 | HF 峰值 15.3GB；vllm 权重 14.25GB + 2327 KV blocks | 单卡 64GB 余量大 |

**由此敲定的正式训练配置**：rollout `tensor_model_parallel_size=1`（单卡装得下 7B，无需 TP=2）；vllm `gpu_memory_utilization` 0.4~0.5 即可（0.6 实测已很宽裕）；FSDP 分摊后每卡总占用预计 ~30GB，64GB 富余。

### 8 卡冒烟两连（正式训练当天先跑，约 40 分钟，与数据集无关）

多卡验证的是管道（Ray 8 worker / FSDP 分片 / hccl / 8 个 vllm 引擎），喂 GSM8K 冒烟数据即可，不需要正式数据集：

⚠️ **跑之前必须先打 vllm-ascend 多卡补丁**（2026-09-14 8 卡实测踩坑：0.9.1rc1 按"引擎独占世界"建 EP/ETP 通信组 `[0]`，与 verl 多 worker 全局域冲突，rank>0 的 worker 必挂 `AssertionError: assert self.cpu_group is not None`；单卡永远不触发）。补丁改的是容器层文件，**每个新作业开头都要跑一次**（start_train.sh 已内置；幂等可重复）：

```bash
python3 /data/home/<你的学号>/project/patch_vllm_ascend.py   # 输出 PATCHED OK / already patched
```

若补丁后仍报 vllm-ascend 相关错误，备选方案 = 引擎升级为 TP=8（通信组自然覆盖全部 rank，不需要补丁）：

```bash
bash smoke_ppo_05b.sh trainer.n_gpus_per_node=8 actor_rollout_ref.rollout.tensor_model_parallel_size=8
```

```bash
cd /data/home/<你的学号>/project
# 从已验证的 0.5B 脚本派生 7B 版（改模型名/实验名/微批三处）
sed -e 's|Qwen2.5-0.5B-Instruct|Qwen2.5-7B-Instruct|' \
    -e 's|smoke_05b_npu|smoke_7b_8c|' \
    -e 's|ppo_micro_batch_size_per_gpu=8|ppo_micro_batch_size_per_gpu=4|g' \
    -e 's|log_prob_micro_batch_size_per_gpu=16|log_prob_micro_batch_size_per_gpu=8|' \
    smoke_ppo_05b.sh > smoke_ppo_7b.sh

# 测试1：0.5B × 8 卡（~10 分钟，纯拓扑验证）
bash smoke_ppo_05b.sh trainer.n_gpus_per_node=8 2>&1 | tee smoke_8c_05b.log
# 测试2：7B × 8 卡（~30 分钟，正式训练彩排）
bash smoke_ppo_7b.sh trainer.n_gpus_per_node=8 2>&1 | tee smoke_8c_7b.log
```

成功标志：日志出现 world size 8 / 8 个 worker → 8 个 vllm 引擎初始化 → `step:0` 奖励滚动。多卡专属故障对照：hccl 初始化超时 → `ls /dev/davinci*` 确认 8 卡可见；OOM → 微批降到 2 或 `gpu_memory_utilization=0.4`；另开终端 `npu-smi info` 看每卡占用（预期 ~30GB）。

---

## 附 B：已知坑与修复对照表（全部实测踩过）

| # | 坑 | 症状 | 修复 |
|---|---|---|---|
| 1 | 注册镜像填了用户名密码 | 拉取报 401 Unauthorized | 清空两栏，匿名拉取 |
| 2 | 新镜像没有 code-server | 运行命令 code-server → command not found 秒挂 | 运行命令指向挂载里的 code-server 全路径（第 3 步） |
| 3 | 平台给运行命令追加 --bind-addr | tail/sleep 被追加参数噎死 | 用 code-server（认这个参数）或忽略参数的脚本 |
| 4 | 平台解压功能按 tar 文件名建目录 | 解压产物藏在 `xxx.tar/` 这层怪目录里 | 解压后 mv 到干净路径 |
| 5 | pip 装包 pyc 断言失败（AssertionError pyc_path） | conda/venv 在 dpc 网络盘上 | `export PIP_NO_COMPILE=1` |
| 6 | 首次 import torch 卡几分钟 | 字节码缓存首次写入 | 耐心等完（或 compileall 预编译），**别 Ctrl+C** |
| 7 | pip 装的包随容器蒸发 | 新作业里 verl 不见了 | 每次作业开头重装（1 分钟）或用持久 venv |
| 8 | source 挂载里的 CANN set_env.sh | 所有算子 ADD_TO_LAUNCHER_LIST 失败 | **永远不要 source**——镜像原生 CANN 就是正确版本 |
| 9 | GitHub clone 抽风 | curl 16 / HTTP/2 stream 1 was not closed cleanly | `git config --global http.version HTTP/1.1`；或使用国内高速镜像 `https://ghfast.top/https://github.com/...` / `https://ghfast.top/https://raw.githubusercontent.com/...` |
| 10 | HF 下载 401/失败 | hf_transfer/Xet 协议走不了镜像 | `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1` |
| 11 | 关电脑/断网后终端里的训练死了 | code-server 断开回收终端，前台进程被终止 | 后台跑：`nohup bash xx.sh > xx.log 2>&1 &`（有 tmux 更稳）；过夜正式跑直接写进作业「运行命令」（start_smoke.sh 模式）+ `trainer.save_freq>0` 存挂载目录 |
| 12 | 筛 2 单线程验证 1.5 万条耗时数小时 | 单核逐题过沙箱 pytest 吞吐低 | 启用 `--workers 16` 多线程并发，利用内置滑动窗口在 28 分钟内全量跑完 |
| 13 | 沙箱判分报找不到 pytest | 虚拟环境只装了 verl，漏装测试驱动器 | `export PIP_NO_COMPILE=1; pip install pytest` |
| 14 | vLLM 实例加载报 unexpected keyword argument | 误把 verl 的 `tensor_model_parallel_size` 当做 vLLM 的初始化形参 | 原生 vLLM 参数名为 `tensor_parallel_size=1`（单卡亦可直接缺省） |
| 15 | 8 卡并行采样多进程 Queue 管道卡死 | `multiprocessing.Queue` 跨进程传输大对象填满 OS 管道 buffer | 每个 Worker 独立直写临时文件 `_tmp_f4_worker_{gpu_id}.jsonl`，主进程安全汇总 |
| 16 | 昇腾 910B 推理时 HBM 碎片 / OOM 告警 | `gpu_memory_utilization` 设为 0.85+ 挤占算子与系统保留内存 | 设为 `0.6` 即可（14.25GB 权重 + 24GB KV Cache，单卡 64GB 非常宽裕） |
