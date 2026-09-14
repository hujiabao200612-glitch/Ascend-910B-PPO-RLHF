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
├── 北理工智算集群用户指南.docx       # 智算平台使用说明手册
└── project/                       # 核心执行脚本与代码
    ├── patch_vllm_ascend.py       # vllm-ascend 0.9.1rc1 多卡通信组修复补丁
    ├── smoke_ppo_05b.sh           # 0.5B PPO 冒烟测试脚本（单卡/多卡均可）
    ├── smoke_ppo_7b.sh            # 7B 8卡 PPO 训练验证脚本（微批自适应）
    ├── start_train.sh             # 无人值守正式训练入口（含保活与 code-server）
    ├── start_smoke.sh             # 自动化冒烟测试与保活
    ├── setup_env_msrl.sh          # MindSpeed-RL 备选环境安装脚本
    ├── mbpp_sanitized.jsonl       # 冒烟验证用小型测试集
    └── rewards/
        └── smoke_gsm8k.py         # 兼容 verl 0.6.1 签名的规则打分 Wrapper
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

### 第 2 步：创建开发容器
* 镜像：选择上一步拉取的镜像；
* 算力：开发/冒烟选 1 卡，正式训练选 8 卡（910B3）；
* 挂载点：勾选 `project` 目录以及模型目录（`Qwen2.5-7B-Instruct` / `Qwen2.5-0.5B-Instruct`）；
* 运行命令：
  ```bash
  chmod +x /data/home/<你的学号>/project/code-server-4.137.0-linux-arm64/bin/code-server && /data/home/<你的学号>/project/code-server-4.137.0-linux-arm64/bin/code-server --auth none
  ```

### 第 3 步：初始化环境（仅首次需建 venv，后续跨容器永久复用）
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
```

> **提示**：建议在 `~/.bashrc` 中写入：
> ```bash
> echo 'source /data/home/<你的学号>/project/envs/verl_env/bin/activate' >> ~/.bashrc
> ```
> 这样每次新开终端都会自动进入正确的 Python 环境。

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

## ⚠️ 常见避坑指南（踩坑总结）

| # | 踩坑现象 | 根本原因 | 正确应对 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'verl'` | 新开终端未激活持久 venv，使用了镜像默认 Python | 运行 `source /data/home/<学号>/project/envs/verl_env/bin/activate` 或使用本仓库脚本（内置自动激活） |
| 2 | `AssertionError: assert self.cpu_group is not None` | vllm-ascend 0.9.1rc1 通信组只分配了 rank 0 | 执行 `python3 patch_vllm_ascend.py` 打入多 worker 通信组补丁 |
| 3 | 算子执行报 `ADD_TO_LAUNCHER_LIST` 失败 | 手动 source 了外置的 CANN `set_env.sh` 导致环境冲突 | **严禁手动 source 任何 CANN 环境变量**，镜像自带的 8.1.rc1 已经就绪 |
| 4 | pip 报错 `AssertionError: pyc_path` | dpc 网络文件系统与 pyc 预编译冲突 | 安装前执行 `export PIP_NO_COMPILE=1` |
| 5 | 断网 / 关闭网页终端导致训练中断 | 终端 session 被杀死 | 使用无人值守入口 `start_train.sh` 或后台运行 `nohup ... &` |

---

## 👥 协作与使用说明
* 欢迎协作团队与班级同学基于此模板克隆并配置自己的训练任务；
* 运行前将 `<你的学号>` 替换为各自平台的主目录路径即可无缝复用。
