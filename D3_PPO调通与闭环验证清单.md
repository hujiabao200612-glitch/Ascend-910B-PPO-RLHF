# D3 阶段任务清单：PPO 调通与全链路闭环验证 v1

> **D3 阶段核心哲学**：**绝不上来就跑全量过夜长训！** 必须先用小规模（20 steps）在真实 8 卡 910B 上完成“数据输入 → Actor 生成 → 沙箱判分打奖励 → Critic 评估 → PPO 梯度反向更新”的**最小完整闭环**，锁定显存、吞吐与 Reward 上行健康度。

---

## 0. D3 关键定位与前序交付

* **前序依赖（输入）**：
  * D1 已调通：8 卡通信补丁、环境持久 venv、0.5B GSM8K 冒烟全通；
  * D2 已交付：`data/rl_pool.jsonl`（2,000 ~ 4,000 条黄金题）+ `data/heldout.jsonl`（200 条隔离考试集）+ `score_kernel` 沙箱判分器。
* **D3 核心目标（输出）**：
  1. 数据包：`data/rlvr/train.parquet`、`data/rlvr/train_smoke.parquet` (128题)、`data/rlvr/test.parquet` (200题)；
  2. 奖励件：`rewards/code_rlvr.py`（正式代码 RLVR 奖励驱动函数，挂载判分沙箱）；
  3. 调优脚本：`run_ppo_7b_smoke.sh`（20-step 验证配置）；
  4. 成果物：20 步训练日志与 TensorBoard 曲线（验证 Reward 稳定上行、显存稳定 <45GB）。

---

## 1. D3 核心任务四步拆解

### 任务 1：数据打包与格式转换（JSONL → Parquet）

verl 框架要求输入为 Parquet 格式，包含对话 Prompt 与标准单测元数据。

* **目标**：编写 `pipeline/pack_to_parquet.py`，将 `rl_pool.jsonl` 转为 verl 标准结构：
  * `data_source`：`"kodcode"`
  * `prompt`：`[{"role": "user", "content": prompt_str}]`
  * `ability`：`"coding"`
  * `reward_model`：`{"style": "rule", "ground_truth": test_code}`
  * `extra_info`：`{"question_id": id, "test": test_code, "solution": solution_code}`
* **产出三个文件**：
  1. `data/rlvr/train_full.parquet`：全量黄金训练池；
  2. `data/rlvr/train_smoke.parquet`：抽样前 128~256 条，专门用于 D3 的 20 步极速验证；
  3. `data/rlvr/test.parquet`：200 条 heldout 独立评测集。

---

### 任务 2：正式代码 RLVR 奖励函数接入（`rewards/code_rlvr.py`）

替代此前冒烟用的玩具正则 `rewards/smoke_gsm8k.py`，接入真正的可验证代码沙箱。

* **接口对齐 verl 规范**：
  ```python
  def compute_score(solution_str, ground_truth, extra_info=None):
      # 1. 抽代码：从模型输出（含思考过程）中提取 Python 代码
      # 2. 过沙箱：调用 score_kernel 跑 pytest
      # 3. 算奖励：
      #    - 格式分 (format_score)：成功提取代码块 +0.1，未提取 0.0
      #    - 准确分 (acc_score)：按 passed / total 给予 0.0 ~ 1.0 的平滑分（或阶梯分）
      #    - 超时惩罚：timed_out 扣减 0.2
      # 返回 final_reward
  ```
* **单测先行**：用已有的 115 项单测对齐奖励返回值，确保对合法代码打高分、对死循环/语法错打 0 分。

---

### 任务 3：7B × 8 卡 PPO 小规模 20-step 闭环验证

在 8 卡 Ascend 910B 上启动小规模冒烟训练，验证整个分布式计算链路的稳定性。

* **运行脚本**：`project/smoke_ppo_7b_rlvr.sh`
* **关键配置参数**：
  * `trainer.n_gpus_per_node=8`
  * `actor_rollout_ref.model.path="../Qwen2.5-7B-Instruct"`
  * `data.train_files="['data/rlvr/train_smoke.parquet']"`
  * `data.val_files="['data/rlvr/test.parquet']"`
  * `trainer.total_steps=20`
  * `data.train_batch_size=64`（每步 64 道题）
  * `actor_rollout_ref.actor.ppo_mini_batch_size=64`
  * `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4`
  * `actor_rollout_ref.rollout.gpu_memory_utilization=0.5`
  * `data.max_response_length=768`（必须支持代码长思考）
* **五大健康检查门槛（验收通过标准）**：
  1. ✅ **显存平稳**：单卡峰值显存稳定在 35GB ~ 45GB 之间（64GB 留有充裕安全余量，无 OOM）；
  2. ✅ **吞吐正常**：单步迭代耗时在 2 ~ 4 分钟以内；
  3. ✅ **Reward 上行**：前 5 步到后 5 步，`rewards/mean` 出现可观测的上升趋势；
  4. ✅ **KL 散度受控**：`kl_divergence` 稳定在 0.001 ~ 0.05 之间，未发生策略突变爆炸；
  5. ✅ **Checkpoint 存取正常**：在 `step:20` 顺利导出检查点文件，且断点能成功恢复。

---

### 任务 4：KL 惩罚系数 $\beta$ 短跑消融（顺手完成）

* **目的**：检验模型在强化学习中的探索自由度。
* **对照组**：
  * 组 1：$\beta = 0.001$（标准约束，防止胡言乱语）；
  * 组 2：$\beta = 0.0005$（低约束，给予模型更激进的代码探索空间）。
* **记录指标**：观察哪一组的策略熵（Entropy）与测试通过率提升更健康，以此定稿 D4 主训练的最佳超参。

---

## 2. D3 验收交付清单（收工标准）

- [ ] `data/rlvr/train_smoke.parquet` 与 `test.parquet` 转换生成完毕；
- [ ] `rewards/code_rlvr.py` 代码奖励函数实现完毕，本地 10 条样本测试通过；
- [ ] 7B × 8 卡 PPO 20-step 完整跑通，无进程挂死、无通信超时；
- [ ] 导出 20-step 训练日志与监控曲线，确认 Reward 上升趋势；
- [ ] 确认 Checkpoint 正常生成并可载入。

---

## 3. D3 常见风险与即时对策

| 故障现象 | 根因排查 | 解决方案 |
|---|---|---|
| **单步显存 OOM** | PPO Actor 与 Critic 共卡，反向传播激活值累积 | 将 `ppo_micro_batch_size_per_gpu` 从 4 降到 2，或开启梯度检查点 `gradient_checkpointing=True` |
| **Reward 均值持续为 0** | 采样输出未能被正确提取或沙箱路径配置错误 | 检查 `code_rlvr.py` 的提取正则，打印第一步采样的 5 条模型输出人工复核 |
| **KL 散度急速飙升 (>1.0)** | 学习率过大导致策略剧烈偏移基座模型 | 将 Actor 学习率从 `1e-6` 下调至 `5e-7`，或将 KL 系数 $\beta$ 临时翻倍 |
| **生成回复被截断 (clip_ratio 高)** | 模型思考过程过长，撞上 256 token 限制 | 将 `data.max_response_length` 调大至 768 或 1024 |
