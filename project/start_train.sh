#!/bin/bash
# start_train.sh — 无人值守训练入口（平台建作业时"运行命令"就填这一行）
#   bash /data/home/<你的学号>/project/start_train.sh
# 它做五件事：
#   ① 把平台追加的 --bind-addr 原样转交 code-server → 作业"连接"按钮照常可用
#   ② 打 vllm-ascend 多卡补丁（容器层文件，每个新作业都要；幂等）
#   ③ 激活持久 venv（verl 等全在里面，新容器免安装）
#   ④ 后台跑 TRAIN_CMD，日志落盘 project/train_月日_时分.log
#   ⑤ 训练结束后保活容器（tail），随时连进去看 checkpoint / 日志

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(cd "$CURRENT_DIR/.." && pwd)
if [[ "$PARENT_DIR" =~ ^/data/home/[^/]+$ ]]; then
    DEFAULT_HOME="$PARENT_DIR"
else
    DEFAULT_HOME="/data/home/<你的学号>"
fi
HOME_DIR="${HOME_DIR:-$DEFAULT_HOME}"

# 启动 code-server 保活（可远程网页连接 VSCode）
if [ -f "$HOME_DIR/project/code-server-4.137.0-linux-arm64/bin/code-server" ]; then
    chmod +x "$HOME_DIR/project/code-server-4.137.0-linux-arm64/bin/code-server"
    "$HOME_DIR/project/code-server-4.137.0-linux-arm64/bin/code-server" --auth none "$@" >/dev/null 2>&1 &
fi

# 打多卡补丁
python3 "$HOME_DIR/project/patch_vllm_ascend.py"

# 激活持久虚拟环境
if [ -f "$HOME_DIR/project/envs/verl_env/bin/activate" ]; then
    source "$HOME_DIR/project/envs/verl_env/bin/activate"
fi

cd "$HOME_DIR/project"
TRAIN_CMD="bash run_ppo_7b_full.sh trainer.n_gpus_per_node=8"
LOG=train_$(date +%m%d_%H%M).log
echo "[start_train] 开始: $TRAIN_CMD （日志: $LOG）"
$TRAIN_CMD > "$LOG" 2>&1
echo "[start_train] 结束 exit=$? （日志: $LOG）"
exec tail -f /dev/null
