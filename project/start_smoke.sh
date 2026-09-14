#!/bin/bash
# start_smoke.sh — 平台"运行命令"指向此脚本；自动忽略平台追加的所有参数
# 职责：① 补装/检查 verl ② 跑 PPO 冒烟（日志落盘到挂载目录）③ 兜底保活
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(cd "$CURRENT_DIR/.." && pwd)
if [[ "$PARENT_DIR" =~ ^/data/home/[^/]+$ ]]; then
    DEFAULT_HOME="$PARENT_DIR"
else
    DEFAULT_HOME="/data/home/<你的学号>"
fi
HOME_DIR="${HOME_DIR:-$DEFAULT_HOME}"

PROJ="$HOME_DIR/project"
LOG="$PROJ/smoke.log"

echo "[start_smoke] $(date) 开始"

# 激活持久虚拟环境
if [ -f "$HOME_DIR/project/envs/verl_env/bin/activate" ]; then
    source "$HOME_DIR/project/envs/verl_env/bin/activate"
fi

# 1) 检查并确保 verl 可用
echo "[start_smoke] 1/3 检查 verl ..."
python3 -c "import verl" 2>/dev/null || pip install -q verl==0.6.1

# 2) 跑冒烟
echo "[start_smoke] 2/3 启动 PPO 冒烟，日志写入 $LOG ..."
cd "$PROJ"
bash smoke_ppo_05b.sh \
    custom_reward_function.path="$PROJ/rewards/smoke_gsm8k.py" \
    > "$LOG" 2>&1
RC=$?
echo "[start_smoke] 冒烟结束，退出码 $RC（0=成功）"

# 3) 兜底保活
echo "[start_smoke] 3/3 进入保活模式。查看进度：tail -50 $LOG"
exec tail -f /dev/null
