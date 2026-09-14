#!/bin/bash
# prepare_gsm8k.sh — 一键准备冒烟用的 GSM8K 数据集（train.parquet / train_300.parquet / test.parquet）
# 适配：自动激活持久虚拟环境并解析用户根目录

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(cd "$CURRENT_DIR/.." && pwd)
if [[ "$PARENT_DIR" =~ ^/data/home/[^/]+$ ]]; then
    DEFAULT_HOME="$PARENT_DIR"
else
    DEFAULT_HOME="/data/home/<你的学号>"
fi
HOME_DIR="${HOME_DIR:-$DEFAULT_HOME}"

# 自动激活持久虚拟环境
if [ -f "$HOME_DIR/project/envs/verl_env/bin/activate" ]; then
    source "$HOME_DIR/project/envs/verl_env/bin/activate"
fi

DATA_DIR="$HOME_DIR/project/data/gsm8k"
mkdir -p "$DATA_DIR"

echo ">>> [1/2] 正在通过 verl 下载并预处理 GSM8K 数据集到 $DATA_DIR ..."
python3 -m verl.utils.dataset.gsm8k --local_dir "$DATA_DIR"

if [ -f "$DATA_DIR/train.parquet" ]; then
    echo ">>> [2/2] 正在裁剪快速冒烟集 train_300.parquet (前 300 条) ..."
    python3 -c "import pandas as pd; df = pd.read_parquet('$DATA_DIR/train.parquet'); df.iloc[:300].to_parquet('$DATA_DIR/train_300.parquet'); print('>>> 裁剪成功，行数:', len(df.iloc[:300]))"
    echo ">>> [OK] GSM8K 数据集准备完毕！"
    ls -lh "$DATA_DIR"
else
    echo ">>> [ERROR] 未能在 $DATA_DIR 下找到 train.parquet，请确认容器网络连接及 verl 安装正常。"
fi
