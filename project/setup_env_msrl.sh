#!/bin/bash
# setup_env_msrl.sh — MindSpeed-RL 备选环境安装脚本（幂等，可重复执行）
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(cd "$CURRENT_DIR/.." && pwd)
if [[ "$PARENT_DIR" =~ ^/data/home/[^/]+$ ]]; then
    DEFAULT_HOME="$PARENT_DIR"
else
    DEFAULT_HOME="/data/home/<你的学号>"
fi
HOME_DIR="${HOME_DIR:-$DEFAULT_HOME}"
PERSIST="$HOME_DIR/project"

ENV=$PERSIST/envs/msrl
CANN_DIR=$PERSIST/cann81
CANN_PKG=$PERSIST/pkg/Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run
MSRL=$PERSIST/MindSpeed-RL

source /root/software/miniconda3/etc/profile.d/conda.sh
[ -d "$ENV" ] || conda create -p "$ENV" python=3.10 -y
conda activate "$ENV"

if [ ! -d "$CANN_DIR/ascend-toolkit" ]; then
  [ -f "$CANN_PKG" ] || { echo ">>> 缺 CANN 安装包：请从昇腾官网下载 8.1.RC1 aarch64 toolkit 放到 $CANN_PKG"; exit 1; }
  chmod +x "$CANN_PKG"
  "$CANN_PKG" --install --quiet --install-path="$CANN_DIR"
fi
source "$CANN_DIR/ascend-toolkit/set_env.sh"

python - <<'EOF' || pip install torch==2.5.1
import importlib.util as u; import sys; sys.exit(0 if u.find_spec("torch") else 1)
EOF
python - <<'EOF' || pip install torch_npu==2.5.1
import importlib.util as u; import sys; sys.exit(0 if u.find_spec("torch_npu") else 1)
EOF

if [ ! -d "$MSRL" ]; then
  git clone -b v2.0.0 https://gitcode.com/ascend/MindSpeed-RL.git "$MSRL" \
    || git clone -b 2.0.0 https://gitcode.com/ascend/MindSpeed-RL.git "$MSRL" \
    || git clone https://gitcode.com/ascend/MindSpeed-RL.git "$MSRL"
fi
cd "$MSRL"
git checkout v2.0.0 2>/dev/null || git checkout 2.0.0 2>/dev/null || true
pip install -r requirements.txt
pip install -e .
