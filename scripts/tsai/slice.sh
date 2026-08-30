#!/usr/bin/env bash
# 快捷运行数据切片：把预处理 CSV 按固定窗口切成 tsai 训练样本文件
#
# 用法:
#   bash scripts/tsai/slice.sh [--config CONFIG] [--limit N] [--workers N] [-v]
#
# 说明:
#   - 读取 config/classify_train.json（默认），或通过 --config 指定
#   - 功能实现在 src/train/slice.py
#   - 依赖 tsai conda 环境（pandas/numpy），环境缺失时提示先创建

set -euo pipefail

ENV_NAME="tsai"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY_SCRIPT="${PROJECT_ROOT}/src/train/slice.py"

# 1. 检查 conda 是否可用
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 未找到 conda，请先安装 Miniconda 或 Anaconda。" >&2
    exit 1
fi

# 2. 非交互 shell 需要先 source conda 的初始化脚本再 activate
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 3. 检查环境是否存在
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[ERROR] conda 环境 '${ENV_NAME}' 不存在，请先执行:" >&2
    echo "        bash scripts/tsai/create_tsai_env.sh" >&2
    exit 1
fi

# 4. 激活环境并运行切片脚本（透传所有参数）
conda activate "${ENV_NAME}"
exec python "${PY_SCRIPT}" "$@"
