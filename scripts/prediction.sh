#!/usr/bin/env bash
# 快捷启动股票分类预测（自动激活 tsai conda 环境）
#
# 用法:
#   bash scripts/prediction.sh                            # 运行预测
#   bash scripts/prediction.sh --config 配置文件路径      # 透传参数
#   bash scripts/prediction.sh --model models/xxx.pkl     # 指定模型
#   bash scripts/prediction.sh --date YYYYMMDD            # 指定输出目录日期
#   bash scripts/prediction.sh -v                         # 输出调试日志
#
# 说明:
#   - 功能实现在 src/prediction/predict.py
#   - 推理依赖 tsai 与 torch，需在 tsai conda 环境运行；本脚本自动激活
#   - 输出: test_output/pred-<date>/dataset 与 test_output/pred-<date>/prediction.log

set -euo pipefail

ENV_NAME="tsai"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREDICT_PY="${PROJECT_ROOT}/scripts/prediction.py"

# 检查 conda 是否可用
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 未找到 conda，请先安装 Miniconda 或 Anaconda。" >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 检查环境是否存在
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[ERROR] conda 环境 '${ENV_NAME}' 不存在，请先执行:" >&2
    echo "        bash scripts/tsai/create_tsai_env.sh" >&2
    exit 1
fi

# 激活环境并运行预测（透传参数）
conda activate "${ENV_NAME}"
exec python "${PREDICT_PY}" "$@"
