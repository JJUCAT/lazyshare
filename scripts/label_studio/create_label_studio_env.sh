#!/usr/bin/env bash
# 创建 label-studio 的 conda 独立环境，并安装 label-studio
#
# 用法:
#   bash scripts/label_studio/create_label_studio_env.sh
#
# 说明:
#   - 环境名: label_studio
#   - Python: 3.11 (label-studio 官方支持 3.9~3.11，3.11 较稳妥)
#   - 环境已存在时只做安装/更新，不重复创建

set -euo pipefail

ENV_NAME="label_studio"
PYTHON_VERSION="3.11"

# 1. 检查 conda 是否可用
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 未找到 conda，请先安装 Miniconda 或 Anaconda。" >&2
    exit 1
fi

# 2. 若环境不存在则创建
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] conda 环境 '${ENV_NAME}' 已存在，跳过创建。"
else
    echo "[INFO] 创建 conda 环境 '${ENV_NAME}' (python=${PYTHON_VERSION}) ..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# 3. 非交互 shell 需要先 source conda 的初始化脚本再 activate
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 4. 安装 label-studio
echo "[INFO] 安装 label-studio ..."
python -m pip install --upgrade pip
python -m pip install -U label-studio

echo ""
echo "[OK] 完成！"
echo "    激活环境:  conda activate ${ENV_NAME}"
echo "    启动服务:  label-studio start"
echo "    默认地址:  http://localhost:8080 (首次启动会创建管理员账号)"
