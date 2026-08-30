#!/usr/bin/env bash
# 创建 tsai 的 conda 独立训练环境，并安装项目内置的 tsai 源码
#
# 用法:
#   bash scripts/tsai/create_tsai_env.sh
#
# 说明:
#   - 环境名: tsai
#   - Python: 3.11 (tsai 1.0 要求 >=3.10)
#   - 以 editable 方式安装项目内置 tsai 源码 (src/ai/tsai-1.0.1)，
#     并安装 extras 可选依赖（sktime 等，ROCKET/MiniRocket 需要）
#   - 环境已存在时只做安装/更新，不重复创建
#
# 兼容性说明:
#   tsai 1.0.1 使用 fastcore 旧版 CLI API（Param / call_parse），
#   与 fastcore 2.x 不兼容（会报 cannot import name 'Param'）。
#   因此固定 fastai==2.8.7 + fastcore==1.14.5：
#     - fastai 2.8.7 依赖 fastcore>=1.8.0（可用 1.14.5）
#     - fastai 2.8.8 依赖 fastcore>=1.14.6，但 fastcore 1.x 最高仅 1.14.5，
#       会迫使 pip 安装 fastcore 2.x，导致 tsai 无法导入
#
# 提示（NVIDIA GPU 用户）:
#   tsai 依赖 torch，macOS 上 pip 默认安装的是 MPS(Apple Silicon)/CPU 版。
#   若需在 Linux + NVIDIA GPU 上使用 CUDA，请在安装前先手动安装对应
#   CUDA 版 torch，例如:
#     pip install torch --index-url https://download.pytorch.org/whl/cu124

set -euo pipefail

ENV_NAME="tsai"
PYTHON_VERSION="3.11"
FASTAI_VERSION="2.8.7"       # tsai 1.0.1 兼容版本（勿升级到 2.8.8）
FASTCORE_VERSION="1.14.5"    # 最高 1.x，含 Param API（勿用 2.x）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TSAI_SRC="${PROJECT_ROOT}/src/ai/tsai-1.0.1"

# 1. 检查 conda 是否可用
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 未找到 conda，请先安装 Miniconda 或 Anaconda。" >&2
    exit 1
fi

# 2. 检查项目内置 tsai 源码是否存在
if [[ ! -d "${TSAI_SRC}" ]]; then
    echo "[ERROR] 未找到项目内置 tsai 源码: ${TSAI_SRC}" >&2
    exit 1
fi

# 3. 若环境不存在则创建
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] conda 环境 '${ENV_NAME}' 已存在，跳过创建。"
else
    echo "[INFO] 创建 conda 环境 '${ENV_NAME}' (python=${PYTHON_VERSION}) ..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# 4. 非交互 shell 需要先 source conda 的初始化脚本再 activate
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# 5. 安装项目内置 tsai（editable 方式，含 extras 可选依赖）
echo "[INFO] 升级 pip ..."
python -m pip install --upgrade pip
echo "[INFO] 安装项目内置 tsai (editable) ... 源码: ${TSAI_SRC}"
python -m pip install -e "${TSAI_SRC}[extras]"

# 6. 固定兼容版本：fastai==2.8.7 + fastcore==1.14.5（见顶部兼容性说明）
echo "[INFO] 固定 fastai==${FASTAI_VERSION} + fastcore==${FASTCORE_VERSION} ..."
python -m pip install "fastai==${FASTAI_VERSION}" "fastcore==${FASTCORE_VERSION}"

echo ""
echo "[OK] 完成！"
echo "    激活环境:  conda activate ${ENV_NAME}"
echo "    验证导入:  python -c \"from tsai.all import *; print('tsai', tsai.__version__)\""
