#!/usr/bin/env bash
# 启动 label-studio 服务（启动成功后自动打开浏览器）
#
# 用法:
#   bash scripts/label_studio/launch_label_studio.sh [--host HOST] [--port PORT] [--no-browser]
#
# 示例:
#   bash scripts/label_studio/launch_label_studio.sh                  # http://127.0.0.1:8080
#   bash scripts/label_studio/launch_label_studio.sh --host 0.0.0.0   # 局域网可访问
#   bash scripts/label_studio/launch_label_studio.sh --port 8090
#   bash scripts/label_studio/launch_label_studio.sh --no-browser     # 不自动打开浏览器

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
# 本地文件服务根目录：直接指向预处理 CSV 目录，
# task 中的 /data/<文件名> 即映射到该目录下的文件，无需复制或符号链接
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/Users/jucat/data/ashare/preprocessed/

set -euo pipefail

ENV_NAME="label_studio"
HOST="127.0.0.1"
PORT="8080"
BROWSER_AUTO=1

# ---------- 解析参数 ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --no-browser)
            BROWSER_AUTO=0; shift ;;
        -h|--help)
            echo "用法: $0 [--host HOST] [--port PORT] [--no-browser]"
            echo "  默认: --host 127.0.0.1 --port 8080（启动后自动打开浏览器）"
            exit 0 ;;
        *)
            echo "[ERROR] 未知参数: $1（使用 --help 查看用法）" >&2
            exit 1 ;;
    esac
done

# 打开浏览器用的 URL 主机（0.0.0.0 对浏览器不可用，改用 127.0.0.1）
URL_HOST="${HOST}"
[[ "${URL_HOST}" == "0.0.0.0" || "${URL_HOST}" == "::" ]] && URL_HOST="127.0.0.1"
URL="http://${URL_HOST}:${PORT}"

# ---------- 等待服务就绪后自动打开浏览器（后台执行） ----------
open_browser_when_ready() {
    [[ "${BROWSER_AUTO}" != "1" ]] && return
    local opener=""
    if command -v open >/dev/null 2>&1; then opener="open"            # macOS
    elif command -v xdg-open >/dev/null 2>&1; then opener="xdg-open"  # Linux
    else
        echo "[INFO] 未找到 open/xdg-open，跳过自动打开浏览器，请手动访问 ${URL}"
        return
    fi
    (
        # 最多等待 60 秒，端口可访问即打开浏览器
        for _ in $(seq 1 60); do
            if curl -s -o /dev/null --max-time 1 "${URL}" 2>/dev/null; then
                "${opener}" "${URL}" >/dev/null 2>&1 || true
                echo "[INFO] 已在浏览器打开 ${URL}"
                break
            fi
            sleep 1
        done
    ) &
}

# ---------- 1. 检查 conda ----------
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 未找到 conda，请先安装 Miniconda 或 Anaconda。" >&2
    exit 1
fi

# ---------- 2. 初始化 conda（非交互 shell 必需） ----------
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ---------- 3. 检查环境是否存在 ----------
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[ERROR] conda 环境 '${ENV_NAME}' 不存在，请先执行:" >&2
    echo "        bash scripts/label_studio/create_label_studio_env.sh" >&2
    exit 1
fi

# ---------- 4. 激活环境并启动服务 ----------
conda activate "${ENV_NAME}"
echo "[INFO] 正在启动 label-studio: ${URL} (Ctrl+C 停止)"
echo "[INFO] 首次访问会引导创建管理员账号"
open_browser_when_ready
exec label-studio start --host "${HOST}" --port "${PORT}"
