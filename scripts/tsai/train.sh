#!/usr/bin/env bash
# 快捷启动 tsai 分类任务的训练 / 验证 / 评分
#
# 用法:
#   bash scripts/tsai/train.sh                          # 无参数：自动 训练→验证→评分 全流程
#   bash scripts/tsai/train.sh [train|verify|evaluate] [选项...]   # 带子命令：单独执行
#
# 子命令:
#   train    仅运行训练，透传参数给 src/train/train.py
#   verify   仅运行验证，透传参数给 src/train/verify.py
#   evaluate 仅运行评分，透传参数给 src/train/evaluate.py
#
# 示例:
#   bash scripts/tsai/train.sh                          # 全流程：训练→验证→评分
#   bash scripts/tsai/train.sh train --limit 300 --epochs 2 -v   # 仅训练（小规模调试）
#   bash scripts/tsai/train.sh verify -v                # 仅验证最新模型
#   bash scripts/tsai/train.sh verify --model models/xxx.pkl     # 验证指定模型
#   bash scripts/tsai/train.sh evaluate <results.json>  # 仅对结构化结果评分
#
# 说明:
#   - 功能实现在 src/train/train.py / verify.py / evaluate.py
#   - 依赖 tsai conda 环境，缺失时提示先创建

set -euo pipefail

ENV_NAME="tsai"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_PY="${PROJECT_ROOT}/src/train/train.py"
VERIFY_PY="${PROJECT_ROOT}/src/train/verify.py"
EVALUATE_PY="${PROJECT_ROOT}/src/train/evaluate.py"

# 1. 解析子命令（train / verify / evaluate / all，默认 all）
CMD="all"
if [[ $# -gt 0 ]]; then
    case "$1" in
        train|verify|evaluate)
            CMD="$1"
            shift ;;
        -h|--help)
            tail -n +2 "$0" | head -n 13 | sed 's/^# \{0,1\}//'
            echo ""
            echo "子命令（不带子命令时为全流程）:"
            echo "  train     仅运行训练"
            echo "  verify    仅运行验证"
            echo "  evaluate  仅运行评分（需传入 results.json）"
            echo "  （无参）   自动 训练→验证→评分"
            echo ""
            echo "透传选项示例:"
            echo "  --config 路径     指定配置文件（默认 config/classify_train.json）"
            echo "  --limit N        最多读取 N 个窗口样本（train）"
            echo "  --dirs N         最多读取 N 个源文件夹样本（train）"
            echo "  --epochs N       覆盖配置中的 epochs（train）"
            echo "  --model 路径     指定模型 pkl（verify）"
            echo "  <results.json>   结构化验证结果路径（evaluate，必填）"
            echo "  -o 路径          评分总结输出文件（evaluate，可选；默认存 results.json 同目录 summary.log）"
            echo "  -v               输出调试日志"
            exit 0 ;;
        *)
            # 未知首参：当作透传参数，按默认 train 处理
            CMD="train"
            ;;
    esac
fi

# 2. 检查 conda 是否可用
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 未找到 conda，请先安装 Miniconda 或 Anaconda。" >&2
    exit 1
fi

# 3. 非交互 shell 需要先 source conda 的初始化脚本再 activate
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# 4. 检查环境是否存在
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[ERROR] conda 环境 '${ENV_NAME}' 不存在，请先执行:" >&2
    echo "        bash scripts/tsai/create_tsai_env.sh" >&2
    exit 1
fi

# 5. 激活环境
conda activate "${ENV_NAME}"

# 6. 无参数：自动执行 训练→验证→评分 全流程
if [[ "${CMD}" == "all" ]]; then
    echo "==== [1/3] 训练 ===="
    python "${TRAIN_PY}"
    echo "==== [2/3] 验证 ===="
    python "${VERIFY_PY}"
    echo "==== [3/3] 评分 ===="
    # 取最新生成的 results.json（validation 下最新模型目录）
    LATEST_RESULTS="$(find "$(python -c "import json,sys; print(json.load(open('${PROJECT_ROOT}/config/classify_train.json'))['validation'])")" -name results.json 2>/dev/null | sort | tail -1)"
    if [[ -z "${LATEST_RESULTS}" ]]; then
        echo "[ERROR] 未找到验证生成的 results.json，评分步骤跳过。" >&2
        exit 1
    fi
    python "${EVALUATE_PY}" "${LATEST_RESULTS}"
    echo ""
    echo "[OK] 全流程完成！"
    exit 0
fi

# 7. 带子命令：单独执行（透传剩余参数）
case "${CMD}" in
    train)
        exec python "${TRAIN_PY}" "$@" ;;
    verify)
        exec python "${VERIFY_PY}" "$@" ;;
    evaluate)
        exec python "${EVALUATE_PY}" "$@" ;;
esac
