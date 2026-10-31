#!/usr/bin/env bash
# 训练流水线：训练 → 验证/测试 → 汇总报告 → 发送邮件
#
# 流程（每一步失败即中断，避免把半成品报告发出去）：
#   1. python src/train/train.py          训练并写 <models>/<模型名>.train.log
#   2. python src/train/verify.py --split all
#                                         对 valid / test 分别验证并写
#                                         <models>/<模型名>.{valid,test}.log
#   3. python scripts/mail_report.py      把三份报告合并成一封邮件发送
#
# 用法:
#   bash scripts/tsai/train_pipeline.sh                       # 全流程（训练+验证+测试+邮件）
#   bash scripts/tsai/train_pipeline.sh --dry-run             # 全流程但不实际发邮件
#   bash scripts/tsai/train_pipeline.sh --no-mail             # 只训练+验证+测试，不发邮件
#   bash scripts/tsai/train_pipeline.sh --skip-train          # 跳过训练，用最新模型验证并发送
#   bash scripts/tsai/train_pipeline.sh --epochs 30 --limit 5000   # 透传训练参数
#   bash scripts/tsai/train_pipeline.sh --config /tmp/x.json  # 指定训练配置
#
# 参数:
#   --config <路径>       训练配置（默认 config/classify_train.json）
#   --mail-config <路径>  邮件配置（默认 config/mail.json）
#   --epochs/--limit/--dirs <N>   透传给 src/train/train.py
#   --dry-run             邮件只打印不发送
#   --no-mail             跳过邮件步骤
#   --skip-train          跳过训练，直接验证最新模型
#   -v                    透传给训练/验证的调试日志开关
#
# 说明:
#   - 依赖 tsai conda 环境（同 scripts/tsai/train.sh）
#   - 报告文件名以"最新一次训练"为准：按 <任务>-*.train.log 的修改时间取最新
set -euo pipefail

ENV_NAME="tsai"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_PY="${PROJECT_ROOT}/src/train/train.py"
VERIFY_PY="${PROJECT_ROOT}/src/train/verify.py"
MAIL_PY="${PROJECT_ROOT}/scripts/mail_report.py"

CONFIG="${PROJECT_ROOT}/config/classify_train.json"
MAIL_CONFIG="${PROJECT_ROOT}/config/mail.json"
DRY_RUN=0
NO_MAIL=0
SKIP_TRAIN=0
VERBOSE=0
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2"; shift 2 ;;
        --mail-config)  MAIL_CONFIG="$2"; shift 2 ;;
        --epochs|--limit|--dirs) TRAIN_ARGS+=("$1" "$2"); shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --no-mail)      NO_MAIL=1; shift ;;
        --skip-train)   SKIP_TRAIN=1; shift ;;
        -v|--verbose)   VERBOSE=1; shift ;;
        -h|--help)
            tail -n +2 "$0" | sed -n '/^#/p' | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "[ERROR] 未知参数: $1" >&2
            exit 1 ;;
    esac
done

# 1. 检查 conda 并激活 tsai 环境
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] 未找到 conda，请先安装 Miniconda 或 Anaconda。" >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[ERROR] conda 环境 '${ENV_NAME}' 不存在，请先执行:" >&2
    echo "        bash scripts/tsai/create_tsai_env.sh" >&2
    exit 1
fi
conda activate "${ENV_NAME}"
cd "${PROJECT_ROOT}"

V_FLAG=()
if [[ ${VERBOSE} -eq 1 ]]; then V_FLAG=(-v); fi

# 2. 训练
if [[ ${SKIP_TRAIN} -eq 0 ]]; then
    echo "==== [1/3] 训练（含训练报告） ===="
    python "${TRAIN_PY}" --config "${CONFIG}" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} ${V_FLAG[@]+"${V_FLAG[@]}"}
else
    echo "==== [1/3] 跳过训练（--skip-train） ===="
fi

# 3. 验证 + 测试
echo "==== [2/3] 验证 + 测试（含验证/测试报告） ===="
python "${VERIFY_PY}" --config "${CONFIG}" --split all ${V_FLAG[@]+"${V_FLAG[@]}"}

# 4. 定位最新模型的三份报告（按 .train.log 修改时间取最新）
eval "$(python3 - "${CONFIG}" <<'PY'
import json, shlex, sys
from pathlib import Path

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
label = (cfg.get("slice", {}).get("label") or ["峰值标签"])[0]
task = f"{label}_{cfg['train']['arch']}"
models = Path(cfg["models"])
logs = sorted(models.glob(f"{task}-*.train.log"),
              key=lambda p: p.stat().st_mtime, reverse=True)
name = logs[0].name[: -len(".train.log")] if logs else ""
print(f"MODELS_DIR={shlex.quote(str(models))}")
print(f"MODEL_NAME={shlex.quote(name)}")
PY
)"

if [[ -z "${MODEL_NAME}" ]]; then
    echo "[ERROR] 未找到任何 <任务>-*.train.log，无法汇总报告" >&2
    exit 1
fi

REPORTS=(
    "${MODELS_DIR}/${MODEL_NAME}.train.log"
    "${MODELS_DIR}/${MODEL_NAME}.valid.log"
    "${MODELS_DIR}/${MODEL_NAME}.test.log"
)
echo "最新模型: ${MODEL_NAME}"
for r in "${REPORTS[@]}"; do
    if [[ -f "$r" ]]; then
        echo "  报告: $r"
    else
        echo "  [WARN] 报告缺失: $r" >&2
    fi
done

# 5. 邮件发送
if [[ ${NO_MAIL} -eq 1 ]]; then
    echo "==== [3/3] 跳过邮件（--no-mail） ===="
    exit 0
fi
echo "==== [3/3] 发送报告邮件 ===="
MAIL_ARGS=(--files "${REPORTS[@]}" --config "${MAIL_CONFIG}" --title "【lazyshare】${MODEL_NAME} 训练/验证/测试报告")
if [[ ${DRY_RUN} -eq 1 ]]; then MAIL_ARGS+=(--dry-run); fi
python "${MAIL_PY}" "${MAIL_ARGS[@]}"

echo ""
echo "[OK] 流水线完成：$(( SKIP_TRAIN == 0 ? 3 : 2 )) 步（训练/验证测试/邮件）"
