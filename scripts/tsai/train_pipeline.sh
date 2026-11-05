#!/usr/bin/env bash
# 训练流水线：训练 → 验证/测试 → 后处理 → 汇总报告 → 发送邮件
#
# 流程（每一步失败即中断，避免把半成品报告发出去）：
#   1. python src/train/train.py          训练并写 <models>/<模型名>.train.log
#   2. python src/train/verify.py --split all
#                                         对 valid / test 分别验证并写
#                                         <models>/<模型名>.{valid,test}.log
#   3. python scripts/postprocess.py      按 M5C 过滤推理出的 B/T 信号并重新评估
#                                         写 <models>/<模型名>.{valid,test}.postprocess.log
#   4. python scripts/mail_report.py      把所有报告合并成一封邮件发送
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
#   --postprocess-config <路径>  后处理配置（默认 config/postprocess.json）
#   --preprocess-config <路径>   预处理配置（读取 postprocess.enabled 开关，
#                                 默认 config/preprocess.json）
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
POSTPROCESS_PY="${PROJECT_ROOT}/scripts/postprocess.py"
MAIL_PY="${PROJECT_ROOT}/scripts/mail_report.py"

CONFIG="${PROJECT_ROOT}/config/classify_train.json"
POSTPROCESS_CONFIG="${PROJECT_ROOT}/config/postprocess.json"
PREPROCESS_CONFIG="${PROJECT_ROOT}/config/preprocess.json"
MAIL_CONFIG="${PROJECT_ROOT}/config/mail.json"
DRY_RUN=0
NO_MAIL=0
SKIP_TRAIN=0
VERBOSE=0
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2"; shift 2 ;;
        --postprocess-config) POSTPROCESS_CONFIG="$2"; shift 2 ;;
        --preprocess-config) PREPROCESS_CONFIG="$2"; shift 2 ;;
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
    echo "==== [1/4] 训练（含训练报告） ===="
    python "${TRAIN_PY}" --config "${CONFIG}" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} ${V_FLAG[@]+"${V_FLAG[@]}"}
else
    echo "==== [1/4] 跳过训练（--skip-train） ===="
fi

# 3. 验证 + 测试
echo "==== [2/4] 验证 + 测试（含验证/测试报告） ===="
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

# 4. 后处理：按 M5C 过滤 B / T 信号并重新评估（与验证/测试同一套指标）
#    开关：预处理配置的 postprocess.enabled（缺省 true）。
#    关闭时跳过过滤，验证/测试的分类指标与推理结果保持不变。
echo "==== [3/4] 后处理（M5C 过滤 B/T 信号） ===="
PP_ENABLED=1
if [[ -f "${PREPROCESS_CONFIG}" ]]; then
    if ! python3 - "${PREPROCESS_CONFIG}" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
flag = cfg.get("postprocess", True)
enabled = flag if isinstance(flag, bool) else bool(flag.get("enabled", True))
sys.exit(0 if enabled else 1)
PY
    then
        PP_ENABLED=0
    fi
fi
if [[ ${PP_ENABLED} -eq 1 ]]; then
    python "${POSTPROCESS_PY}" --config "${POSTPROCESS_CONFIG}" \
        --preprocess-config "${PREPROCESS_CONFIG}" \
        --model "${MODELS_DIR}/${MODEL_NAME}.pkl" --split all ${V_FLAG[@]+"${V_FLAG[@]}"}
else
    echo "  已关闭（${PREPROCESS_CONFIG} 的 postprocess.enabled=false），跳过过滤"
fi

REPORTS=(
    "${MODELS_DIR}/${MODEL_NAME}.train.log"
    "${MODELS_DIR}/${MODEL_NAME}.valid.log"
    "${MODELS_DIR}/${MODEL_NAME}.test.log"
)
if [[ ${PP_ENABLED} -eq 1 ]]; then
    REPORTS+=(
        "${MODELS_DIR}/${MODEL_NAME}.valid.postprocess.log"
        "${MODELS_DIR}/${MODEL_NAME}.test.postprocess.log"
    )
fi
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
    echo "==== [4/4] 跳过邮件（--no-mail） ===="
    exit 0
fi
echo "==== [4/4] 发送报告邮件 ===="
MAIL_ARGS=(--files "${REPORTS[@]}" --config "${MAIL_CONFIG}" --title "【lazyshare】${MODEL_NAME} 训练/验证/测试/后处理报告")
if [[ ${DRY_RUN} -eq 1 ]]; then MAIL_ARGS+=(--dry-run); fi
python "${MAIL_PY}" "${MAIL_ARGS[@]}"

echo ""
echo "[OK] 流水线完成：$(( SKIP_TRAIN == 0 ? 4 : 3 )) 步（训练/验证测试/后处理/邮件）"
