#!/usr/bin/env bash
# 一键流程的定时入口："weekday"（每个工作日 21:00，由 launchd 触发）
#
# 用法:
#   bash scripts/weekday.sh             # 运行全流程（工作日才执行，非工作日自动跳过）
#   bash scripts/weekday.sh --dry-run   # 透传参数（预览命令）
#   bash scripts/weekday.sh --steps pull
#
# 说明:
#   - 功能实现在 src/run.py（--weekday 模式）
#   - 工作日在 src/run.py 内判断（周一至周五），周末/节假日停市时 pull 会自动跳过无新数据
#   - launchd 每天 21:00 触发本脚本，输出由 launchd 重定向到 test_output/weekday*.log

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] 未找到 python3。" >&2
    exit 1
fi

echo "===== $(date '+%F %T') weekday start ====="
exec python3 scripts/run.py --weekday "$@"
