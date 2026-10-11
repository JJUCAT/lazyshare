#!/usr/bin/env bash
# 一键流程的定时入口："weekday"（每个工作日 21:00，由 launchd 触发）
#
# 用法:
#   bash scripts/weekday.sh             # 主动执行：不检查工作日，直接运行全流程
#   bash scripts/weekday.sh --dry-run   # 透传参数（预览命令）
#   bash scripts/weekday.sh --steps pull
#   bash scripts/weekday.sh --scheduled # 定时执行：非工作日自动跳过（launchd 使用）
#
# 说明:
#   - 功能实现在 src/run.py
#   - 只有"定时执行"才检查工作日：由 launchd 传入 --scheduled（或设置环境变量
#     LAZYSHARE_SCHEDULED=1）触发；主动执行本脚本不做任何工作日检查，
#     周末/节假日也可手动跑全流程
#   - 工作日在 src/run.py 内判断（周一至周五，--weekday 模式）
#   - launchd 每天 21:00 触发本脚本（带 --scheduled），输出由 launchd 重定向到
#     test_output/weekday*.log
#
# 注意: macOS 自带 bash 为 3.2，空数组在 set -u 下展开会报错，
#       故使用 ${arr[@]+"${arr[@]}"} 的兼容写法。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] 未找到 python3。" >&2
    exit 1
fi

# 是否定时执行：环境变量或 --scheduled 参数（后者由 launchd 传入）
scheduled=0
if [[ "${LAZYSHARE_SCHEDULED:-0}" == "1" ]]; then
    scheduled=1
fi

# 分离本脚本私有参数（--scheduled）与透传给 run.py 的参数
passthru=()
for arg in "$@"; do
    case "${arg}" in
        --scheduled) scheduled=1 ;;
        *) passthru+=("${arg}") ;;
    esac
done

weekday_flag=()
if [[ "${scheduled}" == "1" ]]; then
    # 定时执行：交给 run.py 的 --weekday 模式过滤非工作日
    weekday_flag=(--weekday)
    echo "===== $(date '+%F %T') weekday(scheduled) start ====="
else
    # 主动执行：不检查工作日
    echo "===== $(date '+%F %T') weekday(manual) start ====="
fi

exec python3 scripts/run.py \
    ${weekday_flag[@]+"${weekday_flag[@]}"} \
    ${passthru[@]+"${passthru[@]}"}
