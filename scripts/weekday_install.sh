#!/usr/bin/env bash
# 安装 launchd 服务："weekday" 每个工作日 21:00 运行 scripts/weekday.sh
#
# 用法:
#   bash scripts/weekday_install.sh      # 安装
#   bash scripts/weekday_install.sh --remove  # 卸载（等价于 weekday_uninstall.sh）
#
# 说明:
#   - 生成 ~/Library/LaunchAgents/com.lazyshare.weekday.plist
#   - launchd 的 StartCalendarInterval 每天 21:00 触发，周一至周五判断在 src/run.py 的
#     --weekday 模式内完成（launchd 无法直接表达"1-5"，故触发后由 weekday 模式过滤）
#   - 卸载: launchctl unload + 删除 plist

set -euo pipefail

LABEL="com.lazyshare.weekday"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTS_DIR="${HOME}/Library/LaunchAgents"
PLIST_DEST="${AGENTS_DIR}/${LABEL}.plist"
LOG_DIR="${PROJECT_ROOT}/test_output"

if [[ "${1:-}" == "--remove" ]]; then
    launchctl unload "${PLIST_DEST}" >/dev/null 2>&1 || true
    rm -f "${PLIST_DEST}"
    echo "已卸载: ${PLIST_DEST}"
    exit 0
fi

mkdir -p "${AGENTS_DIR}" "${LOG_DIR}"

# 生成 plist（每天 21:00；工作日过滤见 src/run.py --weekday）
cat > "${PLIST_DEST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${PROJECT_ROOT}/scripts/weekday.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>21</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/weekday.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/weekday.err.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

# 重新加载（已存在时先卸载，避免 "already loaded" 报错）
launchctl unload "${PLIST_DEST}" >/dev/null 2>&1 || true
launchctl load "${PLIST_DEST}"

echo "已安装 launchd 服务: ${LABEL}"
echo "  plist : ${PLIST_DEST}"
echo "  脚本  : ${PROJECT_ROOT}/scripts/weekday.sh"
echo "  日志  : ${LOG_DIR}/weekday.log / weekday.err.log"
echo "  触发  : 每个工作日 21:00"
echo ""
echo "手动验证: bash scripts/weekday.sh --dry-run"
echo "卸载服务: bash scripts/weekday_install.sh --remove"
