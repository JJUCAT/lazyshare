#!/usr/bin/env bash
# 快捷启动金玥数据日常更新拉取（开源爬虫：requests + pandas）
#
# 用法:
#   bash scripts/pull.sh                              # 增量拉取（日期 > share 最新交易日）
#   bash scripts/pull.sh --date YYYY-MM-DD            # 仅拉取指定日期
#   bash scripts/pull.sh --dry-run                    # 预览待拉取文件，不下载
#   bash scripts/pull.sh --config config/pull.json    # 指定配置
#   bash scripts/pull.sh -v                           # 输出调试日志
#
# 说明:
#   - 功能实现在 src/pull/（config / client / pull）
#   - 检查 share 目录个股数据时间，拉取金玥数据"日常更新"（前复权）日文件
#   - 保存到 download/<年份>/<YYYY-MM-DD>_金玥数据.csv，与 scripts/update.py 兼容
#   - 接口访问按 config/pull.json 的 frequency 控制频率并加随机等待

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 使用系统 python3（含 requests / pandas）
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] 未找到 python3。" >&2
    exit 1
fi

exec python3 -m src.pull.pull "$@"
