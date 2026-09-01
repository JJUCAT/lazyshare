#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动预测结果邮件发送。

功能实现在 src/mail/（config / predictions / sender）。

用法：
    python3 scripts/mail.py                                 # 读取最新预测结果并发送
    python3 scripts/mail.py --date YYYYMMDD                 # 指定预测日期目录
    python3 scripts/mail.py --config config/mail.json       # 指定邮件配置
    python3 scripts/mail.py --dry-run                       # 仅打印内容，不发送
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mail.sender import main

if __name__ == "__main__":
    sys.exit(main())
