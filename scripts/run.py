#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动一键运行（拉取 → 更新 → 预处理 → 预测）。

功能实现在 src/run.py（编排器，复用现有功能接口）。

用法：
    python3 scripts/run.py                              # 全流程
    python3 scripts/run.py --steps pull,update          # 仅拉取 + 更新
    python3 scripts/run.py --dry-run                    # 预览命令，不执行
    python3 scripts/run.py -v                           # 调试日志
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run import main

if __name__ == "__main__":
    sys.exit(main())
