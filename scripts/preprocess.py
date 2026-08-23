#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动数据预处理。

功能实现在 src/preprocess 目录：
    - src/preprocess/handle ：基础工作（配置 / 指标计算 / ST 判断 / 单文件处理 / 多进程编排）
    - src/preprocess/label  ：标签计算（峰值标签 T / B）
本脚本仅负责入口启动。

用法：
    python3 scripts/preprocess.py [--config config/preprocess.json] [--limit N] [--workers N]
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.handle.processor import main

if __name__ == "__main__":
    sys.exit(main())
