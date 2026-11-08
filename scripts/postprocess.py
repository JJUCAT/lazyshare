#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动后处理：按 M5C 过滤推理出的 B / T 信号并重新评估指标。

功能实现在 src/postprocess：
    - src/postprocess/config.py      ：读取独立的 config/postprocess.json
    - src/postprocess/postprocess.py ：价格过滤 + 精度/召回/F1 评估 + 报告输出

开关（config/postprocess.json）：
    "enabled": true
每次运行都会实时读取该配置；关闭时本脚本直接跳过，不改动推理结果
results.json 与验证/测试分类指标；需要临时强制执行可加 --force。

用法：
    python3 scripts/postprocess.py [--config config/postprocess.json] [--force]
        [--model models/<模型名>.pkl] [--split valid|test|all] [--min-conf 0.9] [-o out.log]

输出：
    validation/<模型名>/<split>/postprocess.log   后处理报告
    validation/<模型名>/<split>/postprocess.json  结构化统计
    models/<模型名>.<split>.postprocess.log       模型目录副本
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.postprocess.postprocess import main

if __name__ == "__main__":
    sys.exit(main())
