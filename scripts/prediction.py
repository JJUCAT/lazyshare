#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动股票分类预测。

功能实现在 src/prediction/predict.py（需在 tsai conda 环境运行）。

用法：
    conda activate tsai && python3 scripts/prediction.py
    conda activate tsai && python3 scripts/prediction.py [--config config/classify_train.json]
                                                        [--model models/xxx.pkl]
                                                        [--date YYYYMMDD] [-v]
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.predict import main

if __name__ == "__main__":
    sys.exit(main())
