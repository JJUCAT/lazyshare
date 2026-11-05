# -*- coding: utf-8 -*-
"""后处理：对 AI 推理结果做价格过滤并重新评估 B / T 信号指标。

流程（见 postprocess_plan.md）：
    1. 读取验证 / 测试结果（validation/<模型名>/<split>/results.json）；
    2. 只看推理标签为 B / T 的样本（当天模型给出的买卖信号）；
    3. 价格过滤：
          预测 B 且 当日收盘价 <  M5C → 保留
          预测 T 且 当日收盘价 >  M5C → 保留
       其余 B / T 信号视为噪声剔除；
    4. 对过滤后的 B / T 信号重新计算精度、召回率、F1 等指标
       （与验证 / 测试报告同一套口径，见 src/train/report.py）。

入口：scripts/postprocess.py
"""
from __future__ import annotations

from .config import load_postprocess_config, postprocess_enabled
from .postprocess import (
    DATE_COL,
    MA_COL,
    PRICE_COL,
    postprocess_split,
    score_samples,
)

__all__ = [
    "DATE_COL",
    "MA_COL",
    "PRICE_COL",
    "load_postprocess_config",
    "postprocess_enabled",
    "postprocess_split",
    "score_samples",
]
