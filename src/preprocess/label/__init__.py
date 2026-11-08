# -*- coding: utf-8 -*-
"""标签计算子包：三元屏蔽（triple barrier）标签 B / T / N，盈亏按价格百分比计算。"""
from .triple_barrier import (
    DEFAULT_VALUES,
    PRICE_KEYS,
    VALUE_KEYS,
    compute_triple_barrier_labels,
    future_price_matrix,
    label_lookahead,
    normalize_label_config,
)

__all__ = [
    "DEFAULT_VALUES",
    "PRICE_KEYS",
    "VALUE_KEYS",
    "compute_triple_barrier_labels",
    "future_price_matrix",
    "label_lookahead",
    "normalize_label_config",
]
