# -*- coding: utf-8 -*-
"""标签计算子包：三元屏蔽法（triple barrier）标签 T / B / N。"""
from .triple_barrier import (
    BARRIER_ATR,
    BARRIER_FIXED,
    BARRIER_MODES,
    DEFAULT_PRICE_COLUMN,
    DEFAULT_VALUES,
    VALUE_KEYS,
    barrier_levels,
    compute_triple_barrier_labels,
    normalize_label_config,
)

__all__ = [
    "BARRIER_ATR",
    "BARRIER_FIXED",
    "BARRIER_MODES",
    "DEFAULT_PRICE_COLUMN",
    "DEFAULT_VALUES",
    "VALUE_KEYS",
    "barrier_levels",
    "compute_triple_barrier_labels",
    "normalize_label_config",
]
