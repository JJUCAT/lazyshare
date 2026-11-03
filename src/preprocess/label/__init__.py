# -*- coding: utf-8 -*-
"""标签计算子包：简单三元屏蔽（triple barrier）标签 B / T / N，盈亏用 LU 累加。"""
from .triple_barrier import (
    DEFAULT_VALUES,
    VALUE_KEYS,
    compute_triple_barrier_labels,
    cumulative_lu,
    label_lookahead,
    normalize_label_config,
)

__all__ = [
    "DEFAULT_VALUES",
    "VALUE_KEYS",
    "compute_triple_barrier_labels",
    "cumulative_lu",
    "label_lookahead",
    "normalize_label_config",
]
