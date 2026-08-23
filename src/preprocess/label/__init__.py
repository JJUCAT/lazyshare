# -*- coding: utf-8 -*-
"""标签计算子包。"""
from .peak import PEAK_WINDOW, compute_peak_labels, is_local_peak, is_local_valley

__all__ = ["PEAK_WINDOW", "compute_peak_labels", "is_local_peak", "is_local_valley"]
