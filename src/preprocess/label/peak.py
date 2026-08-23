# -*- coding: utf-8 -*-
"""标签计算：峰值标签（T / B）。

定义见 preprocess_plan.md：
    T：在 M21C 滑动窗口(k)内找出最高值那天 pt_day；
       从 pt_day 往前 k 天（含当天）收盘价最高那天是 ct_day；
       若 ct_day 前后各 k 天都是收盘价最高值，则 ct_day 标记为 T。
    B：在 M21C 滑动窗口(k)内找出最低值那天 pb_day；
       从 pb_day 往前 k 天（含当天）收盘价最低那天是 cb_day；
       若 cb_day 前后各 k 天都是收盘价最低值，则 cb_day 标记为 B。
"""
from __future__ import annotations

import numpy as np

PEAK_WINDOW = 21  # 峰值标签滑动窗口 k


def is_local_peak(arr, idx: int, k: int) -> bool:
    """idx 是否在其前后各 k 天窗口内都是最高值（允许并列最高）。"""
    lo = max(0, idx - k)
    hi = min(len(arr), idx + k + 1)
    before = arr[lo:idx + 1]
    after = arr[idx:hi]
    if before.size == 0 or after.size == 0:
        return False
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        return False
    return float(before[-1]) == float(before.max()) \
        and float(after[0]) == float(after.max())


def is_local_valley(arr, idx: int, k: int) -> bool:
    """idx 是否在其前后各 k 天窗口内都是最低值（允许并列最低）。"""
    lo = max(0, idx - k)
    hi = min(len(arr), idx + k + 1)
    before = arr[lo:idx + 1]
    after = arr[idx:hi]
    if before.size == 0 or after.size == 0:
        return False
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        return False
    return float(before[-1]) == float(before.min()) \
        and float(after[0]) == float(after.min())


def compute_peak_labels(close, m21c, k: int = PEAK_WINDOW) -> list[str]:
    """计算峰值标签列：T（Top）/ B（Bottom）/ 空。

    同一天同时命中 T 与 B 时优先标记为 T。
    """
    n = len(close)
    labels = [""] * n
    close_arr = np.asarray(close, dtype=float)
    m21_arr = np.asarray(m21c, dtype=float)
    for s in range(n - k + 1):
        seg = m21_arr[s:s + k]
        if not np.isfinite(seg).any():
            continue
        # T（Top）
        pt = s + int(np.nanargmax(seg))
        lo = max(0, pt - k + 1)
        seg_close = close_arr[lo:pt + 1]
        if np.isfinite(seg_close).any():
            ct = lo + int(np.nanargmax(seg_close))
            if is_local_peak(close_arr, ct, k):
                labels[ct] = "T"
        # B（Bottom）
        pb = s + int(np.nanargmin(seg))
        lo = max(0, pb - k + 1)
        seg_close = close_arr[lo:pb + 1]
        if np.isfinite(seg_close).any():
            cb = lo + int(np.nanargmin(seg_close))
            if is_local_valley(close_arr, cb, k):
                if labels[cb] != "T":
                    labels[cb] = "B"
    return labels
