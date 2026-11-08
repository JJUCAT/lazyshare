# -*- coding: utf-8 -*-
"""绘图相关的格式化工具。"""
from __future__ import annotations

import math

import numpy as np

AXIS_TICKS = 5  # 纵轴刻度数量

# 取值有明确固定边界的列项：纵轴直接使用该范围，不做 5% 边距放大。
# 目前无固定边界指标（W21LNV 新公式不再保证 [-1, 1]），保留扩展位。
FIXED_RANGES: dict[str, tuple[float, float]] = {}


def is_missing(value) -> bool:
    """判断是否为缺失值（None / NaN）。"""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def format_value(value) -> str:
    """按数值量级自适应格式化单个数值（用于悬浮窗等）。"""
    if is_missing(value):
        return "-"
    av = abs(value)
    if av >= 1e8:
        return f"{value / 1e8:.2f}亿"
    if av >= 1e4:
        return f"{value / 1e4:.2f}万"
    if av >= 1000:
        return f"{value:,.0f}"
    if av >= 100:
        return f"{value:,.1f}"
    if av >= 1:
        return f"{value:,.2f}"
    if av >= 0.01:
        return f"{value:,.3f}"
    if av >= 0.0001:
        return f"{value:,.4f}"
    return f"{value:.3g}"


def format_tick(value: float, step: float) -> str:
    """根据刻度步长自适应格式化刻度标签。"""
    if is_missing(value):
        return "-"
    av = abs(value)
    if av >= 1e8:
        return f"{value / 1e8:.1f}亿"
    if av >= 1e4:
        return f"{value / 1e4:.1f}万"
    s = abs(step) if step else 0.0
    if s >= 100 or av >= 1000:
        return f"{value:,.0f}"
    if s >= 1:
        return f"{value:,.1f}"
    if s >= 0.01:
        return f"{value:,.2f}"
    if s >= 0.0001:
        return f"{value:,.4f}"
    return f"{value:.3g}"


def series_range(series_list, offset: int, show: int) -> tuple[float | None, float | None]:
    """计算若干序列在可见窗口内的数据范围，无有效值返回 (None, None)。

    默认在数据范围上留 5% 边距；若序列属于 FIXED_RANGES（如 W21LNV 的
    [-1, 1]），则纵轴覆盖该固定区间且不再放大边距。
    """
    vmin = None
    vmax = None
    for s in series_list:
        arr = getattr(s, "values", None)
        if arr is None or len(arr) == 0:
            continue
        seg = np.asarray(arr[offset:offset + show], dtype=float)
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        lo = float(seg.min())
        hi = float(seg.max())
        vmin = lo if vmin is None else min(vmin, lo)
        vmax = hi if vmax is None else max(vmax, hi)
    if vmin is None or vmax is None:
        return None, None
    # 固定范围列项：轴范围至少覆盖配置区间，且不做边距放大
    fixed = False
    for s in series_list:
        bounds = FIXED_RANGES.get(getattr(s, "column", ""))
        if bounds is None:
            continue
        lo, hi = bounds
        vmin = min(vmin, lo)
        vmax = max(vmax, hi)
        fixed = True
    if fixed:
        return vmin, vmax
    if vmin == vmax:
        pad = abs(vmin) * 0.05 or 1.0
        vmin -= pad
        vmax += pad
    else:
        pad = (vmax - vmin) * 0.05
        vmin -= pad
        vmax += pad
    return vmin, vmax


def calc_ticks(vmin: float, vmax: float, count: int = AXIS_TICKS) -> list[float]:
    """返回 count 个均匀分布的刻度值。"""
    if count <= 1:
        return [vmin]
    return [vmin + (vmax - vmin) * i / (count - 1) for i in range(count)]


def short_date(full: str) -> str:
    """将 YYYY-MM-DD 缩写为 MM-DD 用于坐标轴。"""
    if len(full) >= 10:
        return full[5:10]
    return full
