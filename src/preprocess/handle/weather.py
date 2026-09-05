# -*- coding: utf-8 -*-
"""weather.csv 加载与 IMV / SIV 计算。

weather.csv（由 scripts/update.py 生成）为宽表：
    行 = 日期，列 = 大盘 + 各行业成交量，值单位为万股。
预处理据此为每只股票逐日计算：
    IMV = 行业成交量 / 大盘成交量      # 表示行业热度
    SIV = 个股成交量 / 行业成交量      # 表示个股在行业中的热度

本模块仅依赖 pandas 与 indicators 中定义的原始数据列常量。
"""
from __future__ import annotations

import pandas as pd

from .indicators import COL_DATE, COL_INDUSTRY, COL_VOLUME

# weather.csv 列名（与 scripts/update.py 保持一致）
COL_WEATHER_DATE = "日期"
MARKET_INDUSTRY = "大盘"

# 股 -> 万股 换算系数（weather 值统一换算为“股”，便于与个股成交量相除）
SHARES_PER_WAN = 10_000.0

# weather 缺失/为空时的空查询表
EMPTY_WEATHER = {"lookup": pd.Series(dtype=float), "mkt_vol": pd.Series(dtype=float)}


def load_weather(path) -> dict:
    """读取 weather.csv（宽表），换算为“股”并摊平，返回查询表。

    weather.csv：行=日期，列=大盘+各行业（单位：万股）。
    返回 dict：
        lookup  : Series，MultiIndex(日期, 行业) -> 行业成交量（股）
        mkt_vol : Series，索引=日期，值为大盘成交量（股）
    文件不存在、为空或缺少日期列时返回 EMPTY_WEATHER。
    """
    try:
        df = pd.read_csv(path, dtype=str)
    except (FileNotFoundError, OSError, ValueError):
        return dict(EMPTY_WEATHER)
    if df is None or df.empty or COL_WEATHER_DATE not in df.columns:
        return dict(EMPTY_WEATHER)
    vol_cols = [c for c in df.columns if c != COL_WEATHER_DATE]
    if not vol_cols:
        return dict(EMPTY_WEATHER)

    df = df.set_index(COL_WEATHER_DATE)
    df.index = df.index.astype(str).str.strip()
    df = df[vol_cols].apply(pd.to_numeric, errors="coerce") * SHARES_PER_WAN

    if MARKET_INDUSTRY in df.columns:
        mkt_vol = df[MARKET_INDUSTRY]
        ind_vol = df.drop(columns=[MARKET_INDUSTRY])
    else:
        mkt_vol = pd.Series(float("nan"), index=df.index)
        ind_vol = df

    # 宽表一次摊平为 (日期, 行业) -> 成交量(股) 查询表，供逐股 reindex 使用
    lookup = ind_vol.stack()
    return {"lookup": lookup, "mkt_vol": mkt_vol}


def compute_imv_siv(df, weather: dict) -> tuple[pd.Series, pd.Series]:
    """按行对齐计算 IMV / SIV，返回与 df 行索引对齐的两个 Series。

    IMV = 行业成交量 / 大盘成交量；SIV = 个股成交量 / 行业成交量。
    df 需包含 COL_DATE / COL_INDUSTRY / COL_VOLUME 列（原始数据列名）。
    weather：load_weather 的返回值（lookup 已按宽表摊平为股，mkt_vol 为大盘股）。
    缺失日期、行业、大盘或成交量为 0 时结果为 NaN。
    """
    lookup: pd.Series = weather["lookup"]
    mkt: pd.Series = weather["mkt_vol"]
    volume = pd.to_numeric(df[COL_VOLUME], errors="coerce")
    empty = pd.Series(float("nan"), index=df.index)
    if lookup.empty or mkt.empty:
        return empty, empty

    dates = pd.Series(df[COL_DATE]).astype(str).str.strip()
    industries = pd.Series(df[COL_INDUSTRY]).astype(str).str.strip()

    # 行业成交量按 (日期, 行业) 对齐；缺失日期/行业时自然为 NaN
    key = pd.MultiIndex.from_arrays([dates.to_numpy(), industries.to_numpy()])
    ind_vol = pd.Series(lookup.reindex(key).to_numpy(), index=df.index, dtype=float)

    # 大盘成交量按 日期 对齐
    mkt_vol = pd.Series(
        mkt.reindex(dates.to_numpy()).to_numpy(), index=df.index, dtype=float
    )

    # 除零防护：成交量 0 视为缺失
    ind_safe = ind_vol.mask(ind_vol == 0)
    mkt_safe = mkt_vol.mask(mkt_vol == 0)

    imv = ind_safe / mkt_safe
    siv = volume / ind_safe
    return imv, siv
