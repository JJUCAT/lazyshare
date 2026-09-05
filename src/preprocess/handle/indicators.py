# -*- coding: utf-8 -*-
"""基础工作：列定义、指标计算、ST 判断、文件名清洗。"""
from __future__ import annotations

import pandas as pd

# 计算窗口（含当日）
CLOSE_WINDOW = 21    # M21C：21 日收盘价均值
VOLUME_WINDOW = 21   # M21V：21 日成交量均值

# 原始数据列名
COL_DATE = "日期"
COL_CODE = "代码"
COL_NAME = "名称"
COL_INDUSTRY = "所属行业"
COL_CLOSE = "收盘价"
COL_HIGH = "最高价"
COL_LOW = "最低价"
COL_VOLUME = "成交量（股）"
COL_ST = "是否ST"

# 输出列
OUTPUT_COLUMNS = [
    "日期",
    "股票代码",
    "股票名称",
    "行业",
    "收盘价",
    "成交量",
    "M21C",
    "M21V",
    "NC",
    "NV",
    "NA",
    "NBear",
    "NBull",
    "SNC",
    "SNV",
    "SNB",
    "M21SNB",
    "IMV",
    "SIV",
    "峰值标签",
]

# 需要保留 3 位小数的浮点列
FLOAT_COLUMNS = [
    "收盘价",
    "成交量",
    "M21C",
    "M21V",
    "NC",
    "NV",
    "NA",
    "NBear",
    "NBull",
    "SNC",
    "SNV",
    "SNB",
    "M21SNB",
    "IMV",
    "SIV",
]

# 文件名中不允许的字符（替换为下划线）
_FILENAME_UNSAFE = set('/\\:*?"<>|')


def sanitize_filename(name: str) -> str:
    """将股票名称清洗为安全的文件名（去除首尾/连续空白、替换非法字符）。"""
    name = str(name).strip()
    name = "".join("_" if c in _FILENAME_UNSAFE else c for c in name)
    # 合并连续空白（包括全角空格），避免文件名中出现多余空格
    return "".join(name.split())


def is_st_stock(df) -> bool:
    """判断是否为 ST 股票（名称含 ST，或最新交易日是否ST=是）。"""
    name = str(df[COL_NAME].iloc[-1]).upper()
    if "ST" in name:
        return True
    if COL_ST in df.columns and len(df):
        if str(df[COL_ST].iloc[-1]).strip() == "是":
            return True
    return False


def compute_indicators(df) -> dict:
    """计算各项指标序列（收盘/最高/最低/成交量及其均值、归一化、累计）。

    返回 dict，键为：close / high / low / volume / m21c / m21v /
    nc / nv / na / nbear / nbull / snc / snv / snb / m21snb。
    """
    close = pd.to_numeric(df[COL_CLOSE], errors="coerce")
    high = pd.to_numeric(df[COL_HIGH], errors="coerce")
    low = pd.to_numeric(df[COL_LOW], errors="coerce")
    volume = pd.to_numeric(df[COL_VOLUME], errors="coerce")

    # 移动均值：窗口含当日，天数不足时为 NaN
    m21c = close.rolling(window=CLOSE_WINDOW, min_periods=CLOSE_WINDOW).mean()
    m21v = volume.rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean()

    # 归一化值（以收盘价为基准）
    nc = (close - m21c) / m21c
    nv = (volume - m21v) / m21v
    na = (high - low) / close
    nbear = (high - close) / close
    nbull = (close - low) / close

    # 累计值
    snc = nc.cumsum()
    snv = nv.cumsum()
    snb = (nbull - nbear).cumsum()

    # SNB 的 21 日均值
    m21snb = snb.rolling(window=CLOSE_WINDOW, min_periods=CLOSE_WINDOW).mean()

    return {
        "close": close, "high": high, "low": low, "volume": volume,
        "m21c": m21c, "m21v": m21v,
        "nc": nc, "nv": nv, "na": na,
        "nbear": nbear, "nbull": nbull,
        "snc": snc, "snv": snv, "snb": snb, "m21snb": m21snb,
    }
