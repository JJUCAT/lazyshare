# -*- coding: utf-8 -*-
"""基础工作：列定义、指标计算、ST/退市判断、文件名清洗。

指标定义见 preprocess_plan.md：
    LU   = (C_t - C_{t-1}) / (max_limit * C_{t-1})   涨跌幅利用率
    M21C = 21 日收盘价均值
    M21V = 21 日成交量均值
    NC   = (C - M21C) / M21C                          价格（归一化）
    NV   = (V - M21V) / M21V                          成交量（归一化）
    ATR  = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|) / C_{t-1} 单日真实波动率（归一化）
    NBull= (H - C) / C                                看多
    NBear= (C - L) / C                                看空
    TR   = 换手率
    IMV  = 行业成交量 / 大盘成交量                     行业热度
    SIV  = 个股成交量 / 行业成交量                     个股在行业中的热度
"""
from __future__ import annotations

import pandas as pd

# 计算窗口（含当日）
CLOSE_WINDOW = 21    # M21C：21 日收盘价均值
VOLUME_WINDOW = 21   # M21V：21 日成交量均值

# 跳过上市后前 N 个交易日（该区间涨跌幅不受限制，LU 无意义）
IPO_SKIP_DAYS = 5

# 浮点数最多保留的小数位数（见 preprocess_plan.md）
FLOAT_DECIMALS = 6

# 原始数据列名
COL_DATE = "日期"
COL_CODE = "代码"
COL_NAME = "名称"
COL_INDUSTRY = "所属行业"
COL_OPEN = "开盘价"
COL_CLOSE = "收盘价"
COL_HIGH = "最高价"
COL_LOW = "最低价"
COL_VOLUME = "成交量（股）"
COL_TURNOVER = "换手率"
COL_ST = "是否ST"
COL_DELIST = "退市时间"

# 涨跌幅限制（按板块/是否 ST/日期判定，见 board_max_limit）
BOARD_MAIN = "主板"
BOARD_GROWTH = "创业板"
BOARD_STAR = "科创板"
BOARD_BSE = "北交所"

# 板块默认最大涨跌幅
BOARD_LIMITS = {
    BOARD_MAIN: 0.10,
    BOARD_GROWTH: 0.20,
    BOARD_STAR: 0.20,
    BOARD_BSE: 0.30,
}
# 沪深主板 ST 股票：2026-07-06 之前为 5%，之后为 10%
ST_MAIN_LIMIT_BEFORE = 0.05
ST_LIMIT_CHANGE_DATE = "2026-07-06"

_GROWTH_PREFIXES = ("300", "301")
_STAR_PREFIXES = ("688", "689")

# 输出列（"日期" 仅作时间轴标识，不作为数值特征）
OUTPUT_COLUMNS = [
    "日期",
    "股票代码",
    "股票名称",
    "行业",
    "收盘价",
    "成交量",
    "LU",
    "M21C",
    "M21V",
    "NC",
    "NV",
    "ATR",
    "NBull",
    "NBear",
    "TR",
    "IMV",
    "SIV",
    "峰值标签",
]

# 需要保留 FLOAT_DECIMALS 位小数的浮点列
FLOAT_COLUMNS = [
    "收盘价",
    "成交量",
    "LU",
    "M21C",
    "M21V",
    "NC",
    "NV",
    "ATR",
    "NBull",
    "NBear",
    "TR",
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


def stock_board(code: str) -> str:
    """按股票代码判断所属板块（用于确定涨跌幅限制）。"""
    code = str(code).strip()
    p3 = code[:3]
    if p3 in _GROWTH_PREFIXES:
        return BOARD_GROWTH
    if p3 in _STAR_PREFIXES:
        return BOARD_STAR
    # 北交所：43/83~88 开头（新三板转板）以及 920 号段
    if code[:1] in ("4", "8") or p3 == "920":
        return BOARD_BSE
    return BOARD_MAIN


def board_max_limit(code: str, df) -> pd.Series:
    """逐日计算该股票当天的最大涨跌幅 max_limit（返回与 df 行索引对齐的 Series）。

    规则（见 preprocess_plan.md / LU）：
        - 沪深主板 10%，创业板 20%，科创板 20%，北交所 30%
        - 沪深主板 ST 股票：2026-07-06 之前 5%，之后 10%
          （创业板/科创板 ST 仍为 20%，北交所 30%）
    """
    board = stock_board(code)
    limit = pd.Series(BOARD_LIMITS[board], index=df.index, dtype=float)
    if board != BOARD_MAIN:
        return limit
    if COL_ST not in df.columns:
        return limit

    # 主板 ST 股票：按是否ST 逐日标记，并按 2026-07-06 分界切换 5% / 10%
    is_st = df[COL_ST].astype(str).str.strip() == "是"
    if not is_st.any():
        return limit
    dates = pd.to_datetime(df[COL_DATE], errors="coerce")
    cutoff = pd.Timestamp(ST_LIMIT_CHANGE_DATE)
    st_old = is_st & (dates < cutoff)
    limit[is_st] = BOARD_LIMITS[BOARD_MAIN]
    limit[st_old] = ST_MAIN_LIMIT_BEFORE
    return limit


def is_skipped_stock(df) -> bool:
    """判断股票是否跳过预处理（ST / *ST / 退市）。

    判定依据（取最新一行）：
        - 名称含 "ST"（覆盖 ST / *ST / SST / S*ST 等）
        - 名称含 "退"（退市整理期股票名称带退）
        - 是否ST 列为 "是"
        - 退市时间列有有效日期（非 "-" / 空）
    """
    if df.empty:
        return False
    name = str(df[COL_NAME].iloc[-1]).upper()
    if "ST" in name or "退" in name:
        return True
    if COL_ST in df.columns and str(df[COL_ST].iloc[-1]).strip() == "是":
        return True
    if COL_DELIST in df.columns:
        delist = str(df[COL_DELIST].iloc[-1]).strip()
        if delist and delist not in {"-", "nan", "None", "NaT"}:
            return True
    return False


def compute_indicators(df, code: str = "") -> dict:
    """计算各项指标序列（LU / 均值 / 归一化 / 波动 / 看多看空 / 换手率）。

    返回 dict，键为：close / high / low / volume / lu / m21c / m21v /
    nc / nv / atr / nbull / nbear / tr。
    """
    close = pd.to_numeric(df[COL_CLOSE], errors="coerce")
    high = pd.to_numeric(df[COL_HIGH], errors="coerce")
    low = pd.to_numeric(df[COL_LOW], errors="coerce")
    volume = pd.to_numeric(df[COL_VOLUME], errors="coerce")
    prev_close = close.shift(1)  # C_{t-1}

    # 涨跌幅利用率：涨跌幅 / 当日最大涨跌幅
    limit = board_max_limit(code, df)
    lu = (close - prev_close) / (limit * prev_close)

    # 移动均值：窗口含当日，天数不足时为 NaN
    m21c = close.rolling(window=CLOSE_WINDOW, min_periods=CLOSE_WINDOW).mean()
    m21v = volume.rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean()

    # 归一化值（以收盘价为基准）
    nc = (close - m21c) / m21c
    nv = (volume - m21v) / m21v

    # 单日真实波动率：考虑隔夜跳空
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range / prev_close

    # 看多 / 看空
    nbull = (high - close) / close
    nbear = (close - low) / close

    # 换手率（原始数据列；缺失时为空）
    if COL_TURNOVER in df.columns:
        tr = pd.to_numeric(df[COL_TURNOVER], errors="coerce")
    else:
        tr = pd.Series(float("nan"), index=df.index, dtype=float)

    return {
        "close": close, "high": high, "low": low, "volume": volume,
        "lu": lu, "m21c": m21c, "m21v": m21v,
        "nc": nc, "nv": nv, "atr": atr,
        "nbull": nbull, "nbear": nbear, "tr": tr,
    }
