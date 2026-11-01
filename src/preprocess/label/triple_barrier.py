# -*- coding: utf-8 -*-
"""标签计算：三元屏蔽法（triple barrier）标签 B / T / N。

参数定义见 preprocess_plan.md 的 "label" 段（config/preprocess.json）：

    max_days           观察天数（向前看多少个交易日）
    price              收益计算使用的价格列名（默认 "开盘价"）
    barrier_mode       屏障计算方式："fixed"（固定百分比）/ "atr"（按波动率自适应）
    take_profit        固定止盈百分比（barrier_mode="fixed" 时使用）
    stop_loss          固定止损百分比（barrier_mode="fixed" 时使用）
    atr_window         ATR 均值窗口（barrier_mode="atr" 时使用）
    take_profit_atr    止盈 = take_profit_atr × ATR
    stop_loss_atr      止损 = stop_loss_atr × ATR
    min/max_take_profit、min/max_stop_loss  屏障的上下限（裁剪，防止极窄/极宽）
    same_day_priority  同一步同时触发止盈与止损时优先标注哪一侧
    values             标签取值 {"take_profit": "B", "stop_loss": "T", "timeout": "N"}

逐日判定规则（第 t 天）：

    入场价 = price[t]
    屏障   = (take_profit_t, stop_loss_t)
             fixed：常数 take_profit / stop_loss
             atr  ：clip(k × ATR_pct[t], min, max)，ATR_pct 为截至 t 日的
                    滚动平均单日真实波动率（只用 t 及之前的数据，无未来信息）
    依次取 k = 1..max_days，计算
        profit_k = (price[t+k] - price[t]) / price[t]
        profit_k >= take_profit_t  -> 止盈（默认 "B"，买入）
        profit_k <= -stop_loss_t   -> 止损（默认 "T"，卖出）
    先触发的一侧即为该日标签；max_days 内均未触发 -> 超时无操作（默认 "N"）。

    同一个 k 同时越过上下两条屏障（跳空等极端情形）时按 same_day_priority 决定。

无效行（价格缺失、未来 max_days 个交易日数据不完整，或 ATR 屏障缺失）标签为
空字符串 ""，由调用方丢弃，避免用不完整的前瞻窗口训练。
"""
from __future__ import annotations

import numpy as np

# 默认价格列（见 preprocess_plan.md：profit 基于开盘价计算）
DEFAULT_PRICE_COLUMN = "开盘价"

# 标签取值的三类（values 必须完整覆盖且互不相同）
VALUE_KEYS = ("take_profit", "stop_loss", "timeout")

DEFAULT_VALUES = {
    "take_profit": "B",  # 止盈 → 买入
    "stop_loss": "T",    # 止损 → 卖出
    "timeout": "N",
}

# 屏障计算方式
BARRIER_FIXED = "fixed"
BARRIER_ATR = "atr"
BARRIER_MODES = (BARRIER_FIXED, BARRIER_ATR)

# same_day_priority 允许取值
PRIORITY_TAKE_PROFIT = "take_profit"
PRIORITY_STOP_LOSS = "stop_loss"

# 无裁剪时的极端值（barrier_mode="atr" 且未配置上下限时使用）
_NO_CLIP_LOW = 0.0
_NO_CLIP_HIGH = 1.0


def normalize_label_config(raw: dict | None) -> dict:
    """校验并补全 label 配置，返回规范化 dict。

    缺省值：max_days=21, price="开盘价", barrier_mode="fixed",
    take_profit=0.15, stop_loss=0.10, atr_window=21, take_profit_atr=3.0,
    stop_loss_atr=2.0, min/max_take_profit=0.06/0.30, min/max_stop_loss=0.04/0.20,
    same_day_priority="stop_loss", values={"take_profit":"B","stop_loss":"T","timeout":"N"}。
    """
    raw = dict(raw or {})
    max_days = int(raw.get("max_days", 21))
    if max_days <= 0:
        raise ValueError(f"label.max_days 必须为正整数，当前 {max_days}")

    price = str(raw.get("price") or DEFAULT_PRICE_COLUMN).strip()
    if not price:
        price = DEFAULT_PRICE_COLUMN

    mode = str(raw.get("barrier_mode") or BARRIER_FIXED).strip().lower()
    if mode not in BARRIER_MODES:
        raise ValueError(
            f'label.barrier_mode 只支持 {" / ".join(BARRIER_MODES)}，当前 {mode!r}')

    take_profit = float(raw.get("take_profit", 0.15))
    if not take_profit > 0:
        raise ValueError(f"label.take_profit 必须为正数，当前 {take_profit}")
    stop_loss = float(raw.get("stop_loss", 0.10))
    if not stop_loss > 0:
        raise ValueError(f"label.stop_loss 必须为正数，当前 {stop_loss}")

    # ---- 波动率自适应屏障参数（barrier_mode="atr" 时使用）----
    atr_window = int(raw.get("atr_window", 21))
    if atr_window <= 0:
        raise ValueError(f"label.atr_window 必须为正整数，当前 {atr_window}")
    take_profit_atr = float(raw.get("take_profit_atr", 3.0))
    if not take_profit_atr > 0:
        raise ValueError(
            f"label.take_profit_atr 必须为正数，当前 {take_profit_atr}")
    stop_loss_atr = float(raw.get("stop_loss_atr", 2.0))
    if not stop_loss_atr > 0:
        raise ValueError(f"label.stop_loss_atr 必须为正数，当前 {stop_loss_atr}")

    min_tp = float(raw.get("min_take_profit", 0.06))
    max_tp = float(raw.get("max_take_profit", 0.30))
    min_sl = float(raw.get("min_stop_loss", 0.04))
    max_sl = float(raw.get("max_stop_loss", 0.20))
    for name, lo, hi in (("take_profit", min_tp, max_tp),
                         ("stop_loss", min_sl, max_sl)):
        if lo < 0:
            raise ValueError(f"label.min_{name} 必须 >= 0，当前 {lo}")
        if not hi > lo:
            raise ValueError(
                f"label.max_{name}({hi}) 必须大于 label.min_{name}({lo})")

    priority = str(raw.get("same_day_priority") or PRIORITY_STOP_LOSS).strip().lower()
    if priority not in (PRIORITY_TAKE_PROFIT, PRIORITY_STOP_LOSS):
        raise ValueError(
            f'label.same_day_priority 只支持 "take_profit" / "stop_loss"，当前 {priority!r}')

    values = dict(DEFAULT_VALUES)
    for k, v in (raw.get("values") or {}).items():
        k = str(k).strip()
        v = str(v).strip()
        if k in DEFAULT_VALUES and v:
            values[k] = v
    for k in VALUE_KEYS:
        if not values.get(k):
            raise ValueError(f"label.values.{k} 不能为空")
    if len({values[k] for k in VALUE_KEYS}) != len(VALUE_KEYS):
        raise ValueError(f"label.values 三类标签取值必须互不相同，当前 {values}")

    return {
        "max_days": max_days,
        "price": price,
        "barrier_mode": mode,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "atr_window": atr_window,
        "take_profit_atr": take_profit_atr,
        "stop_loss_atr": stop_loss_atr,
        "min_take_profit": min_tp,
        "max_take_profit": max_tp,
        "min_stop_loss": min_sl,
        "max_stop_loss": max_sl,
        "same_day_priority": priority,
        "values": {k: values[k] for k in VALUE_KEYS},
    }


def barrier_levels(cfg: dict, volatility=None) -> tuple[np.ndarray, np.ndarray]:
    """返回逐日的 (止盈屏障, 止损屏障) 百分比数组。

    cfg        : normalize_label_config() 的结果
    volatility : 逐日滚动 ATR（百分比，仅用当日及之前的数据），
                 barrier_mode="atr" 时必填
    """
    if cfg["barrier_mode"] != BARRIER_ATR:
        n = 0 if volatility is None else np.asarray(volatility, dtype=float).size
        shape = (n,) if n else (1,)
        return (np.full(shape, cfg["take_profit"], dtype=float),
                np.full(shape, cfg["stop_loss"], dtype=float))

    if volatility is None:
        raise ValueError('label.barrier_mode="atr" 时必须提供 volatility（滚动 ATR）')
    vol = np.asarray(volatility, dtype=float)
    take_profit = np.clip(cfg["take_profit_atr"] * vol,
                          cfg["min_take_profit"], cfg["max_take_profit"])
    stop_loss = np.clip(cfg["stop_loss_atr"] * vol,
                        cfg["min_stop_loss"], cfg["max_stop_loss"])
    return take_profit, stop_loss


def compute_triple_barrier_labels(price, cfg: dict | None = None,
                                  volatility=None) -> np.ndarray:
    """计算三元屏蔽标签，返回 object 数组（与 price 等长，无效行为 ""）。

    price      : 收益计算使用的价格序列（如开盘价）
    cfg        : label 配置（见 normalize_label_config）
    volatility : 逐日滚动 ATR（百分比），barrier_mode="atr" 时必填

    有效行范围：价格、未来 max_days 个价格均有效，且当日屏障有效；其余行
    （尤其文件末尾 max_days 行、ATR 尚未成形的早期行）返回 ""，调用方应丢弃。
    """
    cfg = normalize_label_config(cfg)
    max_days = int(cfg["max_days"])
    values = cfg["values"]

    p = np.asarray(price, dtype=float)
    n = p.size
    labels = np.full(n, "", dtype=object)
    if n <= max_days:
        return labels

    take_profit, stop_loss = barrier_levels(cfg, volatility)
    if take_profit.size == 1:  # fixed 模式：广播成逐日数组
        take_profit = np.repeat(take_profit, n)
        stop_loss = np.repeat(stop_loss, n)

    # 未来收益矩阵 F[k-1, t] = p[t+k] / p[t] - 1（k = 1..max_days）
    F = np.full((max_days, n), np.nan, dtype=float)
    for k in range(1, max_days + 1):
        F[k - 1, : n - k] = (p[k:] - p[: n - k]) / p[: n - k]

    tail = n - max_days
    # 有效行：入场价有效、未来 max_days 个价格全部有效、当日屏障有效
    valid = np.isfinite(p[:tail]) & (p[:tail] > 0) \
        & np.isfinite(F[:, :tail]).all(axis=0) \
        & np.isfinite(take_profit[:tail]) & np.isfinite(stop_loss[:tail])

    # 各屏障首次触发步（0-based，等于 k-1）；未触发记为 max_days
    tp_hit = F >= take_profit[None, :]        # NaN 比较为 False
    sl_hit = F <= -stop_loss[None, :]
    tp_first = np.where(tp_hit.any(axis=0), tp_hit.argmax(axis=0), max_days)
    sl_first = np.where(sl_hit.any(axis=0), sl_hit.argmax(axis=0), max_days)
    a = tp_first[:tail]
    b = sl_first[:tail]

    sub = np.full(tail, "", dtype=object)
    timeout = valid & (a == max_days) & (b == max_days)
    take = valid & (a < b)
    stop = valid & (b < a)
    tie = valid & (a == b) & (a < max_days)  # 同一步同时越过两侧屏障
    if cfg["same_day_priority"] == PRIORITY_TAKE_PROFIT:
        take |= tie
    else:
        stop |= tie

    sub[timeout] = values["timeout"]
    sub[take] = values["take_profit"]
    sub[stop] = values["stop_loss"]
    labels[:tail] = sub
    return labels
