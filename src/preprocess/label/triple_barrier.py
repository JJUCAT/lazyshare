# -*- coding: utf-8 -*-
"""标签计算：三元屏蔽法（triple barrier）标签 T / B / N。

参数定义见 preprocess_plan.md 的 "label" 段（config/preprocess.json）：

    max_days           观察天数（向前看多少个交易日）
    price              收益计算使用的价格列名（默认 "开盘价"）
    take_profit        止盈百分比（0.15 表示 +15%）
    stop_loss          止损百分比（0.10 表示 -10%）
    same_day_priority  同一步同时触发止盈与止损时优先标注哪一侧
    values             标签取值 {"take_profit": "B", "stop_loss": "T", "timeout": "N"}

逐日判定规则（第 t 天）：

    入场价 = price[t]
    依次取 k = 1..max_days，计算
        profit_k = (price[t+k] - price[t]) / price[t]
        profit_k >= take_profit  -> 止盈（默认 "B"，买入）
        profit_k <= -stop_loss   -> 止损（默认 "T"，卖出）
    先触发的一侧即为该日标签；max_days 内均未触发 -> 超时无操作（默认 "N"）。

    同一个 k 同时越过上下两条屏障（跳空等极端情形）时按 same_day_priority 决定。

无效行（价格缺失，或未来 max_days 个交易日数据不完整）标签为空字符串 ""，
由调用方丢弃，避免用不完整的前瞻窗口训练。
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

# same_day_priority 允许取值
PRIORITY_TAKE_PROFIT = "take_profit"
PRIORITY_STOP_LOSS = "stop_loss"


def normalize_label_config(raw: dict | None) -> dict:
    """校验并补全 label 配置，返回规范化 dict。

    缺省值：max_days=21, price="开盘价", take_profit=0.15, stop_loss=0.10,
    same_day_priority="stop_loss", values={"take_profit":"B","stop_loss":"T","timeout":"N"}。
    """
    raw = dict(raw or {})
    max_days = int(raw.get("max_days", 21))
    if max_days <= 0:
        raise ValueError(f"label.max_days 必须为正整数，当前 {max_days}")

    price = str(raw.get("price") or DEFAULT_PRICE_COLUMN).strip()
    if not price:
        price = DEFAULT_PRICE_COLUMN

    take_profit = float(raw.get("take_profit", 0.15))
    if not take_profit > 0:
        raise ValueError(f"label.take_profit 必须为正数，当前 {take_profit}")

    stop_loss = float(raw.get("stop_loss", 0.10))
    if not stop_loss > 0:
        raise ValueError(f"label.stop_loss 必须为正数，当前 {stop_loss}")

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
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "same_day_priority": priority,
        "values": {k: values[k] for k in VALUE_KEYS},
    }


def compute_triple_barrier_labels(price, cfg: dict | None = None) -> np.ndarray:
    """计算三元屏蔽标签，返回 object 数组（与 price 等长，无效行为 ""）。

    price: 收益计算使用的价格序列（如开盘价），可为 list / numpy / pandas Series
    cfg  : label 配置（见 normalize_label_config）

    有效行范围：价格及未来 max_days 个价格均有效的行；其余行（尤其文件末尾
    max_days 行）返回 ""，调用方应丢弃。
    """
    cfg = normalize_label_config(cfg)
    max_days = int(cfg["max_days"])
    values = cfg["values"]

    p = np.asarray(price, dtype=float)
    n = p.size
    labels = np.full(n, "", dtype=object)
    if n <= max_days:
        return labels

    # 未来收益矩阵 F[k-1, t] = p[t+k] / p[t] - 1（k = 1..max_days）
    F = np.full((max_days, n), np.nan, dtype=float)
    for k in range(1, max_days + 1):
        F[k - 1, : n - k] = (p[k:] - p[: n - k]) / p[: n - k]

    tail = n - max_days
    # 有效行：入场价有效，且未来 max_days 个价格全部有效
    valid = np.isfinite(p[:tail]) & (p[:tail] > 0) \
        & np.isfinite(F[:, :tail]).all(axis=0)

    # 各屏障首次触发步（0-based，等于 k-1）；未触发记为 max_days
    tp_hit = F >= cfg["take_profit"]        # NaN 比较为 False
    sl_hit = F <= -cfg["stop_loss"]
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
