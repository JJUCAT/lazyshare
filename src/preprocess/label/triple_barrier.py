# -*- coding: utf-8 -*-
"""标签计算：三元屏蔽（triple barrier）标签 B / T / N，盈亏按价格涨跌幅百分比计算。

参数定义见 preprocess_plan.md 的 "label" 段（config/preprocess.json）：

    price                盈亏计算价格："open"（开盘价）或 "close"（收盘价）
    take_profit          止盈阈值（百分比，正数，如 10.0 表示 +10%）
    take_profit_timeout  止盈时限（在 k = 1..take_profit_timeout 内检查）
    stop_loss            止损阈值（百分比，-8.0 与 8.0 等价，表示 -8%）
    stop_loss_timeout    止损时限（在 k = 1..stop_loss_timeout 内检查）
    values               标签取值 {"take_profit": "B", "stop_loss": "T", "timeout": "N"}

逐日判定规则（第 t 天，入场价 P_t = price 列第 t 天值）：

    profit_k = P_{t+k} / P_t - 1      (k = 1..lookahead)

    止盈：k = 1..take_profit_timeout 内任一 profit_k >= take_profit/100
          -> 止盈（默认 "B"，买入）
    止损：k = 1..stop_loss_timeout 内任一 profit_k <= -|stop_loss|/100
          -> 止损（默认 "T"，卖出）
    其他：两侧窗口内均未触发 -> 无操作（默认 "N"）
    标签优先级 B > T > N：同一行止盈与止损条件都满足时标注止盈（"B"）。

无效行（入场价缺失/非正，或未来 max(take_profit_timeout, stop_loss_timeout)
个交易日价格缺失）标签为空字符串 ""，由调用方丢弃，避免用不完整的前瞻
窗口训练。
"""
from __future__ import annotations

import numpy as np

# 盈亏计算价格："open"（开盘价）/ "close"（收盘价）
PRICE_KEYS = ("open", "close")

# 标签取值的三类（values 必须完整覆盖且互不相同）
VALUE_KEYS = ("take_profit", "stop_loss", "timeout")

DEFAULT_VALUES = {
    "take_profit": "B",  # 止盈 → 买入
    "stop_loss": "T",    # 止损 → 卖出
    "timeout": "N",
}

DEFAULT_PRICE = "open"
DEFAULT_TAKE_PROFIT = 10.0          # 百分比：+10%
DEFAULT_TAKE_PROFIT_TIMEOUT = 15    # 交易日
DEFAULT_STOP_LOSS = 8.0             # 百分比：-8%
DEFAULT_STOP_LOSS_TIMEOUT = 10      # 交易日


def normalize_label_config(raw: dict | None) -> dict:
    """校验并补全 label 配置，返回规范化 dict。

    缺省值：price="open", take_profit=10.0, take_profit_timeout=15,
    stop_loss=8.0, stop_loss_timeout=10,
    values={"take_profit":"B","stop_loss":"T","timeout":"N"}。
    take_profit / stop_loss 均为百分比（10.0 表示 10%）；stop_loss 统一取
    绝对值（配置 -8.0 与 8.0 等价）。
    """
    raw = dict(raw or {})

    price = str(raw.get("price", DEFAULT_PRICE)).strip().lower()
    if price not in PRICE_KEYS:
        raise ValueError(
            f'label.price 只支持 {" / ".join(PRICE_KEYS)}，当前 {price!r}')

    take_profit = float(raw.get("take_profit", DEFAULT_TAKE_PROFIT))
    if not take_profit > 0:
        raise ValueError(f"label.take_profit 必须为正数（百分比），当前 {take_profit}")

    take_profit_timeout = int(
        raw.get("take_profit_timeout", DEFAULT_TAKE_PROFIT_TIMEOUT))
    if take_profit_timeout <= 0:
        raise ValueError(
            f"label.take_profit_timeout 必须为正整数，当前 {take_profit_timeout}")

    stop_loss_raw = float(raw.get("stop_loss", -DEFAULT_STOP_LOSS))
    stop_loss = abs(stop_loss_raw)
    if not stop_loss > 0:
        raise ValueError(f"label.stop_loss 不能为 0，当前 {stop_loss_raw}")

    stop_loss_timeout = int(
        raw.get("stop_loss_timeout", DEFAULT_STOP_LOSS_TIMEOUT))
    if stop_loss_timeout <= 0:
        raise ValueError(
            f"label.stop_loss_timeout 必须为正整数，当前 {stop_loss_timeout}")

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
        "price": price,
        "take_profit": take_profit,
        "take_profit_timeout": take_profit_timeout,
        "stop_loss": stop_loss,
        "stop_loss_timeout": stop_loss_timeout,
        "values": {k: values[k] for k in VALUE_KEYS},
    }


def label_lookahead(cfg: dict | None = None) -> int:
    """标签向前看的交易日数 = max(止盈时限, 止损时限)。"""
    cfg = normalize_label_config(cfg)
    return max(int(cfg["take_profit_timeout"]), int(cfg["stop_loss_timeout"]))


def future_price_matrix(price, lookahead: int) -> np.ndarray:
    """返回未来价格矩阵 F，形状 (lookahead, n)。

    F[k-1, t] = P[t+k]（k = 1..lookahead）；窗口越界处保持 NaN。
    """
    v = np.asarray(price, dtype=float)
    n = v.size
    F = np.full((lookahead, n), np.nan, dtype=float)
    if n == 0:
        return F
    for k in range(1, lookahead + 1):
        if n - k <= 0:
            break
        F[k - 1, : n - k] = v[k:]
    return F


def compute_triple_barrier_labels(price, cfg: dict | None = None) -> np.ndarray:
    """计算三元屏蔽标签，返回 object 数组（与 price 等长，无效行为 ""）。

    price : 逐日价格序列（由 label.price 决定取开盘价还是收盘价）
    cfg   : label 配置（见 normalize_label_config）

    有效行范围：入场价有效且未来 lookahead 个
    （max(止盈时限, 止损时限)）交易日价格均有效；其余行（尤其文件末尾
    lookahead 行）返回 ""，调用方应丢弃。
    """
    cfg = normalize_label_config(cfg)
    tp_days = int(cfg["take_profit_timeout"])
    sl_days = int(cfg["stop_loss_timeout"])
    lookahead = max(tp_days, sl_days)
    values = cfg["values"]
    tp_threshold = cfg["take_profit"] / 100.0
    sl_threshold = -cfg["stop_loss"] / 100.0

    v = np.asarray(price, dtype=float)
    n = v.size
    labels = np.full(n, "", dtype=object)
    if n <= lookahead:
        return labels

    # 未来价格矩阵 F[k-1, t] = P[t+k]（k = 1..lookahead）
    F = future_price_matrix(v, lookahead)
    tail = n - lookahead

    # 有效行：入场价有效且未来 lookahead 个价格全部有效
    entry = v[:tail]
    entry_ok = np.isfinite(entry) & (entry > 0)
    future_ok = np.isfinite(F[:, :tail]).all(axis=0)
    valid = entry_ok & future_ok

    # 涨跌幅：profit[k-1, t] = P[t+k] / P[t] - 1
    denom = np.where(entry_ok, entry, 1.0)
    profit = F[:, :tail] / denom[None, :] - 1.0

    # 各自窗口内是否触发（NaN 比较恒为 False）
    tp_hit = (profit[:tp_days] >= tp_threshold).any(axis=0)
    sl_hit = (profit[:sl_days] <= sl_threshold).any(axis=0)

    # 标签优先级 B > T > N：同一行两类条件都满足时标注止盈
    take = valid & tp_hit
    stop = valid & ~tp_hit & sl_hit
    timeout = valid & ~tp_hit & ~sl_hit

    sub = np.full(tail, "", dtype=object)
    sub[timeout] = values["timeout"]
    sub[take] = values["take_profit"]
    sub[stop] = values["stop_loss"]
    labels[:tail] = sub
    return labels
