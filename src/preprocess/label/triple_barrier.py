# -*- coding: utf-8 -*-
"""标签计算：简单三元屏蔽（triple barrier）标签 B / T / N，盈亏用 LU（涨跌幅利用率）累加。

LU = (C_t - C_{t-1}) / (max_limit × C_{t-1})，即当日涨跌幅占该股票当日最大涨跌幅
的比例（详见 src/preprocess/handle/indicators.py）。用 LU 累加衡量盈亏，天然消除
板块涨跌幅限制差异与个股波动差异（不同板块/ST 的 LU 可比较）。

参数定义见 preprocess_plan.md 的 "label" 段（config/preprocess.json）：

    take_profit           盈利阈值（LU 累计值，正数，如 0.15）
    take_profit_timeout   止盈时限（在 k = 1..take_profit_timeout 内检查）
    stop_loss             亏损阈值（LU 累计值，负数 -0.1 或正数 0.1 均按 0.1 处理）
    stop_loss_timeout     止损时限（在 k = 1..stop_loss_timeout 内检查）
    values                标签取值 {"take_profit": "B", "stop_loss": "T", "timeout": "N"}

逐日判定规则（第 t 天，入场价 = 第 t 日收盘价）：

    止盈：在 k = 1..take_profit_timeout 内，
          profit_k = Σ LU[t+1..t+k]
          profit_k >= take_profit  -> 止盈（默认 "B"，买入）
    止损：在 k = 1..stop_loss_timeout 内，
          profit_k <= -|stop_loss| -> 止损（默认 "T"，卖出）
    止盈 / 止损各自独立窗口，先触发的一侧生效；同一 k 同时触发时优先止损
    （保守，标注 values.stop_loss）。
    两侧窗口内均未触发 -> 无操作（默认 "N"）。

无效行（未来 max(take_profit_timeout, stop_loss_timeout) 个交易日的 LU 缺失）
标签为空字符串 ""，由调用方丢弃，避免用不完整的前瞻窗口训练。
"""
from __future__ import annotations

import numpy as np

# 标签取值的三类（values 必须完整覆盖且互不相同）
VALUE_KEYS = ("take_profit", "stop_loss", "timeout")

DEFAULT_VALUES = {
    "take_profit": "B",  # 止盈 → 买入
    "stop_loss": "T",    # 止损 → 卖出
    "timeout": "N",
}

DEFAULT_TAKE_PROFIT = 0.30
DEFAULT_TAKE_PROFIT_TIMEOUT = 21
DEFAULT_STOP_LOSS = 0.20
DEFAULT_STOP_LOSS_TIMEOUT = 15


def normalize_label_config(raw: dict | None) -> dict:
    """校验并补全 label 配置，返回规范化 dict。

    缺省值：take_profit=0.30, take_profit_timeout=21, stop_loss=0.20,
    stop_loss_timeout=15, values={"take_profit":"B","stop_loss":"T","timeout":"N"}。
    stop_loss 统一取绝对值（配置 -0.1 与 0.1 等价，均表示亏损 0.1 个 LU）。
    阈值单位均为 LU（涨跌幅利用率）累计值。
    """
    raw = dict(raw or {})

    take_profit = float(raw.get("take_profit", DEFAULT_TAKE_PROFIT))
    if not take_profit > 0:
        raise ValueError(f"label.take_profit 必须为正数，当前 {take_profit}")

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
        "take_profit": take_profit,
        "take_profit_timeout": take_profit_timeout,
        "stop_loss": stop_loss,
        "stop_loss_timeout": stop_loss_timeout,
        "values": {k: values[k] for k in VALUE_KEYS},
    }


def label_lookahead(cfg: dict | None = None) -> int:
    """标签向前看的交易日数 = max(止盈时限, 止损时限)。"""
    cfg = cfg if cfg is not None and "values" in cfg else normalize_label_config(cfg)
    return max(int(cfg["take_profit_timeout"]), int(cfg["stop_loss_timeout"]))


def cumulative_lu(lu, lookahead: int) -> np.ndarray:
    """返回未来 LU 累计矩阵 F，形状 (lookahead, n)。

    F[k-1, t] = Σ LU[t+1..t+k]（k = 1..lookahead）；窗口内任一 LU 缺失则为 NaN。
    """
    v = np.asarray(lu, dtype=float)
    n = v.size
    F = np.full((lookahead, n), np.nan, dtype=float)
    if n == 0:
        return F

    nan_flag = np.isnan(v).astype(np.int64)
    clean = np.where(nan_flag > 0, 0.0, v)
    # 前缀和：csum[i] = Σ clean[0..i-1]，cnan[i] = 前 i 个中的缺失个数
    csum = np.concatenate(([0.0], np.cumsum(clean)))
    cnan = np.concatenate(([0], np.cumsum(nan_flag)))
    for k in range(1, lookahead + 1):
        if n - k <= 0:
            break
        t = np.arange(n - k)
        lo = t + 1          # 窗口起点 t+1
        hi = t + k + 1      # 窗口终点 t+k 的右开边界
        sums = csum[hi] - csum[lo]
        bad = (cnan[hi] - cnan[lo]) > 0
        F[k - 1, : n - k] = np.where(bad, np.nan, sums)
    return F


def compute_triple_barrier_labels(lu, cfg: dict | None = None) -> np.ndarray:
    """计算简单三元屏蔽标签，返回 object 数组（与 lu 等长，无效行为 ""）。

    lu  : 逐日涨跌幅利用率 LU 序列
    cfg : label 配置（见 normalize_label_config）

    有效行范围：未来 lookahead 个（max(止盈时限, 止损时限)）LU 均有效；
    其余行（尤其文件末尾 lookahead 行）返回 ""，调用方应丢弃。
    """
    cfg = normalize_label_config(cfg)
    tp_days = int(cfg["take_profit_timeout"])
    sl_days = int(cfg["stop_loss_timeout"])
    lookahead = max(tp_days, sl_days)
    values = cfg["values"]

    v = np.asarray(lu, dtype=float)
    n = v.size
    labels = np.full(n, "", dtype=object)
    if n <= lookahead:
        return labels

    # 未来 LU 累计矩阵 F[k-1, t] = Σ LU[t+1..t+k]（k = 1..lookahead）
    F = cumulative_lu(v, lookahead)
    tail = n - lookahead

    # 有效行：未来 lookahead 个 LU 全部有效
    valid = np.isfinite(F[:, :tail]).all(axis=0)

    # 各自独立窗口内的首次触发步（0-based，等于 k-1）；未触发记为对应窗口长度
    tp_hit = F[:tp_days, :tail] >= cfg["take_profit"]     # NaN 比较为 False
    sl_hit = F[:sl_days, :tail] <= -cfg["stop_loss"]
    a = np.where(tp_hit.any(axis=0), tp_hit.argmax(axis=0), tp_days)
    b = np.where(sl_hit.any(axis=0), sl_hit.argmax(axis=0), sl_days)

    tp_seen = a < tp_days
    sl_seen = b < sl_days
    # 同一 k 同时触发时优先止损（保守）：止损在前一交易日及之前才算止盈胜出
    take = valid & tp_seen & (~sl_seen | (a < b))
    stop = valid & sl_seen & (~tp_seen | (b <= a))
    timeout = valid & ~tp_seen & ~sl_seen

    sub = np.full(tail, "", dtype=object)
    sub[timeout] = values["timeout"]
    sub[take] = values["take_profit"]
    sub[stop] = values["stop_loss"]
    labels[:tail] = sub
    return labels
