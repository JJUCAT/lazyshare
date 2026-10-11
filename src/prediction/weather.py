# -*- coding: utf-8 -*-
"""按 config/classify_train.json 的 weather 配置统计大盘 / 热门行业 / 热门票。

统计口径（见 prediction_plan.md 第 6 条）：
- global    大盘：取最近 days 天成交量均值，与当前（最新交易日）成交量比较
- industrys 热门行业：取各行业最近 days 天成交量均值，当前成交量大于均值者，
            按当前成交量从高到低取前 tops 个
- shares / hot_industrys_shares 热门行业热门票：在上述热门行业内，取每只个股
            最近 days 天成交量均值，当前成交量大于均值者取前 tops 个

数据来源：
- 大盘 / 行业：cfg["weather"]（weather.csv 宽表，行=日期，列=大盘+各行业，单位万股）
- 个股：predict.py 切片的股票窗口（dataset 目录，含"行业"/"成交量"列，单位股）

输出为文本行列表，供 predict.py 追加写入 pred-<date>/prediction.log。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 股 -> 万股（weather.csv 中大盘/行业成交量单位为万股）
WAN = 10_000.0

DATE_COL = "日期"
MARKET_COL = "大盘"
INDUSTRY_COL = "行业"
VOLUME_COL = "成交量"
NAME_COL = "股票名称"
CODE_COL = "股票代码"


def get_weather_config(cfg: dict) -> dict:
    """从 classify_train.json 读取 weather 统计配置。

    优先取顶层 "weather" 为 dict 时的内容（兼容新配置）；
    否则取 "evaluate"."weather"（当前 config/classify_train.json 采用）。
    """
    w = cfg.get("weather")
    if isinstance(w, dict):
        return w
    evaluate = cfg.get("evaluate")
    if isinstance(evaluate, dict) and isinstance(evaluate.get("weather"), dict):
        return evaluate["weather"]
    return {}


def get_weather_path(cfg: dict) -> Path | None:
    """从 classify_train.json 读取 weather.csv 路径（顶层 "weather" 为字符串时）。"""
    for key in ("weather", "weather_path", "weather_csv"):
        value = cfg.get(key)
        if isinstance(value, (str, Path)) and str(value).strip():
            path = Path(str(value))
            return path if path.is_absolute() else PROJECT_ROOT / path
    return None


def load_weather_table(path: Path | None) -> pd.DataFrame | None:
    """读取 weather.csv 宽表，按日期升序排序并将数值列转为 float。

    文件不存在、为空或缺少日期列时返回 None。
    """
    if path is None:
        logging.warning("未配置 weather.csv 路径，跳过天气统计")
        return None
    try:
        df = pd.read_csv(path, dtype=str)
    except (FileNotFoundError, OSError, ValueError):
        logging.warning("weather.csv 不存在或无法读取: %s", path)
        return None
    if df is None or df.empty or DATE_COL not in df.columns:
        logging.warning("weather.csv 为空或缺少日期列: %s", path)
        return None
    df = df.copy()
    df[DATE_COL] = df[DATE_COL].astype(str).str.strip()
    df = df[df[DATE_COL] != ""].sort_values(DATE_COL).reset_index(drop=True)
    if df.empty:
        logging.warning("weather.csv 无有效日期行: %s", path)
        return None
    for col in df.columns:
        if col != DATE_COL:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _stats(values: pd.Series, days: int) -> tuple[float, float, int] | None:
    """取序列末尾 days 天，返回 (当前值, 均值, 参与天数)；无有效值时返回 None。"""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return None
    n = max(1, int(days))
    tail = v.tail(n)
    return float(tail.iloc[-1]), float(tail.mean()), int(len(tail))


def _fmt(value: float) -> str:
    """成交量格式化（万股，千分位 + 两位小数）。"""
    return f"{value:,.2f}"


def _cmp(cur: float, mean: float) -> str:
    if cur > mean:
        return "大于均值"
    if cur < mean:
        return "小于均值"
    return "等于均值"


def build_weather_lines_from_file(config_path, windows: dict) -> list[str]:
    """从原始配置文件读取 weather 段并统计。

    注意：train.data.load_config 返回的是归一化后的配置，会丢弃顶层 "weather"
    路径与 "evaluate"."weather" 段，因此这里直接读取原始 JSON 文件。
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as exc:
        logging.warning("读取配置文件失败，跳过天气统计: %s (%s)", config_path, exc)
        return []
    return build_weather_lines(cfg, windows)


def build_weather_lines(cfg: dict, windows: dict) -> list[str]:
    """按配置生成大盘 / 热门行业 / 热门票统计文本行。

    windows: predict.py 的切片结果 {stem: (窗口 csv 路径, 最新交易日)}。
    未配置 weather 或缺少大盘数据时，返回空列表（不写任何行）。
    """
    weather_cfg = get_weather_config(cfg)
    if not weather_cfg:
        logging.info("未配置 weather 统计，跳过天气概况")
        return []

    table = load_weather_table(get_weather_path(cfg))
    lines: list[str] = ["大盘交易行情概况:"]

    # 1) 大盘：最近 days 天成交量均值 vs 当前成交量
    global_cfg = weather_cfg.get("global") or {}
    global_days = int(global_cfg.get("days", 15) or 15)
    if table is not None and MARKET_COL in table.columns:
        stat = _stats(table[MARKET_COL], global_days)
        if stat:
            cur, mean, n = stat
            date = table[DATE_COL].iloc[-1]
            lines.append(
                f"  大盘: 交易日 {date}，最近 {n} 天成交量均值 {_fmt(mean)} 万股，"
                f"当前 {_fmt(cur)} 万股（{_cmp(cur, mean)}）")
        else:
            lines.append("  大盘: 无有效成交量数据")
    else:
        lines.append("  大盘: weather.csv 缺失或缺少大盘列")

    # 2) 热门行业：当前成交量大于最近 days 天均值者，按当前成交量取前 tops
    industry_cfg = weather_cfg.get("industrys") or {}
    industry_days = int(industry_cfg.get("days", 15) or 15)
    industry_tops = int(industry_cfg.get("tops", 3) or 3)
    hot_industries: list[str] = []
    ind_rows: list[tuple[str, float, float]] = []
    if table is not None:
        ind_cols = [c for c in table.columns if c not in (DATE_COL, MARKET_COL)]
        for col in ind_cols:
            stat = _stats(table[col], industry_days)
            if stat and stat[0] > stat[1]:
                ind_rows.append((col, stat[0], stat[1]))
        ind_rows.sort(key=lambda t: t[1], reverse=True)
        hot_industries = [row[0] for row in ind_rows[:industry_tops]]
    lines.append(
        f"  热门行业（最近 {industry_days} 天成交量均值，当前大于均值，"
        f"按当前成交量取前 {industry_tops}）:")
    if ind_rows:
        for i, (name, cur, mean) in enumerate(ind_rows[:industry_tops], 1):
            lines.append(f"    {i}. {name}  当前 {_fmt(cur)} 万股"
                         f"  {industry_days}天均值 {_fmt(mean)} 万股")
    else:
        lines.append("    无（没有行业当前成交量大于均值）")

    # 3) 热门行业热门票：热门行业内个股当前成交量大于最近 days 天均值者
    share_cfg = (weather_cfg.get("shares")
                 or weather_cfg.get("hot_industrys_shares") or {})
    share_days = int(share_cfg.get("days", 5) or 5)
    share_tops = int(share_cfg.get("tops", 3) or 3)
    lines.append(
        f"  热门行业热门票（最近 {share_days} 天成交量均值，当前大于均值，"
        f"每个热门行业取前 {share_tops}）:")

    # 最新交易日：只统计在最新交易日仍有数据的个股，剔除退市/长期停牌股
    latest_trade_date: str | None = None
    if table is not None and DATE_COL in table.columns and not table.empty:
        latest_trade_date = str(table[DATE_COL].iloc[-1]).strip()
    else:
        end_dates = [str(d).strip() for _p, d in windows.values() if str(d).strip()]
        latest_trade_date = max(end_dates) if end_dates else None

    by_industry: dict[str, list[tuple[str, str, float, float]]] = {}
    if hot_industries and windows:
        hot_set = set(hot_industries)
        for csv_path, end_date in windows.values():
            if latest_trade_date is not None and str(end_date).strip() != latest_trade_date:
                continue  # 非最新交易日数据（退市/长期停牌），不参与热门票统计
            try:
                df = pd.read_csv(Path(csv_path), encoding="utf-8-sig", dtype=str)
            except Exception:  # noqa: BLE001
                continue
            if df.empty or INDUSTRY_COL not in df.columns or VOLUME_COL not in df.columns:
                continue
            industry = str(df[INDUSTRY_COL].iloc[-1]).strip()
            if industry not in hot_set:
                continue
            stat = _stats(pd.to_numeric(df[VOLUME_COL], errors="coerce") / WAN, share_days)
            if not stat:
                continue
            cur, mean, _n = stat
            if cur <= mean:
                continue
            code = (str(df[CODE_COL].iloc[-1]).strip() if CODE_COL in df.columns
                    else Path(csv_path).stem)
            name = str(df[NAME_COL].iloc[-1]).strip() if NAME_COL in df.columns else ""
            by_industry.setdefault(industry, []).append((code, name, cur, mean))

    has_share_row = False
    for industry in hot_industries:
        rows = sorted(by_industry.get(industry, []),
                      key=lambda r: r[2], reverse=True)[:share_tops]
        if not rows:
            continue
        has_share_row = True
        lines.append(f"    [{industry}]")
        for i, (code, name, cur, mean) in enumerate(rows, 1):
            lines.append(f"      {i}. {code} {name}  当前 {_fmt(cur)} 万股"
                         f"  {share_days}天均值 {_fmt(mean)} 万股")
    if not has_share_row:
        lines.append("    无（热门行业内没有个股当前成交量大于均值）")
    return lines
