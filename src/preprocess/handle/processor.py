# -*- coding: utf-8 -*-
"""基础工作：单文件处理与多进程编排（日志、临时文件、文件名分配）。"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from ..label.peak import compute_peak_labels
from .config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from .indicators import (
    COL_CLOSE,
    COL_CODE,
    COL_DATE,
    COL_HIGH,
    COL_INDUSTRY,
    COL_LOW,
    COL_NAME,
    COL_VOLUME,
    FLOAT_COLUMNS,
    OUTPUT_COLUMNS,
    compute_indicators,
    is_st_stock,
    sanitize_filename,
)
from .weather import EMPTY_WEATHER, compute_imv_siv, load_weather

# 日志文件（工作区 test_output 目录）
LOG_FILE = PROJECT_ROOT / "test_output" / "preprocess.log"

# 原始股票数据文件名：代码-名称.csv（或旧格式 代码_名称.csv）。
# 用于排除 weather.csv 等非股票文件（见 raw 目录内的 weather.csv）。
_STOCK_FILE_RE = re.compile(r"^\d{6}[-_]")

# weather.csv 查询表（由进程池 worker 经 initializer 加载一次，供 IMV/SIV 计算）
_WEATHER: dict | None = None


def _init_worker(weather_path: str | None) -> None:
    """进程池 worker 初始化：加载 weather.csv 查询表。"""
    global _WEATHER
    _WEATHER = load_weather(weather_path) if weather_path else None


def setup_logging(log_file: Path = LOG_FILE, verbose: bool = False) -> None:
    """同时输出到控制台与日志文件。"""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    # 清空已有 handler，避免重复
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    # 控制台
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    # 日志文件
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def process_file(task: tuple[Path, Path]) -> dict:
    """计算单只股票并把结果写入临时 CSV（由 worker 进程调用）。

    task: (原始文件路径, 临时输出路径)
    返回状态字典，避免大 DataFrame 跨进程传输。
    """
    raw_file, tmp_path = task
    try:
        # dtype=str：保留股票代码的前导零（如 000001），避免被推断为整数
        df = pd.read_csv(raw_file, encoding="utf-8-sig", dtype=str)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "code": None, "name": None,
                "message": f"读取失败: {exc}"}
    if df.empty:
        return {"status": "empty", "code": None, "name": None,
                "message": "空文件"}

    # 必要列校验（防止 weather.csv 等非股票文件导致任务整体崩溃）
    _REQUIRED_COLS = (COL_DATE, COL_CODE, COL_NAME, COL_INDUSTRY,
                      COL_CLOSE, COL_HIGH, COL_LOW, COL_VOLUME)
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        return {"status": "error", "code": None, "name": None,
                "message": f"缺少必要列: {missing}"}

    # 按日期升序排序（先解析日期，避免字符串排序异常）
    df = df.assign(_dt=pd.to_datetime(df[COL_DATE], errors="coerce"))
    df = (
        df.sort_values("_dt", na_position="last")
        .drop(columns="_dt")
        .reset_index(drop=True)
    )

    code_raw = df[COL_CODE].iloc[-1]
    name_raw = df[COL_NAME].iloc[-1]
    industry_raw = df[COL_INDUSTRY].iloc[-1]
    code = str(code_raw).strip() if not pd.isna(code_raw) else ""
    name = str(name_raw).strip() if not pd.isna(name_raw) else ""
    industry = str(industry_raw).strip() if not pd.isna(industry_raw) else ""

    # ST 股票跳过，不生成预处理数据文件
    if is_st_stock(df):
        return {"status": "st_skip", "code": code, "name": name,
                "message": "ST 股票"}

    ind = compute_indicators(df)

    # IMV：行业成交量 / 大盘成交量；SIV：个股成交量 / 行业成交量（基于 weather.csv）
    imv, siv = compute_imv_siv(df, _WEATHER if _WEATHER is not None else EMPTY_WEATHER)

    # 峰值标签：T（高峰）/ B（低谷）/ N（None）（标签计算在 src/preprocess/label）
    peak_labels = compute_peak_labels(ind["close"], ind["m21c"])

    out = pd.DataFrame(
        {
            "日期": df[COL_DATE].astype(str),
            "股票代码": code,
            "股票名称": name,
            "行业": industry,
            "收盘价": ind["close"],
            "成交量": ind["volume"],
            "M21C": ind["m21c"],
            "M21V": ind["m21v"],
            "NC": ind["nc"],
            "NV": ind["nv"],
            "NA": ind["na"],
            "NBear": ind["nbear"],
            "NBull": ind["nbull"],
            "SNC": ind["snc"],
            "SNV": ind["snv"],
            "SNB": ind["snb"],
            "M21SNB": ind["m21snb"],
            "IMV": imv,
            "SIV": siv,
            "峰值标签": peak_labels,
        }
    )
    result = out[OUTPUT_COLUMNS].copy()
    # 浮点数最多保留 3 位小数
    result[FLOAT_COLUMNS] = result[FLOAT_COLUMNS].round(3)
    result.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    return {"status": "ok", "code": code, "name": name, "message": None}


def needs_update(raw_file: Path, out_dir: Path) -> bool:
    """判断原始文件是否需要重新生成预处理输出（增量更新）。

    - ST 股票不生成输出，无需更新
    - 输出文件不存在 → 需要（新股票）
    - 输出文件缺少原始数据中的任意日期 → 需要
    - 输出文件已包含原始数据全部日期 → 无需
    """
    raw_dates: set[str] = set()
    code = name = ""
    last_name = ""
    last_st_flag = False
    try:
        with open(raw_file, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return False
            try:
                ci = header.index("代码")
                ni = header.index("名称")
                di = header.index("日期")
            except ValueError:
                return False
            st_idx = header.index("是否ST") if "是否ST" in header else -1
            for row in reader:
                if len(row) <= max(ci, ni, di):
                    continue
                if not code:
                    code = row[ci].strip()
                    name = row[ni].strip()
                raw_dates.add(row[di])
                if len(row) > ni:
                    last_name = row[ni].strip()
                if st_idx >= 0 and len(row) > st_idx:
                    last_st_flag = row[st_idx].strip() == "是"
    except OSError:
        return False
    if not code or not raw_dates:
        return False
    # ST 股票不生成预处理文件，无需更新
    if "ST" in last_name.upper() or last_st_flag:
        return False

    # 按代码匹配输出文件（名称可能变化）
    matches = list(out_dir.glob(f"{code}-*.csv"))
    if not matches:
        return True
    out_file = matches[0]

    out_dates: set[str] = set()
    try:
        with open(out_file, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return True
            try:
                di = header.index("日期")
            except ValueError:
                return True
            # 旧输出缺少 IMV/SIV 列时也需重新生成（列结构升级）
            if "IMV" not in header or "SIV" not in header:
                return True
            for row in reader:
                if len(row) > di:
                    out_dates.add(row[di])
    except OSError:
        return True
    return not raw_dates.issubset(out_dates)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A 股日线数据预处理（多进程并行）")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"配置文件路径（默认: {DEFAULT_CONFIG}）",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="仅处理前 N 个文件（调试用）")
    parser.add_argument("--update", action="store_true",
                        help="增量更新：仅重新处理有新数据（新日期/新股票）的文件")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="并行 worker 数量（默认取 CPU 核心数的一半）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)

    cfg = load_config(args.config)
    raw_dir: Path = cfg["raw_dir"]
    out_dir: Path = cfg["out_dir"]
    weather_path: Path | None = cfg["weather"]

    raw_files = sorted(raw_dir.glob("*.csv"))
    # 仅处理股票数据文件（代码-名称.csv），排除 weather.csv 等非股票文件
    raw_files = [f for f in raw_files if _STOCK_FILE_RE.match(f.name)]
    if args.limit is not None:
        raw_files = raw_files[: args.limit]

    if args.update:
        before = len(raw_files)
        raw_files = [f for f in raw_files if needs_update(f, out_dir)]
        logging.info("update 模式：共 %d 个原始文件，%d 个需要更新",
                     before, len(raw_files))

    n_workers = args.workers if args.workers else max(1, (os.cpu_count() or 1) // 2)
    start_time = time.perf_counter()
    logging.info("原始数据目录: %s（共 %d 个文件）", raw_dir, len(raw_files))
    logging.info("输出目录: %s", out_dir)
    logging.info("日志文件: %s", LOG_FILE)
    if weather_path is not None and weather_path.exists():
        logging.info("weather.csv: %s（IMV/SIV 数据来源）", weather_path)
    else:
        logging.warning("weather.csv 不存在，IMV/SIV 列将为空白: %s", weather_path)
    logging.info("并行 worker 数量: %d（CPU 核心数: %s）",
                 n_workers, os.cpu_count())
    out_dir.mkdir(parents=True, exist_ok=True)

    # 清理历史遗留的临时文件
    for stale in out_dir.glob(".tmp_*.csv"):
        stale.unlink(missing_ok=True)

    # 构建任务：每个 worker 写独立临时文件，避免大 DataFrame 跨进程传输
    tmp_dir = out_dir
    tasks = [
        (raw_file, tmp_dir / f".tmp_{i:05d}.csv")
        for i, raw_file in enumerate(raw_files)
    ]

    processed = 0
    skipped = 0
    weather_arg = (
        str(weather_path)
        if weather_path is not None and weather_path.exists()
        else None
    )
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(weather_arg,)
    ) as executor:
        # map 保持输入顺序
        for i, result in enumerate(executor.map(process_file, tasks, chunksize=1)):
            tmp_path: Path = tasks[i][1]
            raw_file: Path = tasks[i][0]
            status = result["status"]
            if status != "ok":
                skipped += 1
                logging.warning("跳过 %s: %s（%s）",
                                raw_file.name, result["message"], status)
                tmp_path.unlink(missing_ok=True)
                continue

            code = result["code"]
            name = result["name"]
            # 输出文件名：股票代码-股票名称（股票代码唯一，见 preprocess_plan.md）
            final_path = out_dir / f"{code}-{sanitize_filename(name)}.csv"
            tmp_path.rename(final_path)
            # 清理同名代码的旧输出文件（名称变更时避免残留）
            for old in out_dir.glob(f"{code}-*.csv"):
                if old != final_path:
                    old.unlink(missing_ok=True)
            processed += 1
            if processed % 500 == 0:
                logging.info("已处理 %d 个文件...", processed)

    elapsed = time.perf_counter() - start_time
    logging.info("完成：预处理文件数量 %d 个，跳过 %d 个，耗时 %.2f 秒",
                 processed, skipped, elapsed)
    logging.info("输出目录: %s", out_dir)
    return 0
