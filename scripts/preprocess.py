#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本（多进程并行版）

读取原始日线数据（raw_data 目录下，文件名为 {股票代码}_金玥数据.csv），
对每只股票计算以下指标，并输出到 preprocessed_data 目录：

    M21C : 21 日收盘价均值
    M5C  : 5 日收盘价均值
    M21V : 21 日成交量均值
    M5V  : 5 日成交量均值
    M5H  : 5 日最高价均值
    M5L  : 5 日最低价均值
    NC   : (M5C - M21C) / M21C
    NV   : (M5V - M21V) / M21V
    SNC  : NC 的累计值
    SNV  : NV 的累计值

说明：
    - 某日期 A 的 M21 为 A 及此前 20 个交易日的均值（窗口为 21，含当日）。
      若截至该日期的历史天数不足 21 天，则该日期没有 M21 数据（空值），
      M5 / NC / NV / SNC / SNV 同理按各自窗口处理。
    - 输出包含原始"成交量"列，以及由原始"收盘价 / 成交量 / 最高价 / 最低价"
      重新计算的均值（M21C / M5C / M21V / M5V / M5H / M5L）与归一化值，
      不沿用原文件中的均线列。
    - 输出为 CSV，文件名为股票名称；若股票名称重复，则追加股票代码以区分。
    - 原始数据路径与输出路径从 config/preprocess.json 读取。
    - 使用多进程并行处理，worker 数量默认取本机 CPU 核心数的一半。
    - 输出浮点数最多保留 3 位小数。
    - 运行日志写入 test_output/preprocess.log，记录预处理文件数量与耗时。

用法：
    python3 scripts/preprocess.py [--config config/preprocess.json] [--limit N] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "preprocess.json"

# 计算窗口（含当日）
CLOSE_WINDOW = 21    # M21C：21 日收盘价均值
VOLUME_WINDOW = 21   # M21V：21 日成交量均值
CLOSE_WINDOW_5 = 5   # M5C：5 日收盘价均值
VOLUME_WINDOW_5 = 5  # M5V：5 日成交量均值

# 原始数据列名
COL_DATE = "日期"
COL_CODE = "代码"
COL_NAME = "名称"
COL_INDUSTRY = "所属行业"
COL_CLOSE = "收盘价"
COL_HIGH = "最高价"
COL_LOW = "最低价"
COL_VOLUME = "成交量（股）"

# 输出列
OUTPUT_COLUMNS = [
    "日期",
    "股票代码",
    "股票名称",
    "行业",
    "收盘价",
    "成交量",
    "M21C",
    "M5C",
    "M21V",
    "M5V",
    "M5H",
    "M5L",
    "NC",
    "NV",
    "SNC",
    "SNV",
]

# 需要保留 3 位小数的浮点列
FLOAT_COLUMNS = [
    "收盘价",
    "成交量",
    "M21C",
    "M5C",
    "M21V",
    "M5V",
    "M5H",
    "M5L",
    "NC",
    "NV",
    "SNC",
    "SNV",
]

# 日志文件（工作区 test_output 目录）
LOG_FILE = PROJECT_ROOT / "test_output" / "preprocess.log"

# 文件名中不允许的字符（替换为下划线）
_FILENAME_UNSAFE = set('/\\:*?"<>|')


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


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    raw_dir = Path(cfg.get("raw_data", "")).expanduser()
    out_dir = Path(cfg.get("preprocessed_data", "")).expanduser()
    if not raw_dir or not raw_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在: {raw_dir}")
    if not out_dir:
        raise ValueError("preprocessed_data 未配置")
    return {"raw_dir": raw_dir, "out_dir": out_dir}


def sanitize_filename(name: str) -> str:
    """将股票名称清洗为安全的文件名（去除首尾/连续空白、替换非法字符）。"""
    name = str(name).strip()
    name = "".join("_" if c in _FILENAME_UNSAFE else c for c in name)
    # 合并连续空白（包括全角空格），避免文件名中出现多余空格
    return "".join(name.split())


def process_file(task: tuple[Path, Path]) -> dict:
    """计算单只股票并把结果写入临时 CSV（由 worker 进程调用）。

    task: (原始文件路径, 临时输出路径)
    返回状态字典，避免大 DataFrame 跨进程传输。
    """
    raw_file, tmp_path = task
    try:
        df = pd.read_csv(raw_file, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "code": None, "name": None,
                "message": f"读取失败: {exc}"}
    if df.empty:
        return {"status": "empty", "code": None, "name": None,
                "message": "空文件"}

    # 按日期升序排序（先解析日期，避免字符串排序异常）
    df = df.assign(_dt=pd.to_datetime(df[COL_DATE], errors="coerce"))
    df = (
        df.sort_values("_dt", na_position="last")
        .drop(columns="_dt")
        .reset_index(drop=True)
    )

    code = str(df[COL_CODE].iloc[-1])
    name = str(df[COL_NAME].iloc[-1]).strip()
    industry = str(df[COL_INDUSTRY].iloc[-1]).strip()

    close = pd.to_numeric(df[COL_CLOSE], errors="coerce")
    high = pd.to_numeric(df[COL_HIGH], errors="coerce")
    low = pd.to_numeric(df[COL_LOW], errors="coerce")
    volume = pd.to_numeric(df[COL_VOLUME], errors="coerce")

    # 移动均值：窗口含当日，天数不足时为 NaN
    m21c = close.rolling(window=CLOSE_WINDOW, min_periods=CLOSE_WINDOW).mean()
    m5c = close.rolling(window=CLOSE_WINDOW_5, min_periods=CLOSE_WINDOW_5).mean()
    m21v = volume.rolling(window=VOLUME_WINDOW, min_periods=VOLUME_WINDOW).mean()
    m5v = volume.rolling(window=VOLUME_WINDOW_5, min_periods=VOLUME_WINDOW_5).mean()
    m5h = high.rolling(window=VOLUME_WINDOW_5, min_periods=VOLUME_WINDOW_5).mean()
    m5l = low.rolling(window=VOLUME_WINDOW_5, min_periods=VOLUME_WINDOW_5).mean()

    # 归一化值
    nc = (m5c - m21c) / m21c
    nv = (m5v - m21v) / m21v

    # 累计值
    snc = nc.cumsum()
    snv = nv.cumsum()

    out = pd.DataFrame(
        {
            "日期": df[COL_DATE].astype(str),
            "股票代码": code,
            "股票名称": name,
            "行业": industry,
            "收盘价": close,
            "成交量": volume,
            "M21C": m21c,
            "M5C": m5c,
            "M21V": m21v,
            "M5V": m5v,
            "M5H": m5h,
            "M5L": m5l,
            "NC": nc,
            "NV": nv,
            "SNC": snc,
            "SNV": snv,
        }
    )
    result = out[OUTPUT_COLUMNS].copy()
    # 浮点数最多保留 3 位小数
    result[FLOAT_COLUMNS] = result[FLOAT_COLUMNS].round(3)
    result.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    return {"status": "ok", "code": code, "name": name, "message": None}


def allocate_path(out_dir: Path, base: str, code: str, used: set[str]) -> Path:
    """在 out_dir 下分配未占用的输出文件路径。

    优先级：{名称}.csv -> {名称}_{代码}.csv -> {名称}_{代码}_{n}.csv
    """
    for cand in (base, f"{base}_{code}"):
        path = out_dir / f"{cand}.csv"
        if str(path) not in used:
            used.add(str(path))
            return path
    i = 2
    while True:
        path = out_dir / f"{base}_{code}_{i}.csv"
        if str(path) not in used:
            used.add(str(path))
            return path
        i += 1


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

    raw_files = sorted(raw_dir.glob("*.csv"))
    if args.limit is not None:
        raw_files = raw_files[: args.limit]

    n_workers = args.workers if args.workers else max(1, (os.cpu_count() or 1) // 2)
    start_time = time.perf_counter()
    logging.info("原始数据目录: %s（共 %d 个文件）", raw_dir, len(raw_files))
    logging.info("输出目录: %s", out_dir)
    logging.info("日志文件: %s", LOG_FILE)
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

    # 输出文件名分配：默认使用股票名称，重名时追加股票代码（按输入顺序确定性分配）
    seen_names: dict[str, str] = {}   # 清洗后的名称 -> 已分配股票代码
    used_paths: set[str] = set()

    processed = 0
    skipped = 0
    collisions = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # map 保持输入顺序，保证重名分配结果确定
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
            base = sanitize_filename(name)
            if base in seen_names:
                collisions += 1
                logging.warning(
                    "股票名称重复: %s（%s 与 %s），输出文件追加代码区分",
                    name, seen_names[base], raw_file.name,
                )
            seen_names[base] = code

            final_path = allocate_path(out_dir, base, code, used_paths)
            tmp_path.rename(final_path)
            processed += 1
            if processed % 500 == 0:
                logging.info("已处理 %d 个文件...", processed)

    elapsed = time.perf_counter() - start_time
    logging.info("完成：预处理文件数量 %d 个，跳过 %d 个，重名追加 %d 个，耗时 %.2f 秒",
                 processed, skipped, collisions, elapsed)
    logging.info("输出目录: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
