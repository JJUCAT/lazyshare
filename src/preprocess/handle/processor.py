# -*- coding: utf-8 -*-
"""基础工作：单文件处理与多进程编排（日志、临时文件、文件名分配）。"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from ..label.peak import compute_peak_labels
from .config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from .indicators import (
    COL_CODE,
    COL_DATE,
    COL_INDUSTRY,
    COL_NAME,
    FLOAT_COLUMNS,
    OUTPUT_COLUMNS,
    compute_indicators,
    is_st_stock,
    sanitize_filename,
)

# 日志文件（工作区 test_output 目录）
LOG_FILE = PROJECT_ROOT / "test_output" / "preprocess.log"


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

    # ST 股票跳过，不生成预处理数据文件
    if is_st_stock(df):
        return {"status": "st_skip", "code": code, "name": name,
                "message": "ST 股票"}

    ind = compute_indicators(df)

    # 峰值标签：T（高峰）/ B（低谷）/ 空（标签计算在 src/preprocess/label）
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
            "峰值标签": peak_labels,
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
