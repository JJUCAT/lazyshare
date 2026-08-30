#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把预处理 CSV 转换为 Label Studio 数据源可识别的 task。

读取 config/preprocess.json：
    - "preprocessed_data"：预处理 CSV 目录（源）
    - "tasks"            ：Label Studio task 输出目录
每个 CSV 生成一条 Label Studio 时间序列（TimeSeries）task，写入
<tasks 目录>/tasks.jsonl（JSON Lines，每行一个 task，Label Studio 支持直接导入）。

task["data"] 字段：
    - "ts"    ：指向 CSV 的本地文件 URL，格式 "/data/<文件名>"。该路径相对于
                Label Studio 本地文件根目录 LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT。
                默认根目录即预处理 CSV 目录（preprocessed_data），因此可直接读取
                源文件，无需复制或符号链接。
    - "time"  ：时间列名（默认 "日期"）
    - "series"：数值序列列名列表（默认排除 日期/股票代码/股票名称/行业/峰值标签）
    - "meta"  ：股票代码 / 股票名称 / 行业 等元信息，便于筛选与参考

用法：
    python3 scripts/csv2task.py [--config config/preprocess.json]
    python3 scripts/csv2task.py --document-root /path/to/root
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.handle.config import DEFAULT_CONFIG  # noqa: E402

# 不作为时间序列数值列展示的列（日期作时间轴，其余为元信息/非数值标签）
NON_SERIES_COLUMNS = {"日期", "股票代码", "股票名称", "行业", "峰值标签"}


def read_csv_head(csv_path: Path) -> tuple[list[str], pd.DataFrame]:
    """只读取列名与首行数据，避免加载整个大文件（仅需列结构与元信息）。"""
    df = pd.read_csv(csv_path, nrows=1, encoding="utf-8-sig")
    return list(df.columns), df


def build_task(csv_path: Path, ts_url: str) -> dict:
    """根据单个 CSV 构建一条 Label Studio 时间序列 task。"""
    columns, head = read_csv_head(csv_path)

    # 时间列：优先 "日期"，否则取第一列
    time_col = "日期" if "日期" in columns else (columns[0] if columns else "")
    # 数值序列列：按 CSV 列顺序排除非序列列
    series = [c for c in columns if c not in NON_SERIES_COLUMNS]

    # 元信息（整列相同，取首行即可；首行为空时留空）
    meta: dict[str, str] = {}
    if not head.empty:
        for key in ("股票代码", "股票名称", "行业"):
            if key in columns:
                val = head[key].iloc[0]
                meta[key] = "" if pd.isna(val) else str(val)
    meta["文件"] = csv_path.name
    meta["数据行数"] = _count_lines(csv_path)

    return {
        "data": {
            "ts": ts_url,
            "time": time_col,
            "series": series,
            "meta": meta,
        }
    }


def _count_lines(csv_path: Path) -> int:
    """统计 CSV 数据行数（不含表头），读取失败返回 0。"""
    try:
        with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            return sum(1 for _ in f) - 1
    except OSError:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="预处理 CSV 转 Label Studio task")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"配置文件路径（默认: {DEFAULT_CONFIG}）",
    )
    parser.add_argument(
        "--document-root", type=Path, default=None,
        help="Label Studio 本地文件根目录（默认即预处理 CSV 目录，"
             "即 ts 的 /data/ 前缀直接映射到源文件）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 读取配置（preprocessed_data -> tasks）
    cfg = load_config(args.config)
    src_dir: Path = cfg["out_dir"]
    out_dir: Path = cfg["tasks_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 本地文件根目录：默认即 CSV 源目录，ts 的 /data/ 前缀直接映射到源文件
    document_root = args.document_root or src_dir
    if document_root != src_dir:
        logging.warning(
            "本地文件根目录 %s 与 CSV 目录 %s 不同，需自行保证文件可访问",
            document_root, src_dir,
        )

    csv_files = sorted(src_dir.glob("*.csv"))
    logging.info("CSV 目录: %s（共 %d 个文件）", src_dir, len(csv_files))
    logging.info("输出目录: %s", out_dir)
    logging.info("本地文件根目录: %s", document_root)

    tasks: list[dict] = []
    skipped = 0
    for csv_path in csv_files:
        try:
            # 本地文件 URL：相对 document_root，统一用 /data/ 前缀
            ts_url = "/data/" + quote(csv_path.name)
            task = build_task(csv_path, ts_url)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logging.warning("跳过 %s: %s", csv_path.name, exc)
            continue
        tasks.append(task)

    # 写出 JSON Lines（每行一个 task，Label Studio 可直接导入）
    out_file = out_dir / "tasks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    logging.info("完成：生成 task %d 条，跳过 %d 条，输出 %s",
                 len(tasks), skipped, out_file)
    logging.info("提示：在 Label Studio 中导入 %s 即可开始标注", out_file)
    return 0


def load_config(config_path: Path) -> dict:
    """读取 config/preprocess.json，返回源目录与 task 输出目录。"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    src_dir = Path(cfg.get("preprocessed_data", "")).expanduser()
    out_dir = Path(cfg.get("tasks", "")).expanduser()
    if not src_dir or not src_dir.exists():
        raise FileNotFoundError(f"preprocessed_data 目录不存在: {src_dir}")
    if not out_dir:
        raise ValueError("tasks 未配置")
    return {"out_dir": src_dir, "tasks_dir": out_dir}


if __name__ == "__main__":
    sys.exit(main())
