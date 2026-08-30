#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""更新 share 目录下的股票数据。

配置文件: config/update.json
    "update": 新数据目录（按年份子目录存放 "YYYY-MM-DD_金玥数据.csv" 日线文件）
    "share" : 待更新目录（按股票存放 "代码-名称.csv"）

规则:
1. 股票代码是股票的唯一标识，股票名称不是。
2. 遍历 share 下的股票，从 update 中获取该代码的新数据并追加（按日期去重）。
3. 检查 update 中是否有新股票，有则创建对应文件写入 share。

用法:
    python3 scripts/update.py [--config config/update.json] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "update.json"

# 6 位数字股票代码
_CODE_RE = re.compile(r"^\d{6}$")
# 日文件命名: YYYY-MM-DD_金玥数据.csv
_DAY_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_")
# 现有 share 文件名: 代码_xxx.csv 或 代码-名称.csv
_SHARE_FILE_RE = re.compile(r"^(\d{6})(?:_|-)")


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> tuple[Path, Path]:
    """返回 (update_dir, share_dir)。"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    update_dir = Path(cfg.get("update", "")).expanduser()
    share_dir = Path(cfg.get("share", "")).expanduser()
    if not update_dir.exists():
        raise FileNotFoundError(f"update 目录不存在: {update_dir}")
    if not share_dir.exists():
        raise FileNotFoundError(f"share 目录不存在: {share_dir}")
    return update_dir, share_dir


def sanitize_name(name: str) -> str:
    """清洗名称中不能用于文件名的字符。"""
    return re.sub(r'[\\/:*?"<>|\r\n\t]', "_", str(name)).strip()


def iter_day_files(update_dir: Path) -> list[tuple[str, Path]]:
    """遍历 update 目录下所有 "YYYY-MM-DD_金玥数据.csv" 日文件，返回 [(日期, 路径)]。"""
    files = []
    for root, _dirs, names in os.walk(update_dir):
        for fn in names:
            if not fn.endswith(".csv"):
                continue
            m = _DAY_FILE_RE.match(fn)
            if m:
                files.append((m.group(1), Path(root) / fn))
    return sorted(files)


def read_csv(path: Path) -> tuple[list[str] | None, list[list[str]]]:
    """读取 CSV，返回 (表头, 数据行)。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    return header, rows


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """写回 CSV（utf-8）。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def align_row(row: list[str], row_header: list[str], target_header: list[str], fill: str = "") -> list[str]:
    """将 row 按 row_header 对齐，映射到 target_header 的列顺序。"""
    col_map = {name: i for i, name in enumerate(row_header)}
    out = []
    for col in target_header:
        idx = col_map.get(col)
        out.append(row[idx] if idx is not None and idx < len(row) else fill)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="更新 share 目录下的股票数据")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要执行的操作，不修改文件")
    args = parser.parse_args(argv)

    update_dir, share_dir = load_config(args.config)

    # ---- 1. 收集 update 所有日文件，建立 code -> {date: row} ----
    day_files = iter_day_files(update_dir)
    print(f"[1/3] 扫描 update 日文件: {len(day_files)} 个")

    upd_header: list[str] | None = None
    upd_rows: dict[str, dict[str, list[str]]] = {}
    for date, path in day_files:
        header, rows = read_csv(path)
        if header is None:
            continue
        if upd_header is None:
            upd_header = header
        try:
            ci = header.index("代码")
            di = header.index("日期")
        except ValueError:
            continue
        for row in rows:
            if len(row) <= max(ci, di):
                continue
            code = row[ci]
            if not _CODE_RE.match(code):
                continue
            upd_rows.setdefault(code, {})[row[di]] = row

    print(f"      update 中唯一股票: {len(upd_rows)} 个")

    # ---- 2. share 现有股票 ----
    share_files: dict[str, Path] = {}
    for fn in os.listdir(share_dir):
        m = _SHARE_FILE_RE.match(fn)
        if m and fn.endswith(".csv"):
            share_files[m.group(1)] = share_dir / fn

    print(f"[2/3] share 现有股票文件: {len(share_files)} 个")

    # share 标准表头：取任一现有 share 文件；若无则以 update 表头 + "是否融资融券"
    std_header: list[str] | None = None
    if share_files:
        first_path = next(iter(share_files.values()))
        h, _ = read_csv(first_path)
        if h:
            std_header = h
    if std_header is None and upd_header is not None:
        std_header = list(upd_header) + ["是否融资融券"]

    # ---- 3. 更新已有股票 + 新增新股票 ----
    updated_codes: list[str] = []
    new_codes: list[str] = []
    total_appended = 0
    skipped = 0

    for code, date_rows in sorted(upd_rows.items()):
        path = share_files.get(code)

        if path is None:
            # 新股票：创建 代码-名称.csv
            if std_header is None:
                skipped += 1
                continue
            latest_date = sorted(date_rows)[-1]
            latest_row = date_rows[latest_date]
            ni = upd_header.index("名称") if "名称" in (upd_header or []) else -1
            latest_name = sanitize_name(latest_row[ni]) if ni >= 0 and ni < len(latest_row) else code

            rows = [align_row(date_rows[d], upd_header, std_header) for d in sorted(date_rows)]
            if "名称" in std_header:
                ni2 = std_header.index("名称")
                for r in rows:
                    r[ni2] = latest_name
            di = std_header.index("日期")
            rows.sort(key=lambda r: r[di] if di < len(r) else "")

            new_path = share_dir / f"{code}-{latest_name}.csv"
            if new_path.exists():
                print(f"      !! 新股票目标已存在，跳过: {new_path.name}")
                skipped += 1
                continue

            if args.dry_run:
                print(f"      新增 {code}-{latest_name}.csv: {len(rows)} 行")
            else:
                write_csv(new_path, std_header, rows)
            new_codes.append(code)
            total_appended += len(rows)
            continue

        # 已有股票：追加新日期数据
        header, rows = read_csv(path)
        if header is None:
            skipped += 1
            continue
        try:
            ci = header.index("代码")
            ni = header.index("名称")
            di = header.index("日期")
        except ValueError:
            skipped += 1
            continue

        existing_dates = {row[di] for row in rows if len(row) > di}
        latest_name = rows[-1][ni] if rows and len(rows[-1]) > ni else None

        to_append: list[list[str]] = []
        for d in sorted(date_rows):
            if d in existing_dates:
                continue
            new_row = align_row(date_rows[d], upd_header, header)
            if latest_name is not None:
                new_row[ni] = latest_name
            to_append.append(new_row)

        if not to_append:
            continue

        rows.extend(to_append)
        rows.sort(key=lambda r: r[di] if di < len(r) else "")

        if args.dry_run:
            print(f"      更新 {path.name}: 追加 {len(to_append)} 行")
        else:
            write_csv(path, header, rows)
        updated_codes.append(code)
        total_appended += len(to_append)

    # ---- 汇总 ----
    print(f"[3/3] 完成")
    print(f"      更新已有股票: {len(updated_codes)} 个，追加 {total_appended} 行")
    print(f"      新增股票    : {len(new_codes)} 个")
    print(f"      跳过        : {skipped} 个")
    for c in new_codes:
        print(f"        新股票: {c}")
    if args.dry_run:
        print("      (dry-run，未写入任何文件)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
