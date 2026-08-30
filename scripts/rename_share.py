#!/usr/bin/env python3
"""将 /Users/jucat/data/ashare/share 中的 CSV 文件重命名为 "代码-名称.csv"。

读取每个 CSV 的表头定位 "代码" / "名称" 列，取第一行数据，重命名文件。
"""
import csv
import os
import re

SRC_DIR = "/Users/jucat/data/ashare/share"


def sanitize(name: str) -> str:
    """去除不能用于文件名的字符。"""
    return re.sub(r'[\\/:*?"<>|\r\n]', "_", name)


def main() -> None:
    renamed = 0
    skipped = 0
    errors = []

    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(SRC_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    skipped += 1
                    continue

                # 优先按表头定位列，找不到则按位置取第 2、3 列
                try:
                    code_idx = header.index("代码")
                    name_idx = header.index("名称")
                except ValueError:
                    code_idx, name_idx = 1, 2

                row = next(reader, None)
                if row is None:
                    skipped += 1
                    continue

                code = row[code_idx].strip()
                name = row[name_idx].strip()
                if not code or not name:
                    skipped += 1
                    continue

                new_name = f"{code}-{sanitize(name)}.csv"
                new_path = os.path.join(SRC_DIR, new_name)
                if os.path.abspath(new_path) == os.path.abspath(path):
                    skipped += 1  # 已经是目标命名
                    continue
                if os.path.exists(new_path):
                    errors.append(f"{fname} -> 目标已存在: {new_name}")
                    continue

                os.rename(path, new_path)
                renamed += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{fname}: {e}")

    print(f"renamed: {renamed}, skipped: {skipped}, errors: {len(errors)}")
    for msg in errors[:50]:
        print("ERROR:", msg)


if __name__ == "__main__":
    main()
