# -*- coding: utf-8 -*-
"""金玥数据日常更新拉取主逻辑。

流程（见 pull_plan.md）：
    1. 检查 "share" 个股的数据时间（取整体最新交易日）
    2. 获取金玥数据"日常更新"文件列表（前复权）
    3. 拉取日期晚于 share 最新交易日的日文件，保存到 "download"
    4. 接口访问按 frequency 控制频率，并加随机等待

输出格式：download/<年份>/<YYYY-MM-DD>_金玥数据.csv（与 scripts/update.py 兼容）

用法：
    python3 -m src.pull.pull [--config config/pull.json] [--date YYYY-MM-DD] [--dry-run] [-v]
"""
from __future__ import annotations

import argparse
import io
import logging
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.pull.client import DyyClient
from src.pull.config import DEFAULT_CONFIG, load_config

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIMESTAMP_RE = re.compile(r" \d{2}:\d{2}:\d{2}$")


def read_last_date(path: Path) -> str:
    """读取 CSV 最后一行的日期列（YYYY-MM-DD），非日期返回空串。"""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 2048))
            data = f.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""
    lines = [ln for ln in data.splitlines() if ln.strip()]
    if not lines:
        return ""
    first = lines[-1].split(",")[0].strip().strip('"')
    return first if _DATE_RE.match(first) else ""


def share_latest_date(share_dir: Path) -> str:
    """扫描 share 目录所有股票 CSV，返回整体最新交易日；无数据返回空串。"""
    latest = ""
    for path in share_dir.glob("*.csv"):
        d = read_last_date(path)
        if d and d > latest:
            latest = d
    return latest


def normalize_csv(content: bytes) -> pd.DataFrame:
    """把下载的 CSV 字节规范化为标准 DataFrame。

    - 自动处理引号包裹的字段（utf-8-sig）
    - 列名去渠道后缀：退市时间_Eq6TNd -> 退市时间
    - 上市时间 / 退市时间去掉 " 00:00:00" 时间戳
    """
    # dtype=str：保留原始字符串格式（代码前导零、浮点原始精度），避免被重新推断/格式化
    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", dtype=str)
    df.columns = [
        re.sub(r"^退市时间.*$", "退市时间", str(c)) for c in df.columns
    ]
    for col in ("上市时间", "退市时间"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(
                _TIMESTAMP_RE, "", regex=True)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="金玥数据日常更新拉取")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="配置文件路径（默认: config/pull.json）")
    parser.add_argument("--date", type=str, default=None,
                        help="仅拉取指定日期 YYYY-MM-DD（默认按 share 最新日期增量拉取）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印将要拉取的文件，不实际下载")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    share_dir: Path = cfg["share_dir"]
    download_dir: Path = cfg["download_dir"]
    frequency: float = cfg["frequency"]
    logging.info("数据源: %s（%s）", cfg["base_url"], cfg["account"])
    logging.info("复权方式: %s（adjust=%s）", cfg["adjust_type"], cfg["adjust"])
    logging.info("share 目录: %s", share_dir)
    logging.info("download 目录: %s", download_dir)

    # 1. 检查 share 最新交易日
    latest = args.date if args.date else share_latest_date(share_dir)
    if not latest:
        logging.warning("无法确定 share 最新交易日（--date 未指定且 share 无数据）")
        return 1
    logging.info("share 最新交易日: %s", latest)

    # 2. 登录并获取日常更新文件列表
    client = DyyClient(cfg["base_url"], cfg["account"], cfg["password"])
    client.login()
    files = client.list_files()

    # 3. 确定需要拉取的文件（日期 > share 最新，且 download 中不存在）
    to_pull = []
    for filename in files:
        date = filename[:-4] if filename.endswith(".csv") else filename
        if not _DATE_RE.match(date) or date <= latest:
            continue
        year = date[:4]
        dst = download_dir / year / f"{date}_金玥数据.csv"
        if dst.exists():
            logging.info("已存在，跳过: %s", dst)
            continue
        to_pull.append((date, filename, dst))

    logging.info("待拉取文件 %d 个（日期 > %s）", len(to_pull), latest)
    if not to_pull:
        logging.info("无需拉取（已是最新）")
        return 0

    if args.dry_run:
        for date, filename, dst in to_pull:
            logging.info("[dry-run] 将拉取 %s -> %s", filename, dst)
        return 0

    # 4. 逐个拉取（frequency 秒 + 随机等待，控制接口访问频率）
    pulled = 0
    for i, (date, filename, dst) in enumerate(to_pull):
        try:
            content = client.download(filename, cfg["adjust"])
        except Exception as exc:  # noqa: BLE001
            logging.error("拉取失败 %s: %s", filename, exc)
            continue
        df = normalize_csv(content)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        pulled += 1
        logging.info("[%d/%d] 已拉取 %s（%d 行）-> %s",
                     i + 1, len(to_pull), filename, len(df), dst)
        if i < len(to_pull) - 1:
            wait = frequency + random.uniform(0, 1)
            time.sleep(wait)

    logging.info("完成：拉取 %d 个文件，共 %d 个待拉取", pulled, len(to_pull))
    return 0


if __name__ == "__main__":
    sys.exit(main())
