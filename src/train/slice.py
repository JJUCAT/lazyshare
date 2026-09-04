# -*- coding: utf-8 -*-
"""数据切片：把预处理 CSV 按固定窗口切成样本文件（tsai 分类训练用）。

读取 config/classify_train.json：
    - data_source：预处理 CSV 目录（源）
    - dataset    ：切片输出目录
    - slice      ：{"seq_len": 21, "jump_step": 21, "throw_last_one": true}
                   窗口大小 / 滑动步长 / 是否丢弃末尾对齐窗口
    - items      ：特征列（不含"日期"，日期仅作标识/时间轴）
    - label      ：标签列（随窗口保留，默认"峰值标签"）

处理规则（见 tsai_classify_train_plan.md）：
    1. 忽略"日期"作为特征，从 items 参数完整的时间开始切片：
       即从所有 items 特征列都非 NaN 的第一天开始（前面指标未形成完整值）。
    2. 按窗口 seq_len、滑动 jump_step 切片，每个窗口为一个样本文件。
    3. 切片保存为 CSV，必须保留"日期"列；文件名 = 源文件名-开始日期-结束日期.csv。
    4. 同一个数据源（源 CSV）的切片保存在 dataset 路径下的同名文件夹中：
       输出路径 = dataset/<源文件名>/<源文件名>-开始日期-结束日期.csv。
    5. 正向平铺切完后若末尾剩余不足一个完整窗口（seq_len），按 throw_last_one
       决定是否补一个"以文件最后一天收尾"的对齐窗口样本：
       - true（默认）：丢弃，只保留正向平铺的窗口；
       - false：补生成该完整窗口（起点 len-seq_len，与前面平铺窗口尾部重叠），
         使训练样本覆盖"预测最新一天"的情形。

用法：
    python3 scripts/tsai/slice.py [--config config/classify_train.json] [--limit N] [--workers N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATE_COL = "日期"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "classify_train.json"


def load_train_config(config_path: Path) -> dict:
    """读取 config/classify_train.json 并校验关键字段。"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_source = Path(cfg.get("data_source", "")).expanduser()
    dataset = Path(cfg.get("dataset", "")).expanduser()
    if not data_source or not data_source.exists():
        raise FileNotFoundError(f"data_source 目录不存在: {data_source}")
    if not dataset:
        raise ValueError("dataset 未配置")

    slice_cfg = cfg.get("slice") or {}
    seq_len = int(slice_cfg.get("seq_len", 63))
    jump_step = int(slice_cfg.get("jump_step", 21))
    # 是否丢弃"以最新一天收尾的对齐窗口"（见 tsai_classify_train_plan.md），默认 true
    throw_last_one = bool(slice_cfg.get("throw_last_one", True))
    # items / label 已整合进 slice；兼容旧格式（顶层 items / label）
    items = slice_cfg.get("items") or cfg.get("items") or []
    label = (slice_cfg.get("label") or cfg.get("label") or ["峰值标签"])[0]
    if not items:
        raise ValueError("items 未配置")
    if DATE_COL not in items:
        items = [DATE_COL] + items  # 保证日期列参与输出，但不作为数值特征
    if seq_len <= 0 or jump_step <= 0:
        raise ValueError("slice.seq_len / slice.jump_step 必须为正整数")

    return {
        "data_source": data_source,
        "dataset": dataset,
        "items": items,
        "label": label,
        "seq_len": seq_len,
        "jump_step": jump_step,
        "throw_last_one": throw_last_one,
    }


SIGNATURE_FILE = ".slice_signature.json"


def dataset_signature(cfg: dict) -> dict:
    """由切片参数生成格式指纹，用于判断本地 dataset 与 config 是否匹配。

    只要 seq_len / jump_step / 特征列 / 标签列 / throw_last_one 任一变化，
    旧切片文件（行数、列、末尾窗口策略）即失效，需要整体重建。
    """
    return {
        "seq_len": cfg["seq_len"],
        "jump_step": cfg["jump_step"],
        "items": [c for c in cfg["items"] if c != DATE_COL],
        "label": cfg["label"],
        "throw_last_one": cfg["throw_last_one"],
    }


def read_dataset_signature(dataset_dir: Path) -> dict | None:
    """读取 dataset 目录下的格式签名；不存在或损坏时返回 None。"""
    p = dataset_dir / SIGNATURE_FILE
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            sig = json.load(f)
        return sig if isinstance(sig, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_dataset_signature(dataset_dir: Path, sig: dict) -> None:
    """把本次切片的格式签名写入 dataset 目录（下次运行据此校验）。"""
    with open(dataset_dir / SIGNATURE_FILE, "w", encoding="utf-8") as f:
        json.dump(sig, f, ensure_ascii=False, indent=2)


def wipe_dataset(dataset_dir: Path) -> int:
    """删除 dataset 下全部内容（源子目录、窗口 csv、旧签名文件）。

    返回删除的文件数量。
    """
    removed = 0
    for p in dataset_dir.iterdir():
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    f.unlink(missing_ok=True)
                    removed += 1
            p.rmdir()
        else:
            p.unlink(missing_ok=True)
            removed += 1
    return removed


def ensure_dataset_consistent(dataset_dir: Path, cfg: dict) -> None:
    """切片前检查 dataset 格式与 config 是否一致，不一致则整体清理再切片。

    - dataset 为空：直接切片；
    - 签名存在且与当前参数一致：保留（仍会逐个刷新源的窗口文件）；
    - 无签名（旧版本产物）或签名不一致：格式不匹配，整体清理后重建。
    """
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sig = dataset_signature(cfg)
    if not any(dataset_dir.iterdir()):
        logging.info("dataset 为空，跳过格式检查")
        return
    cur = read_dataset_signature(dataset_dir)
    if cur == sig:
        logging.info("dataset 格式与 config 匹配（seq_len=%d, jump_step=%d）",
                     cfg["seq_len"], cfg["jump_step"])
        return
    removed = wipe_dataset(dataset_dir)
    logging.warning("dataset 格式与 config 不匹配：期望 %s，实际 %s；清理 %d 个文件后重新切片",
                    sig, cur, removed)


def slice_file(task: tuple[Path, dict]) -> dict:
    """把单个预处理 CSV 切成多个窗口样本文件。

    task: (源文件路径, 配置 dict)
    返回状态字典。
    """
    raw_file, cfg = task
    dataset_dir: Path = cfg["dataset"]
    items: list[str] = cfg["items"]
    label: str = cfg["label"]
    seq_len: int = cfg["seq_len"]
    jump_step: int = cfg["jump_step"]
    throw_last_one: bool = cfg["throw_last_one"]

    # 特征列：去掉"日期"（日期仅标识，不参与数值特征）
    feat_cols = [c for c in items if c != DATE_COL]

    try:
        df = pd.read_csv(raw_file, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "file": raw_file.name, "message": f"读取失败: {exc}"}
    if df.empty:
        return {"status": "empty", "file": raw_file.name, "message": "空文件"}

    # 校验必需列存在
    need_cols = [DATE_COL] + feat_cols + ([label] if label else [])
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        return {"status": "error", "file": raw_file.name,
                "message": f"缺少列: {missing}"}

    # 按日期升序排序
    df = df.assign(_dt=pd.to_datetime(df[DATE_COL], errors="coerce"))
    df = df.sort_values("_dt", na_position="last").reset_index(drop=True)

    # 从 items 特征列完整（全非 NaN）的第一天开始
    valid = df[feat_cols].notna().all(axis=1)
    if not valid.any():
        return {"status": "skip", "file": raw_file.name,
                "message": "items 特征列无完整数据"}
    start0 = int(valid.idxmax())

    # 输出列：日期 + 特征 + 标签
    out_cols = [DATE_COL] + feat_cols + ([label] if label else [])
    stem = raw_file.stem
    # 同一个数据源（源 CSV）的切片保存在 dataset 下的同名文件夹
    out_subdir = dataset_dir / stem
    out_subdir.mkdir(parents=True, exist_ok=True)

    n_windows = 0
    removed_single = 0
    aligned_windows = 0

    def save_window(s: int, aligned: bool = False) -> None:
        """保存起点为 s 的完整窗口；整个窗口同一标签（无分类信息）则跳过。"""
        nonlocal n_windows, removed_single, aligned_windows
        win = df.iloc[s:s + seq_len][out_cols].copy()
        # 切片后检查：整个窗口都是同一个标签（无分类信息）则删除，不生成
        if label:
            win_labels = win[label].dropna().astype(str).str.strip()
            if win_labels.nunique() <= 1:
                removed_single += 1
                return
        start_date = str(win[DATE_COL].iloc[0])
        end_date = str(win[DATE_COL].iloc[-1])
        out_path = out_subdir / f"{stem}-{start_date}-{end_date}.csv"
        win.to_csv(out_path, index=False, encoding="utf-8-sig")
        n_windows += 1
        if aligned:
            aligned_windows += 1

    # 1) 正向平铺：从 items 特征完整日 start0 起，按 jump_step 依次切完整窗口
    for s in range(start0, len(df) - seq_len + 1, jump_step):
        save_window(s)

    # 2) throw_last_one=False：正向平铺未覆盖到文件末尾时，补一个
    #    "以文件最后一天收尾"的完整窗口（起点 len(df)-seq_len），
    #    与前面的平铺窗口尾部重叠；若该起点已被平铺窗口覆盖则跳过，避免重复。
    if not throw_last_one:
        s_end = len(df) - seq_len
        if s_end >= start0 and (s_end - start0) % jump_step != 0:
            save_window(s_end, aligned=True)

    return {"status": "ok", "file": raw_file.name,
            "windows": n_windows, "removed_single": removed_single,
            "aligned": aligned_windows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="预处理 CSV 滑动窗口切片（tsai 训练数据准备）")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="配置文件路径（默认: config/preprocess.json）",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="仅处理前 N 个文件（调试用）")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="并行 worker 数量（默认取 CPU 核心数的一半）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_train_config(args.config)
    data_source: Path = cfg["data_source"]
    dataset_dir: Path = cfg["dataset"]
    # 切片前核对 dataset 格式与 config 是否一致；不一致则整体清理，避免旧格式样本混入训练
    ensure_dataset_consistent(dataset_dir, cfg)

    raw_files = sorted(data_source.glob("*.csv"))
    if args.limit is not None:
        raw_files = raw_files[: args.limit]

    n_workers = args.workers if args.workers else max(1, (__import__("os").cpu_count() or 1) // 2)
    logging.info("数据源目录: %s（共 %d 个文件）", data_source, len(raw_files))
    logging.info("输出目录: %s", dataset_dir)
    logging.info("窗口: seq_len=%d, jump_step=%d", cfg["seq_len"], cfg["jump_step"])
    logging.info("throw_last_one=%s（false 时补生成以最新一天收尾的对齐窗口）", cfg["throw_last_one"])
    logging.info("特征列: %s", [c for c in cfg["items"] if c != DATE_COL])
    logging.info("标签列: %s", cfg["label"])
    logging.info("并行 worker 数量: %d", n_workers)

    # 清理旧切片：本次处理的源文件夹下已有窗口文件会被重新生成，
    # 避免"删除单标签窗口"逻辑生效前生成的旧文件（如全 N 窗口）残留
    cleaned = 0
    for f in raw_files:
        sub = dataset_dir / f.stem
        if sub.is_dir():
            for old in sub.glob("*.csv"):
                old.unlink(missing_ok=True)
                cleaned += 1
    if cleaned:
        logging.info("清理旧窗口文件 %d 个", cleaned)

    tasks = [(f, cfg) for f in raw_files]
    processed = 0
    skipped = 0
    total_windows = 0
    removed_single = 0
    total_aligned = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in executor.map(slice_file, tasks, chunksize=1):
            if result["status"] != "ok":
                skipped += 1
                logging.warning("跳过 %s: %s", result["file"], result["message"])
                continue
            processed += 1
            total_windows += result.get("windows", 0)
            removed_single += result.get("removed_single", 0)
            total_aligned += result.get("aligned", 0)
            if processed % 200 == 0:
                logging.info("已切片 %d 个文件，生成窗口 %d 个...", processed, total_windows)

    logging.info("完成：切片文件 %d 个，跳过 %d 个，生成窗口样本 %d 个"
                 "（含末尾对齐窗口 %d 个；删除单标签窗口 %d 个）",
                 processed, skipped, total_windows, total_aligned, removed_single)

    # 记录切片格式签名，供下次运行校验 dataset 与 config 是否匹配
    write_dataset_signature(dataset_dir, dataset_signature(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
