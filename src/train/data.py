# -*- coding: utf-8 -*-
"""训练/验证公共模块：配置加载、窗口样本加载、数据集划分。

切片后的 dataset 目录结构：
    dataset/<源文件名>/<源文件名>-开始日期-结束日期.csv
每个窗口 csv 为 seq_len 行，含列：日期 + 特征列 + 标签列。

提供：
    - load_config()          读取 config/classify_train.json 完整配置
    - collect_samples()      从 dataset 读取所有窗口样本，组装 X/y
    - window_last_label()    取窗口最后一天的标签（严格 T/B/N，与逐日推理语义一致）
    - train_valid_split()    按 seed 随机划分训练/验证集
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "classify_train.json"

DATE_COL = "日期"
# 空标签归一为 None 类，保证分类任务类别完整
EMPTY_LABEL = "None"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    """读取 config/classify_train.json，返回完整配置 dict。"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_source = Path(cfg.get("data_source", "")).expanduser()
    dataset = Path(cfg.get("dataset", "")).expanduser()
    validation = Path(cfg.get("validation", "")).expanduser()
    models = Path(cfg.get("models", "")).expanduser()
    if not data_source or not data_source.exists():
        raise FileNotFoundError(f"data_source 目录不存在: {data_source}")
    if not dataset.exists():
        raise FileNotFoundError(f"dataset 目录不存在: {dataset}")
    if not validation:
        raise ValueError("validation 未配置")
    if not models:
        raise ValueError("models 未配置")

    slice_cfg = cfg.get("slice") or {}
    seq_len = int(slice_cfg.get("seq_len", 63))
    # items / label 已整合进 slice；兼容旧格式（顶层 items / label）
    items = slice_cfg.get("items") or cfg.get("items") or []
    label = (slice_cfg.get("label") or cfg.get("label") or ["峰值标签"])[0]
    if not items:
        raise ValueError("items 未配置")

    train_cfg = cfg.get("train") or {}
    test_cfg = train_cfg.get("test") or {}
    # train_max：随机取 train_max 个 dataset 样本作为总训练源（测试用）；<0 表示用完整数据
    train_max = int(test_cfg.get("train_max", -1))
    # 损失函数标签权重：{标签: 权重}；None 表示不加权（默认 CrossEntropyLoss）
    loss_label_weight = train_cfg.get("loss_label_weight") or None

    # 随机种子：显式配置整数则固定可复现；缺省 / null / "auto" / "random"
    # 用当前时间生成随机种子，使每次训练抽样的训练集不同
    seed_cfg = train_cfg.get("seed", "auto")
    if seed_cfg is None or str(seed_cfg).strip().lower() in ("", "auto", "random"):
        seed = int(time.time_ns() % (2**31))
    else:
        seed = int(seed_cfg)

    # evaluate：新格式 {"labels": [...], "items": [...], "accuracy_interval": [...]}；
    # 旧格式为评分项列表
    evaluate_cfg = cfg.get("evaluate") or {}
    if isinstance(evaluate_cfg, list):
        evaluate_items = [str(x) for x in evaluate_cfg]
        evaluate_labels: list[str] = []
        accuracy_intervals: list[float] = []
    else:
        evaluate_items = [str(x) for x in (evaluate_cfg.get("items") or [])]
        evaluate_labels = [str(x) for x in (evaluate_cfg.get("labels") or [])]
        # 不同置信度阈值：统计该置信度以上的标签占比与精度
        accuracy_intervals = [float(x) for x in (evaluate_cfg.get("accuracy_interval") or [])]

    parsed = {
        "data_source": data_source,
        "dataset": dataset,
        "validation": validation,
        "models": models,
        "items": [c for c in items if c != DATE_COL],  # 特征列排除日期
        "label": label,
        "seq_len": seq_len,
        "arch": train_cfg.get("arch", "InceptionTimePlus"),
        "tfms": train_cfg.get("tfms", "TSClassification"),
        "batch_tfms": train_cfg.get("batch_tfms", "TSStandardize"),
        "batch_size": int(train_cfg.get("batch_size", 32)),
        "epochs": int(train_cfg.get("epochs", 100)),
        "learning_rate": float(train_cfg.get("learning_rate", 0.001)),
        "validation_set": float(train_cfg.get("validation_set", 0.1)),
        "seed": seed,
        "train_max": train_max,
        "loss_label_weight": loss_label_weight,
        "evaluate": evaluate_items,
        "evaluate_labels": evaluate_labels,
        "accuracy_intervals": accuracy_intervals,
    }
    return parsed


def window_last_label(df, label_col: str, seq_len: int) -> str:
    """取窗口最后一天（第 seq_len 行）的标签，严格用原值（T / B / N）。

    与逐日推理语义一致：用前 seq_len 天（含当天）的特征预测最后一天的类别，
    因此训练标签也取最后一天的峰值标签，而不是"窗口内最后一个非空标签"。
    值为空 / NaN 时归为 None 类。
    """
    raw = df[label_col].iloc[seq_len - 1]
    s = "" if pd.isna(raw) else str(raw).strip()
    return s if s else EMPTY_LABEL


def window_label(label_values) -> str:
    """（兼容旧逻辑）从窗口的标签列取最后一个非空标签；全空则为 None 类。"""
    last = None
    for v in label_values:
        s = "" if pd.isna(v) else str(v).strip()
        if s:
            last = s
    return last if last else EMPTY_LABEL


def _collect_paths(dataset_dir: Path, max_dirs: int | None = None) -> list[Path]:
    """收集 dataset 下所有窗口 csv 路径（按文件夹排序）。"""
    dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if max_dirs is not None:
        dirs = dirs[:max_dirs]
    paths = []
    for d in dirs:
        paths.extend(sorted(d.glob("*.csv")))
    return paths


def collect_samples(dataset_dir: Path, items: list[str], label: str,
                    seq_len: int, max_samples: int | None = None,
                    max_dirs: int | None = None,
                    train_max: int | None = None, seed: int = 42,
                    ) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """从 dataset 读取窗口样本。

    - train_max >= 0：随机抽取 train_max 个样本作为总训练源（测试用）
    - train_max < 0  或 None：使用 dataset 完整数据
    - max_samples / max_dirs：调试用上限

    返回 (X, y, paths)：
        X    : float32 数组 (n_samples, n_vars, seq_len)
        y    : 字符串标签数组 (n_samples,)
        paths: 每个样本对应的窗口 csv 路径
    """
    all_paths = _collect_paths(dataset_dir, max_dirs=max_dirs)
    if not all_paths:
        raise ValueError(f"dataset 目录下未找到任何窗口 csv: {dataset_dir}")

    # 随机抽取 train_max 个（先收集全量路径再按 seed 抽样）
    if train_max is not None and train_max >= 0:
        rng = np.random.default_rng(seed)
        if train_max < len(all_paths):
            all_paths = list(rng.choice(all_paths, size=train_max, replace=False))
        # train_max 大于等于样本总数时直接用全部
        logging.info("train_max=%d：随机抽取 %d 个样本作为总训练源", train_max, len(all_paths))

    X, y, paths = [], [], []
    n = 0
    for csv_path in all_paths:
        if max_samples is not None and n >= max_samples:
            break
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            logging.warning("读取失败 %s: %s", csv_path, exc)
            continue
        if df.empty or len(df) < seq_len:
            continue
        x = df[items].head(seq_len).to_numpy(dtype=np.float32).T  # (vars, seq_len)
        # 训练标签严格取窗口最后一天（第 seq_len 行）的峰值标签
        y.append(window_last_label(df, label, seq_len))
        X.append(x)
        paths.append(csv_path)
        n += 1

    if n == 0:
        raise ValueError(f"dataset 目录下未读取到任何样本: {dataset_dir}")
    return np.stack(X), np.array(y), paths


def train_valid_split(n_samples: int, valid_ratio: float, seed: int):
    """按 seed 随机划分训练/验证索引。

    返回 (train_idx, valid_idx)，均为升序整数数组。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_valid = int(round(n_samples * valid_ratio))
    valid_idx = np.sort(idx[:n_valid])
    train_idx = np.sort(idx[n_valid:])
    return train_idx, valid_idx
