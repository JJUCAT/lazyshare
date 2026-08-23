# -*- coding: utf-8 -*-
"""配置加载：读取 config/preprocess.json 的 raw_data 与 preprocessed_data。"""
from __future__ import annotations

import json
from pathlib import Path

# src/preprocess/handle/config.py -> handle -> preprocess -> src -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "preprocess.json"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    """返回 {"raw_dir", "out_dir"}，路径不存在时抛异常。"""
    config_path = Path(config_path)
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
