# -*- coding: utf-8 -*-
"""读取配置文件（config/preprocess.json），供 GUI 使用。"""
from __future__ import annotations

import json
from pathlib import Path

# src/gui/business/config.py -> src/gui/business -> src/gui -> src -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "preprocess.json"
DEFAULT_GUI_CONFIG = PROJECT_ROOT / "config" / "gui.json"
DEFAULT_TMP_DIR = PROJECT_ROOT / "test_output"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    """读取配置文件，返回 dict（文件不存在或解析失败时返回空 dict）。"""
    if not Path(config_path).exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def get_preprocessed_dir(config_path: Path = DEFAULT_CONFIG) -> Path | None:
    """返回配置中 preprocessed_data 路径（不存在配置时返回 None）。"""
    cfg = load_config(config_path)
    raw = cfg.get("preprocessed_data", "")
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.exists() else None


def load_gui_config(config_path: Path = DEFAULT_GUI_CONFIG) -> dict:
    """读取 config/gui.json，失败时返回空 dict。"""
    if not Path(config_path).exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def get_tmp_dir(config_path: Path = DEFAULT_GUI_CONFIG) -> Path:
    """返回 gui.json 中 tmp 目录（不存在时创建；未配置则用默认目录）。

    该目录当前用于存放最近打开文件记录（RecentFiles）。
    """
    cfg = load_gui_config(config_path)
    raw = cfg.get("tmp", "")
    if raw:
        p = Path(raw).expanduser()
    else:
        p = DEFAULT_TMP_DIR
    p.mkdir(parents=True, exist_ok=True)
    return p
