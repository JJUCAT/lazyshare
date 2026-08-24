# -*- coding: utf-8 -*-
"""图表窗口显示配置记录（后端）。

把 chart_win / sub_win / info_win 的显示配置（例如 sub_win 显示的列项参数、
时间轴 show_days）记录到 config/gui.json 的 "window" 字段。该配置是全局的，
对所有 csv 文件有效：添加/删除数据、点击缩放（+/-）时实时保存，打开 csv 文件时
据此恢复，因此切换文件后依然显示相同的列项参数和 show_days。只读写 gui.json 中
的 window 字段，不影响文件中的其他字段。
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import DEFAULT_GUI_CONFIG

WINDOW_KEY = "window"


def _load_gui_data(config_path: Path) -> dict:
    """读取 gui.json 的完整内容；文件不存在或损坏时返回空 dict。"""
    if not Path(config_path).exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def extract_display_config(model) -> dict:
    """从图表模型提取显示配置（时间轴 show_days + 每个 sub_win 显示的列项），不含数据数值。

    结构：{"show_days": 可见天数,
           "sub_wins": [{"series": [{"column": 列项名, "side": 纵列}, ...]}, ...]}
    """
    return {
        "show_days": model.show_days,
        "sub_wins": [
            {"series": [{"column": s.column, "side": s.side} for s in sw.series]}
            for sw in model.sub_wins
        ],
    }


class WindowConfig:
    """图表窗口显示配置，持久化到 gui.json 的 "window" 字段。"""

    def __init__(self, config_path: Path | None = None) -> None:
        self._path = Path(config_path) if config_path is not None else DEFAULT_GUI_CONFIG

    # ------------------------------------------------------------------
    @property
    def config_path(self) -> Path:
        """gui.json 配置文件路径。"""
        return self._path

    def load(self) -> dict:
        """读取 window 字段，返回 dict（未配置或字段损坏时返回空 dict）。"""
        data = _load_gui_data(self._path)
        window = data.get(WINDOW_KEY, {})
        return dict(window) if isinstance(window, dict) else {}

    def save(self, display_config: dict) -> None:
        """把显示配置写入 gui.json 的 window 字段，并保留其他字段。"""
        data = _load_gui_data(self._path)
        data[WINDOW_KEY] = dict(display_config)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
