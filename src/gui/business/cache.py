# -*- coding: utf-8 -*-
"""显示内容缓存管理（后端）。

把每个文件当前的显示配置（sub_win、show_days、offset 等）打包缓存到
config/gui.json 的 tmp 目录下，**不包含 csv 数据数值**；
最多缓存 MAX_CACHE_FILES 个文件，打开文件/最近文件时先查缓存快速恢复显示。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .chart_model import DEFAULT_SHOW_DAYS
from .config import get_tmp_dir

MAX_CACHE_FILES = 100
CACHE_DIRNAME = "cache"
INDEX_FILENAME = "cache_index.json"


def _sha1(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def extract_display_state(model) -> dict:
    """提取模型的显示状态（不含 csv 数据数值）。"""
    return {
        "show_days": model.show_days,
        "offset": model.offset,
        "sub_wins": [
            {"series": [{"column": s.column, "side": s.side} for s in sw.series]}
            for sw in model.sub_wins
        ],
    }


def has_display_state(state: dict) -> bool:
    """判断显示状态是否有值得缓存的内容。"""
    return bool(state.get("sub_wins")) or state.get("show_days") != DEFAULT_SHOW_DAYS


class DisplayCache:
    """文件显示状态缓存（最多 MAX_CACHE_FILES 个文件）。"""

    def __init__(self, tmp_dir: Path | None = None) -> None:
        self._dir = Path(tmp_dir) if tmp_dir is not None else get_tmp_dir()
        self._cache_dir = self._dir / CACHE_DIRNAME
        self._index_file = self._dir / INDEX_FILENAME
        self._index: list[str] = []
        self._load_index()

    # ------------------------------------------------------------------
    @property
    def index_file(self) -> Path:
        return self._index_file

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def list_cached(self) -> list[str]:
        """按最近缓存顺序返回文件路径。"""
        return list(self._index)

    # ------------------------------------------------------------------
    def save(self, path: str | Path, state: dict) -> None:
        """保存某文件的显示状态，并把它移到最近缓存最前、淘汰超出上限的。"""
        path_str = str(path)
        payload = {"file_path": path_str}
        payload.update(state)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file(path_str).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if path_str in self._index:
            self._index.remove(path_str)
        self._index.insert(0, path_str)
        while len(self._index) > MAX_CACHE_FILES:
            evicted = self._index.pop()
            self._cache_file(evicted).unlink(missing_ok=True)
        self._save_index()

    def load(self, path: str | Path) -> dict | None:
        """读取某文件的缓存显示状态，不存在或损坏时返回 None。"""
        f = self._cache_file(path)
        if not f.exists():
            return None
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("file_path") != str(path):
            return None
        return data

    def remove(self, path: str | Path) -> None:
        """删除某文件的缓存记录（如显示内容已被清空时）。"""
        path_str = str(path)
        self._cache_file(path_str).unlink(missing_ok=True)
        if path_str in self._index:
            self._index.remove(path_str)
            self._save_index()

    def clear(self) -> None:
        """清空全部缓存。"""
        self._index = []
        if self._cache_dir.exists():
            for f in self._cache_dir.glob("*.json"):
                f.unlink(missing_ok=True)
        self._index_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    def _cache_file(self, path: str | Path) -> Path:
        return self._cache_dir / f"{_sha1(str(path))}.json"

    def _load_index(self) -> None:
        if not self._index_file.exists():
            self._index = []
            return
        try:
            data = json.loads(self._index_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._index = [str(x) for x in data if isinstance(x, str)]
            else:
                self._index = []
        except (OSError, json.JSONDecodeError):
            self._index = []

    def _save_index(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_file.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2), encoding="utf-8")
