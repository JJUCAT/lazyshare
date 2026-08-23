# -*- coding: utf-8 -*-
"""最近打开文件记录。

记录保存在 config/gui.json 中 tmp 目录下的 recent_files.json，
最多保留 MAX_RECENT 个，最新打开的在最前。
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import get_tmp_dir

MAX_RECENT = 10
RECENT_FILENAME = "recent_files.json"


class RecentFiles:
    """最近打开文件列表（含持久化）。"""

    def __init__(self, tmp_dir: Path | None = None) -> None:
        self._dir = Path(tmp_dir) if tmp_dir is not None else get_tmp_dir()
        self._file = self._dir / RECENT_FILENAME
        self._items: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    @property
    def file_path(self) -> Path:
        """记录文件路径。"""
        return self._file

    def list_files(self) -> list[str]:
        """按最近打开顺序返回文件路径（最新在前）。"""
        return list(self._items)

    def add(self, path: str | Path) -> None:
        """记录一次打开（去重、最新在前、最多 MAX_RECENT 个）。"""
        path_str = str(Path(path))
        if path_str in self._items:
            self._items.remove(path_str)
        self._items.insert(0, path_str)
        del self._items[MAX_RECENT:]
        self._save()

    def clear(self) -> None:
        """清空最近记录。"""
        self._items = []
        self._save()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._file.exists():
            self._items = []
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._items = [str(x) for x in data if isinstance(x, str)]
            else:
                self._items = []
        except (OSError, json.JSONDecodeError):
            self._items = []

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file.write_text(
            json.dumps(self._items, ensure_ascii=False, indent=2),
            encoding="utf-8")
