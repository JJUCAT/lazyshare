# -*- coding: utf-8 -*-
"""数据缓存层：加载并缓存当前打开的 CSV 文件数据。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# CSV 中的标识列（不作为图表数值列）
IDENTITY_COLUMNS = ("日期", "股票代码", "股票名称", "行业")

# 峰值标签列（T/B/N 字符串，不作为普通数值曲线，用于收盘价上方标注）
PEAK_LABEL_COLUMN = "峰值标签"
CLOSE_COLUMN = "收盘价"


class DataStore:
    """保存当前打开的 CSV 文件数据，并提供给 UI / 模型使用。

    使用简单的观察者模式：监听方通过 add_listener 注册回调，
    数据加载 / 清空后触发通知。
    """

    def __init__(self) -> None:
        self.file_path: str | None = None
        self._df: pd.DataFrame | None = None
        self._values: dict[str, np.ndarray] = {}
        self._dates: np.ndarray | None = None
        self._peak_labels: np.ndarray | None = None
        self._listeners: list[callable] = []

    # ------------------------------------------------------------------
    # 观察者
    # ------------------------------------------------------------------
    def add_listener(self, fn: callable) -> None:
        self._listeners.append(fn)

    def _notify(self) -> None:
        for fn in list(self._listeners):
            fn()

    # ------------------------------------------------------------------
    # 加载 / 清空
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._df is not None

    def load_file(self, path: str | Path) -> None:
        """加载 csv 文件，预计算数值列缓存。"""
        path = Path(path)
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.empty:
            raise ValueError("CSV 文件为空")
        self.file_path = str(path)
        self._df = df
        self._cache_columns()
        self._notify()

    def clear(self) -> None:
        """清理缓存的旧数据。"""
        self.file_path = None
        self._df = None
        self._values = {}
        self._dates = None
        self._peak_labels = None
        self._notify()

    # ------------------------------------------------------------------
    # 数据访问
    # ------------------------------------------------------------------
    @property
    def row_count(self) -> int:
        return 0 if self._df is None else len(self._df)

    @property
    def df(self) -> pd.DataFrame | None:
        return self._df

    def get_columns(self) -> list[str]:
        return list(self._df.columns) if self._df is not None else []

    def get_plot_columns(self) -> list[str]:
        """可加载到图表的数值列（排除日期/代码/名称/行业等标识列）。"""
        cols = []
        for c in self.get_columns():
            if c in IDENTITY_COLUMNS:
                continue
            if c in self._values:
                cols.append(c)
        return cols

    def get_dates(self) -> np.ndarray:
        return self._dates if self._dates is not None else np.array([], dtype=object)

    def get_values(self, column: str) -> np.ndarray:
        """返回某列的 float 数组（缺失值为 NaN），列不存在时返回空数组。"""
        return self._values.get(column, np.array([]))

    def get_peak_labels(self) -> np.ndarray:
        """返回峰值标签数组（'' / 'T' / 'B' / 'N'），无此列时返回空数组。

        调用方（sub_win 绘制）应忽视 N（None）标签，仅绘制 T / B。
        """
        return (self._peak_labels if self._peak_labels is not None
                else np.array([], dtype=object))

    def get_title_info(self) -> dict:
        """从 CSV 列项读取标题信息（股票名称、代码、行业、日期范围）。"""
        if self._df is None or self._df.empty:
            return {}
        row = self._df.iloc[0]
        info = {
            "name": str(row.get("股票名称", "") or "").strip(),
            "code": str(row.get("股票代码", "") or "").strip(),
            "industry": str(row.get("行业", "") or "").strip(),
        }
        if self._dates is not None and len(self._dates):
            info["date_start"] = str(self._dates[0])
            info["date_end"] = str(self._dates[-1])
        info["rows"] = self.row_count
        info["columns"] = self.get_columns()
        return info

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _cache_columns(self) -> None:
        self._values = {}
        self._peak_labels = np.array([], dtype=object)
        for col in self._df.columns:
            if col == PEAK_LABEL_COLUMN:
                self._peak_labels = np.array(
                    [str(x) if pd.notna(x) else "" for x in self._df[col]],
                    dtype=object,
                )
                continue
            try:
                arr = pd.to_numeric(self._df[col], errors="coerce").to_numpy(
                    dtype=float
                )
            except (TypeError, ValueError):
                arr = np.array([], dtype=float)
            self._values[col] = arr
        dates = self._df.get("日期", pd.Series(dtype=object))
        self._dates = np.array([str(d) for d in dates], dtype=object)
