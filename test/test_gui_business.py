# -*- coding: utf-8 -*-
"""business 层单元测试：DataStore 与 ChartModel。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gui.business.chart_model import DEFAULT_SHOW_DAYS, ChartModel
from src.gui.business.data_store import DataStore

COLUMNS = ["日期", "股票代码", "股票名称", "行业", "收盘价",
           "M21C", "M5C", "M21V", "M5V", "NC", "NV", "SNC", "SNV"]


def make_sample_csv(path: Path, n: int = 60) -> pd.DataFrame:
    """生成一份模拟预处理输出的 CSV。"""
    dates = pd.bdate_range("2025-01-02", periods=n).strftime("%Y-%m-%d")
    idx = np.arange(n)
    close = np.round(10.0 + idx * 0.1, 3)
    m21c = np.where(idx < 20, np.nan, np.round(close - 0.2, 3))
    m5c = np.where(idx < 4, np.nan, np.round(close + 0.1, 3))
    df = pd.DataFrame({
        "日期": dates,
        "股票代码": "600827",
        "股票名称": "测试股份",
        "行业": "测试行业",
        "收盘价": close,
        "M21C": m21c,
        "M5C": m5c,
        "M21V": 100000,
        "M5V": 120000,
        "NC": np.where(idx < 20, np.nan, (m5c - m21c) / m21c),
        "NV": 0.01,
        "SNC": np.where(idx < 20, np.nan, 0.0),
        "SNV": 0.01,
    })
    df = df[COLUMNS]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


class DataStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "sample.csv"
        make_sample_csv(self.path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_load_file(self) -> None:
        store = DataStore()
        store.load_file(self.path)
        self.assertTrue(store.is_loaded)
        self.assertEqual(store.file_path, str(self.path))
        self.assertEqual(store.row_count, 60)
        self.assertEqual(store.get_columns(), COLUMNS)

    def test_plot_columns_exclude_identity(self) -> None:
        store = DataStore()
        store.load_file(self.path)
        plot_cols = store.get_plot_columns()
        self.assertNotIn("日期", plot_cols)
        self.assertNotIn("股票代码", plot_cols)
        self.assertNotIn("股票名称", plot_cols)
        self.assertNotIn("行业", plot_cols)
        self.assertIn("收盘价", plot_cols)

    def test_values_keep_nan(self) -> None:
        store = DataStore()
        store.load_file(self.path)
        vals = store.get_values("M21C")
        self.assertEqual(len(vals), 60)
        self.assertTrue(np.isnan(vals[0]))       # 窗口不足
        self.assertFalse(np.isnan(vals[20]))     # 21 日后有值
        close = store.get_values("收盘价")
        self.assertAlmostEqual(close[0], 10.0)

    def test_title_info(self) -> None:
        store = DataStore()
        store.load_file(self.path)
        info = store.get_title_info()
        self.assertEqual(info["name"], "测试股份")
        self.assertEqual(info["code"], "600827")
        self.assertEqual(info["industry"], "测试行业")
        self.assertEqual(info["rows"], 60)

    def test_clear(self) -> None:
        store = DataStore()
        store.load_file(self.path)
        store.clear()
        self.assertFalse(store.is_loaded)
        self.assertIsNone(store.file_path)
        self.assertEqual(store.get_columns(), [])
        self.assertEqual(store.row_count, 0)


class ChartModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "sample.csv"
        make_sample_csv(self.path, n=60)
        self.store = DataStore()
        self.store.load_file(self.path)
        self.model = ChartModel(self.store)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_default_window(self) -> None:
        self.assertEqual(self.model.show_days, DEFAULT_SHOW_DAYS)
        self.assertEqual(self.model.total_days, 60)
        # 默认 offset 在最右侧（最新数据）
        self.assertEqual(self.model.offset, 60 - DEFAULT_SHOW_DAYS)
        self.assertEqual(self.model.max_offset, 60 - DEFAULT_SHOW_DAYS)

    def test_add_new_subwin(self) -> None:
        sw = self.model.add_series(["M21C"])
        self.assertEqual(len(self.model.sub_wins), 1)
        self.assertEqual(sw.name, "M21C")
        self.assertEqual(len(sw.series), 1)
        self.assertEqual(sw.series[0].side, "left")
        self.assertIsNotNone(sw.series[0].color)

    def test_add_multiple_columns_name(self) -> None:
        sw = self.model.add_series(["M21C", "M5C", "NC"])
        self.assertEqual(sw.name, "M21C-M5C-NC")
        self.assertEqual(len(sw.series), 3)
        # 同 sub_win 内不同列项颜色不同
        colors = {s.color for s in sw.series}
        self.assertEqual(len(colors), 3)

    def test_add_to_existing_updates_name(self) -> None:
        self.model.add_series(["M21C"])
        sw = self.model.add_series(["M5C"], target_name="M21C")
        self.assertEqual(len(self.model.sub_wins), 1)
        self.assertEqual(sw.name, "M21C-M5C")
        self.assertEqual(len(sw.series), 2)

    def test_add_duplicate_skipped(self) -> None:
        sw = self.model.add_series(["M21C", "M21C", "M5C"])
        self.assertEqual(len(sw.series), 2)

    def test_sides(self) -> None:
        sw = self.model.add_series(["M21C"], side="left")
        self.model.add_series(["收盘价"], side="right", target_name=sw.name)
        self.assertEqual(len(sw.series_of_side("left")), 1)
        self.assertEqual(len(sw.series_of_side("right")), 1)
        self.assertEqual(sw.name, "M21C-收盘价")

    def test_remove_series_updates_name(self) -> None:
        self.model.add_series(["M21C", "M5C"])
        sw = self.model.sub_wins[0]
        ok = self.model.remove_series(sw, "M21C")
        self.assertTrue(ok)
        self.assertEqual(sw.name, "M5C")
        self.assertEqual(len(sw.series), 1)

    def test_remove_last_series_removes_subwin(self) -> None:
        self.model.add_series(["M21C"])
        ok = self.model.remove_series("M21C", "M21C")
        self.assertTrue(ok)
        self.assertEqual(self.model.sub_wins, [])

    def test_remove_series_by_name_and_side(self) -> None:
        sw = self.model.add_series(["M21C"], side="left")
        self.model.add_series(["M21C"], side="right", target_name=sw.name)
        self.assertEqual(len(sw.series), 2)
        self.model.remove_series(sw.name, "M21C", side="left")
        self.assertEqual(len(sw.series), 1)
        self.assertEqual(sw.series[0].side, "right")
        self.assertEqual(sw.name, "M21C")

    def test_remove_missing_returns_false(self) -> None:
        self.model.add_series(["M21C"])
        self.assertFalse(self.model.remove_series("不存在的窗口", "M21C"))
        sw = self.model.sub_wins[0]
        self.assertFalse(self.model.remove_series(sw, "不存在列项"))

    def test_zoom_in_anchor_right(self) -> None:
        # 默认 show=21, offset=39, right_edge=60
        self.model.zoom_in()
        self.assertEqual(self.model.show_days, 42)
        self.assertEqual(self.model.offset, 60 - 42)
        self.assertEqual(self.model.offset, self.model.max_offset)

    def test_zoom_out(self) -> None:
        self.model.zoom_in()
        self.model.zoom_out()
        self.assertEqual(self.model.show_days, 21)
        self.assertEqual(self.model.offset, 39)

    def test_zoom_out_min_one(self) -> None:
        for _ in range(10):
            self.model.zoom_out()
        self.assertGreaterEqual(self.model.show_days, 1)

    def test_offset_clamped(self) -> None:
        self.model.set_offset(0)
        self.assertEqual(self.model.offset, 0)
        self.model.set_offset(999999)
        self.assertEqual(self.model.offset, self.model.max_offset)

    def test_clear(self) -> None:
        self.model.add_series(["M21C"])
        self.model.clear()
        self.assertEqual(self.model.sub_wins, [])
        self.assertEqual(self.model.show_days, DEFAULT_SHOW_DAYS)
        # 清除后默认仍显示最新数据（offset 在最右侧）
        self.assertEqual(self.model.offset, self.model.max_offset)
        self.assertEqual(self.model.offset, 39)

    def test_notify_on_change(self) -> None:
        calls = []
        self.model.add_listener(lambda: calls.append(1))
        self.model.add_series(["M21C"])
        self.model.zoom_in()
        self.model.set_offset(5)
        self.assertGreaterEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
