# -*- coding: utf-8 -*-
"""business 层单元测试：DataStore / ChartModel / 预处理峰值标签。"""
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
from src.preprocess.handle.indicators import is_st_stock
from src.preprocess.label.peak import compute_peak_labels

COLUMNS = ["日期", "股票代码", "股票名称", "行业", "收盘价",
           "M21C", "M5C", "M21V", "M5V", "NC", "NV", "SNC", "SNV",
           "峰值标签"]


def make_sample_csv(path: Path, n: int = 60) -> pd.DataFrame:
    """生成一份模拟预处理输出的 CSV（含峰值标签列）。"""
    dates = pd.bdate_range("2025-01-02", periods=n).strftime("%Y-%m-%d")
    idx = np.arange(n)
    close = np.round(10.0 + idx * 0.1, 3)
    m21c = np.where(idx < 20, np.nan, np.round(close - 0.2, 3))
    m5c = np.where(idx < 4, np.nan, np.round(close + 0.1, 3))
    labels = [""] * n
    for j in range(5, n, 12):
        labels[j] = "T"
    for j in range(9, n, 12):
        labels[j] = "B"
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
        "峰值标签": labels,
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
        self.assertEqual(store.row_count, 60)
        self.assertEqual(store.get_columns(), COLUMNS)

    def test_plot_columns_exclude_identity_and_peak(self) -> None:
        store = DataStore()
        store.load_file(self.path)
        plot_cols = store.get_plot_columns()
        self.assertNotIn("日期", plot_cols)
        self.assertNotIn("股票代码", plot_cols)
        self.assertNotIn("股票名称", plot_cols)
        self.assertNotIn("行业", plot_cols)
        self.assertNotIn("峰值标签", plot_cols)  # 标签列不作为普通曲线
        self.assertIn("收盘价", plot_cols)

    def test_values_keep_nan(self) -> None:
        store = DataStore()
        store.load_file(self.path)
        vals = store.get_values("M21C")
        self.assertTrue(np.isnan(vals[0]))
        self.assertFalse(np.isnan(vals[20]))

    def test_peak_labels(self) -> None:
        store = DataStore()
        store.load_file(self.path)
        labels = store.get_peak_labels()
        self.assertEqual(len(labels), 60)
        self.assertIn("T", set(labels))
        self.assertIn("B", set(labels))

    def test_title_info(self) -> None:
        store = DataStore()
        store.load_file(self.path)
        info = store.get_title_info()
        self.assertEqual(info["name"], "测试股份")
        self.assertEqual(info["code"], "600827")

    def test_clear(self) -> None:
        store = DataStore()
        store.load_file(self.path)
        store.clear()
        self.assertFalse(store.is_loaded)
        self.assertEqual(store.get_columns(), [])
        self.assertEqual(store.get_peak_labels().size, 0)


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
        self.assertEqual(self.model.offset, 60 - DEFAULT_SHOW_DAYS)

    def test_add_and_name(self) -> None:
        sw = self.model.add_series(["M21C", "M5C", "NC"])
        self.assertEqual(sw.name, "M21C-M5C-NC")
        self.assertEqual(len(sw.series), 3)

    def test_zoom_anchor_right(self) -> None:
        self.model.zoom_in()
        self.assertEqual(self.model.show_days, 42)
        self.assertEqual(self.model.offset, self.model.max_offset)

    def test_remove_series(self) -> None:
        sw = self.model.add_series(["M21C", "M5C"])
        self.model.remove_series(sw, "M21C")
        self.assertEqual(self.model.sub_wins[0].name, "M5C")
        self.model.remove_series(sw, "M5C")
        self.assertEqual(self.model.sub_wins, [])

    def test_restore_state(self) -> None:
        sw = self.model.add_series(["M21C", "M5C"], side="left")
        self.model.add_series(["收盘价"], side="right", target_name=sw.name)
        self.model.show_days = 30
        from src.gui.business.cache import extract_display_state
        state = extract_display_state(self.model)
        m2 = ChartModel(self.store)
        m2.restore_state(state)
        self.assertEqual(m2.show_days, 30)
        self.assertEqual(m2.sub_wins[0].name, "M21C-M5C-收盘价")


class PreprocessHelperTest(unittest.TestCase):
    def test_peak_labels(self) -> None:
        n = 100
        base = 10.0 + np.arange(n) * 0.1
        spike = np.zeros(n)
        spike[::25] = 5.0
        close = base + spike
        m21c = pd.Series(close).rolling(21, min_periods=21).mean().to_numpy()
        labels = compute_peak_labels(close, m21c, k=21)
        self.assertEqual(len(labels), n)
        self.assertIn("T", labels)
        self.assertIn("B", labels)

    def test_is_st_stock(self) -> None:
        df = pd.DataFrame({"名称": ["平安银行"], "是否ST": ["否"]})
        self.assertFalse(is_st_stock(df))
        df2 = pd.DataFrame({"名称": ["ST星源"], "是否ST": ["否"]})
        self.assertTrue(is_st_stock(df2))
        df3 = pd.DataFrame({"名称": ["某公司"], "是否ST": ["是"]})
        self.assertTrue(is_st_stock(df3))


if __name__ == "__main__":
    unittest.main()
