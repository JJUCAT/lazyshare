# -*- coding: utf-8 -*-
"""显示内容缓存（business.cache）单元测试。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gui.business.cache import (
    MAX_CACHE_FILES,
    DisplayCache,
    extract_display_state,
    has_display_state,
)
from src.gui.business.chart_model import ChartModel
from src.gui.business.data_store import DataStore

from test.test_gui_business import make_sample_csv

EMPTY_STATE = {"show_days": 21, "offset": 0, "sub_wins": []}


class DisplayCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = DisplayCache(self.dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_save_load(self) -> None:
        state = {"show_days": 42, "offset": 5,
                 "sub_wins": [{"series": [{"column": "M21C", "side": "left"}]}]}
        self.cache.save("a.csv", state)
        loaded = self.cache.load("a.csv")
        self.assertEqual(loaded["show_days"], 42)
        self.assertEqual(loaded["file_path"], "a.csv")

    def test_list_order(self) -> None:
        self.cache.save("a.csv", dict(EMPTY_STATE))
        self.cache.save("b.csv", dict(EMPTY_STATE))
        self.cache.save("a.csv", dict(EMPTY_STATE))
        self.assertEqual(self.cache.list_cached(), ["a.csv", "b.csv"])

    def test_cap_evicts_oldest(self) -> None:
        for i in range(MAX_CACHE_FILES + 5):
            self.cache.save(f"f{i}.csv", dict(EMPTY_STATE))
        self.assertEqual(len(self.cache.list_cached()), MAX_CACHE_FILES)
        self.assertIsNone(self.cache.load("f0.csv"))

    def test_remove(self) -> None:
        self.cache.save("a.csv", dict(EMPTY_STATE))
        self.cache.save("b.csv", dict(EMPTY_STATE))
        self.cache.remove("a.csv")
        self.assertEqual(self.cache.list_cached(), ["b.csv"])
        self.assertIsNone(self.cache.load("a.csv"))

    def test_clear(self) -> None:
        self.cache.save("a.csv", dict(EMPTY_STATE))
        self.cache.clear()
        self.assertEqual(self.cache.list_cached(), [])


class RestoreStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "sample.csv"
        make_sample_csv(self.path, n=60)
        self.store = DataStore()
        self.store.load_file(self.path)
        self.model = ChartModel(self.store)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_round_trip(self) -> None:
        sw = self.model.add_series(["M21C", "M5C"], side="left")
        self.model.add_series(["收盘价"], side="right", target_name=sw.name)
        self.model.show_days = 30
        state = extract_display_state(self.model)
        self.assertTrue(has_display_state(state))
        m2 = ChartModel(self.store)
        m2.restore_state(state)
        self.assertEqual(m2.show_days, 30)
        self.assertIn("M21C-M5C-收盘价", [s.name for s in m2.sub_wins])

    def test_restore_skips_missing_columns(self) -> None:
        state = {"show_days": 21, "offset": 0, "sub_wins": [{"series": [
            {"column": "M21C", "side": "left"},
            {"column": "不存在的列", "side": "left"},
        ]}]}
        m2 = ChartModel(self.store)
        m2.restore_state(state)
        self.assertEqual(len(m2.sub_wins[0].series), 1)
        self.assertEqual(m2.sub_wins[0].series[0].column, "M21C")


if __name__ == "__main__":
    unittest.main()
