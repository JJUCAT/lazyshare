# -*- coding: utf-8 -*-
"""最近文件记录（business.recent_files）单元测试。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gui.business.recent_files import MAX_RECENT, RecentFiles


class RecentFilesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_add_order_and_dedupe(self) -> None:
        rf = RecentFiles(self.dir)
        rf.add("a.csv")
        rf.add("b.csv")
        rf.add("c.csv")
        self.assertEqual(rf.list_files(), ["c.csv", "b.csv", "a.csv"])
        rf.add("a.csv")
        self.assertEqual(rf.list_files(), ["a.csv", "c.csv", "b.csv"])

    def test_cap(self) -> None:
        rf = RecentFiles(self.dir)
        for i in range(MAX_RECENT + 5):
            rf.add(f"f{i}.csv")
        files = rf.list_files()
        self.assertEqual(len(files), MAX_RECENT)
        self.assertEqual(files[0], f"f{MAX_RECENT + 4}.csv")

    def test_persist(self) -> None:
        rf = RecentFiles(self.dir)
        rf.add("a.csv")
        rf.add("b.csv")
        rf2 = RecentFiles(self.dir)
        self.assertEqual(rf2.list_files(), ["b.csv", "a.csv"])

    def test_clear(self) -> None:
        rf = RecentFiles(self.dir)
        rf.add("a.csv")
        rf.clear()
        self.assertEqual(rf.list_files(), [])


if __name__ == "__main__":
    unittest.main()
