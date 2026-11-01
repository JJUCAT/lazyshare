# -*- coding: utf-8 -*-
"""数据划分单元测试：按时间顺序划分（防未来数据泄漏）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.data import span_range, time_split


def make_spans(n: int = 100, step: int = 42, window: int = 41) -> np.ndarray:
    """构造 n 个不重叠窗口的 (首日, 末日) 数组（间隔 step 天，长度 window+1 天）。"""
    base = np.datetime64("2010-01-01")
    return np.array(
        [(base + np.timedelta64(i * step, "D"),
          base + np.timedelta64(i * step + window, "D")) for i in range(n)],
        dtype="datetime64[D]",
    )


class TimeSplitTest(unittest.TestCase):
    def test_strict_time_order(self) -> None:
        spans = make_spans(100)
        tr, va, te, info = time_split(spans, (0.6, 0.2, 0.2), purge_days=21)
        self.assertEqual(len(tr) + len(va) + len(te) + info["purged"], len(spans))
        # 标签锚点（窗口末日）严格递增：train < valid < test，测试集时间最新
        self.assertLess(spans[tr, 1].max(), spans[va, 1].min())
        self.assertLess(spans[va, 1].max(), spans[te, 1].min())
        self.assertGreater(spans[te, 1].max(), spans[tr, 1].max())
        # 三集合互斥
        self.assertEqual(len(set(tr.tolist()) & set(va.tolist())), 0)
        self.assertEqual(len(set(va.tolist()) & set(te.tolist())), 0)
        self.assertEqual(len(set(tr.tolist()) & set(te.tolist())), 0)

    def test_label_horizon_never_crosses_boundary(self) -> None:
        """train 的最晚末日 + 前瞻窗口 必须早于 valid 的最早末日（防标签前瞻泄漏）。"""
        spans = make_spans(100)
        tr, va, te, info = time_split(spans, (0.6, 0.2, 0.2), purge_days=21)
        # 21 个交易日 → 日历间隔约 31 天
        self.assertGreaterEqual(info["gap_days"], 30)
        gap = np.timedelta64(info["gap_days"], "D")
        self.assertLess(spans[tr, 1].max() + gap, spans[va, 1].min())
        self.assertLess(spans[va, 1].max() + gap, spans[te, 1].min())
        # 只裁掉跨界样本：丢弃数量应远小于集合规模
        self.assertLess(info["purged"], len(spans) // 4)

    def test_without_purge(self) -> None:
        spans = make_spans(50)
        tr, va, te, info = time_split(spans, (0.6, 0.2, 0.2), purge_days=0)
        self.assertEqual(info["purged"], 0)
        self.assertEqual(info["gap_days"], 0)
        self.assertEqual(len(tr) + len(va) + len(te), len(spans))

    def test_span_range(self) -> None:
        spans = make_spans(10)
        start, end = span_range(spans, [0, 1, 2])
        self.assertEqual(start, str(spans[0, 0]))
        self.assertEqual(end, str(spans[2, 1]))
        self.assertEqual(span_range(spans, []), ("-", "-"))

    def test_invalid_ratios(self) -> None:
        spans = make_spans(10)
        with self.assertRaises(ValueError):
            time_split(spans, (0.5, 0.5, 0.5), purge_days=0)

    def test_empty_span_raises(self) -> None:
        with self.assertRaises(ValueError):
            time_split(np.zeros((0, 2), dtype="datetime64[D]"), (0.6, 0.2, 0.2), 0)


if __name__ == "__main__":
    unittest.main()
