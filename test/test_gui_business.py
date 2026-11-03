# -*- coding: utf-8 -*-
"""business 层单元测试：DataStore / ChartModel / 预处理三元屏蔽标签。"""
from __future__ import annotations

import json
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
from src.gui.business.window_config import WindowConfig, extract_display_config
from src.preprocess.handle.indicators import compute_indicators, is_skipped_stock
from src.preprocess.label.triple_barrier import (
    compute_triple_barrier_labels,
    normalize_label_config,
)

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
        state = extract_display_config(self.model)
        m2 = ChartModel(self.store)
        m2.restore_state(state)
        # 恢复显示列项和 show_days
        self.assertEqual(m2.show_days, 30)
        self.assertEqual(m2.sub_wins[0].name, "M21C-M5C-收盘价")


class WindowConfigTest(unittest.TestCase):
    """验证图表显示配置记录到 gui.json 的 window 字段（对所有 csv 有效）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self._tmp.name) / "gui.json"
        self.window_cfg = WindowConfig(self.config_path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_save_load_roundtrip(self) -> None:
        config = {
            "show_days": 42,
            "sub_wins": [
                {"series": [{"column": "M21C", "side": "left"},
                            {"column": "收盘价", "side": "right"}]},
                {"series": [{"column": "NC", "side": "left"}]},
            ],
        }
        self.window_cfg.save(config)
        self.assertEqual(self.window_cfg.load(), config)

    def test_preserve_other_fields(self) -> None:
        # 先写入其他字段（模拟 tmp 等），再保存 window 不应覆盖
        self.config_path.write_text(
            '{"tmp": "/some/path", "window": {}}', encoding="utf-8")
        self.window_cfg.save({
            "sub_wins": [{"series": [{"column": "M21C", "side": "left"}]}],
        })
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(data["tmp"], "/some/path")
        self.assertEqual(data["window"]["sub_wins"][0]["series"][0]["column"], "M21C")

    def test_extract_display_config(self) -> None:
        # 从图表模型提取 show_days 与 sub_win 显示的列项参数（不含数据数值）
        path = Path(self._tmp.name) / "sample.csv"
        make_sample_csv(path)
        store = DataStore()
        store.load_file(path)
        model = ChartModel(store)
        model.add_series(["M21C", "M5C"], side="left")
        model.add_series(["收盘价"], side="right", target_name="M21C-M5C")
        cfg = extract_display_config(model)
        self.assertEqual(cfg["show_days"], DEFAULT_SHOW_DAYS)
        self.assertEqual(
            cfg["sub_wins"][0]["series"],
            [{"column": "M21C", "side": "left"},
             {"column": "M5C", "side": "left"},
             {"column": "收盘价", "side": "right"}],
        )

    def test_load_missing_returns_empty(self) -> None:
        self.assertEqual(self.window_cfg.load(), {})

    def test_load_corrupt_returns_empty(self) -> None:
        self.config_path.write_text("{ not valid json", encoding="utf-8")
        self.assertEqual(self.window_cfg.load(), {})


class PreprocessHelperTest(unittest.TestCase):
    def test_triple_barrier_labels(self) -> None:
        # 简单三元屏蔽（盈亏 = 未来 LU 累加）：止盈 LU>=0.3 → B，止损 LU<=-0.2 → T
        lu = [0.0, 0.1, 0.1, 0.2, 0.05, 0.05, 0.0, -0.1, -0.1, -0.1, -0.1]
        cfg = {
            "take_profit": 0.3,
            "take_profit_timeout": 3,
            "stop_loss": -0.2,
            "stop_loss_timeout": 3,
        }
        labels = compute_triple_barrier_labels(lu, cfg)
        self.assertEqual(len(labels), len(lu))
        # 末尾 max(止盈时限, 止损时限) 行前瞻窗口不完整，标签为空（预处理会丢弃）
        self.assertEqual(list(labels[-3:]), ["", "", ""])
        # 第 0 天：3 日内 LU 累计 0.1+0.1+0.2=0.4 >= 0.3 → 止盈 B
        self.assertEqual(labels[0], "B")
        # 第 3 天：3 日内 LU 累计仅 0.10 → 未触发 → N
        self.assertEqual(labels[3], "N")
        # 第 7 天：3 日内 LU 累计 -0.1-0.1-0.1=-0.3 <= -0.2 → 止损 T
        self.assertEqual(labels[7], "T")

    def test_label_config_defaults(self) -> None:
        cfg = normalize_label_config(None)
        self.assertEqual(cfg["take_profit"], 0.30)
        self.assertEqual(cfg["take_profit_timeout"], 21)
        self.assertEqual(cfg["stop_loss"], 0.20)
        self.assertEqual(cfg["stop_loss_timeout"], 15)
        self.assertEqual(cfg["values"],
                         {"take_profit": "B", "stop_loss": "T", "timeout": "N"})
        # LU 累计阈值：不再有 price（价格列）配置
        self.assertNotIn("price", cfg)
        # stop_loss 正负号等价（统一取绝对值）
        self.assertEqual(normalize_label_config({"stop_loss": -0.2})["stop_loss"], 0.2)
        self.assertEqual(normalize_label_config({"stop_loss": 0.2})["stop_loss"], 0.2)
        with self.assertRaises(ValueError):
            normalize_label_config({"take_profit": 0})
        with self.assertRaises(ValueError):
            normalize_label_config({"take_profit_timeout": 0})
        with self.assertRaises(ValueError):
            normalize_label_config({"stop_loss": 0})
        with self.assertRaises(ValueError):
            normalize_label_config({"stop_loss_timeout": 0})

    def test_separate_windows(self) -> None:
        """止盈 / 止损各自独立窗口：超出各自时限的 LU 累计不计入。"""
        cfg = {
            "take_profit": 0.3,
            "take_profit_timeout": 10,
            "stop_loss": -0.2,
            "stop_loss_timeout": 5,
        }
        # 涨幅出现在 k=8（> 止损时限 5，但 <= 止盈时限 10）→ 止盈 B
        lu = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]
        labels = compute_triple_barrier_labels(lu, cfg)
        self.assertEqual(labels[0], "B")
        # 第 0 天跌幅出现在 k=6（> 止损时限 5）→ 不算止损，标 N
        # 第 1 天同一跌幅出现在 k=5（<= 5）→ 止损 T
        lu = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
        labels = compute_triple_barrier_labels(lu, cfg)
        self.assertEqual(labels[0], "N")
        self.assertEqual(labels[1], "T")
        # LU 缺失（前瞻窗口内有 NaN）的行无效
        lu = [0.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        labels = compute_triple_barrier_labels(lu, cfg)
        self.assertEqual(labels[0], "")
        # 数据过短时全部为空
        self.assertEqual(list(compute_triple_barrier_labels(
            [0.0, 0.0, 0.0], cfg)), ["", "", ""])

    def test_lu_olu_hlu_llu(self) -> None:
        """LU / OLU / HLU / LLU 均按 max_limit × C_{t-1} 归一。"""
        df = pd.DataFrame({
            "日期": ["2025-01-02", "2025-01-03", "2025-01-06"],
            "收盘价": [10.0, 11.0, 9.9],
            "开盘价": [10.0, 10.5, 10.2],
            "最高价": [10.5, 11.2, 10.3],
            "最低价": [9.8, 10.4, 9.8],
            "成交量（股）": [100, 110, 120],
        })
        ind = compute_indicators(df, "000001")   # 沪深主板，max_limit = 10%
        # 第 0 日没有昨收，四项均缺失
        for key in ("lu", "olu", "hlu", "llu"):
            self.assertTrue(np.isnan(ind[key].iloc[0]))
        # 第 1 日：分母 = 0.10 × 10.0 = 1.0
        self.assertAlmostEqual(ind["lu"].iloc[1], (11.0 - 10.0) / 1.0, places=9)
        self.assertAlmostEqual(ind["olu"].iloc[1], (10.5 - 10.0) / 1.0, places=9)
        self.assertAlmostEqual(ind["hlu"].iloc[1], (11.0 - 11.2) / 1.0, places=9)
        self.assertAlmostEqual(ind["llu"].iloc[1], (11.0 - 10.4) / 1.0, places=9)
        # HLU 恒 <= 0（看空回落），LLU 恒 >= 0（看多反弹）
        self.assertTrue((ind["hlu"].dropna() <= 0).all())
        self.assertTrue((ind["llu"].dropna() >= 0).all())

    def test_is_skipped_stock(self) -> None:
        df = pd.DataFrame({"名称": ["平安银行"], "是否ST": ["否"]})
        self.assertFalse(is_skipped_stock(df))
        df2 = pd.DataFrame({"名称": ["ST星源"], "是否ST": ["否"]})
        self.assertTrue(is_skipped_stock(df2))
        df3 = pd.DataFrame({"名称": ["某公司"], "是否ST": ["是"]})
        self.assertTrue(is_skipped_stock(df3))


if __name__ == "__main__":
    unittest.main()
