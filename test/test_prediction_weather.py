# -*- coding: utf-8 -*-
"""大盘 / 热门行业 / 热门票统计（prediction.weather）单元测试。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.weather import (build_weather_lines,
                                    build_weather_lines_from_file)


class WeatherStatsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.weather_path = self.dir / "weather.csv"
        self.weather_path.write_text(
            "日期,大盘,行业A,行业B\n"
            "2026-09-01,100,10,10\n"
            "2026-09-02,100,10,10\n"
            "2026-09-03,200,30,5\n",
            encoding="utf-8",
        )
        self.cfg = {
            "weather": str(self.weather_path),
            "evaluate": {
                "weather": {
                    "global": {"days": 2},
                    "industrys": {"days": 2, "tops": 3},
                    "shares": {"days": 2, "tops": 2},
                }
            },
        }
        # windows: stem -> (csv 路径, 最新交易日)
        self.windows: dict[str, tuple[Path, str]] = {}
        self._add_window("600001-热门股", "600001", "行业A", [10, 30])
        self._add_window("600002-缩量股", "600002", "行业A", [100, 10])
        self._add_window("600003-冷门股", "600003", "行业B", [10, 10])
        self._add_window("600004-停更股", "600004", "行业A", [99, 99],
                         end_date="2026-08-01")  # 非最新交易日，应被剔除

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _add_window(self, stem: str, code: str, industry: str,
                    volumes: list[float], end_date: str = "2026-09-03") -> None:
        lines = ["日期,股票代码,股票名称,行业,成交量"]
        for i, vol in enumerate(volumes):
            lines.append(f"2026-09-0{i + 1},{code},{stem.split('-')[1]},{industry},{vol}")
        path = self.dir / f"{stem}.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.windows[stem] = (path, end_date)

    def test_output_contains_global_industry_and_share(self) -> None:
        text = "\n".join(build_weather_lines(self.cfg, self.windows))
        self.assertIn("大盘交易行情概况:", text)
        # 大盘：当前 200 万股 > 最近 2 天均值 150 万股
        self.assertIn("大于均值", text)
        # 热门行业：仅行业A 当前(30) > 均值(20)，行业B 当前(5) < 均值(7.5) 未入选
        self.assertIn("1. 行业A", text)
        self.assertNotIn("1. 行业B", text)
        # 热门票：行业A 内仅 600001 当前(30) > 均值(20)，600002 缩量被剔除
        self.assertIn("600001", text)
        self.assertNotIn("600002", text)

    def test_stale_stock_excluded(self) -> None:
        text = "\n".join(build_weather_lines(self.cfg, self.windows))
        self.assertNotIn("600004", text)

    def test_missing_weather_path(self) -> None:
        cfg = {"evaluate": {"weather": {"global": {"days": 2}}}}
        lines = build_weather_lines(cfg, {})
        self.assertTrue(lines)  # 仍输出提示行
        self.assertIn("大盘交易行情概况:", lines[0])

    def test_no_config(self) -> None:
        self.assertEqual(build_weather_lines({}, self.windows), [])

    def test_from_file_reads_weather_section(self) -> None:
        """load_config 会丢弃 weather 段，build_weather_lines_from_file 需直读原始 JSON。"""
        config_path = self.dir / "classify_train.json"
        config_path.write_text(json.dumps(self.cfg, ensure_ascii=False), encoding="utf-8")
        text = "\n".join(build_weather_lines_from_file(config_path, self.windows))
        self.assertIn("大盘交易行情概况:", text)
        self.assertIn("1. 行业A", text)
        self.assertIn("600001", text)

    def test_from_file_missing_file(self) -> None:
        self.assertEqual(
            build_weather_lines_from_file(self.dir / "nope.json", self.windows), [])

    def test_parsed_config_without_weather(self) -> None:
        """归一化后的训练配置（evaluate 为列表）不含 weather，应返回空列表。"""
        parsed = {"weather": str(self.weather_path), "evaluate": ["accuracy"]}
        self.assertEqual(build_weather_lines(parsed, self.windows), [])


if __name__ == "__main__":
    unittest.main()
