# -*- coding: utf-8 -*-
"""GUI 冒烟测试（离屏渲染）：主窗口构建、加载、渲染、峰值标签、最近文件、窗口显示配置。"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from src.gui.business.chart_model import ChartModel
from src.gui.business.data_store import DataStore
from src.gui.business.recent_files import RecentFiles
from src.gui.business.window_config import WindowConfig
from src.gui.ui.dialogs import AddDataDialog, DeleteDataDialog
from src.gui.ui.main_window import MainWindow

from test.test_gui_business import make_sample_csv


class GuiSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "sample.csv"
        make_sample_csv(self.path, n=120)
        self._tmp_recent = tempfile.TemporaryDirectory()
        self._tmp_window = tempfile.TemporaryDirectory()
        self.store = DataStore()
        self.model = ChartModel(self.store)
        recent = RecentFiles(Path(self._tmp_recent.name))
        window_cfg = WindowConfig(Path(self._tmp_window.name) / "gui.json")
        self.win = MainWindow(self.store, self.model, recent=recent,
                              window_cfg=window_cfg)
        self.win.load_file(self.path)

    def tearDown(self) -> None:
        self.win.close()
        self._tmp.cleanup()
        self._tmp_recent.cleanup()
        self._tmp_window.cleanup()

    def test_window_loads_and_title(self) -> None:
        self.assertTrue(self.store.is_loaded)
        self.assertIn("sample.csv", self.win.windowTitle())
        self.assertEqual(self.win._info_win._title.text(), "测试股份（600827）")

    def test_add_and_render(self) -> None:
        self.model.add_series(["M21C", "M5C"], side="left")
        self.model.add_series(["收盘价"], side="right", target_name="M21C-M5C")
        self.win.show()
        self.app.processEvents()
        chart = self.win._chart_win
        self.assertEqual(len(chart._subwin_widgets), 1)
        self.assertFalse(chart.grab().isNull())

    def test_peak_label_render(self) -> None:
        # 添加“收盘价”→ 自动显示峰值标签；渲染不崩溃
        self.model.add_series(["收盘价"], side="left")
        self.win.show()
        self.app.processEvents()
        chart = self.win._chart_win
        self.assertGreaterEqual(int((self.store.get_peak_labels() == "T").sum()), 1)
        self.assertFalse(chart.grab().isNull())
        # 缩放不影响标签：缩小到全部数据（列宽极小）仍应正常渲染
        self.model.show_days = self.model.total_days
        self.model.set_offset(0)
        self.app.processEvents()
        self.assertFalse(chart.grab().isNull())

    def test_global_highlight(self) -> None:
        self.model.add_series(["M21C"], side="left")
        self.model.add_series(["SNC"], side="left")
        self.win.show()
        self.app.processEvents()
        chart = self.win._chart_win
        sw0 = chart._subwin_widgets[0]
        day = self.model.offset
        chart.set_active_day(day, sw0, sw0.mapToGlobal(sw0.rect().center()))
        self.assertEqual(chart.active_day, day)
        self.assertIn("日期：", chart._float_win._date_label.text())
        chart.clear_active_day()
        self.assertIsNone(chart.active_day)

    def test_add_data_dialog(self) -> None:
        dlg = AddDataDialog(self.store, self.model, self.win)
        self.assertEqual(dlg.selected_columns(), [])
        items = [dlg._column_list.item(i) for i in range(dlg._column_list.count())]
        self.assertGreaterEqual(len(items), 1)
        items[0].setCheckState(Qt.CheckState.Checked)
        self.assertEqual(len(dlg.selected_columns()), 1)
        # 峰值标签不作为可选列项
        self.assertNotIn("峰值标签", [i.text() for i in items])
        dlg.close()

    def test_delete_data_dialog(self) -> None:
        self.model.add_series(["M21C", "M5C"], side="left")
        dlg = DeleteDataDialog(self.model, self.win)
        self.assertEqual(dlg._series_list.count(), 2)
        dlg._series_list.item(0).setCheckState(Qt.CheckState.Checked)
        self.assertEqual(dlg.selected_series(), [("M21C", "left")])
        dlg.close()
        self.assertTrue(self.win._action_delete_data.isEnabled())

    def test_recent_files_menu(self) -> None:
        rf = self.win._recent
        rf.clear()
        self.assertEqual(rf.list_files(), [])
        self.win.load_file(self.path)
        self.assertEqual(rf.list_files(), [str(self.path)])
        names = [a.text() for a in self.win._recent_menu.actions()
                 if not a.isSeparator() and a.text() != "清空最近文件"]
        self.assertIn(self.path.name, names)

    def test_display_config_saved_after_add(self) -> None:
        self.model.add_series(["M21C", "M5C"], side="left")
        self.win._save_display_config()
        cfg = WindowConfig(Path(self._tmp_window.name) / "gui.json").load()
        cols = [s["column"] for s in cfg["sub_wins"][0]["series"]]
        self.assertEqual(cols, ["M21C", "M5C"])
        self.assertEqual(cfg["show_days"], self.model.show_days)

    def test_zoom_saves_show_days(self) -> None:
        # 点击缩放（+）后 show_days 实时保存到 window 配置
        self.win._chart_win._btn_zoom_in.click()
        cfg = WindowConfig(Path(self._tmp_window.name) / "gui.json").load()
        self.assertEqual(cfg["show_days"], self.model.show_days)
        self.assertGreater(self.model.show_days, 21)

    def test_window_config_restores_columns_on_open(self) -> None:
        # 已有 window 显示配置时，打开 csv 文件依然显示这些列项和 show_days
        WindowConfig(Path(self._tmp_window.name) / "gui.json").save({
            "show_days": 30,
            "sub_wins": [{"series": [{"column": "M21C", "side": "left"},
                                     {"column": "收盘价", "side": "right"}]}],
        })
        self.win.load_file(self.path)
        self.assertEqual(self.win._model.show_days, 30)
        self.assertEqual(self.win._model.sub_wins[0].name, "M21C-收盘价")

    def test_display_config_updated_after_edit(self) -> None:
        self.model.add_series(["M21C", "M5C"], side="left")
        self.win._save_display_config()
        sw = self.model.get_subwin("M21C-M5C")
        self.model.remove_series(sw, "M21C")
        self.model.remove_series(sw, "M5C")
        self.win._save_display_config()
        cfg = WindowConfig(Path(self._tmp_window.name) / "gui.json").load()
        self.assertEqual(cfg.get("sub_wins"), [])


if __name__ == "__main__":
    unittest.main()
