# -*- coding: utf-8 -*-
"""GUI 冒烟测试（使用离屏渲染，验证主窗口可构建、可加载、可渲染）。"""
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

from PySide6.QtWidgets import QApplication

from src.gui.business.cache import DisplayCache
from src.gui.business.chart_model import ChartModel
from src.gui.business.data_store import DataStore
from src.gui.business.recent_files import RecentFiles
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
        self._tmp_cache = tempfile.TemporaryDirectory()
        self.store = DataStore()
        self.model = ChartModel(self.store)
        recent = RecentFiles(Path(self._tmp_recent.name))
        cache = DisplayCache(Path(self._tmp_cache.name))
        self.win = MainWindow(self.store, self.model, recent=recent, cache=cache)
        self.win.load_file(self.path)

    def tearDown(self) -> None:
        self.win.close()
        self._tmp.cleanup()
        self._tmp_recent.cleanup()
        self._tmp_cache.cleanup()

    def test_window_loads_and_title(self) -> None:
        self.assertTrue(self.store.is_loaded)
        self.assertIn("sample.csv", self.win.windowTitle())
        self.assertEqual(self.win._info_win._title.text(), "测试股份（600827）")

    def test_add_and_render(self) -> None:
        self.model.add_series(["M21C", "M5C"], side="left")
        self.model.add_series(["收盘价"], side="right", target_name="M21C-M5C")
        self.model.add_series(["SNC"], side="left")
        self.win.show()
        self.app.processEvents()
        chart = self.win._chart_win
        self.assertEqual(len(chart._subwin_widgets), 2)
        # 强制绘制（触发 paintEvent），不应抛异常
        pixmap = chart.grab()
        self.assertFalse(pixmap.isNull())
        # 缩放与滚动后再次渲染
        self.model.zoom_in()
        self.model.zoom_out()
        self.model.set_offset(0)
        self.app.processEvents()
        self.assertFalse(chart.grab().isNull())

    def test_zoom_buttons_and_scrollbar(self) -> None:
        chart = self.win._chart_win
        before = self.model.show_days
        chart._btn_zoom_in.click()
        self.assertEqual(self.model.show_days, before * 2)
        chart._btn_zoom_out.click()
        self.assertEqual(self.model.show_days, before)
        self.assertEqual(chart._hscroll.maximum(), self.model.max_offset)

    def test_float_win_and_global_highlight(self) -> None:
        self.model.add_series(["M21C"], side="left")
        self.model.add_series(["SNC"], side="left")
        self.win.show()
        self.app.processEvents()
        chart = self.win._chart_win
        sw0 = chart._subwin_widgets[0]
        sw1 = chart._subwin_widgets[1]
        day = self.model.offset
        # 点击时间柱：所有 sub_win 高亮同一天 + 显示悬浮窗
        chart.set_active_day(day, sw0, sw0.mapToGlobal(sw0.rect().center()))
        self.assertEqual(chart.active_day, day)
        self.assertIn("日期：", chart._float_win._date_label.text())
        # 两个 sub_win 使用同一全局高亮
        self.assertEqual(sw0._controller.active_day, day)
        self.assertEqual(sw1._controller.active_day, day)
        # 其他动作清除高亮
        chart.clear_active_day()
        self.assertIsNone(chart.active_day)
        self.assertFalse(chart._float_win.isVisible())

    def test_menu_actions(self) -> None:
        self.assertTrue(self.win._action_add_data.isEnabled())
        # 初始没有 sub_win 时，删除数据不可用
        self.assertFalse(self.win._action_delete_data.isEnabled())

    def test_recent_files_menu(self) -> None:
        rf = self.win._recent
        rf.clear()
        self.assertEqual(rf.list_files(), [])
        # 加载文件后记录到最近文件，并刷新子菜单
        self.win.load_file(self.path)
        self.assertEqual(rf.list_files(), [str(self.path)])
        actions = self.win._recent_menu.actions()
        names = [a.text() for a in actions
                 if not a.isSeparator() and a.text() != "清空最近文件"]
        self.assertIn(self.path.name, names)
        # 点击最近文件菜单项重新加载
        target = next(a for a in actions
                      if not a.isSeparator() and a.text() == self.path.name)
        target.trigger()
        self.assertTrue(self.store.is_loaded)
        # 清空最近文件
        clear = next(a for a in self.win._recent_menu.actions()
                     if a.text() == "清空最近文件")
        clear.trigger()
        self.assertEqual(rf.list_files(), [])
        self.assertEqual(self.win._recent_menu.actions()[0].text(), "（无最近文件）")

    def test_cache_restore_on_reopen(self) -> None:
        # 添加显示内容并改变 show_days
        self.model.add_series(["M21C", "M5C"], side="left")
        self.model.add_series(["SNC"], side="left")
        self.model.show_days = 30
        # 重新打开同一文件：先缓存旧显示，再从缓存快速恢复
        self.win.load_file(self.path)
        self.assertEqual(self.win._model.show_days, 30)
        names = [sw.name for sw in self.win._model.sub_wins]
        self.assertIn("M21C-M5C", names)
        self.assertIn("SNC", names)
        self.assertIn(str(self.path), self.win._cache.list_cached())

    def test_cache_updated_after_edit(self) -> None:
        cache = self.win._cache
        # 编辑（添加数据）后缓存更新为最新显示内容
        self.model.add_series(["M21C", "M5C"], side="left")
        self.win._save_current_cache()
        self.assertIn(str(self.path), cache.list_cached())
        cached = cache.load(str(self.path))
        self.assertEqual(cached["sub_wins"][0]["series"][0]["column"], "M21C")
        self.assertEqual(len(cached["sub_wins"][0]["series"]), 2)
        # 再次编辑（删除全部数据源）后缓存被移除
        sw = self.model.get_subwin("M21C-M5C")
        self.model.remove_series(sw, "M21C")
        self.model.remove_series(sw, "M5C")
        self.win._save_current_cache()
        self.assertNotIn(str(self.path), cache.list_cached())
        self.assertIsNone(cache.load(str(self.path)))

    def test_delete_data_dialog(self) -> None:
        from PySide6.QtCore import Qt
        from src.gui.ui.dialogs import DeleteDataDialog

        self.model.add_series(["M21C", "M5C"], side="left")
        dlg = DeleteDataDialog(self.model, self.win)
        self.assertEqual(dlg._subwin_combo.count(), 1)
        self.assertEqual(dlg._series_list.count(), 2)
        # 勾选一个数据源
        dlg._series_list.item(0).setCheckState(Qt.CheckState.Checked)
        self.assertEqual(dlg.selected_series(), [("M21C", "left")])
        self.assertEqual(dlg.subwin_name(), "M21C-M5C")
        dlg.close()
        # 删除后模型更新：名称按剩余列项重算
        self.model.remove_series(self.model.get_subwin("M21C-M5C"), "M21C")
        self.assertEqual(self.model.sub_wins[0].name, "M5C")
        self.assertEqual(len(self.model.sub_wins[0].series), 1)
        # 删除菜单随 sub_win 存在而启用
        self.assertTrue(self.win._action_delete_data.isEnabled())

    def test_add_data_dialog(self) -> None:
        from PySide6.QtCore import Qt
        from src.gui.ui.dialogs import AddDataDialog

        # 构造不应抛异常（回归：QListWidget 不可直接迭代）
        dlg = AddDataDialog(self.store, self.model, self.win)
        self.assertEqual(dlg.selected_columns(), [])
        # 勾选两个列项
        items = [dlg._column_list.item(i) for i in range(dlg._column_list.count())]
        self.assertGreaterEqual(len(items), 1)
        items[0].setCheckState(Qt.CheckState.Checked)
        if len(items) > 1:
            items[1].setCheckState(Qt.CheckState.Checked)
        self.assertEqual(len(dlg.selected_columns()), min(2, len(items)))
        self.assertEqual(dlg.side(), "left")
        dlg._side_right.setChecked(True)
        self.assertEqual(dlg.side(), "right")
        # 全选 / 清空 按钮
        dlg._check_all_btn.click()
        self.assertEqual(len(dlg.selected_columns()), len(items))
        dlg._clear_all_btn.click()
        self.assertEqual(dlg.selected_columns(), [])
        dlg.close()


if __name__ == "__main__":
    unittest.main()
