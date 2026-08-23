# -*- coding: utf-8 -*-
"""主窗口：菜单栏 + 左侧 chart_win + 右侧 info_win。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
)

from src.gui.business.cache import (
    DisplayCache,
    extract_display_state,
    has_display_state,
)
from src.gui.business.config import get_preprocessed_dir
from src.gui.business.recent_files import RecentFiles
from src.gui.ui.chart_win import ChartWin
from src.gui.ui.dialogs import AddDataDialog, DeleteDataDialog
from src.gui.ui.info_win import InfoWin

CSV_FILTER = "CSV 文件 (*.csv)"


class MainWindow(QMainWindow):
    """GUI 主窗口。"""

    def __init__(self, data_store, chart_model,
                 recent: RecentFiles | None = None,
                 cache: DisplayCache | None = None) -> None:
        super().__init__()
        self._store = data_store
        self._model = chart_model
        self._store.add_listener(self._on_store_changed)
        self._model.add_listener(self._on_model_changed)

        self.setWindowTitle("LazyShare")
        self.resize(1280, 800)

        self._recent = recent if recent is not None else RecentFiles()
        self._cache = cache if cache is not None else DisplayCache()
        self._build_menu()

        self._chart_win = ChartWin(chart_model, data_store)
        self._info_win = InfoWin(data_store)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._chart_win)
        splitter.addWidget(self._info_win)
        splitter.setSizes([900, 380])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self._on_store_changed()

    # ------------------------------------------------------------------
    # 菜单栏
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        # 文件
        file_menu = menu_bar.addMenu("文件(&F)")
        action_open = file_menu.addAction("打开文件(&O)…")
        action_open.setShortcut(QKeySequence.StandardKey.Open)
        action_open.triggered.connect(self.open_file_dialog)
        self._recent_menu = file_menu.addMenu("打开最近文件(&R)")
        file_menu.addSeparator()
        action_exit = file_menu.addAction("退出(&X)")
        action_exit.setShortcut(QKeySequence.StandardKey.Quit)
        action_exit.triggered.connect(self.close)
        self._refresh_recent_menu()

        # 编辑
        edit_menu = menu_bar.addMenu("编辑(&E)")
        self._action_add_data = edit_menu.addAction("添加数据(&D)…")
        self._action_add_data.setShortcut(QKeySequence("Ctrl+D"))
        self._action_add_data.triggered.connect(self.add_data)
        self._action_delete_data = edit_menu.addAction("删除数据(&L)…")
        self._action_delete_data.setShortcut(QKeySequence("Ctrl+Shift+D"))
        self._action_delete_data.triggered.connect(self.delete_data)

    # ------------------------------------------------------------------
    # 文件
    # ------------------------------------------------------------------
    def open_file_dialog(self) -> None:
        start_dir = get_preprocessed_dir() or Path.home()
        path, _ = QFileDialog.getOpenFileName(
            self, "打开文件", str(start_dir), CSV_FILTER)
        if not path:
            return
        self.load_file(path)

    def load_file(self, path: str | Path) -> None:
        """加载 csv 文件：打开新文件前缓存旧文件显示内容，加载后从缓存快速恢复。"""
        path = Path(path)
        # 打开新文件时，把旧文件的显示内容（sub_win / show_days 等，不含数据数值）打包缓存
        if self._store.is_loaded and self._store.file_path:
            old_state = extract_display_state(self._model)
            if has_display_state(old_state):
                self._cache.save(self._store.file_path, old_state)
        try:
            self._store.load_file(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "打开失败",
                                 f"无法加载文件：\n{path}\n\n{exc}")
            return
        # 先从缓存中找记录，有就快速恢复显示；否则清空
        cached = self._cache.load(str(path))
        if cached is not None:
            self._model.restore_state(cached)
        else:
            self._model.clear()
        self.setWindowTitle(f"LazyShare - {path.name}")
        self._recent.add(str(path))
        self._refresh_recent_menu()

    # ------------------------------------------------------------------
    # 最近文件
    # ------------------------------------------------------------------
    def _refresh_recent_menu(self) -> None:
        """刷新“打开最近文件”子菜单（最多 10 个，最新在前）。"""
        self._recent_menu.clear()
        files = self._recent.list_files()
        if not files:
            action = self._recent_menu.addAction("（无最近文件）")
            action.setEnabled(False)
            return
        for path in files:
            action = self._recent_menu.addAction(Path(path).name)
            action.setToolTip(path)
            action.triggered.connect(
                lambda checked=False, p=path: self.load_file(p))
        self._recent_menu.addSeparator()
        clear_action = self._recent_menu.addAction("清空最近文件")
        clear_action.triggered.connect(self._clear_recent)

    def _clear_recent(self) -> None:
        self._recent.clear()
        self._refresh_recent_menu()

    # ------------------------------------------------------------------
    # 编辑
    # ------------------------------------------------------------------
    def add_data(self) -> None:
        if not self._store.is_loaded:
            QMessageBox.information(
                self, "提示", "请先通过“文件 → 打开文件”加载一个 CSV 文件。")
            return
        result = AddDataDialog.ask(self._store, self._model, self)
        if not result or not result["columns"]:
            return
        self._model.add_series(
            columns=result["columns"],
            side=result["side"],
            target_name=result["target_name"],
        )
        self._save_current_cache()

    def delete_data(self) -> None:
        if not self._model.sub_wins:
            QMessageBox.information(
                self, "提示", "当前没有可删除的数据，请先“编辑 → 添加数据”。")
            return
        result = DeleteDataDialog.ask(self._model, self)
        if not result or not result["series"]:
            return
        sw = self._model.get_subwin(result["subwin_name"])
        if sw is None:
            return
        for column, side in result["series"]:
            self._model.remove_series(sw, column, side)
        self._save_current_cache()

    def _save_current_cache(self) -> None:
        """编辑后更新当前文件的缓存记录（显示内容被清空则移除缓存）。"""
        if not (self._store.is_loaded and self._store.file_path):
            return
        state = extract_display_state(self._model)
        if has_display_state(state):
            self._cache.save(self._store.file_path, state)
        else:
            self._cache.remove(self._store.file_path)

    # ------------------------------------------------------------------
    def _on_store_changed(self) -> None:
        self._action_add_data.setEnabled(self._store.is_loaded)
        self._action_delete_data.setEnabled(bool(self._model.sub_wins))

    def _on_model_changed(self) -> None:
        self._action_delete_data.setEnabled(bool(self._model.sub_wins))
