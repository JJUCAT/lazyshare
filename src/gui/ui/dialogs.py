# -*- coding: utf-8 -*-
"""对话框：添加数据。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

# 让勾选框在暗色主题下更醒目
_DIALOG_STYLE = """
QListWidget#columnList {
    background-color: #202025;
    border: 1px solid #4a4a55;
    border-radius: 4px;
    outline: none;
}
QListWidget#columnList::item {
    padding: 4px 6px;
    color: #e6e6e6;
}
QListWidget#columnList::item:hover {
    background-color: #3a3a46;
}
QListWidget#columnList::item:selected {
    background-color: #3a3a46;
}
QListWidget#columnList::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #9a9aa5;
    border-radius: 4px;
    background: #2b2b33;
}
QListWidget#columnList::indicator:hover {
    border-color: #ffffff;
}
QListWidget#columnList::indicator:checked {
    background-color: #4363d8;
    border-color: #4363d8;
}
QPushButton#checkAllBtn, QPushButton#clearAllBtn {
    background-color: #3a3a46;
    border: 1px solid #55555f;
    border-radius: 4px;
    padding: 3px 12px;
    color: #e6e6e6;
}
QPushButton#checkAllBtn:hover, QPushButton#clearAllBtn:hover {
    background-color: #4a4a58;
    border-color: #7a7a88;
}
"""


class AddDataDialog(QDialog):
    """“编辑 → 添加数据”对话框。

    选择列项（可多选）加载到 chart_win 的某个 sub_win（或新建窗口），
    并选择加载到左纵列还是右纵列。
    """

    def __init__(self, data_store, model, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("添加数据")
        self.setMinimumWidth(420)
        self._store = data_store
        self._model = model

        root = QVBoxLayout(self)
        root.setSpacing(10)
        self.setStyleSheet(_DIALOG_STYLE)

        hint = QLabel("选择要加载到图表的列项（点击左侧方框勾选，可多选）：")
        root.addWidget(hint)

        self._column_list = QListWidget()
        self._column_list.setObjectName("columnList")
        for col in data_store.get_plot_columns():
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self._column_list.addItem(item)
        self._column_list.itemChanged.connect(self._on_item_changed)
        root.addWidget(self._column_list, 1)

        btn_row = QHBoxLayout()
        self._check_all_btn = QPushButton("全选")
        self._check_all_btn.setObjectName("checkAllBtn")
        self._check_all_btn.clicked.connect(
            lambda: self._set_all_checked(Qt.CheckState.Checked))
        self._clear_all_btn = QPushButton("清空")
        self._clear_all_btn.setObjectName("clearAllBtn")
        self._clear_all_btn.clicked.connect(
            lambda: self._set_all_checked(Qt.CheckState.Unchecked))
        btn_row.addWidget(self._check_all_btn)
        btn_row.addWidget(self._clear_all_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        form = QFormLayout()
        form.setSpacing(8)

        self._target_combo = QComboBox()
        self._target_combo.addItem("（新建窗口）", None)
        for sw in model.sub_wins:
            self._target_combo.addItem(sw.name, sw.name)
        form.addRow("加载到窗口：", self._target_combo)

        self._side_left = QRadioButton("左纵列")
        self._side_right = QRadioButton("右纵列")
        self._side_left.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self._side_left)
        group.addButton(self._side_right)
        side_box = QHBoxLayout()
        side_box.addWidget(self._side_left)
        side_box.addWidget(self._side_right)
        side_box.addStretch(1)
        form.addRow("加载到纵列：", side_box)
        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._buttons = buttons
        self._update_ok_state()

    # ------------------------------------------------------------------
    def _on_item_changed(self, _item) -> None:
        self._update_ok_state()

    def _update_ok_state(self) -> None:
        btn = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        btn.setEnabled(bool(self.selected_columns()))

    def _on_accept(self) -> None:
        if not self.selected_columns():
            return
        self.accept()

    def _set_all_checked(self, state) -> None:
        for i in range(self._column_list.count()):
            self._column_list.item(i).setCheckState(state)

    # ------------------------------------------------------------------
    def selected_columns(self) -> list[str]:
        # PySide6 的 QListWidget 不可直接迭代，需按索引取 item
        return [
            self._column_list.item(i).text()
            for i in range(self._column_list.count())
            if self._column_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def target_name(self) -> str | None:
        return self._target_combo.currentData()

    def side(self) -> str:
        return "left" if self._side_left.isChecked() else "right"

    @staticmethod
    def ask(data_store, model, parent=None) -> dict | None:
        """便捷调用：返回 {columns, target_name, side} 或 None（取消）。"""
        dlg = AddDataDialog(data_store, model, parent)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return {
            "columns": dlg.selected_columns(),
            "target_name": dlg.target_name(),
            "side": dlg.side(),
        }


class DeleteDataDialog(QDialog):
    """“编辑 → 删除数据”对话框。

    选择 sub_win 及其中要删除的数据源（可多选）。
    """

    def __init__(self, model, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("删除数据")
        self.setMinimumWidth(420)
        self._model = model
        self.setStyleSheet(_DIALOG_STYLE)

        root = QVBoxLayout(self)
        root.setSpacing(10)

        hint = QLabel("选择要删除数据源的 sub_win 与列项（点击左侧方框勾选，可多选）：")
        root.addWidget(hint)

        self._subwin_combo = QComboBox()
        self._subwin_combo.currentIndexChanged.connect(self._reload_series)
        root.addWidget(self._subwin_combo)

        self._series_list = QListWidget()
        self._series_list.setObjectName("columnList")
        self._series_list.itemChanged.connect(self._on_item_changed)
        root.addWidget(self._series_list, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._buttons = buttons
        self._reload_subwins()
        self._update_ok_state()

    # ------------------------------------------------------------------
    def _reload_subwins(self) -> None:
        self._subwin_combo.blockSignals(True)
        self._subwin_combo.clear()
        for sw in self._model.sub_wins:
            self._subwin_combo.addItem(f"{sw.name}（{len(sw.series)} 个数据源）",
                                       sw.name)
        self._subwin_combo.blockSignals(False)
        self._reload_series()

    def _reload_series(self) -> None:
        self._series_list.blockSignals(True)
        self._series_list.clear()
        name = self._subwin_combo.currentData()
        sw = self._model.get_subwin(name) if name else None
        if sw is not None:
            for s in sw.series:
                label = f"{s.column}（{'左纵列' if s.side == 'left' else '右纵列'}）"
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, (s.column, s.side))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self._series_list.addItem(item)
        self._series_list.blockSignals(False)
        self._update_ok_state()

    def _on_item_changed(self, _item) -> None:
        self._update_ok_state()

    def _update_ok_state(self) -> None:
        btn = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        btn.setEnabled(bool(self.selected_series()))

    def _on_accept(self) -> None:
        if not self.selected_series():
            return
        self.accept()

    # ------------------------------------------------------------------
    def subwin_name(self) -> str | None:
        return self._subwin_combo.currentData()

    def selected_series(self) -> list[tuple[str, str]]:
        return [
            self._series_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._series_list.count())
            if self._series_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    @staticmethod
    def ask(model, parent=None) -> dict | None:
        """便捷调用：返回 {subwin_name, series} 或 None（取消）。"""
        dlg = DeleteDataDialog(model, parent)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return {"subwin_name": dlg.subwin_name(), "series": dlg.selected_series()}


def show_error(parent, title: str, message: str) -> None:
    QMessageBox.critical(parent, title, message)
