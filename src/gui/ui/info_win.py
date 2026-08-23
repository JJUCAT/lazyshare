# -*- coding: utf-8 -*-
"""右侧副窗口 info_win：标题从 CSV 列项读取显示股票名称、代码。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class InfoWin(QWidget):
    """显示当前打开股票的信息。"""

    def __init__(self, data_store) -> None:
        super().__init__()
        self._store = data_store
        self._store.add_listener(self._refresh)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        title = QLabel("未加载数据")
        title.setObjectName("infoTitle")
        title.setWordWrap(True)
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        root.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(sep)

        self._title = title
        self._details = QLabel("")
        self._details.setWordWrap(True)
        self._details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._details)

        col_title = QLabel("数据列项")
        col_title.setObjectName("infoSubtitle")
        root.addWidget(col_title)
        self._columns = QLabel("")
        self._columns.setWordWrap(True)
        self._columns.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(self._columns)
        root.addWidget(scroll, 1)

        self._refresh()

    def _refresh(self) -> None:
        info = self._store.get_title_info()
        if not info:
            self._title.setText("未加载数据")
            self._details.setText(
                "请通过“文件 → 打开文件”选择一个预处理后的 CSV 文件。")
            self._columns.setText("")
            return
        # 标题从 csv 列项读取显示股票名称、代码
        self._title.setText(f"{info.get('name', '-')}（{info.get('code', '-')}）")
        lines = []
        if info.get("industry"):
            lines.append(f"行业：{info['industry']}")
        if info.get("date_start") and info.get("date_end"):
            lines.append(f"数据范围：{info['date_start']} ~ {info['date_end']}")
        lines.append(f"数据行数：{info.get('rows', 0)}")
        lines.append(f"文件：{self._store.file_path or '-'}")
        self._details.setText("\n".join(lines))
        self._columns.setText("\n".join(f"· {c}" for c in info.get("columns", [])))
