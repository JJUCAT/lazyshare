# -*- coding: utf-8 -*-
"""悬浮窗：鼠标点击时间柱时显示具体时间和数值。"""
from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .chart_utils import format_value, is_missing

_QSS = """
QFrame#floatWin {
    background-color: #2b2b33;
    border: 1px solid #55555f;
    border-radius: 6px;
}
QLabel {
    color: #e6e6e6;
    background: transparent;
}
QLabel#dateLabel {
    font-weight: bold;
    color: #ffffff;
}
"""


class FloatWin(QFrame):
    """无边框、置顶的悬浮信息窗口。"""

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.ToolTip
                         | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setObjectName("floatWin")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet(_QSS)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(3)

        self._date_label = QLabel("-")
        self._date_label.setObjectName("dateLabel")
        layout.addWidget(self._date_label)

        self._value_container = QVBoxLayout()
        self._value_container.setSpacing(2)
        layout.addLayout(self._value_container)

    # ------------------------------------------------------------------
    def set_content(self, date_str: str, rows: list[tuple[str, str, object]]) -> None:
        """设置内容。

        rows: [(颜色, 列项名, 原始数值), ...]
        """
        self._date_label.setText(f"日期：{date_str}")
        while self._value_container.count():
            item = self._value_container.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        if not rows:
            self._value_container.addWidget(self._make_row("#888888", "（无数据）", None))
        for color, name, value in rows:
            self._value_container.addWidget(self._make_row(color, name, value))

    def _make_row(self, color: str, name: str, value) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {color};")
        value_text = format_value(value) if not is_missing(value) else "-"
        text = QLabel(f"{name}：<b>{value_text}</b>")
        lay.addWidget(dot)
        lay.addWidget(text)
        lay.addStretch(1)
        return row

    def move_near(self, global_pos: QPoint, margin: int = 18) -> None:
        """移动到鼠标附近并保持在屏幕内。"""
        self.adjustSize()
        screen = QGuiApplication.screenAt(global_pos)
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        pos = QPoint(global_pos.x() + margin, global_pos.y() + margin)
        if pos.x() + self.width() > geo.right():
            pos.setX(geo.right() - self.width())
        if pos.y() + self.height() > geo.bottom():
            pos.setY(geo.bottom() - self.height())
        pos.setX(max(geo.left(), pos.x()))
        pos.setY(max(geo.top(), pos.y()))
        self.move(pos)
