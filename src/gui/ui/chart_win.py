# -*- coding: utf-8 -*-
"""左侧主窗口 chart_win：纵向排列多个 sub_win，共享时间轴。

- 多个纵向排列小窗口，横轴为时间轴且对齐
- 右上角 “+”（show_days 翻倍）/ “-”（show_days 减半）
- 底部横向滚动条，默认在最右侧（显示最新数据）
"""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .float_win import FloatWin
from .sub_win import SubWinWidget


class ChartWin(QWidget):
    """左侧图表主窗口。"""

    def __init__(self, model, data_store,
                 on_zoom: Callable[[], None] | None = None) -> None:
        super().__init__()
        self._model = model
        self._data_store = data_store
        self._on_zoom_cb = on_zoom  # 点击缩放（+/-）后的保存回调
        self._float_win = FloatWin()
        self._subwin_widgets: list[SubWinWidget] = []
        self._last_structure_key: tuple | None = None
        # 全局高亮的时间柱（所有 sub_win 高亮同一天）
        self._active_day: int | None = None

        self._model.add_listener(self._on_model_changed)

        # ---- 顶部：缩放按钮 ----
        header = QHBoxLayout()
        self._info_label = QLabel("")
        header.addWidget(self._info_label)
        header.addStretch(1)
        self._btn_zoom_out = QToolButton()
        self._btn_zoom_out.setText("－")
        self._btn_zoom_out.setToolTip("show_days 缩小一半")
        self._btn_zoom_out.clicked.connect(self._zoom_out)
        self._btn_zoom_in = QToolButton()
        self._btn_zoom_in.setText("＋")
        self._btn_zoom_in.setToolTip("show_days 翻倍")
        self._btn_zoom_in.clicked.connect(self._zoom_in)
        header.addWidget(self._btn_zoom_out)
        header.addWidget(self._btn_zoom_in)

        # ---- 中部：sub_win 纵向滚动区 ----
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        self._container = QWidget()
        self._subwin_layout = QVBoxLayout(self._container)
        self._subwin_layout.setContentsMargins(0, 0, 0, 0)
        self._subwin_layout.setSpacing(2)
        self._scroll_area.setWidget(self._container)

        # ---- 底部：横向时间滚动条 ----
        self._hscroll = QScrollBar(Qt.Orientation.Horizontal)
        self._hscroll.valueChanged.connect(self._on_scroll_value_changed)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)
        root.addLayout(header)
        root.addWidget(self._scroll_area, 1)
        root.addWidget(self._hscroll)

        self._on_model_changed()

    # ------------------------------------------------------------------
    # 缩放（+/-）：调整 show_days 后触发保存回调
    # ------------------------------------------------------------------
    def _zoom_out(self) -> None:
        self._model.zoom_out()
        if self._on_zoom_cb is not None:
            self._on_zoom_cb()

    def _zoom_in(self) -> None:
        self._model.zoom_in()
        if self._on_zoom_cb is not None:
            self._on_zoom_cb()

    # ------------------------------------------------------------------
    # 模型变化刷新
    # ------------------------------------------------------------------
    def _on_model_changed(self) -> None:
        # 缩放 / 滚动属于“其他动作”，清除高亮并关闭悬浮窗
        if self._active_day is not None:
            self._active_day = None
            self._float_win.hide()
        key = self._model.structure_key()
        if key != self._last_structure_key:
            self._rebuild_subwins()
            self._last_structure_key = key
        self._update_scrollbar()
        self._update_info()
        for w in self._subwin_widgets:
            w.update()

    def _rebuild_subwins(self) -> None:
        while self._subwin_layout.count():
            item = self._subwin_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._subwin_widgets.clear()
        for sw in self._model.sub_wins:
            widget = SubWinWidget(sw, self._model, self)
            self._subwin_layout.addWidget(widget)
            self._subwin_widgets.append(widget)
        if not self._subwin_widgets:
            hint = QLabel("尚未添加任何数据窗口。\n"
                          "请使用菜单“编辑 → 添加数据”选择列项加载到图表。")
            hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint.setSizePolicy(QSizePolicy.Policy.Expanding,
                               QSizePolicy.Policy.Expanding)
            self._subwin_layout.addWidget(hint)

    def _update_scrollbar(self) -> None:
        sb = self._hscroll
        max_off = self._model.max_offset
        sb.blockSignals(True)
        sb.setRange(0, max_off)
        sb.setPageStep(max(1, self._model.show_days))
        sb.setSingleStep(max(1, self._model.show_days // 20))
        sb.setValue(self._model.offset)
        sb.blockSignals(False)

    def _update_info(self) -> None:
        model = self._model
        dates = self._data_store.get_dates()
        end_idx = model.offset + model.show_days - 1
        start = str(dates[model.offset]) if 0 <= model.offset < len(dates) else "-"
        end = str(dates[end_idx]) if 0 <= end_idx < len(dates) else "-"
        self._info_label.setText(
            f"显示 {model.show_days} 日 · {start} ~ {end}"
            f"（共 {model.total_days} 日）")

    def _on_scroll_value_changed(self, value: int) -> None:
        if value != self._model.offset:
            self._model.set_offset(value)

    # ------------------------------------------------------------------
    # 时间柱高亮 + 悬浮窗管理（全局）
    # ------------------------------------------------------------------
    @property
    def active_day(self) -> int | None:
        """当前高亮的时间柱（全局 day 下标），None 表示无高亮。"""
        return self._active_day

    def set_active_day(self, day_index: int, widget: SubWinWidget, global_pos) -> None:
        """点击时间柱：所有 sub_win 高亮同一天，并显示悬浮窗。"""
        dates = self._data_store.get_dates()
        if not (0 <= day_index < len(dates)):
            return
        self._active_day = day_index
        self.show_float(widget, day_index, global_pos)
        for w in self._subwin_widgets:
            w.update()

    def clear_active_day(self) -> None:
        """其他动作：清除高亮并关闭悬浮窗。"""
        if self._active_day is None:
            self._float_win.hide()
            return
        self._active_day = None
        self._float_win.hide()
        for w in self._subwin_widgets:
            w.update()

    def show_float(self, widget: SubWinWidget, day_index: int, global_pos) -> None:
        dates = self._data_store.get_dates()
        if not (0 <= day_index < len(dates)):
            return
        rows = []
        for s in widget.subwin.series:
            if day_index < len(s.values):
                rows.append((s.color, s.column, s.values[day_index]))
            else:
                rows.append((s.color, s.column, float("nan")))
        self._float_win.set_content(str(dates[day_index]), rows)
        self._float_win.show()
        self._float_win.move_near(global_pos)

    def close_float(self) -> None:
        self._float_win.hide()
