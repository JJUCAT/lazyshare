# -*- coding: utf-8 -*-
"""单个纵向小窗口：绘制点线图，支持时间柱点击显示悬浮窗。

时间轴由 ChartWin 中的共享 ChartModel 决定（show_days / offset），
因此多个 sub_win 之间天然对齐。
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..business.data_store import CLOSE_COLUMN, PEAK_LABEL_COLUMN
from .chart_utils import calc_ticks, format_tick, series_range, short_date

# 布局常量
MARGIN_LEFT = 64
MARGIN_RIGHT = 64
AXIS_BAND = 24          # 底部时间轴高度
MIN_HEIGHT = 170

# 顶部标题行与图例行高度（图例不足一行时自动换行并动态扩展顶部高度）
TITLE_ROW_H = 20
LEGEND_ROW_H = 18
LEGEND_PAD = 6

# 颜色
COLOR_BG = QColor("#1e1e1e")
COLOR_GRID = QColor("#2e2e36")
COLOR_AXIS = QColor("#6b6b76")
COLOR_TEXT = QColor("#d8d8d8")

# 绘制点数过多时的阈值（超过则关闭抗锯齿、不画数据点标记）
DENSE_THRESHOLD = 400

# 峰值标签固定大小（不受缩放影响）
PEAK_LABEL_SIZE = 14.0

# 需绘制的峰值标签：仅绘制 T（Top）/ B（Bottom），忽视 N（None）标签
PEAK_LABELS_DRAWN = ("T", "B")


class SubWinWidget(QWidget):
    """绑定一个 business.SubWin 的绘图控件。"""

    def __init__(self, subwin, model, controller) -> None:
        super().__init__()
        self.subwin = subwin          # business.chart_model.SubWin
        self._model = model           # business.chart_model.ChartModel
        self._controller = controller  # ChartWin（负责管理悬浮窗与全局高亮）
        self.setMinimumHeight(MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    # ------------------------------------------------------------------
    # 几何
    # ------------------------------------------------------------------
    def _top_margin(self) -> float:
        """顶部高度 = 标题行 + 图例行数 * 行高 + 留白（图例自动换行）。"""
        return TITLE_ROW_H + self._legend_rows() * LEGEND_ROW_H + LEGEND_PAD

    def _legend_items(self) -> list[tuple[str, str]]:
        if self.subwin is None:
            return []
        return [(s.color, s.column) for s in self.subwin.series]

    def _legend_item_width(self, fm, name: str) -> float:
        # 颜色方块(10) + 间隙(4) + 文本 + 右间距(12)
        return 14.0 + fm.horizontalAdvance(name) + 12.0

    def _legend_rows(self) -> int:
        """图例按当前宽度换行后的总行数。"""
        items = self._legend_items()
        if not items:
            return 0
        avail = max(1.0, self.width() - 20)
        fm = self.fontMetrics()
        rows = 1
        x = 10.0
        for _, name in items:
            sw = self._legend_item_width(fm, name)
            if x > 10 and x + sw > avail:
                x = 10.0
                rows += 1
            x += sw
        return rows

    def _plot_rect(self) -> QRectF:
        w = self.width()
        h = self.height()
        top = self._top_margin()
        return QRectF(MARGIN_LEFT, top,
                      max(1.0, w - MARGIN_LEFT - MARGIN_RIGHT),
                      max(1.0, h - top - AXIS_BAND))

    def _col_width(self, plot: QRectF) -> float:
        return plot.width() / max(1, self._model.show_days)

    def _x_for_col(self, plot: QRectF, col: int) -> float:
        return plot.left() + (col + 0.5) * self._col_width(plot)

    def _col_at(self, plot: QRectF, x: float) -> int:
        col = int((x - plot.left()) / self._col_width(plot))
        return max(0, min(col, self._model.show_days - 1))

    def _day_at(self, plot: QRectF, x: float) -> int:
        return self._model.offset + self._col_at(plot, x)

    # ------------------------------------------------------------------
    # 事件：时间柱点击 -> 全局高亮 + 悬浮窗
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton and pos.y() >= self._top_margin() - 4:
            plot = self._plot_rect()
            day = self._day_at(plot, pos.x())
            # 所有 sub_win 高亮同一天，并显示悬浮窗
            self._controller.set_active_day(day, self, event.globalPosition().toPoint())
        else:
            # 其他动作（右键等）关闭悬浮窗
            self._controller.clear_active_day()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        # 指针移出原时间柱（或移动到其他时间柱）-> 其他动作，关闭悬浮窗
        if self._controller.active_day is not None:
            plot = self._plot_rect()
            if plot.contains(event.position()):
                day = self._day_at(plot, event.position().x())
                if day != self._controller.active_day:
                    self._controller.clear_active_day()
            else:
                self._controller.clear_active_day()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._controller.clear_active_day()
        super().leaveEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        # 滚动属于其他动作，关闭悬浮窗
        self._controller.clear_active_day()
        super().wheelEvent(event)

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), COLOR_BG)

        model = self._model
        show = model.show_days
        if self.subwin is None or show <= 0 or model.total_days <= 0:
            painter.setPen(COLOR_TEXT)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "（暂无数据，请使用“编辑 → 添加数据”加载列项）")
            painter.end()
            return

        dense = show > DENSE_THRESHOLD
        if not dense:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        offset = model.offset
        plot = self._plot_rect()
        dates = model.data_store.get_dates()

        left_series = self.subwin.series_of_side("left")
        right_series = self.subwin.series_of_side("right")

        self._draw_title_legend(painter, plot)
        self._draw_vgrid(painter, plot, show, dense)
        self._draw_hgrid(painter, plot, left_series, right_series, offset, show)
        self._draw_highlight(painter, plot, offset, show)
        self._draw_series_lines(painter, plot, left_series, right_series, offset, show, dense)
        self._draw_peak_labels(painter, plot, offset, show)
        self._draw_axis_ticks(painter, plot, left_series, right_series, offset, show)
        self._draw_date_axis(painter, plot, dates, offset, show)
        self._draw_border(painter, plot)
        painter.end()

    def _draw_title_legend(self, painter: QPainter, plot: QRectF) -> None:
        # 标题行（左上角）
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(COLOR_TEXT)
        title_rect = QRectF(10, 2, max(10.0, self.width() - 20), TITLE_ROW_H)
        name = self.subwin.name if self.subwin.name else "未命名"
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft
                         | Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(
                             name, Qt.TextElideMode.ElideRight,
                             int(title_rect.width())))
        font.setBold(False)
        painter.setFont(font)

        # 图例行：颜色方块 + 列项名，从左往右，一行放不下自动换行，保证每个序列完整显示
        items = self._legend_items()
        if not items:
            return
        avail = max(1.0, self.width() - 20)
        x = 10.0
        y = TITLE_ROW_H + 2.0
        for color, col_name in items:
            fm = painter.fontMetrics()
            sw = self._legend_item_width(fm, col_name)
            if x > 10 and x + sw > avail:
                x = 10.0
                y += LEGEND_ROW_H
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(color))
            painter.drawRect(QRectF(x, y + 3, 10, 10))
            painter.setPen(COLOR_TEXT)
            painter.drawText(QRectF(x + 14, y - 2,
                                    fm.horizontalAdvance(col_name) + 4, LEGEND_ROW_H),
                             Qt.AlignmentFlag.AlignLeft
                             | Qt.AlignmentFlag.AlignVCenter, col_name)
            x += sw

    def _draw_vgrid(self, painter: QPainter, plot: QRectF, show: int, dense: bool) -> None:
        if dense:
            step = max(1, show // 50)
        else:
            step = 1
        pen = QPen(COLOR_GRID, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        col_w = self._col_width(plot)
        for col in range(0, show, step):
            x = plot.left() + col * col_w
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

    def _draw_hgrid(self, painter: QPainter, plot: QRectF,
                    left_series, right_series, offset: int, show: int) -> None:
        src = left_series or right_series
        vmin, vmax = series_range(src, offset, show)
        if vmin is None:
            return
        pen = QPen(COLOR_GRID, 1, Qt.PenStyle.DotLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for v in calc_ticks(vmin, vmax):
            y = plot.bottom() - (v - vmin) / (vmax - vmin) * plot.height()
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

    def _draw_highlight(self, painter: QPainter, plot: QRectF,
                        offset: int, show: int) -> None:
        """高亮全局点击的时间柱（所有 sub_win 高亮同一天）。"""
        day = self._controller.active_day if self._controller is not None else None
        if day is None:
            return
        col = day - offset
        if col < 0 or col >= show:
            return
        col_w = self._col_width(plot)
        x = plot.left() + col * col_w
        painter.fillRect(QRectF(x, plot.top(), col_w, plot.height()),
                         QColor(255, 255, 255, 22))
        pen = QPen(QColor(255, 255, 255, 130), 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        cx = x + col_w / 2.0
        painter.drawLine(QPointF(cx, plot.top()), QPointF(cx, plot.bottom()))

    def _draw_series_lines(self, painter: QPainter, plot: QRectF,
                           left_series, right_series, offset: int,
                           show: int, dense: bool) -> None:
        # 左纵列先画，右纵列后画（右纵列在上层）
        self._draw_side(painter, plot, left_series, offset, show, dense)
        self._draw_side(painter, plot, right_series, offset, show, dense)

    def _draw_peak_labels(self, painter: QPainter, plot: QRectF,
                          offset: int, show: int) -> None:
        """在“收盘价”曲线数据点上方绘制峰值标签（T 红 / B 绿，忽视 N）。

        仅当 sub_win 含有“收盘价”曲线且数据含“峰值标签”列时绘制；
        N（None）标签不绘制。
        """
        close_series = next(
            (s for s in self.subwin.series if s.column == CLOSE_COLUMN), None)
        if close_series is None:
            return
        labels = self._model.data_store.get_peak_labels()
        if labels is None or len(labels) == 0:
            return
        vmin, vmax = series_range([close_series], offset, show)
        if vmin is None:
            return
        # 固定大小、不受缩放影响；始终绘制（时间柱过窄时也显示）
        size = PEAK_LABEL_SIZE
        col_w = self._col_width(plot)
        for i in range(show):
            idx = offset + i
            if idx < 0 or idx >= len(labels):
                continue
            lab = str(labels[idx]).strip()
            if lab not in PEAK_LABELS_DRAWN:
                # 忽视 N（None）等其他标签，不绘制
                continue
            v = close_series.values[idx]
            if not np.isfinite(v):
                continue
            x = plot.left() + (i + 0.5) * col_w
            y = plot.bottom() - (float(v) - vmin) / (vmax - vmin) * plot.height()
            self._draw_peak_marker(painter, x, y, size, lab, plot)

    def _draw_peak_marker(self, painter: QPainter, x: float, y: float,
                          size: float, lab: str, plot: QRectF) -> None:
        """绘制单个峰值标签：彩色正方形 + 中间白色镂空字母。"""
        top = max(plot.top(), y - size - 3.0)
        rect = QRectF(x - size / 2.0, top, size, size)
        color = QColor("#e6194b") if lab == "T" else QColor("#3cb44b")
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRect(rect)
        painter.setPen(QColor("#ffffff"))
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(max(7, int(size * 0.58)))
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, lab)
        painter.restore()

    def _draw_side(self, painter: QPainter, plot: QRectF, series_list,
                   offset: int, show: int, dense: bool) -> None:
        if not series_list:
            return
        vmin, vmax = series_range(series_list, offset, show)
        if vmin is None:
            return
        col_w = self._col_width(plot)
        for s in series_list:
            arr = s.values
            if len(arr) == 0:
                continue
            color = QColor(s.color)
            path = QPainterPath()
            started = False
            for i in range(show):
                idx = offset + i
                if idx < 0 or idx >= len(arr):
                    started = False
                    continue
                v = arr[idx]
                if not np.isfinite(v):
                    started = False
                    continue
                x = plot.left() + (i + 0.5) * col_w
                y = plot.bottom() - (float(v) - vmin) / (vmax - vmin) * plot.height()
                if not started:
                    path.moveTo(x, y)
                    started = True
                else:
                    path.lineTo(x, y)
            pen = QPen(color, 1.6)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

            # 数据点（密集时跳过，保证性能）
            if not dense:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                r = 2.2
                for i in range(show):
                    idx = offset + i
                    if idx < 0 or idx >= len(arr):
                        continue
                    v = arr[idx]
                    if not np.isfinite(v):
                        continue
                    x = plot.left() + (i + 0.5) * col_w
                    y = plot.bottom() - (float(v) - vmin) / (vmax - vmin) * plot.height()
                    painter.drawEllipse(QPointF(x, y), r, r)

    def _draw_axis_ticks(self, painter: QPainter, plot: QRectF,
                         left_series, right_series, offset: int, show: int) -> None:
        # 左纵列刻度
        self._draw_tick_side(painter, plot, left_series, offset, show,
                             side="left")
        # 右纵列刻度
        self._draw_tick_side(painter, plot, right_series, offset, show,
                             side="right")

    def _draw_tick_side(self, painter: QPainter, plot: QRectF, series_list,
                        offset: int, show: int, side: str) -> None:
        vmin, vmax = series_range(series_list, offset, show)
        if vmin is None:
            return
        ticks = calc_ticks(vmin, vmax)
        step = (vmax - vmin) / (len(ticks) - 1) if len(ticks) > 1 else 0.0
        tick_pen = QPen(COLOR_AXIS, 1)
        tick_pen.setCosmetic(True)
        for v in ticks:
            y = plot.bottom() - (v - vmin) / (vmax - vmin) * plot.height()
            painter.setPen(tick_pen)
            if side == "left":
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.left() - 6, y))
                rect = QRectF(0, y - 9, MARGIN_LEFT - 10, 18)
                flags = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            else:
                painter.drawLine(QPointF(plot.right(), y), QPointF(plot.right() + 6, y))
                rect = QRectF(plot.right() + 8, y - 9, MARGIN_RIGHT - 12, 18)
                flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            painter.setPen(COLOR_TEXT)
            painter.drawText(rect, flags, format_tick(v, step))

    def _draw_date_axis(self, painter: QPainter, plot: QRectF,
                        dates, offset: int, show: int) -> None:
        # 底部时间轴分界线
        line_pen = QPen(COLOR_AXIS, 1)
        line_pen.setCosmetic(True)
        painter.setPen(line_pen)
        painter.drawLine(QPointF(plot.left(), plot.bottom()),
                         QPointF(plot.right(), plot.bottom()))

        label_step = max(1, show // 6)
        painter.setPen(COLOR_TEXT)
        col_w = self._col_width(plot)
        band_rect = QRectF(plot.left(), plot.bottom(),
                           plot.width(), AXIS_BAND)

        for col in range(0, show, label_step):
            idx = offset + col
            if idx < 0 or idx >= len(dates):
                continue
            x = plot.left() + (col + 0.5) * col_w
            rect = QRectF(x - col_w / 2, band_rect.top(), col_w, AXIS_BAND)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             short_date(str(dates[idx])))
        # 最后一个时间柱也标记
        last_col = show - 1
        idx = offset + last_col
        if 0 <= idx < len(dates):
            x = plot.left() + (last_col + 0.5) * col_w
            rect = QRectF(x - col_w / 2, band_rect.top(), col_w, AXIS_BAND)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             short_date(str(dates[idx])))

    def _draw_border(self, painter: QPainter, plot: QRectF) -> None:
        pen = QPen(COLOR_AXIS, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(plot)
