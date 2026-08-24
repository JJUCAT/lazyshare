# -*- coding: utf-8 -*-
"""图表数据模型：管理 sub_win / 序列 / 时间窗口 / 滚动偏移。"""
from __future__ import annotations

from typing import Callable

# 不同列项使用的颜色（同个 sub_win 内不同序列颜色不同）
PALETTE = [
    "#f1144b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff",
]

DEFAULT_SHOW_DAYS = 21


class Series:
    """单个列项数据序列（属于某个 sub_win）。"""

    __slots__ = ("column", "side", "color", "values")

    def __init__(self, column: str, side: str, color: str, values) -> None:
        self.column = column        # 列项名
        self.side = side            # 'left' 左纵列 / 'right' 右纵列
        self.color = color          # 显示颜色
        self.values = values        # 数值数组（含 NaN）

    def __repr__(self) -> str:  # pragma: no cover
        return f"Series({self.column!r}, side={self.side!r})"


class SubWin:
    """单个纵向排列的小窗口，含若干序列。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.series: list[Series] = []

    def add_series(self, series: Series) -> None:
        # 同一 sub_win 内不重复加载相同列项
        for s in self.series:
            if s.column == series.column and s.side == series.side:
                return
        self.series.append(series)
        self._update_name()

    def _update_name(self) -> None:
        # sub_win 用列项名命名，多个列项用“-”拼接
        self.name = "-".join(s.column for s in self.series)

    def series_of_side(self, side: str) -> list[Series]:
        return [s for s in self.series if s.side == side]

    def __repr__(self) -> str:  # pragma: no cover
        return f"SubWin({self.name!r})"


class ChartModel:
    """图表整体模型：sub_win 列表 + 时间窗口（show_days/offset）。

    时间轴由全体 sub_win 共享：show_days 表示可见天数，
    offset 表示可见窗口在完整时间轴上的起始下标（0 为最早）。
    """

    def __init__(self, data_store) -> None:
        self.data_store = data_store
        self.sub_wins: list[SubWin] = []
        self.show_days: int = DEFAULT_SHOW_DAYS
        self._offset: int = 0
        self._color_index: int = 0
        self._listeners: list[Callable[[], None]] = []
        # 默认窗口显示最新数据（offset 在最右侧）
        self._offset = self.max_offset
        # 数据变化（如新打开文件）后同样右对齐
        self.data_store.add_listener(self._on_data_changed)

    def _on_data_changed(self) -> None:
        self.show_days = DEFAULT_SHOW_DAYS
        self._offset = self.max_offset
        self._notify()

    # ------------------------------------------------------------------
    # 观察者
    # ------------------------------------------------------------------
    def add_listener(self, fn: Callable[[], None]) -> None:
        self._listeners.append(fn)

    def _notify(self) -> None:
        for fn in list(self._listeners):
            fn()

    # ------------------------------------------------------------------
    # 基础状态
    # ------------------------------------------------------------------
    @property
    def total_days(self) -> int:
        return self.data_store.row_count

    @property
    def max_offset(self) -> int:
        """offset 最大可取的值（保证窗口不超过数据范围）。"""
        return max(0, self.total_days - self.show_days)

    @property
    def offset(self) -> int:
        return min(self._offset, self.max_offset)

    def set_offset(self, value: int) -> None:
        self._offset = int(value)
        self._offset = max(0, min(self._offset, self.max_offset))
        self._notify()

    def structure_key(self) -> tuple:
        """sub_win 结构签名，用于 UI 判断是否需要重建窗口。"""
        return tuple(
            (sw.name, tuple((s.column, s.side) for s in sw.series))
            for sw in self.sub_wins
        )

    # ------------------------------------------------------------------
    # 时间窗口
    # ------------------------------------------------------------------
    def zoom_in(self) -> None:
        """“+”：show_days 翻倍（保持右边缘数据点不动）。"""
        self._set_show_days(self.show_days * 2)

    def zoom_out(self) -> None:
        """“-”：show_days 缩小一半（保持右边缘数据点不动）。"""
        self._set_show_days(max(1, self.show_days // 2))

    def _set_show_days(self, new_show_days: int) -> None:
        if new_show_days == self.show_days:
            return
        right_edge = min(self.total_days, self.offset + self.show_days)
        new_show_days = max(1, new_show_days)
        if self.total_days > 0:
            new_show_days = min(new_show_days, self.total_days)
        self.show_days = new_show_days
        self._offset = max(0, right_edge - self.show_days)
        self._notify()

    # ------------------------------------------------------------------
    # 序列管理
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """清理缓存的旧数据和显示内容。"""
        self.sub_wins.clear()
        self._color_index = 0
        self.show_days = DEFAULT_SHOW_DAYS
        # 默认窗口显示最新数据（offset 在最右侧）
        self._offset = self.max_offset
        self._notify()

    def add_series(self, columns: list[str], side: str = "left",
                   target_name: str | None = None) -> SubWin:
        """添加一个或多个列项到指定 sub_win（target_name 为 None 时新建窗口）。

        columns 可包含重复，内部会去重；同 sub_win 内相同列项不重复添加。
        """
        side = "right" if side == "right" else "left"
        if target_name is None:
            sw = SubWin(name="")
        else:
            sw = self.get_subwin(target_name)
            if sw is None:
                sw = SubWin(name="")

        added = False
        for col in columns:
            if not col:
                continue
            # 去重：同 sub_win 内已存在相同列项则跳过
            if any(s.column == col and s.side == side for s in sw.series):
                continue
            color = self._next_color()
            values = self.data_store.get_values(col)
            sw.add_series(Series(col, side, color, values))
            added = True
        if added and sw not in self.sub_wins:
            self.sub_wins.append(sw)
        self._notify()
        return sw

    def get_subwin(self, name: str) -> SubWin | None:
        for sw in self.sub_wins:
            if sw.name == name:
                return sw
        return None

    def remove_series(self, subwin, column: str, side: str | None = None) -> bool:
        """从指定 sub_win 删除某列项数据源。

        subwin 可为 SubWin 对象或窗口名；side 指定时仅删除该纵列上的数据源。
        删除后 sub_win 名称按剩余列项重算，若没有剩余列项则整个 sub_win 被移除。
        """
        sw = subwin if isinstance(subwin, SubWin) else self.get_subwin(subwin)
        if sw is None or sw not in self.sub_wins:
            return False
        before = len(sw.series)
        sw.series = [
            s for s in sw.series
            if not (s.column == column and (side is None or s.side == side))
        ]
        if len(sw.series) == before:
            return False
        sw._update_name()
        if not sw.series:
            self.sub_wins.remove(sw)
        self._notify()
        return True

    def restore_state(self, state: dict) -> None:
        """按显示配置恢复（不含数据数值），与 extract_display_config 对应。"""
        self.clear()
        self.show_days = max(1, int(state.get("show_days", DEFAULT_SHOW_DAYS)))
        for sw_data in state.get("sub_wins", []):
            series = sw_data.get("series", [])
            if not series:
                continue
            sw = SubWin(name="")
            for s in series:
                col = str(s.get("column", ""))
                side = "right" if s.get("side") == "right" else "left"
                if not col:
                    continue
                # 跳过当前数据中不存在的列项
                if self.data_store.get_values(col).size == 0:
                    continue
                if any(x.column == col and x.side == side for x in sw.series):
                    continue
                sw.add_series(
                    Series(col, side, self._next_color(),
                           self.data_store.get_values(col)))
            if sw.series:
                self.sub_wins.append(sw)
        if self.total_days > 0:
            self.show_days = min(self.show_days, self.total_days)
        offset = int(state.get("offset", self.max_offset))
        self._offset = max(0, min(offset, self.max_offset))
        self._notify()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _next_color(self) -> str:
        color = PALETTE[self._color_index % len(PALETTE)]
        self._color_index += 1
        return color
