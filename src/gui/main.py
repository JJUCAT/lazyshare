#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LazyShare GUI 入口。

用法：
    python3 src/gui/main.py                # 正常启动
    python3 src/gui/main.py --file PATH    # 启动后自动加载指定 csv
    python3 -m src.gui.main                # 从项目根目录运行
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from src.gui.business.chart_model import ChartModel
from src.gui.business.data_store import DataStore
from src.gui.ui.main_window import MainWindow


def apply_theme(app: QApplication) -> None:
    """应用暗色主题。"""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(37, 37, 38))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(221, 221, 221))
    palette.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(45, 45, 48))
    palette.setColor(QPalette.ColorRole.Text, QColor(221, 221, 221))
    palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 48))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(221, 221, 221))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(67, 99, 216))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(43, 43, 51))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(140, 140, 140))
    app.setPalette(palette)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LazyShare GUI")
    parser.add_argument("--file", type=str, default=None,
                        help="启动时自动加载的 CSV 文件路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication(sys.argv)
    apply_theme(app)

    store = DataStore()
    model = ChartModel(store)
    window = MainWindow(store, model)
    window.show()

    if args.file:
        window.load_file(args.file)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
