#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动 LazyShare GUI，并在出错或崩溃时将错误写入 test_output/gui_error.log。

覆盖两类问题：
    1. Python 异常（含 Qt 事件循环中槽函数抛出的未捕获异常）——通过 sys.excepthook 记录；
    2. C 层崩溃（段错误等致命信号）——通过 faulthandler 记录。
"""
from __future__ import annotations

import argparse
import faulthandler
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = PROJECT_ROOT / "test_output" / "gui_error.log"

_log_handle = None  # faulthandler 使用的文件句柄（保持打开）


def _ensure_log_dir() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _write(text: str) -> None:
    _ensure_log_dir()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text)


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    """未捕获异常（含 Qt 槽内异常）记录到 gui_error.log。"""
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    _write(f"\n===== {_timestamp()} GUI 异常 =====\n{msg}\n")
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def setup_logging() -> None:
    """开启异常与崩溃日志记录。"""
    _ensure_log_dir()
    # C 层致命错误（段错误等）写入同一日志文件
    global _log_handle
    _log_handle = open(LOG_FILE, "a", encoding="utf-8")
    faulthandler.enable(file=_log_handle)
    # Python 未捕获异常
    sys.excepthook = _excepthook
    _write(f"\n===== {_timestamp()} GUI 启动 =====\n")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 LazyShare GUI")
    parser.add_argument("--file", type=str, default=None,
                        help="启动时自动加载的 CSV 文件路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()

    try:
        from src.gui.main import main as gui_main

        gui_args = [f"--file={args.file}"] if args.file else None
        code = gui_main(gui_args)
    except SystemExit as exc:  # app.exec() 正常退出
        code = int(exc.code or 0)
    except Exception:  # noqa: BLE001
        msg = traceback.format_exc()
        _write(f"\n===== {_timestamp()} GUI 启动失败 =====\n{msg}\n")
        print(f"GUI 启动失败，详见 {LOG_FILE}", file=sys.stderr)
        code = 1
    finally:
        if _log_handle is not None:
            try:
                _log_handle.close()
            except Exception:  # noqa: BLE001
                pass
    return code


if __name__ == "__main__":
    sys.exit(main())
