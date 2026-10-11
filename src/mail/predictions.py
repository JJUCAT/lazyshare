# -*- coding: utf-8 -*-
"""预测报告读取：定位最新 pred-* 目录，读取 prediction.log（最终报告）原文。

prediction.log 由 src/prediction/predict.py 生成，是 weekday 流程的最后一份报告，
内容依次为：
    预测日期 / 模型 / 数据源 / 窗口长度(seq_len) / 预测交易日 / 股票总数统计
    T 标签置信度前 10 只股票（按置信度从高到低）
    B 标签置信度前 10 只股票（按置信度从高到低）
    大盘交易行情概况 / 热门行业 / 热门行业热门票
    关注名单（config/watch.json）内每只股票的标签 / 置信度

邮件正文与该报告保持一致（直接发送报告原文），故此处只做"定位 + 读取 + 取交易日"。
关注名单由 predict.py 写入报告，随报告原文一并进入邮件正文。
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_OUTPUT = PROJECT_ROOT / "test_output"

# 预测报告文件名
REPORT_NAME = "prediction.log"

# 报告头部行形如 "预测交易日: 2026-09-11"
_TRADE_DATE_RE = re.compile(r"^\s*预测交易日\s*[:：]\s*(\S+)\s*$", re.MULTILINE)


def find_latest_pred_dir(test_output: Path = TEST_OUTPUT) -> Path | None:
    """返回 test_output 下最新（pred-YYYYMMDD，目录名最大）的预测目录；无则返回 None。"""
    dirs = [p for p in test_output.glob("pred-*") if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name, reverse=True)[0]


def read_report(log_path: Path) -> str:
    """读取预测报告（prediction.log）原文；文件不存在时返回空串。"""
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8")


def parse_trade_date(report: str, fallback: str = "") -> str:
    """从报告原文提取"预测交易日"（用于邮件主题）；取不到时返回 fallback。

    报告中该行由 predict.py 写入，无 T/B 命中时值为 "-"，此时同样回退到 fallback。
    """
    m = _TRADE_DATE_RE.search(report or "")
    if not m:
        return fallback
    value = m.group(1).strip()
    return value if value and value != "-" else fallback
