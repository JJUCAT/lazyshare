# -*- coding: utf-8 -*-
"""预测结果解析：定位最新 pred-* 目录、解析 prediction.log、按置信度过滤。

prediction.log 由 src/prediction/predict.py 生成，T/B 结果行格式:
    排名    股票代码    股票名称    预测    置信度    交易日
    1     603221    爱丽家居     T     0.9432    2026-09-01
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_OUTPUT = PROJECT_ROOT / "test_output"

# 排名 / 6 位代码 / 名称(无空格) / T|B / 置信度 / 日期
_LINE_RE = re.compile(
    r"^\s*(\d+)\s+(\d{6})\s+(\S+)\s+([TB])\s+(\d+(?:\.\d+)?)\s+(\S+)\s*$"
)


def find_latest_pred_dir(test_output: Path = TEST_OUTPUT) -> Path | None:
    """返回 test_output 下最新（pred-YYYYMMDD，目录名最大）的预测目录；无则返回 None。"""
    dirs = [p for p in test_output.glob("pred-*") if p.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name, reverse=True)[0]


def parse_prediction_log(log_path: Path) -> list[dict]:
    """解析 prediction.log 的 T/B 结果行，返回:
    [{rank, code, name, pred, conf, date}, ...]（保持文件顺序，即置信度降序）
    """
    rows: list[dict] = []
    if not log_path.exists():
        return rows
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        rows.append({
            "rank": int(m.group(1)),
            "code": m.group(2),
            "name": m.group(3),
            "pred": m.group(4),
            "conf": float(m.group(5)),
            "date": m.group(6),
        })
    return rows


def filter_high_conf(rows: list[dict], threshold: float) -> list[dict]:
    """仅保留置信度大于 threshold 的结果（保持降序）。"""
    return [r for r in rows if r["conf"] > threshold]
