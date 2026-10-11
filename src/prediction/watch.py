# -*- coding: utf-8 -*-
"""关注名单预测：读取 config/watch.json，记录名单内股票的标签与置信度。

见 prediction_plan.md 第 7 条：读取 config/watch.json，推理分类后把 watch 名单的
股票标签、置信度记录到 pred-<date>/prediction.log 文件中；邮件正文即该报告原文，
故该内容会随邮件一并发出（无需改动邮件模块）。

config/watch.json 结构:
    { "watch": ["600158", "002891"] }

输出为文本行列表，供 predict.py 追加写入 pred-<date>/prediction.log。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 默认关注名单配置路径
DEFAULT_WATCH = PROJECT_ROOT / "config" / "watch.json"

# 名单内股票未出现在预测结果中时的占位（如数据不足 seq_len 天）
_MISSING = "（无预测结果）"


def load_watch_codes(config_path: str | Path = DEFAULT_WATCH) -> list[str]:
    """读取 config/watch.json 的关注股票代码列表。

    文件不存在 / 解析失败 / 未配置 watch 时返回空列表（不阻断预测流程，
    因为关注名单属于附加信息）。代码统一去空白，保持原顺序并去重。
    """
    path = Path(config_path)
    if not path.exists():
        logging.warning("关注名单配置不存在，跳过关注名单: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as exc:
        logging.warning("读取关注名单配置失败，跳过关注名单: %s (%s)", path, exc)
        return []
    raw = cfg.get("watch") if isinstance(cfg, dict) else None
    if not raw:
        logging.info("config/watch.json 未配置 watch，跳过关注名单")
        return []
    codes: list[str] = []
    for item in raw:
        code = str(item).strip()
        if code and code not in codes:
            codes.append(code)
    return codes


def build_watch_lines(rows: list[dict],
                      config_path: str | Path = DEFAULT_WATCH) -> list[str]:
    """生成关注名单股票的标签 / 置信度文本行。

    rows: predict.py 的预测结果，每项含 code / name / pred / conf / date
          （与 top-10 榜单同源，含 T / B / N 全部预测）。
    名单内未出现在预测结果中的股票（如数据不足 / 已退出 data_source）以占位符列出，
    保证任何被关注股票都不会被静默遗漏。
    """
    codes = load_watch_codes(config_path)
    if not codes:
        return []

    # 代码 -> 预测结果（同一代码重复时取第一条）
    by_code: dict[str, dict] = {}
    for r in rows:
        by_code.setdefault(str(r.get("code", "")).strip(), r)

    header = (f"{'排名':<6}{'股票代码':<10}{'股票名称':<16}{'预测':<6}"
              f"{'置信度':<10}{'交易日'}")
    lines = [f"关注名单（{Path(config_path).name}）:", header]
    for i, code in enumerate(codes, 1):
        r = by_code.get(code)
        if r is None:
            lines.append(f"{i:<6}{code:<10}{_MISSING:<16}{'-':<6}{'-':<10}{'-'}")
            continue
        lines.append(f"{i:<6}{r['code']:<10}{r['name']:<16}{r['pred']:<6}"
                     f"{r['conf']:.4f}    {r['date']}")
    return lines
