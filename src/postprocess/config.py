# -*- coding: utf-8 -*-
"""后处理配置：读取 config/postprocess.json（独立于预处理 / 训练配置）。

配置结构：
    {
      "enabled": true,                     # 后处理开关（每次运行实时读取）
      "data_source": "/path/preprocessed", # 预处理目录（读 收盘价 / M5C）
      "validation":  "/path/validation",   # 推理结果目录（results.json 所在）
      "models":      "/path/models",       # 模型目录（默认定位于最新 .pkl）
      "price_col": "收盘价",                # 当日股价列
      "ma_col":    "M5C",                  # 比对的均线列（M5C / M60C）
      "labels": {"buy": "B", "sell": "T"}, # 买入 / 卖出标签
      "min_conf": null                     # 可选：只保留置信度 >= min_conf 的信号
    }

开关："enabled" 只在 config/postprocess.json 维护。训练流水线与
scripts/postprocess.py 每次运行都会重新读取该文件，不使用缓存；
enabled=false 时跳过过滤，不写任何后处理报告，推理结果 results.json
与验证/测试分类指标不受影响。

价格与均线取自预处理目录（data_source）中的源股票 CSV；切片后的窗口 CSV
只保留特征列，不含收盘价 / M5C / M60C，因此需要按（源文件名, 日期）回溯。

用法：
    python3 scripts/postprocess.py [--config config/postprocess.json] ...
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "postprocess.json"

# 以下为默认列名 / 取值，可被 config 覆盖
PRICE_COL = "收盘价"
MA_COL = "M5C"
BUY_LABEL = "B"
SELL_LABEL = "T"


def _parse_enabled(cfg: dict) -> bool:
    """解析 enabled 开关，兼容 bool / {"enabled": bool} / 字符串。"""
    flag = cfg.get("enabled", True)
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, dict):
        return bool(flag.get("enabled", True))
    if isinstance(flag, str):
        return flag.strip().lower() not in ("0", "false", "no", "off")
    return bool(flag)


def postprocess_enabled(config_path: Path = DEFAULT_CONFIG) -> bool:
    """后处理过滤是否启用（每次调用都实时读取 config/postprocess.json）。

    - 配置文件不存在：默认启用（True）；
    - enabled 缺省：默认启用（True）；
    - 不做任何缓存，配置修改后下一次调用立即生效。
    """
    path = Path(config_path)
    if not path.exists():
        return True
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return _parse_enabled(cfg)


def load_postprocess_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    """读取后处理配置，返回后处理所需参数。

    返回 dict：
        enabled                          : 后处理开关（实时读取配置所得）
        data_source / validation / models : 路径
        price_col / ma_col               : 价格与均线列名
        buy / sell                       : 买入 / 卖出标签
        min_conf                         : 置信度下限（None 表示不筛选）
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"后处理配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_source = Path(cfg.get("data_source", "")).expanduser()
    validation = Path(cfg.get("validation", "")).expanduser()
    models = Path(cfg.get("models", "")).expanduser()
    if not data_source:
        raise ValueError("postprocess.data_source 未配置")
    if not validation:
        raise ValueError("postprocess.validation 未配置")
    if not models:
        raise ValueError("postprocess.models 未配置")

    labels = cfg.get("labels") or {}
    min_conf = cfg.get("min_conf")
    if min_conf is not None:
        min_conf = float(min_conf)
        if not 0.0 <= min_conf <= 1.0:
            raise ValueError(f"postprocess.min_conf 必须在 [0, 1] 之间，当前 {min_conf}")

    return {
        "enabled": _parse_enabled(cfg),
        "data_source": data_source,
        "validation": validation,
        "models": models,
        "price_col": str(cfg.get("price_col") or PRICE_COL),
        "ma_col": str(cfg.get("ma_col") or MA_COL),
        "buy": str(labels.get("buy") or BUY_LABEL),
        "sell": str(labels.get("sell") or SELL_LABEL),
        "min_conf": min_conf,
    }
