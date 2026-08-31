# -*- coding: utf-8 -*-
"""配置加载：读取 config/pull.json。

字段说明（见 pull_plan.md）：
    source.dyy            ：金玥数据账号（url / account / password）
    spider.share_individual：爬虫参数（port: 日常更新 / type: 复权方式 / frequency: 接口访问频率秒）
    share                 ：数据库目录（用于检查个股数据时间）
    download              ：拉取数据保存路径
"""
from __future__ import annotations

import json
from pathlib import Path

# src/pull/config.py -> pull -> src -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pull.json"

# 复权方式映射：配置 type -> 日常更新下载接口的子标签参数（qqr=前复权）
ADJUST_MAP = {
    "不复权": "fqr",
    "前复权": "qqr",
    "后复权": "hqr",
}


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    """读取 config/pull.json，返回规范化配置 dict。"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    source = cfg.get("source") or {}
    dyy = source.get("dyy") or {}
    url = str(dyy.get("url", "")).strip()
    account = str(dyy.get("account", "")).strip()
    password = str(dyy.get("password", "")).strip()
    if not url or not account or not password:
        raise ValueError("source.dyy 未完整配置（url / account / password）")

    spider = cfg.get("spider") or {}
    share_individual = spider.get("share_individual") or {}
    port = str(share_individual.get("port", "日常更新"))
    adj_type = str(share_individual.get("type", "前复权"))
    frequency = float(share_individual.get("frequency", 1))
    if adj_type not in ADJUST_MAP:
        raise ValueError(f"不支持的复权方式: {adj_type}（可用: {list(ADJUST_MAP)}）")

    share_dir = Path(cfg.get("share", "")).expanduser()
    download_dir = Path(cfg.get("download", "")).expanduser()
    if not share_dir or not share_dir.exists():
        raise FileNotFoundError(f"share 目录不存在: {share_dir}")
    if not download_dir:
        raise ValueError("download 未配置")

    return {
        "base_url": f"https://{url}" if not url.startswith("http") else url,
        "account": account,
        "password": password,
        "port": port,
        "adjust_type": adj_type,
        "adjust": ADJUST_MAP[adj_type],
        "frequency": max(0.0, frequency),
        "share_dir": share_dir,
        "download_dir": download_dir,
    }
