# -*- coding: utf-8 -*-
"""邮件配置加载：读取 config/mail.json。

config/mail.json 结构:
    {
      "smtp": { "server": "smtp.163.com", "user": "可选发件人", "auth_code": "..." },
      "mails": [ "目标邮箱1", ... ],
      "threshold": 0.9   # 可选，高置信度阈值（默认 0.9）
    }

说明:
    - 163 邮箱用"授权码"登录，需 smtp.user（发件人邮箱）+ smtp.auth_code
    - smtp.user 未配置时，默认取第一个目标邮箱作为发件人
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "mail.json"


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    """返回归一化配置:
    {
        "smtp": {"server", "user", "auth_code"},
        "mails": [str, ...],
        "threshold": float,   # 默认 0.9
    }
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    smtp = cfg.get("smtp") or {}
    mails = [m for m in cfg.get("mails", []) if isinstance(m, str) and m.strip()]
    if not mails:
        raise ValueError("config/mail.json 中 mails 为空，请至少配置一个目标邮箱")

    server = (smtp.get("server") or "").strip()
    auth_code = (smtp.get("auth_code") or "").strip()
    user = (smtp.get("user") or mails[0]).strip()
    if not server or not auth_code or not user:
        raise ValueError("smtp.server / smtp.auth_code / smtp.user 不能为空")

    try:
        threshold = float(cfg.get("threshold", 0.9))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"threshold 必须是数字: {cfg.get('threshold')}") from exc

    return {
        "smtp": {"server": server, "user": user, "auth_code": auth_code},
        "mails": mails,
        "threshold": threshold,
    }
