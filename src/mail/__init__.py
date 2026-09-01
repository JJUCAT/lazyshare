# -*- coding: utf-8 -*-
"""邮件功能包：读取 config/mail.json，把高置信度预测结果发到目标邮箱。

- config      : 邮件配置加载（smtp / mails / threshold）
- predictions : 定位最新预测目录并解析 prediction.log（按置信度过滤）
- sender      : 组装邮件正文并 smtplib 发送
"""
from src.mail.config import load_config
from src.mail.predictions import (
    filter_high_conf,
    find_latest_pred_dir,
    parse_prediction_log,
)
from src.mail.sender import send_mail

__all__ = [
    "load_config",
    "find_latest_pred_dir",
    "parse_prediction_log",
    "filter_high_conf",
    "send_mail",
]
