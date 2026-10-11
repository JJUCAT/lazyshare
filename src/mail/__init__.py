# -*- coding: utf-8 -*-
"""邮件功能包：读取 config/mail.json，把预测报告发到目标邮箱。

- config      : 邮件配置加载（smtp / mails）
- predictions : 定位最新预测目录并读取 prediction.log（最终报告）原文
- sender      : 以报告原文组装邮件正文并 smtplib 发送
"""
from src.mail.config import load_config
from src.mail.predictions import (
    find_latest_pred_dir,
    parse_trade_date,
    read_report,
)
from src.mail.sender import send_mail

__all__ = [
    "load_config",
    "find_latest_pred_dir",
    "read_report",
    "parse_trade_date",
    "send_mail",
]
