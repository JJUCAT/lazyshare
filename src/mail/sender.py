# -*- coding: utf-8 -*-
"""邮件发送：把预测报告（prediction.log）作为邮件正文发到目标邮箱。

流程（见 run_plan.md）：
    1. 读取 config/mail.json
    2. 定位最新 pred-<date>/prediction.log（可用 --date 指定）
    3. 邮件正文 = 报告原文，与 weekday.sh 流程的最终报告（预测报告）完全一致
    4. 报告不存在或为空时跳过发送

用法：
    python3 scripts/mail.py [--config config/mail.json] [--date YYYYMMDD] [--dry-run] [-v]
"""
from __future__ import annotations

import argparse
import html as html_lib
import logging
import smtplib
import sys
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

from src.mail.config import DEFAULT_CONFIG, load_config
from src.mail.predictions import (
    PROJECT_ROOT,
    find_latest_pred_dir,
    parse_trade_date,
    read_report,
)


def build_body(report: str) -> tuple[str, str]:
    """生成 (纯文本正文, HTML 正文)：正文即预测报告原文，不做任何增删。"""
    text = report if report.endswith("\n") else report + "\n"
    html = (
        "<html><body>"
        "<pre style='font-family:Menlo,Consolas,monospace;font-size:13px;"
        "white-space:pre-wrap;margin:0'>"
        f"{html_lib.escape(text)}"
        "</pre></body></html>"
    )
    return text, html


def send_mail(smtp_cfg: dict, subject: str, text: str, html: str,
              mails: list[str]) -> None:
    """通过 SMTP 发送邮件到 mails 列表（163 邮箱用 SSL 465）。"""
    msg = MIMEMultipart("alternative")
    msg["From"] = formataddr(("lazyshare", smtp_cfg["user"]))
    msg["To"] = ", ".join(mails)
    msg["Subject"] = Header(subject, "utf-8")
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP_SSL(smtp_cfg["server"], 465, timeout=30) as server:
        server.login(smtp_cfg["user"], smtp_cfg["auth_code"])
        server.sendmail(smtp_cfg["user"], mails, msg.as_string())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把预测报告（prediction.log）发送到目标邮箱")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="邮件配置文件（默认: config/mail.json）")
    parser.add_argument("--date", type=str, default=None,
                        help="预测目录日期 YYYYMMDD（默认取最新 pred-* 目录）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印将发送的内容，不实际发送")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)

    # 定位预测目录 / 报告文件
    if args.date:
        pred_dir = PROJECT_ROOT / "test_output" / f"pred-{args.date}"
        if not pred_dir.is_dir():
            logging.error("预测目录不存在: %s", pred_dir)
            return 1
    else:
        pred_dir = find_latest_pred_dir()
        if pred_dir is None:
            logging.error("test_output 下未找到 pred-* 预测目录")
            return 1
    log_path = pred_dir / "prediction.log"
    logging.info("预测报告: %s", log_path)

    report = read_report(log_path)
    if not report.strip():
        logging.info("预测报告为空或不存在，本次不发送邮件")
        return 0

    # 交易日取报告中的"预测交易日"，取不到则用预测目录名的日期
    trade_date = parse_trade_date(report, fallback=pred_dir.name.removeprefix("pred-"))
    subject = f"【lazyshare】{trade_date} 预测报告"
    text, html = build_body(report)

    if args.dry_run:
        print(subject)
        # 直接写出报告原文（text 已带结尾换行），避免 print 额外插入空行
        sys.stdout.write(text)
        logging.info("dry-run：不实际发送")
        return 0

    send_mail(cfg["smtp"], subject, text, html, cfg["mails"])
    logging.info("已发送邮件到 %d 个邮箱: %s", len(cfg["mails"]), cfg["mails"])
    logging.info("主题: %s", subject)
    return 0


if __name__ == "__main__":
    sys.exit(main())
