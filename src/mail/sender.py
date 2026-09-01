# -*- coding: utf-8 -*-
"""邮件发送：读取预测结果（置信度 > threshold），组装邮件并发到目标邮箱。

流程（见 run_plan.md）：
    1. 读取 config/mail.json
    2. 定位最新 pred-<date>/prediction.log（可用 --date 指定）
    3. 解析 T/B 结果，过滤置信度 > threshold 的股票
    4. 有高置信度信号时发邮件；无信号则跳过并提示

用法：
    python3 scripts/mail.py [--config config/mail.json] [--date YYYYMMDD] [--dry-run] [-v]
"""
from __future__ import annotations

import argparse
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
    filter_high_conf,
    find_latest_pred_dir,
    parse_prediction_log,
)

# 标签中文释义
_PRED_LABEL = {"T": "高峰(T)", "B": "低谷(B)"}


def build_body(rows: list[dict], latest_date: str, threshold: float) -> tuple[str, str]:
    """生成 (纯文本正文, HTML 正文)。rows 为置信度降序的高置信度结果。"""
    lines = [
        f"交易日: {latest_date}",
        f"高置信度信号（置信度 > {threshold:.2f}）: {len(rows)} 只",
        "",
        f"{'排名':<4}{'代码':<8}{'名称':<12}{'标签':<8}{'置信度':<8}{'交易日'}",
        "-" * 52,
    ]
    for r in rows:
        label = _PRED_LABEL.get(r["pred"], r["pred"])
        lines.append(
            f"{r['rank']:<4}{r['code']:<8}{r['name']:<12}{label:<10}"
            f"{r['conf']:.4f}    {r['date']}"
        )
    text = "\n".join(lines)

    trs = "\n".join(
        "<tr>"
        f"<td>{r['rank']}</td><td>{r['code']}</td><td>{r['name']}</td>"
        f"<td>{_PRED_LABEL.get(r['pred'], r['pred'])}</td>"
        f"<td>{r['conf']:.4f}</td><td>{r['date']}</td>"
        "</tr>"
        for r in rows
    )
    html = (
        "<html><body style='font-family:Menlo,monospace;font-size:13px'>"
        f"<h3>高置信度预测信号（置信度 > {threshold:.2f}）</h3>"
        f"<p>交易日: <b>{latest_date}</b>，共 <b>{len(rows)}</b> 只</p>"
        "<table border='1' cellspacing='0' cellpadding='6' "
        "style='border-collapse:collapse'>"
        "<tr style='background:#f0f0f0'><th>排名</th><th>代码</th><th>名称</th>"
        "<th>标签</th><th>置信度</th><th>交易日</th></tr>"
        f"{trs}"
        "</table></body></html>"
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
        description="把预测结果中置信度 > 阈值的股票标签发送邮件")
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
    smtp_cfg = cfg["smtp"]
    threshold = cfg["threshold"]

    # 定位预测目录 / prediction.log
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
    logging.info("预测结果: %s", log_path)

    rows = parse_prediction_log(log_path)
    high = filter_high_conf(rows, threshold)
    logging.info("T/B 共 %d 条，置信度 > %.2f 的有 %d 条",
                 len(rows), threshold, len(high))

    if not high:
        logging.info("无置信度 > %.2f 的信号，本次不发送邮件", threshold)
        return 0

    # 交易日取最高置信度那行的日期
    latest_date = high[0]["date"]
    subject = f"【lazyshare】{latest_date} 高置信度预测信号（{len(high)} 只）"
    text, html = build_body(high, latest_date, threshold)

    if args.dry_run:
        print(subject)
        print(text)
        logging.info("dry-run：不实际发送")
        return 0

    send_mail(smtp_cfg, subject, text, html, cfg["mails"])
    logging.info("已发送邮件到 %d 个邮箱: %s", len(cfg["mails"]), cfg["mails"])
    logging.info("主题: %s", subject)
    return 0


if __name__ == "__main__":
    sys.exit(main())
