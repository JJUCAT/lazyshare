#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把训练 / 验证 / 测试报告文件合并后作为邮件正文发送。

与 scripts/mail.py（发送预测报告 prediction.log）互补：
    - scripts/mail.py   ：面向每日预测流程，正文固定取 <pred-date>/prediction.log
    - scripts/mail_report.py：面向训练流水线，可把任意多个报告文件拼成一封邮件

典型用法（由 scripts/tsai/train_pipeline.sh 调用）：
    python3 scripts/mail_report.py \\
        --files models/<模型名>.train.log models/<模型名>.valid.log models/<模型名>.test.log \\
        --title "【lazyshare】<模型名> 训练/验证/测试报告"

单独调试：
    python3 scripts/mail_report.py --files a.log b.log --dry-run

复用 src/mail/ 的既有能力（配置加载 + SMTP 发送 + 纯文本/HTML 双格式正文），
不新增邮件协议实现。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mail.config import DEFAULT_CONFIG, load_config  # noqa: E402
from src.mail.sender import build_body, send_mail  # noqa: E402

# 每个报告文件在正文中的分隔线，便于在邮件里快速定位章节
SECTION_SEP = "=" * 72


def build_report(paths: list[Path]) -> str:
    """把多个报告文件按给定顺序拼成一份正文。

    每个文件前加分隔线与文件名；不存在或内容为空的文件跳过（并记 warning），
    以支持"训练成功但验证失败"时仍能发出已有部分的报告。
    """
    chunks: list[str] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            logging.warning("报告文件不存在，跳过: %s", p)
            continue
        text = p.read_text(encoding="utf-8", errors="replace").strip("\n")
        if not text.strip():
            logging.warning("报告文件为空，跳过: %s", p)
            continue
        chunks.append(f"{SECTION_SEP}\n# {p.name}\n{SECTION_SEP}\n{text}")
    if not chunks:
        return ""
    return "\n\n".join(chunks) + "\n"


def default_title(files: list[Path]) -> str:
    """由第一个报告文件名推导标题：<模型名>.train.log -> 【lazyshare】<模型名> 训练报告。"""
    if not files:
        return "【lazyshare】模型报告"
    name = Path(files[0]).name
    model = name.rsplit(".", 2)[0] if name.count(".") >= 2 else Path(name).stem
    return f"【lazyshare】{model} 模型报告"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把训练/验证/测试报告文件合并后发送邮件")
    parser.add_argument("--files", nargs="+", type=Path, required=True,
                        help="要合并进正文的报告文件（按给定顺序拼接）")
    parser.add_argument("--title", default=None,
                        help="邮件标题（默认由第一个报告文件名推导）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="邮件配置文件（默认: config/mail.json）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印标题与正文，不实际发送")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    report = build_report(args.files)
    if not report.strip():
        logging.warning("没有可发送的报告内容，本次不发送邮件")
        return 0

    title = args.title or default_title(args.files)
    text, html = build_body(report)

    if args.dry_run:
        print(f"Subject: {title}")
        sys.stdout.write(text)
        logging.info("dry-run：不实际发送（正文 %d 字符）", len(text))
        return 0

    mail_cfg = load_config(args.config)
    send_mail(mail_cfg["smtp"], title, text, html, mail_cfg["mails"])
    logging.info("已发送邮件到 %d 个邮箱: %s", len(mail_cfg["mails"]), mail_cfg["mails"])
    logging.info("主题: %s（正文 %d 字符）", title, len(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
