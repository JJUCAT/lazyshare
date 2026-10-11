#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键运行：拉取 → 更新 → 预处理 → 预测 → 邮件（尽量调用现有功能接口）。

流程（见 run_plan.md）：
    1. pull      : 使用 pull.json 拉取最新数据
    2. update    : 使用 update.json 更新个股历史数据
    3. preprocess: 使用 preprocess.json 更新预处理数据
    4. retrain   : 仅 3/6/9/12 月的首个工作日：更新 dataset 并训练新模型
    5. prediction: 使用 prediction.sh 估计最新股票标签
    6. mail      : 把置信度 > 阈值的股票及标签发邮件到目标邮箱（config/mail.json）
    7. cleanup   : 清理 test_output 旧预测目录（仅保留最近 KEEP_TEST_OUTPUT_DAYS 天）

本脚本只做编排：通过 subprocess 调用现有入口，不重复实现内部逻辑。
    - pull / update / preprocess / mail 使用系统 python3（可用 --python 指定）
    - prediction 使用 scripts/prediction.sh，自动激活 tsai conda 环境

用法：
    python3 scripts/run.py                              # 全流程
    python3 scripts/run.py --steps pull,update          # 仅拉取 + 更新
    python3 scripts/run.py --dry-run                    # 预览将执行的命令，不执行
    python3 scripts/run.py -v                           # 向子命令透传 -v 调试日志
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# test_output 下保留的最近预测目录数量（按 pred-YYYYMMDD 目录名日期计算）
TEST_OUTPUT_DIR = PROJECT_ROOT / "test_output"
KEEP_TEST_OUTPUT_DAYS = 10

# 季度重训月份：这些月份的"首个工作日"更新 dataset 并训练新模型
QUARTER_MONTHS = (3, 6, 9, 12)

DEFAULT_PULL_CONFIG = PROJECT_ROOT / "config" / "pull.json"
DEFAULT_UPDATE_CONFIG = PROJECT_ROOT / "config" / "update.json"
DEFAULT_PREPROCESS_CONFIG = PROJECT_ROOT / "config" / "preprocess.json"
DEFAULT_PREDICT_CONFIG = PROJECT_ROOT / "config" / "classify_train.json"
DEFAULT_MAIL_CONFIG = PROJECT_ROOT / "config" / "mail.json"

# 固定执行顺序（retrain 为条件步骤，仅季度首月首个工作日实际执行）
STEPS = ("pull", "update", "preprocess", "retrain", "predict", "mail", "cleanup")

STEP_DESCRIPTION = {
    "pull": "拉取最新数据（src.pull.pull）",
    "update": "更新个股历史数据（scripts/update.py）",
    "preprocess": "更新预处理数据（scripts/preprocess.py --update）",
    "retrain": (f"季度重训：更新 dataset 并训练新模型"
                f"（仅 {'/'.join(str(m) for m in QUARTER_MONTHS)} 月首个工作日）"),
    "predict": "估计最新股票标签（scripts/prediction.sh）",
    "mail": "发送预测报告邮件（scripts/mail.py，正文与 prediction.log 一致）",
    "cleanup": f"清理 test_output 旧预测目录（仅保留最近 {KEEP_TEST_OUTPUT_DAYS} 天）",
}


def cleanup_test_output(test_output_dir: Path = TEST_OUTPUT_DIR,
                        keep_days: int = KEEP_TEST_OUTPUT_DAYS,
                        dry_run: bool = False) -> list[Path]:
    """清理 test_output 下旧的 pred-<日期> 目录，仅保留最近 keep_days 天。

    - 只处理形如 pred-YYYYMMDD 的目录，按目录名日期升序，保留最新 keep_days 个，
      其余目录及其内容整体删除（其它文件如日志 / cache 一律不动）；
    - 目录名不符合 pred-YYYYMMDD 的目录不参与清理，避免误删手工产物；
    - dry_run=True 时只计算不删除，便于预览。

    返回将被删除（或已删除）的目录路径列表。
    """
    if keep_days <= 0 or not test_output_dir.is_dir():
        return []
    dated: list[tuple[str, Path]] = []
    for p in test_output_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"pred-(\d{8})", p.name)
        if m:
            dated.append((m.group(1), p))  # 目录名日期 YYYYMMDD 可直接按字符串排序
    if len(dated) <= keep_days:
        return []
    dated.sort(key=lambda t: t[0])  # 日期升序，末尾为最新
    expired = [p for _, p in dated[:-keep_days]]
    if not dry_run:
        for p in expired:
            shutil.rmtree(p, ignore_errors=True)
    return expired


def is_weekday(dt: datetime | None = None) -> bool:
    """是否工作日（周一至周五）。datetime.weekday(): 0=周一 ... 6=周日。"""
    return (dt or datetime.now()).weekday() < 5


def is_retrain_day(dt: datetime | None = None) -> bool:
    """是否季度重训日：3/6/9/12 月的首个工作日。

    weekday 定时任务仅在周一至周五运行；当季度首月 1 号恰逢周末时无法触发，
    因此以"该月第一个工作日"作为季度首日执行重训，避免因非交易日而错过。
    """
    dt = dt or datetime.now()
    if dt.month not in QUARTER_MONTHS or not is_weekday(dt):
        return False
    # 该月 1 号至今天的更早日期中不存在工作日，则今天即本月首个工作日
    return all(not is_weekday(dt.replace(day=d)) for d in range(1, dt.day))


def build_commands(python: str, args: argparse.Namespace) -> dict[str, list[str]]:
    """构建各步骤命令（复用现有入口）。"""
    v = ["-v"] if args.verbose else []
    return {
        "pull": [
            python, "-m", "src.pull.pull",
            "--config", str(args.pull_config), *v,
        ],
        "update": [
            python, str(SCRIPTS_DIR / "update.py"),
            "--config", str(args.update_config),
        ],
        "preprocess": [
            python, str(SCRIPTS_DIR / "preprocess.py"),
            "--config", str(args.preprocess_config), "--update", *v,
        ],
        "predict": [
            "bash", str(SCRIPTS_DIR / "prediction.sh"),
            "--config", str(args.predict_config), *v,
        ],
        "mail": [
            python, str(SCRIPTS_DIR / "mail.py"),
            "--config", str(args.mail_config), *v,
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="一键运行：拉取 → 更新 → 预处理 → 预测 → 邮件（调用现有功能接口）")
    parser.add_argument("--pull-config", type=Path, default=DEFAULT_PULL_CONFIG,
                        help=f"pull 配置文件（默认: {DEFAULT_PULL_CONFIG}）")
    parser.add_argument("--update-config", type=Path, default=DEFAULT_UPDATE_CONFIG,
                        help=f"update 配置文件（默认: {DEFAULT_UPDATE_CONFIG}）")
    parser.add_argument("--preprocess-config", type=Path, default=DEFAULT_PREPROCESS_CONFIG,
                        help=f"preprocess 配置文件（默认: {DEFAULT_PREPROCESS_CONFIG}）")
    parser.add_argument("--predict-config", type=Path, default=DEFAULT_PREDICT_CONFIG,
                        help=f"prediction 配置文件（默认: {DEFAULT_PREDICT_CONFIG}）")
    parser.add_argument("--mail-config", type=Path, default=DEFAULT_MAIL_CONFIG,
                        help=f"mail 配置文件（默认: {DEFAULT_MAIL_CONFIG}）")
    parser.add_argument(
        "--steps", default=",".join(STEPS),
        help=f"逗号分隔要执行的步骤（默认全部）: {', '.join(STEPS)}")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印将执行的命令，不实际运行")
    parser.add_argument("--python", default=sys.executable,
                        help="python 解释器（用于 pull/update/preprocess，默认当前解释器）")
    parser.add_argument("--weekday", action="store_true",
                        help="工作日定时模式：仅周一至周五执行，其余日子直接退出")
    parser.add_argument("--force-retrain", action="store_true",
                        help="强制季度重训步骤（忽略 3/6/9/12 月首个工作日的日期判断）")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="向子命令透传 -v 输出调试日志")
    args = parser.parse_args(argv)

    # weekday 模式：非工作日直接退出（节假日停市时下游会自动跳过无新数据的步骤）
    if args.weekday and not is_weekday():
        print(f"[weekday] 今天不是工作日（{datetime.now():%Y-%m-%d %A}），跳过。")
        return 0

    # 校验步骤名
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        parser.error(f"未知步骤: {', '.join(unknown)}（可选: {', '.join(STEPS)}）")
    # 按固定顺序执行
    order = [s for s in STEPS if s in steps]
    if not order:
        parser.error("未选择任何步骤（--steps 至少包含一个）")

    cmds = build_commands(args.python, args)
    for step in order:
        print(f"\n[{step}] {STEP_DESCRIPTION[step]}")

        # retrain 为内置条件步骤：季度首月首个工作日 更新 dataset 并训练新模型
        if step == "retrain":
            today = datetime.now()
            if not (args.force_retrain or is_retrain_day(today)):
                months = "/".join(str(m) for m in QUARTER_MONTHS)
                print(f"  非季度重训日（{today:%F}），跳过（仅 {months} 月首个工作日重训）。")
                continue
            v = ["-v"] if args.verbose else []
            retrain_cmds = [
                # 1) 更新 dataset：重切预处理数据为训练窗口样本
                ["bash", str(SCRIPTS_DIR / "tsai" / "slice.sh"),
                 "--config", str(args.predict_config), *v],
                # 2) 训练新模型（仅训练；完成后自动保留最新 KEEP_MODELS 个模型）
                ["bash", str(SCRIPTS_DIR / "tsai" / "train.sh"), "train",
                 "--config", str(args.predict_config), *v],
            ]
            for cmd in retrain_cmds:
                print(f"  $ {' '.join(cmd)}")
                if args.dry_run:
                    continue
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
                if proc.returncode != 0:
                    print(f"\n[{step}] 失败（退出码 {proc.returncode}），终止后续步骤。",
                          file=sys.stderr)
                    return proc.returncode
            continue

        # cleanup 为内置步骤（非子进程命令）：清理 test_output 旧预测目录
        if step == "cleanup":
            removed = cleanup_test_output(TEST_OUTPUT_DIR, KEEP_TEST_OUTPUT_DAYS,
                                          dry_run=args.dry_run)
            if removed:
                print(f"  清理旧预测目录 {len(removed)} 个"
                      f"（保留最近 {KEEP_TEST_OUTPUT_DAYS} 天）: "
                      + ", ".join(p.name for p in removed))
            else:
                print(f"  无需清理（pred 目录不超过 {KEEP_TEST_OUTPUT_DAYS} 天）")
            continue

        cmd = cmds[step]
        print(f"  $ {' '.join(cmd)}")
        if args.dry_run:
            continue

        # 复用现有入口，输出直接继承当前终端（进度日志可见）
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if proc.returncode != 0:
            print(f"\n[{step}] 失败（退出码 {proc.returncode}），终止后续步骤。",
                  file=sys.stderr)
            return proc.returncode

    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
