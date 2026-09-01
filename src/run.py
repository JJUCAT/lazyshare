#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键运行：拉取 → 更新 → 预处理 → 预测（尽量调用现有功能接口）。

流程（见 run_plan.md）：
    1. pull      : 使用 pull.json 拉取最新数据
    2. update    : 使用 update.json 更新个股历史数据
    3. preprocess: 使用 preprocess.json 更新预处理数据
    4. prediction: 使用 prediction.sh 估计最新股票标签

本脚本只做编排：通过 subprocess 调用现有入口，不重复实现内部逻辑。
    - pull / update / preprocess 使用系统 python3（可用 --python 指定）
    - prediction 使用 scripts/prediction.sh，自动激活 tsai conda 环境

用法：
    python3 scripts/run.py                              # 全流程
    python3 scripts/run.py --steps pull,update          # 仅拉取 + 更新
    python3 scripts/run.py --dry-run                    # 预览将执行的命令，不执行
    python3 scripts/run.py -v                           # 向子命令透传 -v 调试日志
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

DEFAULT_PULL_CONFIG = PROJECT_ROOT / "config" / "pull.json"
DEFAULT_UPDATE_CONFIG = PROJECT_ROOT / "config" / "update.json"
DEFAULT_PREPROCESS_CONFIG = PROJECT_ROOT / "config" / "preprocess.json"
DEFAULT_PREDICT_CONFIG = PROJECT_ROOT / "config" / "classify_train.json"

# 固定执行顺序
STEPS = ("pull", "update", "preprocess", "predict")

STEP_DESCRIPTION = {
    "pull": "拉取最新数据（src.pull.pull）",
    "update": "更新个股历史数据（scripts/update.py）",
    "preprocess": "更新预处理数据（scripts/preprocess.py --update）",
    "predict": "估计最新股票标签（scripts/prediction.sh）",
}


def is_weekday(dt: datetime | None = None) -> bool:
    """是否工作日（周一至周五）。datetime.weekday(): 0=周一 ... 6=周日。"""
    return (dt or datetime.now()).weekday() < 5


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
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="一键运行：拉取 → 更新 → 预处理 → 预测（调用现有功能接口）")
    parser.add_argument("--pull-config", type=Path, default=DEFAULT_PULL_CONFIG,
                        help=f"pull 配置文件（默认: {DEFAULT_PULL_CONFIG}）")
    parser.add_argument("--update-config", type=Path, default=DEFAULT_UPDATE_CONFIG,
                        help=f"update 配置文件（默认: {DEFAULT_UPDATE_CONFIG}）")
    parser.add_argument("--preprocess-config", type=Path, default=DEFAULT_PREPROCESS_CONFIG,
                        help=f"preprocess 配置文件（默认: {DEFAULT_PREPROCESS_CONFIG}）")
    parser.add_argument("--predict-config", type=Path, default=DEFAULT_PREDICT_CONFIG,
                        help=f"prediction 配置文件（默认: {DEFAULT_PREDICT_CONFIG}）")
    parser.add_argument(
        "--steps", default=",".join(STEPS),
        help=f"逗号分隔要执行的步骤（默认全部）: {', '.join(STEPS)}")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印将执行的命令，不实际运行")
    parser.add_argument("--python", default=sys.executable,
                        help="python 解释器（用于 pull/update/preprocess，默认当前解释器）")
    parser.add_argument("--weekday", action="store_true",
                        help="工作日定时模式：仅周一至周五执行，其余日子直接退出")
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
        cmd = cmds[step]
        print(f"\n[{step}] {STEP_DESCRIPTION[step]}")
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
