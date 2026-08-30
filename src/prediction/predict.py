# -*- coding: utf-8 -*-
"""股票分类预测。

从 data_source（预处理数据）对每只股票取最新 seq_len 天数据，用最新训练模型
预测最新一天的标签（T / B / N），只看 T / B 并按置信度从高到低保存结果。

流程（见 prediction_plan.md）：
1. 在 test_output 创建 pred-<date> 目录（date 为当前日期）
2. 读取 config/classify_train.json
3. 遍历 data_source，切片最新 seq_len 天数据保存到 pred-<date>/dataset
4. 用最新模型推理，仅分类最新一天的标签
5. 分类结果只看 T / B，按置信度从高到低保存到 pred-<date>/prediction.log

用法（需在 tsai conda 环境运行）：
    python3 scripts/prediction.py [--config config/classify_train.json]
                                  [--model models/xxx.pkl] [--date YYYYMMDD] [-v]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.train.data import DATE_COL, DEFAULT_CONFIG, load_config

CODE_COL = "股票代码"
NAME_COL = "股票名称"


def load_learner(model_path: Path):
    """加载 tsai 导出模型（需在 tsai conda 环境运行）。"""
    from tsai.inference import load_learner as _load_learner
    return _load_learner(model_path)


def model_forward(learn, windows: np.ndarray) -> np.ndarray:
    """对模型 forward，返回 softmax 概率 (n, n_classes)，列序 = 训练时 classes。"""
    import torch
    dev = next(learn.model.parameters()).device
    x = torch.as_tensor(windows, dtype=torch.float32, device=dev)
    with torch.no_grad():
        logits = learn.model(x)
    return torch.softmax(logits, dim=-1).cpu().numpy()


def find_model(models_dir: Path, label: str, arch: str) -> Path:
    """取 models 目录下最新的 <label>_<arch>-*.pkl 模型。"""
    pattern = f"{label}_{arch}-*.pkl"
    candidates = sorted(models_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"models 目录下未找到模型: {models_dir}（匹配 {pattern}）")
    return candidates[-1]


def make_latest_window(raw_file: Path, items: list[str], seq_len: int,
                       dataset_dir: Path) -> tuple[Path | None, str]:
    """切片股票最新 seq_len 天数据并保存到 dataset_dir。

    返回 (输出文件路径, 最新交易日)；数据不足 seq_len 天时返回 (None, "")。
    """
    try:
        # dtype=str：保留股票代码前导零（如 000001），避免被推断为整数
        df = pd.read_csv(raw_file, encoding="utf-8-sig", dtype=str)
    except Exception:  # noqa: BLE001
        return None, ""
    if df.empty:
        return None, ""
    missing = [c for c in items if c not in df.columns]
    if missing:
        return None, ""
    # 按日期升序排序
    df = df.assign(_dt=pd.to_datetime(df[DATE_COL], errors="coerce"))
    df = (df.sort_values("_dt", na_position="last")
            .drop(columns="_dt")
            .reset_index(drop=True))
    # 特征列转数值（dtype=str 读取后）
    for c in items:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 从特征列完整（全非 NaN）的第一天开始，与训练切片语义一致，避免窗口含 NaN
    valid = df[items].notna().all(axis=1)
    if not valid.any():
        return None, ""
    start0 = int(valid.idxmax())
    df = df.iloc[start0:].reset_index(drop=True)
    if len(df) < seq_len:
        return None, ""
    win = df.tail(seq_len).reset_index(drop=True)
    start_date = str(win[DATE_COL].iloc[0])
    end_date = str(win[DATE_COL].iloc[-1])
    out_path = dataset_dir / f"{raw_file.stem}-{start_date}-{end_date}.csv"
    win.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path, end_date


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="股票分类预测（用最新模型预测每只股票最新一天标签）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="配置文件路径（默认: config/classify_train.json）")
    parser.add_argument("--model", type=Path, default=None,
                        help="模型 pkl 路径（默认取 models 目录最新 <label>_<arch>-*.pkl）")
    parser.add_argument("--date", type=str, default=None,
                        help="输出目录日期 YYYYMMDD（默认今天）")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    date_str = args.date or datetime.now().strftime("%Y%m%d")
    out_root = PROJECT_ROOT / "test_output" / f"pred-{date_str}"
    dataset_dir = out_root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    logging.info("输出目录: %s", out_root)

    # 最新模型 + 模型清单（items / seq_len / classes / 标签权重）
    model_path = args.model or find_model(cfg["models"], cfg["label"], cfg["arch"])
    logging.info("使用模型: %s", model_path)
    manifest_path = model_path.with_suffix(".valid.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"模型清单不存在: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    items = m.get("items") or cfg["items"]
    seq_len = int(m.get("seq_len", cfg["seq_len"]))
    classes = m.get("classes") or ["B", "N", "T"]
    weights = m.get("loss_label_weight") or {}
    logging.info("特征列: %s, seq_len: %d, 类别顺序: %s", items, seq_len, classes)

    # 遍历 data_source 切片最新窗口
    raw_files = sorted(cfg["data_source"].glob("*.csv"))
    windows: dict[str, tuple[Path, str]] = {}
    for raw_file in raw_files:
        out_path, end_date = make_latest_window(raw_file, items, seq_len, dataset_dir)
        if out_path:
            windows[raw_file.stem] = (out_path, end_date)
    logging.info("切片窗口 %d 个（数据不足 seq_len=%d 天已跳过）",
                 len(windows), seq_len)

    if not windows:
        logging.warning("没有任何股票可预测，退出")
        return 0

    # 批量推理所有窗口
    learn = load_learner(model_path)
    xs: list[np.ndarray] = []
    metas: list[tuple[str, str, str, str]] = []  # (stem, code, name, date)
    for stem, (out_path, _) in windows.items():
        df = pd.read_csv(out_path, encoding="utf-8-sig", dtype=str)
        for c in items:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        xs.append(df[items].head(seq_len).to_numpy(dtype=np.float32).T)
        last = df.iloc[-1]
        code = str(last[CODE_COL]).strip()
        name = str(last[NAME_COL]).strip()
        metas.append((stem, code, name, str(last[DATE_COL])))
    X_all = np.stack(xs)  # (n, vars, seq_len)
    probas = model_forward(learn, X_all)  # (n, n_classes)
    logging.info("推理完成: %d 只股票", len(metas))

    # 决策边界按标签权重调整（与训练/验证一致）
    w = np.array([float(weights.get(c, 1.0)) for c in classes], dtype=float)
    scored = probas * w[None, :]
    pred_idx = np.argmax(scored, axis=-1)

    # 汇总结果：只看 T / B，按置信度降序
    rows: list[dict] = []
    for (stem, code, name, date), p, pi in zip(metas, probas, pred_idx):
        rows.append({
            "code": code, "name": name,
            "pred": classes[int(pi)], "conf": float(p[int(pi)]),
            "date": date, "stem": stem,
        })
    tb = [r for r in rows if r["pred"] in ("T", "B")]
    tb.sort(key=lambda r: r["conf"], reverse=True)

    # 写 prediction.log
    log_path = out_root / "prediction.log"
    lines = [
        f"预测日期: {date_str}",
        f"模型: {model_path.name}",
        f"数据源: {cfg['data_source']}",
        f"窗口长度(seq_len): {seq_len}",
        f"预测交易日: {rows[0]['date'] if rows else '-'}",
        f"股票总数: {len(rows)}，T/B 命中: {len(tb)}",
        "",
        "分类结果（仅 T / B，按置信度从高到低）:",
        f"{'排名':<6}{'股票代码':<10}{'股票名称':<16}{'预测':<6}{'置信度':<10}{'交易日'}",
    ]
    for i, r in enumerate(tb, 1):
        lines.append(f"{i:<6}{r['code']:<10}{r['name']:<16}{r['pred']:<6}"
                     f"{r['conf']:.4f}    {r['date']}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logging.info("分类结果已保存: %s（T/B %d 条）", log_path, len(tb))
    logging.info("dataset 目录: %s", dataset_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
