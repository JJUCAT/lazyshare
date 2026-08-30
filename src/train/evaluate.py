# -*- coding: utf-8 -*-
"""模型评分计算。

支持两种输入：
    1. verify.py 生成的结构化结果 results.json；
    2. validation 模型文件夹下的 csv 文件（遍历评分统计）。

根据 config/classify_train.json 的 "evaluate" 结构完成评分项计算，
输出评分总结（字符串 / 文件）。

"evaluate" 结构（新格式）：
    {
      "labels": ["T", "B"],   # 评分时只考虑的标签（true 与 pred 均在其中才计入）
      "items":  ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    }
旧格式：直接为评分项列表，等价于只配置 items、labels 为空（不限制）。

支持的评分项（scikit-learn）：
    - accuracy   : 准确率
    - precision  : 精确率（多分类 weighted）
    - recall     : 召回率（多分类 weighted）
    - f1_score   : F1（多分类 weighted）
    - roc_auc    : ROC-AUC（多分类 one-vs-rest，需 probas / P_* 概率列）
另有 evaluate.accuracy_interval：对每个置信度阈值，统计"置信度 >= 阈值的标签占比"与"该子集精度"。

用法：
    python3 src/train/evaluate.py <results.json 或 validation 模型文件夹>
           [--config config/classify_train.json] [-o out.txt]
未指定 -o 时，评分总结自动保存到 目标同目录/summary.log。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "classify_train.json"

LABEL_COL = "峰值标签"
INFER_COL = "infer"
CONF_COL = "conf"

# 支持的评分项 -> (函数, 是否需要概率)
_METRICS = {
    "accuracy": ("accuracy_score", False),
    "precision": ("precision_score", False),
    "recall": ("recall_score", False),
    "f1_score": ("f1_score", False),
    "roc_auc": ("roc_auc_score", True),
}


def load_results(results_path: Path) -> dict:
    """读取结构化验证结果 results.json。"""
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"结构化验证结果不存在: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_samples_from_csv(root: Path) -> tuple[list[dict], list[str]]:
    """遍历验证目录下所有窗口 csv，收集逐日样本 {date,true,pred,conf}。

    - true  取"峰值标签"列
    - pred  取"infer"列
    - conf  取"conf"列

    不再读取 P_N / P_T / P_B 等概率列（评分不依赖标签概率列），probas 恒为 None。
    返回 (samples, classes)。
    """
    root = Path(root)
    samples: list[dict] = []
    classes: set[str] = set()
    for f in sorted(root.rglob("*.csv")):
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except Exception:  # noqa: BLE001
            continue
        if INFER_COL not in df.columns or LABEL_COL not in df.columns:
            continue
        for i in range(len(df)):
            raw_true = df[LABEL_COL].iloc[i]
            true = str(raw_true).strip() if pd.notna(raw_true) else ""
            if not true:
                true = "None"
            pred = str(df[INFER_COL].iloc[i]).strip()
            conf = None
            if CONF_COL in df.columns and pd.notna(df[CONF_COL].iloc[i]):
                conf = float(df[CONF_COL].iloc[i])
            classes.add(true)
            samples.append({
                "date": str(df["日期"].iloc[i]) if "日期" in df.columns else "",
                "true": true,
                "pred": pred,
                "probas": None,
                "conf": conf,
            })
    return samples, sorted(classes)


def build_matrix(samples: list[dict], classes: list[str]):
    """从 samples 构造 y_true / y_pred（标签 -> 类索引）。"""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true, y_pred = [], []
    for s in samples:
        if s["true"] not in class_to_idx or s["pred"] not in class_to_idx:
            continue
        y_true.append(class_to_idx[s["true"]])
        y_pred.append(class_to_idx[s["pred"]])
    return np.array(y_true), np.array(y_pred)


def build_probas(samples: list[dict], n_classes: int) -> np.ndarray:
    """从 samples 构造概率矩阵 (n, n_classes)。

    缺失或含 NaN/Inf 的概率替换为均匀分布，避免 roc_auc 等因 NaN 失败。
    """
    probas = np.zeros((len(samples), n_classes), dtype=float)
    for i, s in enumerate(samples):
        p = s.get("probas")
        if p:
            arr = np.asarray(p, dtype=float)[:n_classes]
            if np.isfinite(arr).all():
                probas[i, :len(arr)] = arr
            else:
                probas[i, :] = 1.0 / n_classes  # 非有限概率 -> 均匀
        else:
            probas[i, :] = 1.0 / n_classes  # 缺失概率时均匀
    return probas


def _load_evaluate_config(config_path: Path):
    """读取配置 evaluate 结构，返回 (metrics, labels, accuracy_intervals)。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    evaluate_cfg = cfg.get("evaluate") or {}
    if isinstance(evaluate_cfg, list):
        evaluate_cfg = {"items": evaluate_cfg, "labels": []}
    metrics = [str(x) for x in (evaluate_cfg.get("items") or [])]
    labels = [str(x) for x in (evaluate_cfg.get("labels") or [])]
    intervals = [float(x) for x in (evaluate_cfg.get("accuracy_interval") or [])]
    return metrics, labels, intervals


def _score_samples(samples: list[dict], classes: list[str],
                   metrics: list[str] | None = None,
                   labels: list[str] | None = None,
                   model_name: str = "",
                   config_path: Path = DEFAULT_CONFIG,
                   accuracy_intervals: list[float] | None = None) -> str:
    """计算评分项，返回评分总结字符串。

    - samples: 逐日样本 {true, pred, probas, ...}
    - classes: 类别列表（probas 列序与之对应）
    - metrics: 评分项列表；为 None 时读取配置 evaluate.items
    - labels: 评分时只考虑的标签（true 与 pred 均在其中才计入）；
              为 None 时读取配置 evaluate.labels；空列表表示不限制。
    - accuracy_intervals: 置信度阈值列表；为 None 时读取配置 evaluate.accuracy_interval。
    """
    if metrics is None or labels is None or accuracy_intervals is None:
        _metrics, _labels, _intervals = _load_evaluate_config(config_path)
        if metrics is None:
            metrics = _metrics
        if labels is None:
            labels = _labels
        if accuracy_intervals is None:
            accuracy_intervals = _intervals

    # 只对指定 labels 中的样本评分（true 与 pred 均需命中）
    if labels:
        label_set = set(labels)
        samples = [s for s in samples
                   if s["true"] in label_set and s["pred"] in label_set]
        if not samples:
            return (f"模型: {model_name}\n"
                    f"评分标签: {labels}\n无符合评分标签的样本")

    # 评分使用的类别与概率列：labels 过滤后只保留对应类别列，保证 roc_auc 等一致
    if labels:
        label_cols = [i for i, c in enumerate(classes) if c in label_set]
        eval_classes = [classes[i] for i in label_cols]
    else:
        label_cols = None
        eval_classes = classes

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true, y_pred = build_matrix(samples, eval_classes)
    n_classes = len(eval_classes)
    probas_full = build_probas(samples, len(classes)) if any(
        _METRICS.get(m, (None, False))[1] for m in metrics) else None
    if probas_full is not None and label_cols is not None:
        probas = probas_full[:, label_cols]
    else:
        probas = probas_full

    lines = [f"模型: {model_name}",
             f"类别: {classes}"]
    if labels:
        lines.append(f"评分标签: {labels}")
    lines.append(f"样本数: {len(samples)}")
    for m in metrics:
        key = m.strip().lower()
        if key not in _METRICS:
            lines.append(f"{m}: 不支持的评分项，已跳过")
            continue
        fn_name, need_prob = _METRICS[key]
        try:
            if key == "accuracy":
                val = accuracy_score(y_true, y_pred)
            elif key == "precision":
                val = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            elif key == "recall":
                val = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            elif key == "f1_score":
                val = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            elif key == "roc_auc":
                if not any(s.get("probas") for s in samples):
                    raise ValueError("缺少标签概率列，无法计算 roc_auc")
                if probas is None or n_classes < 2:
                    raise ValueError("roc_auc 需要概率且至少 2 类")
                if n_classes == 2:
                    val = roc_auc_score(y_true, probas[:, 1])
                else:
                    val = roc_auc_score(y_true, probas, multi_class="ovr", average="weighted")
            lines.append(f"{m}: {val:.4f}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"{m}: 计算失败 ({exc})")

    # accuracy_interval：各置信度阈值的标签占比与精度
    if accuracy_intervals:
        c2i = {c: i for i, c in enumerate(classes)}
        for thr in accuracy_intervals:
            total = 0
            correct = 0
            for s in samples:
                p = s.get("probas")
                pred = s.get("pred")
                if p and pred in c2i:
                    idx = c2i[pred]
                    conf = float(p[idx]) if 0 <= idx < len(p) else None
                else:
                    conf = s.get("conf")
                if conf is not None and np.isfinite(conf) and conf >= thr:
                    total += 1
                    if s["true"] == s["pred"]:
                        correct += 1
            ratio = total / len(samples) if samples else 0.0
            acc = correct / total if total else 0.0
            lines.append(f"[{thr}] 占比 {ratio:.2%}（{total}/{len(samples)}），精度 {acc:.4f}")
    return "\n".join(lines)


def evaluate_results(results_path: Path, metrics: list[str] | None = None,
                     labels: list[str] | None = None,
                     config_path: Path = DEFAULT_CONFIG,
                     accuracy_intervals: list[float] | None = None) -> str:
    """从 results.json 计算评分项，返回评分总结字符串。"""
    results = load_results(results_path)
    classes = list(results.get("classes") or [])
    samples = results.get("samples") or []
    if not classes or not samples:
        raise ValueError("results.json 缺少 classes 或 samples")
    return _score_samples(samples, classes, metrics, labels,
                          model_name=str(results.get("model", "")),
                          config_path=config_path,
                          accuracy_intervals=accuracy_intervals)


def evaluate_from_csv(root: Path, metrics: list[str] | None = None,
                      labels: list[str] | None = None,
                      config_path: Path = DEFAULT_CONFIG,
                      accuracy_intervals: list[float] | None = None) -> str:
    """遍历 validation 目录下 csv 文件，计算评分项，返回评分总结字符串。"""
    samples, classes = collect_samples_from_csv(root)
    if not samples:
        raise ValueError(f"validation 目录下未找到含 infer 列的 csv: {root}")
    if not classes:
        # 无 P_* 概率列时，从 true/pred 推断类别
        classes = sorted({s["true"] for s in samples} | {s["pred"] for s in samples})
    return _score_samples(samples, classes, metrics, labels,
                          model_name=str(Path(root).name),
                          config_path=config_path,
                          accuracy_intervals=accuracy_intervals)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="模型评分计算（读 results.json 或遍历 validation 模型文件夹 csv）")
    parser.add_argument("target", type=Path,
                        help="results.json 路径，或 validation 模型文件夹路径")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="配置文件路径（默认: config/classify_train.json）")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="评分总结输出文件路径（默认打印到 stdout）")
    args = parser.parse_args(argv)

    target = args.target
    if target.is_dir():
        summary = evaluate_from_csv(target, config_path=args.config)
        default_out = target / "summary.log"
    else:
        summary = evaluate_results(target, config_path=args.config)
        default_out = target.parent / "summary.log"

    print(summary)
    if args.output is None:
        args.output = default_out
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(summary + "\n", encoding="utf-8")
    print(f"\n评分总结已保存: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
