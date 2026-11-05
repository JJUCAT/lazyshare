# -*- coding: utf-8 -*-
"""后处理：对 AI 推理结果按 M5C 价格过滤 B / T 信号并重新评估指标。

规则（见 postprocess_plan.md）：
    1. 只处理推理标签为 B / T 的样本（当天模型给出的买卖信号）；
    2. 预测 B 且 当日收盘价 <  M5C → 保留；否则剔除；
       预测 T 且 当日收盘价 >  M5C → 保留；否则剔除；
    3. 对保留的信号重新计算 accuracy / precision / recall / f1 / roc_auc
       以及按类别指标（与验证 / 测试报告同一口径，见 src/train/report.py）。

数据来源：
    - 推理结果：validation/<模型名>/<split>/results.json
      （由 src/train/verify.py 生成，samples 含 path / date / true / pred / probas）
    - 价格与均线：预处理目录 data_source/<源文件名>.csv 的
      "收盘价" 与 "M5C" 列（窗口 CSV 不含这两列，只能按源文件 + 日期回溯）

用法：
    python3 scripts/postprocess.py [--config config/postprocess.json]
        [--model models/<模型名>.pkl | <模型名>] [--split valid|test|all]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.postprocess.config import (  # noqa: E402
    DEFAULT_CONFIG,
    PREPROCESS_CONFIG,
    load_postprocess_config,
    postprocess_enabled,
)
from src.train.report import (  # noqa: E402
    classification_report,
    confidence_table,
    write_report,
)

DATE_COL = "日期"
PRICE_COL = "收盘价"
MA_COL = "M5C"

RESULTS_FILE = "results.json"
LOG_FILE = "postprocess.log"
SUMMARY_FILE = "postprocess.json"

SPLIT_NAMES = {"valid": "验证集（valid）", "test": "测试集（test）"}


def _finite(v) -> float | None:
    """把值转成有限浮点数；无法转换 / NaN / Inf 时返回 None。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _date_key(v) -> str:
    """日期归一为 YYYY-MM-DD（解析失败时保留原始字符串）。"""
    try:
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(v).strip()


class PriceCache:
    """按源股票缓存（日期 → 收盘价 / 均线），避免重复读取同一源 CSV。"""

    def __init__(self, data_source: Path, price_col: str = PRICE_COL,
                 ma_col: str = MA_COL):
        self.data_source = Path(data_source)
        self.price_col = price_col
        self.ma_col = ma_col
        self._cache: dict[str, dict[str, tuple[float | None, float | None]]] = {}

    def _load(self, stem: str) -> dict[str, tuple[float | None, float | None]]:
        if stem in self._cache:
            return self._cache[stem]
        index: dict[str, tuple[float | None, float | None]] = {}
        src = self.data_source / f"{stem}.csv"
        if not src.exists():
            logging.warning("后处理：源文件不存在，无法判定价格（%s）", src)
            self._cache[stem] = index
            return index
        try:
            df = pd.read_csv(
                src, encoding="utf-8-sig",
                usecols=[DATE_COL, self.price_col, self.ma_col],
                dtype={DATE_COL: str},
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("后处理：读取源文件失败 %s: %s", src, exc)
            self._cache[stem] = index
            return index
        prices = pd.to_numeric(df[self.price_col], errors="coerce")
        mas = pd.to_numeric(df[self.ma_col], errors="coerce")
        for raw_date, price, ma in zip(df[DATE_COL], prices, mas):
            key = str(raw_date).strip()
            val = (_finite(price), _finite(ma))
            index[key] = val
            index.setdefault(_date_key(key), val)
        self._cache[stem] = index
        return index

    def get(self, rel_path: str, date: str) -> tuple[float | None, float | None]:
        """按窗口相对路径定位源股票，返回该日 (收盘价, 均线)。"""
        if not rel_path:
            return None, None
        stem = Path(rel_path).parts[0]
        index = self._load(stem)
        if not index:
            return None, None
        hit = index.get(str(date).strip()) or index.get(_date_key(date))
        return hit if hit else (None, None)


def _label_stats(samples: list[dict], label: str) -> dict:
    """单个标签在样本集合内的 precision / recall / f1 / support。"""
    tp = sum(1 for s in samples if s["true"] == label and s["pred"] == label)
    support = sum(1 for s in samples if s["true"] == label)
    pred_n = sum(1 for s in samples if s["pred"] == label)
    prec = tp / pred_n if pred_n else 0.0
    rec = tp / support if support else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"support": support, "pred_n": pred_n, "tp": tp,
            "precision": prec, "recall": rec, "f1": f1}


def _subset_report(samples: list[dict], classes: list[str], heading: str) -> list[str]:
    """在完整类别空间上输出分类指标 + 置信度分档。"""
    lines = ["", f"== {heading} =="]
    if not samples:
        lines.append("无样本")
        return lines
    y_true = [s["true"] for s in samples]
    y_pred = [s["pred"] for s in samples]
    has_probas = all(s.get("probas") for s in samples)
    probas = [s["probas"] for s in samples] if has_probas else None
    lines += classification_report(y_true, y_pred, classes, probas=probas)
    lines.append("")
    lines.append("-- 置信度分档 --")
    lines += (confidence_table(y_true, y_pred, probas, classes)
              if probas else ["缺少概率矩阵，跳过置信度分档"])
    return lines


def score_samples(samples: list[dict], classes: list[str], cfg: dict,
                  split: str = "", model_name: str = "",
                  full_samples: list[dict] | None = None) -> tuple[str, dict]:
    """对已按价格过滤后的样本计算指标。

    samples      : 价格过滤后保留的样本（指标计算对象）
    full_samples : 整个 split 的全部样本（仅用于"相对全量召回"的分母）；
                   缺省时退化用 samples
    """
    buy, sell = cfg["buy"], cfg["sell"]
    base = full_samples if full_samples is not None else samples
    bt_samples = [s for s in samples if s["pred"] in (buy, sell)]
    keep_stats = {
        lab: _label_stats(samples, lab) for lab in (buy, sell)
    }
    # 相对全量（整个 split，含被剔除的信号）的召回：
    # 过滤后保留且正确的正例 / 该 split 内全部真实正例
    global_recall = {}
    for lab in (buy, sell):
        total = sum(1 for s in base if s["true"] == lab)
        tp = sum(1 for s in samples if s["true"] == lab and s["pred"] == lab)
        global_recall[lab] = (tp, total)

    split_name = SPLIT_NAMES.get(split, split)
    lines = _subset_report(
        samples, classes,
        f"价格过滤后指标（{split_name}，{'/'.join([buy, sell])} 信号）")
    lines += ["", "== 过滤后逐标签对比 =="]
    lines.append(f"{'类别':<6}{'support':>9}{'预测数':>9}{'precision':>11}"
                 f"{'recall':>9}{'f1':>9}")
    for lab in (buy, sell):
        st = keep_stats[lab]
        lines.append(f"{lab:<6}{st['support']:>9}{st['pred_n']:>9}"
                     f"{st['precision']:>11.4f}{st['recall']:>9.4f}{st['f1']:>9.4f}")

    lines += ["", "== 相对全量（整个 split）的召回 =="]
    for lab in (buy, sell):
        tp, total = global_recall[lab]
        rec = tp / total if total else 0.0
        lines.append(f"{lab}: 过滤后正确 {tp} / 全量真实 {total} = {rec:.4f}")

    stats = {
        "kept_b": keep_stats[buy],
        "kept_t": keep_stats[sell],
        "global_recall": {lab: {"tp": tp, "support": total,
                                "recall": (tp / total if total else 0.0)}
                          for lab, (tp, total) in global_recall.items()},
        "bt_signal_count": len(bt_samples),
        "model": model_name,
        "split": split,
    }
    return "\n".join(lines), stats


def _resolve_model(model_arg: str | None, cfg: dict) -> str:
    """定位模型名：支持 pkl 路径 / 模型名 / 缺省取 models 目录最新 .pkl。"""
    if model_arg:
        p = Path(model_arg)
        return p.stem if p.suffix == ".pkl" else p.name
    candidates = sorted(
        (p for p in cfg["models"].glob("*.pkl") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"models 目录下未找到 pkl 模型: {cfg['models']}")
    return candidates[-1].stem


def postprocess_split(split: str, model_name: str, cfg: dict,
                      write: bool = True) -> tuple[str, dict]:
    """对单个数据集（valid / test）执行后处理，返回 (总结文本, 统计)。

    - 读取 results.json 的逐样本推理结果；
    - B / T 信号按"收盘价 vs 均线"过滤；
    - 重新计算分类指标，写 postprocess.log / postprocess.json。
    """
    split_dir = Path(cfg["validation"]) / model_name / split
    results_path = split_dir / RESULTS_FILE
    if not results_path.exists():
        raise FileNotFoundError(f"推理结果不存在: {results_path}")

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    samples = list(results.get("samples") or [])
    classes = list(results.get("classes") or [])
    if not classes:
        classes = sorted({s["true"] for s in samples} | {s["pred"] for s in samples})
    if not samples:
        raise ValueError(f"results.json 没有样本: {results_path}")

    buy, sell = cfg["buy"], cfg["sell"]
    price_col, ma_col = cfg["price_col"], cfg["ma_col"]

    # 可选：置信度下限（只统计高置信信号）
    min_conf = cfg.get("min_conf")
    total_samples = len(samples)
    # 整个 split 的全量样本：作为"相对全量召回"的分母（不受置信度筛选影响）
    full_samples = list(samples)
    if min_conf is not None:
        samples = [s for s in samples
                   if _finite(s.get("conf")) is not None
                   and _finite(s.get("conf")) >= min_conf]
        if not samples:
            raise ValueError(f"置信度 >= {min_conf} 的样本为空: {results_path}")

    cache = PriceCache(cfg["data_source"], price_col, ma_col)
    bt_samples = [s for s in samples if s["pred"] in (buy, sell)]
    kept: list[dict] = []
    dropped: list[dict] = []
    missing: list[dict] = []
    drop_reason = {buy: 0, sell: 0}   # B: 价格 >= 均线；T: 价格 <= 均线
    for s in bt_samples:
        price, ma = cache.get(str(s.get("path", "")), str(s.get("date", "")))
        if price is None or ma is None:
            missing.append(s)
            continue
        if s["pred"] == buy:
            ok = price < ma
        else:
            ok = price > ma
        if ok:
            kept.append(s)
        else:
            dropped.append(s)
            drop_reason[s["pred"]] += 1

    kept_b = sum(1 for s in kept if s["pred"] == buy)
    kept_t = sum(1 for s in kept if s["pred"] == sell)
    bt_b = sum(1 for s in bt_samples if s["pred"] == buy)
    bt_t = sum(1 for s in bt_samples if s["pred"] == sell)

    lines = [
        f"模型: {model_name}",
        f"数据集: {SPLIT_NAMES.get(split, split)}",
        f"过滤规则: 预测 {buy} 且 {price_col} < {ma_col} 保留；"
        f"预测 {sell} 且 {price_col} > {ma_col} 保留；其余 {buy}/{sell} 信号剔除",
        f"样本总数: {total_samples}"
        + (f"（置信度 >= {min_conf} 后 {len(samples)}）"
           if min_conf is not None else ""),
        f"原始 {buy}/{sell} 信号: {len(bt_samples)}"
        f"（{buy} {bt_b} / {sell} {bt_t}）",
        f"价格缺失无法判定: {len(missing)}（已剔除）",
        f"过滤后保留: {len(kept)}（{buy} {kept_b} / {sell} {kept_t}），"
        f"剔除 {len(dropped)}（{buy} {drop_reason[buy]} / {sell} {drop_reason[sell]}）",
        f"剔除原因: {buy} 收盘价 >= {ma_col} 共 {drop_reason[buy]} 个；"
        f"{sell} 收盘价 <= {ma_col} 共 {drop_reason[sell]} 个",
        "",
    ]
    lines += _subset_report(bt_samples, classes,
                            "原始 B/T 信号指标（未做价格过滤，对照用）")
    lines.append("")
    score_text, stats = score_samples(kept, classes, cfg, split=split,
                                      model_name=model_name,
                                      full_samples=full_samples)
    lines.append(score_text)

    summary = "\n".join(lines)
    stats.update({
        "total_samples": total_samples,
        "after_conf_samples": len(samples),
        "bt_b": bt_b,
        "bt_t": bt_t,
        "kept": len(kept),
        "kept_b": kept_b,
        "kept_t": kept_t,
        "dropped": len(dropped),
        "dropped_b": drop_reason[buy],
        "dropped_t": drop_reason[sell],
        "missing_price": len(missing),
        "price_col": price_col,
        "ma_col": ma_col,
    })

    if write:
        split_dir.mkdir(parents=True, exist_ok=True)
        write_report(split_dir / LOG_FILE, summary.splitlines())
        write_report(Path(cfg["models"]) / f"{model_name}.{split}.postprocess.log",
                     summary.splitlines())
        with open(split_dir / SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    return summary, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="后处理：按 M5C 过滤 B/T 信号并重新评估精度/召回/F1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="后处理配置文件路径（默认: config/postprocess.json）")
    parser.add_argument("--preprocess-config", type=Path, default=PREPROCESS_CONFIG,
                        help="预处理配置（读取 postprocess.enabled 开关，"
                             "默认: config/preprocess.json）")
    parser.add_argument("--force", action="store_true",
                        help="忽略预处理配置中的 postprocess.enabled 开关，强制执行")
    parser.add_argument("--model", default=None,
                        help="模型 pkl 路径或模型名（默认取 models 目录最新）")
    parser.add_argument("--split", choices=("valid", "test", "all"), default="all",
                        help="要后处理的数据集：valid / test / all（默认）")
    parser.add_argument("--min-conf", type=float, default=None, metavar="X",
                        help="只统计置信度 >= X 的信号（覆盖配置 postprocess.min_conf）")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="额外输出合并总结的文件路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 开关：config/preprocess.json 的 postprocess.enabled（缺省 true）。
    # 关闭时直接跳过，不修改推理结果与验证/测试指标。
    if not args.force and not postprocess_enabled(args.preprocess_config):
        logging.info("后处理已关闭（%s 的 postprocess.enabled=false），跳过；"
                     "推理结果与验证/测试分类指标不受影响", args.preprocess_config)
        return 0

    cfg = load_postprocess_config(args.config)
    if args.min_conf is not None:
        if not 0.0 <= args.min_conf <= 1.0:
            raise SystemExit(f"--min-conf 必须在 [0, 1] 之间，当前 {args.min_conf}")
        cfg["min_conf"] = args.min_conf

    model_name = _resolve_model(args.model, cfg)
    logging.info("后处理模型: %s（价格列 %s，均线列 %s）",
                 model_name, cfg["price_col"], cfg["ma_col"])

    targets = ["valid", "test"] if args.split == "all" else [args.split]
    texts: list[str] = []
    ok = 0
    for split in targets:
        try:
            summary, stats = postprocess_split(split, model_name, cfg)
        except FileNotFoundError as exc:
            logging.warning("[%s] 跳过：%s", split, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            logging.error("[%s] 后处理失败：%s", split, exc)
            continue
        print(summary)
        print()
        texts.append(summary)
        ok += 1
        logging.info("[%s] 保留 %d/%d 个 B/T 信号（B %d, T %d），日志: %s",
                     split, stats["kept"], stats["bt_b"] + stats["bt_t"],
                     stats["kept_b"], stats["kept_t"],
                     Path(cfg["validation"]) / model_name / split / LOG_FILE)

    if ok == 0:
        logging.error("没有任何数据集后处理成功")
        return 1
    if args.output is not None:
        write_report(args.output, "\n\n".join(texts).splitlines())
        print(f"合并总结已保存: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
