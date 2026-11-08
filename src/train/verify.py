# -*- coding: utf-8 -*-
"""tsai 时序分类模型验证。

读取训练时保存的验证集清单（<模型名>.valid.json），对其中未参与训练的
验证集窗口样本进行逐日推理，并把每天预测的标签写入该窗口 CSV 的 "infer" 列。

处理规则（见 tsai_classify_train_plan.md）：
    1. 用没有训练的那部分 "validation_set" 数据验证训练好的模型。
    2. 逐日推理：对源股票，只对其验证窗口覆盖的日期区间（各窗口首日向前
       补 seq_len-1 天历史，到最晚窗口末日）逐日推理，每天用其前 seq_len 天
       （含当天）的特征预测当天标签；无法凑齐历史的天标签记为 N。
    3. 验证集的 csv 文件增加 "infer"（分类标签）与 "conf"（标签置信度）列项。
    4. 在 "validation" 路径下创建模型名子文件夹，存放验证集文件和
       模型评分总结日志。

输出结构：
    validation/<模型名>/
        <源文件名>/<源文件名>-开始-结束.csv   # 验证集文件（含 infer 列）
        results.json                          # 结构化验证结果（供 evaluate.py）
        summary.log                           # 模型评分总结日志

用法：
    python3 src/train/verify.py [--config config/classify_train.json] [--model models/<模型名>.pkl] [-v]
    python3 src/train/verify.py --report-only    # 复用已有 results.json 重新生成报告
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.train.data import (  # noqa: E402
    DATE_COL,
    DEFAULT_CONFIG,
    load_config,
)
from src.train.norm import apply_batch_tfms  # noqa: E402
from src.train.report import (  # noqa: E402
    classification_report,
    confidence_table,
    write_report,
)

INFER_COL = "infer"
CONF_COL = "conf"
RESULTS_FILE = "results.json"
SUMMARY_FILE = "summary.log"


def load_learner(model_path: Path):
    """加载 fastai/tsai 导出的模型（延迟导入）。"""
    from tsai.inference import load_learner
    return load_learner(model_path)


@contextlib.contextmanager
def _suppress_progress():
    """抑制 fastai/tqdm 推理进度条输出。

    进度条经 stdout/stderr 输出，批量推理时吞掉即可避免刷屏；
    推理异常会从 with 块传播，traceback 仍在顶层打印，不受影响。
    """
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        yield


def _model_forward(learn, windows: np.ndarray, batch_tfms=None):
    """直接对模型 forward 得到概率与类别索引。

    绕开 tsai get_X_preds 内部的 vocab 解码（decodes），避免个别模型加载后
    vocab 状态异常导致的 IndexError（vocab[o] 越界）或 reshape 错误。
    模型输出列序 = 训练时 TSClassification 的 vocab（sorted(set(y))）= classes。

    batch_tfms 必须与训练一致：TSStandardize 注册在 DataLoader 的 after_batch
    管线里，不在 learn.model 内部，故这里先 apply_batch_tfms 复现同一标准化
    再前向（见 src/train/norm.py），否则训练/推理输入分布不一致。
    """
    import torch
    dev = next(learn.model.parameters()).device
    arr = apply_batch_tfms(windows, batch_tfms)
    x = torch.as_tensor(arr, dtype=torch.float32, device=dev)
    with torch.no_grad():
        logits = learn.model(x)
    probas = torch.softmax(logits, dim=-1).cpu().numpy()
    return probas, np.argmax(probas, axis=-1)


def predict_last_rows(learn, df: pd.DataFrame, items: list[str], seq_len: int,
                      positions: list[int], batch_tfms=None,
                      class_weights: dict | None = None,
                      classes: list[str] | None = None,
                      ) -> dict[int, tuple[str, list[float] | None]]:
    """只对给定位置（各验证窗口的末日）批量推理。

    与训练口径完全一致：训练样本的标签取窗口最后一天，因此验证也只在
    "窗口末日"产生一个样本——用其前 seq_len 天（含当天）的特征预测当天标签，
    不再对窗口内每一天逐日推理（那会使验证样本数与训练样本密度脱节）。

    positions：窗口末日在源序列中的行索引集合。
    历史不足 seq_len 天或越界的位置会被跳过（不产生样本）。

    class_weights / classes：决策边界按标签权重调整——probas 乘以权重后
    再取最大类别，使高权重标签（如 T/B）更易被选中（与训练损失加权一致）。

    一次性把所有窗口末日组成 batch 传给模型 forward，避免逐样本调用。

    返回 {position: (label, probas)}。
    """
    pos = sorted({int(p) for p in positions
                  if p is not None and seq_len - 1 <= int(p) < len(df)})
    if not pos:
        return {}
    # 每个末日位置构造一个完整窗口：(m, vars, seq_len)
    windows = np.stack([
        df.iloc[i - seq_len + 1:i + 1][items].to_numpy(dtype=np.float32).T
        for i in pos
    ])
    probas, preds_idx = _model_forward(learn, windows, batch_tfms=batch_tfms)
    out: dict[int, tuple[str, list[float] | None]] = {}
    # 决策边界考虑标签权重：probas 列序 = classes，乘权重后 argmax
    if class_weights and classes:
        w = np.array([float(class_weights.get(c, 1.0)) for c in classes], dtype=float)
        scored = probas * w[None, :]
        for k, i in enumerate(pos):
            out[i] = (str(classes[int(np.argmax(scored[k]))]), probas[k].tolist())
    else:
        for k, i in enumerate(pos):
            # 模型输出列序 = classes，按索引映射标签；无 classes 时退回数字索引
            label = str(classes[int(preds_idx[k])]) if classes else str(int(preds_idx[k]))
            out[i] = (label, probas[k].tolist())
    return out


def _src_path(src_name: str, cfg: dict) -> Path | None:
    """按源名返回源 csv 路径（data_source/<源名>.csv），不存在时返回 None。"""
    src = Path(cfg["data_source"]) / f"{src_name}.csv"
    return src if src.exists() else None


def resolve_source_path(window_csv: Path, cfg: dict) -> Path | None:
    """由窗口 csv 定位其源 csv：data_source/<源名>.csv。

    源名取窗口 csv 相对 dataset 目录的第一级目录名（源文件夹名）。
    源文件不存在时返回 None，调用方退化为窗口 csv 内部逐日推理。
    """
    try:
        src_name = window_csv.relative_to(cfg["dataset"]).parts[0]
    except ValueError:
        return None
    return _src_path(src_name, cfg)


def _label_conf(probas: list[float] | None, pred: str,
                classes: list[str]) -> float | None:
    """预测标签 pred 的置信度：probas 中 pred 对应类别的概率；无 probas 时返回 None。"""
    if not probas or not classes:
        return None
    try:
        idx = classes.index(pred)
    except ValueError:
        return None
    if 0 <= idx < len(probas):
        v = float(probas[idx])
        return round(v, 4) if np.isfinite(v) else None  # 过滤 NaN/Inf
    return None


def write_validation_csv(src_csv: Path, dst_csv: Path, infers: list[str],
                         confs: list[float | None] | None = None,
                         probas_matrix: list[list[float] | None] | None = None,
                         classes: list[str] | None = None,
                         infer_col: str = INFER_COL,
                         conf_col: str = CONF_COL) -> None:
    """读取窗口 csv，增加逐日 infer / conf 列，并输出各标签概率列（P_<类别>）写入验证目录。"""
    df = pd.read_csv(src_csv, encoding="utf-8-sig")
    if len(infers) != len(df):
        raise ValueError(
            f"infer 数量({len(infers)})与窗口行数({len(df)})不一致: {src_csv.name}")
    df[infer_col] = infers
    if confs is not None:
        if len(confs) != len(df):
            raise ValueError(
                f"conf 数量({len(confs)})与窗口行数({len(df)})不一致: {src_csv.name}")
        df[conf_col] = confs
    # 各标签概率列（供 evaluate 从 csv 计算 roc_auc / conf）
    if probas_matrix is not None and classes:
        for ci, c in enumerate(classes):
            df[f"P_{c}"] = [
                (row[ci] if row and ci < len(row) else None)
                for row in probas_matrix
            ]
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_csv, index=False, encoding="utf-8-sig")


def find_model(model_path_arg: Path | None, cfg: dict) -> Path:
    """定位模型 pkl：指定或取 models 目录最新。"""
    if model_path_arg is not None:
        model_path = Path(model_path_arg)
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")
        return model_path
    pattern = f"{cfg['label']}_{cfg['arch']}-*.pkl"
    candidates = sorted(cfg["models"].glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"models 目录下未找到模型: {cfg['models']}（匹配 {pattern}）")
    return candidates[-1]  # 最新（按文件名时间排序）


def find_manifest(model_path: Path) -> Path:
    """定位验证集清单 <模型名>.valid.json。"""
    manifest = model_path.with_suffix(".valid.json")
    if not manifest.exists():
        raise FileNotFoundError(f"验证集清单不存在: {manifest}")
    return manifest


def _append_last_sample(samples: list[dict], rel: Path, df: pd.DataFrame,
                        pred: str, probas: list[float] | None, label: str,
                        conf: float | None = None) -> None:
    """只把窗口最后一天追加为评估样本（与训练标签口径一致）。

    true = 窗口末日的峰值标签，pred = 同一窗口末日的推理结果。
    """
    i = len(df) - 1
    raw = df[label].iloc[i]
    true_label = str(raw).strip() if pd.notna(raw) else ""
    if not true_label:
        true_label = "None"
    item = {
        "path": str(rel),
        "date": str(df[DATE_COL].iloc[i]),
        "true": true_label,
        "pred": pred,
        "probas": probas,
    }
    if conf is not None:
        item["conf"] = conf
    samples.append(item)


SPLIT_NAMES = {"valid": "验证集（valid）", "test": "测试集（test）"}


def split_window_paths(manifest: dict, split: str) -> list[Path]:
    """从模型清单取指定数据集（valid / test）的窗口路径。

    新清单含 split_paths（train/valid/test）；旧清单只有 valid_paths，
    此时仅 valid 可用（test 返回空列表）。
    """
    split_paths = manifest.get("split_paths") or {}
    paths = split_paths.get(split)
    if paths is None and split == "valid":
        paths = manifest.get("valid_paths") or []
    return [Path(p) for p in (paths or [])]


def evaluate_split(split: str, window_paths: list[Path], learn, manifest: dict,
                   cfg: dict, items: list[str], seq_len: int,
                   classes: list[str], batch_tfms, verbose: bool = False,
                   ) -> tuple[list[dict], int]:
    """对指定数据集逐窗口末日推理，写 CSV 并返回 (samples, 成功窗口数)。

    输出结构：<validation>/<模型名>/<split>/<源名>/<窗口>.csv（含 infer/conf/P_* 列）
    """
    out_root = cfg["validation"] / manifest["model"] / split
    out_root.mkdir(parents=True, exist_ok=True)

    # 按源股票分组（源名 = 窗口相对 dataset 的第一级目录名）
    win_by_src: dict[str | None, list[Path]] = defaultdict(list)
    for p in window_paths:
        try:
            src_name = p.relative_to(cfg["dataset"]).parts[0]
        except ValueError:
            src_name = None
        win_by_src[src_name].append(p)

    samples: list[dict] = []
    ok = 0
    for src_name, wins in win_by_src.items():
        try:
            if src_name is None:
                raise ValueError("无法定位源股票文件夹")
            src_full = _src_path(src_name, cfg)
            if src_full is None:
                raise FileNotFoundError(f"源文件不存在: {src_full}")
            df_src = pd.read_csv(src_full, encoding="utf-8-sig")
            date_to_idx = {str(d): i for i, d in enumerate(df_src[DATE_COL])}

            # 每个窗口只取末日：定位其在源序列中的行索引，一次性批量推理
            windows_meta: list[tuple[Path, pd.DataFrame, int | None]] = []
            positions: list[int] = []
            for w in wins:
                dw = pd.read_csv(w, encoding="utf-8-sig")
                g1 = date_to_idx.get(str(dw[DATE_COL].iloc[-1]))
                windows_meta.append((w, dw, g1))
                if g1 is not None:
                    positions.append(g1)

            # 只推理窗口末日（与训练口径一致）
            pred_by_pos = predict_last_rows(
                learn, df_src, items, seq_len, positions,
                batch_tfms=batch_tfms,
                # 默认 None（纯 argmax）；训练已用加权损失，推理不再重复加权
                class_weights=(manifest.get("loss_label_weight")
                               if str(manifest.get("decision_weight", "none")).lower()
                               == "loss" else None),
                classes=classes)

            for w, dw, g1 in windows_meta:
                rel = w.relative_to(cfg["dataset"])
                n = len(dw)
                # 只写末行 infer / conf / P_*，其余行留空（训练与评估都只用窗口末日）
                win_infers: list[str] = [""] * (n - 1) + ["N"]
                win_probas: list[list[float] | None] = [None] * n
                pred = pred_by_pos.get(g1) if g1 is not None else None
                if pred is not None:
                    win_infers[-1], win_probas[-1] = pred
                conf = _label_conf(win_probas[-1], win_infers[-1], classes)
                win_confs: list[float | None] = [None] * (n - 1) + [conf]
                write_validation_csv(w, out_root / rel, win_infers, confs=win_confs,
                                     probas_matrix=win_probas, classes=classes)
                _append_last_sample(samples, rel, dw, win_infers[-1],
                                    win_probas[-1], cfg["label"], conf=conf)
                ok += 1
                if verbose:
                    logging.debug("[%s] %s -> 末日 %s（infer %s, conf %s）",
                                  split, w.name, dw[DATE_COL].iloc[-1],
                                  win_infers[-1], conf)
        except Exception as exc:  # noqa: BLE001
            logging.warning("[%s] 源股票 %s 处理失败并跳过：%s", split, src_name, exc)
    return samples, ok


def write_split_reports(split: str, samples: list[dict], classes: list[str],
                        manifest: dict, cfg: dict) -> list[Path]:
    """生成该数据集（valid / test）的分类报告，写两份：

    - 模型目录：<models>/<模型名>.<split>.log（用户要求的验证/测试报告）
    - 验证目录：<validation>/<模型名>/<split>/summary.log

    指标在完整类别空间上计算（不做 true/pred 双向过滤），漏报计入 recall 分母。
    """
    y_true = [s["true"] for s in samples]
    y_pred = [s["pred"] for s in samples]
    has_probas = bool(samples) and all(s.get("probas") for s in samples)
    probas = [s["probas"] for s in samples] if has_probas else None

    lines = [
        f"模型: {manifest['model']}",
        f"数据集: {SPLIT_NAMES.get(split, split)}",
        f"生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"类别顺序: {classes}",
        f"样本数: {len(samples)}",
        "",
        "== 分类指标 ==",
        *classification_report(y_true, y_pred, classes, probas=probas),
        "",
        "== 置信度分档指标（conf = 被预测类别的概率；指标同 evaluate.items） ==",
        *(confidence_table(
            y_true, y_pred, probas, classes,
            intervals=cfg.get("confidence_intervals") or None,
            metrics=cfg.get("evaluate") or None,
            labels=cfg.get("evaluate_labels") or None)
          if probas else ["缺少概率矩阵，跳过置信度分档"]),
    ]

    model_report = cfg["models"] / f"{manifest['model']}.{split}.log"
    write_report(model_report, lines)
    summary_path = cfg["validation"] / manifest["model"] / split / SUMMARY_FILE
    write_report(summary_path, lines)
    return [model_report, summary_path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="tsai 时序分类模型验证 / 测试（写 infer 列 + 分类报告）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="配置文件路径（默认: config/classify_train.json）")
    parser.add_argument("--model", type=Path, default=None,
                        help="模型 pkl 路径（默认取 models 目录最新 <label>_<arch>-*.pkl）")
    parser.add_argument("--split", choices=("valid", "test", "all"), default="all",
                        help="要评估的数据集：valid=验证集，test=测试集，all=两者（默认）")
    parser.add_argument("--report-only", action="store_true",
                        help="复用已有 results.json 重新生成报告（不重新推理）")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    model_path = find_model(args.model, cfg)
    logging.info("使用模型: %s", model_path)

    manifest = find_manifest(model_path)
    with open(manifest, "r", encoding="utf-8") as f:
        m = json.load(f)
    items = m.get("items") or cfg["items"]
    seq_len = int(m.get("seq_len", cfg["seq_len"]))
    classes = m.get("classes") or []
    # 训练时的 batch_tfms，推理前必须复现（旧模型清单无该字段时回退到当前配置）
    batch_tfms = m.get("batch_tfms", cfg.get("batch_tfms"))
    logging.info("类别顺序 %s，batch_tfms: %s，划分比例: %s",
                 classes, batch_tfms, m.get("split_ratio"))

    targets = ["valid", "test"] if args.split == "all" else [args.split]

    # --report-only：复用已有 results.json 重新生成报告（不加载模型、不重新推理）
    if args.report_only:
        total = 0
        for split in targets:
            results_path = cfg["validation"] / m["model"] / split / RESULTS_FILE
            if not results_path.exists():
                logging.warning("[%s] results.json 不存在，跳过: %s",
                                split, results_path)
                continue
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            samples = results.get("samples") or []
            res_classes = results.get("classes") or classes
            if not samples:
                logging.warning("[%s] results.json 无样本，跳过", split)
                continue
            reports = write_split_reports(split, samples, res_classes, m, cfg)
            for p in reports:
                logging.info("[%s] 报告已重新生成: %s", split, p)
            total += len(samples)
        if total == 0:
            logging.error("没有任何 results.json 可用于重新生成报告")
            return 1
        logging.info("(--report-only) 完成：复用 %d 个已有推理样本，报告位于 %s",
                     total, cfg["models"])
        return 0

    learn = load_learner(model_path)

    total_ok = 0
    for split in targets:
        window_paths = split_window_paths(m, split)
        if not window_paths:
            logging.warning("模型清单中没有 %s 集样本，跳过"
                            "（旧模型清单只含 valid_paths，需重新训练）", split)
            continue
        logging.info("[%s] 开始评估：%d 个窗口（%s）",
                     split, len(window_paths), SPLIT_NAMES.get(split, split))
        samples, ok = evaluate_split(split, window_paths, learn, m, cfg,
                                    items, seq_len, classes, batch_tfms,
                                    verbose=args.verbose)
        if ok == 0:
            logging.error("[%s] 0 个样本成功，跳过报告生成", split)
            continue

        # 结构化结果（供 evaluate.py / 二次分析）
        results = {
            "model": m["model"],
            "split": split,
            "label": cfg["label"],
            "classes": classes or sorted({s["true"] for s in samples}
                                         | {s["pred"] for s in samples}),
            "samples": samples,
        }
        results_path = cfg["validation"] / m["model"] / split / RESULTS_FILE
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logging.info("[%s] 结构化结果: %s", split, results_path)

        reports = write_split_reports(split, samples, results["classes"], m, cfg)
        for p in reports:
            logging.info("[%s] 报告已保存: %s", split, p)
        total_ok += ok

    if total_ok == 0:
        logging.error("没有任何数据集评估成功")
        return 1
    logging.info("完成：共评估 %d 个窗口，报告位于 %s", total_ok, cfg["models"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
