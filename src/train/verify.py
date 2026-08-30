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
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
from collections import defaultdict
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


def _model_forward(learn, windows: np.ndarray):
    """直接对模型 forward 得到概率与类别索引。

    绕开 tsai get_X_preds 内部的 vocab 解码（decodes），避免个别模型加载后
    vocab 状态异常导致的 IndexError（vocab[o] 越界）或 reshape 错误。
    模型输出列序 = 训练时 TSClassification 的 vocab（sorted(set(y))）= classes。
    """
    import torch
    dev = next(learn.model.parameters()).device
    x = torch.as_tensor(windows, dtype=torch.float32, device=dev)
    with torch.no_grad():
        logits = learn.model(x)
    probas = torch.softmax(logits, dim=-1).cpu().numpy()
    return probas, np.argmax(probas, axis=-1)


def predict_daily(learn, df: pd.DataFrame, items: list[str],
                  seq_len: int, start: int | None = None,
                  end: int | None = None,
                  class_weights: dict | None = None,
                  classes: list[str] | None = None,
                  ) -> tuple[list[str], list[list[float] | None]]:
    """对整个时间序列批量逐日推理（可只推理 [start, end) 区间）。

    对第 i 天，用其前 seq_len 天（含当天）的特征预测当天标签；
    无法凑齐历史的天（i < seq_len-1，或 start/end 限定区间之外）标签记为 N、
    概率为 None。

    start / end：限定逐日推理的日期索引区间 [start, end)，区间外保持 N。
    用于"裁剪到验证窗口覆盖区间"，避免对整段源序列逐日推理造成浪费。

    class_weights / classes：决策边界按标签权重调整——probas 乘以权重后
    再取最大类别，使高权重标签（如 T/B）更易被选中（与训练损失加权一致）。

    一次性把所有可推理的逐日窗口组成 batch 直接传给模型 forward，
    避免逐样本调用导致的进度条刷屏与低吞吐。

    用 _model_forward 直接计算 softmax 概率（列序 = 训练时 vocab = classes），
    并按 classes 映射类别索引为标签字符串（如 'T'），绕开 get_X_preds
    的 vocab 解码，避免个别模型 vocab 状态异常导致的越界 / reshape 错误。

    返回 (infers, probas_list)，长度均为 len(df)。
    """
    n = len(df)
    infers = ["N"] * n
    probas_list: list[list[float] | None] = [None] * n
    i0 = max(seq_len - 1, 0 if start is None else start)
    i1 = n if end is None else min(n, end)
    idxs = list(range(i0, i1))
    if not idxs:
        return infers, probas_list
    # 一次性构造所有逐日窗口：(m, vars, seq_len)
    windows = np.stack([
        df.iloc[i - seq_len + 1:i + 1][items].to_numpy(dtype=np.float32).T
        for i in idxs
    ])
    probas, preds_idx = _model_forward(learn, windows)
    # 决策边界考虑标签权重：probas 列序 = classes，乘权重后 argmax
    if class_weights and classes:
        w = np.array([float(class_weights.get(c, 1.0)) for c in classes], dtype=float)
        scored = probas * w[None, :]
        for k, i in enumerate(idxs):
            infers[i] = str(classes[int(np.argmax(scored[k]))])
            probas_list[i] = probas[k].tolist()
    else:
        for k, i in enumerate(idxs):
            # 模型输出列序 = classes，按索引映射标签；无 classes 时退回数字索引
            infers[i] = str(classes[int(preds_idx[k])]) if classes else str(int(preds_idx[k]))
            probas_list[i] = probas[k].tolist()
    return infers, probas_list


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


def _append_samples(samples: list[dict], rel: Path, df: pd.DataFrame,
                    win_infers: list[str],
                    win_probas: list[list[float] | None],
                    label: str,
                    win_confs: list[float | None] | None = None) -> None:
    """把窗口内逐日样本追加到 samples（true=当天峰值标签，pred=当天推理，conf=标签置信度）。"""
    for i in range(len(df)):
        raw = df[label].iloc[i]
        true_label = str(raw).strip() if pd.notna(raw) else ""
        if not true_label:
            true_label = "None"
        item = {
            "path": str(rel),
            "date": str(df[DATE_COL].iloc[i]),
            "true": true_label,
            "pred": win_infers[i],
            "probas": win_probas[i],
        }
        if win_confs is not None:
            item["conf"] = win_confs[i]
        samples.append(item)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tsai 时序分类模型验证（写 infer 列 + 结构化结果）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="配置文件路径（默认: config/classify_train.json）")
    parser.add_argument("--model", type=Path, default=None,
                        help="模型 pkl 路径（默认取 models 目录最新 <label>_<arch>-*.pkl）")
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
    valid_paths = [Path(p) for p in m["valid_paths"]]
    items = m.get("items") or cfg["items"]
    seq_len = int(m.get("seq_len", cfg["seq_len"]))
    classes = m.get("classes") or []
    logging.info("验证集样本 %d 条，类别顺序 %s", len(valid_paths), classes)

    learn = load_learner(model_path)

    # 输出根目录：validation/<模型名>/
    out_root = cfg["validation"] / m["model"]
    out_root.mkdir(parents=True, exist_ok=True)

    # 按源股票分组验证窗口（源名 = 窗口相对 dataset 的第一级目录名）
    win_by_src: dict[str | None, list[Path]] = defaultdict(list)
    for p in valid_paths:
        try:
            src_name = p.relative_to(cfg["dataset"]).parts[0]
        except ValueError:
            src_name = None
        win_by_src[src_name].append(p)

    # 源股票逐日推理缓存：src_name -> (infers, probas_list, date_to_idx)
    # 同一源股票的多个窗口只推理一次，避免重复计算与进度条刷屏
    daily_cache: dict[str, tuple] = {}

    samples = []
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

            # 计算推理区间 [start, end)：覆盖所有窗口首日(前推 seq_len-1 天)到末日
            start: int | None = None
            end: int | None = None
            windows_meta: list[tuple[Path, pd.DataFrame]] = []
            for w in wins:
                dw = pd.read_csv(w, encoding="utf-8-sig")
                windows_meta.append((w, dw))
                g0 = date_to_idx.get(str(dw[DATE_COL].iloc[0]))
                if g0 is not None:
                    lo = max(0, g0 - (seq_len - 1))
                    start = lo if start is None else min(start, lo)
                g1 = date_to_idx.get(str(dw[DATE_COL].iloc[-1]))
                if g1 is not None:
                    end = g1 + 1 if end is None else max(end, g1 + 1)

            # 只对窗口覆盖区间逐日推理，大幅减少推理量（而非整段源序列）
            infers, probas_list = predict_daily(learn, df_src, items, seq_len,
                                                start=start, end=end,
                                                class_weights=m.get("loss_label_weight"),
                                                classes=classes)
            daily_cache[src_name] = (infers, probas_list, date_to_idx)
            n_infer = (end or len(df_src)) - (start or 0)
            logging.info("源股票 %s 推理区间 [%s, %s)（%d 天 / 源序列 %d 天）",
                         src_name, start, end, n_infer, len(df_src))

            for w, dw in windows_meta:
                rel = w.relative_to(cfg["dataset"])
                dst_csv = out_root / rel
                win_infers: list[str] = []
                win_probas: list[list[float] | None] = []
                for d in dw[DATE_COL]:
                    g = date_to_idx.get(str(d))
                    if g is None or g >= len(infers):
                        win_infers.append("N")
                        win_probas.append(None)
                    else:
                        win_infers.append(infers[g])
                        win_probas.append(probas_list[g])
                # 标签置信度：预测标签 infer 对应类别的原始概率
                win_confs = [_label_conf(win_probas[i], win_infers[i], classes)
                             for i in range(len(dw))]
                write_validation_csv(w, dst_csv, win_infers, confs=win_confs,
                                     probas_matrix=win_probas, classes=classes)
                _append_samples(samples, rel, dw, win_infers, win_probas,
                                cfg["label"], win_confs=win_confs)
                ok += 1
                if args.verbose:
                    logging.debug("已验证 %s -> %d 天（infer 示例 %s）",
                                  w.name, len(win_infers), win_infers[:3])
        except Exception as exc:  # noqa: BLE001
            # 源股票处理失败：不做窗口内部退化推理，直接跳过该源股票（记 warning）
            logging.warning("源股票 %s 验证失败并跳过：%s", src_name, exc)

    if ok == 0:
        logging.error("验证失败：0 个样本成功")
        return 1

    # 结构化验证结果（供 evaluate.py）
    results = {
        "model": m["model"],
        "label": cfg["label"],
        "classes": classes or sorted({s["true"] for s in samples} | {s["pred"] for s in samples}),
        "samples": samples,
    }
    results_path = out_root / RESULTS_FILE
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info("结构化结果已保存: %s", results_path)

    # 生成模型评分总结日志（调用 evaluate 模块）
    from src.train.evaluate import evaluate_results
    summary_path = out_root / SUMMARY_FILE
    summary = evaluate_results(results_path,
                               metrics=cfg["evaluate"],
                               labels=cfg["evaluate_labels"])
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    logging.info("评分总结已保存: %s", summary_path)

    logging.info("完成：验证 %d/%d 个样本，输出目录 %s", ok, len(valid_paths), out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
