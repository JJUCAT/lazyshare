# -*- coding: utf-8 -*-
"""训练 / 验证 / 测试报告生成（统一口径）。

上游：
    - train.py  写训练报告（数据划分统计 + 逐 epoch 训练曲线）
    - verify.py 写验证 / 测试报告（分类指标 + 混淆矩阵 + 置信度分档）

设计要点（与旧 evaluate.py 的差异）：
    - **不做 true/pred 双向过滤**。旧口径 `true in labels and pred in labels`
      会把"真实是 T/B 但模型漏报成 N"的样本整批剔除，导致 recall 恒为 1.0，
      任何模型都拿满分。这里一律在完整类别空间上计算，漏报计入 recall 的分母。
    - 每类指标单独输出（precision / recall / f1 / specificity / support），
      并给出 macro 与 weighted 平均，避免大类（N）掩盖小类（B）的失效。
    - 置信度分档同时给出"命中数 / 占比 / 精度"，便于选择阈值。
"""
from __future__ import annotations

import numpy as np

# 置信度分档默认阈值（与 config.evaluate.accuracy_interval 保持一致）
DEFAULT_INTERVALS = (0.99, 0.95, 0.90, 0.85, 0.80)


def _pct(a: int, b: int) -> str:
    return f"{(a / b * 100):.2f}%" if b else "-"


def split_table(y, splits: dict[str, np.ndarray], classes: list[str]) -> str:
    """输出训练 / 验证 / 测试三集合的样本数与标签分布对比表。

    y      : 全样本标签数组
    splits : {"train": idx, "valid": idx, "test": idx}
    classes: 类别顺序
    """
    y = np.asarray(y)
    total = len(y)
    lines = [f"样本总数: {total}",
             f"整体标签分布: " + "  ".join(
                 f"{c} {int((y == c).sum())}({_pct(int((y == c).sum()), total)})"
                 for c in classes),
             ""]

    header = f"{'集合':<8}{'样本数':>8}{'占比':>9}"
    for c in classes:
        header += f"{c + '数':>8}{c + '占比':>10}"
    lines.append(header)

    for name in ("train", "valid", "test"):
        idx = np.asarray(splits.get(name, []), dtype=int)
        sub = y[idx] if len(idx) else np.asarray([], dtype=y.dtype)
        n = len(sub)
        row = f"{name:<8}{n:>8}{_pct(n, total):>9}"
        for c in classes:
            k = int((sub == c).sum())
            row += f"{k:>8}{_pct(k, n):>10}"
        lines.append(row)

    # 比例一致性检查：各集合每类占比与全集占比的最大偏差（百分点）
    lines.append("")
    overall = {c: float((y == c).sum()) / total for c in classes}
    worst = 0.0
    worst_desc = ""
    for name in ("train", "valid", "test"):
        idx = np.asarray(splits.get(name, []), dtype=int)
        if not len(idx):
            continue
        sub = y[idx]
        for c in classes:
            dev = abs(float((sub == c).sum()) / len(sub) - overall[c]) * 100
            if dev > worst:
                worst, worst_desc = dev, f"{name}/{c}"
    lines.append(f"分层校验: 各集合类别占比与全集的最大偏差 {worst:.2f} 个百分点"
                 f"（{worst_desc}）")
    return "\n".join(lines)


def confusion_matrix(y_true, y_pred, classes: list[str]) -> np.ndarray:
    """混淆矩阵（行=真实，列=预测），顺序按 classes。"""
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def classification_report(y_true, y_pred, classes: list[str],
                          probas=None) -> list[str]:
    """分类指标：accuracy、混淆矩阵、每类 P/R/F1/specificity、macro/weighted 平均。

    y_true / y_pred : 标签序列（完整类别空间，不做过滤）
    probas          : 可选，(n, n_classes) 概率矩阵，用于计算 roc_auc(ovr)
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    if n == 0:
        return ["无样本"]
    cm = confusion_matrix(y_true, y_pred, classes)
    correct = int(np.trace(cm))

    lines = [f"样本数: {n}",
             f"accuracy: {correct / n:.4f}（{correct}/{n}）",
             ""]

    # 混淆矩阵
    lines.append("混淆矩阵（行=真实，列=预测）:")
    head = "真实/预测".ljust(12) + "".join(f"{c:>8}" for c in classes) + f"{'合计':>10}"
    lines.append(head)
    for i, c in enumerate(classes):
        row = f"{c:<10}" + "".join(f"{int(cm[i, j]):>8}" for j in range(len(classes)))
        lines.append(row + f"{int(cm[i].sum()):>10}")
    lines.append(f"{'合计':<10}" + "".join(f"{int(cm[:, j].sum()):>8}"
                                          for j in range(len(classes)))
                 + f"{int(cm.sum()):>10}")

    # 每类指标
    lines.append("")
    lines.append(f"{'类别':<8}{'support':>9}{'预测数':>9}{'precision':>11}"
                 f"{'recall':>9}{'f1':>8}{'specificity':>13}")
    rows = []
    for i, c in enumerate(classes):
        tp = cm[i, i]
        support = cm[i].sum()
        pred_n = cm[:, i].sum()
        fp = pred_n - tp
        fn = support - tp
        tn = cm.sum() - tp - fp - fn
        prec = tp / pred_n if pred_n else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append((prec, rec, f1, support))
        lines.append(f"{c:<8}{int(support):>9}{int(pred_n):>9}{prec:>11.4f}"
                     f"{rec:>9.4f}{f1:>8.4f}{spec:>13.4f}")

    # macro 与 weighted 平均
    precs = [r[0] for r in rows]
    recs = [r[1] for r in rows]
    f1s = [r[2] for r in rows]
    supports = [r[3] for r in rows]
    tot = sum(supports)
    w_prec = sum(p * s for p, s in zip(precs, supports)) / tot if tot else 0.0
    w_rec = sum(r * s for r, s in zip(recs, supports)) / tot if tot else 0.0
    w_f1 = sum(f * s for f, s in zip(f1s, supports)) / tot if tot else 0.0
    lines.append("")
    lines.append(f"{'macro avg':<8}{tot:>9}{'':>9}{np.mean(precs):>11.4f}"
                 f"{np.mean(recs):>9.4f}{np.mean(f1s):>8.4f}")
    lines.append(f"{'weighted':<8}{tot:>9}{'':>9}{w_prec:>11.4f}"
                 f"{w_rec:>9.4f}{w_f1:>8.4f}")

    # ROC-AUC（多分类 one-vs-rest）
    if probas is not None and len(probas):
        try:
            from sklearn.metrics import roc_auc_score
            from sklearn.preprocessing import label_binarize
            idx = {c: i for i, c in enumerate(classes)}
            y_bin = label_binarize([idx[t] for t in y_true],
                                   classes=list(range(len(classes))))
            auc = roc_auc_score(y_bin, np.asarray(probas), multi_class="ovr",
                                average="weighted")
            lines.append(f"\nroc_auc (ovr, weighted): {auc:.4f}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"\nroc_auc: 计算失败（{exc}）")
    return lines


def confidence_table(y_true, y_pred, probas, classes: list[str],
                     intervals=DEFAULT_INTERVALS) -> list[str]:
    """置信度分档统计：conf >= 阈值的样本占比，以及该子集的精度。

    conf 取"被预测类别的概率"（未做任何加权），与实际决策口径一致。
    """
    if not len(y_true):
        return ["无样本"]
    idx = {c: i for i, c in enumerate(classes)}
    confs = []
    for t, p, row in zip(y_true, y_pred, probas):
        i = idx.get(p)
        confs.append(float(row[i]) if (i is not None and i < len(row)) else float("nan"))
    confs = np.asarray(confs, dtype=float)
    correct = np.asarray([t == p for t, p in zip(y_true, y_pred)])

    lines = [f"置信度: 均值 {np.nanmean(confs):.4f}  中位数 {np.nanmedian(confs):.4f}"
             f"  最低 {np.nanmin(confs):.4f}",
             f"{'阈值':<8}{'命中数':>8}{'占比':>10}{'精度':>10}"]
    for thr in intervals:
        mask = np.isfinite(confs) & (confs >= thr)
        n_hit = int(mask.sum())
        acc = float(correct[mask].mean()) if n_hit else 0.0
        lines.append(f"{thr:<8}{n_hit:>8}{_pct(n_hit, len(confs)):>10}{acc:>10.4f}")
    return lines


def write_report(path, lines) -> None:
    """把报告行写入文件（覆盖），自动创建父目录。"""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
