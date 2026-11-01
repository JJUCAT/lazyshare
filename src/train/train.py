# -*- coding: utf-8 -*-
"""tsai 时序分类训练。

从 config/classify_train.json 读取配置：
    - dataset    ：切片后的数据集目录
    - models     ：训练后的模型保存路径
    - train      ：训练参数（arch / tfms / batch_tfms / batch_size / epochs / learning_rate / validation_set / seed）

处理规则（见 tsai_classify_train_plan.md）：
    1. 从 "dataset" 随机读取 (1.0 - "validation_set") 数据，按 "train" 训练参数训练。
    2. 模型用"任务-时间-batch大小-epochs大小"命名，
       如 <label>_<arch>-YYYYMMDD_HHMMSS-bs32-ep40.pkl。
    3. 同时保存验证集样本清单（<模型名>.valid.json），供 verify.py 使用。
    4. 训练后清理旧模型：同任务（<label>_<arch>）下仅保留最新 KEEP_MODELS 个
       模型文件及其验证集清单，避免 models 目录无限增长。

用法：
    python3 src/train/train.py [--config config/classify_train.json] [--limit N] [--dirs N] [-v]
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.train.data import (  # noqa: E402
    DEFAULT_CONFIG,
    collect_samples,
    decision_range,
    label_counts,
    load_config,
    span_range,
    time_split,
)
from src.train.report import split_table, write_report  # noqa: E402

# 同任务（<label>_<arch>）下保留的最新模型数量，训练后清理更旧的模型
KEEP_MODELS = 10


def build_batch_tfms(name):
    """由配置构造 tsai batch 变换；null / 空 / "none" 表示无操作（源数据直接输入）。"""
    if not name:
        return None
    key = str(name).strip().lower()
    if key == "tsstandardize":
        from tsai.all import TSStandardize
        return TSStandardize(by_sample=True)
    if key in ("none", "noop", "identity"):
        return None
    raise ValueError(
        f"不支持的 batch_tfms: {name!r}（可用 \"TSStandardize\"，或 null 表示无操作）")


def build_ts_classifier(X, y, train_idx, valid_idx, cfg: dict):
    """按配置构建 TSClassifier（延迟导入 tsai，避免无环境时报错）。"""
    import torch  # noqa: F401
    import tsai  # noqa: F401
    from tsai.all import (
        TSClassifier,
        TSClassification,
        accuracy,
    )

    splits = (train_idx.tolist(), valid_idx.tolist())
    tfms = [None, TSClassification()]
    batch_tfms = build_batch_tfms(cfg.get("batch_tfms"))

    # 损失函数标签权重：loss_label_weight {标签: 权重}，按类别 vocab 顺序排列。
    # TSClassification 的类别顺序 = sorted(set(y))（如 ['B','N','T']），权重须一一对应。
    loss_func = None
    loss_label_weight = cfg.get("loss_label_weight")
    if loss_label_weight:
        vocab = sorted(set(y.tolist()))
        weight = torch.tensor(
            [float(loss_label_weight.get(lab, 1.0)) for lab in vocab],
            dtype=torch.float32,
        )
        loss_func = torch.nn.CrossEntropyLoss(weight=weight)
        logging.info("损失标签权重: %s（按类别顺序 %s）",
                     dict(zip(vocab, weight.tolist())), vocab)

    clf = TSClassifier(
        X, y, splits=splits,
        arch=cfg["arch"],
        tfms=tfms,
        batch_tfms=batch_tfms,
        loss_func=loss_func,
        metrics=accuracy,
        bs=cfg["batch_size"],
        seed=cfg["seed"],
        verbose=False,
    )
    return clf


def build_train_callbacks(cfg: dict, model_name: str, models_dir: Path):
    """构造"早停 + 最优权重回滚"回调。

    解决的问题：`fit_one_cycle(epochs, lr)` 跑满全部 epoch 后导出的是**最后一轮**
    的权重，而验证损失通常在早中期就已触底（本任务 epoch 13 的 valid_loss 0.390，
    epoch 30 已涨到 1.190）。导出过拟合权重会使少数类（B）的决策边界进一步塌缩。

    两个回调分工：
        - SaveModelCallback   ：每轮按 monitor 比较，刷新最优权重文件；训练结束时
                                fastai 会自动 load 回最优权重（on_train_end）。
        - EarlyStoppingCallback：连续 patience 轮没有超过 min_delta 的改善就提前终止。

    监控指标优先用 valid_loss（连续、稳定）。用 accuracy 这类离散指标时平台期会
    反复抖动，min_delta 建议给 0.001~0.01。

    注意：fit_one_cycle 的 lr 调度按**总 epoch 数**规划，若早停频繁在调度中段触发，
    模型还没走完 lr 衰减段；此时可把 epochs 调大或改用平缓调度（fit_flat_cos）。

    返回 (callbacks, info)。info 含 monitor / patience / min_delta / lower_better /
    best_name / best_file，供日志、训练报告与样本清单记录。patience=0 时只回滚不早停。
    """
    from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback

    es = cfg.get("early_stopping") or {}
    monitor = str(es.get("monitor") or "valid_loss").strip() or "valid_loss"
    patience = int(es.get("patience", 5))
    min_delta = float(es.get("min_delta", 0.0))
    if patience < 0:
        raise ValueError(
            f"early_stopping.patience 必须 >= 0（0 表示不早停），当前 {patience}")
    # loss / error 类指标越小越好，accuracy / f1 等越大越好
    lower_better = any(k in monitor.lower()
                       for k in ("loss", "error", "rmse", "mae", "mse"))
    comp = np.less if lower_better else np.greater

    # 用绝对路径作为 fname：fastai 的 save/load 都做 path/f'{name}.pth'，
    # 而 Path 拼接绝对路径会直接替换左侧，故不受 learn.path 影响。
    # 目录必须先存在：SaveModelCallback 的 after_epoch 直接 torch.save，
    # 父目录缺失会抛 RuntimeError 中断训练（不依赖调用方是否已建目录）。
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    best_name = str(Path(models_dir) / f"{model_name}.best")
    cbs = []
    if cfg.get("save_best", True):
        cbs.append(SaveModelCallback(
            monitor=monitor, comp=comp, min_delta=min_delta,
            fname=best_name, with_opt=False, at_end=False, reset_on_fit=True))
    if patience > 0:
        cbs.append(EarlyStoppingCallback(
            monitor=monitor, comp=comp, min_delta=min_delta, patience=patience))
    info = {
        "monitor": monitor,
        "patience": patience,
        "min_delta": min_delta,
        "lower_better": lower_better,
        "best_name": best_name if cfg.get("save_best", True) else None,
        "best_file": best_name + ".pth",
    }
    return cbs, info


def prune_models(models_dir: Path, task: str, keep: int = KEEP_MODELS) -> list[Path]:
    """清理旧模型：仅保留同任务下最新的 keep 个模型及其验证集清单。

    模型文件名格式为 <label>_<arch>-YYYYMMDD_HHMMSS-bs<bs>-ep<ep>.pkl，
    时间戳定长且高位在前，故文件名字典序即时间先后顺序；保留字典序最大的
    keep 个，删除其余 .pkl 及同名的 <模型名>.valid.json。

    返回被删除的文件路径列表。
    """
    if keep <= 0:
        return []
    candidates = sorted(models_dir.glob(f"{task}-*.pkl"))
    if len(candidates) <= keep:
        return []
    removed: list[Path] = []
    for old in candidates[:-keep]:
        for p in (old, old.with_suffix(".valid.json")):
            if p.exists():
                p.unlink()
                removed.append(p)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="tsai 时序分类训练")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="配置文件路径（默认: config/classify_train.json）")
    parser.add_argument("--limit", type=int, default=None,
                        help="最多读取 N 个窗口样本（调试/小模型用）")
    parser.add_argument("--dirs", type=int, default=None,
                        help="最多读取 N 个源文件夹的样本（调试用）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖配置中的 epochs")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    logging.info("随机种子: %d", cfg["seed"])
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    logging.info("读取样本: dataset=%s (limit=%s, dirs=%s, train_max=%s)",
                 cfg["dataset"], args.limit, args.dirs, cfg["train_max"])
    X, y, paths, spans = collect_samples(
        cfg["dataset"], cfg["items"], cfg["label"], cfg["seq_len"],
        max_samples=args.limit, max_dirs=args.dirs,
        train_max=cfg["train_max"], seed=cfg["seed"],
    )
    n = len(X)
    logging.info("共 %d 个样本，X 形状 %s，特征列 %s",
                 n, X.shape, cfg["items"])
    logging.info("标签分布: %s",
                 {k: int((y == k).sum()) for k in sorted(set(y.tolist()))})

    # 按窗口时间顺序划分 训练/验证/测试（严格 训练 < 验证 < 测试，测试时间最新）：
    # 时间轴连续切三段，边界处按 purge_days 丢弃样本，避免三元屏蔽标签的前瞻
    # 窗口跨进下一个集合（未来数据泄漏，见 src/train/data.py time_split）。
    train_idx, valid_idx, test_idx, split_info = time_split(
        spans, cfg["split_ratio"], cfg["purge_days"])
    split_indices = {"train": train_idx, "valid": valid_idx, "test": test_idx}
    split_paths = {name: [str(paths[i]) for i in idx]
                   for name, idx in split_indices.items()}
    split_stats = {name: label_counts(y, idx) for name, idx in split_indices.items()}
    # 窗口特征区间（可能因长停牌而互相重叠）与决策日区间（窗口末日，严格有序）
    split_ranges = {name: list(span_range(spans, idx))
                    for name, idx in split_indices.items()}
    split_decision = {name: list(decision_range(spans, idx))
                      for name, idx in split_indices.items()}
    logging.info("按时间划分 %s: 训练 %d，验证 %d，测试 %d"
                 "（purge %d 个交易日，仅丢弃标签前瞻跨集合的 %d 个样本）",
                 "/".join(f"{r:g}" for r in cfg["split_ratio"]),
                 len(train_idx), len(valid_idx), len(test_idx),
                 cfg["purge_days"], split_info["purged"])
    for _name in ("train", "valid", "test"):
        _st = split_stats[_name]
        _dec = split_decision[_name]
        logging.info("  %-5s 决策日 %s ~ %s  标签分布: %s",
                     _name, _dec[0], _dec[1], _st)

    clf = build_ts_classifier(X, y, train_idx, valid_idx, cfg)

    # 模型用"任务-时间-batch大小-epochs大小"命名
    # 需在训练前生成：最优权重临时文件与其同名（见 build_train_callbacks）
    task = f"{cfg['label']}_{cfg['arch']}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{task}-{timestamp}-bs{cfg['batch_size']}-ep{cfg['epochs']}"
    cfg["models"].mkdir(parents=True, exist_ok=True)
    model_path = cfg["models"] / f"{model_name}.pkl"

    callbacks, es_info = build_train_callbacks(cfg, model_name, cfg["models"])
    logging.info("开始训练: arch=%s, batch_size=%d, epochs=%d, lr=%s",
                 cfg["arch"], cfg["batch_size"], cfg["epochs"], cfg["learning_rate"])
    logging.info("早停/最优权重回滚: monitor=%s, patience=%d, min_delta=%s, save_best=%s",
                 es_info["monitor"], es_info["patience"], es_info["min_delta"],
                 cfg.get("save_best", True))
    clf.fit_one_cycle(cfg["epochs"], cfg["learning_rate"], cbs=callbacks or None)

    trained_epochs = len(clf.recorder.values)
    stopped_early = trained_epochs < cfg["epochs"]
    # 回滚到监控指标最优的权重。
    # SaveModelCallback 在 on_train_end 已自动 load 过一次，这里显式再 load 一次：
    # 幂等、且能覆盖"回调未注册成功/内部提前返回"的情况，保证导出的一定是最优权重。
    rolled_back = False
    if es_info.get("best_name") and Path(es_info["best_file"]).exists():
        clf.load(es_info["best_name"])
        rolled_back = True
    logging.info("训练完成: 实际 %d/%d 轮%s，导出权重=%s",
                 trained_epochs, cfg["epochs"],
                 "（早停触发）" if stopped_early else "（跑满）",
                 f"最优 epoch（{es_info['monitor']}）" if rolled_back else "最后一轮")

    clf.export(model_path)
    logging.info("模型已保存: %s", model_path)
    # 最优权重临时文件内容已并入 pkl，删除避免 models 目录堆积
    if rolled_back:
        try:
            Path(es_info["best_file"]).unlink()
            logging.info("已清理最优权重临时文件: %s", es_info["best_file"])
        except OSError as exc:  # noqa: BLE001
            logging.warning("清理最优权重临时文件失败: %s", exc)

    # 保存样本清单（供 verify.py 验证 / 测试使用）
    import json
    classes = sorted(set(y.tolist()))  # 与 TSClassification vocab 顺序一致（sort=True）
    manifest = {
        "model": model_name,
        "label": cfg["label"],
        "items": cfg["items"],
        "seq_len": cfg["seq_len"],
        "classes": classes,
        # 训练时使用的 batch_tfms：推理（verify.py / predict.py）据此复现同一
        # 标准化（见 src/train/norm.py），保证训练与推理输入分布一致
        "batch_tfms": cfg.get("batch_tfms"),
        # 训练损失权重（仅作用于训练阶段）
        "loss_label_weight": cfg.get("loss_label_weight"),
        # 推理决策是否再按上面权重对概率加权（"none" 表示纯 argmax）
        "decision_weight": cfg.get("decision_weight"),
        # 早停 / 最优权重回滚的实际执行情况（便于追溯 pkl 对应哪一轮）
        "early_stopping": {
            "monitor": es_info["monitor"],
            "patience": es_info["patience"],
            "min_delta": es_info["min_delta"],
        },
        "trained_epochs": trained_epochs,
        "stopped_early": stopped_early,
        "rolled_back_to_best": rolled_back,
        # 按时间划分（训练 < 验证 < 测试，测试集时间最新）的比例 / 时间范围 /
        # 窗口清单 / 标签分布；purge_days 为边界两侧空出的交易日数（防标签前瞻泄漏）
        "split_method": "time",
        "split_ratio": list(cfg["split_ratio"]),
        "purge_days": cfg["purge_days"],
        "purge_gap_days": split_info.get("gap_days", 0),
        "purged_samples": split_info.get("purged", 0),
        "split_ranges": split_ranges,
        "split_decision_ranges": split_decision,
        "split_paths": split_paths,
        "split_stats": split_stats,
        # 兼容旧字段
        "valid_paths": split_paths["valid"],
    }
    manifest_path = cfg["models"] / f"{model_name}.valid.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logging.info("样本清单已保存: %s（train %d / valid %d / test %d）",
                 manifest_path, len(split_paths["train"]),
                 len(split_paths["valid"]), len(split_paths["test"]))

    # ---- 训练报告：配置 + 数据划分统计 + 逐 epoch 训练曲线 ----
    report = [
        f"模型: {model_name}",
        f"生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"数据源: {cfg['data_source']}",
        f"dataset: {cfg['dataset']}",
        f"特征列: {cfg['items']}",
        f"标签列: {cfg['label']}",
        f"seq_len: {cfg['seq_len']}",
        f"架构: {cfg['arch']}",
        f"batch_size: {cfg['batch_size']}   epochs: {cfg['epochs']}"
        f"   learning_rate: {cfg['learning_rate']}",
        f"batch_tfms: {cfg.get('batch_tfms')}",
        f"loss_label_weight: {cfg.get('loss_label_weight')}",
        f"decision_weight: {cfg.get('decision_weight')}"
        f"（none=推理仅用 argmax，loss=推理再乘一次上述权重）",
        f"early_stopping: {cfg.get('early_stopping')}   save_best: {cfg.get('save_best')}",
        f"随机种子: {cfg['seed']}",
        f"划分方式: 按窗口时间顺序（train < valid < test，测试集时间最新）",
        f"划分比例(train/valid/test): "
        f"{'/'.join(f'{r:g}' for r in cfg['split_ratio'])}"
        f"   purge_days: {cfg['purge_days']}（日历间隔 {split_info.get('gap_days', 0)} 天）",
        "",
        "== 数据划分（按时间顺序切分，边界 purge 防未来数据泄漏） ==",
        split_table(y, split_indices, classes, spans=spans,
                    split_info={**split_info, "purge_days": cfg["purge_days"]}),
        "",
        "== 训练曲线 ==",
    ]
    try:
        metric_names = [str(x) for x in clf.recorder.metric_names[1:]]
        values = clf.recorder.values
        report.append(f"{'epoch':<7}" + "".join(f"{n:>14}" for n in metric_names))
        for ep, row in enumerate(values, 1):
            cells = ""
            for v in row:
                try:
                    cells += f"{float(v):>14.6f}"
                except (TypeError, ValueError):
                    cells += f"{str(v):>14}"
            report.append(f"{ep:<7}{cells}")
        # ---- 早停 / 最优权重回滚摘要 ----
        # 报告必须能回答"这个 pkl 对应哪一轮"，否则无法判断导出的是否过拟合权重
        report += [
            "",
            "== 早停 / 最优权重回滚 ==",
            f"监控指标: {es_info['monitor']}"
            f"（{'越小越好' if es_info['lower_better'] else '越大越好'}）"
            f"  patience: {es_info['patience']}  min_delta: {es_info['min_delta']}",
            f"实际训练轮数: {trained_epochs}/{cfg['epochs']}"
            + ("（早停触发）" if stopped_early else "（跑满全部 epoch）"),
            "导出权重: "
            + (f"已回滚到 {es_info['monitor']} 最优 epoch" if rolled_back
               else "最后一轮（未启用 save_best 或未找到最优权重文件）"),
        ]
        if values and es_info["monitor"] in metric_names:
            mi = metric_names.index(es_info["monitor"])
            col = [float(v[mi]) for v in values]
            bi = int(np.argmin(col)) if es_info["lower_better"] else int(np.argmax(col))
            report.append(f"最优 epoch: {bi + 1}"
                          f"（{es_info['monitor']}={col[bi]:.6f}）")
        if "accuracy" in metric_names and values:
            ai = metric_names.index("accuracy")
            best = max(range(len(values)), key=lambda i: float(values[i][ai]))
            report += ["",
                       f"best valid accuracy: {float(values[best][ai]):.4f}"
                       f"（epoch {best + 1}，仅作记录；权重按 "
                       f"{es_info['monitor']} 选择）",
                       f"last valid accuracy: {float(values[-1][ai]):.4f}"]
    except Exception as exc:  # noqa: BLE001
        report.append(f"训练曲线读取失败: {exc}")
    train_report_path = cfg["models"] / f"{model_name}.train.log"
    write_report(train_report_path, report)
    logging.info("训练报告已保存: %s", train_report_path)

    # 训练后清理旧模型：仅保留同任务下最新的 KEEP_MODELS 个模型及其清单
    removed = prune_models(cfg["models"], task, KEEP_MODELS)
    if removed:
        logging.info("已清理旧模型文件 %d 个（仅保留最新 %d 个 %s-*.pkl）",
                     len(removed), KEEP_MODELS, task)
    else:
        logging.info("模型数量未超过 %d 个，无需清理", KEEP_MODELS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
