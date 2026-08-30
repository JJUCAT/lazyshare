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
    load_config,
    train_valid_split,
)


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
    X, y, paths = collect_samples(
        cfg["dataset"], cfg["items"], cfg["label"], cfg["seq_len"],
        max_samples=args.limit, max_dirs=args.dirs,
        train_max=cfg["train_max"], seed=cfg["seed"],
    )
    n = len(X)
    logging.info("共 %d 个样本，X 形状 %s，特征列 %s",
                 n, X.shape, cfg["items"])
    logging.info("标签分布: %s",
                 {k: int((y == k).sum()) for k in sorted(set(y.tolist()))})

    # 按 seed 随机划分：训练集占 1-validation_set，验证集占 validation_set
    train_idx, valid_idx = train_valid_split(n, cfg["validation_set"], cfg["seed"])
    logging.info("划分: 训练 %d，验证 %d", len(train_idx), len(valid_idx))

    clf = build_ts_classifier(X, y, train_idx, valid_idx, cfg)
    logging.info("开始训练: arch=%s, batch_size=%d, epochs=%d, lr=%s",
                 cfg["arch"], cfg["batch_size"], cfg["epochs"], cfg["learning_rate"])
    clf.fit_one_cycle(cfg["epochs"], cfg["learning_rate"])
    logging.info("训练完成")

    # 模型用"任务-时间-batch大小-epochs大小"命名
    task = f"{cfg['label']}_{cfg['arch']}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{task}-{timestamp}-bs{cfg['batch_size']}-ep{cfg['epochs']}"
    cfg["models"].mkdir(parents=True, exist_ok=True)
    model_path = cfg["models"] / f"{model_name}.pkl"
    clf.export(model_path)
    logging.info("模型已保存: %s", model_path)

    # 保存验证集样本清单（供 verify.py 使用）
    import json
    valid_paths = [str(paths[i]) for i in valid_idx]
    classes = sorted(set(y.tolist()))  # 与 TSClassification vocab 顺序一致（sort=True）
    valid_manifest = {
        "model": model_name,
        "label": cfg["label"],
        "items": cfg["items"],
        "seq_len": cfg["seq_len"],
        "classes": classes,
        # 决策边界按标签权重调整：推理（verify.py）据此对 probas 加权后再取类别
        "loss_label_weight": cfg.get("loss_label_weight"),
        "valid_paths": valid_paths,
    }
    manifest_path = cfg["models"] / f"{model_name}.valid.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(valid_manifest, f, ensure_ascii=False, indent=2)
    logging.info("验证集清单已保存: %s（%d 条）", manifest_path, len(valid_paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
