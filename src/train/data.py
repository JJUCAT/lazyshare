# -*- coding: utf-8 -*-
"""训练/验证公共模块：配置加载、窗口样本加载、数据集划分。

切片后的 dataset 目录结构：
    dataset/<源文件名>/<源文件名>-开始日期-结束日期.csv
每个窗口 csv 为 seq_len 行，含列：日期 + 特征列 + 标签列。

提供：
    - load_config()          读取 config/classify_train.json 完整配置
    - collect_samples()      从 dataset 读取所有窗口样本，组装 X/y/时间跨度
    - window_last_label()    取窗口最后一天的标签（严格 T/B/N，与逐日推理语义一致）
    - time_split()           按窗口时间顺序划分 训练/验证/测试（严格 train<valid<test）
    - stratified_split()     按末日标签分层划分（旧接口，仅保留兼容）
    - train_valid_split()    按 seed 随机划分训练/验证集（旧接口，仅保留兼容）

数据划分的核心约束（防止未来数据泄漏）：
    1. 切片窗口允许重叠（见 slice.py：jump_step 可小于 seq_len），样本量约放大
       seq_len/jump_step 倍；重叠只发生在同一集合内部，跨集合的标签泄漏由第 3
       条的 purge 间隔控制。代价是相邻窗口的特征区间重叠，因此验证 / 测试
       指标可能偏乐观，解读时需注意；
    2. 三集合按窗口末日（决策日）切成连续的三段，训练最早、测试最新；
       由于窗口跨 seq_len 天，特征区间可能跨集合交叠，真正严格有序的是决策日；
    3. 三元屏蔽标签会向前看 label.max_days 个交易日，因此相邻集合之间再留出
       purge 间隔（train.purge_days，默认 21 个交易日 ≈ 31 个日历日），保证
       训练集标签的前瞻窗口不会跨进验证集、验证集不会跨进测试集。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "classify_train.json"

DATE_COL = "日期"
# 空标签归一为 None 类，保证分类任务类别完整
EMPTY_LABEL = "None"


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    """读取 config/classify_train.json，返回完整配置 dict。"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_source = Path(cfg.get("data_source", "")).expanduser()
    dataset = Path(cfg.get("dataset", "")).expanduser()
    validation = Path(cfg.get("validation", "")).expanduser()
    models = Path(cfg.get("models", "")).expanduser()
    if not data_source or not data_source.exists():
        raise FileNotFoundError(f"data_source 目录不存在: {data_source}")
    if not dataset.exists():
        raise FileNotFoundError(f"dataset 目录不存在: {dataset}")
    if not validation:
        raise ValueError("validation 未配置")
    if not models:
        raise ValueError("models 未配置")

    slice_cfg = cfg.get("slice") or {}
    seq_len = int(slice_cfg.get("seq_len", 63))
    # items / label 已整合进 slice；兼容旧格式（顶层 items / label）
    items = slice_cfg.get("items") or cfg.get("items") or []
    label = (slice_cfg.get("label") or cfg.get("label") or ["峰值标签"])[0]
    if not items:
        raise ValueError("items 未配置")

    train_cfg = cfg.get("train") or {}
    test_cfg = train_cfg.get("test") or {}
    # train_max：随机取 train_max 个 dataset 样本作为总训练源（测试用）；<0 表示用完整数据
    train_max = int(test_cfg.get("train_max", -1))
    # 损失函数标签权重：{标签: 权重}；None 表示不加权（默认 CrossEntropyLoss）
    loss_label_weight = train_cfg.get("loss_label_weight") or None
    # 推理决策是否也按 loss_label_weight 对概率加权后再取类别。
    # 默认 "none"（只用 argmax）：训练已用加权损失把偏置学进模型，推理再乘一次
    # 等于同一权重生效两次（如 B=10 时会把大量 N/T 强行判成 B）。
    # 仅在显式配置 "loss" 时沿用旧的二次加权行为。
    decision_weight = str(train_cfg.get("decision_weight", "none")).strip().lower()
    if decision_weight not in ("none", "loss"):
        raise ValueError(f'train.decision_weight 只支持 "none" / "loss"，当前 {decision_weight!r}')

    # 数据集划分比例 (train, valid, test)，默认 6:2:2。
    # 支持两种写法：{"train":0.6,"valid":0.2,"test":0.2} 或 [0.6,0.2,0.2]；
    # 缺省时回退到旧配置 validation_set（(1-v, v, 0)）。
    split_cfg = train_cfg.get("split")
    if isinstance(split_cfg, (list, tuple)) and len(split_cfg) == 3:
        split_ratio = tuple(float(x) for x in split_cfg)
    elif isinstance(split_cfg, dict) and split_cfg:
        split_ratio = (
            float(split_cfg.get("train", 0.6)),
            float(split_cfg.get("valid", 0.2)),
            float(split_cfg.get("test", 0.2)),
        )
    else:
        _v = float(train_cfg.get("validation_set", 0.1))
        split_ratio = (1.0 - _v, _v, 0.0)
    if min(split_ratio) < 0 or abs(sum(split_ratio) - 1.0) > 1e-6:
        raise ValueError(f"train.split 比例必须非负且和为 1，当前 {split_ratio}")

    # 早停 + 最优权重回滚（train.early_stopping）：
    #   monitor   : 监控指标名，必须是 fastai recorder 里的指标（valid_loss / accuracy）
    #   patience  : 连续多少轮无改善就提前终止；0 表示不早停（只做最优权重回滚）
    #   min_delta : 视为"改善"的最小幅度，过滤平台期的数值抖动
    # save_best：是否保存并回滚到监控指标最优的权重（独立于早停开关）
    es_cfg = train_cfg.get("early_stopping") or {}
    es_monitor = str(es_cfg.get("monitor") or "valid_loss").strip() or "valid_loss"
    es_patience = int(es_cfg.get("patience", 5))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    if es_patience < 0:
        raise ValueError(
            f"train.early_stopping.patience 必须 >= 0（0 表示不早停），当前 {es_patience}")
    if es_min_delta < 0:
        raise ValueError(
            f"train.early_stopping.min_delta 必须 >= 0，当前 {es_min_delta}")
    save_best = bool(train_cfg.get("save_best", True))

    # purge 间隔：按时间切分训练/验证/测试时，边界两侧要空出的交易日数。
    # 三元屏蔽标签会向前看 max(止盈时限, 止损时限) 个交易日（config/preprocess.json
    # 的 label.take_profit_timeout / stop_loss_timeout），若不留间隔，紧邻边界
    # 的训练样本标签会用到验证集（乃至测试集）区间的价格，造成未来数据泄漏。
    # 默认 21（大于当前标签前瞻 10 个交易日，留有余量）。
    purge_days = int(train_cfg.get("purge_days", 21))
    if purge_days < 0:
        raise ValueError(f"train.purge_days 必须 >= 0，当前 {purge_days}")

    # 学习率调度（train.scheduler）：
    #   flat_cos  : 从 lr 单调余弦衰减（默认）。one_cycle 的前 30% 是 warmup，
    #               小信号数据上最佳点常落在 warmup 段（LR 还没升到峰值），
    #               导致"欠训练 + 已开始过拟合"，故默认改用 flat_cos。
    #   one_cycle : 含 warmup 的 one-cycle（pct_start 控制 warmup 占比）
    #   cosine    : fastai fit() 的余弦退火，起点为 lr
    scheduler = str(train_cfg.get("scheduler") or "flat_cos").strip().lower()
    if scheduler not in ("flat_cos", "one_cycle", "cosine"):
        raise ValueError(
            f'train.scheduler 只支持 "flat_cos" / "one_cycle" / "cosine"，'
            f'当前 {scheduler!r}')
    pct_start = float(train_cfg.get("pct_start", 0.3))
    if not 0.0 < pct_start < 1.0:
        raise ValueError(f"train.pct_start 必须在 (0, 1) 之间，当前 {pct_start}")

    # 随机种子：显式配置整数则固定可复现；缺省 / null / "auto" / "random"
    # 用当前时间生成随机种子，使每次训练抽样的训练集不同
    seed_cfg = train_cfg.get("seed", "auto")
    if seed_cfg is None or str(seed_cfg).strip().lower() in ("", "auto", "random"):
        seed = int(time.time_ns() % (2**31))
    else:
        seed = int(seed_cfg)

    # evaluate：新格式 {"labels": [...], "items": [...], "confidence_interval": [...]}；
    # 旧格式为评分项列表
    evaluate_cfg = cfg.get("evaluate") or {}
    if isinstance(evaluate_cfg, list):
        evaluate_items = [str(x) for x in evaluate_cfg]
        evaluate_labels: list[str] = []
        confidence_intervals: list[float] = []
    else:
        evaluate_items = [str(x) for x in (evaluate_cfg.get("items") or [])]
        evaluate_labels = [str(x) for x in (evaluate_cfg.get("labels") or [])]
        # 不同置信度阈值：按分档统计 evaluate.items 指标与逐标签指标
        confidence_intervals = [
            float(x) for x in (evaluate_cfg.get("confidence_interval") or [])
        ]

    parsed = {
        "data_source": data_source,
        "dataset": dataset,
        "validation": validation,
        "models": models,
        "items": [c for c in items if c != DATE_COL],  # 特征列排除日期
        "label": label,
        "seq_len": seq_len,
        "arch": train_cfg.get("arch", "InceptionTimePlus"),
        "tfms": train_cfg.get("tfms", "TSClassification"),
        "batch_tfms": train_cfg.get("batch_tfms", "TSStandardize"),
        "batch_size": int(train_cfg.get("batch_size", 32)),
        "epochs": int(train_cfg.get("epochs", 100)),
        "learning_rate": float(train_cfg.get("learning_rate", 0.001)),
        "scheduler": scheduler,
        "pct_start": pct_start,
        "validation_set": float(train_cfg.get("validation_set", 0.1)),
        "split_ratio": split_ratio,
        "seed": seed,
        "train_max": train_max,
        "loss_label_weight": loss_label_weight,
        "decision_weight": decision_weight,
        # 早停 / 最优权重回滚
        "early_stopping": {
            "monitor": es_monitor,
            "patience": es_patience,
            "min_delta": es_min_delta,
        },
        "purge_days": purge_days,
        "save_best": save_best,
        "evaluate": evaluate_items,
        "evaluate_labels": evaluate_labels,
        "confidence_intervals": confidence_intervals,
    }
    return parsed


def window_last_label(df, label_col: str, seq_len: int) -> str:
    """取窗口最后一天（第 seq_len 行）的标签，严格用原值（T / B / N）。

    与逐日推理语义一致：用前 seq_len 天（含当天）的特征预测最后一天的类别，
    因此训练标签也取最后一天的峰值标签，而不是"窗口内最后一个非空标签"。
    值为空 / NaN 时归为 None 类。
    """
    raw = df[label_col].iloc[seq_len - 1]
    s = "" if pd.isna(raw) else str(raw).strip()
    return s if s else EMPTY_LABEL


def window_label(label_values) -> str:
    """（兼容旧逻辑）从窗口的标签列取最后一个非空标签；全空则为 None 类。"""
    last = None
    for v in label_values:
        s = "" if pd.isna(v) else str(v).strip()
        if s:
            last = s
    return last if last else EMPTY_LABEL


def _collect_paths(dataset_dir: Path, max_dirs: int | None = None) -> list[Path]:
    """收集 dataset 下所有窗口 csv 路径（按文件夹排序）。"""
    dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if max_dirs is not None:
        dirs = dirs[:max_dirs]
    paths = []
    for d in dirs:
        paths.extend(sorted(d.glob("*.csv")))
    return paths


def collect_samples(dataset_dir: Path, items: list[str], label: str,
                    seq_len: int, max_samples: int | None = None,
                    max_dirs: int | None = None,
                    train_max: int | None = None, seed: int = 42,
                    ) -> tuple[np.ndarray, np.ndarray, list[Path], np.ndarray]:
    """从 dataset 读取窗口样本。

    - train_max >= 0：随机抽取 train_max 个样本作为总训练源（测试用）
    - train_max < 0  或 None：使用 dataset 完整数据
    - max_samples / max_dirs：调试用上限

    返回 (X, y, paths, spans)：
        X    : float32 数组 (n_samples, n_vars, seq_len)
        y    : 字符串标签数组 (n_samples,)
        paths: 每个样本对应的窗口 csv 路径
        spans: datetime64[D] 数组 (n_samples, 2)，每行 = (窗口首日, 窗口末日)，
               供 time_split() 按时间顺序划分数据集（日期无法解析的样本被跳过）
    """
    all_paths = _collect_paths(dataset_dir, max_dirs=max_dirs)
    if not all_paths:
        raise ValueError(f"dataset 目录下未找到任何窗口 csv: {dataset_dir}")

    # 随机抽取 train_max 个（先收集全量路径再按 seed 抽样）
    if train_max is not None and train_max >= 0:
        rng = np.random.default_rng(seed)
        if train_max < len(all_paths):
            all_paths = list(rng.choice(all_paths, size=train_max, replace=False))
        # train_max 大于等于样本总数时直接用全部
        logging.info("train_max=%d：随机抽取 %d 个样本作为总训练源", train_max, len(all_paths))

    X, y, paths, spans = [], [], [], []
    n = 0
    bad_date = 0
    for csv_path in all_paths:
        if max_samples is not None and n >= max_samples:
            break
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            logging.warning("读取失败 %s: %s", csv_path, exc)
            continue
        if df.empty or len(df) < seq_len:
            continue
        # 窗口时间跨度：首日 / 末日（用于按时间顺序划分数据集）
        dates = pd.to_datetime(df[DATE_COL].head(seq_len), errors="coerce")
        start, end = dates.iloc[0], dates.iloc[-1]
        if pd.isna(start) or pd.isna(end):
            bad_date += 1
            continue
        x = df[items].head(seq_len).to_numpy(dtype=np.float32).T  # (vars, seq_len)
        # 训练标签严格取窗口最后一天（第 seq_len 行）的峰值标签
        y.append(window_last_label(df, label, seq_len))
        X.append(x)
        paths.append(csv_path)
        spans.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        n += 1
        # 逐文件读取在大 dataset（数十万~数百万窗口）下耗时很长，
        # 每 10 万个样本输出一次进度，便于判断是否卡住
        if n % 100000 == 0:
            logging.info("已读取 %d/%d 个窗口样本（%.1f%%）",
                         n, len(all_paths), 100.0 * n / max(len(all_paths), 1))

    if bad_date:
        logging.warning("跳过 %d 个窗口样本：日期无法解析（无法参与时间划分）", bad_date)
    if n == 0:
        raise ValueError(f"dataset 目录下未读取到任何样本: {dataset_dir}")
    span_arr = np.array(spans, dtype="datetime64[D]")
    return np.stack(X), np.array(y), paths, span_arr


def span_range(spans, indices) -> tuple[str, str]:
    """返回给定样本索引集合覆盖的时间范围（最早窗口首日 ~ 最晚窗口末日）。"""
    spans = np.asarray(spans, dtype="datetime64[D]")
    idx = np.asarray(list(indices), dtype=int)
    if len(idx) == 0:
        return "-", "-"
    return (str(spans[idx, 0].min()), str(spans[idx, 1].max()))


def decision_range(spans, indices) -> tuple[str, str]:
    """返回给定样本索引集合的决策日范围（最早 ~ 最晚窗口末日）。

    窗口末日是标签锚点，也是 time_split 排序与 purge 的依据；长停牌股票的
    42 个交易日可能跨很久的日历时间，因此"特征首日"区间会互相重叠，
    真正严格有序的是这里返回的决策日区间。
    """
    spans = np.asarray(spans, dtype="datetime64[D]")
    idx = np.asarray(list(indices), dtype=int)
    if len(idx) == 0:
        return "-", "-"
    return (str(spans[idx, 1].min()), str(spans[idx, 1].max()))


def time_split(spans, ratios=(0.6, 0.2, 0.2), purge_days: int = 21):
    """按窗口时间顺序划分训练 / 验证 / 测试集（严格 train < valid < test）。

    与 stratified_split（随机分层）的区别：本函数只在时间轴上切三刀，
    测试集永远是时间最新的一段，训练集是最早的一段。这样验证/测试指标
    反映的是"用历史预测未来"，而不是"用未来预测过去 + 同一时期随机抽检"。

    spans      : (n, 2) datetime64 数组，每行 = (窗口首日, 窗口末日)
    ratios     : (train, valid, test) 样本占比，之和为 1
    purge_days : 相邻集合之间要空出的交易日数（= 三元屏蔽标签的前瞻天数）。
                 交易日到日历日按 7/5 保守换算并额外 +1 天。
                 判据是"窗口末日"（= 决策日、标签锚点）：只丢弃那些
                 标签前瞻窗口（末日 + purge_days 个交易日）会越过下一个集合
                 最早决策日的样本；其余样本一律保留，不做无谓裁剪。

    返回 (train_idx, valid_idx, test_idx, info)：
        info = {"purged": 边界处丢弃的样本数, "gap_days": 实际间隔日历天数,
                "train_span"/"valid_span"/"test_span": (首日, 末日)}
    并做严格校验：任一集合为空、或"集合内最晚窗口末日 + purge"不早于下一个
    集合的最早窗口末日（即标签前瞻跨集合）都会抛错。
    """
    spans = np.asarray(spans, dtype="datetime64[D]")
    if spans.ndim != 2 or spans.shape[1] != 2:
        raise ValueError(f"spans 需为 (n, 2) 数组，实际 {spans.shape}")
    n = len(spans)
    if n == 0:
        raise ValueError("time_split: 样本为空")
    if min(ratios) < 0 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"划分比例必须非负且和为 1，当前 {ratios}")

    starts, ends = spans[:, 0], spans[:, 1]
    # 按窗口末日（决策日）排序 → 时间上连续切三段。
    # 窗口允许重叠（见 slice.py），因此相邻样本的决策日可能只差 jump_step 天，
    # 跨集合的标签前瞻泄漏靠下面的 purge 间隔裁掉。
    order = np.argsort(ends, kind="stable")
    n_tr = int(round(n * float(ratios[0])))
    n_va = int(round(n * float(ratios[1])))
    n_tr = min(max(n_tr, 0), n)
    n_va = min(max(n_va, 0), n - n_tr)
    tr = order[:n_tr]
    va = order[n_tr:n_tr + n_va]
    te = order[n_tr + n_va:]

    info = {"purged": 0, "gap_days": 0}
    # 交易日 → 日历日：每周 5 个交易日 ≈ 7 个日历日，再留 1 天余量
    gap = (np.timedelta64(int(np.ceil(purge_days * 7 / 5)) + 1, "D")
           if purge_days > 0 else np.timedelta64(0, "D"))
    info["gap_days"] = int(gap.astype("timedelta64[D]").astype(int))
    if purge_days > 0:
        # 只裁掉"标签前瞻会越过下一集合最早决策日"的样本（其余保留，避免无谓丢数据）
        if len(tr) and len(va):
            keep = (ends[tr] + gap) < ends[va].min()
            info["purged"] += int((~keep).sum())
            tr = tr[keep]
        if len(va) and len(te):
            keep = (ends[va] + gap) < ends[te].min()
            info["purged"] += int((~keep).sum())
            va = va[keep]

    if not len(tr) or not len(va) or not len(te):
        raise ValueError(
            f"time_split: 划分后存在空集合（train {len(tr)} / valid {len(va)} / "
            f"test {len(te)}），请增大数据量或调整 proportions / purge_days")
    # 严格校验：上一集合的"末日 + purge"必须早于下一集合的最早末日（标签锚点），
    # 否则训练标签会用到下一集合期间的行情（未来数据泄漏）
    for a, b, name_a, name_b in ((tr, va, "train", "valid"),
                                 (va, te, "valid", "test")):
        horizon_end, first_b = ends[a].max() + gap, ends[b].min()
        if not horizon_end < first_b:
            raise ValueError(
                f"time_split: {name_a} 标签前瞻末 {horizon_end} 不早于 "
                f"{name_b} 最早决策日 {first_b}，时间顺序被破坏（未来数据泄漏）")
        if ends[a].max() >= starts[b].min():
            logging.debug("time_split: %s 决策日 %s 与 %s 最早窗口首日 %s 有特征"
                          "区间重叠（不同股票、或同一股票 jump_step 重叠的相邻"
                          "窗口，属正常现象；标签前瞻已由 purge 间隔隔离）",
                          name_a, ends[a].max(), name_b, starts[b].min())

    info["train_span"] = span_range(spans, tr)
    info["valid_span"] = span_range(spans, va)
    info["test_span"] = span_range(spans, te)
    return np.sort(tr), np.sort(va), np.sort(te), info


def stratified_split(y, ratios=(0.6, 0.2, 0.2), seed: int = 42):
    """按标签分层划分训练 / 验证 / 测试集（旧接口，当前流程改用 time_split）。

    分类任务要求三个集合的类别分布一致，而窗口是按股票时间顺序切出来的，
    直接整体随机划分会让小类（如 B）在某些集合中缺失或比例漂移。因此在
    每个类别内部独立按 ratios 切分，再把各类结果合并。

    y      : 样本标签数组（如 ['B','N','T',...]）
    ratios : (train, valid, test) 比例，之和为 1
    seed   : 随机种子

    返回 (train_idx, valid_idx, test_idx)，均为升序整数数组，三者互斥且并集为全集。
    取整误差由 test 段吸收，因此每个类别在三个集合中的占比与全集一致
    （差异仅来自整数取整，单类别样本量小时最多相差 1 个样本）。
    """
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        raise ValueError("stratified_split: 样本为空")
    r_tr, r_va = float(ratios[0]), float(ratios[1])
    rng = np.random.default_rng(seed)
    tr_idx: list[int] = []
    va_idx: list[int] = []
    te_idx: list[int] = []
    for cls in sorted(set(y.tolist())):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)  # 类别内打乱，保证随机性
        m = len(cls_idx)
        n_tr = int(round(m * r_tr))
        n_va = int(round(m * r_va))
        # 防止四舍五入导致超出，剩余全部归入 test
        n_tr = min(n_tr, m)
        n_va = min(n_va, m - n_tr)
        tr_idx.extend(cls_idx[:n_tr].tolist())
        va_idx.extend(cls_idx[n_tr:n_tr + n_va].tolist())
        te_idx.extend(cls_idx[n_tr + n_va:].tolist())
    return (np.sort(np.asarray(tr_idx, dtype=int)),
            np.sort(np.asarray(va_idx, dtype=int)),
            np.sort(np.asarray(te_idx, dtype=int)))


def label_counts(y, indices) -> dict[str, int]:
    """统计给定索引集合上的标签分布（按标签名排序）。"""
    y = np.asarray(y)
    sub = y[np.asarray(indices, dtype=int)] if len(indices) else np.asarray([])
    return {c: int((sub == c).sum()) for c in sorted(set(y.tolist()))}


def train_valid_split(n_samples: int, valid_ratio: float, seed: int):
    """按 seed 随机划分训练/验证索引（旧接口，保留兼容）。

    新流程请使用 stratified_split()，它额外产出测试集并保证类别比例一致。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    n_valid = int(round(n_samples * valid_ratio))
    valid_idx = np.sort(idx[:n_valid])
    train_idx = np.sort(idx[n_valid:])
    return train_idx, valid_idx
