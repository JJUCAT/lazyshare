# -*- coding: utf-8 -*-
"""推理阶段复现训练时的 batch_tfms（保证训练/推理输入分布一致）。

背景：训练时 `TSClassifier(batch_tfms=...)` 的变换注册在 DataLoader 的
after_batch 管线里，**不在 `learn.model` 内部**。因此 verify / predict 用
`learn.model(x)` 直接前向时，必须手动应用同一变换，否则模型在训练阶段
看到的是标准化后的张量，推理阶段却是原始特征（NC/NV/ATR 与 IMV/SIV
量级相差两个数量级），预测结果系统性偏移。

本模块仅实现 config 允许的两种取值：

- `"TSStandardize"`：等价 `TSStandardize(by_sample=True)`
- `null` / `""` / `"none"` / `"noop"` / `"identity"`：无操作，原样返回

等价性依据 tsai 源码：

`tsai/data/preprocessing.py` `TSStandardize.__init__`
    by_sample=True、by_var=False、by_step=False → axes=(1, 2)

`TSStandardize.encodes`
    mean = torch_nanmean(o, dim=(1, 2), keepdim=True)
    std  = torch.clamp_min(torch_nanstd(o, dim=(1, 2), keepdim=True), eps=1e-8)
    return (o - mean) / std

`tsai/utils.py` 的 `torch_nanmean` / `torch_nanstd` 在**无 NaN** 时分别退化为
`torch.mean` / `torch.std`（`unbiased=True`）。切片（`slice.py`）与推理
（`predict.py`）两侧都保证特征列全非 NaN，故此处直接使用无 NaN 分支。
"""
from __future__ import annotations

import numpy as np

# TSStandardize 默认 eps，与其源码保持一致
EPS = 1e-8

# 表示"无操作"的配置取值（与 train.build_batch_tfms 保持一致）
NONE_KEYS = {"", "none", "noop", "identity", "null"}


def tfm_key(name) -> str:
    """把配置里的 batch_tfms 归一为小写 key；None 视为无操作。"""
    if name is None:
        return ""
    return str(name).strip().lower()


def is_noop(name) -> bool:
    """batch_tfms 是否表示"无操作"。"""
    return tfm_key(name) in NONE_KEYS


def standardize_by_sample(x):
    """逐样本标准化，等价 `TSStandardize(by_sample=True)`。

    x: torch.Tensor，形状 (n_samples, n_vars, n_steps)。
    返回同形状张量：每个样本用其自身全部变量与时间步的均值/标准差归一化。
    """
    import torch  # 延迟导入，避免无 torch 环境时报错

    if not torch.is_tensor(x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float32))
    if x.ndim != 3:
        raise ValueError(f"standardize_by_sample 需要 3D 输入 (n, vars, steps)，实际 {tuple(x.shape)}")
    # by_sample=True, by_var=False, by_step=False -> axes=(1, 2)
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True, unbiased=True).clamp_min(EPS)
    return (x - mean) / std


def apply_batch_tfms(x: np.ndarray, name) -> np.ndarray:
    """对推理输入应用训练时的 batch_tfms，返回 float32 ndarray。

    x   : (n_samples, n_vars, n_steps) 原始特征
    name: 训练时配置的 batch_tfms（来自模型清单，缺省回退到配置）
    """
    arr = np.asarray(x, dtype=np.float32)
    if is_noop(name):
        return arr
    if tfm_key(name) == "tsstandardize":
        return standardize_by_sample(arr).cpu().numpy().astype(np.float32, copy=False)
    raise ValueError(
        f"不支持的 batch_tfms: {name!r}（可用 \"TSStandardize\"，或 null 表示无操作）")
