# postprocess 后处理

对 AI 推理结果做二次过滤，只保留"价格位置合理"的买卖信号，并重新评估精度。

## 目录结构

```
src/postprocess/
    __init__.py       包入口（导出 postprocess_split / score_samples）
    config.py         读取 config/classify_train.json 的 postprocess 段
    postprocess.py    价格过滤 + 指标评估 + 报告输出
scripts/postprocess.py       启动入口（python3，无需 tsai 环境）
scripts/tsai/postprocess.sh  快捷入口
```

## 过滤规则

只处理推理标签为 B / T 的样本（当天模型给出的买卖信号）：

| 推理标签 | 保留条件 | 剔除条件 |
| --- | --- | --- |
| B（买入） | 当日收盘价 < M5C | 当日收盘价 >= M5C |
| T（卖出） | 当日收盘价 > M5C | 当日收盘价 <= M5C |

- 价格与均线取自预处理目录 `data_source/<源文件名>.csv` 的 `收盘价` / `M5C` 列
  （窗口 CSV 只保留特征列，不含这两列，需按"源文件名 + 日期"回溯）。
- 价格或均线缺失（NaN / 源文件不存在）的信号单独计数，并从保留集合中剔除。
- M60C 同样在预处理中生成，可把 `postprocess.ma_col` 改成 `M60C` 换用长均线。

## 评估指标

与验证 / 测试报告同一口径（`src/train/report.py`）：

- 全类别 accuracy、混淆矩阵、每类 precision / recall / f1 / specificity；
- macro / weighted 平均；
- `roc_auc`（ovr, weighted，需要概率列）；
- 置信度分档（conf >= 0.99 / 0.95 / 0.90 / 0.85 / 0.80 的占比与精度）。

报告内容：

1. 原始 B/T 信号指标（未做价格过滤，作为对照）；
2. 价格过滤后指标 + 逐标签对比；
3. 相对全量（整个 split）的召回：过滤后正确的正例 / 全量真实正例；
4. 过滤统计：原始 B/T 数量、保留 / 剔除数量及原因、价格缺失数量。

## 配置（config/postprocess.json）

后处理配置独立于训练配置（`config/classify_train.json`），单独维护：

```json
{
  "data_source": "/Users/jucat/data/ashare/preprocessed",
  "validation": "/Users/jucat/data/ashare/validation",
  "models": "/Users/jucat/data/ashare/models",
  "price_col": "收盘价",
  "ma_col": "M5C",
  "labels": { "buy": "B", "sell": "T" },
  "min_conf": null
}
```

- `data_source`：预处理目录，读 `price_col` / `ma_col`
- `validation`：推理结果目录（`<模型名>/<split>/results.json`）
- `models`：模型目录，未传 `--model` 时取其中最新的 `.pkl`
- `min_conf` 非空时，只统计置信度 >= 该值的信号

## 开关（config/preprocess.json）

是否启用后处理过滤，由**预处理参数 JSON** 控制（缺省 true）：

```json
"postprocess": {
  "enabled": true
}
```

- `enabled: false`：训练流水线与 `scripts/postprocess.py` 都跳过过滤，
  只保留验证/测试的分类指标评测；
- 开关只影响是否写过滤报告，**不会修改** `results.json` 或验证 CSV 的
  `infer` / `conf` / `P_*` 列，因此不影响后续步骤（verify / evaluate）的分类指标；
- 需要临时忽略开关强制执行：`python3 scripts/postprocess.py --force`。

## 输出

```
validation/<模型名>/<split>/postprocess.log    后处理报告
validation/<模型名>/<split>/postprocess.json   结构化统计
models/<模型名>.<split>.postprocess.log        模型目录副本
```

## 用法

```bash
# 最新模型的 valid + test（默认读 config/postprocess.json）
python3 scripts/postprocess.py

# 指定配置 / 模型 / 数据集 / 只看高置信信号
python3 scripts/postprocess.py --config config/postprocess.json \
    --model models/<模型名>.pkl --split test --min-conf 0.9

# 忽略 config/preprocess.json 的开关，强制执行
python3 scripts/postprocess.py --force

# 流水线内自动执行（训练 → 验证/测试 → 后处理 → 邮件）
bash scripts/tsai/train_pipeline.sh
```

## 与切片的关系

`M5C` / `M60C` 是后处理专用列，不进入模型特征：

- `slice.items` 中不允许出现 `M5C` / `M60C`（`src/train/slice.py` 会直接报错）；
- 切片窗口 CSV 只保存 `日期 + 特征列 + 峰值标签`，postprocess 需要时从预处理源文件读取价格。
