#!/usr/bin/env bash
# 后处理：按 M5C 过滤推理出的 B / T 信号并重新评估指标
#
# 用法:
#   bash scripts/tsai/postprocess.sh                       # 处理最新模型的 valid + test
#   bash scripts/tsai/postprocess.sh --split test          # 只处理测试集
#   bash scripts/tsai/postprocess.sh --model models/xxx.pkl
#   bash scripts/tsai/postprocess.sh --min-conf 0.9        # 只看高置信信号
#   bash scripts/tsai/postprocess.sh --config /tmp/x.json  # 指定后处理配置
#   bash scripts/tsai/postprocess.sh --force               # 忽略开关强制过滤
#
# 说明:
#   - 读取独立的 config/postprocess.json（data_source / validation / models /
#     price_col / ma_col / labels / min_conf），与训练配置解耦
#   - 开关在 config/preprocess.json 的 "postprocess": { "enabled": true }；
#     关闭时直接跳过，不影响推理结果与验证/测试分类指标
#   - 功能实现在 src/postprocess/postprocess.py
#   - 仅依赖 pandas / numpy / scikit-learn（无需 tsai 环境）
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "${PROJECT_ROOT}/scripts/postprocess.py" "$@"
