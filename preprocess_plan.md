preprocess 实现：
1.src/preprocess/handle 负责基础工作
2.src/preprocess/label 负责标签计算（峰值标签；三元屏蔽的参数配置见 label 段）
3.config/preprocess.json 配置文件：
  "raw_data"：原始数据路径
  "preprocessed_data"：预处理数据路径
  "weather"：行业成交量，大盘成交量
  "label"：三元屏蔽（止盈 / 止损 / 时限）参数，见下文“label 配置参数说明”
4.scripts/preprocess.py：启动预处理

使用一半 cpu 核心数量多线程加速处理
浮点数小数点最多保留 6 位数
日志保存到 test_output，记录预处理文件数量，耗时
preprocess.py 数据预处理，得到 csv 文件，文件名为“股票代码-股票名称”
st，*st，退市股票跳过，不生成预处理数据文件
跳过股票上市前 5 个交易日的数据（涨跌幅无限制）

列项有：{
  股票代码：
  股票名称：
  行业：
  收盘价：
  成交量：
  LU：(C_{t}-C_{t-1})/(max_limit)*C_{t-1}，涨跌幅利用率，max_limit 该股票当天的最大涨幅，例如沪深主板最大涨跌幅 10%，创业板 20%，科创板 20%，北交所 30%等。注意沪深主板的 st 股票在 2026 年 7 月 6 日之前涨跌幅是 5%，现在涨跌幅是 10%。
  M21C：21日收盘价均值
  M21V：21日成交量均值
  NC：(收盘价-M21C)/M21C，归一化值。表示价格。
  NV：(成交量-M21V)/M21V，归一化值。表示成交量。
  ATR：(max(H_{t}-L_{t},|H_{t}-C_{t-1}|,|L_{t}-C_{t-1}|))/C_{t-1}，单日真实波动率，考虑了隔夜跳空，归一化值。表示波动。
  NBull：(最高价-收盘价)/收盘价，归一化值。表示看多。
  NBear：(收盘价-最低价)/收盘价，归一化值。表示看空。
  TR：换手率
  IMV：行业成交量 / 大盘成交量。表示行业热度。
  SIV：个股成交量 / 行业成交量。表示个股在行业热度。
  峰值标签：T，B，N（列名沿用"峰值标签"，取值由三元屏蔽法给出）
}

峰值标签：使用三元屏蔽法（triple barrier），止盈标注 T，止损标注 B，其他标注无操作 N
preprocess.json 参数说明
"label": {
  "max_days": 21, // 观察天数
  "price": "开盘价", // 收益计算使用的价格，profit_{k}=(O_{k+1}-O_{k})/O_{k}
  "barrier_mode": "atr", // 屏障方式："fixed" 固定百分比 / "atr" 按波动率自适应
  "take_profit": 0.15, // 固定止盈百分比（barrier_mode="fixed" 时使用）
  "stop_loss": 0.1, // 固定止损百分比（barrier_mode="fixed" 时使用）
  "atr_window": 21, // ATR 滚动均值窗口（barrier_mode="atr" 时使用）
  "take_profit_atr": 3, // 止盈屏障 = take_profit_atr × ATR
  "stop_loss_atr": 2, // 止损屏障 = stop_loss_atr × ATR
  "min_take_profit": 0.06, // 止盈屏障下限（防止低波动股屏障过窄）
  "max_take_profit": 0.3, // 止盈屏障上限
  "min_stop_loss": 0.04, // 止损屏障下限
  "max_stop_loss": 0.2, // 止损屏障上限
  "same_day_priority": "stop_loss", // 同时触发优先标注止损
  "values": {
    "take_profit": "B", // 止盈标注
    "stop_loss": "T", // 止损标注
    "timeout": "N" // 无操作
  }
}

波动率自适应屏障（barrier_mode="atr"）
  ATR_pct[t] = 截至第 t 日的 atr_window 日滚动平均单日真实波动率（只用 t 及之前的数据）
  take_profit_t = clip(take_profit_atr × ATR_pct[t], min_take_profit, max_take_profit)
  stop_loss_t   = clip(stop_loss_atr   × ATR_pct[t], min_stop_loss,   max_stop_loss)
  高波动股票屏障自动放宽、低波动股票收窄，避免固定百分比对不同波动股票的偏倚

三元屏蔽标注实现（src/preprocess/label/triple_barrier.py）
对第 t 个交易日：
  1.入场价 = price[t]（默认开盘价）
  2.依次取 k = 1..max_days，计算 profit_k = (price[t+k] - price[t]) / price[t]
       profit_k >= take_profit → 止盈，标注 values.take_profit（B，买入）
       profit_k <= -stop_loss  → 止损，标注 values.stop_loss（T，卖出）
  3.先触发的一侧即为该日标签；max_days 内均未触发 → 标注 values.timeout（N）
  4.同一个 k 同时越过两侧屏障（跳空等极端情形）→ 按 same_day_priority 决定
  5.价格缺失或未来 max_days 个交易日数据不完整的行（含文件末尾 max_days 行）
    标签为空，预处理阶段直接丢弃，避免用不完整的前瞻窗口训练
标签参数写入 preprocessed_data/.label_signature.json；参数变化时自动清理旧
预处理文件后重新生成（--update 也生效）

实现 update 接口：
从 "raw_data" 更新新数据到 "preprocessed_data" 路径
scripts/preprocess.py 增加输入参数接口 "update" 支持更新数据


