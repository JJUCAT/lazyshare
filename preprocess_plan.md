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
  LU：(C_{t}-C_{t-1})/(max_limit * C_{t-1})，涨跌幅利用率，max_limit 该股票当天的最大涨幅，例如沪深主板最大涨跌幅 10%，创业板 20%，科创板 20%，北交所 30%等。注意沪深主板的 st 股票在 2026 年 7 月 6 日之前涨跌幅是 5%，现在涨跌幅是 10%。
  OLU：(O_{t}-C_{t-1})/(max_limit * C_{t-1})，隔夜消息利用率
  HLU：(C_{t}-H_{t})/(max_limit * C_{t-1})，看空利用率
  LLU：(C_{t}-L_{t})/(max_limit * C_{t-1})，看多利用率
  M5C：5日收盘价均值
  M21C：21日收盘价均值
  M60C：60日收盘价均值
  M21V：21日成交量均值
  NC：(收盘价-M21C)/M21C，归一化值。表示价格。
  NV：(成交量-M21V)/M21V，归一化值。表示成交量。
  TR：换手率
  IMV：行业成交量 / 大盘成交量。表示行业热度。
  SIV：个股成交量 / 行业成交量。表示个股在行业热度。
  峰值标签：T，B，N（列名沿用"峰值标签"，取值由三元屏蔽法给出）
}

峰值标签：使用简化的三元屏蔽法（triple barrier），止盈标注 B（买入），止损标注 T（卖出），其他标注无操作 N
preprocess.json 参数说明
"label": {
  "take_profit" // 盈利：LU 累计值阈值
  "take_profit_timeout" // 盈利时限
  "stop_loss" // 亏损：LU 累计值阈值（正负号等价）
  "stop_loss_timeout" // 亏损时限
  "values": {
    "take_profit": // 止盈标注
    "stop_loss": // 止损标注
    "timeout": // 无操作
  }
}

三元屏蔽标注实现（src/preprocess/label/triple_barrier.py）
盈亏用涨跌幅利用率 LU 累加衡量（不用价格收益率，天然消除板块涨跌幅限制差异）：
  对第 t 个交易日（入场 = 第 t 日收盘价），
  1.profit_k = Σ LU[t+1..t+k]（k = 1..对应时限）
  2.止盈：在 k = 1..take_profit_timeout（止盈时限，默认 21 日）内，
       profit_k >= take_profit → 止盈，标注 values.take_profit（B，买入）
  3.止损：在 k = 1..stop_loss_timeout（止损时限，默认 15 日）内，
       profit_k <= -|stop_loss| → 止损，标注 values.stop_loss（T，卖出）
     （stop_loss 配置写 -0.2 或 0.2 等价，均表示亏损 0.2 个 LU）
  4.止盈 / 止损各自独立窗口，先触发的一侧生效；同一 k 同时触发时优先止损（保守）
  5.其他标注为 N
  6.末尾 max(take_profit_timeout, stop_loss_timeout) 个交易日的前瞻窗口不完整，
    标签为空并丢弃；--update 比对时也不计入这些日期
标签参数写入 preprocessed_data/.label_signature.json；参数变化时自动清理旧
预处理文件后重新生成（--update 也生效）

实现 update 接口：
从 "raw_data" 更新新数据到 "preprocessed_data" 路径
scripts/preprocess.py 增加输入参数接口 "update" 支持更新数据


