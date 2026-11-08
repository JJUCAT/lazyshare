preprocess 实现：
1.src/preprocess/handle 负责基础工作
2.src/preprocess/label 负责标签计算（峰值标签；三元屏蔽的参数配置见 label 段）
3.config/preprocess.json 配置文件：
  "raw_data"：原始数据路径
  "preprocessed_data"：预处理数据路径
  "weather"：行业成交量，大盘成交量
  "label"：三元屏蔽（止盈 / 止损 / 时限）参数，见下文“label 配置参数说明”
（后处理开关已迁移到 config/postprocess.json 的 "enabled"，见 postprocess_plan.md）
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
  TR：换手率
  IMV：行业成交量 / 大盘成交量。表示行业热度。
  SIV：个股成交量 / 行业成交量。表示个股在行业热度。
  峰值标签：T，B，N（列名沿用"峰值标签"，取值由三元屏蔽法给出）
}

峰值标签：使用简化的三元屏蔽法（triple barrier），止盈标注 B（买入），止损标注 T（卖出），其他标注无操作 N
preprocess.json 参数说明
"label": {
  “price” // 盈亏计算价格
  "take_profit" // 盈利百分比，单位%
  "take_profit_timeout" // 盈利时限
  "stop_loss" // 亏损百分比，单位%
  "stop_loss_timeout" // 亏损时限
  "values": {
    "take_profit": // 止盈标注
    "stop_loss": // 止损标注
    "timeout": // 无操作
  }
}

三元屏蔽标注实现（src/preprocess/label/triple_barrier.py）
用 price(open | close) 计算盈亏，如果使用 close，profit = (C_{k+t}-C_{k})/C_{k}。如果使用 open，profit = (O_{k+t}-O_{k})/O_{k}。如果在盈利时限内能达到盈利，第 k 条时序标注为 B。如果在止损时限内达到亏损值，第 k 条时序标注为 T。其他情况标注为 N。标签优先级 B > T > N。
标签参数写入 preprocessed_data/.label_signature.json；参数变化时自动清理旧
预处理文件后重新生成（--update 也生效）

实现 update 接口：
从 "raw_data" 更新新数据到 "preprocessed_data" 路径
scripts/preprocess.py 增加输入参数接口 "update" 支持更新数据


