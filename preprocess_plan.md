preprocess 实现：
1.src/preprocess/handle 负责基础工作
2.src/preprocess/label 负责标签计算
3.config/preprocess.json 配置文件：
  "raw_data"：原始数据路径
  "preprocessed_data"：预处理数据路径
  "weather"：行业成交量，大盘成交量
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
  峰值标签：T，B，N
}

峰值标签：
T：Top 表示收盘价在一段时间内是高峰值。
  用滑动窗口 k=21，在 M21C 数据中找出最高值那天 pt_day。
  从 pt_day 开始往前找 k 天（包括 pt_day 当天），收盘价最高那天是 ct_day。
  ct_day 前后 k 天都是最高值的话，ct_day 及其前 4 天都标记为 T，共 5 天。
B：Bottom 表示收盘价在一段时间内是低谷值。
  用滑动窗口 k，在 M21C 数据中找出最低值那天 pb_day。
  从 pb_day 开始往前找 k 天（包括 pb_day 当天），收盘价最低那天是 cb_day。
  cb_day 前后 k 天都是最低值的话，cb_day 及其后 4 天都标记为 B，共 5 天。
N：除了 T 和 B 之外的都是 None。

实现 update 接口：
从 "raw_data" 更新新数据到 "preprocessed_data" 路径
scripts/preprocess.py 增加输入参数接口 "update" 支持更新数据


