配置文件：config/update.json
{
  // 新数据的路径
  "update": "/Users/jucat/data/ashare/update"
  // 需要更新数据的路径
  "share": "/Users/jucat/data/ashare/share"
  // weather.csv 路径：每日各行业成交量、大盘成交量（单位：万股）
  "weather": "/Users/jucat/data/ashare/share/weather.csv"
}

scripts/update.py:
1.股票代码是股票的唯一标识，股票名称不是
2.遍历 "share" 下的股票，从 "update" 中获取新数据
3.检查 "update" 是否有新股票，新股票更新到 "share"
4.在 "share" 股票全部检查更新完成之后，检查，更新 "weather" 文件
  weather.csv 为宽表：行 = 日期，列 = "大盘" + 各行业成交量，单位：万股
  （成交量（股）/10000，保留 3 位小数）
  数据来源：share 下全部股票文件（"代码-名称.csv"），而非 update
  对每行按 日期 汇总：
  - 行业成交量：按 "所属行业" 分组，对 "成交量（股）" 求和
  - 大盘成交量：当天全部股票 "成交量（股）" 求和
  文件不存在则创建；已存在则从 share 全部股票重建（行业列动态扩展）