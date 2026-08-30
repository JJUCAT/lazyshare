配置文件：config/update.json
{
  // 新数据的路径
  "update": "/Users/jucat/data/ashare/update"
  // 需要更新数据的路径
  "share": "/Users/jucat/data/ashare/share"
}

scripts/update.py:
1.股票代码是股票的唯一标识，股票名称不是
2.遍历 "share" 下的股票，从 "update" 中获取新数据
3.检查 "update" 是否有新股票，新股票更新到 "share"
