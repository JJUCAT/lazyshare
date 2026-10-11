在 src/prediction 目录实现功能，可以调用其他文件接口
1.在 test_output 中创建文件夹 pred-date，其中 date 是具体的现在日期
2.读取 config/classify_train.json 配置
3.从 "data_source" 遍历股票，切片最新的"seq_len"长度的数据作为 dataset 保存到 pred-date/dataset 目录
4.使用最新模型，推理分类 dataset 数据，仅分类最新一天的标签
5.分类结果只看标签 T 和 B，剔除退市，st 股，分别列出置信度前 10 的 T 标签股票和 B 标签股票，及其置信度，按置信度从高到低排列，保存到 pred-date/prediction.log 文件中
6.按照 config/classify_train.json 配置统计大盘，热门行业，热门票的情况，记录到 prediction.log 文件中。
  "weather": { // 大盘交易行情概况
    "global": {
      "days": // 大盘最近成交量观察天数，计算均值，判断当前成交量是否大于均值
    },
    "industrys": {
      "days": 15, // 行业最近成交量观察天数，计算均值，判断当前成交量是否大于均值
      "tops": 3 // 列出成交量大于均值，成交量最高前几位作为热门行业
    },
    "hot_industrys_shares": { // 热门行业热门票
      "days": 5, // 成交量观察天数，计算均值，判断当前成交量是否大于均值
      "tops": 3 // 列出成交量大于均值，成交量最高前几位
    }
  }
7.读取 config/watch.json，推理分类后把 watch 名单的股票标签，置信度记录到 prediction.log 文件中。