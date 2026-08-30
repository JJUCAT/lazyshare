在 src/prediction 目录实现功能，可以调用其他文件接口
1.在 test_output 中创建文件夹 pred-date，其中 date 是具体的现在日期
2.读取 config/classify_train.json 配置
3.从 "data_source" 遍历股票，切片最新的"seq_len"长度的数据作为 dataset 保存到 pred-date/dataset 目录
4.使用最新模型，推理分类 dataset 数据，仅分类最新一天的标签
5.分类结果只看标签 T 和 B，按置信度从高到低排列，保存到 pred-date/prediction.log 文件中