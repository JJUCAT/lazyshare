tsai 分类任务
config/classify_train.json 时序分类任务训练的配置
"slice": { // 数据切片配置
  "seq_len": 63, // 窗口大小
  "jump_step": 21 // 一次切片后滑动多大距离
}

"items" // 从 csv 文件读取列项参数

"label": // 需要训练识别分类的标签

"data_source": "/Users/jucat/data/ashare/preprocessed" // 数据源
"dataset": "/Users/jucat/data/ashare/dataset" // 切片后的数据集
"validation": "/Users/jucat/data/ashare/validation" // 验证集
"models": "/Users/jucat/data/ashare/models" // 训练后的模型保存路径


"train": { // 训练参数
  "arch": "InceptionTimePlus", // 模型架构
  "tfms": "TSClassification", // 任务类型
  "batch_tfms": "TSStandardize", // 归一化
  "batch_size": 32, // 单次训练的数据切片量
  "epochs": 100, // 数据循环训练次数
  "learning_rate": 0.001 // 
  "validation_set": 0.1 // 验证集占比
}

"evaluate" // 模型评分项

----------

src/train/slice.py:
1.从“items”参数完整的时间开始切片。
2.切片数据保存为 csv，必须保留"日期"参数，切片文件名是源文件名-开始日期-结束日期
3.同一个数据源的切片保存在 "dataset" 路径下的同名文件夹
4.切片后检查，如果整个文件都是同一个标签，删除该文件

src/train/train.py:
1.从 "dataset" 随机读取(1.0-"validation_set")数据按 "train" 训练参数训练
2.模型用“任务-时间-batch大小-epochs大小”命名


src/train/verify.py:
1.没有训练的那部分"validation_set"数据用来验证训练好的模型
2.验证集的 csv 文件增加"infer"列项，记录验证中分类出的标签
3.验证集的 csv 文件增加"conf"列项，记录分类出的标签置信度
4.在 "validation" 路径下创建模型名子文件夹，存放验证集文件和模型评分总结日志

src/evaluate.py:
1.输入是结构化的验证结果，根据 "evaluate"完成评分项计算，输出评分总结

