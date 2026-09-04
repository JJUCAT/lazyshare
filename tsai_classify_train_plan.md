tsai 分类任务
config/classify_train.json 时序分类任务训练的配置
"slice": { // 数据切片配置
  "seq_len": 21, // 窗口大小
  "jump_step": 21, // 一次切片后滑动多大距离
  "items": [ // 从 csv 文件读取列项参数
    "NC",
    "NV",
    "NA",
    "SNB"
  ],
  "label": [ // 需要训练识别分类的标签
    "峰值标签"
  ],
  "throw_last_one": true // 正向平铺后若末尾不足一窗，true=丢弃"以最新一天收尾"的末尾窗口只保留平铺窗口；false=补生成该窗口（覆盖预测最新一天）
},

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
5.正向平铺（从完整起点按 jump_step 滑动切完整窗口）后，若末尾剩余不足一个完整窗口，按 slice.throw_last_one 决定是否补"以文件最后一天收尾"的窗口样本：
  - true（默认）：丢弃，只保留正向平铺窗口
  - false：补生成该窗口（取最后 seq_len 行，与前面平铺窗口尾部重叠），使训练样本覆盖"预测最新一天"的情形
6.切片前先核对本地 dataset 格式（seq_len/jump_step/特征列/标签列/throw_last_one，依据 dataset 目录下 .slice_signature.json 的签名）与 classify_train.json 是否一致；不一致或无签名（旧版本产物）则整体清理旧切片后再重新切片，避免旧格式样本混入训练

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

