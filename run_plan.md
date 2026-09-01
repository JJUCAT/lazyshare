config/mail.json 邮件配置
{
  "smtp": {
    "server": "smtp.163.com",
    "auth_code": "TPxxv6LcCqUe43R2"
  },

  "mails": [ // 目标邮箱
    "lmr2887@163.com"
  ]
}

src/run.py 尽量调用现有功能接口，实现"weekday"功能：
1.使用 pull.json 拉取最新代码数据
2.使用 update.json 更新个股历史数据
3.使用 preprocess.json 更新预处理数据
4.使用 prediction.sh 估计最新股票的标签
5.将置信度大于 0.9 的股票及其标签状态，发邮件到目标邮箱
6."weekday"功能每个工作日 21:00 启动
