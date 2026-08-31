config/pull.json：
{
  "source": {
    "dyy": { // 金钥数据
      "url": "www.dyyboard.ltd",
      "account": "im25547",
      "password": "u86hqc3F"
    }
  },

  "spider": { // 爬虫
    "share_individual": { // A股个股
      "port": "日常更新", 
      "type": "前复权“,
      "frequency": 1 // 接口访问频率
    }
  },

  "share": "/Users/jucat/data/ashare/share", // 数据库
  "download": "/Users/jucat/data/ashare/update" // 拉取保存路径
}

scripts/pull.sh: 拉取数据的快捷脚本


src/pull/：爬虫
检查 "share" 个股的数据时间，拉取数据到 "download"
拉取数据接口访问注意频率，随机时间
