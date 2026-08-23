# LazyShare

A 股日线数据处理与可视化工具。

- `scripts/preprocess.py`：数据预处理脚本（多进程并行），把原始日线数据转换为
  预处理的 CSV（见 `preprocess_plan.md`）。ST 股票跳过、不生成文件，并输出
  “峰值标签”（T/B）列。
- `src/gui`：基于 PySide6 的可视化 GUI（见 `gui_plan.md`），分为 `business`
  （后端业务逻辑）与 `ui`（前端显示）两层。

## 环境依赖

```bash
python3 -m pip install -r requirements.txt
```

依赖：`PySide6`、`pandas`、`numpy`（`requirements.txt` 见项目根目录）。

## 运行 GUI

推荐通过启动器运行：GUI 出错或崩溃时会把错误/崩溃堆栈记录到
`test_output/gui_error.log`。

```bash
# 正常启动
python3 scripts/launch_gui.py

# 启动后自动加载指定预处理 CSV（可用 config/preprocess.json 的 preprocessed_data 目录下的文件测试）
python3 scripts/launch_gui.py --file "/Users/jucat/data/ashare/preprocessed/百联股份_600827.csv"
```

也可直接运行（不记录错误日志）：

```bash
python3 -m src.gui.main [--file PATH]
```

## GUI 功能

左侧主窗口 `chart_win`（多个纵向对齐小窗口 `sub_win`，横轴为共享时间轴）：

- 点线图显示数据；每个 sub_win 支持左、右两个纵列（各 5 个自适应刻度）。
- 时间轴默认显示 `show_days = 21` 日数据；右上角 “＋” 翻倍、“－” 减半，
  缩放时保持右边缘（最新数据）不变。
- 底部横向滚动条默认在最右侧，即默认显示最新数据。
- 每列有隐形时间柱，鼠标点击某列时所有 sub_win 同时高亮该时间柱，
  悬浮窗 `float_win` 显示具体日期与各序列数值；
  鼠标执行其他动作（移出该列 / 右键 / 滚轮 / 离开）关闭悬浮窗。
- 峰值标签显示在“收盘价”曲线数据点上方：正方形中间白色镂空字母，
  T（高峰）用红色、B（低谷）用绿色；标签大小固定、不受缩放影响，
  缩放（包括缩小到全部数据）时始终显示。

右侧副窗口 `info_win`：标题从 CSV 列项读取显示股票名称、代码，并展示行业、
数据范围、行数、文件路径与列项列表。

菜单栏：

- 文件 → 打开文件：弹窗选择 CSV，加载数据并把显示内容打包进缓存路径，
  方便下次快速显示。
- 文件 → 打开最近文件：以子菜单（右侧展开）显示最近打开的文件，最多 10 个、
  最新在前；点击直接重新加载（同样先查缓存），可一键清空记录。记录保存在
  `config/gui.json` 的 `tmp` 目录下（`recent_files.json`）。
- 编辑 → 添加数据：选择列项（可多选），选择加载到哪个 sub_win 或新建窗口，
  以及加载到左纵列还是右纵列。sub_win 以列项名命名，多列项用 “-” 拼接；
  同一 sub_win 内不同列项用不同颜色显示。
  若添加了“收盘价”则自动在该窗口显示“峰值标签”。
- 编辑 → 删除数据：选择 sub_win 及其中的数据源（可多选）删除；删除后窗口名
  按剩余列项重算，若没有剩余列项则整个 sub_win 被移除。

## 缓存管理（后端）

- 缓存路径来自 `config/gui.json` 的 `tmp` 目录（`cache/` 子目录存单个文件状态，
  `cache_index.json` 为索引）。
- 打开新文件时，把旧文件的显示内容（sub_win 及列项/纵列、`show_days`、offset 等）
  打包缓存，**不包含 csv 数据数值**。
- 每次“编辑”（添加数据 / 删除数据）后都会更新当前文件的缓存记录；
  显示内容被清空时自动移除该缓存。
- 最多缓存 100 个文件的数据，超出时按“最久未打开”淘汰并删除对应缓存。
- 打开文件（或最近文件）时先从缓存中找记录，命中则快速恢复显示。

## 目录结构

```
src/
├── preprocess/             # 数据预处理功能（scripts/preprocess.py 只负责启动）
│   ├── handle/             # 基础工作
│   │   ├── config.py       # 读取 config/preprocess.json（raw_data / preprocessed_data）
│   │   ├── indicators.py   # 列定义、指标计算（M21C…）、ST 判断、文件名清洗
│   │   └── processor.py    # 单文件处理与多进程编排（日志、临时文件、文件名分配）
│   └── label/              # 标签计算
│       └── peak.py         # 峰值标签（T / B）计算
└── gui/                    # PySide6 GUI
    ├── main.py             # 程序入口（暗色主题、--file 参数）
    ├── business/           # 后端业务逻辑（不依赖 Qt）
    │   ├── config.py       # 读取 config/preprocess.json、config/gui.json
    │   ├── data_store.py   # CSV 加载与数据缓存
    │   ├── chart_model.py  # 图表模型（sub_win / 序列 / 时间窗口 / 滚动）
    │   ├── cache.py        # 显示内容缓存（最多 100 个文件）
    │   └── recent_files.py # 最近打开文件记录（保存到 tmp 目录）
    └── ui/                 # 前端显示
        ├── main_window.py  # 主窗口 + 菜单栏（含“打开最近文件”）
        ├── chart_win.py    # 左侧图表主窗口（sub_win 排列、缩放、滚动条、全局高亮、悬浮窗）
        ├── sub_win.py      # 单个小窗口点线图绘制与时间柱高亮交互
        ├── float_win.py    # 悬浮窗
        ├── info_win.py     # 右侧信息窗口
        ├── dialogs.py      # “添加数据 / 删除数据”对话框
        └── chart_utils.py  # 刻度/数值格式化等绘图工具
```

## 测试

```bash
python3 -m unittest discover -s test -p "test_*.py" -v
```

- `test/test_gui_business.py`：数据加载与图表模型单元测试。
- `test/test_gui_smoke.py`：GUI 离屏渲染冒烟测试。

渲染检查截图可查看 `test_output/gui_check.png`。
