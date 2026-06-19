# 个人贷款违约预测

本项目围绕“个人贷款违约预测：面向不平衡与业务代价的系统性改进”展开。当前阶段目标是完成第 1-2 周任务：数据准备、EDA、清洗、基础特征工程和初版基线模型。

## 当前目录

```text
data/raw/          原始数据
data/processed/    清洗后数据与固定划分
notebooks/         探索性 notebook
src/               可复用脚本
reports/figures/   图表输出
reports/tables/    表格输出
models/            模型文件
app/               展示系统
goal/              行动手册与原始汇报材料
```

## 数据来源

主实验数据集计划使用 Kaggle Home Credit Default Risk。该数据通常需要 Kaggle 账号和 API 凭据。

扩展验证数据集使用 UCI Default of Credit Card Clients。UCI 官方页面显示该数据集包含 30000 条样本、23 个特征，任务类型为二分类，目标变量为下月是否违约。

## 推荐运行顺序

安装依赖：

```powershell
py -m pip install -r requirements.txt
```

下载或放置数据：

```powershell
py src/download_data.py
```

运行第 1-2 周流程：

```powershell
py src/run_week1_2.py
```

如果没有 `py` 命令，请改用本机 Python 路径。

## 产出物

运行成功后，重点检查：

- `reports/tables/`：数据摘要、字段说明、缺失值统计、目标分布、模型结果表。
- `reports/figures/`：EDA 图、ROC/PR 曲线、混淆矩阵。
- `data/processed/`：清洗后数据和固定训练/验证/测试划分。
- `models/`：保存的初版模型。
- `reports/week1_2_status.md`：第 1-2 周阶段状态报告。
