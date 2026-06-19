# home_credit EDA 与数据清洗说明

## 目标

本说明用于补齐第 2 阶段“EDA 与数据清洗”的可复现证据。当前数据集为 `home_credit`，目标变量为 `TARGET`，原始训练表路径为 `data\raw\home_credit\application_train.csv`。

## 固定划分

- 划分方式：按 `TARGET` 分层随机划分。
- 随机种子：`42`。
- 比例：训练集 70%，验证集 15%，测试集 15%。
- 划分摘要：`reports\tables\home_credit_split_summary.csv`。
- 完整 processed 数据：`data\processed\home_credit_processed.csv`。
- 独立划分文件：`data\processed\home_credit_train.csv`、`data\processed\home_credit_valid.csv`、`data\processed\home_credit_test.csv`。

## 清洗规则

1. `TARGET` 只作为标签使用，不进入特征矩阵。
2. ID 字段 `SK_ID_CURR` 只用于追踪和关联，不进入模型特征。
3. Home Credit 中 `DAYS_EMPLOYED = 365243` 视为缺失哨兵值，替换为缺失，并新增 `DAYS_EMPLOYED_ANOMALY` 标记。
4. Home Credit 中 `CODE_GENDER = XNA` 视为无效类别，进入类别缺失处理。
5. 金额类字段中不合理的非正/负值先置为缺失，再按训练集统计量处理。
6. 数值特征缺失值使用训练集 median 填补，填补值记录在 `reports\tables\home_credit_missing_imputation_policy.csv`。
7. 类别特征缺失值填为 `__MISSING__`；训练集中低频类别归并为 `__RARE__`，避免后续编码过度稀疏。

## EDA 产物

- 数据集摘要：`reports\tables\home_credit_dataset_summary.csv`。
- 缺失率统计：`reports\tables\home_credit_missing_values.csv`。
- 数值变量摘要：`reports\tables\home_credit_numeric_summary.csv`。
- 异常值与偏态报告：`reports\tables\home_credit_outlier_invalid_report.csv`。
- 违约/非违约数值对比：`reports\tables\home_credit_target_numeric_comparison.csv`。
- 类别变量目标率对比：`reports\tables\home_credit_target_categorical_comparison.csv`。
- 稀有类别清单：`reports\tables\home_credit_rare_categories.csv`。
- 清洗动作审计：`reports\tables\home_credit_cleaning_audit.csv`。
- EDA notebook：`notebooks\home_credit_eda_cleaning.ipynb`。
- EDA 图表目录：`reports\figures`。

## 复现方式

```powershell
$env:TRAIN_DEVICE="gpu"
.\.venv\Scripts\python.exe .\src\run_week1_2.py
```

## 注意事项

- EDA 表保留数据质量视角，因此会展示原始缺失、偏态和异常信号。
- processed 数据是建模入口，缺失填补和低频类别归并只使用训练集统计量。
- 当前基线模型仍只使用数值特征；类别特征清洗是为后续 One-Hot、WOE/IV 或 embedding 实验预留。
