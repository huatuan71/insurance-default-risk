# Home Credit 特征工程说明

## 目标

本说明对应 README 3.3“特征工程”阶段。输入为 `data/processed/home_credit_processed.csv`，沿用第 3.2 阶段固定的 `train`、`valid`、`test` 划分，不重新抽样。

## 关键策略

- 标签 `TARGET`、ID `SK_ID_CURR`、划分字段 `split` 不进入模型特征。
- 新增 18 个业务衍生特征，均由申请主表的行级字段计算。
- 数值处理统计量、截尾边界、类别编码器和 IV 排名均只在 `train` split 拟合。
- 类别编码策略为 `OneHotEncoder(handle_unknown="ignore", min_frequency=50)`；当前不把稀疏 one-hot 矩阵落成 CSV。
- WOE/IV 当前只输出 IV 排名作为解释和筛选参考，不生成正式 WOE 替换特征。

## 产物

- 工程后完整数据：`data\processed\home_credit_features.csv`。
- 工程后训练/验证/测试：`data\processed\home_credit_features_train.csv`、`data\processed\home_credit_features_valid.csv`、`data\processed\home_credit_features_test.csv`。
- 特征清单：`reports\tables\home_credit_feature_catalog.csv`。
- 衍生特征说明：`reports\tables\home_credit_derived_feature_dictionary.csv`。
- 特征筛选记录：`reports\tables\home_credit_feature_selection_record.csv`。
- 类别编码策略：`reports\tables\home_credit_encoding_policy.csv`。
- 编码后特征名清单：`reports\tables\home_credit_encoded_feature_names.csv`。
- 数值处理策略：`reports\tables\home_credit_numeric_processing_policy.csv`。
- IV 排名：`reports\tables\home_credit_iv_summary.csv`。

## 当前结果

- 选择进入后续特征矩阵的字段数：142。
- 类别字段数：16。
- 所有工程后数据保留 `TARGET`、`SK_ID_CURR`、`split` 作为监督学习和追踪元数据。
- `data/processed/home_credit_features*.csv` 属于大体积可再生产物，继续由 `.gitignore` 排除。

## 复现命令

```powershell
.\.venv\Scripts\python.exe .\src\run_stage3_features.py
```
