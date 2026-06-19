# 第 3 阶段：特征工程状态报告

## 运行结果

- 数据集：`home_credit`。
- 输入：`data/processed/home_credit_processed.csv`。
- 输出完整特征数据：`data\processed\home_credit_features.csv`。
- 输出训练集：`data\processed\home_credit_features_train.csv`。
- 输出验证集：`data\processed\home_credit_features_valid.csv`。
- 输出测试集：`data\processed\home_credit_features_test.csv`。
- 新增衍生特征数：18。
- 后续特征矩阵候选字段数：142。
- 类别编码候选字段数：16。

## 验收检查

- [x] 沿用固定 train/valid/test 划分，未重新抽样。
- [x] TARGET、SK_ID_CURR、split 未进入模型特征集合。
- [x] 工程后数值特征无 inf 或 -inf。
- [x] 工程后关键数值特征无缺失。
- [x] 每个新增衍生特征都有公式和业务含义。

## 说明

- 编码器、IV、数值处理边界均只使用 `train` split 拟合。
- One-Hot 稀疏矩阵未落盘，避免提交大体积中间产物。
- `TARGET`、`SK_ID_CURR`、`split` 只作为元数据保留，不作为模型特征。
