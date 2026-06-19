# 第 4 阶段：模型基线构建状态报告

## 运行结果

- 数据集：`home_credit`。
- 输入特征数据：`data/processed/home_credit_features.csv`。
- 固定划分：沿用 `split`，未重新抽样。
- 训练行数：215257。
- 验证行数：46127。
- 测试行数：46127。
- 数值特征数：126。
- 类别特征数：16。
- 训练策略：原始不平衡数据；不启用 class_weight、scale_pos_weight、采样或 SMOTE。

## 模型运行日志

- `stage4_logistic_regression`：requested `cpu`，actual `cpu`，status `success`。
- `stage4_xgboost_cuda`：requested `cuda`，actual `cuda`，status `success`。
- `stage4_lightgbm_gpu`：requested `gpu`，actual `gpu`，status `success`。

## 测试集最佳结果

- Test roc_auc 最优：`stage4_xgboost_cuda` = 0.7592。
- Test pr_auc 最优：`stage4_xgboost_cuda` = 0.2451。
- Test f1 最优：`stage4_lightgbm_gpu` = 0.0472。

## 验收检查

- [x] Logistic Regression 基线完成。
- [x] XGBoost 基线完成。
- [x] LightGBM 强基线完成。
- [x] ROC-AUC、PR-AUC、Precision、Recall、F1 和混淆矩阵已统一输出。
- [x] 模型参数、运行设备和 fallback 原因已记录。
- [x] 编码器、标准化器和模型只在 train split 拟合。

## 复现命令

```powershell
$env:TRAIN_DEVICE="gpu"
.\.venv\Scripts\python.exe .\src\run_stage4_baselines.py
```
