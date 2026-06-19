# 第 5 阶段：不平衡处理策略对比状态报告

## 运行结果

- 数据集：`home_credit`。
- 输入特征数据：`data/processed/home_credit_features.csv`。
- 固定划分：沿用 `split`，未重新抽样。
- 训练行数：215257。
- 验证行数：46127。
- 测试行数：46127。
- 模型范围：LightGBM + XGBoost。
- 策略范围：raw、weighted、random_under_sample、smote、threshold_moving。

## 模型与设备

- `stage5_lightgbm_raw_gpu`：strategy `raw`，requested `gpu`，actual `gpu`，status `success`。
- `stage5_lightgbm_weighted_gpu`：strategy `weighted`，requested `gpu`，actual `gpu`，status `success`。
- `stage5_lightgbm_random_under_sample_gpu`：strategy `random_under_sample`，requested `gpu`，actual `gpu`，status `success`。
- `stage5_lightgbm_smote_gpu`：strategy `smote`，requested `gpu`，actual `gpu`，status `success`。
- `stage5_lightgbm_threshold_moving_gpu`：strategy `threshold_moving`，requested `gpu`，actual `gpu`，status `derived_from_raw`。
- `stage5_xgboost_raw_cuda`：strategy `raw`，requested `cuda`，actual `cuda`，status `success`。
- `stage5_xgboost_weighted_cuda`：strategy `weighted`，requested `cuda`，actual `cuda`，status `success`。
- `stage5_xgboost_random_under_sample_cuda`：strategy `random_under_sample`，requested `cuda`，actual `cuda`，status `success`。
- `stage5_xgboost_smote_cuda`：strategy `smote`，requested `cuda`，actual `cuda`，status `success`。
- `stage5_xgboost_threshold_moving_cuda`：strategy `threshold_moving`，requested `cuda`，actual `cuda`，status `derived_from_raw`。

## 测试集最佳结果

- Test pr_auc 最优训练策略：`stage5_xgboost_raw_cuda`，strategy `raw` = 0.2451。
- Test f1 最优：`stage5_xgboost_threshold_moving_cuda`，strategy `threshold_moving` = 0.3145。
- Test recall 最优：`stage5_lightgbm_random_under_sample_gpu`，strategy `random_under_sample` = 0.6928。
- 说明：threshold_moving 不改变排序分数，因此不单独视为 PR-AUC 提升策略。

## 业务影响解读

- `lightgbm` F1 最优策略：`threshold_moving`，相对 raw 多识别 1130 个违约样本，新增 3041 个误报。
- `lightgbm` Recall 最优策略：`random_under_sample`，相对 raw 多识别 2488 个违约样本，新增 13011 个误报。
- `xgboost` F1 最优策略：`threshold_moving`，相对 raw 多识别 1575 个违约样本，新增 5045 个误报。
- `xgboost` Recall 最优策略：`random_under_sample`，相对 raw 多识别 2503 个违约样本，新增 13080 个误报。

## 阈值移动

- `lightgbm` 在 valid split 上选择阈值 `0.19`，valid F1 = 0.3162。
- `xgboost` 在 valid split 上选择阈值 `0.15`，valid F1 = 0.3178。

## 泄漏控制

- `TARGET`、`SK_ID_CURR`、`split` 未进入模型特征矩阵。
- 编码器、采样器、SMOTE 和模型均只在 train split 拟合。
- `scale_pos_weight` 只使用 train split 的正负样本比例计算。
- threshold_moving 只用 valid split 选择阈值，test split 只做最终评估。
- SMOTE 作为对照实验保留；其合成样本不一定具有真实业务含义，后续解释时需谨慎。

## 验收检查

- [x] LightGBM 和 XGBoost 均完成不平衡策略对比。
- [x] 输出 PR-AUC、Recall、Precision、F1 和混淆矩阵。
- [x] 输出相对 raw 的 TP/FP/FN 业务影响表。
- [x] 输出 PR 曲线、指标对比图和召回率-误报数量对比图。
- [x] GPU 失败时记录 CPU fallback，不伪装设备。

## 复现命令

```powershell
$env:TRAIN_DEVICE="gpu"
.\.venv\Scripts\python.exe .\src\run_stage5_imbalance.py
```
