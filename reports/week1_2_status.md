# 第 1-2 周任务状态报告

## 自动运行结果

- 使用数据集：`home_credit`。
- 原始文件：`data\raw\home_credit\application_train.csv`。
- 样本数：307511。
- 字段数：127。
- 目标变量：`TARGET`。
- Home Credit 主实验数据：已存在。
- 字段说明表：`reports\tables\home_credit_field_dictionary.csv`。
- 数据泄漏风险字段清单：`reports\tables\home_credit_leakage_risk_fields.csv`。
- EDA 表格已输出到：`reports\tables`。
- 异常值与偏态报告：`reports\tables\home_credit_outlier_invalid_report.csv`。
- 违约/非违约数值对比：`reports\tables\home_credit_target_numeric_comparison.csv`。
- 类别目标率对比：`reports\tables\home_credit_target_categorical_comparison.csv`。
- 稀有类别清单：`reports\tables\home_credit_rare_categories.csv`。
- EDA 图表已输出到：`reports\figures`。
- 清洗动作审计：`reports\tables\home_credit_cleaning_audit.csv`。
- 清洗后数据：`data\processed\home_credit_processed.csv`。
- 清洗后缺失单元数：0。
- 训练/验证/测试划分摘要：`reports\tables\home_credit_split_summary.csv`。
- 独立训练集文件：`data\processed\home_credit_train.csv`。
- 独立验证集文件：`data\processed\home_credit_valid.csv`。
- 独立测试集文件：`data\processed\home_credit_test.csv`。
- 数据清洗说明：`reports\home_credit_data_cleaning_notes.md`。
- EDA notebook：`notebooks\home_credit_eda_cleaning.ipynb`。

## 依赖状态

- sklearn: available
- lightgbm: available
- xgboost: available
- joblib: available
- training_device: gpu

## 基线模型

- 基线结果表：`reports\tables\baseline_results.csv`。
- 模型运行日志：`reports\tables\model_run_log.csv`。
- `xgboost_cuda_numeric`：requested `cuda`，actual `cuda`，status `success`。
- `lightgbm_gpu_numeric`：requested `gpu`，actual `gpu`，status `success`。
- Test roc_auc 最优：`xgboost_cuda_numeric` = 0.7451。
- Test pr_auc 最优：`xgboost_cuda_numeric` = 0.2296。
- Test f1 最优：`lightgbm_gpu_numeric` = 0.2679。
- GPU-only 模式已启用：CPU-only 的 Logistic Regression 与 RandomForest 已跳过。
- LightGBM 已完成 GPU 训练。

## 第 1-2 周检查清单

- [x] 数据集基本信息表。
- [x] 字段说明表。
- [x] 数据泄漏风险字段清单。
- [x] 目标变量分布表。
- [x] 缺失值统计表。
- [x] 异常值、无效值与偏态报告。
- [x] 违约组与非违约组差异分析。
- [x] 类别变量频数、目标率和稀有类别分析。
- [x] 数据清洗说明。
- [x] EDA notebook。
- [x] 清洗后数据与固定划分。
- [x] 清洗后训练集、验证集和测试集独立文件。
- [x] XGBoost CUDA 初版结果。
- [x] LightGBM 初版结果。

## 后续建议

1. 继续基于 Home Credit 做第 3-4 周：不平衡策略、概率校准、业务阈值优化和 SHAP 分析。
2. 若 LightGBM 使用了 CPU fallback，后续可继续排查 LightGBM OpenCL/Boost Compute 环境以争取 GPU 版。
3. 将 ROC/PR 曲线、混淆矩阵和基线结果表整理进后续 PPT。
