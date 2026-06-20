# 实验产出说明

本目录保存项目的可提交实验证据。阅读顺序建议为：先看各阶段 `*_status.md` 得到结论，再用 `tables/` 核对数值，最后从 `figures/` 选择 PPT 图。除特别说明外，主线结论均来自 Home Credit 固定 train/valid/test 划分。

## 数据产出

| 路径或文件 | 内容 | 作用 |
| --- | --- | --- |
| `data/raw/home_credit/application_train.csv` | 含 `TARGET` 的原始申请主表 | 主实验原始数据；不可手改。 |
| `data/raw/home_credit/application_test.csv` | 不含标签的官方测试申请表 | 检查 train/test 字段一致性与未来推理可用字段。 |
| `data/raw/home_credit/HomeCredit_columns_description.csv` | 官方字段字典 | 为字段说明和业务释义提供权威来源。 |
| `data/raw/home_credit/{bureau,previous_application,POS_CASH_balance,credit_card_balance,installments_payments,bureau_balance}.csv` | 历史辅助表 | 当前未入模；后续聚合时须按申请时点截断，防止时间泄漏。 |
| `data/raw/taiwan/default of credit card clients.xls` | Taiwan 扩展数据集 | 仅用于数据准备/EDA 对照，尚未影响 Home Credit 主线结论。 |
| `data/processed/home_credit_processed.csv` | 清洗、填补并固定 split 的主表 | 第 3.2 阶段可复现输入。 |
| `data/processed/home_credit_{train,valid,test}.csv` | 清洗后的三份固定划分 | 核对样本量和复现实验；不应重新抽样。 |
| `data/processed/home_credit_features.csv` | 第 3.3 工程后的完整特征表 | Stage 4 至 8 的共同输入。 |
| `data/processed/home_credit_features_{train,valid,test}.csv` | 工程后的三份固定划分 | 训练、阈值选择和最终测试的隔离依据。 |
| `data/processed/taiwan_credit_processed.csv` | Taiwan 数据清洗结果 | 扩展数据集的预处理产物。 |

这些 CSV 与模型文件均为可再生的大文件，按 `.gitignore` 不提交；由 `src/` 脚本重新生成。

## 数据准备与 EDA 表

| 文件 | 内容与作用 |
| --- | --- |
| `tables/home_credit_dataset_summary.csv`、`taiwan_credit_dataset_summary.csv` | 样本数、字段数和标签分布的总览；用于数据集介绍页。 |
| `tables/*_field_dictionary.csv` | 字段类型、缺失率、角色与官方说明；用于说明目标、ID、类别/数值字段。 |
| `tables/home_credit_leakage_risk_fields.csv` | `TARGET`、`SK_ID_CURR`、train/test 差异及辅助表时间泄漏政策；用于证明无泄漏控制。 |
| `tables/*_target_distribution.csv` | 违约/非违约计数和比例；用于说明类别不平衡。 |
| `tables/*_missing_values.csv` | 各字段缺失量和缺失率；用于清洗依据。 |
| `tables/dependency_status.csv` | Python、GPU 与关键建模依赖的可用性快照；用于解释训练环境。 |
| `tables/*_numeric_summary.csv`、`tables/*_categorical_summary.csv` | 数值描述统计与类别频数；用于 EDA 附录或特征选择讨论。 |
| `tables/*_split_summary.csv` | train/valid/test 的行数及违约率；用于证明固定划分稳定。 |
| `tables/home_credit_target_numeric_comparison.csv`、`home_credit_target_categorical_comparison.csv` | 违约与非违约群体的关键变量差异；用于业务洞察。 |
| `tables/home_credit_cleaning_audit.csv`、`home_credit_missing_imputation_policy.csv`、`home_credit_outlier_invalid_report.csv`、`home_credit_rare_categories.csv` | 清洗步骤、缺失填补、异常/非法值和稀有类别的审计证据。 |
| `tables/home_credit_numeric_processing_policy.csv`、`home_credit_stage3_fill_policy.csv` | 训练集拟合的截尾与填补规则；用于证明未用 valid/test 统计量。 |

对应图表的用途如下：

| 图组 | 内容与作用 |
| --- | --- |
| `figures/home_credit_target_distribution.png`、`taiwan_credit_target_distribution.png` | 主数据集和扩展数据集的违约率；用于说明不平衡程度。 |
| `figures/home_credit_missing_top20.png`、`taiwan_credit_missing_top20.png` | 缺失率最高的字段；用于支持清洗策略。 |
| `figures/home_credit_*_hist.png`、`taiwan_credit_*_hist.png` | 核心数值或编码字段的分布；用于发现偏态、异常值和量纲差异。`home_credit_SK_ID_CURR_hist.png` 与 `taiwan_credit_ID_hist.png` 只用于确认 ID 分布，不能做业务解释。 |
| `figures/home_credit_target_compare_*.png` | 关键数值字段在违约/非违约群体中的分布差异；用于说明候选风险信号。 |
| `figures/home_credit_*_category_frequency.png` | 合同、教育、收入、性别、资产等类别字段的样本构成。 |
| `figures/home_credit_*_target_rate.png` | 同一类别字段的组内违约率；用于业务维度的风险差异说明。 |
| `figures/home_credit_target_numeric_mean_diff_top10.png` | 违约与非违约均值差异最大的数值字段；用于引出特征工程或解释性分析。 |

## 特征工程产出（Stage 3）

| 文件 | 内容与作用 |
| --- | --- |
| `tables/home_credit_feature_catalog.csv` | 工程后可用特征清单、类型和来源；建模字段的总目录。 |
| `tables/home_credit_derived_feature_dictionary.csv` | 比率、年龄/工龄、外部评分聚合、文档/联系标记、查询计数和对数特征的公式与业务含义。 |
| `tables/home_credit_feature_selection_record.csv` | 每列保留、排除、衍生或仅作 ID/标签的决定；重点证明 `TARGET`、`SK_ID_CURR`、`split` 未入模。 |
| `tables/home_credit_encoding_policy.csv` | One-Hot 编码与 `min_frequency` 规则；说明编码器仅在 train 拟合。 |
| `tables/home_credit_encoded_feature_names.csv` | 编码后的特征名清单；用于定位模型中的 one-hot 变量。 |
| `tables/home_credit_iv_summary.csv` | IV 排名参考；用于可解释性和后续 WOE 讨论，不代表正式 WOE 替换。 |
| `home_credit_feature_engineering_notes.md`、`stage3_feature_status.md` | 特征工程口径、产物位置与验收结论；Stage 3 的首读文档。 |

## 基线模型产出（Stage 4）

| 文件 | 内容与作用 |
| --- | --- |
| `tables/stage4_baseline_results.csv` | Logistic Regression、XGBoost、LightGBM 在 valid/test 的 ROC-AUC、PR-AUC、Precision、Recall、F1 与混淆矩阵；正式基线对比表。 |
| `tables/stage4_baseline_test_predictions.csv` | 测试集逐行预测审计数据；用于复算指标，不建议放入 PPT。 |
| `tables/stage4_model_params.csv`、`stage4_model_run_log.csv` | 模型超参数、请求/实际设备、耗时和 GPU 回退原因；用于复现。 |
| `tables/stage4_feature_matrix_summary.csv`、`stage4_transformed_feature_names.csv` | 三个 split 的特征矩阵规模及转换后字段；用于核验编码和泄漏控制。 |
| `figures/stage4_test_{roc,pr}_curves.png` | 三个基线的排序能力和少数类识别能力对比；PPT 首选。 |
| `figures/stage4_{roc_auc,pr_auc,precision,recall,f1}.png` | 单项指标柱状对比；用于一句话解释模型优劣。 |
| `figures/stage4_*_confusion_matrix.png` | 各模型 TP/FP/TN/FN；用于讲清错误类型。 |
| `stage4_baseline_status.md` | 基线阶段的实际设备、最佳结果和复现命令。 |

`tables/baseline_results.csv`、`baseline_test_predictions.csv`、`model_run_log.csv` 与 `figures/baseline_*` 是第 1-2 周的早期数值基线记录；可作为历史对照，最终汇报以 `stage4_*` 为准。

## 不平衡策略产出（Stage 5）

| 文件 | 内容与作用 |
| --- | --- |
| `tables/stage5_imbalance_results.csv` | LightGBM/XGBoost 在 raw、weighted、欠采样、SMOTE、阈值移动下的统一 test 指标；核心策略对比表。 |
| `tables/stage5_business_impact.csv` | 相对 raw 的新增 TP、FP、减少 FN 和指标变化；直接回答“多抓到多少违约、代价是多少”。 |
| `tables/stage5_threshold_search.csv` | 仅在 valid 上选择 F1 最优阈值的网格记录；用于证明没有用 test 调阈值。 |
| `tables/stage5_strategy_params.csv`、`stage5_strategy_run_log.csv` | 每种策略的采样/权重参数、设备、耗时和回退信息。 |
| `tables/stage5_feature_matrix_summary.csv` | 采样前后的特征矩阵与固定 split 核验。 |
| `figures/stage5_{lightgbm,xgboost}_test_pr_curves.png` | 同一模型不同不平衡策略的 PR 曲线；用于选择主策略。 |
| `figures/stage5_{roc_auc,pr_auc,precision,recall,f1}.png` | 各策略指标比较。 |
| `figures/stage5_recall_false_positive_tradeoff.png` | 召回提高时的误报代价；业务解释重点图。 |
| `figures/stage5_*_confusion_matrix.png` | 每个模型-策略组合的测试集错误构成。 |
| `stage5_imbalance_status.md` | 最佳策略、风险提示和结论摘要。 |

## 成本阈值与概率校准产出（两条 Stage 6 轨道）

文件名中的 `stage6_business_*` 是**业务成本阈值优化**，`stage6_probability_calibration_*` 是后补的**概率校准实验**；二者回答的问题不同，不能互相替代。

| 文件 | 内容与作用 |
| --- | --- |
| `tables/stage6_threshold_cost_grid.csv` | 各模型/成本场景/阈值的 valid 成本网格；用于寻找最小业务成本。 |
| `tables/stage6_optimal_thresholds.csv`、`stage6_recommended_thresholds.csv` | 每个 FN:FP 成本比的最优阈值和最终推荐；风控决策主表。 |
| `tables/stage6_business_threshold_impact.csv` | 推荐阈值相对 0.5 的 TP、FP、FN 与成本变化；用于量化业务收益。 |
| `tables/stage6_feature_matrix_summary.csv` | 成本优化使用的特征和 split 核验。 |
| `figures/stage6_cost_curve_fn{5,10,20}_fp1.png` | 三种 FN:FP 假设下的阈值-成本曲线；解释为何阈值不是 0.5。 |
| `figures/stage6_recommended_{test_cost,confusion_matrix_*.png}` | 推荐策略的测试成本与错误构成；用于最终方案页。 |
| `tables/stage6_probability_calibration_results.csv` | raw、Platt/sigmoid、isotonic 在 valid/test 的 Brier、log loss、ECE、MCE、ROC-AUC、PR-AUC。 |
| `tables/stage6_probability_calibration_bins.csv` | 每个概率分箱的平均预测概率和真实违约率；可靠性曲线的数据来源。 |
| `tables/stage6_probability_calibration_thresholds.csv`、`stage6_probability_calibration_decision_results.csv` | valid F1 阈值搜索与对应 test 决策指标。 |
| `tables/stage6_probability_calibration_summary.csv`、`stage6_probability_calibration_run_log.csv` | 各模型的最佳校准方法和可复现运行日志。 |
| `figures/stage6_probability_calibration_reliability_*.png` | 预测概率与真实频率的一致性；校准效果主图。 |
| `figures/stage6_probability_calibration_{brier_score,ece,log_loss,decision_*.png}` | 校准质量和阈值后业务指标对比。 |
| `stage6_business_threshold_status.md`、`stage6_probability_calibration_status.md` | 两条轨道的结论：成本阈值建议与“保留 raw score”的校准结论。 |

## 可解释性与扩展模型产出

| 文件 | 内容与作用 |
| --- | --- |
| `tables/stage7_lightgbm_feature_importance.csv` | 推荐 LightGBM 的内置特征重要性。 |
| `tables/stage7_shap_global_importance.csv` | SHAP 平均绝对贡献排名；回答模型整体依赖哪些变量。 |
| `tables/stage7_shap_sample_cases.csv` | TP、FP、TN 等典型样本的预测分数、阈值和主要正/负贡献特征。 |
| `tables/stage7_key_feature_business_interpretation.csv` | 将技术字段翻译为风险管理语言；是 SHAP 汇报讲稿依据。 |
| `figures/stage7_*_lightgbm_feature_importance.png`、`stage7_*_shap_{summary,bar}.png` | 全局变量重要性和风险方向；可解释性主图。 |
| `figures/stage7_*_waterfall.png` | 单个客户被判为高风险、误拒或低风险的原因；用于案例页。 |
| `stage7_explainability_status.md` | SHAP 的范围、限制和结论；需强调解释不等于因果。 |
| `tables/stage8_extension_model_results.csv` | MLP+Embedding 在 valid/test 的指标。 |
| `tables/stage8_lightgbm_comparison.csv` | MLP 与 LightGBM 主线模型的可比指标；用于判断深度模型是否值得采用。 |
| `tables/stage8_training_history.csv`、`stage8_threshold_search.csv` | 每轮训练曲线和 valid 阈值选择记录。 |
| `tables/stage8_extension_model_params.csv`、`stage8_extension_model_run_log.csv` | 网络结构、训练参数、设备、最佳 epoch 和耗时。 |
| `figures/stage8_training_curve.png` | MLP loss/PR-AUC 随 epoch 变化；证明 early stopping 的选择。 |
| `figures/stage8_test_{roc,pr}_curves.png`、`stage8_comparison_*.png`、`stage8_mlp_confusion_matrix.png` | 深度模型和 LightGBM 的性能、曲线和错误对比。 |
| `stage8_extension_model_status.md` | 扩展实验结论；不应取代 GBDT 主线。 |

## 状态报告、模型和复现入口

| 产出 | 内容与作用 |
| --- | --- |
| `week1_2_status.md`、`home_credit_data_cleaning_notes.md` | 数据准备与清洗阶段的过程记录。 |
| `stage3_feature_status.md` 至 `stage8_extension_model_status.md` | 各阶段验收结论、输入、主要结果、限制和复现命令；每次汇报前先核对这里。 |
| `models/*.joblib`、`models/*.pt` | 本地训练模型，用于 Stage 6-8 的复用与演示；体积较大且可再生，不提交 Git。 |
| `src/run_week1_2.py`、`run_stage3_features.py`、`run_stage4_baselines.py`、`run_stage5_imbalance.py`、`run_stage6_business_thresholds.py`、`run_stage6_probability_calibration.py`、`run_stage7_explainability.py`、`run_stage8_extension_models.py` | 从数据准备到扩展实验的顺序化复现入口。 |
| `src/inference.py` | 本地应用共用的 Home Credit 输入校验、特征工程、Pipeline 评分、成本阈值读取与单样本 SHAP 服务。 |
| `app/streamlit_app.py`、`app/README.md` | 课堂演示决策台及启动说明；将风险分数、业务阈值、批量 CSV 评分、实验图表和 SHAP 解释整合为可交互页面。 |

## 最终汇报的最小图表组合

1. `home_credit_target_distribution.png`：说明类别不平衡。
2. `stage4_test_pr_curves.png`：说明 LightGBM/XGBoost 强基线。
3. `stage5_recall_false_positive_tradeoff.png`：说明不平衡策略不是单看分数。
4. `stage6_cost_curve_fn10_fp1.png` 和对应推荐混淆矩阵：说明业务阈值决策。
5. 一张 `stage7_*_shap_summary.png` 加一张 `stage7_*_waterfall.png`：说明模型为何这样判断。
6. `stage8_comparison_pr_auc.png`：用一句话交代扩展模型与 GBDT 的关系。

所有图表和表格均应在 PPT 中标注模型、策略、数据划分及阈值口径；不要把不同阶段、不同阈值的数字混用。
