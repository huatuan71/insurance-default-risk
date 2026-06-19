# 第 7 阶段：可解释性分析状态报告

## 运行结果

- 数据集：`home_credit`。
- 解释对象：Stage 6 推荐中使用到的 `stage5_lightgbm_smote_gpu` 与 `stage5_lightgbm_raw_gpu`。
- SHAP 全局解释样本：test split 抽样 2000 行，优先保留违约样本。
- 解释口径：SHAP 值解释模型风险分数方向，不代表因果关系。

## 全局关键变量

- `stage5_lightgbm_raw_gpu` 主要依赖：ext_source_mean、annuity_credit_ratio、goods_credit_ratio、EXT_SOURCE_3、NAME_EDUCATION_TYPE_Higher education、CODE_GENDER_M、EXT_SOURCE_1、ext_source_max。
- `stage5_lightgbm_smote_gpu` 主要依赖：ext_source_mean、NAME_INCOME_TYPE_Working、ext_source_min、CODE_GENDER_M、FLAG_WORK_PHONE、REGION_RATING_CLIENT_W_CITY、FLAG_OWN_CAR_N、NAME_EDUCATION_TYPE_Secondary / secondary special。

## 单样本解释案例

- `main_fn10_tp_high_risk`：SK_ID_CURR `440963`，真实标签 `1`，预测分数 `0.7186`，阈值 `0.10`，预测 `1`。
  - 推高风险：ext_source_mean (+0.6635; 三个外部信用评分的平均水平，用来概括申请人的综合外部信用质量。)
  - 降低风险：NAME_INCOME_TYPE_Working (-0.3231; 收入来源类型。)
- `main_fn10_fp_false_reject`：SK_ID_CURR `226076`，真实标签 `0`，预测分数 `0.1000`，阈值 `0.10`，预测 `1`。
  - 推高风险：EXT_SOURCE_1 (+0.1085; 外部信用评分相关变量，通常反映第三方或外部数据源对申请人风险质量的综合判断。)
  - 降低风险：ext_source_min (-0.6540; 三个外部信用评分中的最低值，用来捕捉外部评分中的短板信号。)
- `main_fn10_tn_low_risk`：SK_ID_CURR `455551`，真实标签 `0`，预测分数 `0.0030`，阈值 `0.10`，预测 `0`。
  - 推高风险：DAYS_BIRTH (+0.0412; 申请人年龄，通常通过年龄阶段反映生命周期和收入稳定性。)
  - 降低风险：ext_source_mean (-1.0717; 三个外部信用评分的平均水平，用来概括申请人的综合外部信用质量。)
- `fn20_raw_tp_high_risk`：SK_ID_CURR `173014`，真实标签 `1`，预测分数 `0.7873`，阈值 `0.05`，预测 `1`。
  - 推高风险：ext_source_mean (+1.3327; 三个外部信用评分的平均水平，用来概括申请人的综合外部信用质量。)
  - 降低风险：AMT_REQ_CREDIT_BUREAU_MON (-0.1461; 申请前一个月内的征信查询次数，反映近期信贷申请活跃度。)

## 业务解释

- `age_and_tenure`：代表字段包括 DAYS_BIRTH、DAYS_EMPLOYED、DAYS_ID_PUBLISH、DAYS_LAST_PHONE_CHANGE。
- `amount_and_ratio`：代表字段包括 AMT_ANNUITY、annuity_credit_ratio、annuity_income_ratio、credit_income_ratio。
- `categorical_profile`：代表字段包括 CODE_GENDER、NAME_EDUCATION_TYPE、NAME_FAMILY_STATUS、NAME_INCOME_TYPE。
- `external_score`：代表字段包括 EXT_SOURCE_1、EXT_SOURCE_2、EXT_SOURCE_3、ext_source_max。
- `flags_and_contacts`：代表字段包括 FLAG_OWN_CAR、FLAG_WORK_PHONE、FLAG_DOCUMENT_3、FLAG_OWN_REALTY。
- `other`：代表字段包括 REGION_POPULATION_RELATIVE、log_AMT_ANNUITY、DEF_30_CNT_SOCIAL_CIRCLE、EMERGENCYSTATE_MODE。

## 验收检查

- [x] 已输出 LightGBM feature importance 图。
- [x] 已输出 SHAP summary/bar 图。
- [x] 已输出 TP、FP、TN 和 20:1 raw TP 单样本 waterfall 图。
- [x] 已生成关键特征业务解释表。
- [x] 只解释 test split，未重新训练、未重新划分。

## 复现命令

```powershell
.\.venv\Scripts\python.exe .\src\run_stage7_explainability.py
```
