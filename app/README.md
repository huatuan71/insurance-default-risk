# 本地贷款违约风险决策台

这是一个用于课堂演示的 Streamlit 应用。它不会重新训练模型，而是加载本地 Stage 5 LightGBM Pipeline、Stage 6 推荐阈值和 Stage 7 SHAP 产物。

## 前置条件

以下本地资产必须存在，且保持在 `.gitignore` 中：

- `data/processed/home_credit_features.csv`
- `models/stage5_lightgbm_smote_gpu.joblib`
- `models/stage5_lightgbm_raw_gpu.joblib`
- `reports/tables/stage6_recommended_thresholds.csv`
- `reports/tables/stage7_shap_sample_cases.csv`

缺失时按顺序运行：

```powershell
.\.venv\Scripts\python.exe .\src\run_stage3_features.py
.\.venv\Scripts\python.exe .\src\run_stage5_imbalance.py
.\.venv\Scripts\python.exe .\src\run_stage6_business_thresholds.py
.\.venv\Scripts\python.exe .\src\run_stage7_explainability.py
```

## 启动

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app\streamlit_app.py
```

默认使用 `FN:FP = 10:1`，可切换到 `5:1` 或 `20:1` 成本场景。上传 CSV 必须包含 Home Credit `application_train/application_test` 的原始申请字段；`TARGET` 若存在会被忽略。

应用仅在内存中处理上传文件，不写入磁盘。它是公开竞赛数据的课程演示，不能用于真实放贷或个人信用决策。
