# Notebooks

当前主要 notebook：

- `home_credit_eda_cleaning.ipynb`：第 2 阶段 EDA 与数据清洗检查，汇总缺失率、异常值、目标组对比、类别频数、稀有类别、清洗规则和固定划分。

建议将探索性分析拆为：

- `01_eda.ipynb`：数据理解、违约率、缺失值、变量分布。
- `02_baseline_models.ipynb`：Logistic Regression、XGBoost、LightGBM 基线。
- `03_imbalance_experiments.ipynb`：权重法、采样法、SMOTE、阈值移动。
- `04_shap_analysis.ipynb`：全局和单样本 SHAP 分析。

当前 notebook 由 `src/run_week1_2.py` 生成，建议先重跑脚本再打开查看。
