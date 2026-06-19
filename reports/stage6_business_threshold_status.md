# 第 6 阶段：业务代价阈值优化状态报告

## 运行结果

- 数据集：`home_credit`。
- 输入特征数据：`data/processed/home_credit_features.csv`。
- 候选模型：Stage 5 已训练的 LightGBM 与 XGBoost 策略模型。
- 成本假设：FN:FP = 5:1、10:1、20:1。
- 阈值网格：0.01 到 0.99，步长 0.01。
- 阈值选择：只使用 valid split；test split 只做最终评估。
- 训练行数：215257。
- 验证行数：46127。
- 测试行数：46127。

## 推荐阈值

- `5:1`：推荐 `lightgbm/smote`，阈值 `0.18`，test 总成本 `15684`。
  - 相比同模型 0.5 阈值节省 `2614`，多识别 `1241` 个违约样本，新增 `3591` 个误报。
- `10:1`：推荐 `lightgbm/smote`，阈值 `0.10`，test 总成本 `23976`。
  - 相比同模型 0.5 阈值节省 `12542`，多识别 `2251` 个违约样本，新增 `9968` 个误报。
- `20:1`：推荐 `lightgbm/raw`，阈值 `0.05`，test 总成本 `33237`。
  - 相比同模型 0.5 阈值节省 `39488`，多识别 `2990` 个违约样本，新增 `20312` 个误报。

## 为什么不是 0.5

- 在所有成本假设下，推荐阈值均低于 0.5，因为漏判违约的业务成本高于误拒正常客户。
- 降低阈值会提升违约召回率，但会增加误报；因此最终阈值必须跟 FN/FP 成本假设绑定解释。
- 本阶段未做概率校准，目标是基于现有模型分数形成业务阈值建议；校准实验仍留在后续阶段。

## 验收检查

- [x] 已生成 `reports\tables\stage6_threshold_cost_grid.csv`，共 2376 行 valid 阈值-成本记录。
- [x] 已生成 `reports\tables\stage6_optimal_thresholds.csv`，覆盖每个模型/策略/成本假设的最优阈值。
- [x] 已生成 `reports\tables\stage6_recommended_thresholds.csv`，给出每组成本假设的推荐决策。
- [x] 已输出阈值-业务成本曲线和推荐策略混淆矩阵。
- [x] 阈值选择只使用 valid split，test split 只用于最终评估。

## 复现命令

```powershell
.\.venv\Scripts\python.exe .\src\run_stage6_business_thresholds.py
```
