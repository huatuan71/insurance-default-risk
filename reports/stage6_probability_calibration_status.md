# 概率校准实验状态报告

## 运行结果

- 数据集：`home_credit`。
- 校准对象：Stage 6 推荐 LightGBM 模型与 Stage 5 XGBoost raw 强基线。
- 校准方法：raw、sigmoid/Platt、isotonic。
- 校准器拟合：只使用 valid split；test split 只做最终评估。
- 评估指标：Brier score、log loss、ECE、MCE、ROC-AUC、PR-AUC，以及 valid-F1 阈值下的混淆矩阵指标。

## 最佳校准结果

- Test Brier score 最优：`stage5_xgboost_raw_cuda` / `raw` = 0.06769。
- Test ECE 最优：`stage5_xgboost_raw_cuda` / `raw` = 0.00103。
- valid-F1 阈值后 Test F1 最优：`stage5_xgboost_raw_cuda` / `raw` = 0.3145，阈值 `0.15`。

## 分模型结论

- `stage5_lightgbm_raw_gpu`：Brier 最优 `raw`，较 raw 改善 `0.00000`；ECE 最优 `raw`，较 raw 改善 `0.00000`。
- `stage5_lightgbm_smote_gpu`：Brier 最优 `raw`，较 raw 改善 `0.00000`；ECE 最优 `raw`，较 raw 改善 `0.00000`。
- `stage5_xgboost_raw_cuda`：Brier 最优 `raw`，较 raw 改善 `0.00000`；ECE 最优 `raw`，较 raw 改善 `0.00000`。
- 结论：sigmoid 和 isotonic 在 test split 上没有超过 raw 分数，说明当前候选 GBDT 分数已经具备较好的概率一致性，展示和业务阈值阶段可继续使用 raw score。

## 解释口径

- 概率校准改善的是“预测概率是否接近真实违约率”，不一定提升 ROC-AUC 或 PR-AUC，因为排序能力主要由原模型决定。
- 本阶段没有重新训练基础模型，也没有使用 test split 拟合校准器。
- 若校准方法没有改善 Brier/ECE，应保留 raw score，而不是为了形式强行替换概率。

## 验收检查

- [x] 已输出 raw、sigmoid、isotonic 的校准指标。
- [x] 已输出 reliability curve、Brier/ECE/log loss 对比图。
- [x] 已输出校准后 valid-F1 阈值与 test 混淆矩阵指标。
- [x] 校准器只使用 valid split 拟合，test split 只用于最终评估。

## 复现命令

```powershell
.\.venv\Scripts\python.exe .\src\run_stage6_probability_calibration.py
```
