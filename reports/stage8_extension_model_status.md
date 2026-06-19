# 第 8 阶段：扩展模型实验状态报告

## 运行结果

- 扩展模型：PyTorch MLP + categorical embeddings。
- 实际设备：`cpu`。
- 最佳 epoch：`8`；训练 epoch 数：`11`。
- valid F1 最优阈值：`0.72`。
- Test ROC-AUC：`0.7463`；PR-AUC：`0.2247`；F1：`0.2907`。

## 与 LightGBM 对比

- PR-AUC 最优：`stage5_lightgbm_raw_gpu` = 0.2424。
- F1 最优：`stage8_pytorch_mlp_embedding` = 0.2907。
- 扩展模型在部分指标上有竞争力，但仍需更多调参和校准后才能进入主线。
- 深度模型可能受训练预算、超参数、类别嵌入维度和不平衡处理方式影响；本阶段只作为扩展探索。

## 验收检查

- [x] 已训练 PyTorch MLP+Embedding 扩展模型。
- [x] 已输出 ROC-AUC、PR-AUC、Precision、Recall、F1 和混淆矩阵。
- [x] 已与 Stage 5/6 LightGBM 主线模型对比。
- [x] scaler、类别映射、pos_weight 只使用 train split；阈值选择只使用 valid split。
- [x] 模型文件仅保存到本地 `models/`，不提交到 Git。

## 复现命令

```powershell
.\.venv\Scripts\python.exe .\src\run_stage8_extension_models.py
```
