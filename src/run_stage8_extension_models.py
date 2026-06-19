from __future__ import annotations

import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from features import ID_COLUMNS, SPLIT, TARGET
from project_paths import FIGURES, MODELS, REPORTS, TABLES, ensure_project_dirs, rel
from run_stage4_baselines import ensure_feature_file, split_feature_target
from train_baseline import evaluate_predictions, has_module


RANDOM_STATE = 42
MODEL_NAME = "stage8_pytorch_mlp_embedding"
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
HIDDEN_LAYERS = [256, 128, 64]
DROPOUT = 0.25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 4096
DEFAULT_MAX_EPOCHS = 20
DEFAULT_PATIENCE = 3
MIN_DELTA = 1e-5
LIGHTGBM_COMPARISON = [
    ("stage5_lightgbm_smote_gpu", "fn10_fp1"),
    ("stage5_lightgbm_raw_gpu", "fn20_fp1"),
]


@dataclass(frozen=True)
class EncodedData:
    numeric: np.ndarray
    categorical: np.ndarray
    labels: np.ndarray


def configure_plotting() -> None:
    os.environ.setdefault("WINDIR", r"C:\Windows")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", str(FIGURES.parent / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)


def require_torch():
    if not has_module("torch"):
        raise RuntimeError(
            "PyTorch is required for Stage 8. Install dependencies first, for example: "
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        )
    import torch

    return torch


def set_reproducible_seed(torch_module: object) -> None:
    np.random.seed(RANDOM_STATE)
    torch_module.manual_seed(RANDOM_STATE)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(RANDOM_STATE)


def embedding_dim(cardinality: int) -> int:
    return int(min(32, max(4, round(cardinality**0.5))))


def fit_category_maps(X_train: pd.DataFrame, categorical_features: list[str]) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for column in categorical_features:
        values = X_train[column].fillna("__MISSING__").astype(str)
        categories = sorted(values.unique().tolist())
        maps[column] = {category: idx + 1 for idx, category in enumerate(categories)}
    return maps


def transform_categories(X: pd.DataFrame, categorical_features: list[str], category_maps: dict[str, dict[str, int]]) -> np.ndarray:
    if not categorical_features:
        return np.zeros((len(X), 0), dtype=np.int64)
    columns = []
    for column in categorical_features:
        values = X[column].fillna("__MISSING__").astype(str)
        encoded = values.map(category_maps[column]).fillna(0).astype("int64").to_numpy()
        columns.append(encoded)
    return np.stack(columns, axis=1).astype(np.int64)


def encode_split(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    scaler: object,
    category_maps: dict[str, dict[str, int]],
) -> EncodedData:
    numeric = scaler.transform(X[numeric_features]).astype(np.float32) if numeric_features else np.zeros((len(X), 0), dtype=np.float32)
    categorical = transform_categories(X, categorical_features, category_maps)
    labels = y.to_numpy(dtype=np.float32)
    return EncodedData(numeric=numeric, categorical=categorical, labels=labels)


def make_loader(torch_module: object, encoded: EncodedData, batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch_module.from_numpy(encoded.numeric),
        torch_module.from_numpy(encoded.categorical),
        torch_module.from_numpy(encoded.labels),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_model_class(torch_module: object):
    nn = torch_module.nn

    class MLPEmbeddingModel(nn.Module):
        def __init__(self, numeric_dim: int, category_cardinalities: list[int], embedding_dims: list[int]) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList(
                [nn.Embedding(cardinality, dim, padding_idx=0) for cardinality, dim in zip(category_cardinalities, embedding_dims)]
            )
            input_dim = numeric_dim + sum(embedding_dims)
            layers: list[object] = []
            previous = input_dim
            for hidden in HIDDEN_LAYERS:
                layers.extend([nn.Linear(previous, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(DROPOUT)])
                previous = hidden
            layers.append(nn.Linear(previous, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, numeric, categorical):
            embeddings = []
            for idx, embedding in enumerate(self.embeddings):
                embeddings.append(embedding(categorical[:, idx]))
            if embeddings:
                combined = torch_module.cat([numeric, *embeddings], dim=1)
            else:
                combined = numeric
            return self.network(combined).squeeze(1)

    return MLPEmbeddingModel


def predict_scores_torch(torch_module: object, model: object, loader: object, device: object) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch_module.no_grad():
        for numeric, categorical, _ in loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            logits = model(numeric, categorical)
            scores.append(torch_module.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(scores)


def choose_threshold(y_valid: pd.Series, valid_scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in THRESHOLD_GRID:
        metrics = evaluate_predictions(y_valid, valid_scores, threshold=float(threshold))
        rows.append({"selection_split": "valid", **metrics})
    search = pd.DataFrame(rows)
    best = search.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return float(best["threshold"]), search


def train_model(
    torch_module: object,
    model: object,
    train_loader: object,
    valid_loader: object,
    y_valid: pd.Series,
    device: object,
    pos_weight: float,
    max_epochs: int,
    patience: int,
) -> tuple[object, pd.DataFrame, dict[str, object]]:
    from sklearn.metrics import average_precision_score

    model.to(device)
    criterion = torch_module.nn.BCEWithLogitsLoss(pos_weight=torch_module.tensor([pos_weight], dtype=torch_module.float32, device=device))
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history_rows: list[dict[str, object]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_valid_pr_auc = -np.inf
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for numeric, categorical, labels in train_loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(numeric, categorical)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.shape[0])
            train_loss += float(loss.detach().cpu()) * batch_size
            train_count += batch_size

        valid_scores = predict_scores_torch(torch_module, model, valid_loader, device)
        valid_pr_auc = float(average_precision_score(y_valid, valid_scores))
        average_loss = train_loss / max(train_count, 1)
        improved = valid_pr_auc > best_valid_pr_auc + MIN_DELTA
        if improved:
            best_valid_pr_auc = valid_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": average_loss,
                "valid_pr_auc": valid_pr_auc,
                "best_valid_pr_auc": best_valid_pr_auc,
                "improved": improved,
            }
        )
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    model.to(device)
    history = pd.DataFrame(history_rows)
    summary = {
        "best_epoch": best_epoch,
        "best_valid_pr_auc": best_valid_pr_auc,
        "epochs_trained": int(history["epoch"].max()) if not history.empty else 0,
        "early_stopped": int(history["epoch"].max()) < max_epochs if not history.empty else False,
    }
    return model, history, summary


def result_row(split_name: str, y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, object]:
    metrics = evaluate_predictions(y_true, scores, threshold=threshold)
    return {"model": MODEL_NAME, "split": split_name, **metrics}


def load_lightgbm_comparison(stage8_test_row: dict[str, object]) -> pd.DataFrame:
    stage5_path = TABLES / "stage5_imbalance_results.csv"
    stage6_path = TABLES / "stage6_recommended_thresholds.csv"
    rows = [
        {
            "model": MODEL_NAME,
            "source": "stage8_valid_f1_threshold",
            "threshold": stage8_test_row["threshold"],
            "roc_auc": stage8_test_row["roc_auc"],
            "pr_auc": stage8_test_row["pr_auc"],
            "precision": stage8_test_row["precision"],
            "recall": stage8_test_row["recall"],
            "f1": stage8_test_row["f1"],
            "tn": stage8_test_row["tn"],
            "fp": stage8_test_row["fp"],
            "fn": stage8_test_row["fn"],
            "tp": stage8_test_row["tp"],
        }
    ]
    if stage5_path.exists() and stage6_path.exists():
        stage5 = pd.read_csv(stage5_path)
        stage6 = pd.read_csv(stage6_path)
        for model_name, scenario in LIGHTGBM_COMPARISON:
            ranking = stage5[(stage5["model"].eq(model_name)) & stage5["split"].eq("test")]
            decision = stage6[(stage6["model"].eq(model_name)) & stage6["cost_scenario"].eq(scenario)]
            if ranking.empty or decision.empty:
                continue
            ranking_row = ranking.iloc[0]
            decision_row = decision.iloc[0]
            rows.append(
                {
                    "model": model_name,
                    "source": f"stage6_{scenario}",
                    "threshold": decision_row["selected_threshold"],
                    "roc_auc": ranking_row["roc_auc"],
                    "pr_auc": ranking_row["pr_auc"],
                    "precision": decision_row["test_precision"],
                    "recall": decision_row["test_recall"],
                    "f1": decision_row["test_f1"],
                    "tn": decision_row["test_tn"],
                    "fp": decision_row["test_fp"],
                    "fn": decision_row["test_fn"],
                    "tp": decision_row["test_tp"],
                }
            )
    return pd.DataFrame(rows)


def configure_curves_for_lightgbm(comparison: pd.DataFrame, y_test: pd.Series, X_test: pd.DataFrame) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    if not has_module("joblib"):
        return scores
    import joblib

    for model_name, _ in LIGHTGBM_COMPARISON:
        path = MODELS / f"{model_name}.joblib"
        if path.exists():
            model = joblib.load(path)
            scores[model_name] = model.predict_proba(X_test)[:, 1]
    return scores


def write_figures(
    history: pd.DataFrame,
    y_test: pd.Series,
    stage8_scores: np.ndarray,
    threshold: float,
    comparison: pd.DataFrame,
    lightgbm_scores: dict[str, np.ndarray],
) -> None:
    configure_plotting()
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

    if not history.empty:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(history["epoch"], history["train_loss"], color="#4E79A7", label="train_loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train loss")
        ax2 = ax1.twinx()
        ax2.plot(history["epoch"], history["valid_pr_auc"], color="#59A14F", label="valid_pr_auc")
        ax2.set_ylabel("Valid PR-AUC")
        ax1.set_title("Stage 8 MLP training curve")
        fig.tight_layout()
        fig.savefig(FIGURES / "stage8_training_curve.png", dpi=160)
        plt.close(fig)

    curve_scores = {MODEL_NAME: stage8_scores, **lightgbm_scores}
    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, scores in curve_scores.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax.plot(fpr, tpr, label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_title("Stage 8 test ROC curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage8_test_roc_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, scores in curve_scores.items():
        precision, recall, _ = precision_recall_curve(y_test, scores)
        ax.plot(recall, precision, label=model_name)
    ax.set_title("Stage 8 test precision-recall curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage8_test_pr_curves.png", dpi=160)
    plt.close(fig)

    metrics = ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        comparison.sort_values(metric).plot(kind="barh", x="model", y=metric, legend=False, ax=ax, color="#4E79A7")
        ax.set_title(f"Stage 8 comparison: {metric}")
        ax.set_xlim(0, 1)
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(FIGURES / f"stage8_comparison_{metric}.png", dpi=160)
        plt.close(fig)

    y_pred = (stage8_scores >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("Stage 8 MLP confusion matrix")
    ax.set_xlabel(f"threshold={threshold:.2f}")
    ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
    ax.set_yticks([0, 1], labels=["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111111")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage8_mlp_confusion_matrix.png", dpi=160)
    plt.close(fig)


def write_status_report(results: pd.DataFrame, run_log: pd.DataFrame, comparison: pd.DataFrame) -> Path:
    status_path = REPORTS / "stage8_extension_model_status.md"
    test_results = results[results["split"].eq("test")]
    mlp = test_results.iloc[0]
    best_pr = comparison.sort_values("pr_auc", ascending=False).iloc[0]
    best_f1 = comparison.sort_values("f1", ascending=False).iloc[0]
    mlp_pr_best = bool(best_pr["model"] == MODEL_NAME)
    mlp_f1_best = bool(best_f1["model"] == MODEL_NAME)
    lines = [
        "# 第 8 阶段：扩展模型实验状态报告",
        "",
        "## 运行结果",
        "",
        "- 扩展模型：PyTorch MLP + categorical embeddings。",
        f"- 实际设备：`{run_log.iloc[0]['actual_device']}`。",
        f"- 最佳 epoch：`{int(run_log.iloc[0]['best_epoch'])}`；训练 epoch 数：`{int(run_log.iloc[0]['epochs_trained'])}`。",
        f"- valid F1 最优阈值：`{float(mlp['threshold']):.2f}`。",
        f"- Test ROC-AUC：`{float(mlp['roc_auc']):.4f}`；PR-AUC：`{float(mlp['pr_auc']):.4f}`；F1：`{float(mlp['f1']):.4f}`。",
        "",
        "## 与 LightGBM 对比",
        "",
        f"- PR-AUC 最优：`{best_pr['model']}` = {float(best_pr['pr_auc']):.4f}。",
        f"- F1 最优：`{best_f1['model']}` = {float(best_f1['f1']):.4f}。",
    ]
    if mlp_pr_best or mlp_f1_best:
        lines.append("- 扩展模型在部分指标上有竞争力，但仍需更多调参和校准后才能进入主线。")
    else:
        lines.append("- 扩展模型未超过 LightGBM 主线，说明该表格任务上 GBDT 仍然更稳健。")
    lines.extend(
        [
            "- 深度模型可能受训练预算、超参数、类别嵌入维度和不平衡处理方式影响；本阶段只作为扩展探索。",
            "",
            "## 验收检查",
            "",
            "- [x] 已训练 PyTorch MLP+Embedding 扩展模型。",
            "- [x] 已输出 ROC-AUC、PR-AUC、Precision、Recall、F1 和混淆矩阵。",
            "- [x] 已与 Stage 5/6 LightGBM 主线模型对比。",
            "- [x] scaler、类别映射、pos_weight 只使用 train split；阈值选择只使用 valid split。",
            "- [x] 模型文件仅保存到本地 `models/`，不提交到 Git。",
            "",
            "## 复现命令",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe .\\src\\run_stage8_extension_models.py",
            "```",
        ]
    )
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status_path


def update_readme_checklist() -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    checked = "- [x] 已完成扩展模型实验。"
    unchecked = "- [ ] 已完成扩展模型实验。"
    if unchecked in text:
        text = text.replace(unchecked, checked)
    elif checked not in text:
        anchor = "- [x] 已完成单样本解释案例。"
        text = text.replace(anchor, f"{anchor}\n{checked}")
    readme.write_text(text, encoding="utf-8")


def run_stage8_extension_models() -> dict[str, Path]:
    ensure_project_dirs()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    torch = require_torch()
    set_reproducible_seed(torch)

    from sklearn.preprocessing import StandardScaler

    start_time = time.perf_counter()
    feature_path = ensure_feature_file()
    df = pd.read_csv(feature_path)
    X, y, split, numeric_features, categorical_features = split_feature_target(df)
    train_mask = split.eq("train")
    valid_mask = split.eq("valid")
    test_mask = split.eq("test")
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    scaler = StandardScaler()
    scaler.fit(X_train[numeric_features])
    category_maps = fit_category_maps(X_train, categorical_features)
    category_cardinalities = [len(category_maps[column]) + 1 for column in categorical_features]
    embedding_dims = [embedding_dim(cardinality) for cardinality in category_cardinalities]

    train_data = encode_split(X_train, y_train, numeric_features, categorical_features, scaler, category_maps)
    valid_data = encode_split(X_valid, y_valid, numeric_features, categorical_features, scaler, category_maps)
    test_data = encode_split(X_test, y_test, numeric_features, categorical_features, scaler, category_maps)

    batch_size = int(os.environ.get("STAGE8_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    max_epochs = int(os.environ.get("STAGE8_MAX_EPOCHS", DEFAULT_MAX_EPOCHS))
    patience = int(os.environ.get("STAGE8_PATIENCE", DEFAULT_PATIENCE))
    train_loader = make_loader(torch, train_data, batch_size=batch_size, shuffle=True)
    valid_loader = make_loader(torch, valid_data, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(torch, test_data, batch_size=batch_size, shuffle=False)

    requested_device = "cuda_if_available"
    actual_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(actual_device)
    positives = int(y_train.sum())
    negatives = int(y_train.shape[0] - positives)
    pos_weight = negatives / positives

    ModelClass = build_model_class(torch)
    model = ModelClass(len(numeric_features), category_cardinalities, embedding_dims)
    model, history, training_summary = train_model(
        torch,
        model,
        train_loader,
        valid_loader,
        y_valid,
        device,
        pos_weight=pos_weight,
        max_epochs=max_epochs,
        patience=patience,
    )

    valid_scores = predict_scores_torch(torch, model, valid_loader, device)
    test_scores = predict_scores_torch(torch, model, test_loader, device)
    selected_threshold, threshold_search = choose_threshold(y_valid, valid_scores)
    result_rows = [
        result_row("valid", y_valid, valid_scores, selected_threshold),
        result_row("test", y_test, test_scores, selected_threshold),
    ]
    results = pd.DataFrame(result_rows)
    test_row = results[results["split"].eq("test")].iloc[0].to_dict()
    comparison = load_lightgbm_comparison(test_row)
    lightgbm_scores = configure_curves_for_lightgbm(comparison, y_test, X_test)

    model_path = MODELS / f"{MODEL_NAME}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "category_cardinalities": category_cardinalities,
            "embedding_dims": embedding_dims,
            "hidden_layers": HIDDEN_LAYERS,
            "dropout": DROPOUT,
            "selected_threshold": selected_threshold,
        },
        model_path,
    )

    elapsed = time.perf_counter() - start_time
    run_log = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "requested_device": requested_device,
                "actual_device": actual_device,
                "status": "success",
                "best_epoch": training_summary["best_epoch"],
                "best_valid_pr_auc": training_summary["best_valid_pr_auc"],
                "epochs_trained": training_summary["epochs_trained"],
                "early_stopped": training_summary["early_stopped"],
                "train_seconds": elapsed,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "selected_threshold": selected_threshold,
                "train_rows": int(y_train.shape[0]),
                "positive_count": positives,
                "negative_count": negatives,
                "pos_weight": pos_weight,
            }
        ]
    )
    params = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "numeric_feature_count": len(numeric_features),
                "categorical_feature_count": len(categorical_features),
                "hidden_layers": json.dumps(HIDDEN_LAYERS),
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "embedding_cardinalities": json.dumps(category_cardinalities),
                "embedding_dims": json.dumps(embedding_dims),
                "scaler_fit_split": "train",
                "category_mapping_fit_split": "train",
                "threshold_selection_split": "valid",
            }
        ]
    )

    outputs = {
        "stage8_results": TABLES / "stage8_extension_model_results.csv",
        "stage8_run_log": TABLES / "stage8_extension_model_run_log.csv",
        "stage8_params": TABLES / "stage8_extension_model_params.csv",
        "stage8_comparison": TABLES / "stage8_lightgbm_comparison.csv",
        "stage8_threshold_search": TABLES / "stage8_threshold_search.csv",
        "stage8_history": TABLES / "stage8_training_history.csv",
    }
    results.to_csv(outputs["stage8_results"], index=False)
    run_log.to_csv(outputs["stage8_run_log"], index=False)
    params.to_csv(outputs["stage8_params"], index=False)
    comparison.to_csv(outputs["stage8_comparison"], index=False)
    threshold_search.to_csv(outputs["stage8_threshold_search"], index=False)
    history.to_csv(outputs["stage8_history"], index=False)
    write_figures(history, y_test, test_scores, selected_threshold, comparison, lightgbm_scores)
    outputs["status"] = write_status_report(results, run_log, comparison)
    update_readme_checklist()
    return outputs


def main() -> int:
    try:
        outputs = run_stage8_extension_models()
    except Exception as exc:  # noqa: BLE001
        print(f"Stage 8 extension model experiment failed: {type(exc).__name__}: {exc}")
        return 1

    print("Stage 8 extension model experiment complete.")
    for key, path in outputs.items():
        print(f"- {key}: {rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
