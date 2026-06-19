from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from features import ID_COLUMNS, SPLIT, TARGET
from project_paths import FIGURES, MODELS, REPORTS, TABLES, ensure_project_dirs, rel
from run_stage4_baselines import (
    ensure_feature_file,
    make_preprocessor,
    predict_scores,
    split_feature_target,
    training_device,
)
from train_baseline import configure_lightgbm_gpu_cache, evaluate_predictions, has_module


RANDOM_STATE = 42
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
MODEL_FAMILIES = ["lightgbm", "xgboost"]
FIT_STRATEGIES = ["raw", "weighted", "random_under_sample", "smote"]
ALL_STRATEGIES = [*FIT_STRATEGIES, "threshold_moving"]


@dataclass(frozen=True)
class Stage5Spec:
    model_family: str
    strategy: str
    requested_device: str
    actual_device: str
    model: object
    params: dict[str, object]
    uses_scale_pos_weight: bool
    scale_pos_weight: float | None
    uses_sampling: bool
    sampler_name: str

    @property
    def name(self) -> str:
        return f"stage5_{self.model_family}_{self.strategy}_{self.actual_device}"


def class_balance(y: pd.Series) -> tuple[int, int, float]:
    positives = int(y.sum())
    negatives = int(y.shape[0] - positives)
    if positives == 0:
        raise ValueError("Training target has no positive samples; scale_pos_weight cannot be computed.")
    return positives, negatives, negatives / positives


def build_sampler(strategy: str):
    if strategy == "random_under_sample":
        from imblearn.under_sampling import RandomUnderSampler

        return RandomUnderSampler(sampling_strategy=1.0, random_state=RANDOM_STATE), "RandomUnderSampler"
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE

        return SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=RANDOM_STATE), "SMOTE"
    return None, ""


def build_pipeline(numeric_features: list[str], categorical_features: list[str], sampler: object | None, estimator: object):
    preprocessor = make_preprocessor(numeric_features, categorical_features, scale_numeric=False)
    if sampler is None:
        from sklearn.pipeline import Pipeline

        return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    from imblearn.pipeline import Pipeline

    return Pipeline([("preprocessor", preprocessor), ("sampler", sampler), ("model", estimator)])


def build_lightgbm_estimator(actual_device: str, strategy: str, scale_pos_weight: float | None):
    from lightgbm import LGBMClassifier

    params: dict[str, object] = {
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 48,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if actual_device == "gpu":
        configure_lightgbm_gpu_cache(FIGURES)
        params.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
    if strategy == "weighted" and scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return LGBMClassifier(**params), params


def build_xgboost_estimator(actual_device: str, strategy: str, scale_pos_weight: float | None):
    from xgboost import XGBClassifier

    params: dict[str, object] = {
        "n_estimators": 350,
        "learning_rate": 0.04,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": 1 if actual_device == "cuda" else -1,
        "tree_method": "hist",
    }
    if actual_device == "cuda":
        params["device"] = "cuda"
    if strategy == "weighted" and scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**params), params


def build_spec(
    model_family: str,
    strategy: str,
    device_request: str,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    requested_device: str | None = None,
) -> Stage5Spec | None:
    if model_family == "lightgbm" and not has_module("lightgbm"):
        return None
    if model_family == "xgboost" and not has_module("xgboost"):
        return None
    if strategy in {"random_under_sample", "smote"} and not has_module("imblearn"):
        return None

    _, _, ratio = class_balance(y_train)
    uses_scale_pos_weight = strategy == "weighted"
    scale_pos_weight = ratio if uses_scale_pos_weight else None
    sampler, sampler_name = build_sampler(strategy)
    uses_sampling = sampler is not None

    if model_family == "lightgbm":
        actual_device = "gpu" if device_request == "gpu" else "cpu"
        requested = requested_device or actual_device
        estimator, params = build_lightgbm_estimator(actual_device, strategy, scale_pos_weight)
    elif model_family == "xgboost":
        actual_device = "cuda" if device_request == "gpu" else "cpu"
        requested = requested_device or actual_device
        estimator, params = build_xgboost_estimator(actual_device, strategy, scale_pos_weight)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    model = build_pipeline(numeric_features, categorical_features, sampler, estimator)
    return Stage5Spec(
        model_family=model_family,
        strategy=strategy,
        requested_device=requested,
        actual_device=actual_device,
        model=model,
        params=params,
        uses_scale_pos_weight=uses_scale_pos_weight,
        scale_pos_weight=scale_pos_weight,
        uses_sampling=uses_sampling,
        sampler_name=sampler_name,
    )


def run_log_row(
    spec: Stage5Spec | None,
    model_family: str,
    strategy: str,
    requested_device: str,
    actual_device: str,
    status: str,
    fallback_from: str = "",
    fallback_reason: str = "",
    fit_rows: int = 0,
    positives: int = 0,
    negatives: int = 0,
    scale_pos_weight: float | None = None,
    sampler_name: str = "",
) -> dict[str, object]:
    return {
        "model": spec.name if spec is not None else f"stage5_{model_family}_{strategy}_missing",
        "model_family": model_family,
        "strategy": strategy,
        "requested_device": requested_device,
        "actual_device": actual_device,
        "status": status,
        "fallback_from": fallback_from,
        "fallback_reason": fallback_reason,
        "fit_rows_before_sampling": fit_rows,
        "positive_count_before_sampling": positives,
        "negative_count_before_sampling": negatives,
        "scale_pos_weight": scale_pos_weight if scale_pos_weight is not None else "",
        "sampler": sampler_name,
    }


def fit_with_fallback(
    spec: Stage5Spec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[Stage5Spec | None, dict[str, object]]:
    positives, negatives, ratio = class_balance(y_train)
    try:
        spec.model.fit(X_train, y_train)
        return spec, run_log_row(
            spec,
            spec.model_family,
            spec.strategy,
            spec.requested_device,
            spec.actual_device,
            "success",
            fit_rows=int(y_train.shape[0]),
            positives=positives,
            negatives=negatives,
            scale_pos_weight=ratio if spec.uses_scale_pos_weight else None,
            sampler_name=spec.sampler_name,
        )
    except Exception as exc:  # noqa: BLE001
        if spec.actual_device not in {"gpu", "cuda"}:
            return None, run_log_row(
                spec,
                spec.model_family,
                spec.strategy,
                spec.requested_device,
                spec.actual_device,
                "failed",
                fallback_reason=f"{type(exc).__name__}: {exc}",
                fit_rows=int(y_train.shape[0]),
                positives=positives,
                negatives=negatives,
                scale_pos_weight=ratio if spec.uses_scale_pos_weight else None,
                sampler_name=spec.sampler_name,
            )

        fallback = build_spec(
            spec.model_family,
            spec.strategy,
            "cpu",
            y_train,
            numeric_features,
            categorical_features,
            requested_device=spec.requested_device,
        )
        if fallback is None:
            return None, run_log_row(
                spec,
                spec.model_family,
                spec.strategy,
                spec.requested_device,
                "cpu",
                "failed",
                fallback_from=spec.name,
                fallback_reason=f"{type(exc).__name__}: {exc}; CPU fallback spec could not be built.",
                fit_rows=int(y_train.shape[0]),
                positives=positives,
                negatives=negatives,
                scale_pos_weight=ratio if spec.uses_scale_pos_weight else None,
                sampler_name=spec.sampler_name,
            )

        first_reason = f"{type(exc).__name__}: {exc}"
        try:
            fallback.model.fit(X_train, y_train)
            return fallback, run_log_row(
                fallback,
                fallback.model_family,
                fallback.strategy,
                spec.requested_device,
                fallback.actual_device,
                "fallback_success",
                fallback_from=spec.name,
                fallback_reason=first_reason,
                fit_rows=int(y_train.shape[0]),
                positives=positives,
                negatives=negatives,
                scale_pos_weight=ratio if fallback.uses_scale_pos_weight else None,
                sampler_name=fallback.sampler_name,
            )
        except Exception as fallback_exc:  # noqa: BLE001
            return None, run_log_row(
                fallback,
                fallback.model_family,
                fallback.strategy,
                spec.requested_device,
                fallback.actual_device,
                "failed",
                fallback_from=spec.name,
                fallback_reason=f"{first_reason}; CPU fallback failed with {type(fallback_exc).__name__}: {fallback_exc}",
                fit_rows=int(y_train.shape[0]),
                positives=positives,
                negatives=negatives,
                scale_pos_weight=ratio if fallback.uses_scale_pos_weight else None,
                sampler_name=fallback.sampler_name,
            )


def choose_threshold(y_valid: pd.Series, scores: np.ndarray, model_family: str, source_model: str) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in THRESHOLD_GRID:
        metrics = evaluate_predictions(y_valid, scores, threshold=float(threshold))
        rows.append(
            {
                "model_family": model_family,
                "source_model": source_model,
                "selection_split": "valid",
                **metrics,
            }
        )
    search = pd.DataFrame(rows)
    best = search.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return float(best["threshold"]), search


def append_metrics(
    rows: list[dict[str, object]],
    model_name: str,
    model_family: str,
    strategy: str,
    actual_device: str,
    split_name: str,
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    metrics = evaluate_predictions(y_true, scores, threshold=threshold)
    row = {
        "model": model_name,
        "model_family": model_family,
        "strategy": strategy,
        "actual_device": actual_device,
        "split": split_name,
        **metrics,
    }
    rows.append(row)
    return row


def write_predictions_for_figures(
    rows: list[pd.DataFrame],
    model_name: str,
    model_family: str,
    strategy: str,
    actual_device: str,
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> None:
    rows.append(
        pd.DataFrame(
            {
                "model": model_name,
                "model_family": model_family,
                "strategy": strategy,
                "actual_device": actual_device,
                "threshold": threshold,
                "y_true": y_true.to_numpy(),
                "y_score": scores,
                "y_pred": (scores >= threshold).astype(int),
            }
        )
    )


def strategy_param_row(spec: Stage5Spec, threshold: float) -> dict[str, object]:
    return {
        "model": spec.name,
        "model_family": spec.model_family,
        "strategy": spec.strategy,
        "requested_device": spec.requested_device,
        "actual_device": spec.actual_device,
        "random_state": RANDOM_STATE,
        "uses_scale_pos_weight": spec.uses_scale_pos_weight,
        "scale_pos_weight": spec.scale_pos_weight if spec.scale_pos_weight is not None else "",
        "uses_sampling": spec.uses_sampling,
        "sampler": spec.sampler_name,
        "threshold": threshold,
        "params_json": json.dumps(spec.params, ensure_ascii=False, sort_keys=True),
    }


def threshold_param_row(raw_spec: Stage5Spec, threshold: float, model_name: str) -> dict[str, object]:
    return {
        "model": model_name,
        "model_family": raw_spec.model_family,
        "strategy": "threshold_moving",
        "requested_device": raw_spec.requested_device,
        "actual_device": raw_spec.actual_device,
        "random_state": RANDOM_STATE,
        "uses_scale_pos_weight": False,
        "scale_pos_weight": "",
        "uses_sampling": False,
        "sampler": "",
        "threshold": threshold,
        "params_json": json.dumps(raw_spec.params, ensure_ascii=False, sort_keys=True),
    }


def build_business_impact(results: pd.DataFrame) -> pd.DataFrame:
    test_results = results[results["split"].eq("test")].copy()
    rows: list[dict[str, object]] = []
    for model_family, family_rows in test_results.groupby("model_family"):
        raw_rows = family_rows[family_rows["strategy"].eq("raw")]
        if raw_rows.empty:
            continue
        raw = raw_rows.iloc[0]
        for _, row in family_rows.iterrows():
            rows.append(
                {
                    "model_family": model_family,
                    "model": row["model"],
                    "strategy": row["strategy"],
                    "compared_to": raw["model"],
                    "threshold": row["threshold"],
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "tp_gain": int(row["tp"] - raw["tp"]),
                    "fp_gain": int(row["fp"] - raw["fp"]),
                    "fn_reduction": int(raw["fn"] - row["fn"]),
                    "recall_delta": float(row["recall"] - raw["recall"]),
                    "precision_delta": float(row["precision"] - raw["precision"]),
                    "f1_delta": float(row["f1"] - raw["f1"]),
                }
            )
    return pd.DataFrame(rows)


def configure_matplotlib() -> None:
    os.environ.setdefault("WINDIR", r"C:\Windows")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", str(FIGURES.parent / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)


def safe_file_part(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_").replace("\\", "_")


def write_stage5_figures(results: pd.DataFrame, predictions: pd.DataFrame) -> None:
    if not has_module("matplotlib") or predictions.empty:
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, precision_recall_curve

    FIGURES.mkdir(parents=True, exist_ok=True)
    test_results = results[results["split"].eq("test")].copy()
    test_results["label"] = test_results["model_family"] + " / " + test_results["strategy"]

    for metric in ["roc_auc", "pr_auc", "recall", "precision", "f1"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        test_results.sort_values(metric).plot(kind="barh", x="label", y=metric, legend=False, ax=ax, color="#4E79A7")
        ax.set_title(f"Stage 5 test {metric}")
        ax.set_xlim(0, 1)
        ax.set_xlabel(metric)
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(FIGURES / f"stage5_{metric}.png", dpi=160)
        plt.close(fig)

    for model_family, family_predictions in predictions[predictions["strategy"].ne("threshold_moving")].groupby("model_family"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for strategy, group in family_predictions.groupby("strategy"):
            precision, recall, _ = precision_recall_curve(group["y_true"], group["y_score"])
            ax.plot(recall, precision, label=strategy)
        ax.set_title(f"Stage 5 {model_family} test precision-recall curves")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / f"stage5_{model_family}_test_pr_curves.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_family, family_rows in test_results.groupby("model_family"):
        ax.scatter(family_rows["fp"], family_rows["recall"], label=model_family, s=60)
        for _, row in family_rows.iterrows():
            ax.annotate(row["strategy"], (row["fp"], row["recall"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_title("Stage 5 recall vs false positives")
    ax.set_xlabel("False positives on test set")
    ax.set_ylabel("Recall")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage5_recall_false_positive_tradeoff.png", dpi=160)
    plt.close(fig)

    for (model_family, strategy, actual_device), group in predictions.groupby(["model_family", "strategy", "actual_device"]):
        cm = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        image = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{model_family} {strategy} confusion matrix")
        ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
        ax.set_yticks([0, 1], labels=["true 0", "true 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111111")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        filename = f"stage5_{safe_file_part(model_family)}_{safe_file_part(strategy)}_{safe_file_part(actual_device)}_confusion_matrix.png"
        fig.savefig(FIGURES / filename, dpi=160)
        plt.close(fig)


def write_status_report(
    results: pd.DataFrame,
    run_log: pd.DataFrame,
    impact: pd.DataFrame,
    split_summary: pd.DataFrame,
    threshold_search: pd.DataFrame,
) -> Path:
    status_path = REPORTS / "stage5_imbalance_status.md"
    test_results = results[results["split"].eq("test")].copy()
    lines = [
        "# 第 5 阶段：不平衡处理策略对比状态报告",
        "",
        "## 运行结果",
        "",
        "- 数据集：`home_credit`。",
        "- 输入特征数据：`data/processed/home_credit_features.csv`。",
        "- 固定划分：沿用 `split`，未重新抽样。",
        f"- 训练行数：{int(split_summary.loc[split_summary['split'].eq('train'), 'rows'].iloc[0])}。",
        f"- 验证行数：{int(split_summary.loc[split_summary['split'].eq('valid'), 'rows'].iloc[0])}。",
        f"- 测试行数：{int(split_summary.loc[split_summary['split'].eq('test'), 'rows'].iloc[0])}。",
        "- 模型范围：LightGBM + XGBoost。",
        "- 策略范围：raw、weighted、random_under_sample、smote、threshold_moving。",
        "",
        "## 模型与设备",
        "",
    ]
    for _, row in run_log.iterrows():
        lines.append(
            f"- `{row['model']}`：strategy `{row['strategy']}`，requested `{row['requested_device']}`，actual `{row['actual_device']}`，status `{row['status']}`。"
        )
        reason = row.get("fallback_reason", "")
        if isinstance(reason, str) and reason.strip():
            lines.append(f"  - 回退原因：{reason.splitlines()[0]}")

    lines.extend(["", "## 测试集最佳结果", ""])
    if not test_results.empty:
        trained_results = test_results[test_results["strategy"].ne("threshold_moving")]
        if not trained_results.empty:
            best_pr = trained_results.sort_values("pr_auc", ascending=False).iloc[0]
            lines.append(
                f"- Test pr_auc 最优训练策略：`{best_pr['model']}`，strategy `{best_pr['strategy']}` = {best_pr['pr_auc']:.4f}。"
            )
        for metric in ["f1", "recall"]:
            best = test_results.sort_values(metric, ascending=False).iloc[0]
            lines.append(
                f"- Test {metric} 最优：`{best['model']}`，strategy `{best['strategy']}` = {best[metric]:.4f}。"
            )
        lines.append("- 说明：threshold_moving 不改变排序分数，因此不单独视为 PR-AUC 提升策略。")

    lines.extend(["", "## 业务影响解读", ""])
    if not impact.empty:
        for model_family, family_impact in impact.groupby("model_family"):
            best_f1 = family_impact.sort_values("f1", ascending=False).iloc[0]
            best_recall = family_impact.sort_values("recall", ascending=False).iloc[0]
            lines.append(
                f"- `{model_family}` F1 最优策略：`{best_f1['strategy']}`，相对 raw 多识别 {int(best_f1['tp_gain'])} 个违约样本，新增 {int(best_f1['fp_gain'])} 个误报。"
            )
            lines.append(
                f"- `{model_family}` Recall 最优策略：`{best_recall['strategy']}`，相对 raw 多识别 {int(best_recall['tp_gain'])} 个违约样本，新增 {int(best_recall['fp_gain'])} 个误报。"
            )

    lines.extend(["", "## 阈值移动", ""])
    if not threshold_search.empty:
        selected = threshold_search.sort_values(["model_family", "f1", "recall", "precision"], ascending=[True, False, False, False])
        selected = selected.groupby("model_family", as_index=False).head(1)
        for _, row in selected.iterrows():
            lines.append(
                f"- `{row['model_family']}` 在 valid split 上选择阈值 `{row['threshold']:.2f}`，valid F1 = {row['f1']:.4f}。"
            )

    lines.extend(
        [
            "",
            "## 泄漏控制",
            "",
            "- `TARGET`、`SK_ID_CURR`、`split` 未进入模型特征矩阵。",
            "- 编码器、采样器、SMOTE 和模型均只在 train split 拟合。",
            "- `scale_pos_weight` 只使用 train split 的正负样本比例计算。",
            "- threshold_moving 只用 valid split 选择阈值，test split 只做最终评估。",
            "- SMOTE 作为对照实验保留；其合成样本不一定具有真实业务含义，后续解释时需谨慎。",
            "",
            "## 验收检查",
            "",
            "- [x] LightGBM 和 XGBoost 均完成不平衡策略对比。",
            "- [x] 输出 PR-AUC、Recall、Precision、F1 和混淆矩阵。",
            "- [x] 输出相对 raw 的 TP/FP/FN 业务影响表。",
            "- [x] 输出 PR 曲线、指标对比图和召回率-误报数量对比图。",
            "- [x] GPU 失败时记录 CPU fallback，不伪装设备。",
            "",
            "## 复现命令",
            "",
            "```powershell",
            '$env:TRAIN_DEVICE="gpu"',
            ".\\.venv\\Scripts\\python.exe .\\src\\run_stage5_imbalance.py",
            "```",
        ]
    )
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status_path


def update_readme_checklist() -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    replacements = {
        "- [ ] 已完成不平衡策略对比。": "- [x] 已完成不平衡策略对比。",
        "- [ ] 已整理不平衡策略对比表。": "- [x] 已整理不平衡策略对比表。",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    readme.write_text(text, encoding="utf-8")


def run_stage5_imbalance() -> dict[str, Path]:
    ensure_project_dirs()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if not has_module("sklearn"):
        raise RuntimeError("scikit-learn is required for stage 5 imbalance experiments.")
    if not has_module("imblearn"):
        raise RuntimeError("imbalanced-learn is required for random under-sampling and SMOTE.")

    feature_path = ensure_feature_file()
    df = pd.read_csv(feature_path)
    X, y, split, numeric_features, categorical_features = split_feature_target(df)
    train_mask = split.eq("train")
    valid_mask = split.eq("valid")
    test_mask = split.eq("test")

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    device = training_device()
    result_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    threshold_rows: list[pd.DataFrame] = []
    raw_specs: dict[str, Stage5Spec] = {}
    raw_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    import joblib

    for model_family in MODEL_FAMILIES:
        if model_family == "lightgbm" and not has_module("lightgbm"):
            run_rows.append(
                run_log_row(None, model_family, "all", "gpu" if device == "gpu" else "cpu", "", "missing_dependency")
            )
            continue
        if model_family == "xgboost" and not has_module("xgboost"):
            run_rows.append(
                run_log_row(None, model_family, "all", "cuda" if device == "gpu" else "cpu", "", "missing_dependency")
            )
            continue

        for strategy in FIT_STRATEGIES:
            spec = build_spec(model_family, strategy, device, y_train, numeric_features, categorical_features)
            if spec is None:
                run_rows.append(
                    run_log_row(None, model_family, strategy, "gpu" if device == "gpu" else "cpu", "", "missing_dependency")
                )
                continue
            fitted_spec, run_row = fit_with_fallback(spec, X_train, y_train, numeric_features, categorical_features)
            run_rows.append(run_row)
            if fitted_spec is None:
                continue

            valid_scores = predict_scores(fitted_spec.model, X_valid)
            test_scores = predict_scores(fitted_spec.model, X_test)
            append_metrics(
                result_rows,
                fitted_spec.name,
                fitted_spec.model_family,
                fitted_spec.strategy,
                fitted_spec.actual_device,
                "valid",
                y_valid,
                valid_scores,
                threshold=0.5,
            )
            append_metrics(
                result_rows,
                fitted_spec.name,
                fitted_spec.model_family,
                fitted_spec.strategy,
                fitted_spec.actual_device,
                "test",
                y_test,
                test_scores,
                threshold=0.5,
            )
            write_predictions_for_figures(
                prediction_rows,
                fitted_spec.name,
                fitted_spec.model_family,
                fitted_spec.strategy,
                fitted_spec.actual_device,
                y_test,
                test_scores,
                threshold=0.5,
            )
            param_rows.append(strategy_param_row(fitted_spec, threshold=0.5))
            joblib.dump(fitted_spec.model, MODELS / f"{fitted_spec.name}.joblib")

            if strategy == "raw":
                raw_specs[model_family] = fitted_spec
                raw_scores[model_family] = (valid_scores, test_scores)

        raw_spec = raw_specs.get(model_family)
        if raw_spec is None or model_family not in raw_scores:
            continue

        valid_scores, test_scores = raw_scores[model_family]
        threshold, search = choose_threshold(y_valid, valid_scores, model_family, raw_spec.name)
        threshold_rows.append(search)
        threshold_model_name = f"stage5_{model_family}_threshold_moving_{raw_spec.actual_device}"
        append_metrics(
            result_rows,
            threshold_model_name,
            model_family,
            "threshold_moving",
            raw_spec.actual_device,
            "valid",
            y_valid,
            valid_scores,
            threshold=threshold,
        )
        append_metrics(
            result_rows,
            threshold_model_name,
            model_family,
            "threshold_moving",
            raw_spec.actual_device,
            "test",
            y_test,
            test_scores,
            threshold=threshold,
        )
        write_predictions_for_figures(
            prediction_rows,
            threshold_model_name,
            model_family,
            "threshold_moving",
            raw_spec.actual_device,
            y_test,
            test_scores,
            threshold=threshold,
        )
        run_rows.append(
            run_log_row(
                raw_spec,
                model_family,
                "threshold_moving",
                raw_spec.requested_device,
                raw_spec.actual_device,
                "derived_from_raw",
                fit_rows=int(y_train.shape[0]),
                positives=int(y_train.sum()),
                negatives=int(y_train.shape[0] - y_train.sum()),
            )
            | {"model": threshold_model_name}
        )
        param_rows.append(threshold_param_row(raw_spec, threshold=threshold, model_name=threshold_model_name))
        joblib.dump(raw_spec.model, MODELS / f"{threshold_model_name}.joblib")

    results = pd.DataFrame(result_rows)
    run_log = pd.DataFrame(run_rows)
    params = pd.DataFrame(param_rows)
    threshold_search = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    impact = build_business_impact(results) if not results.empty else pd.DataFrame()
    split_summary = (
        df.groupby(SPLIT)
        .size()
        .rename("rows")
        .reset_index()
        .assign(
            numeric_feature_count=len(numeric_features),
            categorical_feature_count=len(categorical_features),
            raw_feature_count=len(numeric_features) + len(categorical_features),
            target_column=TARGET,
            id_columns="; ".join(ID_COLUMNS),
        )
    )

    outputs = {
        "stage5_results": TABLES / "stage5_imbalance_results.csv",
        "stage5_run_log": TABLES / "stage5_strategy_run_log.csv",
        "stage5_params": TABLES / "stage5_strategy_params.csv",
        "stage5_threshold_search": TABLES / "stage5_threshold_search.csv",
        "stage5_business_impact": TABLES / "stage5_business_impact.csv",
        "stage5_split_summary": TABLES / "stage5_feature_matrix_summary.csv",
    }
    results.to_csv(outputs["stage5_results"], index=False)
    run_log.to_csv(outputs["stage5_run_log"], index=False)
    params.to_csv(outputs["stage5_params"], index=False)
    threshold_search.to_csv(outputs["stage5_threshold_search"], index=False)
    impact.to_csv(outputs["stage5_business_impact"], index=False)
    split_summary.to_csv(outputs["stage5_split_summary"], index=False)
    write_stage5_figures(results, predictions)
    outputs["status"] = write_status_report(results, run_log, impact, split_summary, threshold_search)
    update_readme_checklist()
    return outputs


def main() -> int:
    try:
        outputs = run_stage5_imbalance()
    except Exception as exc:  # noqa: BLE001
        print(f"Stage 5 imbalance experiments failed: {type(exc).__name__}: {exc}")
        return 1

    print("Stage 5 imbalance experiments complete.")
    for key, path in outputs.items():
        print(f"- {key}: {rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
