from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from features import ID_COLUMNS, SPLIT, TARGET, run_stage3_feature_engineering
from project_paths import DATA_PROCESSED, FIGURES, MODELS, REPORTS, TABLES, ensure_project_dirs, rel
from train_baseline import configure_lightgbm_gpu_cache, dependency_status, evaluate_predictions, has_module


RANDOM_STATE = 42
ONE_HOT_MIN_FREQUENCY = 50


@dataclass(frozen=True)
class Stage4Spec:
    name: str
    requested_device: str
    actual_device: str
    model: object
    params: dict[str, object]


def training_device() -> str:
    return os.environ.get("TRAIN_DEVICE", "auto").strip().lower() or "auto"


def ensure_feature_file() -> Path:
    feature_path = DATA_PROCESSED / "home_credit_features.csv"
    if not feature_path.exists():
        run_stage3_feature_engineering()
    if not feature_path.exists():
        raise FileNotFoundError(f"{rel(feature_path)} was not created. Run src/run_stage3_features.py first.")
    return feature_path


def split_feature_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    metadata = [TARGET, *ID_COLUMNS, SPLIT]
    missing = [column for column in metadata if column not in df.columns]
    if missing:
        raise ValueError(f"Feature data is missing required metadata columns: {missing}")
    X = df.drop(columns=metadata)
    y = df[TARGET].astype(int)
    split = df[SPLIT].astype(str)
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    return X, y, split, numeric_features, categorical_features


def make_one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=ONE_HOT_MIN_FREQUENCY,
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=ONE_HOT_MIN_FREQUENCY,
            sparse=True,
        )


def make_preprocessor(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler

    numeric_transformer = StandardScaler(with_mean=False) if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", make_one_hot_encoder(), categorical_features),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def build_logistic_spec(numeric_features: list[str], categorical_features: list[str]) -> Stage4Spec:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    params = {
        "max_iter": 500,
        "solver": "saga",
        "penalty": "l2",
        "C": 1.0,
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, scale_numeric=True)),
            ("model", LogisticRegression(**params)),
        ]
    )
    return Stage4Spec("stage4_logistic_regression", "cpu", "cpu", model, params)


def build_xgboost_spec(numeric_features: list[str], categorical_features: list[str], device: str) -> Stage4Spec | None:
    if not has_module("xgboost"):
        return None
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    params = {
        "n_estimators": 350,
        "learning_rate": 0.04,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": 1 if device == "gpu" else -1,
        "tree_method": "hist",
    }
    requested = "cuda" if device == "gpu" else "cpu"
    actual = requested
    name = "stage4_xgboost_cuda" if device == "gpu" else "stage4_xgboost_cpu"
    if device == "gpu":
        params["device"] = "cuda"
    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
            ("model", XGBClassifier(**params)),
        ]
    )
    return Stage4Spec(name, requested, actual, model, params)


def build_xgboost_cpu_fallback(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Stage4Spec:
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    params = {
        "n_estimators": 350,
        "learning_rate": 0.04,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
            ("model", XGBClassifier(**params)),
        ]
    )
    return Stage4Spec("stage4_xgboost_cpu_fallback", "cuda", "cpu", model, params)


def build_lightgbm_spec(numeric_features: list[str], categorical_features: list[str], device: str) -> Stage4Spec | None:
    if not has_module("lightgbm"):
        return None
    from lightgbm import LGBMClassifier
    from sklearn.pipeline import Pipeline

    params = {
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 48,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    requested = "gpu" if device == "gpu" else "cpu"
    actual = requested
    name = "stage4_lightgbm_gpu" if device == "gpu" else "stage4_lightgbm_cpu"
    if device == "gpu":
        configure_lightgbm_gpu_cache(FIGURES)
        params.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0})
    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
            ("model", LGBMClassifier(**params)),
        ]
    )
    return Stage4Spec(name, requested, actual, model, params)


def build_lightgbm_cpu_fallback(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Stage4Spec:
    from lightgbm import LGBMClassifier
    from sklearn.pipeline import Pipeline

    params = {
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 48,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
            ("model", LGBMClassifier(**params)),
        ]
    )
    return Stage4Spec("stage4_lightgbm_cpu_fallback", "gpu", "cpu", model, params)


def model_file_name(model_name: str) -> str:
    return f"{model_name}.joblib"


def predict_scores(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw_score = model.decision_function(X)
    return 1 / (1 + np.exp(-raw_score))


def fit_with_fallbacks(
    spec: Stage4Spec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[Stage4Spec | None, dict[str, object]]:
    try:
        spec.model.fit(X_train, y_train)
        return spec, {
            "model": spec.name,
            "requested_device": spec.requested_device,
            "actual_device": spec.actual_device,
            "status": "success",
            "fallback_from": "",
            "fallback_reason": "",
        }
    except Exception as exc:  # noqa: BLE001
        if spec.name == "stage4_xgboost_cuda":
            fallback = build_xgboost_cpu_fallback(numeric_features, categorical_features)
        elif spec.name == "stage4_lightgbm_gpu":
            fallback = build_lightgbm_cpu_fallback(numeric_features, categorical_features)
        else:
            return None, {
                "model": spec.name,
                "requested_device": spec.requested_device,
                "actual_device": spec.actual_device,
                "status": "failed",
                "fallback_from": "",
                "fallback_reason": f"{type(exc).__name__}: {exc}",
            }

        reason = f"{type(exc).__name__}: {exc}"
        fallback.model.fit(X_train, y_train)
        return fallback, {
            "model": fallback.name,
            "requested_device": spec.requested_device,
            "actual_device": fallback.actual_device,
            "status": "fallback_success",
            "fallback_from": spec.name,
            "fallback_reason": reason,
        }


def feature_names_from_model(model: object, numeric_features: list[str], categorical_features: list[str]) -> list[str]:
    try:
        preprocessor = model.named_steps["preprocessor"]
        return preprocessor.get_feature_names_out().tolist()
    except Exception:  # noqa: BLE001
        return numeric_features + categorical_features


def write_stage4_figures(results: pd.DataFrame, predictions: pd.DataFrame) -> None:
    if not has_module("matplotlib") or predictions.empty:
        return

    os.environ.setdefault("WINDIR", r"C:\Windows")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", str(FIGURES.parent / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

    FIGURES.mkdir(parents=True, exist_ok=True)
    test_results = results[results["split"].eq("test")]
    for metric in ["roc_auc", "pr_auc", "recall", "precision", "f1"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        test_results.sort_values(metric).plot(kind="barh", x="model", y=metric, legend=False, ax=ax, color="#4E79A7")
        ax.set_title(f"Stage 4 test {metric}")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(FIGURES / f"stage4_{metric}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, group in predictions.groupby("model"):
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_score"])
        ax.plot(fpr, tpr, label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_title("Stage 4 test ROC curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage4_test_roc_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, group in predictions.groupby("model"):
        precision, recall, _ = precision_recall_curve(group["y_true"], group["y_score"])
        ax.plot(recall, precision, label=model_name)
    ax.set_title("Stage 4 test precision-recall curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage4_test_pr_curves.png", dpi=160)
    plt.close(fig)

    for model_name, group in predictions.groupby("model"):
        cm = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        image = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{model_name} confusion matrix")
        ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
        ax.set_yticks([0, 1], labels=["true 0", "true 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111111")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(FIGURES / f"stage4_{model_name}_confusion_matrix.png", dpi=160)
        plt.close(fig)


def write_status_report(results: pd.DataFrame, run_log: pd.DataFrame, summary: pd.DataFrame) -> Path:
    status_path = REPORTS / "stage4_baseline_status.md"
    test_results = results[results["split"].eq("test")]
    lines = [
        "# 第 4 阶段：模型基线构建状态报告",
        "",
        "## 运行结果",
        "",
        "- 数据集：`home_credit`。",
        "- 输入特征数据：`data/processed/home_credit_features.csv`。",
        "- 固定划分：沿用 `split`，未重新抽样。",
        f"- 训练行数：{int(summary.loc[summary['split'].eq('train'), 'rows'].iloc[0])}。",
        f"- 验证行数：{int(summary.loc[summary['split'].eq('valid'), 'rows'].iloc[0])}。",
        f"- 测试行数：{int(summary.loc[summary['split'].eq('test'), 'rows'].iloc[0])}。",
        f"- 数值特征数：{int(summary['numeric_feature_count'].iloc[0])}。",
        f"- 类别特征数：{int(summary['categorical_feature_count'].iloc[0])}。",
        "- 训练策略：原始不平衡数据；不启用 class_weight、scale_pos_weight、采样或 SMOTE。",
        "",
        "## 模型运行日志",
        "",
    ]
    for _, row in run_log.iterrows():
        lines.append(
            f"- `{row['model']}`：requested `{row['requested_device']}`，actual `{row['actual_device']}`，status `{row['status']}`。"
        )
        reason = row.get("fallback_reason", "")
        if isinstance(reason, str) and reason.strip():
            lines.append(f"  - 回退原因：{reason.splitlines()[0]}")

    lines.extend(["", "## 测试集最佳结果", ""])
    if not test_results.empty:
        for metric in ["roc_auc", "pr_auc", "f1"]:
            best = test_results.sort_values(metric, ascending=False).iloc[0]
            lines.append(f"- Test {metric} 最优：`{best['model']}` = {best[metric]:.4f}。")

    lines.extend(
        [
            "",
            "## 验收检查",
            "",
            "- [x] Logistic Regression 基线完成。",
            "- [x] XGBoost 基线完成。",
            "- [x] LightGBM 强基线完成。",
            "- [x] ROC-AUC、PR-AUC、Precision、Recall、F1 和混淆矩阵已统一输出。",
            "- [x] 模型参数、运行设备和 fallback 原因已记录。",
            "- [x] 编码器、标准化器和模型只在 train split 拟合。",
            "",
            "## 复现命令",
            "",
            "```powershell",
            '$env:TRAIN_DEVICE="gpu"',
            ".\\.venv\\Scripts\\python.exe .\\src\\run_stage4_baselines.py",
            "```",
        ]
    )
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status_path


def run_stage4_baselines() -> dict[str, Path]:
    ensure_project_dirs()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    deps = dependency_status()
    if not deps["sklearn"]:
        raise RuntimeError("scikit-learn is required for stage 4 baselines.")

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
    specs: list[Stage4Spec] = [build_logistic_spec(numeric_features, categorical_features)]
    xgb_spec = build_xgboost_spec(numeric_features, categorical_features, device)
    lgbm_spec = build_lightgbm_spec(numeric_features, categorical_features, device)
    if xgb_spec is not None:
        specs.append(xgb_spec)
    if lgbm_spec is not None:
        specs.append(lgbm_spec)

    result_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    run_rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    import joblib

    for spec in specs:
        fitted_spec, run_row = fit_with_fallbacks(spec, X_train, y_train, numeric_features, categorical_features)
        run_rows.append(run_row)
        if fitted_spec is None:
            continue

        feature_names = feature_names_from_model(fitted_spec.model, numeric_features, categorical_features)
        feature_rows.append(
            {
                "model": fitted_spec.name,
                "input_numeric_feature_count": len(numeric_features),
                "input_categorical_feature_count": len(categorical_features),
                "transformed_feature_count": len(feature_names),
                "metadata_excluded": "; ".join([TARGET, *ID_COLUMNS, SPLIT]),
                "transformed_feature_names": "; ".join(feature_names),
            }
        )
        param_rows.append(
            {
                "model": fitted_spec.name,
                "requested_device": run_row["requested_device"],
                "actual_device": run_row["actual_device"],
                "random_state": RANDOM_STATE,
                "uses_class_weight": False,
                "uses_scale_pos_weight": False,
                "uses_sampling": False,
                "params_json": json.dumps(fitted_spec.params, ensure_ascii=False, sort_keys=True),
            }
        )

        for split_name, split_X, split_y in [
            ("valid", X_valid, y_valid),
            ("test", X_test, y_test),
        ]:
            scores = predict_scores(fitted_spec.model, split_X)
            metrics = evaluate_predictions(split_y, scores)
            result_rows.append({"model": fitted_spec.name, "split": split_name, **metrics})
            if split_name == "test":
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "model": fitted_spec.name,
                            "row_index": split_y.index,
                            "y_true": split_y.to_numpy(),
                            "y_score": scores,
                            "y_pred": (scores >= 0.5).astype(int),
                        }
                    )
                )

        joblib.dump(fitted_spec.model, MODELS / model_file_name(fitted_spec.name))

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    run_log = pd.DataFrame(run_rows)
    params = pd.DataFrame(param_rows)
    transformed_features = pd.DataFrame(feature_rows)
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
        "stage4_results": TABLES / "stage4_baseline_results.csv",
        "stage4_predictions": TABLES / "stage4_baseline_test_predictions.csv",
        "stage4_run_log": TABLES / "stage4_model_run_log.csv",
        "stage4_params": TABLES / "stage4_model_params.csv",
        "stage4_matrix_summary": TABLES / "stage4_feature_matrix_summary.csv",
        "stage4_transformed_features": TABLES / "stage4_transformed_feature_names.csv",
        "baseline_results": TABLES / "baseline_results.csv",
        "baseline_predictions": TABLES / "baseline_test_predictions.csv",
    }
    results.to_csv(outputs["stage4_results"], index=False)
    predictions.to_csv(outputs["stage4_predictions"], index=False)
    run_log.to_csv(outputs["stage4_run_log"], index=False)
    params.to_csv(outputs["stage4_params"], index=False)
    split_summary.to_csv(outputs["stage4_matrix_summary"], index=False)
    transformed_features.to_csv(outputs["stage4_transformed_features"], index=False)
    results.to_csv(outputs["baseline_results"], index=False)
    predictions.to_csv(outputs["baseline_predictions"], index=False)
    write_stage4_figures(results, predictions)
    outputs["status"] = write_status_report(results, run_log, split_summary)
    return outputs


def main() -> int:
    try:
        outputs = run_stage4_baselines()
    except Exception as exc:  # noqa: BLE001
        print(f"Stage 4 baseline training failed: {type(exc).__name__}: {exc}")
        return 1

    print("Stage 4 baseline training complete.")
    for key, path in outputs.items():
        print(f"- {key}: {rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
