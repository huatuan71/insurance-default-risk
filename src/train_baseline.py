from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import MODELS


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_status() -> dict[str, bool]:
    return {name: has_module(name) for name in ["sklearn", "lightgbm", "xgboost", "joblib"]}


def training_device() -> str:
    return os.environ.get("TRAIN_DEVICE", "auto").strip().lower() or "auto"


@dataclass
class ModelSpec:
    name: str
    model: object
    requested_device: str
    actual_device: str
    fallback_from: str = ""
    fallback_reason: str = ""


def prepare_numeric_xy(df: pd.DataFrame, target: str, id_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    excluded = set(id_columns + [target, "split"])
    features = df.drop(columns=[col for col in excluded if col in df.columns], errors="ignore")
    features = features.select_dtypes(include="number")
    labels = df[target].astype(int)
    return features, labels


def evaluate_predictions(y_true: pd.Series, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def configure_lightgbm_gpu_cache(figure_dir: Path) -> Path:
    gpu_cache = figure_dir.parent / ".gpu_cache"
    appdata = gpu_cache / "appdata"
    localappdata = gpu_cache / "localappdata"
    boost_cache = gpu_cache / "boost_compute"
    for path in [appdata, localappdata, boost_cache]:
        path.mkdir(parents=True, exist_ok=True)
    os.environ["APPDATA"] = str(appdata)
    os.environ["LOCALAPPDATA"] = str(localappdata)
    os.environ["BOOST_COMPUTE_CACHE_PATH"] = str(boost_cache)
    return gpu_cache


def train_baselines(
    processed_path: Path,
    target: str,
    id_columns: list[str],
    table_dir: Path,
    figure_dir: Path,
) -> Path:
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    deps = dependency_status()
    device = training_device()
    pd.DataFrame([{**deps, "training_device": device}]).to_csv(
        table_dir / "dependency_status.csv",
        index=False,
    )
    if not deps["sklearn"]:
        skipped = pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "scikit-learn is not installed. Install requirements.txt before running baseline training.",
                }
            ]
        )
        output = table_dir / "baseline_results.csv"
        skipped.to_csv(output, index=False)
        return output

    if device == "gpu" and not deps["xgboost"]:
        skipped = pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "TRAIN_DEVICE=gpu was requested, but xgboost is not installed.",
                }
            ]
        )
        output = table_dir / "baseline_results.csv"
        skipped.to_csv(output, index=False)
        return output

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    try:
        import joblib
    except ImportError:  # pragma: no cover
        joblib = None

    df = pd.read_csv(processed_path)
    X, y = prepare_numeric_xy(df, target, id_columns)
    train_mask = df["split"] == "train"
    valid_mask = df["split"] == "valid"
    test_mask = df["split"] == "test"

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    model_specs: list[ModelSpec] = []

    if device != "gpu":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        model_specs.extend(
            [
                ModelSpec(
                    name="logistic_regression_numeric",
                    requested_device="cpu",
                    actual_device="cpu",
                    model=
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            (
                                "model",
                                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
                            ),
                        ]
                    ),
                ),
                ModelSpec(
                    name="random_forest_numeric",
                    requested_device="cpu",
                    actual_device="cpu",
                    model=
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "model",
                                RandomForestClassifier(
                                    n_estimators=200,
                                    max_depth=12,
                                    min_samples_leaf=20,
                                    n_jobs=-1,
                                    class_weight="balanced_subsample",
                                    random_state=42,
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        )

    if deps["lightgbm"]:
        from lightgbm import LGBMClassifier

        lightgbm_params = {
            "n_estimators": 400,
            "learning_rate": 0.04,
            "num_leaves": 48,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        if device == "gpu":
            configure_lightgbm_gpu_cache(figure_dir)
            gpu_params = {
                **lightgbm_params,
                "device_type": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
            }
            model_specs.append(
                ModelSpec(
                    name="lightgbm_gpu_numeric",
                    requested_device="gpu",
                    actual_device="gpu",
                    model=Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("model", LGBMClassifier(**gpu_params)),
                        ]
                    ),
                )
            )
        else:
            model_specs.append(
                ModelSpec(
                    name="lightgbm_numeric",
                    requested_device="cpu",
                    actual_device="cpu",
                    model=Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("model", LGBMClassifier(**lightgbm_params)),
                        ]
                    ),
                )
            )

    if deps["xgboost"]:
        from xgboost import XGBClassifier

        negatives = int((y_train == 0).sum())
        positives = max(int((y_train == 1).sum()), 1)
        xgb_params = {
            "n_estimators": 350,
            "learning_rate": 0.04,
            "max_depth": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "eval_metric": "logloss",
            "scale_pos_weight": negatives / positives,
            "random_state": 42,
            "n_jobs": 1 if device == "gpu" else -1,
            "tree_method": "hist",
        }
        model_name = "xgboost_cuda_numeric" if device == "gpu" else "xgboost_numeric"
        if device == "gpu":
            xgb_params["device"] = "cuda"
        model_specs.append(
            ModelSpec(
                name=model_name,
                requested_device="cuda" if device == "gpu" else "cpu",
                actual_device="cuda" if device == "gpu" else "cpu",
                model=Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            XGBClassifier(**xgb_params),
                        ),
                    ]
                ),
            )
        )

    if not model_specs:
        skipped = pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": f"No compatible models were available for TRAIN_DEVICE={device}.",
                }
            ]
        )
        output = table_dir / "baseline_results.csv"
        skipped.to_csv(output, index=False)
        return output

    if device == "gpu":
        model_specs.sort(key=lambda spec: 0 if spec.name.startswith("xgboost") else 1)

    result_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    run_log_rows: list[dict[str, object]] = []
    for spec in model_specs:
        name = spec.name
        model = spec.model
        try:
            model.fit(X_train, y_train)
        except Exception as exc:  # noqa: BLE001
            if name == "lightgbm_gpu_numeric" and device == "gpu":
                from lightgbm import LGBMClassifier

                fallback_reason = f"{type(exc).__name__}: {exc}"
                name = "lightgbm_cpu_fallback_numeric"
                model = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            LGBMClassifier(
                                n_estimators=400,
                                learning_rate=0.04,
                                num_leaves=48,
                                subsample=0.85,
                                colsample_bytree=0.85,
                                class_weight="balanced",
                                random_state=42,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )
                model.fit(X_train, y_train)
                run_log_rows.append(
                    {
                        "model": name,
                        "requested_device": "gpu",
                        "actual_device": "cpu",
                        "status": "fallback_success",
                        "fallback_from": "lightgbm_gpu_numeric",
                        "fallback_reason": fallback_reason,
                    }
                )
            else:
                run_log_rows.append(
                    {
                        "model": name,
                        "requested_device": spec.requested_device,
                        "actual_device": spec.actual_device,
                        "status": "failed",
                        "fallback_from": spec.fallback_from,
                        "fallback_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
        else:
            run_log_rows.append(
                {
                    "model": name,
                    "requested_device": spec.requested_device,
                    "actual_device": spec.actual_device,
                    "status": "success",
                    "fallback_from": spec.fallback_from,
                    "fallback_reason": spec.fallback_reason,
                }
            )

        for split_name, split_X, split_y in [
            ("valid", X_valid, y_valid),
            ("test", X_test, y_test),
        ]:
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(split_X)[:, 1]
            else:
                raw_score = model.decision_function(split_X)
                score = 1 / (1 + np.exp(-raw_score))
            metrics = evaluate_predictions(split_y, score)
            result_rows.append({"model": name, "split": split_name, **metrics})
            if split_name == "test":
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "model": name,
                            "row_index": split_y.index,
                            "y_true": split_y.to_numpy(),
                            "y_score": score,
                            "y_pred": (score >= 0.5).astype(int),
                        }
                    )
                )

        if joblib is not None:
            joblib.dump(model, MODELS / f"{name}.joblib")

    results = pd.DataFrame(result_rows)
    output = table_dir / "baseline_results.csv"
    results.to_csv(output, index=False)
    pd.DataFrame(run_log_rows).to_csv(table_dir / "model_run_log.csv", index=False)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    if not predictions.empty:
        predictions.to_csv(table_dir / "baseline_test_predictions.csv", index=False)
    write_metric_figures(results, figure_dir)
    write_curve_figures(predictions, figure_dir)
    return output


def write_metric_figures(results: pd.DataFrame, figure_dir: Path) -> None:
    if not has_module("matplotlib"):
        return

    os.environ.setdefault("WINDIR", r"C:\Windows")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(figure_dir.parent / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    test = results[results["split"] == "test"].copy()
    if test.empty:
        return

    for metric in ["roc_auc", "pr_auc", "recall", "precision", "f1"]:
        fig, ax = plt.subplots(figsize=(8, 4))
        test.sort_values(metric).plot(kind="barh", x="model", y=metric, legend=False, ax=ax, color="#4E79A7")
        ax.set_title(f"Test {metric} by model")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(figure_dir / f"baseline_{metric}.png", dpi=160)
        plt.close(fig)


def write_curve_figures(predictions: pd.DataFrame, figure_dir: Path) -> None:
    if predictions.empty or not has_module("matplotlib"):
        return

    os.environ.setdefault("WINDIR", r"C:\Windows")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(figure_dir.parent / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, group in predictions.groupby("model"):
        fpr, tpr, _ = roc_curve(group["y_true"], group["y_score"])
        ax.plot(fpr, tpr, label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.set_title("Test ROC curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "baseline_test_roc_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for model_name, group in predictions.groupby("model"):
        precision, recall, _ = precision_recall_curve(group["y_true"], group["y_score"])
        ax.plot(recall, precision, label=model_name)
    ax.set_title("Test precision-recall curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "baseline_test_pr_curves.png", dpi=160)
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
        fig.savefig(figure_dir / f"baseline_{model_name}_confusion_matrix.png", dpi=160)
        plt.close(fig)
