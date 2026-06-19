from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import ID_COLUMNS, SPLIT, TARGET
from project_paths import FIGURES, MODELS, REPORTS, TABLES, ensure_project_dirs, rel
from run_stage4_baselines import ensure_feature_file, split_feature_target
from train_baseline import evaluate_predictions, has_module


RANDOM_STATE = 42
EPS = 1e-6
BIN_COUNT = 10
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
CALIBRATION_MODELS = [
    {"model": "stage5_lightgbm_smote_gpu", "model_family": "lightgbm", "strategy": "smote", "role": "stage6_5_1_and_10_1_recommended"},
    {"model": "stage5_lightgbm_raw_gpu", "model_family": "lightgbm", "strategy": "raw", "role": "stage6_20_1_recommended"},
    {"model": "stage5_xgboost_raw_cuda", "model_family": "xgboost", "strategy": "raw", "role": "stage5_strong_ranking_baseline"},
]
CALIBRATION_METHODS = ["raw", "sigmoid", "isotonic"]


def configure_plotting() -> None:
    os.environ.setdefault("WINDIR", r"C:\Windows")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", str(FIGURES.parent / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)


def require_dependencies() -> None:
    missing = [name for name in ["joblib", "sklearn", "matplotlib"] if not has_module(name)]
    if missing:
        raise RuntimeError(f"Missing dependencies for probability calibration: {missing}")


def model_path(model_name: str) -> Path:
    path = MODELS / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{rel(path)} not found. Run src/run_stage5_imbalance.py first.")
    return path


def clip_scores(scores: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=float), EPS, 1 - EPS)


def predict_scores(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return clip_scores(model.predict_proba(X)[:, 1])
    raw_score = model.decision_function(X)
    return clip_scores(1 / (1 + np.exp(-raw_score)))


def fit_calibrators(valid_scores: np.ndarray, y_valid: pd.Series) -> dict[str, object | None]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    sigmoid = LogisticRegression(solver="lbfgs", random_state=RANDOM_STATE)
    sigmoid.fit(valid_scores.reshape(-1, 1), y_valid)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    isotonic.fit(valid_scores, y_valid)
    return {"raw": None, "sigmoid": sigmoid, "isotonic": isotonic}


def apply_calibrator(method: str, calibrator: object | None, scores: np.ndarray) -> np.ndarray:
    if method == "raw":
        return clip_scores(scores)
    if method == "sigmoid":
        return clip_scores(calibrator.predict_proba(scores.reshape(-1, 1))[:, 1])  # type: ignore[union-attr]
    if method == "isotonic":
        return clip_scores(calibrator.predict(scores))  # type: ignore[union-attr]
    raise ValueError(f"Unknown calibration method: {method}")


def calibration_bins(y_true: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    y_array = y_true.to_numpy(dtype=int)
    score_array = np.asarray(scores, dtype=float)
    edges = np.linspace(0, 1, BIN_COUNT + 1)
    bin_ids = np.digitize(score_array, edges[1:-1], right=False)
    rows = []
    for bin_id in range(BIN_COUNT):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        lower = float(edges[bin_id])
        upper = float(edges[bin_id + 1])
        if count == 0:
            rows.append(
                {
                    "bin": bin_id + 1,
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_default_rate": np.nan,
                    "absolute_gap": np.nan,
                    "weighted_gap": 0.0,
                }
            )
            continue
        mean_pred = float(score_array[mask].mean())
        observed = float(y_array[mask].mean())
        gap = abs(mean_pred - observed)
        rows.append(
            {
                "bin": bin_id + 1,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_predicted_probability": mean_pred,
                "observed_default_rate": observed,
                "absolute_gap": gap,
                "weighted_gap": gap * count / len(score_array),
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(y_true: pd.Series, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    bins = calibration_bins(y_true, scores)
    non_empty = bins[bins["count"].gt(0)]
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "log_loss": float(log_loss(y_true, clip_scores(scores), labels=[0, 1])),
        "ece": float(bins["weighted_gap"].sum()),
        "mce": float(non_empty["absolute_gap"].max()),
        "mean_predicted_probability": float(np.mean(scores)),
        "observed_default_rate": float(y_true.mean()),
    }


def choose_threshold(y_valid: pd.Series, scores: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in THRESHOLD_GRID:
        metrics = evaluate_predictions(y_valid, scores, threshold=float(threshold))
        rows.append({"selection_split": "valid", **metrics})
    search = pd.DataFrame(rows)
    best = search.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return float(best["threshold"]), search


def write_figures(results: pd.DataFrame, bins: pd.DataFrame, decision_results: pd.DataFrame) -> None:
    configure_plotting()
    import matplotlib.pyplot as plt

    test_results = results[results["split"].eq("test")].copy()
    for metric in ["brier_score", "ece", "log_loss"]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        test_results.assign(label=lambda x: x["model"] + " / " + x["calibration_method"]).sort_values(metric).plot(
            kind="barh", x="label", y=metric, legend=False, ax=ax, color="#4E79A7"
        )
        ax.set_title(f"Probability calibration test {metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(FIGURES / f"stage6_probability_calibration_{metric}.png", dpi=160)
        plt.close(fig)

    test_decisions = decision_results[decision_results["split"].eq("test")].copy()
    for metric in ["precision", "recall", "f1"]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        test_decisions.assign(label=lambda x: x["model"] + " / " + x["calibration_method"]).sort_values(metric).plot(
            kind="barh", x="label", y=metric, legend=False, ax=ax, color="#59A14F"
        )
        ax.set_title(f"Calibrated valid-F1 threshold test {metric}")
        ax.set_xlim(0, 1)
        ax.set_xlabel(metric)
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(FIGURES / f"stage6_probability_calibration_decision_{metric}.png", dpi=160)
        plt.close(fig)

    for model_name, group in bins[bins["split"].eq("test")].groupby("model"):
        fig, ax = plt.subplots(figsize=(6, 5))
        for method, method_bins in group.groupby("calibration_method"):
            method_bins = method_bins[method_bins["count"].gt(0)]
            ax.plot(
                method_bins["mean_predicted_probability"],
                method_bins["observed_default_rate"],
                marker="o",
                label=method,
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
        ax.set_title(f"{model_name} reliability curve on test")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed default rate")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / f"stage6_probability_calibration_reliability_{model_name}.png", dpi=160)
        plt.close(fig)


def summarize_results(results: pd.DataFrame, decision_results: pd.DataFrame) -> pd.DataFrame:
    test = results[results["split"].eq("test")].copy()
    decision_test = decision_results[decision_results["split"].eq("test")].copy()
    rows = []
    for model_name, group in test.groupby("model"):
        raw = group[group["calibration_method"].eq("raw")].iloc[0]
        best_brier = group.sort_values("brier_score", ascending=True).iloc[0]
        best_ece = group.sort_values("ece", ascending=True).iloc[0]
        decision_group = decision_test[decision_test["model"].eq(model_name)]
        best_f1 = decision_group.sort_values("f1", ascending=False).iloc[0]
        rows.append(
            {
                "model": model_name,
                "raw_brier_score": raw["brier_score"],
                "best_brier_method": best_brier["calibration_method"],
                "best_brier_score": best_brier["brier_score"],
                "brier_improvement_vs_raw": raw["brier_score"] - best_brier["brier_score"],
                "raw_ece": raw["ece"],
                "best_ece_method": best_ece["calibration_method"],
                "best_ece": best_ece["ece"],
                "ece_improvement_vs_raw": raw["ece"] - best_ece["ece"],
                "best_decision_f1_method": best_f1["calibration_method"],
                "best_decision_f1": best_f1["f1"],
                "best_decision_threshold": best_f1["threshold"],
            }
        )
    return pd.DataFrame(rows)


def write_status_report(summary: pd.DataFrame, results: pd.DataFrame, decision_results: pd.DataFrame) -> Path:
    status_path = REPORTS / "stage6_probability_calibration_status.md"
    test = results[results["split"].eq("test")]
    best_brier = test.sort_values("brier_score", ascending=True).iloc[0]
    best_ece = test.sort_values("ece", ascending=True).iloc[0]
    decision_test = decision_results[decision_results["split"].eq("test")]
    best_f1 = decision_test.sort_values("f1", ascending=False).iloc[0]
    lines = [
        "# 概率校准实验状态报告",
        "",
        "## 运行结果",
        "",
        "- 数据集：`home_credit`。",
        "- 校准对象：Stage 6 推荐 LightGBM 模型与 Stage 5 XGBoost raw 强基线。",
        "- 校准方法：raw、sigmoid/Platt、isotonic。",
        "- 校准器拟合：只使用 valid split；test split 只做最终评估。",
        "- 评估指标：Brier score、log loss、ECE、MCE、ROC-AUC、PR-AUC，以及 valid-F1 阈值下的混淆矩阵指标。",
        "",
        "## 最佳校准结果",
        "",
        f"- Test Brier score 最优：`{best_brier['model']}` / `{best_brier['calibration_method']}` = {best_brier['brier_score']:.5f}。",
        f"- Test ECE 最优：`{best_ece['model']}` / `{best_ece['calibration_method']}` = {best_ece['ece']:.5f}。",
        f"- valid-F1 阈值后 Test F1 最优：`{best_f1['model']}` / `{best_f1['calibration_method']}` = {best_f1['f1']:.4f}，阈值 `{best_f1['threshold']:.2f}`。",
        "",
        "## 分模型结论",
        "",
    ]
    raw_wins_all = True
    for _, row in summary.iterrows():
        raw_wins_all = raw_wins_all and row["best_brier_method"] == "raw" and row["best_ece_method"] == "raw"
        lines.append(
            f"- `{row['model']}`：Brier 最优 `{row['best_brier_method']}`，较 raw 改善 `{row['brier_improvement_vs_raw']:.5f}`；ECE 最优 `{row['best_ece_method']}`，较 raw 改善 `{row['ece_improvement_vs_raw']:.5f}`。"
        )
    if raw_wins_all:
        lines.append("- 结论：sigmoid 和 isotonic 在 test split 上没有超过 raw 分数，说明当前候选 GBDT 分数已经具备较好的概率一致性，展示和业务阈值阶段可继续使用 raw score。")
    lines.extend(
        [
            "",
            "## 解释口径",
            "",
            "- 概率校准改善的是“预测概率是否接近真实违约率”，不一定提升 ROC-AUC 或 PR-AUC，因为排序能力主要由原模型决定。",
            "- 本阶段没有重新训练基础模型，也没有使用 test split 拟合校准器。",
            "- 若校准方法没有改善 Brier/ECE，应保留 raw score，而不是为了形式强行替换概率。",
            "",
            "## 验收检查",
            "",
            "- [x] 已输出 raw、sigmoid、isotonic 的校准指标。",
            "- [x] 已输出 reliability curve、Brier/ECE/log loss 对比图。",
            "- [x] 已输出校准后 valid-F1 阈值与 test 混淆矩阵指标。",
            "- [x] 校准器只使用 valid split 拟合，test split 只用于最终评估。",
            "",
            "## 复现命令",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe .\\src\\run_stage6_probability_calibration.py",
            "```",
        ]
    )
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status_path


def update_readme_checklist() -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("- [ ] 已完成概率校准实验。", "- [x] 已完成概率校准实验。")
    readme.write_text(text, encoding="utf-8")


def run_probability_calibration() -> dict[str, Path]:
    ensure_project_dirs()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    require_dependencies()

    import joblib

    feature_path = ensure_feature_file()
    df = pd.read_csv(feature_path)
    X, y, split, _, _ = split_feature_target(df)
    valid_mask = split.eq("valid")
    test_mask = split.eq("test")
    X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    metric_rows: list[dict[str, object]] = []
    bin_rows: list[pd.DataFrame] = []
    threshold_rows: list[pd.DataFrame] = []
    decision_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    for spec in CALIBRATION_MODELS:
        model_name = spec["model"]
        loaded_model = joblib.load(model_path(model_name))
        valid_raw = predict_scores(loaded_model, X_valid)
        test_raw = predict_scores(loaded_model, X_test)
        calibrators = fit_calibrators(valid_raw, y_valid)
        for method in CALIBRATION_METHODS:
            calibrator = calibrators[method]
            valid_scores = apply_calibrator(method, calibrator, valid_raw)
            test_scores = apply_calibrator(method, calibrator, test_raw)
            for split_name, split_y, split_scores in [
                ("valid", y_valid, valid_scores),
                ("test", y_test, test_scores),
            ]:
                metrics = calibration_metrics(split_y, split_scores)
                metric_rows.append(
                    {
                        **spec,
                        "calibration_method": method,
                        "split": split_name,
                        **metrics,
                    }
                )
                bins = calibration_bins(split_y, split_scores)
                bins.insert(0, "split", split_name)
                bins.insert(0, "calibration_method", method)
                bins.insert(0, "strategy", spec["strategy"])
                bins.insert(0, "model_family", spec["model_family"])
                bins.insert(0, "model", model_name)
                bin_rows.append(bins)

            threshold, search = choose_threshold(y_valid, valid_scores)
            search.insert(0, "calibration_method", method)
            search.insert(0, "strategy", spec["strategy"])
            search.insert(0, "model_family", spec["model_family"])
            search.insert(0, "model", model_name)
            threshold_rows.append(search)
            for split_name, split_y, split_scores in [
                ("valid", y_valid, valid_scores),
                ("test", y_test, test_scores),
            ]:
                decision_rows.append(
                    {
                        **spec,
                        "calibration_method": method,
                        "split": split_name,
                        **evaluate_predictions(split_y, split_scores, threshold=threshold),
                    }
                )

            run_rows.append(
                {
                    **spec,
                    "calibration_method": method,
                    "calibration_fit_split": "valid",
                    "test_usage": "final_evaluation_only",
                    "valid_rows": int(y_valid.shape[0]),
                    "test_rows": int(y_test.shape[0]),
                    "valid_positive_count": int(y_valid.sum()),
                    "test_positive_count": int(y_test.sum()),
                    "status": "success",
                }
            )

    results = pd.DataFrame(metric_rows)
    bins = pd.concat(bin_rows, ignore_index=True)
    thresholds = pd.concat(threshold_rows, ignore_index=True)
    decision_results = pd.DataFrame(decision_rows)
    summary = summarize_results(results, decision_results)
    run_log = pd.DataFrame(run_rows)

    outputs = {
        "calibration_results": TABLES / "stage6_probability_calibration_results.csv",
        "calibration_bins": TABLES / "stage6_probability_calibration_bins.csv",
        "calibration_thresholds": TABLES / "stage6_probability_calibration_thresholds.csv",
        "calibration_decisions": TABLES / "stage6_probability_calibration_decision_results.csv",
        "calibration_summary": TABLES / "stage6_probability_calibration_summary.csv",
        "calibration_run_log": TABLES / "stage6_probability_calibration_run_log.csv",
    }
    results.to_csv(outputs["calibration_results"], index=False)
    bins.to_csv(outputs["calibration_bins"], index=False)
    thresholds.to_csv(outputs["calibration_thresholds"], index=False)
    decision_results.to_csv(outputs["calibration_decisions"], index=False)
    summary.to_csv(outputs["calibration_summary"], index=False)
    run_log.to_csv(outputs["calibration_run_log"], index=False)
    write_figures(results, bins, decision_results)
    outputs["status"] = write_status_report(summary, results, decision_results)
    update_readme_checklist()
    return outputs


def main() -> int:
    try:
        outputs = run_probability_calibration()
    except Exception as exc:  # noqa: BLE001
        print(f"Probability calibration experiment failed: {type(exc).__name__}: {exc}")
        return 1
    print("Probability calibration experiment complete.")
    for key, path in outputs.items():
        print(f"- {key}: {rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
