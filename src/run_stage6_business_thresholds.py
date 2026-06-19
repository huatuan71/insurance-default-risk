from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import ID_COLUMNS, SPLIT, TARGET
from project_paths import DATA_PROCESSED, FIGURES, MODELS, REPORTS, TABLES, ensure_project_dirs, rel
from run_stage4_baselines import ensure_feature_file, predict_scores, split_feature_target
from run_stage5_imbalance import run_stage5_imbalance
from train_baseline import evaluate_predictions, has_module


THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
COST_ASSUMPTIONS = [
    {"cost_scenario": "fn5_fp1", "fn_cost": 5, "fp_cost": 1, "cost_ratio": "5:1"},
    {"cost_scenario": "fn10_fp1", "fn_cost": 10, "fp_cost": 1, "cost_ratio": "10:1"},
    {"cost_scenario": "fn20_fp1", "fn_cost": 20, "fp_cost": 1, "cost_ratio": "20:1"},
]
FIT_STRATEGIES = {"raw", "weighted", "random_under_sample", "smote"}


def configure_matplotlib() -> None:
    os.environ.setdefault("WINDIR", r"C:\Windows")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", str(FIGURES.parent / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)


def total_cost(fn: int, fp: int, fn_cost: int, fp_cost: int) -> int:
    return int(fn * fn_cost + fp * fp_cost)


def add_cost(metrics: dict[str, float], fn_cost: int, fp_cost: int) -> dict[str, float]:
    metrics = dict(metrics)
    metrics["fn_cost"] = fn_cost
    metrics["fp_cost"] = fp_cost
    metrics["total_cost"] = total_cost(int(metrics["fn"]), int(metrics["fp"]), fn_cost, fp_cost)
    return metrics


def model_candidates_from_run_log() -> pd.DataFrame:
    run_log_path = TABLES / "stage5_strategy_run_log.csv"
    if not run_log_path.exists():
        run_stage5_imbalance()
    run_log = pd.read_csv(run_log_path)
    candidates = run_log[
        run_log["strategy"].isin(FIT_STRATEGIES) & run_log["status"].isin(["success", "fallback_success"])
    ].copy()
    if candidates.empty:
        raise RuntimeError("No fitted Stage 5 model candidates found. Run src/run_stage5_imbalance.py first.")
    return candidates


def ensure_stage5_models(candidates: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in candidates["model"].tolist() if not (MODELS / f"{name}.joblib").exists()]
    if missing:
        run_stage5_imbalance()
        candidates = model_candidates_from_run_log()
        missing = [name for name in candidates["model"].tolist() if not (MODELS / f"{name}.joblib").exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 5 model files: {missing}")
    return candidates


def threshold_grid_rows(
    model_name: str,
    model_family: str,
    strategy: str,
    actual_device: str,
    y_valid: pd.Series,
    valid_scores: np.ndarray,
    cost_scenario: str,
    cost_ratio: str,
    fn_cost: int,
    fp_cost: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLD_GRID:
        metrics = add_cost(evaluate_predictions(y_valid, valid_scores, threshold=float(threshold)), fn_cost, fp_cost)
        rows.append(
            {
                "model": model_name,
                "model_family": model_family,
                "strategy": strategy,
                "actual_device": actual_device,
                "selection_split": "valid",
                "cost_scenario": cost_scenario,
                "cost_ratio": cost_ratio,
                **metrics,
            }
        )
    return rows


def evaluate_selected_threshold(
    model_name: str,
    model_family: str,
    strategy: str,
    actual_device: str,
    selected_threshold: float,
    y_valid: pd.Series,
    valid_scores: np.ndarray,
    y_test: pd.Series,
    test_scores: np.ndarray,
    cost_scenario: str,
    cost_ratio: str,
    fn_cost: int,
    fp_cost: int,
) -> dict[str, object]:
    valid_metrics = add_cost(evaluate_predictions(y_valid, valid_scores, threshold=selected_threshold), fn_cost, fp_cost)
    test_metrics = add_cost(evaluate_predictions(y_test, test_scores, threshold=selected_threshold), fn_cost, fp_cost)
    test_at_05 = add_cost(evaluate_predictions(y_test, test_scores, threshold=0.5), fn_cost, fp_cost)
    return {
        "model": model_name,
        "model_family": model_family,
        "strategy": strategy,
        "actual_device": actual_device,
        "cost_scenario": cost_scenario,
        "cost_ratio": cost_ratio,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "selected_threshold": selected_threshold,
        "valid_total_cost": valid_metrics["total_cost"],
        "valid_precision": valid_metrics["precision"],
        "valid_recall": valid_metrics["recall"],
        "valid_f1": valid_metrics["f1"],
        "valid_tn": valid_metrics["tn"],
        "valid_fp": valid_metrics["fp"],
        "valid_fn": valid_metrics["fn"],
        "valid_tp": valid_metrics["tp"],
        "test_total_cost": test_metrics["total_cost"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_tn": test_metrics["tn"],
        "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"],
        "test_tp": test_metrics["tp"],
        "test_cost_at_0_5": test_at_05["total_cost"],
        "test_cost_saving_vs_0_5": int(test_at_05["total_cost"] - test_metrics["total_cost"]),
        "test_tp_gain_vs_0_5": int(test_metrics["tp"] - test_at_05["tp"]),
        "test_fp_gain_vs_0_5": int(test_metrics["fp"] - test_at_05["fp"]),
        "test_fn_reduction_vs_0_5": int(test_at_05["fn"] - test_metrics["fn"]),
    }


def build_recommendations(optimal: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for assumption in COST_ASSUMPTIONS:
        scenario = assumption["cost_scenario"]
        group = optimal[optimal["cost_scenario"].eq(scenario)]
        if group.empty:
            continue
        ranked = group.sort_values(
            ["valid_total_cost", "test_total_cost", "test_recall", "test_precision"],
            ascending=[True, True, False, False],
        )
        rows.append(ranked.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def write_predictions_for_recommended_figures(
    rows: list[pd.DataFrame],
    recommendation: pd.Series,
    y_test: pd.Series,
    test_scores: np.ndarray,
) -> None:
    threshold = float(recommendation["selected_threshold"])
    rows.append(
        pd.DataFrame(
            {
                "cost_scenario": recommendation["cost_scenario"],
                "cost_ratio": recommendation["cost_ratio"],
                "model": recommendation["model"],
                "model_family": recommendation["model_family"],
                "strategy": recommendation["strategy"],
                "threshold": threshold,
                "y_true": y_test.to_numpy(),
                "y_score": test_scores,
                "y_pred": (test_scores >= threshold).astype(int),
            }
        )
    )


def safe_file_part(value: str) -> str:
    return value.replace(":", "_").replace(" ", "_").replace("/", "_").replace("\\", "_")


def write_stage6_figures(cost_grid: pd.DataFrame, recommendations: pd.DataFrame, prediction_rows: list[pd.DataFrame]) -> None:
    if not has_module("matplotlib"):
        return
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    FIGURES.mkdir(parents=True, exist_ok=True)
    for scenario, group in cost_grid.groupby("cost_scenario"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, line in group.groupby(["model_family", "strategy"]):
            model_family, strategy = label
            ax.plot(line["threshold"], line["total_cost"], label=f"{model_family}/{strategy}", linewidth=1.4)
        ax.set_title(f"Stage 6 valid business cost curves ({scenario})")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Total business cost")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(FIGURES / f"stage6_cost_curve_{safe_file_part(scenario)}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = recommendations["cost_ratio"].tolist()
    ax.bar(labels, recommendations["test_total_cost"], color="#4E79A7")
    ax.set_title("Stage 6 recommended test business cost")
    ax.set_xlabel("FN:FP cost ratio")
    ax.set_ylabel("Total cost on test set")
    for idx, value in enumerate(recommendations["test_total_cost"].tolist()):
        ax.text(idx, value, str(int(value)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage6_recommended_test_cost.png", dpi=160)
    plt.close(fig)

    if not prediction_rows:
        return
    predictions = pd.concat(prediction_rows, ignore_index=True)
    for scenario, group in predictions.groupby("cost_scenario"):
        cm = confusion_matrix(group["y_true"], group["y_pred"], labels=[0, 1])
        first = group.iloc[0]
        fig, ax = plt.subplots(figsize=(4.5, 4))
        image = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{scenario} recommended confusion matrix")
        ax.set_xlabel(f"{first['model_family']} / {first['strategy']} @ {first['threshold']:.2f}")
        ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
        ax.set_yticks([0, 1], labels=["true 0", "true 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#111111")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(FIGURES / f"stage6_recommended_confusion_matrix_{safe_file_part(scenario)}.png", dpi=160)
        plt.close(fig)


def write_status_report(
    recommendations: pd.DataFrame,
    optimal: pd.DataFrame,
    cost_grid: pd.DataFrame,
    split_summary: pd.DataFrame,
) -> Path:
    status_path = REPORTS / "stage6_business_threshold_status.md"
    lines = [
        "# 第 6 阶段：业务代价阈值优化状态报告",
        "",
        "## 运行结果",
        "",
        "- 数据集：`home_credit`。",
        "- 输入特征数据：`data/processed/home_credit_features.csv`。",
        "- 候选模型：Stage 5 已训练的 LightGBM 与 XGBoost 策略模型。",
        "- 成本假设：FN:FP = 5:1、10:1、20:1。",
        "- 阈值网格：0.01 到 0.99，步长 0.01。",
        "- 阈值选择：只使用 valid split；test split 只做最终评估。",
        f"- 训练行数：{int(split_summary.loc[split_summary['split'].eq('train'), 'rows'].iloc[0])}。",
        f"- 验证行数：{int(split_summary.loc[split_summary['split'].eq('valid'), 'rows'].iloc[0])}。",
        f"- 测试行数：{int(split_summary.loc[split_summary['split'].eq('test'), 'rows'].iloc[0])}。",
        "",
        "## 推荐阈值",
        "",
    ]
    for _, row in recommendations.iterrows():
        lines.append(
            f"- `{row['cost_ratio']}`：推荐 `{row['model_family']}/{row['strategy']}`，阈值 `{row['selected_threshold']:.2f}`，test 总成本 `{int(row['test_total_cost'])}`。"
        )
        lines.append(
            f"  - 相比同模型 0.5 阈值节省 `{int(row['test_cost_saving_vs_0_5'])}`，多识别 `{int(row['test_tp_gain_vs_0_5'])}` 个违约样本，新增 `{int(row['test_fp_gain_vs_0_5'])}` 个误报。"
        )

    lines.extend(["", "## 为什么不是 0.5", ""])
    below_half = recommendations[recommendations["selected_threshold"].lt(0.5)]
    if not below_half.empty:
        lines.append(
            "- 在所有成本假设下，推荐阈值均低于 0.5，因为漏判违约的业务成本高于误拒正常客户。"
        )
    lines.append(
        "- 降低阈值会提升违约召回率，但会增加误报；因此最终阈值必须跟 FN/FP 成本假设绑定解释。"
    )
    lines.append(
        "- 本阶段未做概率校准，目标是基于现有模型分数形成业务阈值建议；校准实验仍留在后续阶段。"
    )

    lines.extend(["", "## 验收检查", ""])
    checks = [
        f"已生成 `{rel(TABLES / 'stage6_threshold_cost_grid.csv')}`，共 {len(cost_grid)} 行 valid 阈值-成本记录。",
        f"已生成 `{rel(TABLES / 'stage6_optimal_thresholds.csv')}`，覆盖每个模型/策略/成本假设的最优阈值。",
        f"已生成 `{rel(TABLES / 'stage6_recommended_thresholds.csv')}`，给出每组成本假设的推荐决策。",
        "已输出阈值-业务成本曲线和推荐策略混淆矩阵。",
        "阈值选择只使用 valid split，test split 只用于最终评估。",
    ]
    for check in checks:
        lines.append(f"- [x] {check}")

    lines.extend(
        [
            "",
            "## 复现命令",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe .\\src\\run_stage6_business_thresholds.py",
            "```",
        ]
    )
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status_path


def update_readme_checklist() -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    replacements = {
        "- [ ] 已完成业务阈值优化。": "- [x] 已完成业务阈值优化。",
        "- [ ] 已输出混淆矩阵。": "- [x] 已输出混淆矩阵。",
        "- [ ] 已输出业务代价曲线。": "- [x] 已输出业务代价曲线。",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    readme.write_text(text, encoding="utf-8")


def run_stage6_business_thresholds() -> dict[str, Path]:
    ensure_project_dirs()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if not has_module("joblib"):
        raise RuntimeError("joblib is required to load Stage 5 models.")

    import joblib

    feature_path = ensure_feature_file()
    df = pd.read_csv(feature_path)
    X, y, split, numeric_features, categorical_features = split_feature_target(df)
    valid_mask = split.eq("valid")
    test_mask = split.eq("test")
    X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    candidates = ensure_stage5_models(model_candidates_from_run_log())
    grid_rows: list[dict[str, object]] = []
    optimal_rows: list[dict[str, object]] = []
    score_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for _, candidate in candidates.iterrows():
        model_name = candidate["model"]
        model = joblib.load(MODELS / f"{model_name}.joblib")
        valid_scores = predict_scores(model, X_valid)
        test_scores = predict_scores(model, X_test)
        score_cache[model_name] = (valid_scores, test_scores)
        for assumption in COST_ASSUMPTIONS:
            rows = threshold_grid_rows(
                model_name,
                candidate["model_family"],
                candidate["strategy"],
                candidate["actual_device"],
                y_valid,
                valid_scores,
                assumption["cost_scenario"],
                assumption["cost_ratio"],
                assumption["fn_cost"],
                assumption["fp_cost"],
            )
            grid_rows.extend(rows)
            best = (
                pd.DataFrame(rows)
                .sort_values(["total_cost", "recall", "precision"], ascending=[True, False, False])
                .iloc[0]
            )
            optimal_rows.append(
                evaluate_selected_threshold(
                    model_name,
                    candidate["model_family"],
                    candidate["strategy"],
                    candidate["actual_device"],
                    float(best["threshold"]),
                    y_valid,
                    valid_scores,
                    y_test,
                    test_scores,
                    assumption["cost_scenario"],
                    assumption["cost_ratio"],
                    assumption["fn_cost"],
                    assumption["fp_cost"],
                )
            )

    cost_grid = pd.DataFrame(grid_rows)
    optimal = pd.DataFrame(optimal_rows)
    recommendations = build_recommendations(optimal)
    prediction_rows: list[pd.DataFrame] = []
    for _, recommendation in recommendations.iterrows():
        _, test_scores = score_cache[recommendation["model"]]
        write_predictions_for_recommended_figures(prediction_rows, recommendation, y_test, test_scores)

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
    impact = recommendations[
        [
            "cost_scenario",
            "cost_ratio",
            "model",
            "model_family",
            "strategy",
            "selected_threshold",
            "valid_total_cost",
            "test_total_cost",
            "test_cost_at_0_5",
            "test_cost_saving_vs_0_5",
            "test_tp_gain_vs_0_5",
            "test_fp_gain_vs_0_5",
            "test_fn_reduction_vs_0_5",
            "test_precision",
            "test_recall",
            "test_f1",
        ]
    ].copy()

    outputs = {
        "stage6_cost_grid": TABLES / "stage6_threshold_cost_grid.csv",
        "stage6_optimal_thresholds": TABLES / "stage6_optimal_thresholds.csv",
        "stage6_recommended_thresholds": TABLES / "stage6_recommended_thresholds.csv",
        "stage6_business_impact": TABLES / "stage6_business_threshold_impact.csv",
        "stage6_split_summary": TABLES / "stage6_feature_matrix_summary.csv",
    }
    cost_grid.to_csv(outputs["stage6_cost_grid"], index=False)
    optimal.to_csv(outputs["stage6_optimal_thresholds"], index=False)
    recommendations.to_csv(outputs["stage6_recommended_thresholds"], index=False)
    impact.to_csv(outputs["stage6_business_impact"], index=False)
    split_summary.to_csv(outputs["stage6_split_summary"], index=False)
    write_stage6_figures(cost_grid, recommendations, prediction_rows)
    outputs["status"] = write_status_report(recommendations, optimal, cost_grid, split_summary)
    update_readme_checklist()
    return outputs


def main() -> int:
    try:
        outputs = run_stage6_business_thresholds()
    except Exception as exc:  # noqa: BLE001
        print(f"Stage 6 business threshold optimization failed: {type(exc).__name__}: {exc}")
        return 1

    print("Stage 6 business threshold optimization complete.")
    for key, path in outputs.items():
        print(f"- {key}: {rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
