from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import ID_COLUMNS, SPLIT, TARGET
from project_paths import DATA_PROCESSED, FIGURES, MODELS, REPORTS, TABLES, ensure_project_dirs, rel
from run_stage4_baselines import ensure_feature_file, predict_scores, split_feature_target
from train_baseline import has_module


RANDOM_STATE = 42
MAX_SHAP_ROWS = 2000
POSITIVE_SHAP_ROWS = 1000
EXPLAIN_MODELS = ["stage5_lightgbm_smote_gpu", "stage5_lightgbm_raw_gpu"]
MAIN_SCENARIO = "fn10_fp1"
RAW_SCENARIO = "fn20_fp1"


BUSINESS_TRANSLATIONS = {
    "EXT_SOURCE": "外部信用评分相关变量，通常反映第三方或外部数据源对申请人风险质量的综合判断。",
    "ext_source_mean": "三个外部信用评分的平均水平，用来概括申请人的综合外部信用质量。",
    "ext_source_min": "三个外部信用评分中的最低值，用来捕捉外部评分中的短板信号。",
    "ext_source_max": "三个外部信用评分中的最高值，用来反映申请人最好的一侧外部信用信号。",
    "ext_source_std": "三个外部信用评分的离散程度，用来衡量外部评分是否一致。",
    "ext_source_count": "可用外部评分字段数量，反映外部评分信息完整度。",
    "AMT_INCOME_TOTAL": "申请人收入水平，用于衡量还款能力。",
    "AMT_CREDIT": "贷款授信金额，金额越高，潜在敞口越大。",
    "AMT_ANNUITY": "贷款年金/分期还款压力。",
    "AMT_GOODS_PRICE": "消费贷款对应商品价格。",
    "log_AMT_INCOME_TOTAL": "收入金额的对数变换，用来降低极端收入值对模型的影响。",
    "log_AMT_CREDIT": "授信金额的对数变换，用来降低极端授信金额对模型的影响。",
    "log_AMT_ANNUITY": "分期还款金额的对数变换，用来降低极端年金值对模型的影响。",
    "log_AMT_GOODS_PRICE": "商品价格的对数变换，用来降低极端商品价格对模型的影响。",
    "DAYS_BIRTH": "申请人年龄，通常通过年龄阶段反映生命周期和收入稳定性。",
    "DAYS_EMPLOYED": "当前工作年限，反映就业稳定性。",
    "age_years": "申请人年龄（年）。",
    "employment_years": "当前就业年限（年）。",
    "annuity_credit_ratio": "分期还款额相对授信金额的压力。",
    "annuity_income_ratio": "分期还款额相对收入的压力。",
    "credit_income_ratio": "授信金额相对收入的杠杆水平。",
    "goods_credit_ratio": "商品价格相对授信金额的比例。",
    "income_per_person": "家庭人均收入。",
    "credit_per_person": "家庭人均授信金额。",
    "children_ratio": "家庭中儿童占比，反映家庭抚养压力。",
    "NAME_CONTRACT_TYPE": "贷款产品类型。",
    "CODE_GENDER": "申请人性别。",
    "NAME_INCOME_TYPE": "收入来源类型。",
    "NAME_EDUCATION_TYPE": "教育程度。",
    "NAME_FAMILY_STATUS": "婚姻/家庭状态。",
    "NAME_HOUSING_TYPE": "居住情况。",
    "OCCUPATION_TYPE": "职业类型。",
    "ORGANIZATION_TYPE": "工作单位/组织类型。",
    "REGION_RATING_CLIENT": "客户所在区域评级。",
    "REGION_RATING_CLIENT_W_CITY": "结合城市后的客户所在区域评级。",
    "WEEKDAY_APPR_PROCESS_START": "贷款申请提交的星期几。",
    "DAYS_LAST_PHONE_CHANGE": "最近一次更换电话号码距申请日的时间，可能反映联系方式稳定性。",
    "DAYS_REGISTRATION": "最近一次变更登记信息距申请日的时间，可能反映居住或身份信息稳定性。",
    "DAYS_ID_PUBLISH": "身份证件签发距申请日的时间。",
    "bureau_request_total": "征信查询次数汇总，可能反映近期信贷活跃度。",
    "AMT_REQ_CREDIT_BUREAU_MON": "申请前一个月内的征信查询次数，反映近期信贷申请活跃度。",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "申请前一年内的征信查询次数，反映中期信贷申请活跃度。",
    "document_flag_sum": "提交文件数量汇总，反映申请材料完整性。",
    "contact_flag_sum": "联系方式标记数量汇总，反映联系信息完整程度。",
    "FLAG_OWN_CAR": "是否拥有车辆，可能反映资产状况。",
    "FLAG_WORK_PHONE": "是否提供工作电话，反映可联系性和工作信息完整度。",
    "FLAG_PHONE": "是否提供联系电话，反映联系信息完整度。",
    "FLAG_DOCUMENT": "申请材料中的文件提交标记，反映材料完整性。",
    "FLAG_": "布尔标记类字段，反映申请材料、联系方式或资产信息是否存在。",
    "REGION_POPULATION_RELATIVE": "申请人所在地区的人口相对密度。",
    "EMERGENCYSTATE_MODE": "居住房屋所在区域的紧急状态标记。",
    "BASEMENTAREA_AVG": "居住房屋地下室面积相关标准化信息，属于居住条件特征。",
    "DEF_30_CNT_SOCIAL_CIRCLE": "申请人社交圈中出现 30 天以上逾期的人数，反映周边信用环境。",
}


def configure_plotting() -> None:
    os.environ.setdefault("WINDIR", r"C:\Windows")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", str(FIGURES.parent / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)


def require_dependencies() -> None:
    missing = [name for name in ["joblib", "shap", "matplotlib", "lightgbm"] if not has_module(name)]
    if missing:
        raise RuntimeError(f"Missing dependencies for Stage 7 explainability: {missing}")


def load_stage6_recommendations() -> pd.DataFrame:
    path = TABLES / "stage6_recommended_thresholds.csv"
    if not path.exists():
        raise FileNotFoundError(f"{rel(path)} not found. Run src/run_stage6_business_thresholds.py first.")
    recommendations = pd.read_csv(path)
    missing_models = [model for model in EXPLAIN_MODELS if model not in recommendations["model"].unique()]
    if missing_models:
        raise ValueError(f"Stage 6 recommendations do not contain expected models: {missing_models}")
    return recommendations


def load_dictionary() -> dict[str, dict[str, str]]:
    dictionary: dict[str, dict[str, str]] = {}
    field_path = TABLES / "home_credit_field_dictionary.csv"
    if field_path.exists():
        fields = pd.read_csv(field_path)
        for _, row in fields.iterrows():
            dictionary[str(row["column"])] = {
                "official_description": str(row.get("official_description", "")),
                "formula": str(row.get("derived_formula", "")),
            }
    derived_path = TABLES / "home_credit_derived_feature_dictionary.csv"
    if derived_path.exists():
        derived = pd.read_csv(derived_path)
        for _, row in derived.iterrows():
            dictionary[str(row["column"])] = {
                "official_description": str(row.get("business_meaning", "")),
                "formula": str(row.get("formula", "")),
            }
    return dictionary


def business_meaning(feature: str, dictionary: dict[str, dict[str, str]]) -> str:
    if feature in BUSINESS_TRANSLATIONS:
        return BUSINESS_TRANSLATIONS[feature]
    for prefix, meaning in BUSINESS_TRANSLATIONS.items():
        if feature.startswith(prefix):
            return meaning
    details = dictionary.get(feature, {})
    official = details.get("official_description", "")
    if official and official != "nan":
        return official
    return "模型使用的申请表字段；需要结合字段字典和 SHAP 方向进一步解释。"


def feature_family(feature: str) -> str:
    if feature.startswith("EXT_SOURCE") or feature.startswith("ext_source"):
        return "external_score"
    if feature.startswith("AMT_") or "ratio" in feature or feature in {"income_per_person", "credit_per_person"}:
        return "amount_and_ratio"
    if feature.startswith("DAYS_") or feature in {"age_years", "employment_years", "employment_age_ratio"}:
        return "age_and_tenure"
    if feature.startswith("NAME_") or feature in {"CODE_GENDER", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"}:
        return "categorical_profile"
    if feature.startswith("FLAG_") or feature.endswith("_flag_sum"):
        return "flags_and_contacts"
    if feature.startswith("CNT_") or "children" in feature:
        return "family_structure"
    return "other"


def parse_feature_name(encoded_feature: str, numeric_features: list[str], categorical_features: list[str]) -> dict[str, str]:
    if encoded_feature in numeric_features:
        return {
            "encoded_feature": encoded_feature,
            "original_feature": encoded_feature,
            "encoded_category": "",
            "feature_type": "numeric",
        }
    matches = [feature for feature in categorical_features if encoded_feature.startswith(f"{feature}_")]
    if matches:
        original = max(matches, key=len)
        return {
            "encoded_feature": encoded_feature,
            "original_feature": original,
            "encoded_category": encoded_feature[len(original) + 1 :],
            "feature_type": "one_hot",
        }
    return {
        "encoded_feature": encoded_feature,
        "original_feature": encoded_feature,
        "encoded_category": "",
        "feature_type": "unknown",
    }


def transformed_dense(preprocessor: object, X: pd.DataFrame) -> np.ndarray:
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        return transformed.toarray()
    return np.asarray(transformed)


def select_shap_sample(X_test: pd.DataFrame, y_test: pd.Series) -> pd.Index:
    positive_index = y_test[y_test.eq(1)].index
    negative_index = y_test[y_test.eq(0)].index
    positive_count = min(POSITIVE_SHAP_ROWS, len(positive_index), MAX_SHAP_ROWS)
    negative_count = min(MAX_SHAP_ROWS - positive_count, len(negative_index))
    sampled_positive = positive_index.to_series().sample(positive_count, random_state=RANDOM_STATE) if positive_count else []
    sampled_negative = negative_index.to_series().sample(negative_count, random_state=RANDOM_STATE) if negative_count else []
    sampled = pd.Index(list(sampled_positive) + list(sampled_negative))
    return sampled.sort_values()


def normalize_shap_values(values: object) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[1] if len(values) > 1 else values[0])
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == 2:
        return array[:, :, 1]
    return array


def normalize_expected_value(expected_value: object) -> float:
    if isinstance(expected_value, list):
        return float(expected_value[1] if len(expected_value) > 1 else expected_value[0])
    array = np.asarray(expected_value)
    if array.ndim > 0 and array.shape[0] > 1:
        return float(array[1])
    return float(array.ravel()[0])


def model_file(model_name: str) -> Path:
    path = MODELS / f"{model_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{rel(path)} not found. Run src/run_stage5_imbalance.py before Stage 7.")
    return path


def explain_model(
    model_name: str,
    X_sample: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    dictionary: dict[str, dict[str, str]],
) -> dict[str, object]:
    import joblib
    import shap

    pipeline = joblib.load(model_file(model_name))
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out().tolist()
    X_dense = transformed_dense(preprocessor, X_sample)
    explainer = shap.TreeExplainer(estimator)
    shap_values = normalize_shap_values(explainer.shap_values(X_dense))
    expected_value = normalize_expected_value(explainer.expected_value)

    parsed = [parse_feature_name(name, numeric_features, categorical_features) for name in feature_names]
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    shap_rows = []
    for idx, details in enumerate(parsed):
        original = details["original_feature"]
        shap_rows.append(
            {
                "model": model_name,
                "rank": 0,
                "encoded_feature": details["encoded_feature"],
                "original_feature": original,
                "encoded_category": details["encoded_category"],
                "feature_type": details["feature_type"],
                "feature_family": feature_family(original),
                "mean_abs_shap": float(mean_abs[idx]),
                "mean_shap": float(mean_signed[idx]),
                "business_meaning": business_meaning(original, dictionary),
            }
        )
    shap_df = pd.DataFrame(shap_rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["rank"] = np.arange(1, len(shap_df) + 1)

    booster = estimator.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    importance_rows = []
    for idx, details in enumerate(parsed):
        original = details["original_feature"]
        importance_rows.append(
            {
                "model": model_name,
                "rank": 0,
                "encoded_feature": details["encoded_feature"],
                "original_feature": original,
                "encoded_category": details["encoded_category"],
                "feature_type": details["feature_type"],
                "feature_family": feature_family(original),
                "importance_gain": float(gain[idx]),
                "importance_split": int(split[idx]),
                "business_meaning": business_meaning(original, dictionary),
            }
        )
    importance_df = pd.DataFrame(importance_rows).sort_values("importance_gain", ascending=False).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    return {
        "pipeline": pipeline,
        "preprocessor": preprocessor,
        "estimator": estimator,
        "explainer": explainer,
        "expected_value": expected_value,
        "feature_names": feature_names,
        "X_dense": X_dense,
        "shap_values": shap_values,
        "shap_df": shap_df,
        "importance_df": importance_df,
    }


def safe_file_part(value: str) -> str:
    return value.replace(":", "_").replace(" ", "_").replace("/", "_").replace("\\", "_")


def write_global_figures(model_name: str, explanation: dict[str, object]) -> list[Path]:
    configure_plotting()
    import matplotlib.pyplot as plt
    import shap

    paths: list[Path] = []
    shap_df: pd.DataFrame = explanation["shap_df"]  # type: ignore[assignment]
    importance_df: pd.DataFrame = explanation["importance_df"]  # type: ignore[assignment]
    feature_names: list[str] = explanation["feature_names"]  # type: ignore[assignment]
    X_dense: np.ndarray = explanation["X_dense"]  # type: ignore[assignment]
    shap_values: np.ndarray = explanation["shap_values"]  # type: ignore[assignment]

    top_importance = importance_df.head(20).sort_values("importance_gain")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_importance["encoded_feature"], top_importance["importance_gain"], color="#4E79A7")
    ax.set_title(f"{model_name} LightGBM gain importance")
    ax.set_xlabel("Gain importance")
    fig.tight_layout()
    path = FIGURES / f"stage7_{safe_file_part(model_name)}_lightgbm_feature_importance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    top_shap = shap_df.head(20).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_shap["encoded_feature"], top_shap["mean_abs_shap"], color="#59A14F")
    ax.set_title(f"{model_name} SHAP mean absolute contribution")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    path = FIGURES / f"stage7_{safe_file_part(model_name)}_shap_bar.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    shap.summary_plot(shap_values, X_dense, feature_names=feature_names, max_display=20, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    fig.tight_layout()
    path = FIGURES / f"stage7_{safe_file_part(model_name)}_shap_summary.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def choose_case_index(y_true: pd.Series, scores: np.ndarray, threshold: float, desired_y: int, desired_pred: int, mode: str) -> int:
    frame = pd.DataFrame({"y_true": y_true, "score": scores}, index=y_true.index)
    frame["y_pred"] = (frame["score"] >= threshold).astype(int)
    candidates = frame[(frame["y_true"].eq(desired_y)) & (frame["y_pred"].eq(desired_pred))]
    if candidates.empty:
        raise ValueError(f"No case found for y={desired_y}, pred={desired_pred}, threshold={threshold}.")
    if mode == "highest_score":
        return int(candidates["score"].idxmax())
    if mode == "lowest_score":
        return int(candidates["score"].idxmin())
    if mode == "near_threshold":
        return int((candidates["score"] - threshold).abs().idxmin())
    raise ValueError(f"Unknown case selection mode: {mode}")


def format_top_features(
    shap_row: np.ndarray,
    feature_names: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    dictionary: dict[str, dict[str, str]],
    positive: bool,
    top_n: int = 5,
) -> str:
    order = np.argsort(shap_row)
    selected = order[-top_n:][::-1] if positive else order[:top_n]
    parts = []
    for idx in selected:
        parsed = parse_feature_name(feature_names[int(idx)], numeric_features, categorical_features)
        original = parsed["original_feature"]
        label = parsed["encoded_feature"]
        meaning = business_meaning(original, dictionary)
        contribution = float(shap_row[int(idx)])
        parts.append(f"{label} ({contribution:+.4f}; {meaning})")
    return " | ".join(parts)


def write_waterfall(
    model_name: str,
    case_name: str,
    explanation: dict[str, object],
    X_case_dense: np.ndarray,
    shap_row: np.ndarray,
) -> Path:
    configure_plotting()
    import matplotlib.pyplot as plt
    import shap

    feature_names: list[str] = explanation["feature_names"]  # type: ignore[assignment]
    expected_value = float(explanation["expected_value"])
    shap_explanation = shap.Explanation(
        values=shap_row,
        base_values=expected_value,
        data=X_case_dense.ravel(),
        feature_names=feature_names,
    )
    shap.plots.waterfall(shap_explanation, max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    fig.tight_layout()
    path = FIGURES / f"stage7_{safe_file_part(model_name)}_{safe_file_part(case_name)}_waterfall.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def build_case_rows(
    explanations: dict[str, dict[str, object]],
    recommendations: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ids_test: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    dictionary: dict[str, dict[str, str]],
) -> pd.DataFrame:
    import shap

    case_specs = [
        {
            "case_name": "main_fn10_tp_high_risk",
            "model": "stage5_lightgbm_smote_gpu",
            "cost_scenario": MAIN_SCENARIO,
            "case_type": "TP high-risk default correctly rejected",
            "desired_y": 1,
            "desired_pred": 1,
            "mode": "highest_score",
        },
        {
            "case_name": "main_fn10_fp_false_reject",
            "model": "stage5_lightgbm_smote_gpu",
            "cost_scenario": MAIN_SCENARIO,
            "case_type": "FP non-default incorrectly rejected",
            "desired_y": 0,
            "desired_pred": 1,
            "mode": "near_threshold",
        },
        {
            "case_name": "main_fn10_tn_low_risk",
            "model": "stage5_lightgbm_smote_gpu",
            "cost_scenario": MAIN_SCENARIO,
            "case_type": "TN low-risk non-default correctly approved",
            "desired_y": 0,
            "desired_pred": 0,
            "mode": "lowest_score",
        },
        {
            "case_name": "fn20_raw_tp_high_risk",
            "model": "stage5_lightgbm_raw_gpu",
            "cost_scenario": RAW_SCENARIO,
            "case_type": "20:1 raw TP high-risk default correctly rejected",
            "desired_y": 1,
            "desired_pred": 1,
            "mode": "highest_score",
        },
    ]
    rows: list[dict[str, object]] = []
    for spec in case_specs:
        model_name = spec["model"]
        rec = recommendations[
            recommendations["model"].eq(model_name) & recommendations["cost_scenario"].eq(spec["cost_scenario"])
        ].iloc[0]
        threshold = float(rec["selected_threshold"])
        pipeline = explanations[model_name]["pipeline"]
        scores = predict_scores(pipeline, X_test)
        case_index = choose_case_index(
            y_test,
            scores,
            threshold,
            desired_y=int(spec["desired_y"]),
            desired_pred=int(spec["desired_pred"]),
            mode=str(spec["mode"]),
        )
        X_case = X_test.loc[[case_index]]
        preprocessor = explanations[model_name]["preprocessor"]
        explainer = explanations[model_name]["explainer"]
        feature_names: list[str] = explanations[model_name]["feature_names"]  # type: ignore[assignment]
        X_case_dense = transformed_dense(preprocessor, X_case)
        shap_values = normalize_shap_values(explainer.shap_values(X_case_dense))
        shap_row = shap_values[0]
        waterfall_path = write_waterfall(model_name, str(spec["case_name"]), explanations[model_name], X_case_dense, shap_row)
        score = float(scores[list(X_test.index).index(case_index)])
        y_pred = int(score >= threshold)
        rows.append(
            {
                "case_name": spec["case_name"],
                "case_type": spec["case_type"],
                "model": model_name,
                "cost_scenario": spec["cost_scenario"],
                "cost_ratio": rec["cost_ratio"],
                "business_threshold": threshold,
                "row_index": case_index,
                "SK_ID_CURR": int(ids_test.loc[case_index]),
                "y_true": int(y_test.loc[case_index]),
                "y_score": score,
                "y_pred": y_pred,
                "waterfall_path": rel(waterfall_path),
                "top_positive_features": format_top_features(
                    shap_row, feature_names, numeric_features, categorical_features, dictionary, positive=True
                ),
                "top_negative_features": format_top_features(
                    shap_row, feature_names, numeric_features, categorical_features, dictionary, positive=False
                ),
            }
        )
    return pd.DataFrame(rows)


def build_business_interpretation(global_shap: pd.DataFrame, feature_importance: pd.DataFrame) -> pd.DataFrame:
    top_shap = global_shap[global_shap["rank"].le(20)]
    top_gain = feature_importance[feature_importance["rank"].le(20)]
    combined = pd.concat(
        [
            top_shap[["model", "encoded_feature", "original_feature", "encoded_category", "feature_family", "business_meaning"]],
            top_gain[["model", "encoded_feature", "original_feature", "encoded_category", "feature_family", "business_meaning"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    rows = []
    for _, row in combined.iterrows():
        rows.append(
            {
                "model": row["model"],
                "encoded_feature": row["encoded_feature"],
                "original_feature": row["original_feature"],
                "encoded_category": row["encoded_category"],
                "feature_family": row["feature_family"],
                "business_meaning": row["business_meaning"],
                "interpretation_guidance": "SHAP 为正表示该特征在该样本上推高违约风险分数；SHAP 为负表示降低风险分数，不能解释为因果关系。",
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "feature_family", "original_feature", "encoded_feature"])


def write_status_report(
    global_shap: pd.DataFrame,
    sample_cases: pd.DataFrame,
    interpretation: pd.DataFrame,
    shap_sample_size: int,
) -> Path:
    status_path = REPORTS / "stage7_explainability_status.md"
    lines = [
        "# 第 7 阶段：可解释性分析状态报告",
        "",
        "## 运行结果",
        "",
        "- 数据集：`home_credit`。",
        "- 解释对象：Stage 6 推荐中使用到的 `stage5_lightgbm_smote_gpu` 与 `stage5_lightgbm_raw_gpu`。",
        f"- SHAP 全局解释样本：test split 抽样 {shap_sample_size} 行，优先保留违约样本。",
        "- 解释口径：SHAP 值解释模型风险分数方向，不代表因果关系。",
        "",
        "## 全局关键变量",
        "",
    ]
    for model_name, group in global_shap.groupby("model"):
        top = group.sort_values("mean_abs_shap", ascending=False).head(8)
        features = "、".join(top["encoded_feature"].tolist())
        lines.append(f"- `{model_name}` 主要依赖：{features}。")

    lines.extend(["", "## 单样本解释案例", ""])
    for _, row in sample_cases.iterrows():
        lines.append(
            f"- `{row['case_name']}`：SK_ID_CURR `{row['SK_ID_CURR']}`，真实标签 `{row['y_true']}`，预测分数 `{row['y_score']:.4f}`，阈值 `{row['business_threshold']:.2f}`，预测 `{row['y_pred']}`。"
        )
        lines.append(f"  - 推高风险：{row['top_positive_features'].split(' | ')[0]}")
        lines.append(f"  - 降低风险：{row['top_negative_features'].split(' | ')[0]}")

    lines.extend(["", "## 业务解释", ""])
    for family, group in interpretation.groupby("feature_family"):
        examples = "、".join(group["original_feature"].drop_duplicates().head(4).tolist())
        lines.append(f"- `{family}`：代表字段包括 {examples}。")

    lines.extend(
        [
            "",
            "## 验收检查",
            "",
            "- [x] 已输出 LightGBM feature importance 图。",
            "- [x] 已输出 SHAP summary/bar 图。",
            "- [x] 已输出 TP、FP、TN 和 20:1 raw TP 单样本 waterfall 图。",
            "- [x] 已生成关键特征业务解释表。",
            "- [x] 只解释 test split，未重新训练、未重新划分。",
            "",
            "## 复现命令",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe .\\src\\run_stage7_explainability.py",
            "```",
        ]
    )
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status_path


def update_readme_checklist() -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    replacements = {
        "- [ ] 已完成 SHAP 全局解释。": "- [x] 已完成 SHAP 全局解释。",
        "- [ ] 已完成单样本解释案例。": "- [x] 已完成单样本解释案例。",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    readme.write_text(text, encoding="utf-8")


def run_stage7_explainability() -> dict[str, Path]:
    ensure_project_dirs()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    require_dependencies()

    feature_path = ensure_feature_file()
    recommendations = load_stage6_recommendations()
    dictionary = load_dictionary()
    df = pd.read_csv(feature_path)
    X, y, split, numeric_features, categorical_features = split_feature_target(df)
    test_mask = split.eq("test")
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    ids_test = df.loc[test_mask, ID_COLUMNS[0]]
    sample_index = select_shap_sample(X_test, y_test)
    X_sample = X_test.loc[sample_index]

    explanations: dict[str, dict[str, object]] = {}
    feature_importance_rows: list[pd.DataFrame] = []
    shap_global_rows: list[pd.DataFrame] = []
    for model_name in EXPLAIN_MODELS:
        explanation = explain_model(model_name, X_sample, numeric_features, categorical_features, dictionary)
        explanations[model_name] = explanation
        feature_importance_rows.append(explanation["importance_df"])  # type: ignore[arg-type]
        shap_global_rows.append(explanation["shap_df"])  # type: ignore[arg-type]
        write_global_figures(model_name, explanation)

    feature_importance = pd.concat(feature_importance_rows, ignore_index=True)
    global_shap = pd.concat(shap_global_rows, ignore_index=True)
    sample_cases = build_case_rows(
        explanations,
        recommendations,
        X_test,
        y_test,
        ids_test,
        numeric_features,
        categorical_features,
        dictionary,
    )
    interpretation = build_business_interpretation(global_shap, feature_importance)

    outputs = {
        "stage7_feature_importance": TABLES / "stage7_lightgbm_feature_importance.csv",
        "stage7_global_shap": TABLES / "stage7_shap_global_importance.csv",
        "stage7_sample_cases": TABLES / "stage7_shap_sample_cases.csv",
        "stage7_business_interpretation": TABLES / "stage7_key_feature_business_interpretation.csv",
    }
    feature_importance.to_csv(outputs["stage7_feature_importance"], index=False)
    global_shap.to_csv(outputs["stage7_global_shap"], index=False)
    sample_cases.to_csv(outputs["stage7_sample_cases"], index=False)
    interpretation.to_csv(outputs["stage7_business_interpretation"], index=False)
    outputs["status"] = write_status_report(global_shap, sample_cases, interpretation, len(sample_index))
    update_readme_checklist()
    return outputs


def main() -> int:
    try:
        outputs = run_stage7_explainability()
    except Exception as exc:  # noqa: BLE001
        print(f"Stage 7 explainability failed: {type(exc).__name__}: {exc}")
        return 1

    print("Stage 7 explainability complete.")
    for key, path in outputs.items():
        print(f"- {key}: {rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
