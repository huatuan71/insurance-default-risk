from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocess import MISSING_CATEGORY, add_home_credit_features, clean_home_credit_application
from features import DERIVED_FEATURES, ID_COLUMNS, SPLIT, TARGET, add_home_credit_stage3_features
from project_paths import DATA_PROCESSED, DATA_RAW, MODELS, ROOT, TABLES


REVIEW_MARGIN = 0.02
SCENARIO_ORDER = ("fn5_fp1", "fn10_fp1", "fn20_fp1")
STAGE2_DERIVED_COLUMNS = {
    "DAYS_EMPLOYED_ANOMALY",
    "credit_income_ratio",
    "annuity_income_ratio",
    "goods_credit_ratio",
    "employment_age_ratio",
}


class InferenceSetupError(RuntimeError):
    """Raised when local artifacts required for the classroom app are missing."""


class InferenceInputError(ValueError):
    """Raised when an uploaded CSV does not follow the Home Credit contract."""


@dataclass(frozen=True)
class ScoredBatch:
    results: pd.DataFrame
    prepared_features: pd.DataFrame


class HomeCreditInferenceService:
    """Load the trained local artifacts and score Home Credit application rows."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or ROOT)
        self.tables = self.root / "reports" / "tables"
        self.models_dir = self.root / "models"
        self.processed_dir = self.root / "data" / "processed"
        self.recommendations = self._load_recommendations()
        self.models = self._load_models()
        self.reference_pipeline = next(iter(self.models.values()))
        self.preprocessor = self.reference_pipeline.named_steps["preprocessor"]
        self.model_features = list(self.preprocessor.feature_names_in_)
        self.categorical_features = self._categorical_features()
        self.numeric_features = [column for column in self.model_features if column not in self.categorical_features]
        self.raw_feature_columns = self._raw_feature_columns()
        self.numeric_fill_values, self.categorical_fill_values = self._load_fill_policy()
        self.feature_meanings = self._load_feature_meanings()

    @property
    def scenarios(self) -> list[str]:
        available = set(self.recommendations["cost_scenario"])
        return [scenario for scenario in SCENARIO_ORDER if scenario in available]

    @property
    def raw_template_columns(self) -> list[str]:
        return ["SK_ID_CURR", *self.raw_feature_columns]

    def _load_recommendations(self) -> pd.DataFrame:
        path = self.tables / "stage6_recommended_thresholds.csv"
        if not path.exists():
            raise InferenceSetupError(
                "缺少 reports/tables/stage6_recommended_thresholds.csv。"
                "请先运行 src/run_stage6_business_thresholds.py。"
            )
        recommendations = pd.read_csv(path)
        required = {"cost_scenario", "cost_ratio", "model", "model_family", "selected_threshold"}
        missing = sorted(required - set(recommendations.columns))
        if missing:
            raise InferenceSetupError(f"推荐阈值表缺少字段：{', '.join(missing)}。")
        recommendations = recommendations.loc[recommendations["model_family"].eq("lightgbm")].copy()
        if recommendations.empty:
            raise InferenceSetupError("推荐阈值表中没有可用于应用的 LightGBM 记录。")
        scenarios = set(recommendations["cost_scenario"])
        missing_scenarios = [scenario for scenario in SCENARIO_ORDER if scenario not in scenarios]
        if missing_scenarios:
            raise InferenceSetupError(f"推荐阈值表缺少成本场景：{', '.join(missing_scenarios)}。")
        return recommendations.set_index("cost_scenario", drop=False)

    def _load_models(self) -> dict[str, object]:
        try:
            import joblib
        except ImportError as exc:
            raise InferenceSetupError("缺少 joblib。请先执行 pip install -r requirements.txt。") from exc

        model_names = self.recommendations["model"].drop_duplicates().tolist()
        missing = [name for name in model_names if not (self.models_dir / f"{name}.joblib").exists()]
        if missing:
            raise InferenceSetupError(
                "缺少 Stage 5 模型："
                f"{', '.join(missing)}。请先运行 src/run_stage5_imbalance.py。"
            )
        return {name: joblib.load(self.models_dir / f"{name}.joblib") for name in model_names}

    def _categorical_features(self) -> list[str]:
        for name, _, columns in self.preprocessor.transformers_:
            if name == "categorical":
                return list(columns)
        return []

    def _raw_feature_columns(self) -> list[str]:
        stage3_columns = {feature.column for feature in DERIVED_FEATURES}
        derived = STAGE2_DERIVED_COLUMNS | stage3_columns
        return [column for column in self.model_features if column not in derived]

    def _load_fill_policy(self) -> tuple[dict[str, float], dict[str, str]]:
        path = self.tables / "home_credit_missing_imputation_policy.csv"
        if not path.exists():
            raise InferenceSetupError(
                "缺少 reports/tables/home_credit_missing_imputation_policy.csv。"
                "请先运行 src/run_week1_2.py。"
            )
        policy = pd.read_csv(path)
        required = {"column", "strategy", "fill_value"}
        if not required.issubset(policy.columns):
            raise InferenceSetupError("缺失值策略表格式不完整。请重新运行 src/run_week1_2.py。")

        numeric = policy.loc[policy["strategy"].eq("train_median"), ["column", "fill_value"]]
        numeric_values = {
            str(row["column"]): float(row["fill_value"])
            for _, row in numeric.iterrows()
            if pd.notna(row["fill_value"])
        }
        categorical = policy.loc[
            policy["strategy"].eq("fill_missing_and_group_train_rare_categories"), ["column", "fill_value"]
        ]
        categorical_values = {
            str(row["column"]): str(row["fill_value"])
            for _, row in categorical.iterrows()
            if pd.notna(row["fill_value"])
        }
        return numeric_values, categorical_values

    def _load_feature_meanings(self) -> dict[tuple[str, str], str]:
        path = self.tables / "stage7_key_feature_business_interpretation.csv"
        if not path.exists():
            return {}
        table = pd.read_csv(path)
        required = {"model", "encoded_feature", "business_meaning"}
        if not required.issubset(table.columns):
            return {}
        return {
            (str(row["model"]), str(row["encoded_feature"])): str(row["business_meaning"])
            for _, row in table.iterrows()
        }

    def recommendation(self, scenario: str) -> pd.Series:
        if scenario not in self.recommendations.index:
            raise InferenceInputError(f"未知成本场景：{scenario}。")
        row = self.recommendations.loc[scenario]
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row

    def validate_raw_frame(self, raw: pd.DataFrame) -> None:
        if raw.empty:
            raise InferenceInputError("上传的 CSV 没有可评分的记录。")
        missing = [column for column in self.raw_template_columns if column not in raw.columns]
        if missing:
            preview = "、".join(missing[:12])
            suffix = " 等" if len(missing) > 12 else ""
            raise InferenceInputError(
                f"CSV 缺少 {len(missing)} 个 Home Credit 原始字段：{preview}{suffix}。"
                "请下载页面提供的模板，或上传 application_train/application_test 格式文件。"
            )

    def prepare_raw_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        self.validate_raw_frame(raw)
        out = raw.copy()
        out = out.drop(columns=[TARGET, SPLIT], errors="ignore")
        out = clean_home_credit_application(out)
        out = add_home_credit_features(out)

        for column, fill_value in self.numeric_fill_values.items():
            if column in out.columns:
                out[column] = pd.to_numeric(out[column], errors="coerce")
                out[column] = out[column].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
        for column, fill_value in self.categorical_fill_values.items():
            if column in out.columns:
                out[column] = out[column].astype("string").fillna(fill_value or MISSING_CATEGORY).astype("object")

        out = add_home_credit_stage3_features(out)
        missing_features = [column for column in self.model_features if column not in out.columns]
        if missing_features:
            raise InferenceInputError(f"预处理后仍缺少模型字段：{', '.join(missing_features[:10])}。")

        prepared = out.reindex(columns=self.model_features).copy()
        for column in self.numeric_features:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
            prepared[column] = prepared[column].replace([np.inf, -np.inf], np.nan)
            if prepared[column].isna().any():
                prepared[column] = prepared[column].fillna(self.numeric_fill_values.get(column, 0.0))
        for column in self.categorical_features:
            prepared[column] = prepared[column].astype("string").fillna(MISSING_CATEGORY).astype("object")
        return prepared

    def prepare_engineered_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.model_features if column not in frame.columns]
        if missing:
            raise InferenceInputError(f"工程后案例缺少模型字段：{', '.join(missing[:10])}。")
        prepared = frame.reindex(columns=self.model_features).copy()
        for column in self.numeric_features:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            prepared[column] = prepared[column].fillna(self.numeric_fill_values.get(column, 0.0))
        for column in self.categorical_features:
            prepared[column] = prepared[column].astype("string").fillna(MISSING_CATEGORY).astype("object")
        return prepared

    def score_raw(self, raw: pd.DataFrame, scenario: str) -> ScoredBatch:
        prepared = self.prepare_raw_frame(raw)
        identifiers = raw["SK_ID_CURR"].reset_index(drop=True)
        return ScoredBatch(self._score_prepared(prepared, identifiers, scenario), prepared)

    def score_engineered(self, frame: pd.DataFrame, scenario: str) -> ScoredBatch:
        if "SK_ID_CURR" not in frame.columns:
            raise InferenceInputError("案例数据缺少 SK_ID_CURR。")
        prepared = self.prepare_engineered_frame(frame)
        identifiers = frame["SK_ID_CURR"].reset_index(drop=True)
        return ScoredBatch(self._score_prepared(prepared, identifiers, scenario), prepared)

    def _score_prepared(self, prepared: pd.DataFrame, identifiers: pd.Series, scenario: str) -> pd.DataFrame:
        recommendation = self.recommendation(scenario)
        model_name = str(recommendation["model"])
        scores = self.models[model_name].predict_proba(prepared)[:, 1]
        threshold = float(recommendation["selected_threshold"])
        high_risk = scores >= threshold
        near_boundary = np.abs(scores - threshold) <= REVIEW_MARGIN
        return pd.DataFrame(
            {
                "SK_ID_CURR": identifiers.to_numpy(),
                "risk_score": scores,
                "risk_probability_percent": scores * 100,
                "cost_scenario": recommendation["cost_ratio"],
                "scenario_key": scenario,
                "model": model_name,
                "decision_threshold": threshold,
                "decision": np.where(high_risk, "高风险建议拦截", "低于阈值建议通过"),
                "near_decision_boundary": near_boundary,
                "review_note": np.where(near_boundary, "接近决策边界，建议人工复核", ""),
            }
        )

    def demo_cases(self) -> pd.DataFrame:
        path = self.tables / "stage7_shap_sample_cases.csv"
        if not path.exists():
            raise InferenceSetupError("缺少 Stage 7 案例表。请先运行 src/run_stage7_explainability.py。")
        return pd.read_csv(path)

    def load_demo_case_features(self, case_name: str) -> pd.DataFrame:
        cases = self.demo_cases()
        match = cases.loc[cases["case_name"].eq(case_name)]
        if match.empty:
            raise InferenceInputError(f"找不到演示案例：{case_name}。")
        row_index = int(match.iloc[0]["row_index"])
        path = self.processed_dir / "home_credit_features.csv"
        if not path.exists():
            raise InferenceSetupError(
                "缺少 data/processed/home_credit_features.csv。请先运行 src/run_stage3_features.py。"
            )
        sample = pd.read_csv(path, skiprows=lambda line: line != 0 and line - 1 != row_index)
        if len(sample) != 1:
            raise InferenceSetupError(f"无法从工程后特征文件读取演示案例行 {row_index}。")
        return sample

    def explain_prepared(self, prepared: pd.DataFrame, scenario: str, top_k: int = 5) -> pd.DataFrame:
        try:
            import shap
        except ImportError as exc:
            raise InferenceSetupError("缺少 shap。请先执行 pip install -r requirements.txt。") from exc

        recommendation = self.recommendation(scenario)
        model_name = str(recommendation["model"])
        pipeline = self.models[model_name]
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]
        transformed = preprocessor.transform(prepared)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed)
        values = shap.TreeExplainer(estimator).shap_values(transformed)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)
        if values.ndim == 3 and values.shape[-1] == 2:
            values = values[:, :, 1]
        contributions = values[0]
        feature_names = preprocessor.get_feature_names_out().tolist()
        rows = []
        for name, value in zip(feature_names, contributions, strict=True):
            rows.append(
                {
                    "encoded_feature": name,
                    "shap_value": float(value),
                    "direction": "推高风险" if value > 0 else "降低风险",
                    "business_meaning": self.feature_meanings.get(
                        (model_name, name), "模型使用的申请表字段；需要结合字段字典进一步解释。"
                    ),
                }
            )
        explanation = pd.DataFrame(rows)
        positive = explanation.loc[explanation["shap_value"].gt(0)].nlargest(top_k, "shap_value")
        negative = explanation.loc[explanation["shap_value"].lt(0)].nsmallest(top_k, "shap_value")
        return pd.concat([positive, negative], ignore_index=True)

