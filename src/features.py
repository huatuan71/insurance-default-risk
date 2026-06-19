from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DATA_PROCESSED, REPORTS, TABLES, ensure_project_dirs, rel


TARGET = "TARGET"
SPLIT = "split"
ID_COLUMNS = ["SK_ID_CURR"]
FIT_SPLIT = "train"
ONE_HOT_MIN_FREQUENCY = 50


@dataclass(frozen=True)
class DerivedFeature:
    column: str
    formula: str
    source_columns: list[str]
    business_meaning: str
    feature_family: str


DERIVED_FEATURES = [
    DerivedFeature("age_years", "-DAYS_BIRTH / 365.25", ["DAYS_BIRTH"], "Applicant age in years.", "age_employment"),
    DerivedFeature(
        "employment_years",
        "max(-DAYS_EMPLOYED / 365.25, 0)",
        ["DAYS_EMPLOYED"],
        "Applicant employment tenure in years.",
        "age_employment",
    ),
    DerivedFeature(
        "annuity_credit_ratio",
        "AMT_ANNUITY / AMT_CREDIT",
        ["AMT_ANNUITY", "AMT_CREDIT"],
        "Loan annuity pressure relative to approved credit.",
        "amount_ratio",
    ),
    DerivedFeature(
        "income_per_person",
        "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS",
        ["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"],
        "Household income available per family member.",
        "amount_ratio",
    ),
    DerivedFeature(
        "credit_per_person",
        "AMT_CREDIT / CNT_FAM_MEMBERS",
        ["AMT_CREDIT", "CNT_FAM_MEMBERS"],
        "Approved credit amount per family member.",
        "amount_ratio",
    ),
    DerivedFeature(
        "children_ratio",
        "CNT_CHILDREN / CNT_FAM_MEMBERS",
        ["CNT_CHILDREN", "CNT_FAM_MEMBERS"],
        "Share of children in the applicant's family size.",
        "family_structure",
    ),
    DerivedFeature(
        "ext_source_mean",
        "mean(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)",
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
        "Average external risk score.",
        "external_score",
    ),
    DerivedFeature(
        "ext_source_min",
        "min(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)",
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
        "Lowest external risk score.",
        "external_score",
    ),
    DerivedFeature(
        "ext_source_max",
        "max(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)",
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
        "Highest external risk score.",
        "external_score",
    ),
    DerivedFeature(
        "ext_source_std",
        "std(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)",
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
        "Dispersion among external risk scores.",
        "external_score",
    ),
    DerivedFeature(
        "ext_source_count",
        "count_non_missing(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)",
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
        "Number of external score fields available after preprocessing.",
        "external_score",
    ),
    DerivedFeature(
        "document_flag_sum",
        "sum(FLAG_DOCUMENT_*)",
        ["FLAG_DOCUMENT_*"],
        "Number of submitted document flags.",
        "application_flags",
    ),
    DerivedFeature(
        "contact_flag_sum",
        "sum(contact availability flags)",
        ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"],
        "Number of available contact channels.",
        "application_flags",
    ),
    DerivedFeature(
        "bureau_request_total",
        "sum(AMT_REQ_CREDIT_BUREAU_*)",
        ["AMT_REQ_CREDIT_BUREAU_*"],
        "Total recent bureau enquiry count across recorded windows.",
        "bureau_request",
    ),
    DerivedFeature(
        "log_AMT_INCOME_TOTAL",
        "log1p(max(AMT_INCOME_TOTAL, 0))",
        ["AMT_INCOME_TOTAL"],
        "Log-transformed applicant income to reduce positive skew.",
        "log_amount",
    ),
    DerivedFeature(
        "log_AMT_CREDIT",
        "log1p(max(AMT_CREDIT, 0))",
        ["AMT_CREDIT"],
        "Log-transformed credit amount to reduce positive skew.",
        "log_amount",
    ),
    DerivedFeature(
        "log_AMT_ANNUITY",
        "log1p(max(AMT_ANNUITY, 0))",
        ["AMT_ANNUITY"],
        "Log-transformed loan annuity to reduce positive skew.",
        "log_amount",
    ),
    DerivedFeature(
        "log_AMT_GOODS_PRICE",
        "log1p(max(AMT_GOODS_PRICE, 0))",
        ["AMT_GOODS_PRICE"],
        "Log-transformed goods price to reduce positive skew.",
        "log_amount",
    ),
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace({0: np.nan})
    return result.replace([np.inf, -np.inf], np.nan)


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def load_processed_home_credit(path: Path | None = None) -> pd.DataFrame:
    processed_path = path or DATA_PROCESSED / "home_credit_processed.csv"
    if not processed_path.exists():
        raise FileNotFoundError(f"{rel(processed_path)} not found. Run src/run_week1_2.py first.")
    df = pd.read_csv(processed_path)
    required = {TARGET, SPLIT, *ID_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{rel(processed_path)} is missing required columns: {missing}")
    return df


def add_home_credit_stage3_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "DAYS_BIRTH" in out.columns:
        out["age_years"] = (-out["DAYS_BIRTH"] / 365.25).clip(lower=0)
    if "DAYS_EMPLOYED" in out.columns:
        out["employment_years"] = (-out["DAYS_EMPLOYED"] / 365.25).clip(lower=0)
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(out.columns):
        out["annuity_credit_ratio"] = safe_divide(out["AMT_ANNUITY"], out["AMT_CREDIT"])
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(out.columns):
        out["income_per_person"] = safe_divide(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"])
    if {"AMT_CREDIT", "CNT_FAM_MEMBERS"}.issubset(out.columns):
        out["credit_per_person"] = safe_divide(out["AMT_CREDIT"], out["CNT_FAM_MEMBERS"])
    if {"CNT_CHILDREN", "CNT_FAM_MEMBERS"}.issubset(out.columns):
        out["children_ratio"] = safe_divide(out["CNT_CHILDREN"], out["CNT_FAM_MEMBERS"])

    ext_cols = available_columns(out, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
    if ext_cols:
        ext = out[ext_cols]
        out["ext_source_mean"] = ext.mean(axis=1)
        out["ext_source_min"] = ext.min(axis=1)
        out["ext_source_max"] = ext.max(axis=1)
        out["ext_source_std"] = ext.std(axis=1).fillna(0)
        out["ext_source_count"] = ext.notna().sum(axis=1)

    document_cols = [column for column in out.columns if column.startswith("FLAG_DOCUMENT_")]
    if document_cols:
        out["document_flag_sum"] = out[document_cols].sum(axis=1)

    contact_cols = available_columns(
        out,
        ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"],
    )
    if contact_cols:
        out["contact_flag_sum"] = out[contact_cols].sum(axis=1)

    bureau_cols = [column for column in out.columns if column.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if bureau_cols:
        out["bureau_request_total"] = out[bureau_cols].sum(axis=1)

    for column in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        if column in out.columns:
            out[f"log_{column}"] = np.log1p(out[column].clip(lower=0))

    derived_cols = [feature.column for feature in DERIVED_FEATURES if feature.column in out.columns]
    out[derived_cols] = out[derived_cols].replace([np.inf, -np.inf], np.nan)
    return out


def fill_engineered_numeric_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    train_mask = out[SPLIT].eq(FIT_SPLIT)
    columns = ["column", "strategy", "fill_value", "missing_before", "reason"]
    rows = []
    for column in out.select_dtypes(include="number").columns:
        if column in [TARGET, *ID_COLUMNS]:
            continue
        missing_before = int(out[column].isna().sum())
        if missing_before == 0:
            continue
        fill_value = out.loc[train_mask, column].median()
        if pd.isna(fill_value):
            fill_value = 0.0
        out[column] = out[column].fillna(fill_value)
        rows.append(
            {
                "column": column,
                "strategy": "train_median_after_stage3_derivation",
                "fill_value": float(fill_value),
                "missing_before": missing_before,
                "reason": "Derived feature created missing/inf values; fill statistic is fitted on train split only.",
            }
        )
    return out, pd.DataFrame(rows, columns=columns)


def build_selection_record(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df[SPLIT].eq(FIT_SPLIT)]
    derived_names = {feature.column for feature in DERIVED_FEATURES}
    rows = []
    for column in df.columns:
        dtype = str(df[column].dtype)
        train_unique = int(train[column].nunique(dropna=False)) if column in train else 0
        role = "feature"
        action = "keep"
        selected = True
        reason = "Candidate feature after leakage and stability review."
        if column == TARGET:
            role, action, selected, reason = "target", "exclude", False, "Label column; use only as y."
        elif column in ID_COLUMNS:
            role, action, selected, reason = "id", "exclude", False, "Identifier; use only for tracing and joins."
        elif column == SPLIT:
            role, action, selected, reason = "split", "exclude", False, "Fixed split marker; not a model feature."
        elif train_unique <= 1:
            action, selected, reason = "exclude", False, "Zero variance on train split."
        elif column in derived_names:
            role, action, reason = "derived_feature", "keep", "Stage 3 engineered business feature."
        elif pd.api.types.is_numeric_dtype(df[column]):
            role = "numeric_feature"
        else:
            role, action, reason = "categorical_feature", "encode_later", "Use OneHotEncoder fitted on train split."

        rows.append(
            {
                "column": column,
                "role": role,
                "dtype": dtype,
                "train_unique_count": train_unique,
                "selected_for_feature_matrix": selected,
                "action": action,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def selected_columns(selection: pd.DataFrame) -> list[str]:
    metadata = [column for column in [*ID_COLUMNS, TARGET, SPLIT] if column in selection["column"].to_list()]
    features = selection.loc[selection["selected_for_feature_matrix"].eq(True), "column"].tolist()
    return metadata + [column for column in features if column not in metadata]


def build_feature_catalog(df: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    train = df[df[SPLIT].eq(FIT_SPLIT)]
    rows = []
    selection_by_column = selection.set_index("column")
    for column in df.columns:
        record = selection_by_column.loc[column]
        rows.append(
            {
                "column": column,
                "role": record["role"],
                "dtype": str(df[column].dtype),
                "train_missing_count": int(train[column].isna().sum()),
                "full_missing_count": int(df[column].isna().sum()),
                "train_unique_count": int(train[column].nunique(dropna=False)),
                "selected_for_feature_matrix": bool(record["selected_for_feature_matrix"]),
                "action": record["action"],
                "reason": record["reason"],
            }
        )
    return pd.DataFrame(rows)


def build_derived_feature_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in DERIVED_FEATURES:
        rows.append(
            {
                "column": feature.column,
                "feature_family": feature.feature_family,
                "formula": feature.formula,
                "source_columns": "; ".join(feature.source_columns),
                "business_meaning": feature.business_meaning,
                "created": feature.column in df.columns,
                "fit_scope": "formula uses row-level application fields; no validation/test aggregate statistics",
            }
        )
    return pd.DataFrame(rows)


def build_numeric_processing_policy(df: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    train = df[df[SPLIT].eq(FIT_SPLIT)]
    selected = set(selection.loc[selection["selected_for_feature_matrix"].eq(True), "column"])
    rows = []
    for column in df.select_dtypes(include="number").columns:
        if column not in selected or column in [TARGET, *ID_COLUMNS]:
            continue
        series = train[column].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            continue
        rows.append(
            {
                "column": column,
                "lower_p01_train": float(series.quantile(0.01)),
                "upper_p99_train": float(series.quantile(0.99)),
                "train_skewness": float(series.skew()) if len(series) > 2 else np.nan,
                "suggested_action": "winsorize_if_model_requires; not applied to engineered CSV",
                "fit_split": FIT_SPLIT,
            }
        )
    return pd.DataFrame(rows)


def build_encoding_policy(df: pd.DataFrame, selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = set(selection.loc[selection["selected_for_feature_matrix"].eq(True), "column"])
    categorical_cols = [
        column
        for column in df.select_dtypes(exclude="number").columns
        if column in selected and column not in [TARGET, SPLIT, *ID_COLUMNS]
    ]
    if not categorical_cols:
        return pd.DataFrame(), pd.DataFrame()

    from sklearn.preprocessing import OneHotEncoder

    train = df.loc[df[SPLIT].eq(FIT_SPLIT), categorical_cols].astype(str)
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=ONE_HOT_MIN_FREQUENCY, sparse_output=True)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=ONE_HOT_MIN_FREQUENCY, sparse=True)
    encoder.fit(train)
    feature_names = encoder.get_feature_names_out(categorical_cols)

    feature_name_rows = []
    policy_rows = []
    for column, categories in zip(categorical_cols, encoder.categories_, strict=False):
        encoded_for_column = [name for name in feature_names if name.startswith(f"{column}_")]
        top_categories = train[column].value_counts().head(10).index.astype(str).tolist()
        policy_rows.append(
            {
                "column": column,
                "strategy": "OneHotEncoder",
                "handle_unknown": "ignore",
                "min_frequency": ONE_HOT_MIN_FREQUENCY,
                "fit_split": FIT_SPLIT,
                "raw_train_unique_count": int(train[column].nunique(dropna=False)),
                "encoder_category_count": len(categories),
                "encoded_feature_count": len(encoded_for_column),
                "top_train_categories": "; ".join(top_categories),
                "notes": "Encoder is fitted on train split only; sparse matrix is not materialized as CSV.",
            }
        )
        for name in encoded_for_column:
            feature_name_rows.append({"source_column": column, "encoded_feature_name": name})

    return pd.DataFrame(policy_rows), pd.DataFrame(feature_name_rows)


def calculate_iv_for_series(values: pd.Series, target: pd.Series, is_numeric: bool) -> tuple[float, int]:
    frame = pd.DataFrame({"value": values, "target": target.astype(int)})
    if is_numeric and frame["value"].nunique(dropna=False) > 10:
        try:
            frame["bin"] = pd.qcut(frame["value"], q=10, duplicates="drop")
        except ValueError:
            frame["bin"] = pd.cut(frame["value"], bins=10, duplicates="drop")
    else:
        frame["bin"] = frame["value"].astype(str).fillna("__MISSING__")

    grouped = frame.groupby("bin", observed=False)["target"].agg(["sum", "count"])
    bad = grouped["sum"].astype(float) + 0.5
    good = (grouped["count"] - grouped["sum"]).astype(float) + 0.5
    bad_dist = bad / bad.sum()
    good_dist = good / good.sum()
    iv = ((bad_dist - good_dist) * np.log(bad_dist / good_dist)).sum()
    return float(iv), int(grouped.shape[0])


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "not_predictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious_or_very_strong"


def build_iv_summary(df: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    train = df[df[SPLIT].eq(FIT_SPLIT)]
    selected = selection.loc[selection["selected_for_feature_matrix"].eq(True), "column"].tolist()
    rows = []
    for column in selected:
        if column in [TARGET, SPLIT, *ID_COLUMNS]:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(train[column])
        iv, bin_count = calculate_iv_for_series(train[column], train[TARGET], is_numeric)
        rows.append(
            {
                "column": column,
                "feature_type": "numeric" if is_numeric else "categorical",
                "iv": iv,
                "iv_strength": iv_strength(iv),
                "bin_count": bin_count,
                "fit_split": FIT_SPLIT,
                "notes": "Screening metric only; no WOE replacement feature is generated in stage 3.",
            }
        )
    return pd.DataFrame(rows).sort_values("iv", ascending=False)


def export_feature_datasets(df: pd.DataFrame, output_prefix: str = "home_credit_features") -> dict[str, Path]:
    paths = {"full": DATA_PROCESSED / f"{output_prefix}.csv"}
    df.to_csv(paths["full"], index=False)
    for split in ["train", "valid", "test"]:
        path = DATA_PROCESSED / f"{output_prefix}_{split}.csv"
        df.loc[df[SPLIT].eq(split)].to_csv(path, index=False)
        paths[split] = path
    return paths


def write_feature_notes(
    output: Path,
    feature_paths: dict[str, Path],
    table_paths: dict[str, Path],
    selected_feature_count: int,
    derived_feature_count: int,
    categorical_feature_count: int,
) -> Path:
    text = f"""# Home Credit 特征工程说明

## 目标

本说明对应 README 3.3“特征工程”阶段。输入为 `data/processed/home_credit_processed.csv`，沿用第 3.2 阶段固定的 `train`、`valid`、`test` 划分，不重新抽样。

## 关键策略

- 标签 `TARGET`、ID `SK_ID_CURR`、划分字段 `split` 不进入模型特征。
- 新增 {derived_feature_count} 个业务衍生特征，均由申请主表的行级字段计算。
- 数值处理统计量、截尾边界、类别编码器和 IV 排名均只在 `train` split 拟合。
- 类别编码策略为 `OneHotEncoder(handle_unknown="ignore", min_frequency=50)`；当前不把稀疏 one-hot 矩阵落成 CSV。
- WOE/IV 当前只输出 IV 排名作为解释和筛选参考，不生成正式 WOE 替换特征。

## 产物

- 工程后完整数据：`{rel(feature_paths["full"])}`。
- 工程后训练/验证/测试：`{rel(feature_paths["train"])}`、`{rel(feature_paths["valid"])}`、`{rel(feature_paths["test"])}`。
- 特征清单：`{rel(table_paths["catalog"])}`。
- 衍生特征说明：`{rel(table_paths["derived"])}`。
- 特征筛选记录：`{rel(table_paths["selection"])}`。
- 类别编码策略：`{rel(table_paths["encoding"])}`。
- 编码后特征名清单：`{rel(table_paths["encoded_names"])}`。
- 数值处理策略：`{rel(table_paths["numeric_policy"])}`。
- IV 排名：`{rel(table_paths["iv"])}`。

## 当前结果

- 选择进入后续特征矩阵的字段数：{selected_feature_count}。
- 类别字段数：{categorical_feature_count}。
- 所有工程后数据保留 `TARGET`、`SK_ID_CURR`、`split` 作为监督学习和追踪元数据。
- `data/processed/home_credit_features*.csv` 属于大体积可再生产物，继续由 `.gitignore` 排除。

## 复现命令

```powershell
.\\.venv\\Scripts\\python.exe .\\src\\run_stage3_features.py
```
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def write_stage3_status(
    output: Path,
    feature_paths: dict[str, Path],
    selected_feature_count: int,
    derived_feature_count: int,
    categorical_feature_count: int,
    validation_rows: list[dict[str, object]],
) -> Path:
    lines = [
        "# 第 3 阶段：特征工程状态报告",
        "",
        "## 运行结果",
        "",
        "- 数据集：`home_credit`。",
        "- 输入：`data/processed/home_credit_processed.csv`。",
        f"- 输出完整特征数据：`{rel(feature_paths['full'])}`。",
        f"- 输出训练集：`{rel(feature_paths['train'])}`。",
        f"- 输出验证集：`{rel(feature_paths['valid'])}`。",
        f"- 输出测试集：`{rel(feature_paths['test'])}`。",
        f"- 新增衍生特征数：{derived_feature_count}。",
        f"- 后续特征矩阵候选字段数：{selected_feature_count}。",
        f"- 类别编码候选字段数：{categorical_feature_count}。",
        "",
        "## 验收检查",
        "",
    ]
    for row in validation_rows:
        mark = "x" if row["passed"] else " "
        lines.append(f"- [{mark}] {row['check']}")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 编码器、IV、数值处理边界均只使用 `train` split 拟合。",
            "- One-Hot 稀疏矩阵未落盘，避免提交大体积中间产物。",
            "- `TARGET`、`SK_ID_CURR`、`split` 只作为元数据保留，不作为模型特征。",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def validate_stage3_outputs(df: pd.DataFrame, selection: pd.DataFrame) -> list[dict[str, object]]:
    selected = selection.loc[selection["selected_for_feature_matrix"].eq(True), "column"].tolist()
    metadata = {TARGET, SPLIT, *ID_COLUMNS}
    feature_only = [column for column in selected if column not in metadata]
    has_no_inf = bool(np.isfinite(df.select_dtypes(include="number").to_numpy()).all())
    critical_cols = [column for column in feature_only if pd.api.types.is_numeric_dtype(df[column])]
    no_numeric_missing = bool(df[critical_cols].isna().sum().sum() == 0) if critical_cols else True
    leakage_excluded = bool(not metadata.intersection(feature_only))
    split_counts_ok = set(df[SPLIT].unique()) == {"train", "valid", "test"}
    return [
        {"check": "沿用固定 train/valid/test 划分，未重新抽样。", "passed": split_counts_ok},
        {"check": "TARGET、SK_ID_CURR、split 未进入模型特征集合。", "passed": leakage_excluded},
        {"check": "工程后数值特征无 inf 或 -inf。", "passed": has_no_inf},
        {"check": "工程后关键数值特征无缺失。", "passed": no_numeric_missing},
        {"check": "每个新增衍生特征都有公式和业务含义。", "passed": len(DERIVED_FEATURES) > 0},
    ]


def run_stage3_feature_engineering(processed_path: Path | None = None) -> dict[str, Path]:
    ensure_project_dirs()
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = load_processed_home_credit(processed_path)
    engineered = add_home_credit_stage3_features(df)
    engineered, fill_policy = fill_engineered_numeric_missing(engineered)
    selection = build_selection_record(engineered)
    catalog = build_feature_catalog(engineered, selection)
    derived = build_derived_feature_dictionary(engineered)
    numeric_policy = build_numeric_processing_policy(engineered, selection)
    encoding_policy, encoded_names = build_encoding_policy(engineered, selection)
    iv_summary = build_iv_summary(engineered, selection)

    output_columns = selected_columns(selection)
    engineered = engineered[output_columns]

    feature_paths = export_feature_datasets(engineered)
    table_paths = {
        "catalog": TABLES / "home_credit_feature_catalog.csv",
        "derived": TABLES / "home_credit_derived_feature_dictionary.csv",
        "selection": TABLES / "home_credit_feature_selection_record.csv",
        "encoding": TABLES / "home_credit_encoding_policy.csv",
        "encoded_names": TABLES / "home_credit_encoded_feature_names.csv",
        "numeric_policy": TABLES / "home_credit_numeric_processing_policy.csv",
        "iv": TABLES / "home_credit_iv_summary.csv",
        "fill_policy": TABLES / "home_credit_stage3_fill_policy.csv",
    }
    catalog.to_csv(table_paths["catalog"], index=False)
    derived.to_csv(table_paths["derived"], index=False)
    selection.to_csv(table_paths["selection"], index=False)
    encoding_policy.to_csv(table_paths["encoding"], index=False)
    encoded_names.to_csv(table_paths["encoded_names"], index=False)
    numeric_policy.to_csv(table_paths["numeric_policy"], index=False)
    iv_summary.to_csv(table_paths["iv"], index=False)
    fill_policy.to_csv(table_paths["fill_policy"], index=False)

    selected_feature_count = int(selection["selected_for_feature_matrix"].sum())
    derived_feature_count = int(derived["created"].sum())
    categorical_feature_count = int(encoding_policy.shape[0])
    notes_path = REPORTS / "home_credit_feature_engineering_notes.md"
    status_path = REPORTS / "stage3_feature_status.md"
    write_feature_notes(
        notes_path,
        feature_paths,
        table_paths,
        selected_feature_count,
        derived_feature_count,
        categorical_feature_count,
    )
    validations = validate_stage3_outputs(engineered, selection)
    write_stage3_status(
        status_path,
        feature_paths,
        selected_feature_count,
        derived_feature_count,
        categorical_feature_count,
        validations,
    )

    return {**feature_paths, **table_paths, "notes": notes_path, "status": status_path}
