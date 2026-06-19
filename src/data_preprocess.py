from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DATA_PROCESSED, DATA_RAW, ensure_project_dirs, rel


RANDOM_STATE = 42
MISSING_CATEGORY = "__MISSING__"
RARE_CATEGORY = "__RARE__"
RARE_CATEGORY_MIN_COUNT = 50
RARE_CATEGORY_MIN_RATE = 0.001

HOME_CREDIT_DERIVED_FEATURES = {
    "DAYS_EMPLOYED_ANOMALY": {
        "description": "Derived flag: raw DAYS_EMPLOYED used Home Credit's 365243 sentinel for missing employment history.",
        "formula": "1 if raw DAYS_EMPLOYED == 365243 else 0",
        "action": "candidate feature; keep as an anomaly/missingness indicator",
    },
    "credit_income_ratio": {
        "description": "Derived feature: credit amount divided by applicant income.",
        "formula": "AMT_CREDIT / AMT_INCOME_TOTAL",
        "action": "candidate feature; recompute from application train/test fields before modeling",
    },
    "annuity_income_ratio": {
        "description": "Derived feature: loan annuity divided by applicant income.",
        "formula": "AMT_ANNUITY / AMT_INCOME_TOTAL",
        "action": "candidate feature; recompute from application train/test fields before modeling",
    },
    "goods_credit_ratio": {
        "description": "Derived feature: goods price divided by credit amount.",
        "formula": "AMT_GOODS_PRICE / AMT_CREDIT",
        "action": "candidate feature; recompute from application train/test fields before modeling",
    },
    "employment_age_ratio": {
        "description": "Derived feature: employment days divided by birth days, both relative to application date.",
        "formula": "DAYS_EMPLOYED / DAYS_BIRTH",
        "action": "candidate feature; recompute from application train/test fields before modeling",
    },
}

HOME_CREDIT_AUXILIARY_TABLES = [
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    source_path: Path
    frame: pd.DataFrame
    target: str
    id_columns: list[str]


def find_home_credit_train() -> Path | None:
    candidates = [
        DATA_RAW / "home_credit" / "application_train.csv",
        DATA_RAW / "application_train.csv",
    ]
    return next((path for path in candidates if path.exists()), None)


def find_taiwan_file() -> Path | None:
    candidates = [
        DATA_RAW / "taiwan" / "default of credit card clients.csv",
        DATA_RAW / "taiwan" / "default of credit card clients.xls",
        DATA_RAW / "default of credit card clients.csv",
        DATA_RAW / "default of credit card clients.xls",
    ]
    return next((path for path in candidates if path.exists()), None)


def load_home_credit(path: Path) -> DatasetBundle:
    df = pd.read_csv(path)
    if "TARGET" not in df.columns:
        raise ValueError(f"{rel(path)} does not contain TARGET.")
    df = clean_home_credit_application(df)
    df = add_home_credit_features(df)
    return DatasetBundle(
        name="home_credit",
        source_path=path,
        frame=df,
        target="TARGET",
        id_columns=[col for col in ["SK_ID_CURR"] if col in df.columns],
    )


def load_taiwan(path: Path) -> DatasetBundle:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        try:
            df = pd.read_excel(path, header=1)
        except ImportError as exc:
            raise RuntimeError(
                "Reading the UCI .xls file requires xlrd. Install dependencies with "
                "`pip install -r requirements.txt`, or provide a CSV copy in data/raw/taiwan/."
            ) from exc

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    rename_map = {
        "default payment next month": "default_payment_next_month",
        "PAY_0": "PAY_1",
    }
    df = df.rename(columns=rename_map)
    target = "default_payment_next_month"
    if target not in df.columns:
        possible_targets = [col for col in df.columns if "default" in str(col).lower()]
        if not possible_targets:
            raise ValueError(f"Could not identify target column in {rel(path)}.")
        target = possible_targets[0]

    df = add_taiwan_features(df)
    return DatasetBundle(
        name="taiwan_credit",
        source_path=path,
        frame=df,
        target=target,
        id_columns=[col for col in ["ID"] if col in df.columns],
    )


def load_available_dataset() -> DatasetBundle:
    home_credit = find_home_credit_train()
    if home_credit:
        return load_home_credit(home_credit)

    taiwan = find_taiwan_file()
    if taiwan:
        return load_taiwan(taiwan)

    raise FileNotFoundError(
        "No raw dataset found. Put Home Credit application_train.csv under "
        "data/raw/home_credit/ or run `python src/download_data.py --uci-only` "
        "for the UCI Taiwan dataset."
    )


def clean_home_credit_application(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "DAYS_EMPLOYED" in df.columns:
        anomaly = df["DAYS_EMPLOYED"].eq(365243)
        df["DAYS_EMPLOYED_ANOMALY"] = anomaly.astype("int8")
        df.loc[anomaly, "DAYS_EMPLOYED"] = np.nan

    if "CODE_GENDER" in df.columns:
        df.loc[df["CODE_GENDER"].eq("XNA"), "CODE_GENDER"] = np.nan

    for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT"]:
        if col in df.columns:
            df.loc[df[col].le(0), col] = np.nan

    for col in ["AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        if col in df.columns:
            df.loc[df[col].lt(0), col] = np.nan

    return df


def add_home_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    safe_divisions = {
        "credit_income_ratio": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        "annuity_income_ratio": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        "goods_credit_ratio": ("AMT_GOODS_PRICE", "AMT_CREDIT"),
        "employment_age_ratio": ("DAYS_EMPLOYED", "DAYS_BIRTH"),
    }
    for new_col, (num, denom) in safe_divisions.items():
        if num in df.columns and denom in df.columns:
            denominator = df[denom].replace({0: np.nan})
            df[new_col] = df[num] / denominator
    return df


def add_taiwan_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bill_cols = [col for col in df.columns if str(col).startswith("BILL_AMT")]
    pay_cols = [col for col in df.columns if str(col).startswith("PAY_AMT")]
    if bill_cols:
        df["bill_total"] = df[bill_cols].sum(axis=1)
        if "LIMIT_BAL" in df.columns:
            df["bill_to_limit_ratio"] = df["bill_total"] / df["LIMIT_BAL"].replace({0: np.nan})
    if pay_cols:
        df["payment_total"] = df[pay_cols].sum(axis=1)
    if bill_cols and pay_cols:
        df["payment_to_bill_ratio"] = df["payment_total"] / df["bill_total"].replace({0: np.nan})
    return df


def add_split_column(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    split = pd.Series(index=df.index, dtype="object")

    for _, group in df.groupby(target, dropna=False):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n = len(indices)
        train_end = int(n * 0.70)
        valid_end = int(n * 0.85)
        split.loc[indices[:train_end]] = "train"
        split.loc[indices[train_end:valid_end]] = "valid"
        split.loc[indices[valid_end:]] = "test"

    out = df.copy()
    out["split"] = split
    return out


def apply_missing_value_policy(
    df: pd.DataFrame,
    target: str,
    id_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    train_mask = out["split"].eq("train")
    excluded = set(id_columns + [target, "split"])
    policy_rows = []

    for col in [column for column in out.columns if column not in excluded]:
        missing_before = int(out[col].isna().sum())
        train_missing_before = int(out.loc[train_mask, col].isna().sum())
        if pd.api.types.is_numeric_dtype(out[col]):
            fill_value = out.loc[train_mask, col].median()
            if pd.isna(fill_value):
                fill_value = 0.0
            out[col] = out[col].fillna(fill_value)
            policy_rows.append(
                {
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "strategy": "train_median",
                    "fill_value": float(fill_value),
                    "missing_before": missing_before,
                    "train_missing_before": train_missing_before,
                    "rare_categories_grouped": 0,
                    "rare_rows_grouped": 0,
                    "reason": "Numeric missing values are filled with the training-split median to avoid validation/test leakage.",
                }
            )
            continue

        series = out[col].astype("string").fillna(MISSING_CATEGORY)
        train_values = series.loc[train_mask]
        train_counts = train_values.value_counts(dropna=False)
        min_count = max(RARE_CATEGORY_MIN_COUNT, int(len(train_values) * RARE_CATEGORY_MIN_RATE))
        rare_values = set(train_counts[train_counts < min_count].index.astype(str))
        rare_values.discard(MISSING_CATEGORY)
        rare_mask = series.astype(str).isin(rare_values)
        out[col] = series.astype(str).mask(rare_mask, RARE_CATEGORY).astype("object")
        policy_rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "strategy": "fill_missing_and_group_train_rare_categories",
                "fill_value": MISSING_CATEGORY,
                "missing_before": missing_before,
                "train_missing_before": train_missing_before,
                "rare_categories_grouped": len(rare_values),
                "rare_rows_grouped": int(rare_mask.sum()),
                "reason": "Categorical missing values use a stable sentinel; rare levels are grouped using training-split counts.",
            }
        )

    return out, pd.DataFrame(policy_rows)


def export_split_files(bundle: DatasetBundle, processed: pd.DataFrame) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for split in ["train", "valid", "test"]:
        output = DATA_PROCESSED / f"{bundle.name}_{split}.csv"
        processed.loc[processed["split"].eq(split)].to_csv(output, index=False)
        paths[split] = output
    return paths


def save_processed_dataset(bundle: DatasetBundle, table_dir: Path | None = None) -> Path:
    ensure_project_dirs()
    processed = add_split_column(bundle.frame, bundle.target)
    processed, imputation_policy = apply_missing_value_policy(processed, bundle.target, bundle.id_columns)
    output = DATA_PROCESSED / f"{bundle.name}_processed.csv"
    processed.to_csv(output, index=False)
    export_split_files(bundle, processed)
    if table_dir is not None:
        table_dir.mkdir(parents=True, exist_ok=True)
        imputation_policy.to_csv(table_dir / f"{bundle.name}_missing_imputation_policy.csv", index=False)
    return output


def write_cleaning_audit(bundle: DatasetBundle, output: Path) -> Path:
    rows = [
        {
            "rule_id": "fixed_stratified_split",
            "column": "split",
            "affected_rows": len(bundle.frame),
            "action": "create train/valid/test split with seed 42 and target stratification",
            "reason": "A fixed split keeps all downstream model comparisons fair and reproducible.",
        }
    ]

    if bundle.name == "home_credit":
        available_columns = set(pd.read_csv(bundle.source_path, nrows=0).columns)
        audit_columns = [
            col
            for col in [
                "DAYS_EMPLOYED",
                "CODE_GENDER",
                "AMT_INCOME_TOTAL",
                "AMT_CREDIT",
                "AMT_ANNUITY",
                "AMT_GOODS_PRICE",
            ]
            if col in available_columns
        ]
        raw = pd.read_csv(bundle.source_path, usecols=audit_columns)
        if "DAYS_EMPLOYED" in raw.columns:
            rows.append(
                {
                    "rule_id": "home_credit_days_employed_sentinel",
                    "column": "DAYS_EMPLOYED",
                    "affected_rows": int(raw["DAYS_EMPLOYED"].eq(365243).sum()),
                    "action": "replace raw 365243 sentinel with NaN and add DAYS_EMPLOYED_ANOMALY",
                    "reason": "365243 is Home Credit's special missing-employment marker, not a real employment duration.",
                }
            )
        if "CODE_GENDER" in raw.columns:
            rows.append(
                {
                    "rule_id": "home_credit_code_gender_xna",
                    "column": "CODE_GENDER",
                    "affected_rows": int(raw["CODE_GENDER"].eq("XNA").sum()),
                    "action": "replace XNA with missing before categorical imputation",
                    "reason": "XNA is an invalid gender category and should not be treated as a normal level.",
                }
            )
        for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT"]:
            if col in raw.columns:
                rows.append(
                    {
                        "rule_id": f"{col.lower()}_non_positive",
                        "column": col,
                        "affected_rows": int(raw[col].le(0).sum()),
                        "action": "replace non-positive values with NaN before ratio features and imputation",
                        "reason": "These amount fields are used as denominators or core financial measures.",
                    }
                )
        for col in ["AMT_ANNUITY", "AMT_GOODS_PRICE"]:
            if col in raw.columns:
                rows.append(
                    {
                        "rule_id": f"{col.lower()}_negative",
                        "column": col,
                        "affected_rows": int(raw[col].lt(0).sum()),
                        "action": "replace negative values with NaN before imputation",
                        "reason": "Negative loan annuity or goods price would be invalid for this application table.",
                    }
                )

    rows.extend(
        [
            {
                "rule_id": "numeric_missing_values",
                "column": "numeric feature columns",
                "affected_rows": "",
                "action": "fill with training-split median; values recorded in missing imputation policy table",
                "reason": "The imputation statistic must be learned from train only to avoid validation/test leakage.",
            },
            {
                "rule_id": "categorical_missing_and_rare_values",
                "column": "categorical feature columns",
                "affected_rows": "",
                "action": f"fill missing with {MISSING_CATEGORY}; group train-rare levels as {RARE_CATEGORY}",
                "reason": "Stable category handling prevents sparse one-off levels from dominating later encoders.",
            },
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def write_data_cleaning_notes(
    bundle: DatasetBundle,
    output: Path,
    processed_path: Path,
    split_summary_path: Path,
    table_dir: Path,
    figure_dir: Path,
    notebook_path: Path,
) -> Path:
    split_paths = {
        split: DATA_PROCESSED / f"{bundle.name}_{split}.csv" for split in ["train", "valid", "test"]
    }
    text = f"""# {bundle.name} EDA 与数据清洗说明

## 目标

本说明用于补齐第 2 阶段“EDA 与数据清洗”的可复现证据。当前数据集为 `{bundle.name}`，目标变量为 `{bundle.target}`，原始训练表路径为 `{rel(bundle.source_path)}`。

## 固定划分

- 划分方式：按 `{bundle.target}` 分层随机划分。
- 随机种子：`42`。
- 比例：训练集 70%，验证集 15%，测试集 15%。
- 划分摘要：`{rel(split_summary_path)}`。
- 完整 processed 数据：`{rel(processed_path)}`。
- 独立划分文件：`{rel(split_paths["train"])}`、`{rel(split_paths["valid"])}`、`{rel(split_paths["test"])}`。

## 清洗规则

1. `TARGET` 只作为标签使用，不进入特征矩阵。
2. ID 字段 `{", ".join(bundle.id_columns) if bundle.id_columns else "无"}` 只用于追踪和关联，不进入模型特征。
3. Home Credit 中 `DAYS_EMPLOYED = 365243` 视为缺失哨兵值，替换为缺失，并新增 `DAYS_EMPLOYED_ANOMALY` 标记。
4. Home Credit 中 `CODE_GENDER = XNA` 视为无效类别，进入类别缺失处理。
5. 金额类字段中不合理的非正/负值先置为缺失，再按训练集统计量处理。
6. 数值特征缺失值使用训练集 median 填补，填补值记录在 `{rel(table_dir / f"{bundle.name}_missing_imputation_policy.csv")}`。
7. 类别特征缺失值填为 `{MISSING_CATEGORY}`；训练集中低频类别归并为 `{RARE_CATEGORY}`，避免后续编码过度稀疏。

## EDA 产物

- 数据集摘要：`{rel(table_dir / f"{bundle.name}_dataset_summary.csv")}`。
- 缺失率统计：`{rel(table_dir / f"{bundle.name}_missing_values.csv")}`。
- 数值变量摘要：`{rel(table_dir / f"{bundle.name}_numeric_summary.csv")}`。
- 异常值与偏态报告：`{rel(table_dir / f"{bundle.name}_outlier_invalid_report.csv")}`。
- 违约/非违约数值对比：`{rel(table_dir / f"{bundle.name}_target_numeric_comparison.csv")}`。
- 类别变量目标率对比：`{rel(table_dir / f"{bundle.name}_target_categorical_comparison.csv")}`。
- 稀有类别清单：`{rel(table_dir / f"{bundle.name}_rare_categories.csv")}`。
- 清洗动作审计：`{rel(table_dir / f"{bundle.name}_cleaning_audit.csv")}`。
- EDA notebook：`{rel(notebook_path)}`。
- EDA 图表目录：`{rel(figure_dir)}`。

## 复现方式

```powershell
$env:TRAIN_DEVICE="gpu"
.\\.venv\\Scripts\\python.exe .\\src\\run_week1_2.py
```

## 注意事项

- EDA 表保留数据质量视角，因此会展示原始缺失、偏态和异常信号。
- processed 数据是建模入口，缺失填补和低频类别归并只使用训练集统计量。
- 当前基线模型仍只使用数值特征；类别特征清洗是为后续 One-Hot、WOE/IV 或 embedding 实验预留。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def home_credit_raw_dir(bundle: DatasetBundle) -> Path:
    if bundle.name == "home_credit":
        return bundle.source_path.parent
    return DATA_RAW / "home_credit"


def load_home_credit_official_descriptions(bundle: DatasetBundle) -> pd.DataFrame:
    description_path = home_credit_raw_dir(bundle) / "HomeCredit_columns_description.csv"
    columns = ["source_table", "column", "official_description", "official_special"]
    if not description_path.exists():
        return pd.DataFrame(columns=columns)

    descriptions = pd.read_csv(description_path, encoding="latin1")
    descriptions = descriptions.rename(
        columns={
            "Table": "source_table",
            "Row": "column",
            "Description": "official_description",
            "Special": "official_special",
        }
    )
    descriptions = descriptions[columns]
    descriptions["column"] = descriptions["column"].astype(str)

    application_rows = descriptions[
        descriptions["source_table"].astype(str).eq("application_{train|test}.csv")
    ].copy()
    return application_rows.drop_duplicates("column", keep="first")


def home_credit_application_columns(bundle: DatasetBundle) -> tuple[set[str], set[str]]:
    raw_dir = home_credit_raw_dir(bundle)
    train_path = raw_dir / "application_train.csv"
    test_path = raw_dir / "application_test.csv"
    train_columns = set(pd.read_csv(train_path, nrows=0).columns) if train_path.exists() else set()
    test_columns = set(pd.read_csv(test_path, nrows=0).columns) if test_path.exists() else set()
    return train_columns, test_columns


def modeling_action_for_field(column: str, role: str, is_in_train: bool, is_in_test: bool) -> str:
    if role == "target":
        return "use as label only; exclude from model features"
    if role == "id":
        return "use only for joins/tracking; exclude from model features"
    if column in HOME_CREDIT_DERIVED_FEATURES:
        return HOME_CREDIT_DERIVED_FEATURES[column]["action"]
    if is_in_train and not is_in_test:
        return "exclude from model features because the field is not available in application_test.csv"
    return "candidate feature after normal preprocessing and leakage review"


def write_field_dictionary(bundle: DatasetBundle, output: Path) -> None:
    rows = []
    df = bundle.frame
    descriptions = pd.DataFrame()
    train_columns: set[str] = set()
    test_columns: set[str] = set()
    if bundle.name == "home_credit":
        descriptions = load_home_credit_official_descriptions(bundle)
        train_columns, test_columns = home_credit_application_columns(bundle)

    for col in df.columns:
        role = "target" if col == bundle.target else "id" if col in bundle.id_columns else "feature"
        is_in_train = col in train_columns if bundle.name == "home_credit" else ""
        is_in_test = col in test_columns if bundle.name == "home_credit" else ""
        derived = HOME_CREDIT_DERIVED_FEATURES.get(col, {})
        rows.append(
            {
                "column": col,
                "role": role,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isna().sum()),
                "missing_rate": float(df[col].isna().mean()),
                "unique_count": int(df[col].nunique(dropna=True)),
                "is_in_application_train": is_in_train,
                "is_in_application_test": is_in_test,
                "derived_formula": derived.get("formula", ""),
                "modeling_action": modeling_action_for_field(col, role, bool(is_in_train), bool(is_in_test))
                if bundle.name == "home_credit"
                else "",
            }
        )
    field_dictionary = pd.DataFrame(rows)
    if bundle.name == "home_credit":
        field_dictionary = field_dictionary.merge(descriptions, on="column", how="left")
        field_dictionary["source_table"] = field_dictionary["source_table"].fillna("")
        field_dictionary["official_description"] = field_dictionary["official_description"].fillna("")
        field_dictionary["official_special"] = field_dictionary["official_special"].fillna("")
        for col, metadata in HOME_CREDIT_DERIVED_FEATURES.items():
            mask = field_dictionary["column"].eq(col)
            field_dictionary.loc[mask, "source_table"] = "derived_from_application_{train|test}.csv"
            field_dictionary.loc[mask, "official_description"] = metadata["description"]
            field_dictionary.loc[mask, "official_special"] = "project_derived"

        ordered_columns = [
            "column",
            "role",
            "dtype",
            "missing_count",
            "missing_rate",
            "unique_count",
            "source_table",
            "official_description",
            "official_special",
            "is_in_application_train",
            "is_in_application_test",
            "derived_formula",
            "modeling_action",
        ]
        field_dictionary = field_dictionary[ordered_columns]
    field_dictionary.to_csv(output, index=False)


def write_home_credit_leakage_risk_fields(bundle: DatasetBundle, output: Path) -> Path:
    if bundle.name != "home_credit":
        raise ValueError("Leakage risk field output is only defined for Home Credit.")

    train_columns, test_columns = home_credit_application_columns(bundle)
    train_only = sorted(train_columns - test_columns)
    test_only = sorted(test_columns - train_columns)
    rows = [
        {
            "column_or_source": "TARGET",
            "risk_type": "label_leakage",
            "availability": "application_train_only",
            "risk_level": "high",
            "modeling_action": "use only as y; exclude from feature matrix",
            "reason": "TARGET is the supervised label and is absent from application_test.csv.",
        },
        {
            "column_or_source": "SK_ID_CURR",
            "risk_type": "identifier",
            "availability": "application_train_and_test",
            "risk_level": "medium",
            "modeling_action": "use only for joins, tracking, and output alignment; exclude from model features",
            "reason": "High-cardinality loan ID can let a model memorize sample identity instead of learning risk patterns.",
        },
        {
            "column_or_source": "application_train_only_columns",
            "risk_type": "test_unavailable_columns",
            "availability": "; ".join(train_only) if train_only else "none",
            "risk_level": "high" if train_only else "low",
            "modeling_action": "exclude all train-only columns from model features",
            "reason": "Fields unavailable in application_test.csv cannot be used for deployable prediction.",
        },
        {
            "column_or_source": "application_test_only_columns",
            "risk_type": "schema_difference",
            "availability": "; ".join(test_only) if test_only else "none",
            "risk_level": "low" if not test_only else "medium",
            "modeling_action": "review before feature engineering",
            "reason": "Test-only columns would need a train-side equivalent or must be ignored.",
        },
    ]

    for col, metadata in HOME_CREDIT_DERIVED_FEATURES.items():
        rows.append(
            {
                "column_or_source": col,
                "risk_type": "derived_feature",
                "availability": "derived from application train/test fields",
                "risk_level": "low",
                "modeling_action": metadata["action"],
                "reason": f"Formula: {metadata['formula']}. Inputs are application-level fields available before decision time.",
            }
        )

    for table in HOME_CREDIT_AUXILIARY_TABLES:
        rows.append(
            {
                "column_or_source": table,
                "risk_type": "temporal_leakage_risk",
                "availability": "raw auxiliary table",
                "risk_level": "medium",
                "modeling_action": "only aggregate historical information available before the current application",
                "reason": "Auxiliary records may include time-relative balances, payments, or statuses; future events must not enter features.",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
