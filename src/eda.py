from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from data_preprocess import DatasetBundle


CORE_NUMERIC_BY_DATASET = {
    "home_credit": [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "credit_income_ratio",
        "annuity_income_ratio",
        "goods_credit_ratio",
        "employment_age_ratio",
    ],
    "taiwan_credit": [
        "LIMIT_BAL",
        "AGE",
        "PAY_1",
        "PAY_2",
        "bill_total",
        "payment_total",
        "bill_to_limit_ratio",
        "payment_to_bill_ratio",
    ],
}

CORE_CATEGORICAL_BY_DATASET = {
    "home_credit": [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
    ],
    "taiwan_credit": ["SEX", "EDUCATION", "MARRIAGE"],
}


def matplotlib_available() -> bool:
    return importlib.util.find_spec("matplotlib") is not None


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def existing_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in df.columns]


def binary_target_values(df: pd.DataFrame, target: str) -> list:
    values = sorted(df[target].dropna().unique().tolist())
    return values[:2] if len(values) >= 2 else values


def write_eda_tables(bundle: DatasetBundle, table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    df = bundle.frame
    target = bundle.target

    dataset_summary = pd.DataFrame(
        [
            {
                "dataset": bundle.name,
                "source_path": str(bundle.source_path),
                "rows": len(df),
                "columns": len(df.columns),
                "target": target,
                "positive_count": int(df[target].sum()) if pd.api.types.is_numeric_dtype(df[target]) else None,
                "positive_rate": float(df[target].mean()) if pd.api.types.is_numeric_dtype(df[target]) else None,
                "duplicate_rows": int(df.duplicated().sum()),
            }
        ]
    )
    dataset_summary.to_csv(table_dir / f"{bundle.name}_dataset_summary.csv", index=False)

    missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_rate=lambda x: x["missing_count"] / len(df))
        .sort_values(["missing_count", "missing_rate"], ascending=False)
    )
    missing.to_csv(table_dir / f"{bundle.name}_missing_values.csv")

    target_dist = (
        df[target]
        .value_counts(dropna=False)
        .rename_axis(target)
        .reset_index(name="count")
        .assign(rate=lambda x: x["count"] / len(df))
    )
    target_dist.to_csv(table_dir / f"{bundle.name}_target_distribution.csv", index=False)

    numeric = df.select_dtypes(include="number")
    numeric.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.to_csv(
        table_dir / f"{bundle.name}_numeric_summary.csv"
    )

    numeric_rows = []
    for col in [column for column in numeric.columns if column not in bundle.id_columns + [target]]:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        non_missing = series.dropna()
        numeric_rows.append(
            {
                "column": col,
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "zero_count": int(series.eq(0).sum()),
                "negative_count": int(series.lt(0).sum()),
                "skewness": float(non_missing.skew()) if len(non_missing) > 2 else np.nan,
                "q1": float(q1) if pd.notna(q1) else np.nan,
                "q3": float(q3) if pd.notna(q3) else np.nan,
                "iqr": float(iqr) if pd.notna(iqr) else np.nan,
                "iqr_lower_fence": float(lower) if pd.notna(lower) else np.nan,
                "iqr_upper_fence": float(upper) if pd.notna(upper) else np.nan,
                "iqr_outlier_count": int(series.lt(lower).sum() + series.gt(upper).sum())
                if pd.notna(iqr)
                else 0,
                "iqr_outlier_rate": float((series.lt(lower).sum() + series.gt(upper).sum()) / len(series))
                if pd.notna(iqr)
                else 0.0,
                "special_note": "raw 365243 sentinel replaced; see cleaning audit"
                if col == "DAYS_EMPLOYED"
                else "",
            }
        )
    pd.DataFrame(numeric_rows).to_csv(table_dir / f"{bundle.name}_outlier_invalid_report.csv", index=False)

    target_values = binary_target_values(df, target)
    comparison_rows = []
    if len(target_values) == 2:
        low_target, high_target = target_values
        comparison_cols = existing_columns(df, CORE_NUMERIC_BY_DATASET.get(bundle.name, []))
        if not comparison_cols:
            comparison_cols = [col for col in numeric.columns if col not in bundle.id_columns + [target]][:20]
        for col in comparison_cols:
            group0 = df.loc[df[target].eq(low_target), col]
            group1 = df.loc[df[target].eq(high_target), col]
            mean0 = group0.mean()
            mean1 = group1.mean()
            comparison_rows.append(
                {
                    "column": col,
                    f"mean_target_{low_target}": float(mean0) if pd.notna(mean0) else np.nan,
                    f"mean_target_{high_target}": float(mean1) if pd.notna(mean1) else np.nan,
                    f"median_target_{low_target}": float(group0.median()) if group0.notna().any() else np.nan,
                    f"median_target_{high_target}": float(group1.median()) if group1.notna().any() else np.nan,
                    f"missing_rate_target_{low_target}": float(group0.isna().mean()),
                    f"missing_rate_target_{high_target}": float(group1.isna().mean()),
                    "mean_diff_high_minus_low": float(mean1 - mean0) if pd.notna(mean0) and pd.notna(mean1) else np.nan,
                    "relative_mean_diff": float((mean1 - mean0) / abs(mean0))
                    if pd.notna(mean0) and pd.notna(mean1) and mean0 != 0
                    else np.nan,
                }
            )
    pd.DataFrame(comparison_rows).to_csv(
        table_dir / f"{bundle.name}_target_numeric_comparison.csv", index=False
    )

    category_rows = []
    target_rate_rows = []
    rare_rows = []
    overall_target_rate = float(df[target].mean()) if pd.api.types.is_numeric_dtype(df[target]) else np.nan
    for col in df.select_dtypes(exclude="number").columns:
        values = df[col].fillna("__MISSING__").astype(str)
        counts = values.value_counts(dropna=False)
        category_rows.append(
            {
                "column": col,
                "unique_count": int(df[col].nunique(dropna=True)),
                "top_value": df[col].mode(dropna=True).iloc[0] if not df[col].mode(dropna=True).empty else None,
                "top_count": int(df[col].value_counts(dropna=True).iloc[0])
                if not df[col].value_counts(dropna=True).empty
                else 0,
            }
        )
        min_count = max(50, int(len(df) * 0.001))
        for value, count in counts.items():
            rare_rows.append(
                {
                    "column": col,
                    "category": value,
                    "count": int(count),
                    "rate": float(count / len(df)),
                    "is_rare": bool(count < min_count),
                    "rare_threshold_count": min_count,
                }
            )
        for value, count in counts.head(20).items():
            mask = values.eq(value)
            target_rate = df.loc[mask, target].mean() if pd.api.types.is_numeric_dtype(df[target]) else np.nan
            target_rate_rows.append(
                {
                    "column": col,
                    "category": value,
                    "count": int(count),
                    "rate": float(count / len(df)),
                    "target_rate": float(target_rate) if pd.notna(target_rate) else np.nan,
                    "target_rate_lift": float(target_rate / overall_target_rate)
                    if pd.notna(target_rate) and overall_target_rate
                    else np.nan,
                }
            )
    pd.DataFrame(category_rows).to_csv(table_dir / f"{bundle.name}_categorical_summary.csv", index=False)
    pd.DataFrame(target_rate_rows).to_csv(
        table_dir / f"{bundle.name}_target_categorical_comparison.csv", index=False
    )
    pd.DataFrame(rare_rows).to_csv(table_dir / f"{bundle.name}_rare_categories.csv", index=False)


def write_eda_figures(bundle: DatasetBundle, figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not matplotlib_available():
        return ["matplotlib is not installed; EDA figures were skipped."]

    os.environ.setdefault("WINDIR", r"C:\Windows")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(figure_dir.parent / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    df = bundle.frame
    target = bundle.target
    messages: list[str] = []

    target_counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    target_counts.plot(kind="bar", ax=ax, color=["#4E79A7", "#E15759"][: len(target_counts)])
    ax.set_title(f"{bundle.name} target distribution")
    ax.set_xlabel(target)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{bundle.name}_target_distribution.png", dpi=160)
    plt.close(fig)

    missing = df.isna().mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 5))
    missing.sort_values().plot(kind="barh", ax=ax, color="#F28E2B")
    ax.set_title(f"{bundle.name} top missing-value rates")
    ax.set_xlabel("missing rate")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{bundle.name}_missing_top20.png", dpi=160)
    plt.close(fig)

    numeric_cols = [col for col in df.select_dtypes(include="number").columns if col != target][:8]
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[col].dropna().sample(min(5000, df[col].dropna().shape[0]), random_state=42).hist(ax=ax, bins=40)
        ax.set_title(f"{bundle.name}: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{bundle.name}_{col}_hist.png", dpi=160)
        plt.close(fig)

    target_values = binary_target_values(df, target)
    core_numeric = existing_columns(df, CORE_NUMERIC_BY_DATASET.get(bundle.name, []))
    if len(target_values) == 2 and core_numeric:
        low_target, high_target = target_values
        comparison_rows = []
        for col in core_numeric:
            mean0 = df.loc[df[target].eq(low_target), col].mean()
            mean1 = df.loc[df[target].eq(high_target), col].mean()
            if pd.notna(mean0) and pd.notna(mean1):
                comparison_rows.append((col, mean1 - mean0))
        if comparison_rows:
            comparison = (
                pd.DataFrame(comparison_rows, columns=["column", "mean_diff"])
                .assign(abs_diff=lambda x: x["mean_diff"].abs())
                .sort_values("abs_diff", ascending=False)
                .head(10)
                .sort_values("mean_diff")
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(comparison["column"], comparison["mean_diff"], color="#59A14F")
            ax.axvline(0, color="#444444", linewidth=1)
            ax.set_title(f"{bundle.name} target=1 minus target=0 mean difference")
            ax.set_xlabel("mean difference")
            fig.tight_layout()
            fig.savefig(figure_dir / f"{bundle.name}_target_numeric_mean_diff_top10.png", dpi=160)
            plt.close(fig)

        for col in core_numeric[:6]:
            groups = []
            labels = []
            for value in target_values:
                sample = df.loc[df[target].eq(value), col].dropna()
                if sample.empty:
                    continue
                groups.append(sample.sample(min(3000, len(sample)), random_state=42))
                labels.append(str(value))
            if len(groups) < 2:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            try:
                ax.boxplot(groups, tick_labels=labels, showfliers=False)
            except TypeError:
                ax.boxplot(groups, labels=labels, showfliers=False)
            ax.set_title(f"{bundle.name}: {col} by {target}")
            ax.set_xlabel(target)
            ax.set_ylabel(col)
            fig.tight_layout()
            fig.savefig(figure_dir / f"{bundle.name}_target_compare_{safe_name(col)}.png", dpi=160)
            plt.close(fig)

    core_categorical = existing_columns(df, CORE_CATEGORICAL_BY_DATASET.get(bundle.name, []))
    overall_target_rate = df[target].mean() if pd.api.types.is_numeric_dtype(df[target]) else np.nan
    for col in core_categorical[:6]:
        values = df[col].fillna("__MISSING__").astype(str)
        top_counts = values.value_counts().head(10).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        top_counts.plot(kind="barh", ax=ax, color="#4E79A7")
        ax.set_title(f"{bundle.name}: {col} top categories")
        ax.set_xlabel("count")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{bundle.name}_{safe_name(col)}_category_frequency.png", dpi=160)
        plt.close(fig)

        target_rates = []
        for category in top_counts.index:
            mask = values.eq(category)
            target_rates.append((category, df.loc[mask, target].mean(), int(mask.sum())))
        target_rate_df = pd.DataFrame(target_rates, columns=["category", "target_rate", "count"]).sort_values(
            "target_rate"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(target_rate_df["category"], target_rate_df["target_rate"], color="#E15759")
        if pd.notna(overall_target_rate):
            ax.axvline(overall_target_rate, color="#444444", linewidth=1, linestyle="--")
        ax.set_title(f"{bundle.name}: {col} target rate")
        ax.set_xlabel("target rate")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{bundle.name}_{safe_name(col)}_target_rate.png", dpi=160)
        plt.close(fig)

    return messages
