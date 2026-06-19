from __future__ import annotations

import json
from pathlib import Path


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def write_eda_notebook(output: Path, dataset_name: str = "home_credit") -> Path:
    cells = [
        markdown_cell(
            f"""
# {dataset_name} EDA 与数据清洗检查

本 notebook 汇总第 2 阶段的核心证据：缺失率、异常值、违约/非违约差异、类别频数、清洗规则和固定数据划分。
先运行 `src/run_week1_2.py` 生成表格和图片，再从上到下执行本 notebook。
"""
        ),
        code_cell(
            """
from pathlib import Path
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"
DATA_PROCESSED = ROOT / "data" / "processed"
DATASET = "home_credit"

def read_table(name):
    return pd.read_csv(TABLES / f"{DATASET}_{name}.csv")
"""
        ),
        markdown_cell(
            """
## 数据集与类别不平衡

先确认样本量、字段数、重复行，以及 `TARGET` 的违约率。
"""
        ),
        code_cell(
            """
dataset_summary = read_table("dataset_summary")
target_distribution = read_table("target_distribution")
display(dataset_summary)
display(target_distribution)
"""
        ),
        markdown_cell(
            """
![target distribution](../reports/figures/home_credit_target_distribution.png)
"""
        ),
        markdown_cell(
            """
## 缺失值、异常值与偏态

这些表用于决定清洗规则和后续是否需要更稳健的特征变换。
"""
        ),
        code_cell(
            """
missing = read_table("missing_values")
outliers = read_table("outlier_invalid_report")
display(missing.head(20))
display(outliers.sort_values("iqr_outlier_rate", ascending=False).head(15))
"""
        ),
        markdown_cell(
            """
![missing top20](../reports/figures/home_credit_missing_top20.png)
"""
        ),
        markdown_cell(
            """
## 违约组与非违约组对比

数值变量对比看均值、分位数和缺失率差异；类别变量对比看各类别目标率和相对总体违约率的 lift。
"""
        ),
        code_cell(
            """
numeric_compare = read_table("target_numeric_comparison")
categorical_compare = read_table("target_categorical_comparison")
display(numeric_compare)
display(categorical_compare.sort_values("target_rate_lift", ascending=False).head(20))
"""
        ),
        markdown_cell(
            """
![numeric target mean diff](../reports/figures/home_credit_target_numeric_mean_diff_top10.png)
"""
        ),
        markdown_cell(
            """
## 类别频数与稀有类别

稀有类别清单用于后续 One-Hot、WOE/IV 或 embedding 特征工程，避免训练集中的极低频类别造成不稳定。
"""
        ),
        code_cell(
            """
categorical_summary = read_table("categorical_summary")
rare_categories = read_table("rare_categories")
display(categorical_summary.sort_values("unique_count", ascending=False))
display(rare_categories[rare_categories["is_rare"]].head(20))
"""
        ),
        markdown_cell(
            """
## 清洗规则与固定划分

processed 数据使用训练集统计量完成缺失填补，并导出了 train/valid/test 三个独立文件。
"""
        ),
        code_cell(
            """
cleaning_audit = read_table("cleaning_audit")
imputation_policy = read_table("missing_imputation_policy")
split_summary = read_table("split_summary")
display(cleaning_audit)
display(imputation_policy.head(20))
display(split_summary)
"""
        ),
        code_cell(
            """
processed_cols = pd.read_csv(DATA_PROCESSED / f"{DATASET}_processed.csv", nrows=0).columns.tolist()
split_counts = pd.read_csv(
    DATA_PROCESSED / f"{DATASET}_processed.csv",
    usecols=["split", "TARGET"],
)["split"].value_counts()
print(f"processed columns: {len(processed_cols)}")
display(split_counts)
"""
        ),
        markdown_cell(
            """
## 后续动作

- 将目标组对比图和缺失 Top20 图整理进 PPT 的数据理解部分。
- 第 3 阶段继续基于同一套 processed 数据做特征工程和模型对比。
- 若启用类别特征模型，沿用当前缺失和稀有类别策略，避免训练/验证/测试口径不一致。
"""
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
