from __future__ import annotations

import sys
from pathlib import Path
import os

import pandas as pd

from data_preprocess import (
    load_available_dataset,
    save_processed_dataset,
    write_cleaning_audit,
    write_data_cleaning_notes,
    write_field_dictionary,
    write_home_credit_leakage_risk_fields,
)
from eda import write_eda_figures, write_eda_tables
from notebook_artifacts import write_eda_notebook
from project_paths import FIGURES, NOTEBOOKS, REPORTS, TABLES, ensure_project_dirs, rel
from train_baseline import dependency_status, train_baselines


def update_status_report(lines: list[str]) -> None:
    status_path = REPORTS / "week1_2_status.md"
    text = "\n".join(lines).rstrip() + "\n"
    status_path.write_text(text, encoding="utf-8")


def main() -> int:
    ensure_project_dirs()
    status_lines = [
        "# 第 1-2 周任务状态报告",
        "",
        "## 自动运行结果",
        "",
    ]

    try:
        bundle = load_available_dataset()
    except Exception as exc:  # noqa: BLE001
        status_lines.extend(
            [
                "- 状态：未完成真实数据流程。",
                f"- 原因：{exc}",
                "",
                "## 下一步",
                "",
                "1. 将 `application_train.csv` 放入 `data/raw/home_credit/`，或运行 `python src/download_data.py --uci-only` 获取 UCI 数据。",
                "2. 安装 `requirements.txt` 中的依赖。",
                "3. 重新运行 `python src/run_week1_2.py`。",
            ]
        )
        update_status_report(status_lines)
        print(exc)
        return 1

    status_lines.extend(
        [
            f"- 使用数据集：`{bundle.name}`。",
            f"- 原始文件：`{rel(bundle.source_path)}`。",
            f"- 样本数：{len(bundle.frame)}。",
            f"- 字段数：{len(bundle.frame.columns)}。",
            f"- 目标变量：`{bundle.target}`。",
        ]
    )
    home_credit_path = Path("data/raw/home_credit/application_train.csv")
    if home_credit_path.exists():
        status_lines.append("- Home Credit 主实验数据：已存在。")
    else:
        status_lines.append("- Home Credit 主实验数据：暂未获取；Kaggle 下载需要完成账号认证。")
    if bundle.name == "taiwan_credit":
        status_lines.append("- 当前结果基于 UCI Taiwan 数据集，适合作为第 1-2 周流程联调和扩展验证。")

    field_dict = TABLES / f"{bundle.name}_field_dictionary.csv"
    write_field_dictionary(bundle, field_dict)
    status_lines.append(f"- 字段说明表：`{rel(field_dict)}`。")
    leakage_risk_path = None
    if bundle.name == "home_credit":
        leakage_risk_path = TABLES / "home_credit_leakage_risk_fields.csv"
        write_home_credit_leakage_risk_fields(bundle, leakage_risk_path)
        status_lines.append(f"- 数据泄漏风险字段清单：`{rel(leakage_risk_path)}`。")

    write_eda_tables(bundle, TABLES)
    figure_messages = write_eda_figures(bundle, FIGURES)
    status_lines.append(f"- EDA 表格已输出到：`{rel(TABLES)}`。")
    status_lines.append(f"- 异常值与偏态报告：`{rel(TABLES / f'{bundle.name}_outlier_invalid_report.csv')}`。")
    status_lines.append(f"- 违约/非违约数值对比：`{rel(TABLES / f'{bundle.name}_target_numeric_comparison.csv')}`。")
    status_lines.append(f"- 类别目标率对比：`{rel(TABLES / f'{bundle.name}_target_categorical_comparison.csv')}`。")
    status_lines.append(f"- 稀有类别清单：`{rel(TABLES / f'{bundle.name}_rare_categories.csv')}`。")
    if figure_messages:
        status_lines.extend([f"- {message}" for message in figure_messages])
    else:
        status_lines.append(f"- EDA 图表已输出到：`{rel(FIGURES)}`。")

    cleaning_audit_path = TABLES / f"{bundle.name}_cleaning_audit.csv"
    write_cleaning_audit(bundle, cleaning_audit_path)
    status_lines.append(f"- 清洗动作审计：`{rel(cleaning_audit_path)}`。")

    processed_path = save_processed_dataset(bundle, TABLES)
    processed = pd.read_csv(processed_path)
    split_summary = (
        processed.groupby(["split", bundle.target])
        .size()
        .rename("count")
        .reset_index()
    )
    split_summary_path = TABLES / f"{bundle.name}_split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False)
    notebook_path = NOTEBOOKS / f"{bundle.name}_eda_cleaning.ipynb"
    write_eda_notebook(notebook_path, bundle.name)
    cleaning_notes_path = REPORTS / f"{bundle.name}_data_cleaning_notes.md"
    write_data_cleaning_notes(
        bundle,
        cleaning_notes_path,
        processed_path,
        split_summary_path,
        TABLES,
        FIGURES,
        notebook_path,
    )
    status_lines.append(f"- 清洗后数据：`{rel(processed_path)}`。")
    status_lines.append(f"- 清洗后缺失单元数：{int(processed.isna().sum().sum())}。")
    status_lines.append(f"- 训练/验证/测试划分摘要：`{rel(split_summary_path)}`。")
    status_lines.append(f"- 独立训练集文件：`{rel(processed_path.parent / f'{bundle.name}_train.csv')}`。")
    status_lines.append(f"- 独立验证集文件：`{rel(processed_path.parent / f'{bundle.name}_valid.csv')}`。")
    status_lines.append(f"- 独立测试集文件：`{rel(processed_path.parent / f'{bundle.name}_test.csv')}`。")
    status_lines.append(f"- 数据清洗说明：`{rel(cleaning_notes_path)}`。")
    status_lines.append(f"- EDA notebook：`{rel(notebook_path)}`。")

    deps = dependency_status()
    status_lines.append("")
    status_lines.append("## 依赖状态")
    status_lines.append("")
    for package, installed in deps.items():
        status_lines.append(f"- {package}: {'available' if installed else 'missing'}")
    train_device = os.environ.get("TRAIN_DEVICE", "auto").strip().lower() or "auto"
    status_lines.append(f"- training_device: {train_device}")

    baseline_results = train_baselines(processed_path, bundle.target, bundle.id_columns, TABLES, FIGURES)
    model_run_log_path = TABLES / "model_run_log.csv"
    model_run_log = pd.read_csv(model_run_log_path) if model_run_log_path.exists() else pd.DataFrame()
    status_lines.append("")
    status_lines.append("## 基线模型")
    status_lines.append("")
    status_lines.append(f"- 基线结果表：`{rel(baseline_results)}`。")
    if not model_run_log.empty:
        status_lines.append(f"- 模型运行日志：`{rel(model_run_log_path)}`。")
        for _, row in model_run_log.iterrows():
            model = row.get("model", "")
            requested = row.get("requested_device", "")
            actual = row.get("actual_device", "")
            status = row.get("status", "")
            status_lines.append(f"- `{model}`：requested `{requested}`，actual `{actual}`，status `{status}`。")
            fallback_reason = row.get("fallback_reason", "")
            if isinstance(fallback_reason, str) and fallback_reason.strip():
                short_reason = fallback_reason.strip().splitlines()[0]
                status_lines.append(f"  - 回退原因：{short_reason}")
    results = pd.read_csv(baseline_results)
    if "model" in results.columns and "split" in results.columns:
        test_results = results[results["split"] == "test"]
        if not test_results.empty:
            for metric in ["roc_auc", "pr_auc", "f1"]:
                best = test_results.sort_values(metric, ascending=False).iloc[0]
                status_lines.append(f"- Test {metric} 最优：`{best['model']}` = {best[metric]:.4f}。")
    if not deps["sklearn"]:
        status_lines.append("- 基线训练被跳过：缺少 scikit-learn。")
    elif train_device == "gpu":
        status_lines.append("- GPU-only 模式已启用：CPU-only 的 Logistic Regression 与 RandomForest 已跳过。")
        has_lgbm_fallback = (
            not model_run_log.empty
            and model_run_log["model"].astype(str).str.contains("lightgbm_cpu_fallback").any()
        )
        if has_lgbm_fallback:
            status_lines.append("- LightGBM 已完成 CPU fallback 训练；GPU 尝试失败原因见模型运行日志。")
        else:
            status_lines.append("- LightGBM 已完成 GPU 训练。")
    elif not deps["lightgbm"]:
        status_lines.append("- LightGBM 初版未运行：缺少 lightgbm；已运行可用降级基线。")
    else:
        status_lines.append("- LightGBM 初版已完成。")

    status_lines.append("")
    status_lines.append("## 第 1-2 周检查清单")
    status_lines.append("")
    status_lines.append("- [x] 数据集基本信息表。")
    status_lines.append("- [x] 字段说明表。")
    if bundle.name == "home_credit" and leakage_risk_path is not None:
        status_lines.append("- [x] 数据泄漏风险字段清单。")
    status_lines.append("- [x] 目标变量分布表。")
    status_lines.append("- [x] 缺失值统计表。")
    status_lines.append("- [x] 异常值、无效值与偏态报告。")
    status_lines.append("- [x] 违约组与非违约组差异分析。")
    status_lines.append("- [x] 类别变量频数、目标率和稀有类别分析。")
    status_lines.append("- [x] 数据清洗说明。")
    status_lines.append("- [x] EDA notebook。")
    status_lines.append("- [x] 清洗后数据与固定划分。")
    status_lines.append("- [x] 清洗后训练集、验证集和测试集独立文件。")
    if train_device == "gpu":
        has_xgb_cuda = "model" in results.columns and results["model"].astype(str).eq("xgboost_cuda_numeric").any()
        has_lightgbm = "model" in results.columns and results["model"].astype(str).str.startswith("lightgbm_").any()
        status_lines.append(f"- [{'x' if has_xgb_cuda else ' '}] XGBoost CUDA 初版结果。")
        status_lines.append(f"- [{'x' if has_lightgbm else ' '}] LightGBM 初版结果。")
    else:
        status_lines.append(f"- [{'x' if deps['lightgbm'] else ' '}] LightGBM 初版结果。")
    status_lines.append("")
    status_lines.append("## 后续建议")
    status_lines.append("")
    if bundle.name == "home_credit":
        status_lines.append("1. 继续基于 Home Credit 做第 3-4 周：不平衡策略、概率校准、业务阈值优化和 SHAP 分析。")
    else:
        status_lines.append("1. 优先补齐 Kaggle Home Credit 主实验数据。")
    if train_device == "gpu":
        status_lines.append("2. 若 LightGBM 使用了 CPU fallback，后续可继续排查 LightGBM OpenCL/Boost Compute 环境以争取 GPU 版。")
    else:
        status_lines.append("2. 安装完整建模依赖后重跑本脚本，生成 LightGBM/XGBoost 结果。")
    status_lines.append("3. 将 ROC/PR 曲线、混淆矩阵和基线结果表整理进后续 PPT。")

    update_status_report(status_lines)
    print(f"Done. Status report: {rel(REPORTS / 'week1_2_status.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
