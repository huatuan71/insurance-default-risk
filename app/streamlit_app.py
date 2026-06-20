from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports" / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference import HomeCreditInferenceService, InferenceInputError, InferenceSetupError, REVIEW_MARGIN


SCENARIO_LABELS = {
    "fn5_fp1": "FN:FP = 5:1（平衡风险与误拒）",
    "fn10_fp1": "FN:FP = 10:1（默认课堂场景）",
    "fn20_fp1": "FN:FP = 20:1（高漏判成本场景）",
}


st.set_page_config(page_title="贷款违约风险决策台", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="正在加载本地模型与阈值资产...")
def load_service() -> HomeCreditInferenceService:
    return HomeCreditInferenceService(ROOT)


@st.cache_data(show_spinner=False)
def load_csv(uploaded_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO

    return pd.read_csv(BytesIO(uploaded_bytes))


def scenario_display(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario)


def render_decision(service: HomeCreditInferenceService, result: pd.Series, scenario: str) -> None:
    recommendation = service.recommendation(scenario)
    model_name = str(result["model"])
    model_label = "LightGBM（SMOTE）" if "smote" in model_name else "LightGBM（raw）"
    col_score, col_threshold, col_model, col_cost = st.columns(4)
    col_score.metric("违约风险分数", f"{float(result['risk_probability_percent']):.2f}%")
    col_threshold.metric("业务阈值", f"{float(result['decision_threshold']):.2f}")
    col_model.metric("决策模型", model_label)
    col_cost.metric("测试集总成本", f"{int(recommendation['test_total_cost']):,}")

    if result["decision"] == "高风险建议拦截":
        st.error(f"决策建议：{result['decision']}。当前场景下，风险分数达到或超过阈值。")
    else:
        st.success(f"决策建议：{result['decision']}。当前场景下，风险分数低于阈值。")
    if bool(result["near_decision_boundary"]):
        st.warning(f"{result['review_note']}（距阈值不超过 {REVIEW_MARGIN:.2f}）。")

    with st.expander("业务口径与实验依据", expanded=False):
        st.write(
            f"成本假设为 {recommendation['cost_ratio']}；推荐阈值只在验证集上选择。"
            f"测试集 Recall 为 {float(recommendation['test_recall']):.1%}，"
            f"相对 0.5 阈值节省成本 {int(recommendation['test_cost_saving_vs_0_5']):,}。"
        )
        st.caption("分数为原始 GBDT 输出。概率校准实验未优于 raw score，因此未额外变换。")


def render_explanation_plot(explanation: pd.DataFrame) -> None:
    if explanation.empty:
        st.info("该案例没有可展示的局部特征贡献。")
        return
    chart_data = explanation.sort_values("shap_value").copy()
    colors = ["#d1495b" if value > 0 else "#007c91" for value in chart_data["shap_value"]]
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(chart_data) * 0.42)))
    ax.barh(chart_data["encoded_feature"], chart_data["shap_value"], color=colors)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to risk score")
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, width="stretch")
    st.dataframe(
        explanation[["direction", "encoded_feature", "shap_value", "business_meaning"]],
        hide_index=True,
        width="stretch",
    )


def render_precomputed_case(service: HomeCreditInferenceService, case: pd.Series, scenario: str) -> bool:
    recommendation = service.recommendation(scenario)
    if str(case["model"]) != str(recommendation["model"]):
        return False
    waterfall = ROOT / str(case["waterfall_path"])
    if waterfall.exists():
        st.image(str(waterfall), caption="预生成的 Stage 7 SHAP waterfall 图", width="stretch")
    st.markdown("**推高风险的主要因素**")
    st.write(str(case["top_positive_features"]))
    st.markdown("**降低风险的主要因素**")
    st.write(str(case["top_negative_features"]))
    st.caption("SHAP 解释模型分数方向，不等同于因果关系。")
    return True


def render_dynamic_explanation(service: HomeCreditInferenceService, prepared: pd.DataFrame, scenario: str, key: str) -> None:
    if st.button("生成当前案例的 SHAP 解释", key=f"explain_{key}"):
        with st.spinner("正在计算单样本 SHAP 贡献..."):
            st.session_state[f"explanation_{key}"] = service.explain_prepared(prepared.iloc[[0]], scenario)
    explanation = st.session_state.get(f"explanation_{key}")
    if explanation is not None:
        render_explanation_plot(explanation)


def render_risk_decision(service: HomeCreditInferenceService, scenario: str) -> None:
    st.subheader("单案例风险决策")
    st.caption("选择已验证的测试案例，或上传 Home Credit 原始格式 CSV 后选择其中一条记录。")
    source = st.radio("案例来源", ["内置演示案例", "上传 CSV"], horizontal=True)

    case_metadata: pd.Series | None = None
    if source == "内置演示案例":
        cases = service.demo_cases()
        labels = {
            row["case_name"]: f"{row['case_name']} | {row['case_type']} | ID {int(row['SK_ID_CURR'])}"
            for _, row in cases.iterrows()
        }
        default_name = "main_fn10_tp_high_risk"
        options = cases["case_name"].tolist()
        selected_name = st.selectbox(
            "选择案例",
            options,
            index=options.index(default_name) if default_name in options else 0,
            format_func=lambda name: labels[name],
        )
        case_metadata = cases.loc[cases["case_name"].eq(selected_name)].iloc[0]
        sample = service.load_demo_case_features(selected_name)
        scored = service.score_engineered(sample, scenario)
        result = scored.results.iloc[0]
        prepared = scored.prepared_features.iloc[[0]]
        render_decision(service, result, scenario)
        with st.expander("仅供演示的真实标签", expanded=False):
            actual = "违约" if int(case_metadata["y_true"]) == 1 else "未违约"
            st.write(f"该测试集案例的真实标签：{actual}。线上推理时不会拥有此字段。")
    else:
        upload = st.file_uploader("上传 application_train/application_test 格式 CSV", type="csv", key="decision_upload")
        if upload is None:
            template = pd.DataFrame(columns=service.raw_template_columns)
            st.download_button(
                "下载原始输入字段模板",
                template.to_csv(index=False).encode("utf-8-sig"),
                "home_credit_input_template.csv",
                "text/csv",
            )
            return
        try:
            raw = load_csv(upload.getvalue())
            scored = service.score_raw(raw, scenario)
        except (UnicodeDecodeError, pd.errors.ParserError, InferenceInputError) as exc:
            st.error(f"无法评分：{exc}")
            return
        position = st.selectbox(
            "选择要解释的记录",
            list(range(len(scored.results))),
            format_func=lambda index: f"第 {index + 1} 行 | ID {int(scored.results.iloc[index]['SK_ID_CURR'])}",
        )
        result = scored.results.iloc[position]
        prepared = scored.prepared_features.iloc[[position]]
        render_decision(service, result, scenario)

    st.subheader("风险原因")
    if case_metadata is not None and render_precomputed_case(service, case_metadata, scenario):
        return
    render_dynamic_explanation(service, prepared, scenario, f"{scenario}_{int(result['SK_ID_CURR'])}")


def render_batch_scoring(service: HomeCreditInferenceService, scenario: str) -> None:
    st.subheader("批量 CSV 评分")
    st.caption("上传数据仅在当前会话内处理，不写入磁盘。`TARGET` 列存在时会被忽略。")
    template = pd.DataFrame(columns=service.raw_template_columns)
    st.download_button(
        "下载原始输入字段模板",
        template.to_csv(index=False).encode("utf-8-sig"),
        "home_credit_input_template.csv",
        "text/csv",
        key="batch_template",
    )
    upload = st.file_uploader("上传待评分 CSV", type="csv", key="batch_upload")
    if upload is None:
        return
    try:
        raw = load_csv(upload.getvalue())
        scored = service.score_raw(raw, scenario)
    except (UnicodeDecodeError, pd.errors.ParserError, InferenceInputError) as exc:
        st.error(f"无法评分：{exc}")
        return
    st.success(f"已完成 {len(scored.results):,} 条记录的评分。")
    st.dataframe(scored.results.head(100), hide_index=True, width="stretch")
    st.caption("页面最多预览 100 行；下载文件包含全部结果。")
    st.download_button(
        "下载评分结果 CSV",
        scored.results.to_csv(index=False).encode("utf-8-sig"),
        "home_credit_scored_results.csv",
        "text/csv",
    )


def image_if_available(title: str, filename: str, caption: str) -> None:
    path = ROOT / "reports" / "figures" / filename
    st.markdown(f"**{title}**")
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"未找到 {filename}。请先运行对应实验阶段。")


def render_experiment_overview(service: HomeCreditInferenceService) -> None:
    st.subheader("实验结论总览")
    recommendation = service.recommendation("fn10_fp1")
    calibration_path = ROOT / "reports" / "tables" / "stage6_probability_calibration_summary.csv"
    calibration = pd.read_csv(calibration_path) if calibration_path.exists() else pd.DataFrame()
    col1, col2, col3 = st.columns(3)
    col1.metric("默认成本场景", str(recommendation["cost_ratio"]))
    col2.metric("默认决策阈值", f"{float(recommendation['selected_threshold']):.2f}")
    col3.metric("默认测试集 Recall", f"{float(recommendation['test_recall']):.1%}")
    if not calibration.empty:
        best = calibration.loc[calibration["best_brier_score"].idxmin()]
        st.caption(
            f"概率校准结论：{best['model']} 的 raw score Brier score 最优（{float(best['best_brier_score']):.4f}），"
            "因此应用展示未校准的 GBDT 分数。"
        )

    left, right = st.columns(2)
    with left:
        image_if_available("Stage 4：基线 PR 曲线", "stage4_test_pr_curves.png", "LightGBM/XGBoost/Logistic Regression 的少数类识别能力。")
        image_if_available("Stage 5：召回与误报权衡", "stage5_recall_false_positive_tradeoff.png", "不平衡策略提高召回时的误报代价。")
        image_if_available("Stage 7：全局 SHAP", "stage7_stage5_lightgbm_smote_gpu_shap_summary.png", "10:1 推荐 LightGBM 的全局风险信号。")
    with right:
        image_if_available("Stage 6：10:1 成本曲线", "stage6_cost_curve_fn10_fp1.png", "阈值不固定为 0.5，而是以业务成本最小化为目标。")
        image_if_available("Stage 6：概率可靠性", "stage6_probability_calibration_reliability_stage5_lightgbm_smote_gpu.png", "raw、sigmoid 与 isotonic 的预测概率一致性比较。")
        image_if_available("Stage 8：扩展模型对比", "stage8_comparison_pr_auc.png", "MLP+Embedding 作为扩展对照，GBDT 仍是主线。")


def render_limitations() -> None:
    st.subheader("使用边界")
    st.warning("这是基于公开竞赛数据的课程演示系统，不能用于真实放贷、自动拒绝或实际个人信用决策。")
    st.markdown(
        "- 仅接受与 Home Credit `application_train/application_test` 一致的输入字段。\n"
        "- `TARGET` 只用于离线实验评估，不会进入模型，也不会影响上传记录的预测。\n"
        "- 人工复核提示仅标记接近业务阈值的记录，不代表已完成双阈值或人工审核成本优化。\n"
        "- SHAP 描述模型分数的局部贡献，不代表变量与违约之间的因果关系。\n"
        "- 模型、原始数据与处理后 CSV 为本地可再生产物，Git 仓库不包含这些大文件。"
    )


def main() -> None:
    st.title("个人贷款违约风险决策台")
    st.caption("Home Credit | LightGBM | 不平衡策略 | 业务成本阈值 | SHAP 解释")
    try:
        service = load_service()
    except InferenceSetupError as exc:
        st.error(str(exc))
        st.code(
            ".\\.venv\\Scripts\\python.exe .\\src\\run_stage3_features.py\n"
            ".\\.venv\\Scripts\\python.exe .\\src\\run_stage5_imbalance.py\n"
            ".\\.venv\\Scripts\\python.exe .\\src\\run_stage6_business_thresholds.py\n"
            ".\\.venv\\Scripts\\python.exe .\\src\\run_stage7_explainability.py",
            language="powershell",
        )
        return

    with st.sidebar:
        st.header("决策场景")
        scenario = st.radio(
            "漏判违约成本 : 误拒正常客户成本",
            service.scenarios,
            index=service.scenarios.index("fn10_fp1") if "fn10_fp1" in service.scenarios else 0,
            format_func=scenario_display,
        )
        recommendation = service.recommendation(scenario)
        st.caption(
            f"模型：{recommendation['model']}\n\n"
            f"阈值：{float(recommendation['selected_threshold']):.2f}\n\n"
            "阈值由验证集成本最小化确定。"
        )

    tab_decision, tab_batch, tab_overview, tab_limits = st.tabs(["风险决策", "批量评分", "实验总览", "使用边界"])
    with tab_decision:
        render_risk_decision(service, scenario)
    with tab_batch:
        render_batch_scoring(service, scenario)
    with tab_overview:
        render_experiment_overview(service)
    with tab_limits:
        render_limitations()


if __name__ == "__main__":
    main()
