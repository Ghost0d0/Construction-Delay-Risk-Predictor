"""
streamlit_app.py
----------------
Interactive dashboard for the Construction Delay Risk Predictor.

Launch:
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

# ── Path resolution ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_generator import generate_dataset
from train_model import (
    load_data, build_preprocessor, train_all_models,
    save_artifacts, predict_delay_risk, load_model,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from risk_analysis import (
    monte_carlo_duration, analyse_portfolio,
    risk_correlation_matrix, compare_scenarios, compute_risk_score,
    INDUSTRY_BENCHMARKS,
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Construction Delay Risk Predictor",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Sans', sans-serif; font-weight: 700; }

.risk-card {
    border-radius: 10px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.risk-low      { background: #052e16; border: 2px solid #16a34a; }
.risk-moderate { background: #431407; border: 2px solid #ea580c; }
.risk-high     { background: #450a0a; border: 2px solid #dc2626; }
.risk-critical { background: #1c0a0a; border: 2px solid #7f1d1d; }

.risk-pct {
    font-size: 3.5rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    line-height: 1;
}
.risk-label {
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.4rem;
    font-family: 'DM Mono', monospace;
}
.factor-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0;
    border-bottom: 1px solid #1e293b;
}
.factor-name { flex: 1; font-size: 0.85rem; color: #cbd5e1; }
.factor-bar  { height: 6px; border-radius: 3px; background: #ef4444; }
.kpi-box {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.kpi-val   { font-size: 1.7rem; font-weight: 700; color: #38bdf8;
             font-family: 'DM Mono', monospace; }
.kpi-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase;
             letter-spacing: 0.08em; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

CHART_BG   = "#0f172a"
AXIS_COLOR = "#334155"
TEXT_COLOR = "#cbd5e1"
ACCENT     = "#38bdf8"


# ─── Data & Model Loading ─────────────────────────────────────────────────────

@st.cache_resource
def load_or_train_model():
    """Load model if exists, otherwise train."""
    model_path = ROOT / "models" / "delay_model.pkl"
    data_path  = ROOT / "data"  / "construction_projects.csv"

    if not data_path.exists():
        generate_dataset(save_path=str(data_path))

    if not model_path.exists():
        from sklearn.model_selection import train_test_split
        os.makedirs(ROOT / "models", exist_ok=True)
        X, y = load_data(str(data_path))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        preprocessor = build_preprocessor()
        pipe, evaluation, feat_df = train_all_models(
            X_train, X_test, y_train, y_test, preprocessor
        )
        save_artifacts(pipe, evaluation, feat_df)
        return pipe, evaluation, feat_df
    else:
        pipe = load_model(str(model_path))
        eval_path = ROOT / "models" / "model_evaluation.json"
        feat_path = ROOT / "models" / "feature_importance.csv"
        evaluation = json.load(open(eval_path)) if eval_path.exists() else {}
        feat_df    = pd.read_csv(feat_path) if feat_path.exists() else pd.DataFrame()
        return pipe, evaluation, feat_df


@st.cache_data
def load_dataset():
    path = ROOT / "data" / "construction_projects.csv"
    return pd.read_csv(path)


# ─── Plot Helpers ─────────────────────────────────────────────────────────────

def dark_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.spines[:].set_color(AXIS_COLOR)
    return fig, ax


def plot_feature_importance(feat_df: pd.DataFrame) -> plt.Figure:
    top = feat_df.head(12)
    fig, ax = dark_fig(10, 4.5)
    bars = ax.barh(top["feature"][::-1], top["importance_pct"][::-1],
                   color=ACCENT, alpha=0.85, edgecolor=CHART_BG)
    ax.set_xlabel("Importance (%)", color=TEXT_COLOR, fontsize=9)
    ax.set_title("Top Feature Importances (Gradient Boosting)",
                 color=TEXT_COLOR, fontsize=11, fontweight="700")
    for bar in bars:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%", va="center",
                color=TEXT_COLOR, fontsize=7.5, fontfamily="monospace")
    fig.tight_layout()
    return fig


def plot_delay_by_type(df: pd.DataFrame) -> plt.Figure:
    grp = df.groupby("project_type")["delayed"].mean() * 100
    fig, ax = dark_fig(7, 3.5)
    colors = ["#ef4444" if v > 60 else "#f97316" if v > 45 else "#22c55e"
              for v in grp.values]
    bars = ax.bar(grp.index, grp.values, color=colors, alpha=0.85, edgecolor=CHART_BG)
    ax.axhline(INDUSTRY_BENCHMARKS["global_delay_rate_pct"], color="#94a3b8",
               linestyle="--", lw=1.2, label=f"Global avg ({INDUSTRY_BENCHMARKS['global_delay_rate_pct']}%)")
    ax.set_ylabel("Delay Rate (%)", color=TEXT_COLOR, fontsize=9)
    ax.set_title("Delay Rate by Project Type", color=TEXT_COLOR,
                 fontsize=11, fontweight="700")
    ax.legend(facecolor="#1e293b", edgecolor=AXIS_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.0f}%", ha="center", va="bottom",
                color=TEXT_COLOR, fontsize=8, fontfamily="monospace")
    fig.tight_layout()
    return fig


def plot_risk_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = dark_fig(8, 3.5)
    delayed     = df[df["delayed"] == 1]["delay_probability"] if "delay_probability" in df else pd.Series()
    not_delayed = df[df["delayed"] == 0]["delay_probability"] if "delay_probability" in df else pd.Series()
    ax.hist(not_delayed, bins=40, alpha=0.7, color="#22c55e", label="Not Delayed", density=True)
    ax.hist(delayed,     bins=40, alpha=0.7, color="#ef4444", label="Delayed",     density=True)
    ax.set_xlabel("Delay Probability", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Density", color=TEXT_COLOR, fontsize=9)
    ax.set_title("Delay Probability Distribution", color=TEXT_COLOR,
                 fontsize=11, fontweight="700")
    ax.legend(facecolor="#1e293b", edgecolor=AXIS_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return fig


def plot_monte_carlo(mc: dict) -> plt.Figure:
    planned = mc["planned_days"]
    hist_vals, bin_edges = mc["histogram_data"]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = dark_fig(9, 3.8)
    colors = ["#22c55e" if b <= planned else
              "#f97316" if b <= planned * 1.20 else "#ef4444"
              for b in bin_centers]
    ax.bar(bin_centers, hist_vals, width=bin_edges[1] - bin_edges[0],
           color=colors, alpha=0.80, edgecolor=CHART_BG)
    ax.axvline(planned,       color="#38bdf8", lw=1.8, linestyle="--", label=f"Planned ({planned}d)")
    ax.axvline(mc["p50_days"], color="#facc15", lw=1.5, linestyle=":",  label=f"P50 ({mc['p50_days']}d)")
    ax.axvline(mc["p80_days"], color="#f97316", lw=1.5, linestyle=":",  label=f"P80 ({mc['p80_days']}d)")
    ax.axvline(mc["p90_days"], color="#ef4444", lw=1.5, linestyle=":",  label=f"P90 ({mc['p90_days']}d)")
    ax.set_xlabel("Completion Duration (days)", color=TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Simulations", color=TEXT_COLOR, fontsize=9)
    ax.set_title(f"Monte Carlo Duration Simulation ({mc['simulations']:,} runs)",
                 color=TEXT_COLOR, fontsize=11, fontweight="700")
    ax.legend(facecolor="#1e293b", edgecolor=AXIS_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return fig


def plot_region_heatmap(df: pd.DataFrame) -> plt.Figure:
    if "region" not in df.columns or "delayed" not in df.columns:
        return None
    pivot = df.groupby(["region", "project_type"])["delayed"].mean().unstack() * 100
    fig, ax = dark_fig(8, 3.5)
    import matplotlib.colors as mcolors
    cmap = plt.cm.get_cmap("RdYlGn_r")
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, color=TEXT_COLOR, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, color=TEXT_COLOR, fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="600")
    plt.colorbar(im, ax=ax, label="Delay Rate (%)")
    ax.set_title("Delay Rate Heatmap: Region × Project Type",
                 color=TEXT_COLOR, fontsize=11, fontweight="700")
    fig.tight_layout()
    return fig


# ─── Sidebar: Project Input ───────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.5rem 0 1rem 0;">
            <div style="font-size:1.8rem">⚠️</div>
            <div style="font-family:'DM Mono',monospace; font-size:0.68rem;
                        color:#64748b; text-transform:uppercase; letter-spacing:0.1em;">
                Delay Risk Predictor
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📋 Project Details")
        project_type  = st.selectbox("Project Type",
            ["Commercial", "Residential", "Infrastructure", "Industrial"])
        contract_type = st.selectbox("Contract Type",
            ["Fixed Price", "Cost Plus", "Design-Build"])
        region        = st.selectbox("Region",
            ["North", "South", "East", "West", "Central"])

        st.markdown("### 📐 Project Scale")
        project_size = st.number_input("Project Size (m²)", 100, 100000, 5000, step=500)
        num_workers  = st.slider("Peak Workers on Site", 5, 300, 40, step=5)
        num_subs     = st.slider("Number of Subcontractors", 1, 17, 8)
        planned_days = st.number_input("Planned Duration (days)", 30, 900, 180, step=10)
        budget       = st.number_input("Budget (USD)", 100000, 100000000, 5000000, step=100000)
        buffer       = st.slider("Schedule Buffer (days)", 0, 30, 10)

        st.markdown("### 🎯 Risk Factors")
        weather_risk   = st.slider("Weather Risk Score",      1, 10, 5)
        material_risk  = st.slider("Material Delivery Risk",  1, 10, 5)
        labour_avail   = st.slider("Labour Availability",     1, 10, 6)
        design_cmplx   = st.slider("Design Complexity",       1, 10, 5)
        site_access    = st.slider("Site Accessibility",      1, 10, 6)
        prev_delays    = st.slider("Previous Delays (count)", 0,  4, 1)

        predict_btn = st.button("🔍 Assess Delay Risk", type="primary",
                                use_container_width=True)

    project = {
        "project_size_m2":        project_size,
        "num_workers":            num_workers,
        "num_subcontractors":     num_subs,
        "planned_duration_days":  planned_days,
        "budget_usd":             budget,
        "schedule_buffer_days":   buffer,
        "weather_risk_score":     weather_risk,
        "material_delivery_risk": material_risk,
        "labour_availability":    labour_avail,
        "design_complexity":      design_cmplx,
        "site_accessibility":     site_access,
        "previous_delays":        prev_delays,
        "contract_type":          contract_type,
        "project_type":           project_type,
        "region":                 region,
    }
    return project, predict_btn


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div style="border-bottom:1px solid #1e3a5f; padding-bottom:1rem; margin-bottom:1.5rem;">
        <h1 style="margin:0; font-size:1.9rem; color:#f1f5f9;">
            ⚠️ Construction Delay Risk Predictor
        </h1>
        <p style="margin:0.3rem 0 0 0; color:#64748b; font-size:0.85rem;
                  font-family:'DM Mono',monospace;">
            Machine Learning · Risk Modelling · Project Management Analytics · MSc Portfolio
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading model …"):
        model, evaluation, feat_df = load_or_train_model()
        df = load_dataset()

    project, predict_btn = render_sidebar()

    # Run prediction on button or first load
    if predict_btn or "prediction" not in st.session_state:
        result = predict_delay_risk(model, project)
        st.session_state.prediction = result
        st.session_state.project    = project

    result  = st.session_state.prediction
    proj    = st.session_state.project

    # ── Risk Result Card ─────────────────────────────────────────────────────
    risk_class = {
        "Low": "risk-low", "Moderate": "risk-moderate",
        "High": "risk-high", "Critical": "risk-critical"
    }.get(result["risk_level"], "risk-high")

    risk_color = {
        "Low": "#22c55e", "Moderate": "#f97316",
        "High": "#ef4444", "Critical": "#7f1d1d"
    }.get(result["risk_level"], "#ef4444")

    col_risk, col_factors = st.columns([1, 2])

    with col_risk:
        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <div style="color:#94a3b8; font-size:0.75rem; font-family:'DM Mono',monospace;
                        text-transform:uppercase; letter-spacing:0.1em;">
                Delay Probability
            </div>
            <div class="risk-pct" style="color:{risk_color};">
                {result['delay_probability_pct']}%
            </div>
            <div class="risk-label" style="color:{risk_color};">
                {result['risk_level']} Risk
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Monte Carlo
        mc = monte_carlo_duration(proj["planned_duration_days"],
                                   result["delay_probability"])
        st.markdown("#### Duration Forecast")
        c1, c2 = st.columns(2)
        c1.metric("P50 Completion", f"{mc['p50_days']} days",
                  delta=f"+{mc['p50_days'] - proj['planned_duration_days']} vs plan")
        c2.metric("P80 Completion", f"{mc['p80_days']} days",
                  delta=f"+{mc['p80_days'] - proj['planned_duration_days']} vs plan")
        st.metric("On-Time Probability", f"{mc['prob_on_time']*100:.0f}%")

    with col_factors:
        st.markdown("#### 🔴 Top Risk Factors")
        if result["top_risk_factors"]:
            for rf in result["top_risk_factors"]:
                bar_w = int(rf["impact"] * 280)
                st.markdown(f"""
                <div class="factor-row">
                    <div class="factor-name">• {rf['factor']}</div>
                    <div style="width:{bar_w}px; height:6px; border-radius:3px;
                                background:{risk_color}; opacity:0.85;"></div>
                    <div style="font-size:0.75rem; color:#94a3b8;
                                font-family:'DM Mono',monospace; min-width:32px;">
                        {int(rf['impact']*100)}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No major risk factors identified.")

        st.markdown("#### Composite Risk Score")
        score = compute_risk_score(proj)
        st.progress(int(score), text=f"{score}/100")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio", "🔬 Model", "🎲 Simulation", "🗺️ Heatmap", "📈 Scenarios"
    ])

    with tab1:
        st.subheader("Portfolio Delay Analysis")
        stats = analyse_portfolio(df)

        c1, c2, c3, c4 = st.columns(4)
        kpis = [
            (c1, "TOTAL PROJECTS",  f"{stats['total_projects']:,}"),
            (c2, "DELAY RATE",      f"{stats['delay_rate_pct']}%"),
            (c3, "AVG DURATION",    f"{stats['avg_planned_duration_days']:.0f}d"),
            (c4, "GLOBAL BENCHMARK",f"{INDUSTRY_BENCHMARKS['global_delay_rate_pct']}%"),
        ]
        for col, label, val in kpis:
            with col:
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-val">{val}</div>
                    <div class="kpi-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_delay_by_type(df), use_container_width=True)
        with col2:
            st.pyplot(plot_risk_distribution(df), use_container_width=True)

    with tab2:
        st.subheader("Model Performance")
        best = evaluation.get("best_model", "Gradient Boosting")
        metrics = evaluation.get("metrics", {}).get(best, {})

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, key in [
            (c1, "ROC-AUC",   "roc_auc"),
            (c2, "F1 Score",  "f1"),
            (c3, "Precision", "precision"),
            (c4, "Recall",    "recall"),
            (c5, "Accuracy",  "accuracy"),
        ]:
            with col:
                val = metrics.get(key, "—")
                st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-val">{val}</div>
                    <div class="kpi-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"<br>**Best model:** `{best}` &nbsp;|&nbsp; "
                    f"CV AUC: `{metrics.get('cv_roc_auc_mean','—')} "
                    f"± {metrics.get('cv_roc_auc_std','—')}`",
                    unsafe_allow_html=True)

        if not feat_df.empty:
            st.pyplot(plot_feature_importance(feat_df), use_container_width=True)

        # All models comparison table
        if evaluation.get("metrics"):
            rows = []
            for name, m in evaluation["metrics"].items():
                rows.append({"Model": name, "AUC": m.get("roc_auc"),
                             "F1": m.get("f1"), "Precision": m.get("precision"),
                             "Recall": m.get("recall"), "CV AUC": m.get("cv_roc_auc_mean")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Monte Carlo Duration Simulation")
        mc = monte_carlo_duration(
            proj["planned_duration_days"],
            result["delay_probability"]
        )
        st.pyplot(plot_monte_carlo(mc), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P50 (Median)",  f"{mc['p50_days']} days")
        c2.metric("P80",           f"{mc['p80_days']} days")
        c3.metric("P90",           f"{mc['p90_days']} days")
        c4.metric("On-Time Prob",  f"{mc['prob_on_time']*100:.0f}%")

        st.info(f"Based on {mc['simulations']:,} simulations · "
                f"10% overrun probability: {mc['prob_10pct_overrun']*100:.0f}% · "
                f"25% overrun probability: {mc['prob_25pct_overrun']*100:.0f}%")

    with tab4:
        st.subheader("Regional Risk Heatmap")
        fig = plot_region_heatmap(df)
        if fig:
            st.pyplot(fig, use_container_width=True)

        corr = risk_correlation_matrix(df)
        st.markdown("#### Risk Factor Correlations with Delay")
        corr_df = corr.reset_index()
        corr_df.columns = ["Feature", "Correlation with Delay"]
        st.dataframe(
            corr_df.style.background_gradient(
                subset=["Correlation with Delay"], cmap="RdYlGn_r", vmin=-1, vmax=1
            ).format({"Correlation with Delay": "{:.3f}"}),
            use_container_width=True, hide_index=True
        )

    with tab5:
        st.subheader("What-If Scenario Analysis")
        st.markdown("Compare how mitigation actions reduce risk for the current project.")

        base = {k: v for k, v in proj.items()}
        mods = [
            {"label": "Improve Labour (+3)",      "labour_availability": min(10, proj["labour_availability"] + 3)},
            {"label": "Add 15 Buffer Days",        "schedule_buffer_days": proj["schedule_buffer_days"] + 15},
            {"label": "Reduce Material Risk (−3)", "material_delivery_risk": max(1, proj["material_delivery_risk"] - 3)},
            {"label": "Switch to Cost Plus",       "contract_type": "Cost Plus"},
            {"label": "All Mitigations",
             "labour_availability":    min(10, proj["labour_availability"] + 3),
             "schedule_buffer_days":   proj["schedule_buffer_days"] + 15,
             "material_delivery_risk": max(1, proj["material_delivery_risk"] - 3),
             "contract_type":          "Cost Plus"},
        ]
        scenario_df = compare_scenarios(base, mods)

        fig, ax = dark_fig(9, 3.5)
        colors = ["#ef4444" if s > 60 else "#f97316" if s > 40 else "#22c55e"
                  for s in scenario_df["risk_score"]]
        bars = ax.bar(scenario_df["scenario"], scenario_df["risk_score"],
                      color=colors, alpha=0.85, edgecolor=CHART_BG)
        ax.set_ylabel("Risk Score (0–100)", color=TEXT_COLOR, fontsize=9)
        ax.set_title("Risk Score by Mitigation Scenario", color=TEXT_COLOR,
                     fontsize=11, fontweight="700")
        ax.tick_params(colors=TEXT_COLOR, axis="both")
        plt.xticks(rotation=25, ha="right", fontsize=8)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.0f}", ha="center", va="bottom",
                    color=TEXT_COLOR, fontsize=8, fontfamily="monospace")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("""
    <div style="margin-top:3rem; border-top:1px solid #1e3a5f; padding-top:1rem;
                font-size:0.72rem; color:#475569; font-family:'DM Mono',monospace;">
        Construction Delay Risk Predictor · Gradient Boosting Classifier ·
        Monte Carlo Simulation · MSc Research Portfolio
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
