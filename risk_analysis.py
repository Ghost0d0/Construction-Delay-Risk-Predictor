"""
risk_analysis.py
----------------
Advanced risk analysis, scenario simulation, and reporting utilities
for the Construction Delay Risk Predictor.

Provides:
  - Portfolio-level risk analysis across multiple projects
  - Monte Carlo delay simulation
  - Risk factor correlation analysis
  - Benchmarking against industry averages
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DATA_PATH = "data/construction_projects.csv"

# ─── Industry Benchmarks ──────────────────────────────────────────────────────
INDUSTRY_BENCHMARKS = {
    "global_delay_rate_pct":       65.0,   # % of projects delayed globally
    "avg_delay_days":              70,     # average delay in days
    "avg_cost_overrun_pct":        28.0,   # average cost overrun %
    "Residential_delay_rate":      55.0,
    "Commercial_delay_rate":       62.0,
    "Infrastructure_delay_rate":   74.0,
    "Industrial_delay_rate":       68.0,
}

# ─── Risk Scoring ─────────────────────────────────────────────────────────────

def compute_risk_score(project: Dict) -> float:
    """
    Compute a 0–100 composite risk score for a project without ML model.
    Useful for quick heuristic assessment.
    """
    score = 0.0
    score += project.get("weather_risk_score", 5)    * 4.0
    score += project.get("material_delivery_risk", 5) * 5.0
    score += (10 - project.get("labour_availability", 5)) * 4.0
    score += project.get("design_complexity", 5)      * 3.5
    score += (10 - project.get("site_accessibility", 5)) * 2.5
    score += project.get("previous_delays", 0)        * 6.0
    score += max(0, 10 - project.get("schedule_buffer_days", 10)) * 3.0
    score += (project.get("num_subcontractors", 5) - 5) * 1.5

    contract_adj = {"Fixed Price": 8, "Design-Build": 4, "Cost Plus": 0}
    score += contract_adj.get(project.get("contract_type", "Fixed Price"), 0)

    return round(min(score, 100), 1)


# ─── Monte Carlo Simulation ───────────────────────────────────────────────────

def monte_carlo_duration(
    planned_days: int,
    delay_probability: float,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> Dict:
    """
    Simulate project completion duration using Monte Carlo sampling.

    Returns percentile completion days and probability distribution stats.
    """
    rng = np.random.default_rng(seed)

    completions = []
    for _ in range(n_simulations):
        if rng.uniform() < delay_probability:
            # Delayed: multiply by a lognormal overrun factor
            overrun = rng.lognormal(mean=np.log(1.25), sigma=0.20)
        else:
            # On time: slight variation around planned
            overrun = rng.normal(loc=1.0, scale=0.05)
        completions.append(planned_days * max(overrun, 0.85))

    completions = np.array(completions)
    return {
        "planned_days": planned_days,
        "p10_days": int(np.percentile(completions, 10)),
        "p50_days": int(np.percentile(completions, 50)),
        "p80_days": int(np.percentile(completions, 80)),
        "p90_days": int(np.percentile(completions, 90)),
        "mean_days": round(float(completions.mean()), 1),
        "std_days":  round(float(completions.std()), 1),
        "prob_on_time": round(float((completions <= planned_days).mean()), 3),
        "prob_10pct_overrun": round(float((completions > planned_days * 1.10).mean()), 3),
        "prob_25pct_overrun": round(float((completions > planned_days * 1.25).mean()), 3),
        "simulations": n_simulations,
        "histogram_data": np.histogram(completions, bins=30),
    }


# ─── Portfolio Analysis ───────────────────────────────────────────────────────

def analyse_portfolio(df: pd.DataFrame) -> Dict:
    """
    Aggregate statistics across a dataset of projects.
    """
    total = len(df)
    delayed = df["delayed"].sum() if "delayed" in df.columns else 0

    stats = {
        "total_projects": total,
        "delayed_projects": int(delayed),
        "delay_rate_pct": round(delayed / total * 100, 1) if total > 0 else 0,
        "avg_project_size_m2": round(df["project_size_m2"].mean(), 0),
        "avg_planned_duration_days": round(df["planned_duration_days"].mean(), 0),
        "avg_budget_usd": round(df["budget_usd"].mean(), 0),
        "avg_weather_risk": round(df["weather_risk_score"].mean(), 2),
        "avg_material_risk": round(df["material_delivery_risk"].mean(), 2),
        "avg_labour_availability": round(df["labour_availability"].mean(), 2),
    }

    # By project type
    if "project_type" in df.columns and "delayed" in df.columns:
        by_type = (
            df.groupby("project_type")["delayed"]
            .agg(["count", "sum", "mean"])
            .rename(columns={"count": "n", "sum": "delayed", "mean": "delay_rate"})
        )
        by_type["delay_rate_pct"] = (by_type["delay_rate"] * 100).round(1)
        stats["by_project_type"] = by_type.to_dict("index")

    # By region
    if "region" in df.columns and "delayed" in df.columns:
        by_region = (
            df.groupby("region")["delayed"]
            .mean()
            .mul(100)
            .round(1)
            .to_dict()
        )
        stats["delay_rate_by_region"] = by_region

    # By contract type
    if "contract_type" in df.columns and "delayed" in df.columns:
        by_contract = (
            df.groupby("contract_type")["delayed"]
            .mean()
            .mul(100)
            .round(1)
            .to_dict()
        )
        stats["delay_rate_by_contract"] = by_contract

    return stats


# ─── Risk Factor Correlation ──────────────────────────────────────────────────

def risk_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation between numeric risk features and delay outcome.
    """
    numeric_cols = [
        "weather_risk_score", "material_delivery_risk", "labour_availability",
        "design_complexity", "site_accessibility", "previous_delays",
        "schedule_buffer_days", "num_subcontractors", "delayed",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    corr = df[available].corr()["delayed"].drop("delayed").sort_values(
        key=abs, ascending=False
    )
    return corr.round(3)


# ─── Scenario Comparison ─────────────────────────────────────────────────────

def compare_scenarios(base_project: Dict, modifications: List[Dict]) -> pd.DataFrame:
    """
    Compare delay risk across a base project and a set of modifications.

    Parameters
    ----------
    base_project  : dict of feature values for the base scenario
    modifications : list of dicts, each containing fields to override

    Returns
    -------
    DataFrame with risk scores for each scenario
    """
    rows = []
    scenarios = [{"label": "Base Scenario", **base_project}]
    for mod in modifications:
        scenario = {**base_project, **mod}
        label = mod.get("label", f"Scenario {len(scenarios)}")
        scenario["label"] = label
        scenarios.append(scenario)

    for s in scenarios:
        score = compute_risk_score(s)
        rows.append({
            "scenario": s["label"],
            "risk_score": score,
            "weather_risk": s.get("weather_risk_score", "-"),
            "material_risk": s.get("material_delivery_risk", "-"),
            "labour_avail": s.get("labour_availability", "-"),
            "buffer_days": s.get("schedule_buffer_days", "-"),
            "prev_delays": s.get("previous_delays", "-"),
        })

    return pd.DataFrame(rows)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Construction Delay Risk Predictor")
    print("Step 3a / 3  –  Risk Analysis")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        from data_generator import generate_dataset
        generate_dataset()

    df = pd.read_csv(DATA_PATH)

    # Portfolio analysis
    stats = analyse_portfolio(df)
    print("\n[risk_analysis] Portfolio Summary:")
    for k, v in stats.items():
        if not isinstance(v, dict):
            print(f"  {k:<40} {v}")

    # Risk correlations
    corr = risk_correlation_matrix(df)
    print("\n[risk_analysis] Risk Factor Correlations with Delay:")
    print(corr.to_string())

    # Monte Carlo demo
    mc = monte_carlo_duration(planned_days=180, delay_probability=0.68)
    print(f"\n[risk_analysis] Monte Carlo (180-day project, 68% delay risk):")
    print(f"  P50 completion : {mc['p50_days']} days")
    print(f"  P80 completion : {mc['p80_days']} days")
    print(f"  P90 completion : {mc['p90_days']} days")
    print(f"  Prob on-time   : {mc['prob_on_time']*100:.1f}%")

    # Scenario comparison
    base = {
        "weather_risk_score": 6, "material_delivery_risk": 7,
        "labour_availability": 5, "design_complexity": 6,
        "site_accessibility": 6, "previous_delays": 1,
        "schedule_buffer_days": 10, "num_subcontractors": 8,
        "contract_type": "Fixed Price",
    }
    mods = [
        {"label": "Improved Labour",     "labour_availability": 8},
        {"label": "More Buffer",         "schedule_buffer_days": 20},
        {"label": "Lower Material Risk", "material_delivery_risk": 3},
        {"label": "Best Case",           "labour_availability": 9,
         "material_delivery_risk": 3,   "schedule_buffer_days": 25},
    ]
    print("\n[risk_analysis] Scenario Comparison:")
    print(compare_scenarios(base, mods).to_string(index=False))
