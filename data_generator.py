"""
data_generator.py
-----------------
Generates a synthetic construction project dataset for delay risk prediction.

Each record represents one construction project with features known at
project initiation — the target variable is whether the project experienced
a significant delay (>10% over planned duration).

Run:
    python src/data_generator.py

Output:
    data/construction_projects.csv
"""

import os
import numpy as np
import pandas as pd

# ─── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
N_SAMPLES = 1200  # realistic dataset size for ML training


# ─── Feature Distributions ────────────────────────────────────────────────────

def generate_dataset(n: int = N_SAMPLES, save_path: str = "data/construction_projects.csv") -> pd.DataFrame:
    """
    Generate synthetic construction project data with realistic feature
    correlations and a logistic delay probability model.

    Features
    --------
    project_size_m2         : building footprint in square metres
    num_workers             : peak workforce on site
    num_subcontractors      : number of specialist subcontractor firms
    planned_duration_days   : originally scheduled project duration
    budget_usd              : total approved project budget
    schedule_buffer_days    : contingency days built into the schedule
    weather_risk_score      : 1–10 composite climate/season risk index
    material_delivery_risk  : 1–10 risk of procurement/supply chain delays
    labour_availability     : 1–10 score (10 = abundant local labour)
    design_complexity       : 1–10 architectural/structural complexity
    site_accessibility      : 1–10 (10 = easy access, city-centre site)
    previous_delays         : number of delays on earlier phases (0–4)
    contract_type           : Fixed Price / Cost Plus / Design-Build
    project_type            : Residential / Commercial / Infrastructure / Industrial
    region                  : North / South / East / West / Central
    """
    rng = np.random.default_rng(seed=42)

    n = int(n)

    # ── Continuous features ───────────────────────────────────────────────────
    project_size   = rng.lognormal(mean=7.5, sigma=0.8, size=n).clip(200, 80000)
    num_workers    = (project_size / 120 + rng.normal(0, 5, n)).clip(5, 300).astype(int)
    num_subs       = rng.integers(1, 18, size=n)
    planned_days   = (project_size / 30 + rng.normal(0, 20, n)).clip(30, 900).astype(int)
    budget         = project_size * rng.uniform(1800, 3500, n)
    schedule_buf   = rng.integers(0, 31, size=n)

    # Risk scores (1–10)
    weather_risk   = rng.integers(1, 11, size=n)
    material_risk  = rng.integers(1, 11, size=n)
    labour_avail   = rng.integers(1, 11, size=n)
    design_cmplx   = rng.integers(1, 11, size=n)
    site_access    = rng.integers(1, 11, size=n)
    prev_delays    = rng.integers(0, 5,  size=n)

    # ── Categorical features ──────────────────────────────────────────────────
    contract_types = rng.choice(
        ["Fixed Price", "Cost Plus", "Design-Build"],
        size=n, p=[0.50, 0.25, 0.25]
    )
    project_types  = rng.choice(
        ["Residential", "Commercial", "Infrastructure", "Industrial"],
        size=n, p=[0.35, 0.30, 0.20, 0.15]
    )
    regions        = rng.choice(
        ["North", "South", "East", "West", "Central"],
        size=n
    )

    # ── Delay probability model (logistic) ────────────────────────────────────
    # Higher risk factors → higher delay probability
    log_odds = (
        -3.5
        + 0.25 * weather_risk
        + 0.30 * material_risk
        - 0.20 * labour_avail          # higher avail = lower risk
        + 0.22 * design_cmplx
        - 0.18 * site_access           # better access = lower risk
        + 0.40 * prev_delays
        - 0.04 * schedule_buf          # more buffer = lower risk
        + 0.0001 * num_subs * design_cmplx
        + 0.15 * (num_subs > 10).astype(float)
        + 0.20 * (contract_types == "Fixed Price").astype(float)
        - 0.15 * (contract_types == "Cost Plus").astype(float)
        + 0.18 * (project_types == "Infrastructure").astype(float)
        + rng.normal(0, 0.4, n)        # residual noise
    )
    delay_prob  = 1 / (1 + np.exp(-log_odds))
    delay_label = (rng.uniform(size=n) < delay_prob).astype(int)

    # Actual duration (for context — not used as feature)
    duration_multiplier = np.where(
        delay_label == 1,
        rng.uniform(1.10, 1.60, n),
        rng.uniform(0.90, 1.08, n),
    )
    actual_days = (planned_days * duration_multiplier).astype(int)
    delay_days  = np.maximum(0, actual_days - planned_days)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "project_id":             [f"PRJ{i+1:04d}" for i in range(n)],
        "project_size_m2":        project_size.round(1),
        "num_workers":            num_workers,
        "num_subcontractors":     num_subs,
        "planned_duration_days":  planned_days,
        "budget_usd":             budget.round(0).astype(int),
        "schedule_buffer_days":   schedule_buf,
        "weather_risk_score":     weather_risk,
        "material_delivery_risk": material_risk,
        "labour_availability":    labour_avail,
        "design_complexity":      design_cmplx,
        "site_accessibility":     site_access,
        "previous_delays":        prev_delays,
        "contract_type":          contract_types,
        "project_type":           project_types,
        "region":                 regions,
        "delay_probability":      delay_prob.round(4),
        "actual_duration_days":   actual_days,
        "delay_days":             delay_days,
        "delayed":                delay_label,          # TARGET variable
    })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"[data_generator] Dataset saved → {save_path}")
    print(f"  Samples      : {len(df):,}")
    print(f"  Delayed      : {delay_label.sum():,}  ({delay_label.mean()*100:.1f}%)")
    print(f"  Not delayed  : {(1-delay_label).sum():,}  ({(1-delay_label).mean()*100:.1f}%)")
    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Construction Delay Risk Predictor")
    print("Step 1 / 3  –  Data Generation")
    print("=" * 60)
    df = generate_dataset()
    print("\nSample records:")
    print(df[["project_id", "project_size_m2", "num_workers",
              "material_delivery_risk", "weather_risk_score",
              "delayed"]].head(8).to_string(index=False))
