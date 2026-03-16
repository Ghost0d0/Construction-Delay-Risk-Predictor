"""
train_model.py
--------------
Trains, evaluates, and persists machine learning models for construction
delay risk prediction.

Models trained:
  1. Gradient Boosting Classifier  (primary)
  2. Random Forest Classifier       (benchmark)
  3. Logistic Regression            (baseline)

Run:
    python src/train_model.py

Output:
    models/delay_model.pkl
    models/model_evaluation.json
    models/feature_importance.csv
"""

import json
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/construction_projects.csv"
MODEL_DIR  = "models"

# ─── Feature Configuration ────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "project_size_m2",
    "num_workers",
    "num_subcontractors",
    "planned_duration_days",
    "budget_usd",
    "schedule_buffer_days",
    "weather_risk_score",
    "material_delivery_risk",
    "labour_availability",
    "design_complexity",
    "site_accessibility",
    "previous_delays",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "project_type",
    "region",
]

TARGET = "delayed"


# ─── Preprocessing Pipeline ───────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])


# ─── Model Definitions ────────────────────────────────────────────────────────

def get_models() -> Dict[str, object]:
    return {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            min_samples_leaf=20,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    }


# ─── Training & Evaluation ────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    drop_cols = ["project_id", "delay_probability", "actual_duration_days",
                 "delay_days", TARGET]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET]
    return X, y


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    num_names = NUMERIC_FEATURES
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        CATEGORICAL_FEATURES
    ).tolist()
    return num_names + cat_names


def train_all_models(
    X_train, X_test, y_train, y_test,
    preprocessor: ColumnTransformer,
) -> Tuple[Pipeline, Dict, pd.DataFrame]:
    """Train all models, return best pipeline + evaluation report."""

    models     = get_models()
    results    = {}
    best_auc   = 0
    best_pipe  = None
    best_name  = ""

    for name, clf in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train,
                                    cv=cv, scoring="roc_auc", n_jobs=-1)

        pipe.fit(X_train, y_train)
        metrics = evaluate_model(pipe, X_test, y_test)
        metrics["cv_roc_auc_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_roc_auc_std"]  = round(cv_scores.std(), 4)
        results[name] = metrics

        print(f"  {name:<25} AUC={metrics['roc_auc']:.4f}  "
              f"F1={metrics['f1']:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

        if metrics["roc_auc"] > best_auc:
            best_auc  = metrics["roc_auc"]
            best_pipe = pipe
            best_name = name

    print(f"\n  ✓ Best model: {best_name}  (AUC={best_auc:.4f})")

    # Feature importance from best model
    feat_names = get_feature_names(best_pipe.named_steps["preprocessor"])
    clf = best_pipe.named_steps["classifier"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        importances = np.abs(clf.coef_[0])

    feat_df = pd.DataFrame({
        "feature":    feat_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    feat_df["importance_pct"] = (feat_df["importance"] / feat_df["importance"].sum() * 100).round(2)

    return best_pipe, {"best_model": best_name, "metrics": results}, feat_df


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_artifacts(pipe: Pipeline, evaluation: Dict, feat_df: pd.DataFrame) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(f"{MODEL_DIR}/delay_model.pkl", "wb") as f:
        pickle.dump(pipe, f)

    with open(f"{MODEL_DIR}/model_evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)

    feat_df.to_csv(f"{MODEL_DIR}/feature_importance.csv", index=False)
    print(f"[train_model] Artifacts saved → {MODEL_DIR}/")


def load_model(path: str = f"{MODEL_DIR}/delay_model.pkl") -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Prediction API ───────────────────────────────────────────────────────────

def predict_delay_risk(model: Pipeline, project: Dict) -> Dict:
    """
    Predict delay probability for a single project.

    Parameters
    ----------
    model   : fitted sklearn Pipeline
    project : dict with feature values

    Returns
    -------
    dict with delay_probability, risk_level, top_risk_factors
    """
    X = pd.DataFrame([project])

    # Ensure all expected columns are present
    for col in NUMERIC_FEATURES:
        if col not in X.columns:
            X[col] = 0
    for col in CATEGORICAL_FEATURES:
        if col not in X.columns:
            X[col] = "Unknown"

    proba = model.predict_proba(X)[0][1]

    if proba < 0.30:
        risk_level = "Low"
        risk_color = "green"
    elif proba < 0.55:
        risk_level = "Moderate"
        risk_color = "orange"
    elif proba < 0.75:
        risk_level = "High"
        risk_color = "red"
    else:
        risk_level = "Critical"
        risk_color = "darkred"

    # Top contributing risk factors (rule-based interpretation)
    risk_factors = _identify_risk_factors(project)

    return {
        "delay_probability": round(float(proba), 4),
        "delay_probability_pct": round(float(proba) * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "top_risk_factors": risk_factors,
    }


def _identify_risk_factors(project: Dict) -> list:
    """Rule-based risk factor identification for explainability."""
    factors = []

    if project.get("material_delivery_risk", 0) >= 7:
        factors.append(("Material procurement delay", project["material_delivery_risk"] / 10))
    if project.get("weather_risk_score", 0) >= 7:
        factors.append(("Adverse weather conditions", project["weather_risk_score"] / 10))
    if project.get("labour_availability", 10) <= 4:
        factors.append(("Low labour availability", (10 - project["labour_availability"]) / 10))
    if project.get("design_complexity", 0) >= 7:
        factors.append(("High design complexity", project["design_complexity"] / 10))
    if project.get("previous_delays", 0) >= 2:
        factors.append(("History of previous delays", project["previous_delays"] / 4))
    if project.get("num_subcontractors", 0) >= 12:
        factors.append(("Large subcontractor network", min(project["num_subcontractors"] / 17, 1.0)))
    if project.get("schedule_buffer_days", 10) <= 5:
        factors.append(("Insufficient schedule buffer", (10 - project["schedule_buffer_days"]) / 10))
    if project.get("site_accessibility", 10) <= 4:
        factors.append(("Poor site accessibility", (10 - project["site_accessibility"]) / 10))
    if project.get("contract_type") == "Fixed Price":
        factors.append(("Fixed price contract pressure", 0.55))

    # Sort by impact score descending
    factors.sort(key=lambda x: x[1], reverse=True)
    return [{"factor": f, "impact": round(s, 2)} for f, s in factors[:5]]


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Construction Delay Risk Predictor")
    print("Step 2 / 3  –  Model Training")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print("[train_model] Generating dataset first …")
        from data_generator import generate_dataset
        generate_dataset()

    X, y = load_data()
    print(f"\n[train_model] Loaded {len(X):,} samples | "
          f"Delay rate: {y.mean()*100:.1f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor()
    print("\n[train_model] Training models …\n")
    best_pipe, evaluation, feat_df = train_all_models(
        X_train, X_test, y_train, y_test, preprocessor
    )

    save_artifacts(best_pipe, evaluation, feat_df)

    print("\nTop 10 Feature Importances:")
    print(feat_df.head(10).to_string(index=False))

    # Demo prediction
    demo = {
        "project_size_m2": 5000,
        "num_workers": 45,
        "num_subcontractors": 8,
        "planned_duration_days": 180,
        "budget_usd": 12_000_000,
        "schedule_buffer_days": 10,
        "weather_risk_score": 7,
        "material_delivery_risk": 8,
        "labour_availability": 4,
        "design_complexity": 6,
        "site_accessibility": 5,
        "previous_delays": 2,
        "contract_type": "Fixed Price",
        "project_type": "Commercial",
        "region": "North",
    }
    result = predict_delay_risk(best_pipe, demo)
    print(f"\nDemo prediction → Delay Risk: {result['delay_probability_pct']}%  "
          f"[{result['risk_level']}]")
    print("Top risk factors:")
    for rf in result["top_risk_factors"]:
        print(f"  • {rf['factor']}  (impact: {rf['impact']})")
