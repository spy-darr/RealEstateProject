"""
evaluate.py
-----------
Standalone model evaluation script.
Loads trained models and generates a comprehensive evaluation report.
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error
)

MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data():
    """Regenerate test split from saved artefacts."""
    from src.data_collection.data_loader import load_data
    from src.preprocessing.preprocess    import DataPreprocessor

    df   = load_data()
    prep = DataPreprocessor.load_artifacts()

    from src.feature_engineering.feature_engineer import FeatureEngineer
    fe  = FeatureEngineer()
    df  = fe.transform(df)

    from sklearn.model_selection import train_test_split
    feature_df = df.drop(columns=["price", "price_per_sqft", "property_id",
                                   "amenities", "property_age_group"], errors="ignore")
    cat_cols = ["city", "locality", "property_type"]
    for col in cat_cols:
        if col in feature_df.columns:
            feature_df.drop(columns=[col], inplace=True)

    X = df.drop(columns=["price", "price_per_sqft", "property_id",
                          "amenities", "property_age_group"], errors="ignore")

    # Use preprocessor transform
    X_scaled = prep.transform(df.drop(columns=["price"], errors="ignore"))
    y        = df["price"].values

    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42)
    return X_test, y_test


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Compute full metrics for a single model."""
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)

    return {
        "model":    name,
        "r2":       round(float(r2_score(y_test, preds)), 4),
        "rmse":     round(float(np.sqrt(mean_squared_error(y_test, preds))), 2),
        "mae":      round(float(mean_absolute_error(y_test, preds)), 2),
        "mape":     round(float(mean_absolute_percentage_error(y_test, preds)) * 100, 2),
        "preds":    preds.tolist(),
    }


def plot_actual_vs_predicted(preds, y_test, name: str):
    """Scatter plot of actual vs predicted prices."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test / 1e6, np.array(preds) / 1e6, alpha=0.3, s=10, color="#6c47ff")
    lim = [0, max(y_test.max(), np.array(preds).max()) / 1e6 * 1.05]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_title(f"Actual vs Predicted — {name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Price (₹M)")
    ax.set_ylabel("Predicted Price (₹M)")
    ax.legend()
    ax.set_xlim(lim); ax.set_ylim(lim)
    plt.tight_layout()
    safe = name.lower().replace(" ", "_")
    path = OUTPUTS_DIR / f"eval_{safe}_actual_vs_pred.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_residuals(preds, y_test, name: str):
    """Residual distribution plot."""
    residuals = np.array(preds) - y_test
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals / 1e6, bins=60, color="#00d4a8", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"Residual Distribution — {name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Residual (₹M)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    safe = name.lower().replace(" ", "_")
    path = OUTPUTS_DIR / f"eval_{safe}_residuals.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_evaluation():
    """Main evaluation routine."""
    print("\n[Evaluator] Loading test data …")
    try:
        X_test, y_test = load_test_data()
    except Exception as e:
        print(f"[Error] Could not load test data: {e}")
        print("  Run python train_pipeline.py first.")
        return

    all_metrics = []

    # Evaluate each saved model
    model_names = {
        "linear_regression": "Linear Regression",
        "ridge_regression":  "Ridge Regression",
        "random_forest":     "Random Forest",
        "xgboost":           "XGBoost",
        "gradientboosting":  "GradientBoosting",
    }

    for fname, display_name in model_names.items():
        path = MODELS_DIR / f"{fname}.pkl"
        if not path.exists():
            continue
        print(f"  Evaluating {display_name} …")
        with open(path, "rb") as f:
            model = pickle.load(f)
        metrics = evaluate_model(model, X_test, y_test, display_name)
        all_metrics.append({k: v for k, v in metrics.items() if k != "preds"})
        plot_actual_vs_predicted(metrics["preds"], y_test, display_name)
        plot_residuals(metrics["preds"], y_test, display_name)

    if not all_metrics:
        print("[Warning] No saved models found. Run train_pipeline.py first.")
        return

    # Summary table
    df_metrics = pd.DataFrame(all_metrics).sort_values("r2", ascending=False)
    print("\n" + "=" * 65)
    print(f"{'Model':<25} {'R²':>8} {'RMSE':>14} {'MAE':>14} {'MAPE%':>7}")
    print("-" * 65)
    for _, row in df_metrics.iterrows():
        print(f"  {row['model']:<23} {row['r2']:>8.4f} "
              f"₹{row['rmse']:>12,.0f} ₹{row['mae']:>12,.0f} {row['mape']:>6.1f}%")
    print("=" * 65)

    # Save summary
    path = OUTPUTS_DIR / "evaluation_report.json"
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[Evaluator] Report saved → {path}")
    print(f"[Evaluator] Plots saved  → {OUTPUTS_DIR}")


if __name__ == "__main__":
    run_evaluation()
