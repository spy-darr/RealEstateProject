"""
train_pipeline.py
-----------------
End-to-end training pipeline.
Run this script once to:
  1. Load / generate data
  2. Run EDA
  3. Engineer features
  4. Preprocess
  5. Train all ML models
  6. Train deep learning model
  7. Fit recommender
  8. Fit anomaly detector
  9. Generate sample predictions & investment report
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_collection.data_loader            import load_data, save_data
from src.eda.eda                                import EDAAnalyzer
from src.feature_engineering.feature_engineer  import run_feature_engineering
from src.preprocessing.preprocess              import DataPreprocessor
from src.models.train_model                    import ModelTrainer
from src.deep_learning.deep_learning_model     import DeepLearningModel
from src.recommendation.recommendation         import PropertyRecommender
from src.investment_analysis.investment_analysis import InvestmentAnalyzer
from src.anomaly_detection.anomaly_detection   import AnomalyDetector

OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\n" + "═" * 60)
    print("   AI Real Estate Analytics — Training Pipeline")
    print("═" * 60)

    # ── 1. Data ──────────────────────────────────────────────────────
    print("\n[1/8] Loading dataset …")
    df_raw = load_data()
    print(f"      Shape: {df_raw.shape}")

    # ── 2. EDA ───────────────────────────────────────────────────────
    print("\n[2/8] Running EDA …")
    try:
        eda = EDAAnalyzer(df_raw)
        eda.run_all()
    except Exception as e:
        print(f"      [EDA Warning] {e}")

    # ── 3. Feature Engineering ────────────────────────────────────────
    print("\n[3/8] Feature Engineering …")
    df_enriched = run_feature_engineering(df_raw, save=True)
    print(f"      New columns: {[c for c in df_enriched.columns if c not in df_raw.columns]}")

    # ── 4. Preprocessing ─────────────────────────────────────────────
    print("\n[4/8] Preprocessing …")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.fit_transform(df_enriched)

    # ── 5. ML Models ─────────────────────────────────────────────────
    print("\n[5/8] Training ML models …")
    trainer = ModelTrainer()
    results = trainer.train(X_train, X_test, y_train, y_test)
    trainer.print_comparison_table()
    trainer.save_best_model()
    trainer.save_all_models()
    fi = trainer.get_feature_importance(feature_cols)
    print(f"      Top features: {fi['feature'].head(5).tolist() if len(fi) else 'N/A'}")

    # ── 6. Deep Learning ─────────────────────────────────────────────
    print("\n[6/8] Training Deep Learning model …")
    try:
        dl = DeepLearningModel()
        dl.build(X_train.shape[1])
        dl.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=64)
        dl_metrics = dl.evaluate(X_test, y_test)
        dl.save()
    except Exception as e:
        print(f"      [DL Warning] {e}")
        dl_metrics = {}

    # ── 7. Recommender ───────────────────────────────────────────────
    print("\n[7/8] Fitting Recommender …")
    recommender = PropertyRecommender(top_n=5)
    recommender.fit(df_enriched)

    # ── 8. Anomaly Detector ──────────────────────────────────────────
    print("\n[8/8] Fitting Anomaly Detector …")
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(df_enriched)
    flagged   = detector.detect(df_enriched)
    stats     = detector.summary(flagged)
    print(f"      Anomalies found: {stats['total_anomalies']} ({stats['anomaly_pct']}%)")

    # ── Sample Predictions ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  SAMPLE PREDICTIONS")
    print("─" * 60)
    sample_props = [
        {
            "city": "Pune", "locality": "Koregaon Park", "property_type": "Apartment",
            "bedrooms": 3, "bathrooms": 2, "area_sqft": 1350, "age_years": 4,
            "floor_number": 6, "total_floors": 14, "location_score": 8.5,
            "amenities": "Gym|Pool|Parking|Security|Lift|CCTV",
        },
        {
            "city": "Mumbai", "locality": "Bandra", "property_type": "Penthouse",
            "bedrooms": 4, "bathrooms": 4, "area_sqft": 2800, "age_years": 2,
            "floor_number": 20, "total_floors": 22, "location_score": 9.5,
            "amenities": "Gym|Pool|Parking|Security|Lift|Club House|CCTV",
        },
        {
            "city": "Jaipur", "locality": "Vaishali Nagar", "property_type": "Villa",
            "bedrooms": 4, "bathrooms": 3, "area_sqft": 2200, "age_years": 8,
            "floor_number": 0, "total_floors": 2, "location_score": 7.2,
            "amenities": "Garden|Parking|Security|Power Backup",
        },
    ]

    predictions = []
    for sp in sample_props:
        sp_df = pd.DataFrame([sp])
        try:
            X = preprocessor.transform(sp_df)
            price = float(trainer.best_model.predict(X)[0])
            price = max(price, 0)
            ppsf  = round(price / sp["area_sqft"], 2)
            print(f"  {sp['property_type']:18s} | {sp['city']:12s} | {sp['area_sqft']} sqft → "
                  f"₹{price:>12,.0f}  (₹{ppsf:,.0f}/sqft)")
            predictions.append({"input": sp, "predicted_price": price, "price_per_sqft": ppsf})
        except Exception as e:
            print(f"  Prediction failed: {e}")

    preds_path = OUTPUTS_DIR / "sample_predictions.json"
    with open(preds_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\n  Saved → {preds_path}")

    # ── Sample Investment Report ──────────────────────────────────────
    print("\n" + "─" * 60)
    analyzer = InvestmentAnalyzer()
    report   = analyzer.analyze(price=8_500_000, city="Pune")
    analyzer.print_report(report)

    # ── Model Comparison Summary ──────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE ✓")
    print("═" * 60)
    print(f"  Best ML Model : {trainer.best_model_name}")
    print(f"  Best R²       : {results[trainer.best_model_name]['r2']:.4f}")
    if dl_metrics:
        print(f"  DL Model R²   : {dl_metrics.get('r2', 'N/A')}")
    print(f"  Outputs saved : {OUTPUTS_DIR}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
