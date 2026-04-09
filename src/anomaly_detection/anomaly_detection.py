"""
anomaly_detection.py
--------------------
Anomaly Detection Module.
Detects price anomalies and suspicious listings using Isolation Forest,
Z-score analysis, and IQR fencing.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODELS_DIR  = Path(__file__).resolve().parents[2] / "models"
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ANOMALY_FEATURES = [
    "price", "area_sqft", "price_per_sqft",
    "bedrooms", "bathrooms", "location_score",
]


# ─────────────────────────────────────────────
class AnomalyDetector:
    """
    Multi-method anomaly detection for real estate listings.

    Methods
    -------
    1. Isolation Forest  — model-based global anomaly score
    2. Z-score           — statistical outlier per numeric feature
    3. IQR Fence         — interquartile range outlier flag
    4. Domain rules      — price/sqft extreme thresholds
    """

    def __init__(self, contamination: float = 0.05):
        """
        Parameters
        ----------
        contamination : estimated fraction of anomalies in dataset (0-0.5)
        """
        self.contamination  = contamination
        self.iso_forest     = IsolationForest(
            n_estimators=200, contamination=contamination,
            max_samples="auto", random_state=42, n_jobs=-1
        )
        self.scaler         = StandardScaler()
        self._is_fitted     = False
        self.thresholds: dict = {}

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """
        Fit detector on a clean (or full) dataset.

        Parameters
        ----------
        df : DataFrame with at least ANOMALY_FEATURES columns
        """
        df = self._add_ppsf(df)
        feature_cols = [c for c in ANOMALY_FEATURES if c in df.columns]
        X = df[feature_cols].fillna(df[feature_cols].median()).values

        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)

        # Store IQR thresholds for later use
        for col in feature_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            self.thresholds[col] = {
                "q1": q1, "q3": q3, "iqr": iqr,
                "lower": q1 - 3 * iqr,
                "upper": q3 + 3 * iqr,
            }

        self._is_fitted  = True
        self.feature_cols = feature_cols
        self._save()
        print(f"[AnomalyDetector] Fitted on {len(df)} samples. Features: {feature_cols}")
        return self

    # ------------------------------------------------------------------
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every row in df for anomalies.

        Returns
        -------
        df with added columns:
          anomaly_score         : Isolation Forest score (-1 worst .. +1 best)
          is_anomaly_iso        : 1 if IF flags as anomaly
          z_score_price         : Z-score of price
          is_anomaly_zscore     : 1 if |z| > 3 on price
          is_anomaly_iqr        : 1 if any feature outside 3*IQR
          anomaly_flag          : combined flag (1 = suspicious listing)
          anomaly_reason        : human-readable reason string
        """
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        df = df.copy()
        df = self._add_ppsf(df)

        # ── Isolation Forest ─────────────────────────────────────────
        feature_cols = [c for c in self.feature_cols if c in df.columns]
        X = df[feature_cols].fillna(df[feature_cols].median()).values
        X_scaled   = self.scaler.transform(X)
        if_preds   = self.iso_forest.predict(X_scaled)           # +1 normal / -1 anomaly
        if_scores  = self.iso_forest.score_samples(X_scaled)     # lower = more anomalous

        df["anomaly_score"]      = np.round(if_scores, 4)
        df["is_anomaly_iso"]     = (if_preds == -1).astype(int)

        # ── Z-score on price ─────────────────────────────────────────
        price_mean = df["price"].mean()
        price_std  = df["price"].std() + 1e-9
        df["z_score_price"]   = ((df["price"] - price_mean) / price_std).round(4)
        df["is_anomaly_zscore"] = (df["z_score_price"].abs() > 3).astype(int)

        # ── IQR fencing ──────────────────────────────────────────────
        def _iqr_flag(row):
            for col, bounds in self.thresholds.items():
                if col not in df.columns:
                    continue
                if row[col] < bounds["lower"] or row[col] > bounds["upper"]:
                    return 1
            return 0
        df["is_anomaly_iqr"] = df.apply(_iqr_flag, axis=1)

        # ── Domain rules ─────────────────────────────────────────────
        if "price_per_sqft" in df.columns:
            df["is_anomaly_domain"] = (
                (df["price_per_sqft"] < 1000) | (df["price_per_sqft"] > 100_000)
            ).astype(int)
        else:
            df["is_anomaly_domain"] = 0

        # ── Combined flag ─────────────────────────────────────────────
        df["anomaly_flag"] = (
            (df["is_anomaly_iso"]    == 1) |
            (df["is_anomaly_zscore"] == 1) |
            (df["is_anomaly_iqr"]    == 1) |
            (df["is_anomaly_domain"] == 1)
        ).astype(int)

        # ── Human-readable reason ─────────────────────────────────────
        def _reason(row):
            reasons = []
            if row["is_anomaly_iso"]:
                reasons.append("IF-score outlier")
            if row["is_anomaly_zscore"]:
                reasons.append(f"price Z={row['z_score_price']:.1f}")
            if row["is_anomaly_iqr"]:
                reasons.append("IQR outlier")
            if row["is_anomaly_domain"]:
                reasons.append("price/sqft extreme")
            return "; ".join(reasons) if reasons else "normal"
        df["anomaly_reason"] = df.apply(_reason, axis=1)

        # ── Summary ──────────────────────────────────────────────────
        n_anomalies = df["anomaly_flag"].sum()
        pct         = n_anomalies / len(df) * 100
        print(f"[AnomalyDetector] Detected {n_anomalies} anomalies ({pct:.1f}%)")

        # Save anomalous subset
        anomalies = df[df["anomaly_flag"] == 1]
        path = OUTPUTS_DIR / "anomalies.csv"
        anomalies.to_csv(path, index=False)
        print(f"[AnomalyDetector] Anomalies saved → {path}")

        return df

    # ------------------------------------------------------------------
    def summary(self, df_flagged: pd.DataFrame) -> dict:
        """Return summary statistics about detected anomalies."""
        anomalies = df_flagged[df_flagged["anomaly_flag"] == 1]
        stats = {
            "total_records":   len(df_flagged),
            "total_anomalies": int(anomalies.shape[0]),
            "anomaly_pct":     round(anomalies.shape[0] / len(df_flagged) * 100, 2),
            "if_count":        int(df_flagged["is_anomaly_iso"].sum()),
            "zscore_count":    int(df_flagged["is_anomaly_zscore"].sum()),
            "iqr_count":       int(df_flagged["is_anomaly_iqr"].sum()),
            "domain_count":    int(df_flagged.get("is_anomaly_domain", pd.Series([0])).sum()),
        }
        path = OUTPUTS_DIR / "anomaly_summary.json"
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats

    # ------------------------------------------------------------------
    @staticmethod
    def _add_ppsf(df: pd.DataFrame) -> pd.DataFrame:
        if "price_per_sqft" not in df.columns and "price" in df.columns and "area_sqft" in df.columns:
            df = df.copy()
            df["price_per_sqft"] = df["price"] / df["area_sqft"].replace(0, 1)
        return df

    # ------------------------------------------------------------------
    def _save(self):
        path = MODELS_DIR / "anomaly_detector.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "iso_forest":   self.iso_forest,
                "scaler":       self.scaler,
                "thresholds":   self.thresholds,
                "feature_cols": self.feature_cols,
            }, f)
        print(f"[AnomalyDetector] Model saved → {path}")

    @classmethod
    def load(cls) -> "AnomalyDetector":
        path = MODELS_DIR / "anomaly_detector.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls()
        inst.iso_forest   = data["iso_forest"]
        inst.scaler       = data["scaler"]
        inst.thresholds   = data["thresholds"]
        inst.feature_cols = data["feature_cols"]
        inst._is_fitted   = True
        return inst


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data_collection.data_loader import load_data

    df = load_data()
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(df)

    flagged = detector.detect(df)
    stats   = detector.summary(flagged)
    print(json.dumps(stats, indent=2))
    print(flagged[flagged["anomaly_flag"] == 1][["property_id", "price", "area_sqft",
                                                  "anomaly_score", "anomaly_reason"]].head(10))
