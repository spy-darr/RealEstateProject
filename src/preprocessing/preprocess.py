"""
preprocess.py
-------------
Data Preprocessing Module.
Handles missing values, encoding, scaling, and train/test splitting.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = ["city", "locality", "property_type"]
NUMERIC_COLS     = ["bedrooms", "bathrooms", "area_sqft", "age_years",
                    "floor_number", "total_floors", "location_score",
                    "price_per_sqft", "amenities_count"]
TARGET_COL       = "price"


# ─────────────────────────────────────────────
class DataPreprocessor:
    """
    Full preprocessing pipeline:
      1. Drop irrelevant columns
      2. Engineer amenities count from pipe-separated string
      3. Impute missing values
      4. Label-encode categorical features
      5. Standard-scale numeric features
      6. Split into train / test sets
    """

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []

    # ------------------------------------------------------------------
    def _engineer_amenities_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert pipe-separated amenities string to integer count."""
        if "amenities" in df.columns:
            df = df.copy()
            df["amenities_count"] = (
                df["amenities"]
                .fillna("")
                .apply(lambda x: len(x.split("|")) if x else 0)
            )
            df.drop(columns=["amenities"], inplace=True)
        return df

    # ------------------------------------------------------------------
    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing numeric values with column median."""
        df = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                n_missing  = int(df[col].isnull().sum())
                df[col]    = df[col].fillna(median_val)
                print(f"  [Impute] {col}: filled {n_missing} NaN → {median_val:.2f}")
        return df

    # ------------------------------------------------------------------
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode categorical columns."""
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda v: v if v in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        return df

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame):
        """
        Full fit + transform on training data.

        Returns
        -------
        X_train, X_test, y_train, y_test, feature_columns
        """
        print("[Preprocessor] Starting fit_transform …")
        df = df.copy()

        # Drop ID column if present
        if "property_id" in df.columns:
            df.drop(columns=["property_id"], inplace=True)

        df = self._engineer_amenities_count(df)
        df = self._impute_missing(df)
        df = self._encode_categoricals(df, fit=True)

        # Drop price_per_sqft to avoid data leakage during prediction
        # (it's derived from price; keep for EDA/FE only)
        # Also drop any remaining object/string columns that weren't encoded
        feature_df = df.drop(columns=[TARGET_COL, "price_per_sqft"], errors="ignore")
        feature_df = feature_df.select_dtypes(include=[np.number])
        self.feature_columns = list(feature_df.columns)

        X = feature_df.values
        y = df[TARGET_COL].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        self._save_artifacts()
        print(f"[Preprocessor] Done. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, self.feature_columns

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new/unseen data using fitted encoders and scaler.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input features (without 'price' column).

        Returns
        -------
        np.ndarray of scaled features.
        """
        df = df.copy()
        if "property_id" in df.columns:
            df.drop(columns=["property_id"], inplace=True)

        df = self._engineer_amenities_count(df)
        df = self._impute_missing(df)
        df = self._encode_categoricals(df, fit=False)

        # Keep only expected feature columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]

        return self.scaler.transform(df.values)

    # ------------------------------------------------------------------
    def _save_artifacts(self):
        """Persist encoders and scaler to disk."""
        with open(MODELS_DIR / "label_encoders.pkl", "wb") as f:
            pickle.dump(self.label_encoders, f)
        with open(MODELS_DIR / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(MODELS_DIR / "feature_columns.pkl", "wb") as f:
            pickle.dump(self.feature_columns, f)
        print("[Preprocessor] Artifacts saved.")

    @classmethod
    def load_artifacts(cls):
        """Reconstruct a preprocessor from saved artifacts."""
        inst = cls()
        with open(MODELS_DIR / "label_encoders.pkl", "rb") as f:
            inst.label_encoders = pickle.load(f)
        with open(MODELS_DIR / "scaler.pkl", "rb") as f:
            inst.scaler = pickle.load(f)
        with open(MODELS_DIR / "feature_columns.pkl", "rb") as f:
            inst.feature_columns = pickle.load(f)
        return inst


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data_collection.data_loader import load_data

    df = load_data()
    prep = DataPreprocessor()
    X_train, X_test, y_train, y_test, feats = prep.fit_transform(df)
    print("Feature columns:", feats)
    print("X_train sample:", X_train[:2])
