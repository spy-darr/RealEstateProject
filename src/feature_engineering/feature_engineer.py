"""
feature_engineer.py
-------------------
Feature Engineering Module.
Creates derived / enriched features to improve model performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# City-level median price per sqft (₹) — baseline for relative scoring
CITY_MEDIAN_PPSF = {
    "Mumbai": 25000, "Delhi": 18000, "Bangalore": 12000,
    "Pune": 9000,    "Hyderabad": 8500, "Chennai": 9500,
    "Kolkata": 7000, "Ahmedabad": 6500, "Jaipur": 6000, "Surat": 5500,
}

PREMIUM_LOCALITIES = {
    "Bandra", "Juhu", "Worli", "Koregaon Park", "Koramangala",
    "Indiranagar", "Banjara Hills", "Jubilee Hills", "Adyar",
    "Vasant Kunj", "Alipore", "C-Scheme", "Vesu", "Satellite",
}


class FeatureEngineer:
    """
    Adds domain-specific engineered features to the raw property DataFrame.

    New columns produced
    --------------------
    price_per_sqft      : price / area_sqft  (if price present)
    amenities_count     : count of amenities from pipe-separated string
    amenities_score     : normalised 0-10 amenities score
    bed_bath_ratio      : bedrooms / bathrooms
    floor_ratio         : floor_number / total_floors
    area_per_bedroom    : area_sqft / bedrooms
    city_price_index    : city baseline price per sqft normalised to [0,1]
    is_premium_locality : 1 if locality is a premium area, else 0
    property_age_group  : categorical: New / Mid / Old
    value_score         : composite investment attractiveness score (0-10)
    """

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, has_price: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering steps.

        Parameters
        ----------
        df        : raw (or partially processed) DataFrame
        has_price : include price-derived features only when price is available

        Returns
        -------
        pd.DataFrame with additional feature columns
        """
        df = df.copy()

        df = self._amenities_features(df)
        df = self._ratio_features(df)
        df = self._location_features(df)
        df = self._age_group(df)
        df = self._value_score(df)

        if has_price and "price" in df.columns and "area_sqft" in df.columns:
            df["price_per_sqft"] = (df["price"] / df["area_sqft"]).round(2)

        print(f"[FeatureEngineer] Added features. Shape: {df.shape}")
        return df

    # ------------------------------------------------------------------
    def _amenities_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "amenities" in df.columns:
            df["amenities_count"] = (
                df["amenities"].fillna("").apply(lambda x: len(x.split("|")) if x else 0)
            )
            max_amenities = 10
            df["amenities_score"] = (df["amenities_count"] / max_amenities * 10).clip(0, 10)
        else:
            df.setdefault("amenities_count", 0)
            df.setdefault("amenities_score", 0)
        return df

    # ------------------------------------------------------------------
    def _ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bed_bath_ratio"] = (
            df["bedrooms"] / df["bathrooms"].replace(0, 1)
        ).round(2)

        df["floor_ratio"] = (
            df.get("floor_number", pd.Series(np.zeros(len(df)))) /
            df.get("total_floors",  pd.Series(np.ones(len(df)))).replace(0, 1)
        ).round(3)

        df["area_per_bedroom"] = (
            df["area_sqft"] / df["bedrooms"].replace(0, 1)
        ).round(2)

        return df

    # ------------------------------------------------------------------
    def _location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # City price index: normalise city median ppsf to [0,1]
        max_ppsf = max(CITY_MEDIAN_PPSF.values())
        df["city_price_index"] = df["city"].map(
            {c: round(v / max_ppsf, 4) for c, v in CITY_MEDIAN_PPSF.items()}
        ).fillna(0.5)

        # Premium locality flag
        df["is_premium_locality"] = (
            df["locality"].isin(PREMIUM_LOCALITIES).astype(int)
        )

        return df

    # ------------------------------------------------------------------
    def _age_group(self, df: pd.DataFrame) -> pd.DataFrame:
        def _group(age):
            if pd.isna(age):
                return "Unknown"
            if age <= 3:
                return "New"
            elif age <= 12:
                return "Mid"
            else:
                return "Old"

        df["property_age_group"] = df.get(
            "age_years", pd.Series(["Unknown"] * len(df))
        ).apply(_group)

        return df

    # ------------------------------------------------------------------
    def _value_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Composite value / investment attractiveness score (0-10).
        Weights: location_score 40%, amenities_score 30%, age inverse 20%, floor_ratio 10%.
        """
        age_score = (
            (1 - df.get("age_years", pd.Series(np.zeros(len(df)))).fillna(15) / 30)
            .clip(0, 1) * 10
        )
        location_part  = df.get("location_score",  pd.Series(np.full(len(df), 5.0))) * 0.40
        amenities_part = df.get("amenities_score", pd.Series(np.zeros(len(df))))    * 0.30
        age_part       = age_score * 0.20
        floor_part     = df.get("floor_ratio", pd.Series(np.zeros(len(df))))         * 10 * 0.10

        df["value_score"] = (location_part + amenities_part + age_part + floor_part).round(2).clip(0, 10)
        return df


# ─────────────────────────────────────────────
def run_feature_engineering(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Convenience function: run FE and optionally save enriched dataset.

    Returns
    -------
    Enriched DataFrame
    """
    fe = FeatureEngineer()
    df_enriched = fe.transform(df)
    if save:
        out_path = DATA_DIR / "properties_enriched.csv"
        df_enriched.to_csv(out_path, index=False)
        print(f"[FeatureEngineer] Enriched dataset saved → {out_path}")
    return df_enriched


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data_collection.data_loader import load_data

    raw = load_data()
    enriched = run_feature_engineering(raw)
    print(enriched[["price_per_sqft", "amenities_score", "value_score", "property_age_group"]].head(10))
