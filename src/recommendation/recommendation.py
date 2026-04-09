"""
recommendation.py
-----------------
Property Recommendation System.
Uses cosine similarity on engineered feature vectors to recommend
similar properties to a query listing.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing  import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Features used for similarity computation
SIMILARITY_FEATURES = [
    "bedrooms", "bathrooms", "area_sqft", "location_score",
    "amenities_count", "floor_number", "age_years",
    "city_encoded", "property_type_encoded",
]


# ─────────────────────────────────────────────
class PropertyRecommender:
    """
    Content-based property recommendation engine.

    Workflow
    --------
    1. fit(df)     — encodes & normalises the property catalogue
    2. recommend() — returns top-N most similar properties to a query
    """

    def __init__(self, top_n: int = 5):
        self.top_n       = top_n
        self.scaler      = MinMaxScaler()
        self.df_catalogue: pd.DataFrame = None
        self.feature_matrix: np.ndarray  = None
        self._city_map:  dict = {}
        self._type_map:  dict = {}

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "PropertyRecommender":
        """
        Build the similarity matrix from a property catalogue.

        Parameters
        ----------
        df : pd.DataFrame  (must contain the raw property columns)
        """
        df = df.copy()
        df = self._prepare(df)

        feature_cols = [c for c in SIMILARITY_FEATURES if c in df.columns]
        X = df[feature_cols].fillna(0).values
        self.feature_matrix = self.scaler.fit_transform(X)
        self.df_catalogue   = df.reset_index(drop=True)
        self.feature_cols   = feature_cols

        self._save()
        print(f"[Recommender] Fitted on {len(df)} properties. Features: {feature_cols}")
        return self

    # ------------------------------------------------------------------
    def recommend(
        self,
        query: dict,
        price_tolerance_pct: float = 30.0,
        same_city: bool = True,
    ) -> pd.DataFrame:
        """
        Find top-N properties similar to a query property.

        Parameters
        ----------
        query               : dict of property attributes (same keys as dataset)
        price_tolerance_pct : filter to ±x% of query price (0 = no filter)
        same_city           : restrict results to same city

        Returns
        -------
        pd.DataFrame of top-N recommended properties with similarity score
        """
        if self.feature_matrix is None:
            raise RuntimeError("Recommender not fitted. Call fit() first.")

        # Build query vector
        q_df = pd.DataFrame([query])
        q_df = self._prepare(q_df)

        q_vec = np.zeros((1, len(self.feature_cols)))
        for i, col in enumerate(self.feature_cols):
            if col in q_df.columns:
                q_vec[0, i] = float(q_df[col].fillna(0).iloc[0])
        q_vec_scaled = self.scaler.transform(q_vec)

        # Compute cosine similarity with all catalogue items
        sims = cosine_similarity(q_vec_scaled, self.feature_matrix).flatten()

        results = self.df_catalogue.copy()
        results["similarity_score"] = np.round(sims, 4)

        # Optional filters
        if same_city and "city" in query and "city" in results.columns:
            results = results[results["city"] == query["city"]]

        if price_tolerance_pct > 0 and "price" in query and "price" in results.columns:
            lo = query["price"] * (1 - price_tolerance_pct / 100)
            hi = query["price"] * (1 + price_tolerance_pct / 100)
            results = results[(results["price"] >= lo) & (results["price"] <= hi)]

        results = (
            results
            .sort_values("similarity_score", ascending=False)
            .head(self.top_n)
            .reset_index(drop=True)
        )

        # Return human-readable columns
        display_cols = [
            c for c in [
                "property_id", "city", "locality", "property_type",
                "bedrooms", "area_sqft", "price", "location_score",
                "similarity_score",
            ] if c in results.columns
        ]
        return results[display_cols]

    # ------------------------------------------------------------------
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categoricals needed for similarity."""
        df = df.copy()

        # Amenities count
        if "amenities" in df.columns and "amenities_count" not in df.columns:
            df["amenities_count"] = (
                df["amenities"].fillna("").apply(lambda x: len(x.split("|")) if x else 0)
            )

        # City encoding
        if "city" in df.columns:
            if not self._city_map:
                unique_cities = df["city"].dropna().unique()
                self._city_map = {c: i for i, c in enumerate(sorted(unique_cities))}
            df["city_encoded"] = df["city"].map(self._city_map).fillna(0).astype(int)

        # Property type encoding
        if "property_type" in df.columns:
            if not self._type_map:
                unique_types = df["property_type"].dropna().unique()
                self._type_map = {t: i for i, t in enumerate(sorted(unique_types))}
            df["property_type_encoded"] = (
                df["property_type"].map(self._type_map).fillna(0).astype(int)
            )

        return df

    # ------------------------------------------------------------------
    def _save(self):
        path = MODELS_DIR / "recommender.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "scaler":         self.scaler,
                "feature_matrix": self.feature_matrix,
                "df_catalogue":   self.df_catalogue,
                "feature_cols":   self.feature_cols,
                "city_map":       self._city_map,
                "type_map":       self._type_map,
            }, f)
        print(f"[Recommender] Model saved → {path}")

    @classmethod
    def load(cls, top_n: int = 5) -> "PropertyRecommender":
        path = MODELS_DIR / "recommender.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        inst = cls(top_n=top_n)
        inst.scaler         = data["scaler"]
        inst.feature_matrix = data["feature_matrix"]
        inst.df_catalogue   = data["df_catalogue"]
        inst.feature_cols   = data["feature_cols"]
        inst._city_map      = data["city_map"]
        inst._type_map      = data["type_map"]
        print(f"[Recommender] Loaded model — {len(inst.df_catalogue)} properties.")
        return inst


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data_collection.data_loader import load_data

    df = load_data()
    rec = PropertyRecommender(top_n=5)
    rec.fit(df)

    query_property = {
        "city":          "Pune",
        "locality":      "Baner",
        "property_type": "Apartment",
        "bedrooms":      3,
        "bathrooms":     2,
        "area_sqft":     1200,
        "age_years":     5,
        "floor_number":  4,
        "location_score": 7.0,
        "amenities":     "Gym|Parking|Security|Lift",
        "price":         8_500_000,
    }

    recs = rec.recommend(query_property)
    print("\nTop Recommendations:")
    print(recs.to_string(index=False))
