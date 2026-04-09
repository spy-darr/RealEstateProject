"""
data_loader.py
--------------
Data Collection Module for Real Estate Analytics Platform.
Loads or generates synthetic property datasets for model training.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASET_FILE = DATA_DIR / "properties.csv"

CITIES = ["Mumbai", "Pune", "Bangalore", "Hyderabad", "Chennai",
          "Delhi", "Kolkata", "Ahmedabad", "Jaipur", "Surat"]

PROPERTY_TYPES = ["Apartment", "Villa", "Independent House", "Studio", "Penthouse"]

LOCALITIES = {
    "Mumbai":     ["Bandra", "Andheri", "Worli", "Juhu", "Powai"],
    "Pune":       ["Koregaon Park", "Hinjewadi", "Kothrud", "Viman Nagar", "Baner"],
    "Bangalore":  ["Whitefield", "Koramangala", "Indiranagar", "HSR Layout", "Marathahalli"],
    "Hyderabad":  ["Gachibowli", "Hitech City", "Banjara Hills", "Jubilee Hills", "Madhapur"],
    "Chennai":    ["Anna Nagar", "Velachery", "OMR", "Adyar", "T.Nagar"],
    "Delhi":      ["Dwarka", "Vasant Kunj", "Rohini", "Saket", "Lajpat Nagar"],
    "Kolkata":    ["Salt Lake", "New Town", "Alipore", "Ballygunge", "Park Street"],
    "Ahmedabad":  ["Bodakdev", "Satellite", "Prahlad Nagar", "Navrangpura", "Thaltej"],
    "Jaipur":     ["C-Scheme", "Vaishali Nagar", "Malviya Nagar", "Tonk Road", "Jagatpura"],
    "Surat":      ["Adajan", "Vesu", "Piplod", "City Light", "Katargam"],
}

AMENITIES = ["Swimming Pool", "Gym", "Parking", "Security", "Lift",
             "Garden", "Club House", "Power Backup", "CCTV", "Children Play Area"]


# ─────────────────────────────────────────────
# Synthetic Data Generator
# ─────────────────────────────────────────────
def generate_synthetic_data(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic housing dataset for Indian cities.

    Parameters
    ----------
    n_samples : int
        Number of property records to generate.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with realistic property features and price.
    """
    np.random.seed(random_state)
    records = []

    for _ in range(n_samples):
        city = np.random.choice(CITIES)
        locality = np.random.choice(LOCALITIES[city])
        prop_type = np.random.choice(PROPERTY_TYPES)

        bedrooms    = np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.30, 0.35, 0.18, 0.07])
        bathrooms   = min(bedrooms + np.random.choice([0, 1]), 5)
        area_sqft   = int(np.random.normal(loc=bedrooms * 400 + 300, scale=200))
        area_sqft   = max(300, min(area_sqft, 5000))

        age_years   = np.random.randint(0, 30)
        floor_num   = np.random.randint(0, 25)
        total_floors= max(floor_num + np.random.randint(1, 10), floor_num + 1)

        n_amenities = np.random.randint(2, len(AMENITIES) + 1)
        amenity_list= np.random.choice(AMENITIES, size=n_amenities, replace=False).tolist()
        amenities_str = "|".join(amenity_list)

        # Location score (0-10): premium areas score higher
        premium_localities = ["Bandra", "Juhu", "Koregaon Park", "Koramangala",
                              "Banjara Hills", "Jubilee Hills", "Adyar", "Vasant Kunj",
                              "Alipore", "C-Scheme", "Vesu", "Satellite"]
        location_score = (
            round(np.random.uniform(7.0, 10.0), 2)
            if locality in premium_localities
            else round(np.random.uniform(4.0, 7.5), 2)
        )

        # Base price per sqft (INR) by city
        city_base = {
            "Mumbai": 25000, "Delhi": 18000, "Bangalore": 12000,
            "Pune": 9000,    "Hyderabad": 8500, "Chennai": 9500,
            "Kolkata": 7000, "Ahmedabad": 6500, "Jaipur": 6000, "Surat": 5500,
        }
        base_ppsf = city_base[city]

        # Modifiers
        type_factor = {"Penthouse": 1.8, "Villa": 1.5, "Independent House": 1.2,
                       "Apartment": 1.0, "Studio": 0.85}
        age_discount = max(0, 1 - age_years * 0.01)
        amenity_bonus = 1 + len(amenity_list) * 0.02
        location_bonus = 1 + (location_score - 5) * 0.05
        noise = np.random.normal(1.0, 0.08)

        price_per_sqft = int(
            base_ppsf * type_factor[prop_type] * age_discount
            * amenity_bonus * location_bonus * noise
        )
        price = int(price_per_sqft * area_sqft)

        # Introduce ~3% missing values
        if np.random.rand() < 0.03:
            age_years = np.nan
        if np.random.rand() < 0.02:
            floor_num = np.nan

        records.append({
            "property_id":   f"PROP{_ + 1:05d}",
            "city":          city,
            "locality":      locality,
            "property_type": prop_type,
            "bedrooms":      bedrooms,
            "bathrooms":     bathrooms,
            "area_sqft":     area_sqft,
            "age_years":     age_years,
            "floor_number":  floor_num,
            "total_floors":  total_floors,
            "amenities":     amenities_str,
            "location_score": location_score,
            "price_per_sqft": price_per_sqft,
            "price":         price,
        })

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────
# Load / Save helpers
# ─────────────────────────────────────────────
def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load dataset from CSV or generate synthetic data if file not found.

    Parameters
    ----------
    filepath : str, optional
        Path to CSV file. Defaults to data/properties.csv.

    Returns
    -------
    pd.DataFrame
    """
    fp = Path(filepath) if filepath else DATASET_FILE
    if fp.exists():
        print(f"[DataLoader] Loading from {fp}")
        return pd.read_csv(fp)
    print("[DataLoader] File not found — generating synthetic dataset …")
    df = generate_synthetic_data()
    save_data(df, fp)
    return df


def save_data(df: pd.DataFrame, filepath: str = None) -> None:
    """Persist DataFrame to CSV."""
    fp = Path(filepath) if filepath else DATASET_FILE
    fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fp, index=False)
    print(f"[DataLoader] Saved {len(df)} records to {fp}")


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.head(3))
    print(df.dtypes)
