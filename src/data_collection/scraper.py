"""
scraper.py
----------
Data Collection — Web Scraper (Placeholder / Demo).

In a production system, this module would scrape real estate portals
such as MagicBricks, 99acres, or Housing.com.

For the academic project:
  - The scraper structure is demonstrated with dummy/mock responses.
  - Actual HTTP requests are NOT made to avoid rate-limiting or TOS issues.
  - Replace `_mock_response()` with real requests + BeautifulSoup parsing.

Usage
-----
    from src.data_collection.scraper import PropertyScraper
    scraper = PropertyScraper(city="Pune", pages=3)
    df = scraper.scrape()
"""

import time
import random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Simulated listing templates ───────────────────────────────────────
_TITLES = [
    "{bed} BHK {ptype} in {locality}",
    "Spacious {bed} BHK {ptype} for Sale — {locality}",
    "Premium {bed} BHK {ptype} | {locality}, {city}",
    "Well-maintained {bed} BHK {ptype} near IT Hub",
    "Ready-to-move {bed} BHK {ptype} — {city}",
]

_DESCRIPTIONS = [
    "East-facing unit on {floor}th floor. Fully ventilated. Modular kitchen.",
    "Corner unit with panoramic views. {amenities}. Near metro station.",
    "Vaastu-compliant layout. Close to schools and hospitals. {amenities}.",
    "Gated community. 24×7 security. Club house access. {amenities}.",
]


class PropertyScraper:
    """
    Simulated property listing scraper.

    In production: replace `_fetch_page()` with real HTTP requests
    and `_parse_listing()` with actual HTML parsing logic.

    Parameters
    ----------
    city   : Target city name
    pages  : Number of result pages to scrape
    delay  : Polite delay (seconds) between requests
    """

    SOURCE_URL = "https://www.example-realestate.com/property-for-sale/{city}/"

    def __init__(self, city: str = "Pune", pages: int = 5, delay: float = 1.0):
        self.city   = city
        self.pages  = pages
        self.delay  = delay
        self._records: list = []

    # ------------------------------------------------------------------
    def scrape(self) -> pd.DataFrame:
        """
        Execute the scraping workflow across `self.pages` pages.

        Returns
        -------
        pd.DataFrame of scraped listings (mock data in this demo).
        """
        print(f"[Scraper] Starting scrape for '{self.city}' ({self.pages} pages) …")

        for page in range(1, self.pages + 1):
            print(f"  Page {page}/{self.pages} …", end=" ")
            listings = self._fetch_page(page)
            for listing in listings:
                parsed = self._parse_listing(listing)
                if parsed:
                    self._records.append(parsed)
            print(f"{len(listings)} listings found")
            time.sleep(self.delay * 0.1)  # reduced for demo

        df = pd.DataFrame(self._records)
        print(f"[Scraper] Total: {len(df)} listings scraped.")
        return df

    # ------------------------------------------------------------------
    def _fetch_page(self, page: int) -> list:
        """
        Fetch a single results page.

        Production implementation:
        --------------------------
        import requests
        from bs4 import BeautifulSoup

        url = self.SOURCE_URL.format(city=self.city.lower()) + f"?page={page}"
        headers = {"User-Agent": "Mozilla/5.0 ..."}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.select(".listing-card")  # CSS selector for listing elements
        """
        # ── MOCK: return synthetic listing dicts ─────────────────────
        n = random.randint(8, 15)
        return [self._mock_listing() for _ in range(n)]

    # ------------------------------------------------------------------
    def _parse_listing(self, raw: dict) -> dict | None:
        """
        Parse a raw listing element into a structured dict.

        Production implementation:
        --------------------------
        try:
            title    = raw.select_one(".listing-title").text.strip()
            price    = int(raw.select_one(".price").text.replace(",","").replace("₹",""))
            area     = int(raw.select_one(".area").text.split()[0])
            locality = raw.select_one(".locality").text.strip()
            ...
        except (AttributeError, ValueError):
            return None
        """
        # ── MOCK: raw is already a dict in demo mode ─────────────────
        return raw

    # ------------------------------------------------------------------
    def _mock_listing(self) -> dict:
        """Generate one realistic dummy listing for the target city."""
        from src.data_collection.data_loader import (
            LOCALITIES, PROPERTY_TYPES, AMENITIES, CITY_BASE_PPSF
        )
        # Fallback if city not in map
        city_locs = LOCALITIES.get(self.city, ["Area 1", "Area 2", "Area 3"])
        locality  = random.choice(city_locs)
        ptype     = random.choice(PROPERTY_TYPES)
        bed       = random.choice([1, 2, 3, 4])
        bath      = min(bed + random.choice([0, 1]), 4)
        area      = int(np.random.normal(bed * 400 + 300, 150))
        area      = max(300, min(area, 4000))
        age       = random.randint(0, 20)
        floor     = random.randint(0, 20)
        total_fl  = floor + random.randint(1, 8)
        n_amen    = random.randint(2, 8)
        amenities = "|".join(random.sample(AMENITIES, n_amen))
        loc_score = round(random.uniform(4.0, 9.5), 1)

        base_ppsf  = CITY_BASE_PPSF.get(self.city, 9000)
        noise      = random.uniform(0.88, 1.12)
        ppsf       = int(base_ppsf * noise * (1 + (loc_score - 5) * 0.04))
        price      = ppsf * area

        title = random.choice(_TITLES).format(
            bed=bed, ptype=ptype, locality=locality, city=self.city
        )

        return {
            "source":        "mock_scraper",
            "scraped_at":    datetime.utcnow().isoformat(),
            "title":         title,
            "city":          self.city,
            "locality":      locality,
            "property_type": ptype,
            "bedrooms":      bed,
            "bathrooms":     bath,
            "area_sqft":     area,
            "age_years":     age,
            "floor_number":  floor,
            "total_floors":  total_fl,
            "amenities":     amenities,
            "location_score": loc_score,
            "price_per_sqft": ppsf,
            "price":         price,
            "listing_url":   f"https://example.com/property/{random.randint(10000, 99999)}",
        }

    # ------------------------------------------------------------------
    def save(self, filename: str = None) -> Path:
        """Save scraped data to CSV."""
        if not self._records:
            print("[Scraper] No records to save.")
            return None
        df  = pd.DataFrame(self._records)
        fp  = DATA_DIR / (filename or f"scraped_{self.city.lower()}.csv")
        df.to_csv(fp, index=False)
        print(f"[Scraper] Saved {len(df)} records → {fp}")
        return fp


# ─────────────────────────────────────────────
if __name__ == "__main__":
    scraper = PropertyScraper(city="Bangalore", pages=3, delay=0.5)
    df      = scraper.scrape()
    scraper.save()
    print(df[["city", "locality", "property_type", "bedrooms", "area_sqft", "price"]].head(10))
