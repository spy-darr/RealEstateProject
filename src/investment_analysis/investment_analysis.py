"""
investment_analysis.py
----------------------
Investment Analysis Module.
Computes ROI, rental yield, break-even period, and investment grades
for properties.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Market assumptions (can be overridden per-city) ─────────────────
DEFAULT_CONFIG = {
    "annual_appreciation_pct":  7.0,   # property value appreciation %/yr
    "gross_rental_yield_pct":   3.5,   # % of property value as annual rent
    "maintenance_pct":          1.0,   # % of property value per year
    "vacancy_rate_pct":         8.0,   # expected vacancy %
    "tax_rate_pct":             30.0,  # income tax on rental income
    "mortgage_rate_pct":        8.5,   # home loan interest rate
    "down_payment_pct":         20.0,  # % of purchase price as down payment
    "loan_tenure_years":        20,    # mortgage tenure
}

CITY_YIELD_ADJUSTMENTS = {
    "Mumbai":    -0.5,   # lower yield — high prices
    "Bangalore":  0.5,
    "Hyderabad":  0.8,
    "Pune":       0.3,
    "Chennai":    0.2,
    "Delhi":     -0.3,
    "Kolkata":    0.6,
    "Ahmedabad":  0.9,
    "Jaipur":     1.0,
    "Surat":      1.1,
}


# ─────────────────────────────────────────────
@dataclass
class InvestmentReport:
    property_price:       float
    city:                 str
    # Rental
    gross_annual_rent:    float
    net_annual_rent:      float
    gross_rental_yield:   float
    net_rental_yield:     float
    # ROI
    roi_5yr_pct:          float
    roi_10yr_pct:         float
    projected_value_5yr:  float
    projected_value_10yr: float
    # Mortgage
    loan_amount:          float
    monthly_emi:          float
    break_even_years:     float
    # Grade
    investment_grade:     str
    recommendation:       str

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────
class InvestmentAnalyzer:
    """
    Compute comprehensive investment metrics for a given property.

    Usage
    -----
    analyzer = InvestmentAnalyzer()
    report   = analyzer.analyze(price=8_500_000, city="Pune")
    print(report)
    """

    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    # ------------------------------------------------------------------
    def analyze(
        self,
        price: float,
        city: str = "Pune",
        area_sqft: float = None,
        bedrooms: int = None,
    ) -> InvestmentReport:
        """
        Full investment analysis for a property.

        Parameters
        ----------
        price     : property purchase price (INR)
        city      : city name (adjusts yield assumptions)
        area_sqft : optional — used for context
        bedrooms  : optional — used for context

        Returns
        -------
        InvestmentReport dataclass
        """
        cfg = self.config.copy()

        # City-adjusted rental yield
        yield_adj = CITY_YIELD_ADJUSTMENTS.get(city, 0.0)
        gross_yield_pct = cfg["gross_rental_yield_pct"] + yield_adj
        gross_yield_pct = max(1.0, gross_yield_pct)

        # ── Rental calculations ─────────────────────────────────────
        gross_annual_rent = price * gross_yield_pct / 100
        effective_rent    = gross_annual_rent * (1 - cfg["vacancy_rate_pct"] / 100)
        maintenance_cost  = price * cfg["maintenance_pct"] / 100
        tax_on_rent       = effective_rent * cfg["tax_rate_pct"] / 100
        net_annual_rent   = effective_rent - maintenance_cost - tax_on_rent

        gross_rental_yield = round(gross_annual_rent / price * 100, 2)
        net_rental_yield   = round(net_annual_rent   / price * 100, 2)

        # ── Appreciation & ROI ───────────────────────────────────────
        appr = cfg["annual_appreciation_pct"] / 100
        projected_5yr  = price * (1 + appr) ** 5
        projected_10yr = price * (1 + appr) ** 10

        total_rental_5yr  = net_annual_rent * 5
        total_rental_10yr = net_annual_rent * 10

        roi_5yr  = ((projected_5yr  - price + total_rental_5yr)  / price * 100)
        roi_10yr = ((projected_10yr - price + total_rental_10yr) / price * 100)

        # ── Mortgage / EMI ───────────────────────────────────────────
        down_payment  = price * cfg["down_payment_pct"] / 100
        loan_amount   = price - down_payment
        monthly_rate  = cfg["mortgage_rate_pct"] / 100 / 12
        n_months      = cfg["loan_tenure_years"] * 12

        if monthly_rate > 0:
            monthly_emi = loan_amount * monthly_rate * (1 + monthly_rate) ** n_months / \
                          ((1 + monthly_rate) ** n_months - 1)
        else:
            monthly_emi = loan_amount / n_months

        annual_emi_outflow = monthly_emi * 12
        annual_cash_flow   = net_annual_rent - annual_emi_outflow

        # Break-even: years until cumulative cash-flow + appreciation covers down payment
        # Simplified: breakeven when cumulative net rent covers down payment
        if net_annual_rent > 0:
            break_even = down_payment / net_annual_rent
        else:
            break_even = 999.9

        # ── Investment Grade ─────────────────────────────────────────
        grade, recommendation = self._grade(net_rental_yield, roi_10yr, break_even)

        return InvestmentReport(
            property_price       = round(price, 2),
            city                 = city,
            gross_annual_rent    = round(gross_annual_rent, 2),
            net_annual_rent      = round(net_annual_rent, 2),
            gross_rental_yield   = gross_rental_yield,
            net_rental_yield     = net_rental_yield,
            roi_5yr_pct          = round(roi_5yr, 2),
            roi_10yr_pct         = round(roi_10yr, 2),
            projected_value_5yr  = round(projected_5yr, 2),
            projected_value_10yr = round(projected_10yr, 2),
            loan_amount          = round(loan_amount, 2),
            monthly_emi          = round(monthly_emi, 2),
            break_even_years     = round(break_even, 1),
            investment_grade     = grade,
            recommendation       = recommendation,
        )

    # ------------------------------------------------------------------
    def bulk_analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse an entire catalogue and append investment columns.

        Returns
        -------
        DataFrame with new investment metric columns.
        """
        reports = []
        for _, row in df.iterrows():
            r = self.analyze(
                price     = row["price"],
                city      = row.get("city", "Pune"),
                area_sqft = row.get("area_sqft"),
                bedrooms  = row.get("bedrooms"),
            )
            reports.append(r.to_dict())

        report_df = pd.DataFrame(reports)
        result = pd.concat(
            [df.reset_index(drop=True), report_df.drop(columns=["property_price", "city"])],
            axis=1
        )

        path = OUTPUTS_DIR / "investment_analysis.csv"
        result.to_csv(path, index=False)
        print(f"[InvestmentAnalyzer] Bulk analysis saved → {path}")
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _grade(net_yield: float, roi_10yr: float, breakeven: float):
        """Assign A/B/C/D investment grade with a recommendation."""
        score = 0
        if net_yield >= 3.0:   score += 3
        elif net_yield >= 2.0: score += 2
        elif net_yield >= 1.0: score += 1

        if roi_10yr >= 200:   score += 3
        elif roi_10yr >= 150: score += 2
        elif roi_10yr >= 100: score += 1

        if breakeven <= 10:  score += 2
        elif breakeven <= 18: score += 1

        if score >= 7:
            return "A+", "Excellent investment — strong rental yield and appreciation potential."
        elif score >= 5:
            return "A",  "Good investment — solid returns with moderate risk."
        elif score >= 3:
            return "B",  "Moderate investment — acceptable returns; consider location growth."
        else:
            return "C",  "Below-average investment — rental yield is low; evaluate carefully."

    # ------------------------------------------------------------------
    def print_report(self, report: InvestmentReport):
        """Pretty-print an investment report."""
        d = report.to_dict()
        print("\n" + "═" * 55)
        print(f"  INVESTMENT REPORT — {d['city']}")
        print("═" * 55)
        print(f"  Property Price     : ₹{d['property_price']:>15,.0f}")
        print(f"  Gross Annual Rent  : ₹{d['gross_annual_rent']:>15,.0f}  ({d['gross_rental_yield']}%)")
        print(f"  Net Annual Rent    : ₹{d['net_annual_rent']:>15,.0f}  ({d['net_rental_yield']}%)")
        print(f"  Projected (5yr)    : ₹{d['projected_value_5yr']:>15,.0f}")
        print(f"  Projected (10yr)   : ₹{d['projected_value_10yr']:>15,.0f}")
        print(f"  ROI 5yr            : {d['roi_5yr_pct']:>8.1f}%")
        print(f"  ROI 10yr           : {d['roi_10yr_pct']:>8.1f}%")
        print(f"  Loan Amount        : ₹{d['loan_amount']:>15,.0f}")
        print(f"  Monthly EMI        : ₹{d['monthly_emi']:>15,.0f}")
        print(f"  Break-even         : {d['break_even_years']:>8.1f} years")
        print(f"  Grade              : {d['investment_grade']}")
        print(f"  Recommendation     : {d['recommendation']}")
        print("═" * 55)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = InvestmentAnalyzer()
    r = analyzer.analyze(price=8_500_000, city="Pune", area_sqft=1200, bedrooms=3)
    analyzer.print_report(r)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data_collection.data_loader import load_data
    df = load_data()
    analyzer.bulk_analyze(df.head(100))
