"""
eda.py
------
Exploratory Data Analysis Module.
Generates and saves statistical plots for the housing dataset.
"""

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
FIGSIZE = (12, 6)


# ─────────────────────────────────────────────
class EDAAnalyzer:
    """Run a full suite of EDA plots and summary statistics."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._add_amenities_count()

    def _add_amenities_count(self):
        if "amenities" in self.df.columns and "amenities_count" not in self.df.columns:
            self.df["amenities_count"] = (
                self.df["amenities"].fillna("").apply(lambda x: len(x.split("|")) if x else 0)
            )

    # ------------------------------------------------------------------
    def summary_stats(self) -> pd.DataFrame:
        """Print and return descriptive statistics."""
        stats = self.df.describe(include="all")
        print("=== Summary Statistics ===")
        print(stats)
        return stats

    # ------------------------------------------------------------------
    def plot_price_distribution(self):
        """Plot price histogram and log-price distribution."""
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

        axes[0].hist(self.df["price"] / 1e6, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[0].set_title("Price Distribution (₹ Millions)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Price (₹ M)")
        axes[0].set_ylabel("Count")

        log_prices = np.log1p(self.df["price"])
        axes[1].hist(log_prices, bins=60, color="#55A868", edgecolor="white", alpha=0.85)
        axes[1].set_title("Log-Price Distribution", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("log(Price + 1)")
        axes[1].set_ylabel("Count")

        fig.suptitle("Property Price Distributions", fontsize=15, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "01_price_distribution.png")

    # ------------------------------------------------------------------
    def plot_price_by_city(self):
        """Box plot of price by city."""
        fig, ax = plt.subplots(figsize=(14, 6))
        order = (
            self.df.groupby("city")["price"].median()
            .sort_values(ascending=False).index
        )
        sns.boxplot(data=self.df, x="city", y="price", order=order,
                    palette="Set2", ax=ax, showfliers=False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
        ax.set_title("Property Price by City (Median Ordered)", fontsize=13, fontweight="bold")
        ax.set_xlabel("City")
        ax.set_ylabel("Price")
        plt.xticks(rotation=30)
        plt.tight_layout()
        self._save(fig, "02_price_by_city.png")

    # ------------------------------------------------------------------
    def plot_price_by_property_type(self):
        """Bar chart of average price per property type."""
        fig, ax = plt.subplots(figsize=(10, 5))
        avg = (
            self.df.groupby("property_type")["price"]
            .mean()
            .sort_values(ascending=False)
            / 1e6
        )
        avg.plot(kind="bar", color="#DD8452", edgecolor="black", ax=ax)
        ax.set_title("Avg Price by Property Type (₹ Millions)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Avg Price (₹ M)")
        ax.set_xlabel("")
        plt.xticks(rotation=20)
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"₹{bar.get_height():.1f}M",
                ha="center", va="bottom", fontsize=9
            )
        plt.tight_layout()
        self._save(fig, "03_price_by_type.png")

    # ------------------------------------------------------------------
    def plot_correlation_heatmap(self):
        """Heatmap of numeric feature correlations."""
        numeric_df = self.df.select_dtypes(include=[np.number]).drop(
            columns=["price_per_sqft"], errors="ignore"
        )
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(12, 9))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.5,
            annot_kws={"size": 9}, ax=ax
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        self._save(fig, "04_correlation_heatmap.png")

    # ------------------------------------------------------------------
    def plot_area_vs_price(self):
        """Scatter: area vs price coloured by bedrooms."""
        fig, ax = plt.subplots(figsize=FIGSIZE)
        scatter = ax.scatter(
            self.df["area_sqft"],
            self.df["price"] / 1e6,
            c=self.df["bedrooms"],
            cmap="viridis", alpha=0.4, s=15
        )
        plt.colorbar(scatter, ax=ax, label="Bedrooms")
        ax.set_title("Area vs Price (coloured by Bedrooms)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Area (sqft)")
        ax.set_ylabel("Price (₹ M)")
        plt.tight_layout()
        self._save(fig, "05_area_vs_price.png")

    # ------------------------------------------------------------------
    def plot_bedrooms_distribution(self):
        """Count plot of bedroom categories."""
        fig, ax = plt.subplots(figsize=(8, 5))
        order = sorted(self.df["bedrooms"].unique())
        sns.countplot(data=self.df, x="bedrooms", order=order,
                      palette="pastel", edgecolor="black", ax=ax)
        ax.set_title("Bedroom Count Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Bedrooms")
        ax.set_ylabel("Count")
        plt.tight_layout()
        self._save(fig, "06_bedrooms_distribution.png")

    # ------------------------------------------------------------------
    def plot_location_score_vs_price(self):
        """Scatter: location score vs price."""
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.scatter(
            self.df["location_score"],
            self.df["price"] / 1e6,
            alpha=0.35, s=12, color="#C44E52"
        )
        # regression line
        m, b = np.polyfit(self.df["location_score"], self.df["price"] / 1e6, 1)
        xs = np.linspace(self.df["location_score"].min(), self.df["location_score"].max(), 100)
        ax.plot(xs, m * xs + b, "k--", linewidth=1.5, label="Trend")
        ax.set_title("Location Score vs Price", fontsize=13, fontweight="bold")
        ax.set_xlabel("Location Score (0-10)")
        ax.set_ylabel("Price (₹ M)")
        ax.legend()
        plt.tight_layout()
        self._save(fig, "07_location_score_vs_price.png")

    # ------------------------------------------------------------------
    def run_all(self):
        """Execute all EDA plots sequentially."""
        print("[EDA] Running full analysis suite …")
        self.summary_stats()
        self.plot_price_distribution()
        self.plot_price_by_city()
        self.plot_price_by_property_type()
        self.plot_correlation_heatmap()
        self.plot_area_vs_price()
        self.plot_bedrooms_distribution()
        self.plot_location_score_vs_price()
        print(f"[EDA] All plots saved to {OUTPUTS_DIR}")

    # ------------------------------------------------------------------
    @staticmethod
    def _save(fig: plt.Figure, filename: str):
        fp = OUTPUTS_DIR / filename
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fp.name}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data_collection.data_loader import load_data

    df = load_data()
    eda = EDAAnalyzer(df)
    eda.run_all()
