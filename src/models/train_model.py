"""
train_model.py
--------------
Machine Learning Models Module.
Trains Linear Regression, Random Forest, and XGBoost models,
evaluates them, and persists the best model to disk.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

from sklearn.linear_model   import LinearRegression, Ridge
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics        import mean_squared_error, r2_score, mean_absolute_error

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[Warning] XGBoost not installed — will use GradientBoosting as fallback.")

MODELS_DIR  = Path(__file__).resolve().parents[2] / "models"
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)


# ─────────────────────────────────────────────
class ModelTrainer:
    """
    Trains multiple regression models and selects the best by R².

    Usage
    -----
    trainer = ModelTrainer()
    results = trainer.train(X_train, X_test, y_train, y_test)
    trainer.save_best_model()
    """

    def __init__(self):
        self.models: dict = {}
        self.results: dict = {}
        self.best_model_name: str = ""
        self.best_model = None
        self._build_model_registry()

    # ------------------------------------------------------------------
    def _build_model_registry(self):
        """Define all candidate models with tuned hyper-parameters."""
        self.model_registry = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression":  Ridge(alpha=10.0),
            "Random Forest":     RandomForestRegressor(
                n_estimators=200, max_depth=15, min_samples_leaf=4,
                n_jobs=-1, random_state=42
            ),
        }
        if XGBOOST_AVAILABLE:
            self.model_registry["XGBoost"] = XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            )
        else:
            self.model_registry["GradientBoosting"] = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42
            )

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray, X_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
    ) -> dict:
        """
        Train all models, evaluate on test set, store results.

        Returns
        -------
        dict  {model_name: {r2, rmse, mae, mape}}
        """
        print("\n[ModelTrainer] Training models …")
        for name, model in self.model_registry.items():
            print(f"  ▸ {name} …", end=" ", flush=True)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = {
                "r2":   round(r2_score(y_test, preds), 4),
                "rmse": round(rmse(y_test, preds), 2),
                "mae":  round(mean_absolute_error(y_test, preds), 2),
                "mape": round(mape(y_test, preds), 2),
            }
            self.models[name]  = model
            self.results[name] = metrics
            print(f"R²={metrics['r2']:.4f}  RMSE=₹{metrics['rmse']:,.0f}  MAPE={metrics['mape']:.1f}%")

        # Best model by R²
        self.best_model_name = max(self.results, key=lambda n: self.results[n]["r2"])
        self.best_model      = self.models[self.best_model_name]
        print(f"\n[ModelTrainer] ✓ Best model: {self.best_model_name}  "
              f"(R²={self.results[self.best_model_name]['r2']:.4f})")

        self._save_results()
        return self.results

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """Run inference with a specific or best model."""
        model = self.models.get(model_name, self.best_model)
        if model is None:
            raise RuntimeError("No trained model found. Call train() first.")
        return model.predict(X)

    # ------------------------------------------------------------------
    def save_best_model(self):
        """Persist best model to models/best_model.pkl."""
        if self.best_model is None:
            raise RuntimeError("No model to save. Run train() first.")
        path = MODELS_DIR / "best_model.pkl"
        with open(path, "wb") as f:
            pickle.dump({"name": self.best_model_name, "model": self.best_model}, f)
        print(f"[ModelTrainer] Best model saved → {path}")

    def save_all_models(self):
        """Persist every trained model."""
        for name, model in self.models.items():
            safe_name = name.lower().replace(" ", "_")
            path = MODELS_DIR / f"{safe_name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
        print(f"[ModelTrainer] All {len(self.models)} models saved to {MODELS_DIR}")

    @staticmethod
    def load_best_model():
        """Load persisted best model."""
        path = MODELS_DIR / "best_model.pkl"
        with open(path, "rb") as f:
            payload = pickle.load(f)
        print(f"[ModelTrainer] Loaded model: {payload['name']}")
        return payload["model"], payload["name"]

    # ------------------------------------------------------------------
    def _save_results(self):
        """Write metrics JSON to outputs/."""
        path = OUTPUTS_DIR / "model_comparison.json"
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[ModelTrainer] Metrics saved → {path}")

    # ------------------------------------------------------------------
    def print_comparison_table(self):
        """Pretty-print model comparison."""
        print("\n" + "=" * 68)
        print(f"{'Model':<25} {'R²':>8} {'RMSE (₹)':>14} {'MAE (₹)':>12} {'MAPE%':>7}")
        print("-" * 68)
        for name, m in self.results.items():
            marker = "★" if name == self.best_model_name else " "
            print(f"{marker} {name:<23} {m['r2']:>8.4f} {m['rmse']:>14,.0f} "
                  f"{m['mae']:>12,.0f} {m['mape']:>6.1f}%")
        print("=" * 68)

    # ------------------------------------------------------------------
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Return feature importances for tree-based best model.

        Returns
        -------
        pd.DataFrame sorted by importance descending.
        """
        model = self.best_model
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_)
        else:
            return pd.DataFrame()

        df = pd.DataFrame({
            "feature":    feature_names[:len(imp)],
            "importance": imp,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        path = OUTPUTS_DIR / "feature_importances.csv"
        df.to_csv(path, index=False)
        print(f"[ModelTrainer] Feature importances saved → {path}")
        return df


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data_collection.data_loader  import load_data
    from src.preprocessing.preprocess     import DataPreprocessor

    df   = load_data()
    prep = DataPreprocessor()
    X_train, X_test, y_train, y_test, feats = prep.fit_transform(df)

    trainer = ModelTrainer()
    trainer.train(X_train, X_test, y_train, y_test)
    trainer.print_comparison_table()
    trainer.save_best_model()
    trainer.save_all_models()

    fi = trainer.get_feature_importance(feats)
    print(fi.head(10))
