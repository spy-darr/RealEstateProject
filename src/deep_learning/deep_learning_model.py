"""
deep_learning_model.py
----------------------
Deep Learning Module.
Builds, trains, and evaluates a TensorFlow/Keras neural network
for property price regression.
"""

import numpy as np
import json
from pathlib import Path

MODELS_DIR  = Path(__file__).resolve().parents[2] / "models"
OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[Warning] TensorFlow not installed — DL module will use sklearn MLPRegressor.")

if not TF_AVAILABLE:
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ─────────────────────────────────────────────
class DeepLearningModel:
    """
    Neural network for property price prediction.

    Architecture (TF/Keras)
    -----------------------
    Input → Dense(256, relu) → Dropout(0.3)
          → Dense(128, relu) → Dropout(0.2)
          → Dense(64,  relu)
          → Dense(1)   (linear output = price)

    Falls back to sklearn MLPRegressor when TF is unavailable.
    """

    def __init__(self, input_dim: int = None):
        self.input_dim   = input_dim
        self.model       = None
        self.history     = None
        self.use_tf      = TF_AVAILABLE
        self._is_trained = False

    # ------------------------------------------------------------------
    def build(self, input_dim: int = None):
        """Construct the neural network graph."""
        if input_dim:
            self.input_dim = input_dim

        if self.use_tf:
            self.model = self._build_keras_model(self.input_dim)
            print("[DL] Keras model built.")
            self.model.summary()
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
                verbose=False,
            )
            print("[DL] MLPRegressor (sklearn) model built.")

    # ------------------------------------------------------------------
    def _build_keras_model(self, input_dim: int) -> "keras.Model":
        inputs = keras.Input(shape=(input_dim,), name="features")

        x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.30)(x)

        x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.20)(x)

        x = layers.Dense(64)(x)
        x = layers.Activation("relu")(x)

        output = layers.Dense(1, name="price")(x)

        model = keras.Model(inputs=inputs, outputs=output, name="PropertyPricer")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="huber",          # robust to outliers
            metrics=["mae"],
        )
        return model

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray = None,  y_val: np.ndarray = None,
        epochs: int = 80, batch_size: int = 64,
    ) -> dict:
        """
        Train the neural network.

        Returns
        -------
        dict  training metrics / history
        """
        if self.model is None:
            self.build(X_train.shape[1])

        print(f"[DL] Training on {X_train.shape[0]} samples …")

        if self.use_tf:
            cb_list = [
                callbacks.EarlyStopping(
                    monitor="val_loss", patience=12,
                    restore_best_weights=True, verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5,
                    patience=6, min_lr=1e-6, verbose=0
                ),
            ]
            validation = (X_val, y_val) if X_val is not None else None
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=validation,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=cb_list,
                verbose=1,
            )
            hist_dict = {
                "loss":     self.history.history.get("loss", []),
                "val_loss": self.history.history.get("val_loss", []),
                "mae":      self.history.history.get("mae", []),
            }
        else:
            self.model.fit(X_train, y_train)
            hist_dict = {"info": "sklearn MLPRegressor trained"}

        self._is_trained = True
        self._save_history(hist_dict)
        return hist_dict

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict property prices."""
        if not self._is_trained:
            raise RuntimeError("Model not trained yet.")
        preds = self.model.predict(X)
        return preds.flatten() if self.use_tf else preds

    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Compute R², RMSE, MAE on test set."""
        preds = self.predict(X_test)

        if not TF_AVAILABLE:
            r2  = float(self.model.score(X_test, y_test))
        else:
            from sklearn.metrics import r2_score
            r2 = float(r2_score(y_test, preds))

        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse_ = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae_  = float(mean_absolute_error(y_test, preds))

        metrics = {"r2": round(r2, 4), "rmse": round(rmse_, 2), "mae": round(mae_, 2)}
        print(f"[DL] Evaluation → R²={metrics['r2']:.4f}  "
              f"RMSE=₹{metrics['rmse']:,.0f}  MAE=₹{metrics['mae']:,.0f}")

        path = OUTPUTS_DIR / "dl_metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    # ------------------------------------------------------------------
    def save(self):
        """Persist model weights."""
        if self.use_tf:
            path = str(MODELS_DIR / "deep_learning_model.h5")
            self.model.save(path)
            print(f"[DL] Keras model saved → {path}")
        else:
            import pickle
            path = MODELS_DIR / "dl_mlp_model.pkl"
            with open(path, "wb") as f:
                pickle.dump(self.model, f)
            print(f"[DL] MLPRegressor saved → {path}")

    def load(self):
        """Load persisted model."""
        if self.use_tf:
            path = str(MODELS_DIR / "deep_learning_model.h5")
            self.model = keras.models.load_model(path)
        else:
            import pickle
            path = MODELS_DIR / "dl_mlp_model.pkl"
            with open(path, "rb") as f:
                self.model = pickle.load(f)
        self._is_trained = True
        print("[DL] Model loaded.")

    # ------------------------------------------------------------------
    def _save_history(self, hist: dict):
        path = OUTPUTS_DIR / "dl_training_history.json"
        # Convert numpy floats to native float for JSON serialisation
        clean = {k: [float(v) for v in vals] if isinstance(vals, list) else vals
                 for k, vals in hist.items()}
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data_collection.data_loader  import load_data
    from src.preprocessing.preprocess     import DataPreprocessor

    df   = load_data()
    prep = DataPreprocessor()
    X_train, X_test, y_train, y_test, _ = prep.fit_transform(df)

    dl = DeepLearningModel()
    dl.build(X_train.shape[1])
    dl.train(X_train, y_train, X_test, y_test, epochs=60)
    dl.evaluate(X_test, y_test)
    dl.save()
