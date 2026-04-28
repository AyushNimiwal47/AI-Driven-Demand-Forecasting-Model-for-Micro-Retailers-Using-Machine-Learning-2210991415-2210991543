"""
Forecasting models:
1. ARIMA      - statistical baseline (auto-ordered via AIC)
2. RandomForest - 300 trees, max depth 15
3. LSTM       - 2 stacked layers (64, 32 units), Adam optimiser

Each class exposes fit() / predict() with consistent signatures so main.py
can compare them directly.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class ARIMAModel:
    """Univariate ARIMA model with automatic order selection via AIC."""

    def __init__(self, max_p=5, max_d=2, max_q=5):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.model = None
        self.best_order = None
        self.history = None

    def fit(self, train_series):
        from statsmodels.tsa.arima.model import ARIMA

        self.history = list(train_series)
        best_aic = np.inf
        best_order = (1, 1, 1)

        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    try:
                        m = ARIMA(self.history, order=(p, d, q))
                        res = m.fit()
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue

        self.best_order = best_order
        self.model = ARIMA(self.history, order=best_order).fit()
        return self

    def predict(self, steps):
        forecast = self.model.forecast(steps=steps)
        return np.asarray(forecast).clip(min=0)


class RandomForestModel:
    """Random Forest regressor over engineered features."""

    def __init__(self, n_estimators=300, max_depth=15, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        preds = self.model.predict(X_test)
        return np.asarray(preds).clip(min=0)


class LSTMModel:
    """
    Two stacked LSTM layers (64 -> 32 units) followed by a Dense layer.
    Trained with Adam optimiser at lr=0.001, batch size 32, with early
    stopping on validation loss.
    """

    def __init__(self, sequence_length=14, epochs=30, batch_size=32, lr=0.001):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None

    def _build(self, n_features):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            LSTM(64, return_sequences=True,
                 input_shape=(self.sequence_length, n_features)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation="linear"),
        ])
        model.compile(optimizer=Adam(learning_rate=self.lr), loss="mse")
        return model

    @staticmethod
    def to_sequences(values, seq_len):
        """Slide a window of length `seq_len` over `values`."""
        values = np.asarray(values, dtype=float)
        X, y = [], []
        for i in range(len(values) - seq_len):
            X.append(values[i:i + seq_len])
            y.append(values[i + seq_len])
        return np.asarray(X), np.asarray(y)

    def fit(self, train_series, val_series=None):
        from tensorflow.keras.callbacks import EarlyStopping

        X_train, y_train = self.to_sequences(train_series, self.sequence_length)
        if X_train.ndim == 2:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        validation_data = None
        if val_series is not None and len(val_series) > self.sequence_length:
            X_val, y_val = self.to_sequences(val_series, self.sequence_length)
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
            validation_data = (X_val, y_val)

        self.model = self._build(n_features=1)
        cb = [EarlyStopping(monitor="val_loss" if validation_data else "loss",
                            patience=8, restore_best_weights=True)]
        self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=cb,
            verbose=0,
        )
        return self

    def predict(self, history, steps):
        """Roll forward `steps` predictions starting from the end of history."""
        history = list(np.asarray(history, dtype=float))
        preds = []
        for _ in range(steps):
            window = np.asarray(history[-self.sequence_length:],
                                dtype=float).reshape(1, self.sequence_length, 1)
            yhat = float(self.model.predict(window, verbose=0)[0, 0])
            yhat = max(0.0, yhat)
            preds.append(yhat)
            history.append(yhat)
        return np.asarray(preds)
