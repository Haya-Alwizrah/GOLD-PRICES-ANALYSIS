import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers, regularizers, callbacks

class GoldPricePredictor:
    TROY_OUNCE_GRAMS = 31.1034768

    def __init__(self):
        self.direct_model = None
        self.recursive_model = None
        self.features = None
        self.scaler = None

        self.history_direct = None
        self.history_recursive = None

        self.y_test_direct = None
        self.y_pred_direct = None

        self.y_test_recursive = None
        self.y_pred_recursive = None

    def _build_model(self, input_shape, units1=128, units2=64, units3=32, dropout1=0.3, dropout2=0.2, l2reg=1e-3, lr=1e-3):

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(units1, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2reg)),
            tf.keras.layers.Dropout(dropout1),
            tf.keras.layers.Dense(units2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2reg)),
            tf.keras.layers.Dropout(dropout2),
            tf.keras.layers.Dense(units3, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr
            ),
            loss="mae",
            metrics=["mae", "mse"]
        )

        return model

    # TRAIN DIRECT MODEL
    # Target = Gold_Close after 21 days

    def train_direct(self, X_train, y_train, X_val, y_val, features, lr=0.001, validation_split=0.2, epochs=200, batch_size=32, es_patience=10):
        self.features = features

        self.direct_model = self._build_model(input_shape=X_train.shape[1],lr=lr)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=es_patience,
            restore_best_weights=True
        )

        history = self.direct_model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        self.history_direct = history.history

    # TRAIN RECURSIVE MODEL
    # Target = Gold_Close next day

    def train_recursive(self, X_train, y_train, X_val, y_val, features, lr=0.001, validation_split=0.2, epochs=200, batch_size=32, es_patience=10):
        self.features = features

        self.recursive_model = self._build_model(input_shape=X_train.shape[1], lr=lr)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=es_patience,
            restore_best_weights=True
        )

        history = self.recursive_model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        self.history_recursive = history.history

    def evaluate(self, model, X_test, y_test, label="Model"):
        y_pred = model.predict(X_test, verbose=0).flatten()

        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("RMSE :", rmse)
        print("MAE  :", mae)
        print("R²   :", r2)

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }


    # DIRECT PREDICTION
    # Predict price after 21 days

    def predict_after_21_days(self, latest_data):
        if self.direct_model is None:
            raise ValueError("Direct model has not been trained.")

        if self.scaler is None:
            raise ValueError("Scaler has not been set.")

        X_current = pd.DataFrame(
            [[
                latest_data[feature]
                for feature in self.features
            ]],
            columns=self.features
        )

        X_scaled = self.scaler.transform(X_current)
        prediction = self.direct_model.predict(X_scaled, verbose=0).flatten()[0]

        return prediction

    # RECURSIVE FORECAST
    # Predict next day repeatedly for 21 days

    def forecast_21_days(self, latest_data, historical_data, days=21):
        if self.recursive_model is None:
            raise ValueError("Recursive model has not been trained.")

        if self.scaler is None:
            raise ValueError("Scaler has not been set.")

        # 1. Prepare historical Gold prices
        price_history = historical_data["Gold_Close"].tolist()
        current = latest_data.copy()

        # 2. Recursive forecasting
        predictions = []
        for _ in range(days):

            # Prepare current input
            X_current = pd.DataFrame([[current[feature] for feature in self.features]], columns=self.features)
            X_current_scaled = self.scaler.transform(X_current)

            # Predict next day's Gold Close
            next_price = self.recursive_model.predict( X_current_scaled, verbose=0).flatten()[0]
            predictions.append(next_price)
            price_history.append(next_price)

            # Update Gold_Close
            previous_close = current["Gold_Close"]
            current["Gold_Close"] = next_price

            # Lag1 + Lag7
            current["Lag1"] = previous_close
            if len(price_history) >= 8:
                current["Lag7"] = price_history[-8]

            # MA7 + MA30
            current["MA7"] = np.mean(price_history[-7:])
            current["MA30"] = np.mean(price_history[-30:])

            # EMA20
            alpha = 2 / (20 + 1)
            current["EMA20"] = alpha * next_price + (1 - alpha) * current["EMA20"]

            # Daily Return
            current["Daily_Return"] = (next_price - previous_close) / previous_close

            # RSI
            prices = pd.Series(price_history)
            delta = prices.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean().iloc[-1]
            avg_loss = loss.rolling(14).mean().iloc[-1]

            if avg_loss == 0:
                current["RSI"] = 100
            else:
                rs = avg_gain / avg_loss
                current["RSI"] = 100 - (100/(1 + rs))

            # MACD
            ema12 = prices.ewm(span=12, adjust=False).mean()
            ema26 = prices.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()

            current["MACD"] = macd.iloc[-1]
            current["MACD_Signal"] = macd_signal.iloc[-1]
            current["MACD_Hist"] = current["MACD"] - current["MACD_Signal"]

            # Bollinger Bands
            rolling_20 = pd.Series(price_history[-20:])
            bb_middle = rolling_20.mean()
            bb_std = rolling_20.std()

            current["BB_Middle"] = bb_middle
            current["BB_Upper"] = bb_middle + 2 * bb_std
            current["BB_Lower"] = bb_middle - 2 * bb_std

            # DXY / SP500 / Oil / VIX
            current["DXY"] = latest_data["DXY"]
            current["SP500"] = latest_data["SP500"]
            current["Oil"] = latest_data["Oil"]
            current["VIX"] = latest_data["VIX"]

        # 3. Create forecast dates
        last_date = pd.to_datetime(latest_data.name)
        forecast_dates = pd.bdate_range(
            start= last_date + pd.Timedelta(days=1),
            periods=days
        )

        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted_Close": predictions
        })


        # 4. Find lowest and highest predicted prices
        min_row = forecast_df.loc[forecast_df["Predicted_Close"].idxmin()]
        max_row = forecast_df.loc[forecast_df["Predicted_Close"].idxmax()]

        # 5. Print results
        print(f"\n{days}-Day Recursive Forecast")
        print(f"Lowest predicted price :  {min_row['Predicted_Close']:.2f}")
        print(f"Lowest price date      : {min_row['Date'].date()}")
        print(f"Highest predicted price: {max_row['Predicted_Close']:.2f}")
        print(f"Highest price date     : {max_row['Date'].date()}")

        return forecast_df

    def plot_history(self, history, title="Training History"):
        plt.figure(figsize=(10, 5))
        plt.plot(history["loss"], label="Training Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def save(self, direct_path="direct_gold_model.keras", recursive_path="recursive_gold_model.keras"):
        if self.direct_model is not None:
            self.direct_model.save(direct_path)

        if self.recursive_model is not None:
            self.recursive_model.save(recursive_path)

    def load(self, direct_path=None, recursive_path=None):
        if direct_path is not None:
            self.direct_model = tf.keras.models.load_model(direct_path)

        if recursive_path is not None:
            self.recursive_model = tf.keras.models.load_model(recursive_path)
