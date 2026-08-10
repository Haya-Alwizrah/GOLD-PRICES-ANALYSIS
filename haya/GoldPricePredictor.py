import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class GoldPricePredictor:
    def __init__(self):
        self.model = None
        self.history = None
        self.features = None
        self.history_train = None
        self.y_test = None
        self.y_pred = None
        self.price_history = None
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, lr=0.001, validation_split=0.2, epochs=200, batch_size=32, es_patience=10):

        self.features = X_train.columns.tolist()

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae"]
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor = "val_loss",
            patience = es_patience,
            restore_best_weights = True
        )

        history = self.model.fit(
            X_train,
            y_train,
            validation_split = validation_split,
            epochs = epochs,
            batch_size = batch_size,
            callbacks = [early_stop],
            verbose = 1
        )

        self.history_train = history.history
    

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test, verbose=0).flatten()

        self.y_test = y_test
        self.y_pred = y_pred

        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("RMSE :", rmse)
        print("MAE  :", mae)
        print("R²   :", r2)

        plt.figure(figsize=(14, 6))
        plt.plot(self.history_train["loss"], label="Training Loss")
        plt.plot(self.history_train["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("ANN Training vs Validation Loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(14, 6))
        plt.plot(self.y_test.values, label="Actual")
        plt.plot(self.y_pred, label="Prediction")
        plt.xlabel("Samples")
        plt.ylabel("Gold Price")
        plt.title("Actual vs Predicted Gold Prices")
        plt.legend()
        plt.grid(True)
        plt.show()

    def save(self, model_path):
        self.model.save(model_path)

    def load(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def forecast(self, latest_data, historical_data, days=30):

        # 1. Prepare historical Gold prices
        price_history = historical_data["Gold_Close"].tolist()
        current = latest_data.copy()

        # 2. Recursive forecasting
        predictions = []
        for _ in range(days):

            # Prepare current input
            X_current = pd.DataFrame([[current[feature] for feature in self.features]],columns=self.features)
            X_current_scaled = self.scaler.transform(X_current)

            # Predict next day's Gold Close
            next_price = self.model.predict( X_current_scaled, verbose=0)[0][0]
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
                current["RSI"] = 100 - (100 / (1 + rs))

            # MACD
            prices = pd.Series(price_history)
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
            start=last_date + pd.Timedelta(days=1),
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
        print(f"{days}-Day Gold Price Forecast")
        print(f"Lowest predicted price : {min_row['Predicted_Close']:.2f}")
        print(f"Lowest price date      : {min_row['Date'].date()}")

        print(f"Highest predicted price: {max_row['Predicted_Close']:.2f}")
        print(f"Highest price date     : {max_row['Date'].date()}")

        return forecast_df