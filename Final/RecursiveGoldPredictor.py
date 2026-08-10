import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

class RecursiveGoldPredictor:
    def __init__(self):
        self.model = None
        self.history = None
        self.features = None
        self.history_train = None
        self.y_test = None
        self.y_pred = None
        self.price_history = None
        self.scaler = None

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
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mae",
            metrics=["mae", "mse"]
        )

        return model
    
    def tune(self, X_train, y_train, X_val, y_val, configs, features=None, epochs=80, batch_size=32, patience=10):

        self.features = features
        self.tuning_results = []

        for i, config in enumerate(configs):
            tf.keras.backend.clear_session()
            model = self._build_model(input_shape=X_train.shape[1],**config)

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True
            )

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )

            best_val_mae = min(history.history["val_mae"])
            self.tuning_results.append({
                "config": config,
                "val_mae": best_val_mae
            })

            print(f"Config {i}: {config} -> val_mae={best_val_mae:.4f}")

        # Select best configuration
        self.best_config = min(self.tuning_results, key=lambda x: x["val_mae"])["config"]
        print("\nBest config:")
        print(self.best_config)

        return self.best_config

    def train(self, X_train, y_train, X_val, y_val, configs, features=None, best_config=None, epochs=300, batch_size=32, patience=20):

        self.features = features
        if best_config is None:
            if self.best_config is None:
                self.tune(X_train, y_train, X_val, y_val, configs, epochs, batch_size, patience)

            best_config = self.best_config

        self.best_config = best_config
        tf.keras.backend.clear_session()

        self.model = self._build_model(input_shape=X_train.shape[1], **best_config)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True
            ),

            tf.keras.callbacks.ModelCheckpoint(
                "best_recursive_gold_model.keras",
                monitor="val_loss",
                save_best_only=True
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=8,
                min_lr=1e-6
            )
        ]

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        self.history_train = history.history
        return history

    def evaluate(self, X_test, y_test, scaler_y=None):
        y_pred = self.model.predict(X_test, verbose=0).flatten()

        if scaler_y is not None:
            y_test_original = scaler_y.inverse_transform(np.asarray(y_test).reshape(-1, 1)).ravel()
            y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        else:
            y_test_original = np.asarray(y_test)
            y_pred_original = y_pred
    
        self.y_test = y_test_original
        self.y_pred = y_pred_original

        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)

        print("\nRecursive Model Results")
        print(f"RMSE : {rmse:.3f}")
        print(f"MAE  : {mae:.3f}")
        print(f"R²   : {r2:.4f}")

        plt.figure(figsize=(14, 6))
        plt.plot(self.history_train["loss"], label="Training Loss")
        plt.plot(self.history_train["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("ANN Training vs Validation Loss")
        plt.legend()
        plt.show()
        
        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    def save(self, model_path):
        self.model.save(model_path)

    def load(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def forecast(self, latest_data, historical_data, scaler, days=30):

        # 1. Prepare historical Gold prices
        price_history = historical_data["Gold_Close"].tolist()
        current = latest_data.copy()

        # 2. Recursive forecasting
        last_date = pd.to_datetime(latest_data.name)
        predictions = []
        for i in range(days):

            # Date of the day we are predicting
            forecast_date = last_date + pd.offsets.BDay(i + 1)
            forecast_dates.append(forecast_date)

            # Calendar features
            dow = forecast_date.dayofweek
            current["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            current["dow_cos"] = np.cos(2 * np.pi * dow / 7)

            month = forecast_date.month
            current["month_sin"] = np.sin(2 * np.pi * month / 12)
            current["month_cos"] = np.cos(2 * np.pi * month / 12)

            # Prepare current input
            X_current = pd.DataFrame([[current[feature] for feature in self.features]],columns=self.features)
            X_current_scaled = scaler.transform(X_current)

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