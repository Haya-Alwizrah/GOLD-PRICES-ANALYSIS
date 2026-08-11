import numpy as np
import pandas as pd
from BaseGoldPredictor import BaseGoldPredictor

class RecursiveGoldPredictor(BaseGoldPredictor):
    def __init__(self):
        super().__init__("Recursive")

    def forecast(self, latest_data, historical_data, scaler_X, days=30):

        # 1. Prepare historical Gold prices
        price_history = historical_data["Gold_Close"].tolist()
        current = latest_data.copy()

        # 2. Recursive forecasting
        last_date = pd.to_datetime(latest_data.name)

        forecast_dates = []
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
            X_current_scaled = scaler_X.transform(X_current)

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

        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted_Close": predictions
        })

        # 4. Find lowest and highest predicted prices
        min_row = forecast_df.loc[forecast_df["Predicted_Close"].idxmin()]
        max_row = forecast_df.loc[forecast_df["Predicted_Close"].idxmax()]

        lowest_24k = min_row["Predicted_Close"] / 31.1034768
        highest_24k = max_row["Predicted_Close"] / 31.1034768

        # 5. Print results
        print(f"{days}-Day Gold Price Forecast")
        print(f"Lowest predicted 24K price : {lowest_24k:.2f}")
        print(f"Lowest price date          : {min_row['Date'].date()}")

        print(f"Highest predicted 24K price: {highest_24k:.2f}")
        print(f"Highest price date         : {max_row['Date'].date()}")

        return forecast_df