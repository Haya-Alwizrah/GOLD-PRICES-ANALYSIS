import pandas as pd
from Final.models.BaseGoldPredictor import BaseGoldPredictor

class DirectGoldPredictor(BaseGoldPredictor):
    TROY_OUNCE_GRAMS = 31.1034768
    def __init__(self):
        super().__init__("Direct")

    def predict_after_21_days(self, latest_data, scaler_X, current_price):
        X_current = pd.DataFrame([[latest_data[feature] for feature in self.features]], columns=self.features)
        X_current_scaled = scaler_X.transform(X_current)

        predicted_price = self.model.predict(X_current_scaled, verbose=0).ravel()[0]

        current_24k = current_price / self.TROY_OUNCE_GRAMS
        predicted_24k =  predicted_price / self.TROY_OUNCE_GRAMS

        pct_change = ((predicted_24k - current_24k) / current_24k) * 100

        karat_forecast = {
            "24K": predicted_24k,
            "21K": predicted_24k * 21 / 24,
            "18K": predicted_24k * 18 / 24
        }

        # Trading signal
        BUY_THRESHOLD = 2.0
        SELL_THRESHOLD = -2.0

        if pct_change > BUY_THRESHOLD:
            signal = (
                "BUY NOW / SELL LATER — "
                "price expected to rise"
            )

        elif pct_change < SELL_THRESHOLD:
            signal = (
                "WAIT TO BUY / SELL NOW — "
                "price expected to fall"
            )

        else:
            signal = (
                "HOLD — "
                "no strong trend expected"
            )

        # Print results
        print("\n21-Day Direct Forecast")

        print(f"Current 24K price/gram: {current_24k:.2f}")
        print(f"Predicted 24K price/gram in ~1 month:")
        for k, v in karat_forecast.items():
            print(f"  {k}: {v:.2f}")

        print(f"\nExpected change: {pct_change:+.2f}%")
        print(f"Signal: {signal}")

        return {
            "predicted_ounce": predicted_price,
            "current_24k": current_24k,
            "predicted_24k": predicted_24k,
            "karat_forecast": karat_forecast,
            "pct_change": pct_change,
            "signal": signal
        }