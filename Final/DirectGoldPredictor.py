import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class DirectGoldPredictor:
    TROY_OUNCE_GRAMS = 31.1034768
    def __init__(self):
        self.model = None
        self.history = None
        self.features = None
        self.y_test = None
        self.y_pred = None
        self.tuning_results = None
        self.best_config = None

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
                "best_direct_gold_model.keras",
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

        self.history = history.history
        return history

    def evaluate(self, X_test, y_test, scaler_y=None):
        if self.model is None:
            raise ValueError("Model has not been trained.")

        y_pred = self.model.predict(X_test, verbose=0).flatten()

        if scaler_y is not None:
            y_test_original = scaler_y.inverse_transform(np.array(y_test).reshape(-1, 1)).ravel()
            y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        else:
            y_test_original = np.asarray(y_test)
            y_pred_original = y_pred

        self.y_test = y_test_original
        self.y_pred = y_pred_original

        rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)

        print("\nDirect Model Results")
        print(f"RMSE : {rmse:.3f}")
        print(f"MAE  : {mae:.3f}")
        print(f"R²   : {r2:.4f}")

        plt.figure(figsize=(10, 4))
        plt.plot(self.history["loss"], label="Training Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Direct Model Training History")
        plt.legend()
        plt.grid(True)
        plt.show()

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    def predict_after_21_days(self, latest_data, scaler_X, scaler_y, current_price):
        X_current = pd.DataFrame([[latest_data[feature] for feature in self.features]], columns=self.features)
        X_current_scaled = scaler_X.transform(X_current)

        # Predict scaled price
        predicted_scaled = self.model.predict(X_current_scaled, verbose=0).ravel()[0]

        # Convert back to original gold price
        predicted_price = scaler_y.inverse_transform([[predicted_scaled]])[0][0]

        # Convert Troy ounce -> gram
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

    def save(self, model_path="direct_gold_model.keras"):
        self.model.save(model_path)

    def load(self, model_path="direct_gold_model.keras"):
        self.model = tf.keras.models.load_model(model_path)