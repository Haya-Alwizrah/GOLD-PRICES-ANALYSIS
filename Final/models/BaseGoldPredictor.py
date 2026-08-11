import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from pathlib import Path
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

class BaseGoldPredictor:
    def __init__(self, model_name):
        self.model = None
        self.features = None
        self.history = None
        self.best_config = None
        self.tuning_results = []
        self.model_name = model_name

        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.CHECKPOINTS_DIR = self.BASE_DIR / "checkpoints"
        self.CHECKPOINTS_DIR.mkdir(exist_ok=True)

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
            optimizer=tf.keras.optimizers.Adam(lr),
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
                self.CHECKPOINTS_DIR /f"best_{self.model_name.lower().replace(' ', '_')}_gold_model.keras",
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

        history_path = self.CHECKPOINTS_DIR /f"{self.model_name.lower().replace(' ', '_')}_history.pkl"
        with open(history_path, "wb") as f:
            pickle.dump(self.history, f)

        return history

    def evaluate(self, X_test, y_test):

        y_pred = self.model.predict(X_test, verbose=0).flatten()

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n{self.model_name} Results")
        print(f"RMSE : {rmse:.3f}")
        print(f"MAE  : {mae:.3f}")
        print(f"R²   : {r2:.4f}")

        plt.figure(figsize=(10, 4))
        plt.plot(self.history["loss"], label="Training Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title(f"{self.model_name} Model Training History")
        plt.legend()
        plt.grid(True)
        plt.show()

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        }

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

        history_path = self.CHECKPOINTS_DIR / f"{self.model_name.lower().replace(' ', '_')}_history.pkl"
        with open(history_path, "rb") as f:
            self.history = pickle.load(f)