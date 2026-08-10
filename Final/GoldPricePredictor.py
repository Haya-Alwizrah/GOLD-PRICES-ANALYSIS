import numpy as np
import tensorflow as tf

from RecursiveGoldPredictor import RecursiveGoldPredictor
from DirectGoldPredictor import DirectGoldPredictor

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

class GoldPricePredictor:
    def __init__(self):
        self.recursive_model = RecursiveGoldPredictor()
        self.direct_model = DirectGoldPredictor()

        self.results = {}

        self.configs = [
            dict(
                units1=64,
                units2=32,
                units3=16,
                dropout1=0.2,
                dropout2=0.1,
                l2reg=1e-4,
                lr=1e-3
            ),
            dict(
                units1=128,
                units2=64,
                units3=32,
                dropout1=0.3,
                dropout2=0.2,
                l2reg=1e-3,
                lr=1e-3
            ),
            dict(
                units1=128,
                units2=64,
                units3=32,
                dropout1=0.4,
                dropout2=0.3,
                l2reg=1e-3,
                lr=5e-4
            ),
            dict(
                units1=256,
                units2=128,
                units3=64,
                dropout1=0.4,
                dropout2=0.3,
                l2reg=1e-2,
                lr=1e-3
            )
        ]


    def train_recursive(self, X_train, y_train, X_val, y_val, epochs=300, batch_size=32, tune_epochs=80, tune_patience=10, patience=20):
        print("--------- Training Recursive Model --------------")

        self.recursive_model.tune(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            configs=self.configs,
            epochs=tune_epochs,
            batch_size=batch_size,
            patience=tune_patience
        )

        self.recursive_model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            configs=self.configs,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )

    def train_direct(self, X_train, y_train, X_val, y_val, epochs=300, batch_size=32, tune_epochs=80, tune_patience=10, patience=20):
        print(" ------------ Training Direct Model---------- ")

        self.direct_model.tune(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            configs=self.configs,
            epochs=tune_epochs,
            batch_size=batch_size,
            patience=tune_patience
        )

        self.direct_model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            configs=self.configs,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )

    def evaluate_recursive(self, X_test, y_test):
        result = self.recursive_model.evaluate(X_test, y_test)
        self.results["recursive"] = result
        return result

    def evaluate_direct(self, X_test, y_test):
        result = self.direct_model.evaluate(X_test, y_test)
        self.results["direct"] = result
        return result

    def predict_recursive(self, latest_data, historical_data, scaler_X, days=21):
        return self.recursive_model.forecast(
            latest_data=latest_data,
            historical_data=historical_data,
            scaler_X=scaler_X,
            days=days
        )

    def predict_direct(self, latest_data, X_columns, scaler):
        return self.direct_model.predict(
            latest_data=latest_data,
            X_columns=X_columns,
            scaler=scaler
        )

    def save_models(self, recursive_path="recursive_gold_model.keras", direct_path="direct_gold_model.keras"):
        self.recursive_model.save(recursive_path)
        self.direct_model.save(direct_path)

        print("Models saved successfully.")

    def load_models(self, recursive_path=None, direct_path=None):
        if recursive_path is not None:
            self.recursive_model.load(recursive_path)

        if direct_path is not None:
            self.direct_model.load(direct_path)

        print("Models loaded successfully.")

        