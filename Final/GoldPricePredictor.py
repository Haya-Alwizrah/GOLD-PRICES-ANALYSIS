import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import layers, regularizers, callbacks

from RecursiveGoldPredictor import RecursiveGoldPredictor
from DirectGoldPredictor import DirectGoldPredictor

class GoldPricePredictor:
    def __init__(self):
        self.recursive_model = RecursiveGoldPredictor()
        self.direct_model = DirectGoldPredictor()

        self.results = {}

    def train_recursive(self, X_train, y_train, X_val, y_val, configs, epochs=300, batch_size=32, tune_epochs=80, tune_patience=10, patience=20):
        print("--------- Training Recursive Model --------------")

        self.recursive_model.tune(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            configs=configs,
            epochs=tune_epochs,
            batch_size=batch_size,
            patience=tune_patience
        )

        self.recursive_model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            configs=configs,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )

    def train_direct(self, X_train, y_train, X_val, y_val, configs, epochs=300, batch_size=32, tune_epochs=80, tune_patience=10, patience=20):
        print(" ------------ Training Direct Model---------- ")

        self.direct_model.tune(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            configs=configs,
            epochs=tune_epochs,
            batch_size=batch_size,
            patience=tune_patience
        )

        self.direct_model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            configs=configs,
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

    def forecast_recursive(self, latest_data, historical_data, scaler, days=21):
        return self.recursive_model.forecast(
            latest_data=latest_data,
            historical_data=historical_data,
            scaler=scaler,
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

        