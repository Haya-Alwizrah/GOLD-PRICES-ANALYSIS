import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class GoldPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.history = None
        self.features = None
        self.history_train = None
        self.y_test = None
        self.y_pred = None

    def train(self, X_train, y_train, lr=0.001, validation_split=0.2, epochs=200, batch_size=32, es_patience=20):

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

    def forecast(self):
        pass