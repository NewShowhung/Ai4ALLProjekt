import math
import numpy as np
import sklearn.metrics
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input # type: ignore

def build_model(window_length: int, features: int, verbose = False) -> tf.keras.models.Sequential: # type: ignore
    """
        Baut und kompiliert ein Zeitreihen-CNN.

        Argumente:
            window_length: int -> Fensterlänge, also wie viele Werte in einer Zeitreihe enthalten sind
            features: int -> Anzahl Features
    """    
    model = tf.keras.models.Sequential([ # type: ignore
        Input((window_length, features)),
        Conv1D(filters=128, kernel_size=3, activation="relu"),
        MaxPooling1D(2),
        Conv1D(filters=128, kernel_size=3, activation="relu"),
        Flatten(),
        Dropout(.01),
        Dense(256, activation="relu"),
        Dense(256, activation="relu"),
        Dropout(.01),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()] # type: ignore
    )

    if verbose: model.summary()
    else: print(f"Anzahl Parameter: {model.count_params()}")

    return model

def custom_predict(model: tf.keras.models.Sequential, X_scaled: np.ndarray, scaler_y: StandardScaler, y_min: float, y_max: float, depth: int = 1) -> np.ndarray: # type: ignore
    """
        Sagt Werte in vorgegebenen Schranken nach depth Zeitschritten vorher.

        Nutzt model.predict() um Werte für y vorherzusagen und sorgt dafür, das diese nur zwischen y_min und y_max liegen.
        Größere Werte werden auf y_max und kleinere Werte auf y_min gesetzt. Der vorhergesagte Wert wird den Eingabewerten angehängt
        und erneut eine Vorhersage getroffen. Dies wird so oft wiederholt, bis depth-viele Zeitschritte vorhergesagt wurden.
        Die letzte Vorhersage wird zurückgegeben

        Argumente:
            model: tf.keras.models.Sequential -> Modell das zur Vorhersage genutzt wird
            X_scaled: np.ndarray -> skalierte Daten, für die Vorhersagen getroffen werden sollen
            scaler_y: StandardScaler -> StandardScaler für Vorhersagen
            y_min: float -> minimaler Wert für Vorhersagen
            y_max: float -> maximaler Wert für Vorhersagen
            depth: int (optional) -> wie viele Zeitschritte vorhergesagt werden sollen, standardmäßig depth=1

        Gibt zurück:
            y: np.ndarray -> Array mit Vorhersagen
    """
    y: np.ndarray = np.array(None)
    # sagt so lange Parkplatzwerte voraus und hängt sie den Inputwerten an, bis depth erreicht ist
    for d in range(0, depth):
        y_scaled: list[float] = model.predict(X_scaled)
        y = scaler_y.inverse_transform(y_scaled)
        
        y = np.clip(y, a_min=y_min, a_max=y_max) # auf [y_min, y_max] beschränken
        
        # schiebt alle Parkplatzwerte in X eine Stelle nach vorne und hängt y_scaled ans Ende an -> Grundlage für nächste Vorhersage
        for i in range(0, len(X_scaled)):
            for j in range(0, len(X_scaled[0][0]) - 1):
                X_scaled[i][0][j] = X_scaled[i][0][j + 1] # Parkplatzwerte nach vorne schieben
            X_scaled[i][0][len(X_scaled[0][0]) - 1] = y_scaled[i] # aktuelle Vorhersage anhängen

    return y

def evaluate_model(model: tf.keras.models.Sequential, X_scaled: np.ndarray, y_true: np.ndarray, scaler_y: StandardScaler, depth: int = 1, verbose: bool = False) -> list[float]: # type: ignore
    """
        Berechnet Metriken für ein Modell.

        Berechnet Vorhersagen für X nach depth Zeitschritten und daraus MAE, MSE, RMSE und R2-Score des Modells.

        Argumente:
            model: tf.keras.models.Sequential -> Modell das evaluiert werden soll
            X_scaled: np.ndarray -> skalierter Input für das Modell
            y_true: np.ndarray -> richtige Vorhersagen für X_scaled
            scaler_y: StandardScaler -> StandardScaler um Skalierung der Vorhersagen zu reversen
            depth: int (optional) -> wie viele Zeitschritte vorhergesagt werden sollen, standardmäßig depth=1
            verbose: bool (optional) -> falls True werden absolute & relative Abweichung jedes Datenpaars ausgegeben, standardmäßig verbode=False
        
        Gibt zurück:
            list[float] -> Liste mit Metriken des Modells. [MAE, MSE, RMSE, R2] 
    """
    y_pred: np.ndarray = custom_predict(model, X_scaled, scaler_y, y_min=0, y_max=87, depth=depth)

    mae: float = 0
    mse: float = 0
    rmse: float = 0
    r2: float = 0
    mae_rel: float = 0

    for i in range(0, len(y_true)):
        diff: float = abs(y_true[i] - y_pred[i])[0]
        mae += diff
        mse += diff**2
        # Division durch 0 abfangen, wenn keine Parkplätze frei waren (y_val = 0)
        if y_true[i] != 0:
            mae_rel += diff / y_true[i]
        else:
            # kein Plan was man jetzt hier addieren soll, ich nehme einfach mal diff so wies ist
            mae_rel += diff

        if verbose: print(f"real: {y_true[i]}, predicted: {y_pred[i]}, diff: {diff}, relative error: {(diff / y_true[i]) if y_true[i] != 0 else "undefined"}")

    mae = mae / len(y_pred)
    mse = mse / len(y_pred)
    rmse = math.sqrt(mse)
    r2 = float(sklearn.metrics.r2_score(y_true, y_pred))
    mae_rel = mae_rel / len(y_pred)
    print(f"\n-- Metriken --\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nMAE(relative): {mae_rel}\nR2: {r2}\n")

    return [mae, mse, rmse, r2]