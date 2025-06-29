from datetime import datetime
import gc
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import tensorflow as tf
import zeitreihenmodell as zm

ROTE_BEETE = 0

if __name__ == "__main__":
    print("--- Test verschiedener Fensterlängen ---")
    
    MIN_WINDOW_LENGTH: int = int(sys.argv[1])
    MAX_WINDOW_LENGTH: int = int(sys.argv[2]) # nach kurzer Überschlagsrechnung wird das Modell mies riesig -> ich setzte jetzt mal bei 75 die Grenze..
    MODELS_PER_WINDOW_LENGTH: int = 20
    NUMBER_OF_MODELS: int = MODELS_PER_WINDOW_LENGTH * (MAX_WINDOW_LENGTH - MIN_WINDOW_LENGTH + 1)
    models_tested: int = 0

    print(f"min. Fensterlänge: {MIN_WINDOW_LENGTH}, max. Fensterlänge: {MAX_WINDOW_LENGTH}")
    
    PROJECT_DIR: str = os.path.join(os.path.dirname(__file__), "..")
    DATA_PATH: str = os.path.join(PROJECT_DIR, "data/data.csv")
    COLUMN_INDEX_END_ALSTERHAUS: int = 12 # letzte Spalte mit Daten zum Parkhaus "Alsterhaus"
    COLUMN_INDEX_BEGIN_WEATHER: int = 539 # erste Spalte mit Wetterdaten
    
    WINDOW_OVERLAP: float = .8
    PREDICTION_OFFSET: int = 1
    FEATURES: int = 13

    print("\n> Daten einlesen..\n")
    df: pd.DataFrame = pd.read_csv(DATA_PATH)

    print(f"Anzahl Merkmale: {len(df.columns)}")
    print(f"Anzahl Beobachtungen: {len(df.index)}")

    print("\n> unnütze Spalten und NaNs entfernen..\n")
    df = zm.delete_bullshit(df, COLUMN_INDEX_END_ALSTERHAUS+1, COLUMN_INDEX_BEGIN_WEATHER)

    print(f"\n> Fensterlängen von {MIN_WINDOW_LENGTH} bis {MAX_WINDOW_LENGTH} testen..\n")
    mae_list: list[float] = []
    mse_list: list[float] = []
    rmse_list: list[float] = []
    r2_list: list[float] = []

    for window_length in range(MIN_WINDOW_LENGTH, MAX_WINDOW_LENGTH+1):
        print(f"Fensterlänge={window_length}:")

        X: np.ndarray = np.array([])
        y: np.ndarray = np.array([])
        X, y = zm.convert_2_timeseries(df, window_length, WINDOW_OVERLAP, PREDICTION_OFFSET)

        X_train: np.ndarray = np.array([])
        X_val: np.ndarray = np.array([])
        X_test: np.ndarray = np.array([])
        y_train: np.ndarray = np.array([])
        y_val: np.ndarray = np.array([])
        y_test: np.ndarray = np.array([])
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.6, random_state=ROTE_BEETE)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=.5, random_state=ROTE_BEETE)
        # zu float32 casten -> spart Speicher, sonst float64
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        print(f"Datensätze in Trainingsdaten: {len(y_train)}\nDatensätze in Validierungsdaten: {len(y_val)}\nDatensätze in Testdaten: {len(y_test)}\n")
        # Variablen so bald wie möglich freigeben
        del X, y
        gc.collect()

        X_train_scaled: np.ndarray = np.array([])
        X_val_scaled: np.ndarray = np.array([])
        X_test_scaled: np.ndarray = np.array([])
        y_train_scaled: np.ndarray = np.array([])
        y_val_scaled: np.ndarray = np.array([])
        y_test_scaled: np.ndarray = np.array([])
        scaler_X: StandardScaler = StandardScaler()
        scaler_y: StandardScaler = StandardScaler()
        X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_y = zm.scale_data(X_train, X_val, X_test, y_train, y_val, y_test, FEATURES, window_length)
        # zu float32 casten -> spart Speicher, sonst float64
        X_train_scaled = X_train_scaled.astype(np.float32)
        X_val_scaled = X_val_scaled.astype(np.float32)
        y_train_scaled = y_train_scaled.astype(np.float32)
        y_val_scaled = y_val_scaled.astype(np.float32)
        # Variablen so bald wie möglich freigeben
        del X_test, y_test, X_test_scaled, y_test_scaled
        gc.collect()

        model = tf.keras.models.Sequential # type: ignore

        mae_sum: float = 0
        mse_sum: float = 0
        rmse_sum: float = 0
        r2_sum: float = 0
        # mehrere Modelle mit Fensterlänge trainieren um Metriken zu mitteln -> Ausreißer ausgleichen
        for i in range(0, MODELS_PER_WINDOW_LENGTH):
            metrics: list[float] = []

            print(f"({100*(models_tested / NUMBER_OF_MODELS)}%) Modell {i} von {MODELS_PER_WINDOW_LENGTH}, window_length={window_length}")

            model = zm.build_model(window_length, FEATURES, verbose=False)
            model.fit(X_train_scaled, y_train_scaled, epochs=20, batch_size=8, verbose=0)
            metrics = zm.evaluate_model(model, X_val_scaled, y_val, scaler_y, depth=1)

            # Modell freigeben -> brauchen ggf viel Speicher
            tf.keras.backend.clear_session() # type: ignore
            del model
            gc.collect()

            mae_sum += metrics[0]
            mse_sum += metrics[1]
            rmse_sum += metrics[2]
            r2_sum += metrics[3]

            models_tested += 1

        mae_list.append(mae_sum / MODELS_PER_WINDOW_LENGTH)
        mse_list.append(mse_sum / MODELS_PER_WINDOW_LENGTH)
        rmse_list.append(rmse_sum / MODELS_PER_WINDOW_LENGTH)
        r2_list.append(r2_sum / MODELS_PER_WINDOW_LENGTH)
        # Metriken zwischenspeichern, damit nach Abbruch nicht alles verloren ist
        np.savez(os.path.join(PROJECT_DIR, f"data/metrics_partial_{MIN_WINDOW_LENGTH}_{MAX_WINDOW_LENGTH}.npz"), mae=mae_list, mse=mse_list, rmse=rmse_list, r2=r2_list)

        # Speicher aufräumen - ich dachte das muss ich nur in C.. :O
        del X_train, X_val, y_train, y_val
        del X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled
        del scaler_y
        tf.keras.backend.clear_session() # type: ignore
        gc.collect()
    
    # Ergebnisse plotten
    m: float = 0
    b: float = 0
    x: np.ndarray = np.arange(MIN_WINDOW_LENGTH, MAX_WINDOW_LENGTH + 1)

    a, b, c = np.polyfit(x, mae_list, 2)
    plt.subplot(2, 2, 1)
    plt.plot(x, a*x**2+b*x+c, linestyle='-', color="#FF484B") # Regressionspolynom -> Verlauf des MAE, analog dazu später für andere Metriken
    plt.plot(x, mae_list, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50")
    plt.title("Mean Absolute Error")
    plt.xlabel("Fensterlänge")
    plt.ylabel("MAE")

    a, b, c = np.polyfit(x, mse_list, 2)
    plt.subplot(2, 2, 2)
    plt.plot(x, a*x**2+b*x+c, linestyle='-', color="#FF484B")
    plt.plot(x, mse_list, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50")
    plt.title("Mean Squared Error")
    plt.xlabel("Fensterlänge")
    plt.ylabel("MSE")

    a, b, c = np.polyfit(x, rmse_list, 2)
    plt.subplot(2, 2, 3)
    plt.plot(x, a*x**2+b*x+c, linestyle='-', color="#FF484B")
    plt.plot(x, rmse_list, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50")
    plt.title("Root Mean Squared Error")
    plt.xlabel("Fensterlänge")
    plt.ylabel("RMSE")

    a, b, c = np.polyfit(x, r2_list, 2)
    plt.subplot(2, 2, 4)
    plt.plot(x, a*x**2+b*x+c, linestyle='-', color="#FF484B")
    plt.plot(x, r2_list, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50")
    plt.title("R2 Score")
    plt.xlabel("Fensterlänge")
    plt.ylabel("R2")

    plt.tight_layout()

    save_path: str = os.path.join(PROJECT_DIR, "img/fensterlaengen.png")
    print(f"Diagramm wird unter {save_path} gespeichert.")
    plt.savefig(save_path)

    show_plot: str = input("\nDiagramm öffnen? (y/n) ")
    if show_plot == 'y':
        os.system(f"xdg-open {save_path}")