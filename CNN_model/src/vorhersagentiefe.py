import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split

from model import *
from preprocess_data import *

ROTE_BEETE: int = 0

def plot_model_predictions(X_scaled: np.ndarray, y_true: np.ndarray, scaler_y: StandardScaler, save_path: str, depth: int = 1) -> None:
    """
        Visualisiert die Vorhersagen eines Modells in einem Diagramm.

        Plottet die vom Modell vorhergesagten Werte in Abhängigkeit der zu ihnen gehörenden "richtigen" Werte. (aus Testdaten) 
        Im Vergleich dazu wird eine Linie für ein perfektes Modell geplottet, um die Güte des Modells anschaulich darzustellen.

        Argumente:
            y_true: list -> Liste mit richtigen Werten
            y_pred: list -> Liste mit vom Modell vorhergesagten Werten
    """
    y_pred: np.ndarray = custom_predict(model, X_scaled, scaler_y, y_min=0, y_max=87, depth=depth)

    perfect_x: list[int] = [0, 87]
    perfect_y: list[int] = [0, 87]
    plt.plot(perfect_x, perfect_y, linestyle='-', color="#FF484B") # Linie für perfekte Vorhersage
    plt.plot(y_true, y_pred, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50") # Vorhersagen des Modells

    # Beschriftung hinzufügen
    font1: dict = {'family':'serif','color':'black','size':20}
    font2: dict = {'family':'serif','color':'black','size':10}
    plt.title("Vorhersagen Zeitreihenmodell", fontdict=font1)
    plt.xlabel("echte Werte", fontdict=font2)
    plt.ylabel("vorhergesagte Werte", fontdict=font2)

    plt.grid(color="#808080", linestyle=":", linewidth=.5)

    print(f"Diagramm wird unter {save_path} gespeichert.")
    plt.savefig(save_path)

"""
    Testet ein Modell für verschiedene Vorhersagetiefen aus.

    Dazu werden Datensätze aus X und y für verschiedene Zeitabstände (bis max. Vorhersagetiefe) vom letztem X-Wert bis zum ersten y-Wert erstellt.
    Anschließend wird das Modell mit diesen Daten evaluiert und die Ergebnisse der einzelnen Durchläufe geplottet.
"""

if __name__ == "__main__":
    PROJECT_DIR: str = os.path.join(os.path.dirname(__file__), "..")
    DATA_PATH: str = os.path.join(PROJECT_DIR, "data/data_20250612.csv")
    COLUMN_INDEX_END_ALSTERHAUS: int = 12 # letzte Spalte mit Daten zum Parkhaus "Alsterhaus"
    COLUMN_INDEX_BEGIN_WEATHER: int = 539 # erste Spalte mit Wetterdaten
    
    WINDOW_LENGTH: int = 18
    WINDOW_OVERLAP: float = .9
    PREDICTION_OFFSET: int = 1
    FEATURES: int = 13

    MAX_DEPTH: int = 48

    print("\n> Daten einlesen..\n")
    df: pd.DataFrame = pd.read_csv(DATA_PATH)

    print(f"Anzahl Merkmale: {len(df.columns)}")
    print(f"Anzahl Beobachtungen: {len(df.index)}")

    print("\n> unnütze Spalten und NaNs entfernen..\n")
    df = delete_bullshit(df, COLUMN_INDEX_END_ALSTERHAUS+1, COLUMN_INDEX_BEGIN_WEATHER)

    # print(df.describe().transpose())

    print("\n> in Zeitreihen umwandeln..\n")
    X: np.ndarray = np.array([])
    y: np.ndarray = np.array([])
    X, y = convert_2_timeseries(df, WINDOW_LENGTH, WINDOW_OVERLAP, PREDICTION_OFFSET)

    print("\n> in Trainings-, Validierungs- & Testdaten aufteilen..\n")
    X_train: np.ndarray = np.array([])
    X_val: np.ndarray = np.array([])
    X_test: np.ndarray = np.array([])
    y_train: np.ndarray = np.array([])
    y_val: np.ndarray = np.array([])
    y_test: np.ndarray = np.array([])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.6, random_state=ROTE_BEETE)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=.5, random_state=ROTE_BEETE)

    print(f"Datensätze in Trainingsdaten: {len(y_train)}\nDatensätze in Validierungsdaten: {len(y_val)}\nDatensätze in Testdaten: {len(y_test)}\n")

    print("\n> Daten standardisieren..\n")
    X_train_scaled: np.ndarray = np.array([])
    X_val_scaled: np.ndarray = np.array([])
    X_test_scaled: np.ndarray = np.array([])
    y_train_scaled: np.ndarray = np.array([])
    y_val_scaled: np.ndarray = np.array([])
    y_test_scaled: np.ndarray = np.array([])
    scaler_X: StandardScaler = StandardScaler()
    scaler_y: StandardScaler = StandardScaler()
    _, _, _, _, _, _, scaler_X, scaler_y = scale_data(X_train, X_val, X_test, y_train, y_val, y_test, FEATURES, WINDOW_LENGTH)

    print("Modell laden..")
    model: keras.models.Sequential = keras.models.load_model(os.path.join(PROJECT_DIR, "models/model_small.keras")) # type: ignore

    log_file = open(os.path.join(PROJECT_DIR, "logs/vorhersagentiefe_small.log"), "w")

    for i in range(1, MAX_DEPTH+1):
        print(f"\noffset={i} -> {i*5} min")
        log_file.write(f"\noffset={i} -> {i*5} min\n")

        X, y = convert_2_timeseries(df, WINDOW_LENGTH, WINDOW_OVERLAP, prediction_offset=i, cutoff=7500) # Datensätze mit passendem Abstand zwischen X und y generieren        

        # GANZ WICHTIG: train, test & val behalten, auch wenn später nur val benutzt wird
        # -> sonst funktioniert Evaluation nicht (gibt dann immer gleiche Metriken aus, kein Plan warum)
        X_train: np.ndarray = np.array([])
        X_val: np.ndarray = np.array([])
        X_test: np.ndarray = np.array([])
        y_train: np.ndarray = np.array([])
        y_val: np.ndarray = np.array([])
        y_test: np.ndarray = np.array([])
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.6, random_state=ROTE_BEETE)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=.5, random_state=ROTE_BEETE)

        # skalieren
        n_samples, _, _ = X_val.shape
        X_2D: np.ndarray = X_val.reshape(-1, FEATURES)
        X_scaled: np.ndarray = scaler_X.transform(X_2D).reshape(n_samples, WINDOW_LENGTH, FEATURES)
        y_scaled: np.ndarray = scaler_y.transform(y_val.reshape(-1, 1))

        print("\n> Evaluation..\n")
        mae, mse, rmse, r2 = evaluate_model(model, X_scaled, y_val, scaler_y, depth=i)

        print(f"mae:\t{mae}\nmse:\t{mse}\nrmse:\t{rmse}\nr2:\t{r2}")
        log_file.write(f"mae:\t{mae}\nmse:\t{mse}\nrmse:\t{rmse}\nr2:\t{r2}\n")

        plot_model_predictions(X_scaled, y_val, scaler_y, os.path.join(PROJECT_DIR, f"img/vorhersagentiefe/small/{i*5}min.png"), depth=i)
