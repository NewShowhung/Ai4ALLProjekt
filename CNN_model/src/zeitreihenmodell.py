import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

from model import *
from preprocess_data import *

ROTE_BEETE: int = 0

def plot_model_predictions(X_scaled: np.ndarray, y_true: np.ndarray, scaler_y: StandardScaler) -> None:
    """
        Visualisiert die Vorhersagen eines Modells in einem Diagramm.

        Plottet die vom Modell vorhergesagten Werte in Abhängigkeit der zu ihnen gehörenden "richtigen" Werte. (aus Testdaten) 
        Im Vergleich dazu wird eine Linie für ein perfektes Modell geplottet, um die Güte des Modells anschaulich darzustellen.

        Argumente:
            y_true: list -> Liste mit richtigen Werten
            y_pred: list -> Liste mit vom Modell vorhergesagten Werten
    """
    y_pred: np.ndarray = custom_predict(model, X_scaled, scaler_y, y_min=0, y_max=87, depth=6)

    metrics = evaluate_model(model, X_scaled, y_true, scaler_y)

    perfect_x: list[int] = [0, 87]
    perfect_y: list[int] = [0, 87]
    plt.plot(perfect_x, perfect_y, linestyle='-', color="#FF484B") # Linie für perfekte Vorhersage
    plt.plot(y_true, y_pred, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50") # Vorhersagen des Modells

    # Beschriftung hinzufügen
    font1: dict = {'family':'serif','color':'black','size':20}
    font2: dict = {'family':'serif','color':'black','size':10}
    plt.title("Vorhersage vs. Ist - Alsterhaus", fontdict=font1)
    plt.xlabel("Ist-Wert (Auslastung)", fontdict=font2)
    plt.ylabel("Vorhergesagt (Auslastung)", fontdict=font2)

    plt.grid(color="#808080", linestyle=":", linewidth=.5)

    text_metrics = f"MAE: {metrics[0]:.2f}\nMSE: {metrics[1]:.2f}\nRMSE: {metrics[2]:.2f}\nR²: {metrics[3]:.2f}"
    plt.text(0.95, 0.05, text_metrics, transform=plt.gca().transAxes, 
             verticalalignment='bottom', horizontalalignment='right', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    save_path: str = input("Wo soll das Diagramm gespeichert werden? ")
    # save_path: str = f"{SRC_DIR}/../img/temp.png"
    print(f"Diagramm wird unter {save_path} gespeichert.")
    plt.savefig(save_path)

    show_plot: str = input("\nDiagramm öffnen? (y/n) ").lower()
    if show_plot == 'y':
        os.system(f"xdg-open {save_path}")

if __name__ == "__main__":
    PROJECT_DIR: str = os.path.join(os.path.dirname(__file__), "..")
    DATA_PATH: str = os.path.join(PROJECT_DIR, "data/data_20250612.csv")
    COLUMN_INDEX_END_ALSTERHAUS: int = 12 # letzte Spalte mit Daten zum Parkhaus "Alsterhaus"
    COLUMN_INDEX_BEGIN_WEATHER: int = 539 # erste Spalte mit Wetterdaten
    
    WINDOW_LENGTH: int = 18
    WINDOW_OVERLAP: float = .9
    PREDICTION_OFFSET: int = 1
    FEATURES: int = 13

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
    X, y = convert_2_timeseries(df, WINDOW_LENGTH, WINDOW_OVERLAP, PREDICTION_OFFSET, cutoff=7500) # cutoff=7500, damit beim Vorhersagetiefetest gleiche Daten vorliegen
    # Erzeugt Testdaten für Vorhersage in 6 Zeitschritten (30 min)
    # Problem: Trainingsdaten des Modells sind mit enthalten, hab gerade kein Plan, wie man die rausfiltert
    # X6, y6 = convert_2_timeseries(df, WINDOW_LENGTH, WINDOW_OVERLAP, 6)

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
    X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_sca5led, scaler_X, scaler_y = scale_data(X_train, X_val, X_test, y_train, y_val, y_test, FEATURES, WINDOW_LENGTH)

    # Skalierung der Testwerte für Vorhersage in 6 Zeitschritten
    # n_samples, _, _ = X6.shape
    # X6_2D: np.ndarray = X6.reshape(-1, FEATURES)
    # X6_scaled: np.ndarray = scaler_X.transform(X6_2D).reshape(n_samples, WINDOW_LENGTH, FEATURES)
    # y6_scaled: np.ndarray = scaler_y.transform(y6.reshape(-1, 1))

    model: keras.models.Sequential = keras.models.Sequential()
    load_existing_model = input("Gespeichertes Modell laden? (ansonsten wird ein neues trainiert) (y/n)").lower()
    if load_existing_model == 'y':
        model_path = input("Pfad zum Modell: ")
        model = keras.models.load_model(model_path) # type: ignore
    else:
        print("\n> Model kompilieren..\n")
        model = build_model(WINDOW_LENGTH, FEATURES, verbose=True)

        print("\n> Training..\n")
        model.fit(X_train_scaled, y_train_scaled, epochs=20, batch_size=32, verbose="1")

    print("\n> Evaluation..\n")
    evaluate_model(model, X_val_scaled, y_val, scaler_y)

    save = input("Modell speichern? (y/n) ").lower()
    if save == 'y':
        model_name = input("Modellname: ")
        # model_path = f"{SRC_DIR}/../models/model1.keras"
        model.save(os.path.join(PROJECT_DIR, f"models/{model_name}"))

    plot_model_predictions(X_val_scaled, y_val, scaler_y)
