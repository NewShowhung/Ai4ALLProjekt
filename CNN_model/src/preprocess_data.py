import math
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import StandardScaler

def delete_bullshit(df: pd.DataFrame, column_begin_delete: int, column_end_delete: int) -> pd.DataFrame:
    """
        Entfernt Spalten zwischen den Indizes (inklusiv) und Zeilen mit NaN.

        Argumente:
            df: pd.DataFrame -> DataFrame der bereinigt werden soll
            column_begin_delete: int -> Index der ersten Spalte die gelöscht werden soll
            column_end_delete: int -> Index der letzten Spalte die gelöscht werden soll
        
        Gibt zurück
            df: Dataframe -> bereinigter Dataframe
    """
    # nicht erwünschte Spalten entfernen
    df = df.drop(df.iloc[:, column_begin_delete:column_end_delete].columns, axis=1)

    # Indizes der Zeilen mit NaN & Störungen heraussuchen
    indexes_to_delete: list = []
    for c in df.columns:
        for i in range(0, len(df.index)):
            if isinstance(df[c].iloc[i], (int, float)) and math.isnan(df[c].iloc[i]) and i not in indexes_to_delete: indexes_to_delete.append(i) # NaN -> irgednwas stimmt mit Daten nicht
            elif df[c].iloc[i] == "Störung" and i not in indexes_to_delete: indexes_to_delete.append(i) # Störung -> Daten sind Quatsch
    # Zeilen mit entfernen
    df = df.drop(indexes_to_delete, axis=0)

    print(f"Anzahl Merkmale nach Bereinigung: {len(df.columns)}")
    print(f"Anzahl Beobachtungen nach Bereinigung: {len(df.index)}")

    return df

def get_features_2_append_2_timeseries(df: pd.DataFrame, i: int, onehot: bool = False) -> list[float]:
    """
        Sammelt für Zeitreihe benötigte Features aus einer Zeile des Dataframes.

        Argumente:
            df: pf.DataFrame -> Daten
            i: int -> Zeilenindex aus welcher Zeile von df Daten ausgelesen werden sollen
            onehot: bool (optional) -> legt fest ob Wochentage onehot-encoded werden sollen, standardmäßig onehot=False
    """

    ret: list[float] = []

    weekday: int = datetime.strptime(df["timestamp"].iloc[i], "%Y-%m-%d %H:%M:%S").weekday()

    ret.append(df["Alsterhaus_frei"].iloc[i])
    ret.append(df["temp"].iloc[i])
    ret.append(df["sunshine"].iloc[i])
    ret.append(df["wind_speed"].iloc[i])
    ret.append(df["wind_gust"].iloc[i]) # Windböhen
    ret.append(df["precip"].iloc[i]) # Niederschlag
    # Uhrzeit wäre vielleicht noch toll
    if onehot: # One-Hot-Encoding für Wochentage, damit manche Wochentage nicht als besser gewertet werden
        ret.append(1 if weekday == 0 else 0)
        ret.append(1 if weekday == 1 else 0)
        ret.append(1 if weekday == 2 else 0)
        ret.append(1 if weekday == 3 else 0)
        ret.append(1 if weekday == 4 else 0)
        ret.append(1 if weekday == 5 else 0)
        ret.append(1 if weekday == 6 else 0)
    else:
        ret.append(weekday)

    return ret

def convert_2_timeseries(df: pd.DataFrame, window_length: int, window_overlap: float, prediction_offset: int, cutoff: int = -1) -> list[np.ndarray]:
    """
        Wandelt DataFrame in X und y für ein Zeitreihenmodell um.

        Argumente:
            df: pd.DataFrame -> DataFrame mit Daten
            window_length: int -> Anzahl Zeitpunkte pro Zeitreihe
            window_overlap: float -> zu wie viel % sich aufeinanderfolgende Fenster überlappen
            predicted_value_offset: int -> der wie vielte nach dem letzten im Fenster liegende Wert als als Vorhersagewert dienen soll
            cutoff: int (optional) -> falls cutoff>0 werden nur die ersten cutoff-vielen Zeitreihen zurückgegeben, standardmäßig cutoff=-1

        Gibt zurück:
            list[np.ndarray] -> X und y mit Zeitreihen
    """
    i: int = 0
    X: list[list[float]] = []
    y: list[float] = []
    while (i + window_length + prediction_offset) < len(df.index):
        temp: list = []
        # Anzahl freier Parkplätze aus $window_length aufeinanderfolgenden Zeilen in eine Zeile von X schreiben
        for j in range(i, i+window_length):
            temp.append(get_features_2_append_2_timeseries(df, j, onehot=True))
        X.append(temp)
        # vorherzusagenden Wert zu y hinzufügen (Wert 1/2 Stunde nach dem letzten Wert in X)
        y.append(df["Alsterhaus_frei"].iloc[i + window_length + prediction_offset])

        i += max(int(window_length * (1 - window_overlap)), 1)

    # sorgt für einheitliche Anzahl an Zeitreihen -> mit zunehmender Tiefe entstehen weniger Zeitreihen, was manchmal stört
    if cutoff > 0:
        X = X[:cutoff][:][:]
        y = y[:cutoff]

    print(f"\nAnzahl Zeitreihen: {len(y)}")

    return [np.array(X), np.array(y)]

def scale_data(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, features: int, window_length: int) -> list:
    """
        Standardisiert die Daten, dh. sie werden so skaliert, dass Mittelwert = 0 und Varianz = 1 sind.

        Argumente:
            X_train: np.array, 
            X_val: np.array, 
            X_test: np.array, 
            y_train: np.array, 
            y_val: np.array, 
            y_test: np.array -> unstandardisierte Daten
            features: int -> Anzahl Features
            window_length: int -> Fensterlänge
            
        Gibt zurück:
            list -> standardisierte Daten + die verwendeten StandardScaler
            [X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_y]
    """

    n_samples: int = 0
    n_samples, _, _ = X_train.shape

    X_train_2D: np.ndarray = X_train.reshape(-1, features)

    scaler_X: StandardScaler = StandardScaler()
    X_train_scaled_2D: np.ndarray = scaler_X.fit_transform(X_train_2D)

    X_train_scaled: np.ndarray = X_train_scaled_2D.reshape(n_samples, window_length, features)

    n_samples, _, _ = X_val.shape
    X_val_2D: np.ndarray = X_val.reshape(-1, features)
    X_val_scaled: np.ndarray = scaler_X.transform(X_val_2D).reshape(n_samples, window_length, features)

    n_samples, _, _ = X_test.shape
    X_test_2D: np.ndarray = X_test.reshape(-1, features)
    X_test_scaled: np.ndarray = scaler_X.transform(X_test_2D).reshape(n_samples, window_length, features)

    scaler_y: StandardScaler = StandardScaler()
    X_train_1st_col_2D: np.ndarray = X_train[:,:,0].reshape(-1, 1)
    scaler_y.fit(X_train_1st_col_2D)

    y_train_scaled: np.ndarray = scaler_y.transform(y_train.reshape(-1, 1))
    y_val_scaled: np.ndarray = scaler_y.transform(y_val.reshape(-1, 1))
    y_test_scaled: np.ndarray = scaler_y.transform(y_test.reshape(-1, 1))

    return [X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_y]