import os
import re
import pandas as pd
import matplotlib            
matplotlib.use('Agg')        
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

#Pfad zur CSV-Datei
CSV_PATH = 'parkhaeuser_wetter_merged_allvars_mit_preisen_cleaned.csv'

#Verzeichnis anlegen, in dem die Modelle, Importances, Plots und Metriken gespeichert werden
OUTPUT_DIR = 'C:/Users/Showhung/OneDrive/py/data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Liste zum Sammeln der Metriken aller Parkhäuser
all_metrics = []

#Datensatz einlesen (inkl. Timestamp)
data = pd.read_csv(CSV_PATH, parse_dates=['timestamp'])

#Wetter-Spalten definieren (anpassen, falls nötig)
weather_cols = [
    'wind_dir', 'wind_speed', 'wind_speed_min', 'wind_speed_max', 'wind_gust',
    'pressure', 'rad_diffuse', 'rad_global', 'rad_long', 'humidity',
    'precip_dur', 'precip_ind', 'precip', 'sunshine',
    'temp', 'temp_min', 'temp_max', 'soiltemp', 'dewpoint'
]

#Parkhäuser ermitteln\all_columns = data.columns.tolist()
#Korrektur: all_columns richtig definieren
all_columns = data.columns.tolist()
garage_prefixes = {
    col.rsplit('_', 1)[0] for col in all_columns if col.endswith('_auslastung')
}
print(f"Gefundene Parkhäuser ({len(garage_prefixes)}): {sorted(garage_prefixes)}")

#Safe-Dateiname-Funktion
def make_safe_filename(name: str) -> str:
    safe = re.sub(r'[^0-9A-Za-z]+', '_', name)
    return re.sub(r'_{2,}', '_', safe).strip('_')

#Für jedes Parkhaus: trainieren, evaluieren, Outlier-CSV erzeugen
for garage in sorted(garage_prefixes):
    target_col = f'{garage}_auslastung'
    print(f'\n--- Verarbeite Parkhaus: {garage} ---')

    #Arbeits-DF klonen
    df_g = data.copy()

    #Features und Ziel
    X = df_g[weather_cols]
    y = df_g[target_col]
    timestamps = df_g['timestamp']

    #NaN-freie Indizes
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)
    timestamps = timestamps.loc[valid_idx].reset_index(drop=True)

    #Train/Test-Split
    X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(
        X, y, timestamps, test_size=0.2, random_state=42
    )

    #RandomForest initialisieren & trainieren
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    #Vorhersage & Bewertung
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f'MAE: {mae:.2f} | MSE: {mse:.2f} | R²: {r2:.2f}')

    #Outlier Erkennung
    resid = y_test - y_pred
    abs_resid = resid.abs()
    #Threshold = Mittelwert + 2*StdDev
    thresh = abs_resid.mean() + 2 * abs_resid.std()
    #Outlier-DF
    df_out = pd.DataFrame({
        'timestamp': ts_test,
        'actual':      y_test,
        'predicted':   y_pred,
        'residual':    resid,
        'abs_residual': abs_resid
    })
    outliers = df_out[abs_resid > thresh].sort_values('abs_residual', ascending=False)
    n_outliers = len(outliers)
    print(f'Anzahl Outlier: {n_outliers} (Threshold: {thresh:.2f})')

    #Metriken sammeln
    all_metrics.append({
        'garage': garage,
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'n_outliers': n_outliers,
        'outlier_threshold': thresh
    })

    #Safe-Name fürs Speichern
    safe_name = make_safe_filename(garage)

    #Outliers als CSV speichern
    out_csv = os.path.join(OUTPUT_DIR, f'{safe_name}_outliers.csv')
    outliers.to_csv(out_csv, index=False)
    print(f'Outlier-Tabelle gespeichert in: {out_csv}')

    #Feature-Importances
    importances = pd.Series(rf.feature_importances_, index=weather_cols)\
                    .sort_values(ascending=False)
    fi_csv = os.path.join(OUTPUT_DIR, f'{safe_name}_feature_importances.csv')
    importances.to_csv(fi_csv, header=['importance'])

    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', edgecolor='black')
    plt.gca().invert_yaxis()
    plt.title(f'Feature Importances – {garage}')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{safe_name}_importances.png'), dpi=150)
    plt.close()

    #Regression-Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
    plt.title(f'Vorhersage vs. Ist – {garage}')
    plt.xlabel('Ist-Wert (Auslastung)')
    plt.ylabel('Vorhergesagt (Auslastung)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{safe_name}_regression.png'), dpi=150)
    plt.close()

    #Modell speichern
    model_path = os.path.join(OUTPUT_DIR, f'rf_model_{safe_name}.pkl')
    joblib.dump(rf, model_path)

#Am Ende: alle Metriken in eine einzelne CSV schreiben
metrics_df = pd.DataFrame(all_metrics)
metrics_csv = os.path.join(OUTPUT_DIR, 'all_garage_metrics.csv')
metrics_df.to_csv(metrics_csv, index=False)
print(f'Alle Metriken gespeichert in: {metrics_csv}')

print('\nAlle Parkhaus-Modelle, Diagramme, Outlier-Tabellen und Metriken wurden erzeugt.')
