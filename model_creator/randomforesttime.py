import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# === CSV laden ===
df = pd.read_csv("parkhaeuser_wetter_merged_allvars_mit_preisen_cleaned.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

# === relative Auslastung berechnen, wenn belegte_ + gesamt_ Spalten vorhanden ===
rel_auslastung_spalten = []
for col in df.columns:
    if col.startswith("belegte_Parkplaetze_"):
        parkhaus = col.replace("belegte_Parkplaetze_", "")
        gesamt_col = f"gesamt_Parkplaetze_{parkhaus}"
        if gesamt_col in df.columns:
            rel_col = f"{parkhaus}_auslastung"
            df[rel_col] = df[col] / df[gesamt_col]
            rel_auslastung_spalten.append(rel_col)

# === alternativ alle *_auslastung-Spalten sammeln (falls bereits vorhanden) ===
auslastung_spalten = rel_auslastung_spalten or [col for col in df.columns if col.endswith('_auslastung')]

# === Plot-Ordner vorbereiten ===
output_folder = "randomforest_plots_hourly"
os.makedirs(output_folder, exist_ok=True)

# Referenz-Zeit 0–23
stunden_raster = pd.DataFrame({'hour': list(range(24))})

# === Für jedes Parkhaus ===
for spalte in auslastung_spalten:
    temp_df = df[['hour', spalte]].dropna()
    grouped = temp_df.groupby('hour')[spalte].mean().reset_index()
    grouped = stunden_raster.merge(grouped, on='hour', how='left')
    grouped[spalte] = grouped[spalte].interpolate(limit_direction='both')

    X = grouped[['hour']]
    y = grouped[spalte]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred_all = model.predict(X.sort_values('hour'))
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # === Plot ===
    plt.figure(figsize=(10, 6))
    plt.plot(X['hour'], y, label='Ist', marker='o')
    plt.plot(X.sort_values('hour')['hour'], y_pred_all, label='Vorhergesagt', marker='x')
    plt.xlabel("Uhrzeit (Stunde)")
    plt.ylabel("Auslastung (relativ)")
    plt.title(f"RF-Modell stündlich – {spalte.replace('_auslastung', '')}")
    plt.xticks(range(24))

    plt.text(0.01, 0.99,
             "Zeitintervall: Stundenweise\nModell: Random Forest\nTrain/Test: 80 % / 20 %",
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(facecolor='white', alpha=0.4, edgecolor='gray'))

    textstr = f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nR²: {r2:.3f}"
    plt.text(0.99, 0.01, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

    plt.legend()
    plt.tight_layout()

    safe_name = re.sub(r'[^\w\-_\. ]', '_', spalte)
    plot_path = os.path.join(output_folder, f"{safe_name}_hourly_rf.png")
    plt.savefig(plot_path)
    plt.close()

print("✅ Alle stündlichen RELATIVEN Auslastungsplots gespeichert in 'randomforest_plots_hourly'.")
