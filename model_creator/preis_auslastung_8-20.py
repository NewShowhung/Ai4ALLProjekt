import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# DATEN EINLESEN
df = pd.read_csv('parkhaeuser_wetter_merged_allvars_mit_preisen.csv', parse_dates=['timestamp'])

# DF FILTERN: NUR ZEITSTEMPEL VON 08:00 BIS 20:00 UHR
df_filtered = df[(df['timestamp'].dt.hour >= 8) & (df['timestamp'].dt.hour <= 20)]

# SPALTEN FÜR AUSLASTUNG IDENTIFIZIEREN
occ_cols = [col for col in df_filtered.columns if col.endswith('_auslastung')]

# MITTLERE ABSOLUTE AUSLASTUNG UND PREIS BERECHNEN
data = []
for occ_col in occ_cols:
    fac = occ_col[:-len('_auslastung')]
    price_col = f"{fac}_preis_pro_stunde"
    if price_col in df_filtered.columns:
        mean_occ = df_filtered[occ_col].dropna().mean()
        price_vals = df_filtered[price_col].dropna().unique()
        price = price_vals[0] if len(price_vals) > 0 else np.nan
        data.append({'facility': fac, 'mean_occ': mean_occ, 'price': price})

summary = pd.DataFrame(data).dropna()

# LINEARE REGRESSION RECHNEN
X = summary[['price']].values
y = summary['mean_occ'].values
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# R²-Wert AUSGABE
print(f"R²-Wert (8-20 Uhr): {r2:.4f}")

# VERANSCHAULICHUNG
plt.figure()
plt.scatter(summary['price'], summary['mean_occ'])
plt.plot(summary['price'], y_pred)
plt.xlabel('Preis pro Stunde (€)')
plt.ylabel('Durchschnittliche Auslastung (Stellplätze)')
plt.title(f'Preis vs. Auslastung (8-20 Uhr, R² = {r2:.4f})')
plt.grid(True)
plt.show()