import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Daten einlesen
df = pd.read_csv('parkhaeuser_wetter_merged_allvars_mit_preisen.csv', parse_dates=['timestamp'])

# 2. Spalten für Auslastung und Preis identifizieren
occ_cols = [col for col in df.columns if col.endswith('_auslastung')]
price_cols = [col for col in df.columns if col.endswith('_preis_pro_stunde')]

# 3. Mittlere absolute Auslastung und Stundenpreis je Parkhaus berechnen
data = []
for occ_col in occ_cols:
    fac = occ_col[:-len('_auslastung')]
    price_col = f"{fac}_preis_pro_stunde"
    if price_col in df.columns:
        mean_occ = df[occ_col].dropna().mean()
        price_vals = df[price_col].dropna().unique()
        price = price_vals[0] if len(price_vals) > 0 else np.nan
        data.append({'facility': fac, 'mean_occ': mean_occ, 'price': price})

summary = pd.DataFrame(data).dropna()

# 4. Lineare Regression
X = summary[['price']].values
y = summary['mean_occ'].values
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"R²-Wert: {r2:.4f}")

# 5. Veranschaulichung
plt.figure()
plt.scatter(summary['price'], summary['mean_occ'])
plt.plot(summary['price'], y_pred)
plt.xlabel('Preis pro Stunde (€)')
plt.ylabel('Durchschnittliche Auslastung (Stellplätze)')
plt.title(f'Preis vs. Auslastung (R² = {r2:.4f})')
plt.show()