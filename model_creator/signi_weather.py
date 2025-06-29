import pandas as pd
import statsmodels.api as sm

# CSV einlesen
df = pd.read_csv('parkhaeuser_wetter_merged_allvars_mit_preisen.csv')

# Spaltenlisten
weather_cols       = ['wind_dir','wind_speed','wind_speed_min','wind_speed_max','wind_gust',
                      'pressure','humidity','temp','temp_min','temp_max','soiltemp']
hourly_price_cols  = [col for col in df.columns if col.endswith('_preis_pro_stunde')]
daily_price_cols   = [col for col in df.columns if col.endswith('_max_tagespreis')]
occupancy_cols     = [col for col in df.columns if col.endswith('_auslastung')]

# Für jedes Parkhaus & jeden Prädiktor jeweils getrennt dropna -> so bleibt möglichst viel Data erhalten
for occ in occupancy_cols:
    print(f"\n===== Parkhaus-Auslastung: {occ} =====")
    for pred in weather_cols + hourly_price_cols + daily_price_cols:
        # nur die beiden Spalten betrachten und NaNs wegwerfen
        sub = df[[pred, occ]].dropna()
        
        # Check, ob genug Varianz da ist
        if sub[pred].nunique() < 2 or sub[occ].nunique() < 2:
            print(f"{pred:40s} → übersprungen (zu wenig Varianz: "
                  f"{pred} hat {sub[pred].nunique()} Werte, "
                  f"{occ} hat {sub[occ].nunique()} Werte)")
            continue
        
        # Regression
        X = sm.add_constant(sub[pred])
        y = sub[occ]
        model = sm.OLS(y, X).fit()
        coef  = model.params[pred]
        p_val = model.pvalues[pred]
        sig   = "✔" if p_val <= 0.05 else "✘"
        
        print(f"{pred:40s} | Coef: {coef: .4f} | p-Wert: {p_val:.4e} {sig}")
