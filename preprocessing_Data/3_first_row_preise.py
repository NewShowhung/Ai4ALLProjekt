import pandas as pd
import os
import re

# --- Pfade anpassen ---
base_dir    = r'XXXXXX'
input_csv   = os.path.join(base_dir, 'parkhaeuser_wetter_merged_allvars.csv')
output_csv  = os.path.join(base_dir, 'preise_first_row.csv')

# 1) Einlesen
df = pd.read_csv(input_csv)

# 2) Spalten mit *_preise finden
preise_cols = [c for c in df.columns if re.search(r'_preise$', c)]
if not preise_cols:
    raise ValueError("Keine Spalten mit '_preise' gefunden!")

# 3) Erste Zeile dieser Spalten extrahieren
first_row = df.loc[0, preise_cols]  # Ergebnis: Series

# 4) In DataFrame umwandeln und speichern
first_row.to_frame().T.to_csv(output_csv, index=False)

print(f"✅ Erste Zeile der *_preise-Spalten gespeichert in:\n   {output_csv}")
