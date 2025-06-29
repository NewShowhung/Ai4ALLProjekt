import pandas as pd
import numpy as np

# 1. CSV laden und nach Zeit sortieren
input_path  = 'parkhaeuser_wetter_merged_allvars_mit_preisen.csv'
output_path = 'parkhaeuser_wetter_merged_allvars_mit_preisen_cleaned.csv'

df = pd.read_csv(input_path, parse_dates=['timestamp'], low_memory=False)
df = df.sort_values('timestamp')

# 2. Status-Spalten finden und numerische Spalten bei "Störung" auf NaN setzen
status_cols = [c for c in df.columns if c.endswith('_status')]

for status_col in status_cols:
    prefix = status_col[:-7]  # z.B. "Alsterhaus" von "Alsterhaus_status"
    # numerische Spalten dieses Parkhauses
    num_cols = [f'{prefix}_auslastung',
                f'{prefix}_behindertenst',
                f'{prefix}_frauenst',
                f'{prefix}_frei',
                f'{prefix}_gesamt']
    mask = df[status_col] == 'Störung'
    df.loc[mask, num_cols] = np.nan

# 3. Interpolation für alle numerischen Auslastungs-Spalten
#    (limit_direction='both' füllt auch an Rändern)
all_num = [c for c in df.columns
           if any(c.endswith(suf) for suf in 
                  ['_auslastung','_behindertenst','_frauenst','_frei','_gesamt'])]
df[all_num] = df[all_num].interpolate(method='linear', limit_direction='both')

# 4. Ergebnis speichern
df.to_csv(output_path, index=False)
print(f'Bereinigte Datei gespeichert unter: {output_path}')