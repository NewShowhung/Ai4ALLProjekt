import pandas as pd
import glob, os, re

# — Pfade —
base_dir    = r'OneDrive\py'
park_csv    = os.path.join(base_dir, 'parkhaeuser_timeseries_wide.csv')
weather_dir = os.path.join(base_dir, 'wetter')
output_csv  = os.path.join(base_dir, 'parkhaeuser_wetter_merged_allvars.csv')

# 1) Parkhaus-Daten laden und timestamp erzeugen
wide = pd.read_csv(
    park_csv,
    parse_dates={'timestamp': ['Tag', 'Zeit']},
    infer_datetime_format=True,
    low_memory=False
)

# 2) Unerwünschte Spalten droppen
patterns = [
    "betreiber", "datenherkunft", "einfahrt", "25832", "id", "hausnr",
    "wgs84", "link", "oeffnungszeit", "punkt", "pur", "received",
    "stellplaetze_gesamt", "strasse"
]
regex = re.compile("|".join(patterns), re.IGNORECASE)
to_drop = [c for c in wide.columns if regex.search(c)]
wide.drop(columns=to_drop, inplace=True)

# 3) Mapping für Wetter-Variablen
mapping = {
    'OBS_DEU_PT10M_T2M':   'temp',
    'OBS_DEU_PT10M_T2M_X': 'temp_max',
    'OBS_DEU_PT10M_T2M_N': 'temp_min',
    'OBS_DEU_PT10M_T5CM':  'soiltemp',
    'OBS_DEU_PT10M_T5CM_X':'soiltemp_max',
    'OBS_DEU_PT10M_PP':    'pressure',
    'OBS_DEU_PT10M_TD':    'dewpoint',
    'OBS_DEU_PT10M_RF':    'humidity',
    'OBS_DEU_PT10M_RR-I':  'precip_ind',
    'OBS_DEU_PT10M_RR-D':  'precip_dur',
    'OBS_DEU_PT10M_RR':    'precip',
    'OBS_DEU_PT10M_SD':    'sunshine',
    'OBS_DEU_PT10M_RAD-L':'rad_long',
    'OBS_DEU_PT10M_RAD-G':'rad_global',
    'OBS_DEU_PT10M_RAD-F':'rad_diffuse',
    'OBS_DEU_PT10M_F_X':   'wind_gust',
    'OBS_DEU_PT10M_F':     'wind_speed',
    'OBS_DEU_PT10M_D':     'wind_dir',
    'OBS_DEU_PT10M_F_MX':  'wind_speed_max',
    'OBS_DEU_PT10M_F_MN':  'wind_speed_min'
}

# 4) Wetter-Dateien einlesen, resamplen & interpolieren
weather_series = []
for fn in glob.glob(os.path.join(weather_dir, 'data_OBS_DEU_PT10M_*.csv')):
    var = os.path.basename(fn).replace('data_','').replace('.csv','')
    if var not in mapping:
        print(f"⚠️ Überspringe unbekannt: {var}")
        continue
    short = mapping[var]
    print(f"📥 Lade {var} → Spalte '{short}'")
    
    dfw = pd.read_csv(fn, encoding='utf-8-sig', index_col=False)
    dfw.columns = dfw.columns.str.replace('\ufeff','', regex=False)
    if 'Zeitstempel' not in dfw.columns:
        print(f"   ❌ Keine Zeitstempel-Spalte in {fn}")
        continue
    dfw['Zeitstempel'] = pd.to_datetime(dfw['Zeitstempel'], infer_datetime_format=True)
    
    dfw = (
        dfw[['Zeitstempel','Wert']]
          .drop_duplicates('Zeitstempel')
          .set_index('Zeitstempel')
          .resample('5T')
          .interpolate(method='time')
    )
    dfw.rename(columns={'Wert': short}, inplace=True)
    print(f"   ✓ {short}: {dfw[short].notna().sum()}/{len(dfw)} Werte")
    weather_series.append(dfw[short])

# 5) Zusammenfügen aller Wetter-Serien
if not weather_series:
    raise RuntimeError("Keine Wetter-Daten geladen!")
weather_resampled = pd.concat(weather_series, axis=1).reset_index()

# 6) As-of-Merge mit Parkhaus-Daten (Toleranz 5 Min)
wide.sort_values('timestamp', inplace=True)
weather_resampled.sort_values('Zeitstempel', inplace=True)
merged = pd.merge_asof(
    wide,
    weather_resampled,
    left_on='timestamp',
    right_on='Zeitstempel',
    direction='nearest',
    tolerance=pd.Timedelta('5min')
)

# 7) Aufräumen & speichern
merged.drop(columns=['Zeitstempel'], inplace=True)
merged.to_csv(output_csv, index=False)
print(f"✅ Fertig! Datei liegt nun bei:\n   {output_csv}")
