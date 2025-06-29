import os
import glob
import pandas as pd

# 1. Pfade anpassen
SNAPSHOT_FOLDER = r"snapshots"
PICKLE_PATH     = r"parkhaeuser_timeseries.pkl"
CSV_PATH        = r"parkhaeuser_timeseries_wide.csv"


def extract_timestamp_from_filename(filepath: str) -> pd.Timestamp:
    """
    Erwartet Dateinamen wie 'parkhaeuser_20250508T194739Z.csv'
    und gibt den Timestamp als pandas.Timestamp zurück.
    """
    fname = os.path.basename(filepath)
    ts_str = fname.split('_', 1)[1].rsplit('.', 1)[0]
    return pd.to_datetime(ts_str, format='%Y%m%dT%H%M%SZ')


def process_file(filepath: str) -> pd.DataFrame:
    """
    Liest eine CSV, entfernt doppelte 'name'-Einträge (falls vorhanden),
    stacked sie zu einer 1-Zeilen-DataFrame mit MultiIndex-Spalten
    (parkhaus, feld) und dem Timestamp als Index.
    """
    ts = extract_timestamp_from_filename(filepath)
    df = pd.read_csv(filepath)

    # doppelte Parkhaus-Namen sicher entfernen
    df = df.drop_duplicates(subset=['name'], keep='first')

    # stack → Series mit MultiIndex (parkhaus,feld)
    s = df.set_index('name').stack()
    s.index.set_names(['parkhaus', 'feld'], inplace=True)

    # in 1-Zeilen-DF transformieren
    row = s.to_frame().T
    row.index = pd.DatetimeIndex([ts])
    return row


def build_timeseries():
    # 1) Alle Snapshots auflisten
    pattern = os.path.join(SNAPSHOT_FOLDER, '*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        print("❗️ Keine CSV-Dateien im Snapshot-Ordner gefunden.")
        return

    # 2) Vorhandene Zeitreihe laden oder leeren DF anlegen
    if os.path.exists(PICKLE_PATH):
        df_all = pd.read_pickle(PICKLE_PATH)
        processed_ts = set(df_all.index)
        mode = 'Folgelauf'
    else:
        df_all = pd.DataFrame()
        processed_ts = set()
        mode = 'Erstlauf'

    # 3) Nur neue Dateien verarbeiten
    new_files = [
        f for f in files
        if extract_timestamp_from_filename(f) not in processed_ts
    ]
    if not new_files:
        print("✅ Keine neuen Snapshots gefunden.")
        return df_all

    # 4) Jede neue Datei einlesen und in Liste sammeln
    rows = []
    for f in new_files:
        try:
            rows.append(process_file(f))
        except Exception as e:
            print(f"⚠️ Fehler bei {os.path.basename(f)}: {e}")

    if not rows:
        print("❗️ Aus den neuen Dateien konnten keine Zeilen erzeugt werden.")
        return df_all

    # 5) Neue Zeilen zusammenführen
    df_new = pd.concat(rows)

    # 6) In das Gesamt-DF integrieren
    if df_all.empty:
        df_all = df_new
    else:
        df_all = pd.concat([df_all, df_new])

    # 7) Index und Spalten sortieren
    df_all.sort_index(inplace=True)
    df_all.sort_index(axis=1, level=[0,1], inplace=True)

    # 8) Spalten flach und Tag/Zeit
    # MultiIndex-Spalten flat: 'Parkhaus_Feld'
    df_all.columns = [f"{p}_{f}" for p, f in df_all.columns]
    # Tag und Zeit als separate Spalten vorne einfügen
    df_all.insert(0, 'Tag', df_all.index.date)
    df_all.insert(1, 'Zeit', df_all.index.time)

    # 9) Speichern
    df_all.to_pickle(PICKLE_PATH)
    df_all.to_csv(CSV_PATH, index=True)

    print(f"🔄 {mode}: {len(new_files)} neue Snapshots verarbeitet und angehängt.")
    return df_all


if __name__ == '__main__':
    build_timeseries()
