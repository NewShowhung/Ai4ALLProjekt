import pandas as pd
import numpy as np

# 1. CSV-Datei laden (Windows-Pfad anpassen)
input_path = r'parkhaeuser_wetter_merged_allvars.csv'
df = pd.read_csv(input_path, low_memory=False)

# 2. Alle *_preise-Spalten identifizieren und löschen
preis_spalten = [col for col in df.columns if col.lower().endswith('_preise')]
df = df.drop(columns=preis_spalten)

# 3. Manuelle Zuordnung von Parkhausnamen zu den extrahierten numerischen Preisen
#    Format: "Parkhaus": (Preis_pro_Stunde, Max_Tagespreis)
price_mapping = {
    "Alsterhaus":                    (4.00,  28.00),
    "Alter Wall / BUCERIUS-Passage": (3.90,  32.00),
    "Am Hauptbahnhof":               (4.00,  20.00),
    "Bahnhof Altona":                (1.30,  12.00),
    "Bavaria Office":                (1.50,  11.00),
    "Berliner Tor":                  (2.80,  18.00),
    "City-Parkhaus":                 (3.30,  20.00),
    "Deichtorhallen":                (2.00,  20.00),
    "Deutsch Japanisches Zentrum":    (3.00,  22.00),
    "EKZ Nedderfeld":                (1.50,  10.00),
    "Elbarkaden":                    (2.50,  12.00),
    "Elbphilharmonie":               (5.00,  90.00),
    "Europa-Passage":                (2.50,  16.00),
    "Facharztklinik":                (5.00,  20.00),
    "Falkenried":                    (2.50,  16.00),
    "Fleethof":                      (3.00,  18.00),
    "Friedrich-Ebert-Damm":          (1.50,   9.00),
    "Große Reichenstraße":           (4.00,  20.00),
    "Gänsemarkt":                    (4.50,  30.00),
    "Hafentor":                      (4.00,  20.00),
    "Hahnenkamp":                    (1.33,  12.00),
    "Hamburger Meile":               (1.00,   8.00),
    "Hansa- Theater":                (2.00,  10.00),
    "Hanse-Viertel":                 (3.50,  28.00),
    "Hanseatic Trade Center":        (None,  None),
    "Holzdamm / Ibis":               (3.00,  29.00),
    "Holzhafen":                     (2.50,  20.00),
    "Hühnerposten":                  (None,  14.00),
    "Karstadt Eimsbüttel":           (2.00,   4.00),
    "Karstadt Wandsbek Parkhaus 1":  (1.20,   4.50),
    "Karstadt Wandsbek Parkhaus 2":  (1.20,   4.50),
    "Kaufhof (Galeria Kaufhof)":     (3.00,  None),
    "Kunsthalle":                    (4.00,  30.00),
    "Madison":                       (2.50,  32.00),
    "Marie-Jonas-Platz":             (3.00,  20.00),
    "Marriott":                      (4.50,  32.00),
    "Mercado (Einkaufszentrum)":     (1.00,   8.00),
    "Messe-Mitte":                   (2.00,  16.00),
    "Messe-Ost":                     (2.00,  16.00),
    "Michel-Garage":                 (4.00,  20.00),
    "Millerntor":                    (3.00,  15.00),
    "Mundsburg Center":              (1.00,   3.00),
    "Neue Flora":                    (2.50,  14.00),
    "Neues Forum":                   (1.50,   3.00),
    "Neues Steintor":                (1.30,  13.00),
    "Nomis Quartier":                (5.60,  25.00),
    "Othmarschen Park":              (2.00,  None),
    "Ottensen":                      (0.80,   6.00),
    "Parkhaus Stadthöfe (Bleichenhof)": (3.90,  32.00),
    "Parkplatz Cruise Center":       (2.00,  10.00),
    "Radison Blu":                   (3.00,  25.00),
    "Reeperbahn-Garagen (Spielbudenplatz)": (3.00,  15.00),
    "Rindermarkthalle":              (2.50,  50.00),
    "Rödingsmarkt":                  (3.00,  20.00),
    "Saturn":                        (3.00,  None),
    "Schillerstraße (CCA)":          (1.00,  15.00),
    "Speicherstadt":                 (2.50,  15.00),
    "StadtKontor Parking":           (1.00,  12.00),
    "Stadtlagerhaus":                (None,  None),
    "Tanzende Türme":                (4.00,  22.00),
    "Uniklinik Eppendorf":           (2.50,  18.00),
    "W 1":                           (1.20,   4.50),
    "Winterhuder Markt":             (1.50,   4.50),
    "Überseequartier Nord":          (2.50,  15.00),
}

# 4. Neue Spalten für numerische Preise hinzufügen
for parkhaus, (stundenpreis, tagespreis) in price_mapping.items():
    col_stunde = f"{parkhaus}_preis_pro_stunde"
    col_tag    = f"{parkhaus}_max_tagespreis"

    df[col_stunde] = stundenpreis if stundenpreis is not None else np.nan
    df[col_tag]    = tagespreis if tagespreis is not None else np.nan

# 5. Neue CSV-Datei abspeichern (ebenfalls in C:\Users\betti\OneDrive\py)
output_path = r'parkhaeuser_wetter_merged_allvars_mit_preisen.csv'
df.to_csv(output_path, index=False)

print(f"Neue Datei wurde gespeichert unter:\n{output_path}")
