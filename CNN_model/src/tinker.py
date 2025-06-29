import pandas as pd

DATA_PATH = "data/data.csv"
COLUMN_INDEX_END_ALSTERHAUS = 12 # letzte Spalte mit Daten zum Parkhaus "Alsterhaus"
COLUMN_INDEX_BEGIN_WEATHER = 539 # erste Spalte mit Wetterdaten

df = pd.read_csv(DATA_PATH)

print

for col in df.columns:
    print(col)

print(df.columns[len(df.columns) - 19])
print(len(df.columns) - 19)