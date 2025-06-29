import pandas as pd
import folium
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

# Daten laden und Merkmale aufbauen
df = pd.read_csv('parkhaeuser_wetter_merged_allvars_mit_preisen.csv')

# Parkhäuser identifizieren
price_cols = [col for col in df.columns if col.endswith('_preis_pro_stunde')]
garages = [col.replace('_preis_pro_stunde', '') for col in price_cols]

data = []
for garage in garages:
    cols = [f"{garage}_preis_pro_stunde", f"{garage}_gesamt", f"{garage}_frei", f"{garage}_lat", f"{garage}_lon"]
    if all(col in df.columns for col in cols):
        price = pd.to_numeric(df[f"{garage}_preis_pro_stunde"].iloc[0], errors='coerce')
        capacity = pd.to_numeric(df[f"{garage}_gesamt"], errors='coerce')
        free = pd.to_numeric(df[f"{garage}_frei"], errors='coerce')
        occupied = capacity - free
        mean_occ = occupied.mean()
        lat = pd.to_numeric(df[f"{garage}_lat"].iloc[0], errors='coerce')
        lon = pd.to_numeric(df[f"{garage}_lon"].iloc[0], errors='coerce')
        data.append([garage, price, mean_occ, lat, lon])

feat_df = pd.DataFrame(data, columns=['Garage', 'Preis', 'Besetzt', 'Lat', 'Lon']).set_index('Garage')

# Daten skalieren und Clustering
scaler = StandardScaler()
scaled = scaler.fit_transform(feat_df[['Preis', 'Besetzt', 'Lat', 'Lon']])
Z = linkage(scaled, method='ward')
clusters = fcluster(Z, t=3, criterion='maxclust')

feat_df['Cluster'] = clusters

# Interaktive Karte erstellen
center = [feat_df['Lat'].mean(), feat_df['Lon'].mean()]
m = folium.Map(location=center, zoom_start=13)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']
for (garage, row), cl in zip(feat_df.iterrows(), clusters):
    folium.CircleMarker(
        location=[row['Lat'], row['Lon']],
        radius=6,
        popup=f"{garage}\nPreis: €{row['Preis']:.2f}\nBelegt: {row['Besetzt']:.0f}\nCluster: {cl}",
        color=colors[(cl-1) % len(colors)],
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

# Karte anzeigen
m
