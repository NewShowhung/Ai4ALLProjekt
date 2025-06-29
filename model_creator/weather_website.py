#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_parkhaus_clusters.py

Erstellt eine interaktive Karte der Parkhäuser mit Cluster-Färbung und Legende.

Voraussetzungen:
    pip install pandas scikit-learn folium

Nutzung:
    python3 plot_parkhaus_clusters.py
    -> erzeugt 'parkhaus_clusters_map.html' im aktuellen Verzeichnis
    -> im Browser öffnen
"""

import pandas as pd
import folium
from sklearn.cluster import AgglomerativeClustering

def main():
    # 1) Feature-Importances laden und Cluster berechnen
    feats = pd.read_csv('feature_importances_per_garage.csv', index_col=0)
    clust = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    feats['Cluster'] = clust.fit_predict(feats.values)

    # 2) Koordinaten-Tabelle laden (mit Preisen oder ohne, je nach Datei)
    df_coord = pd.read_csv('parkhaeuser_wetter_merged_allvars_mit_preisen.csv')
    coords = {}
    for col in df_coord.columns:
        if col.endswith('_lat'):
            name = col[:-4]
            lon_col = name + '_lon'
            if lon_col in df_coord.columns:
                coords[name] = (df_coord[col].iloc[0], df_coord[lon_col].iloc[0])

    # 3) DataFrame für Plot zusammenbauen
    rows = []
    for name, row in feats.iterrows():
        if name in coords:
            lat, lon = coords[name]
            rows.append({
                'Garage': name,
                'Latitude': lat,
                'Longitude': lon,
                'Cluster': int(row['Cluster'])
            })
    plot_df = pd.DataFrame(rows)

    # 4) Farbzuordnung
    colors = {
        0: 'red',     # Cluster 0: Temperatur & Feuchtigkeit
        1: 'blue',    # Cluster 1: Sonneneinstrahlung
        2: 'green',   # Cluster 2: Druck & Strahlung
        3: 'purple',  # Cluster 3: Kein Wetter-Einfluss
        4: 'orange'   # Cluster 4: Druck-gesteuert
    }

    # 5) Karte erstellen
    m = folium.Map(location=[
        plot_df['Latitude'].mean(),
        plot_df['Longitude'].mean()
    ], zoom_start=32)

    # Marker setzen
    for _, r in plot_df.iterrows():
        folium.CircleMarker(
            location=(r['Latitude'], r['Longitude']),
            radius=6,
            color=colors[r['Cluster']],
            fill=True,
            fill_color=colors[r['Cluster']],
            fill_opacity=0.7,
            popup=r['Garage']
        ).add_to(m)

    # 6) Legende einfügen
    legend = """
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 200px; height: 160px;
        border:2px solid grey; background-color:white; z-index:9999;
        font-size:14px; padding:10px;
    ">
      <b>Cluster-Legende</b><br>
      <i style="color:red;">&#9679;</i> Cluster 0: Temperatur &amp; Feuchtigkeit<br>
      <i style="color:blue;">&#9679;</i> Cluster 1: Sonneneinstrahlung<br>
      <i style="color:green;">&#9679;</i> Cluster 2: Druck &amp; Strahlung<br>
      <i style="color:purple;">&#9679;</i> Cluster 3: Kein Wetter-Einfluss<br>
      <i style="color:orange;">&#9679;</i> Cluster 4: Druck-gesteuert
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    # 7) Speichern
    out = 'parkhaus_clusters_map.html'
    m.save(out)
    print(f"Karte erstellt: {out}")

if __name__ == '__main__':
    main()
