#!/usr/bin/env python3
"""
fetch_parkhaeuser.py
====================

Dependencies
------------
    pip install geopandas requests

Run
---
    python fetch_parkhaeuser.py

Stop with Ctrl+C.
"""

import os
import time
from datetime import datetime
import geopandas as gpd

# --- Configuration ----------------------------------------------------------

WFS_URL = (
    "https://geodienste.hamburg.de/wfs_parkhaeuser?"
    "service=WFS&version=1.1.0&request=GetFeature&srsName=EPSG%3A25832&"
    "typeName=parkhaeuser&bbox=557357.6687469253,5929041.604135078,"
    "572597.6605173299,5939624.931753415,EPSG:25832"
)

SNAPSHOT_DIR = "snapshots"      # Sub‑folder to hold CSV files
INTERVAL_SEC = 5 * 60           # 5 minutes  (300 s)

# ---------------------------------------------------------------------------

def to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a copy of *gdf* transformed to EPSG:4326 (WGS‑84)."""
    if gdf.crs is None or gdf.crs.to_epsg() != 25832:
        gdf = gdf.set_crs(epsg=25832, inplace=False)
    return gdf.to_crs(epsg=4326)


def fetch_and_save() -> None:
    """Fetch data from the WFS endpoint and save to a CSV file."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    try:
        # --- Fetch ----------------------------------------------------------
        gdf = gpd.read_file(WFS_URL)

        # --- Transform to WGS‑84 -------------------------------------------
        gdf_wgs = to_wgs84(gdf)

        # --- Prepare columns for CSV ---------------------------------------
        # Original geometry in WKT (EPSG:25832)
        gdf["geometry_25832"] = gdf.geometry.apply(lambda g: g.wkt if g is not None else None)

        # WGS‑84 geometry in WKT
        gdf["geometry_wgs84"] = gdf_wgs.geometry.apply(lambda g: g.wkt if g is not None else None)

        # Latitude / Longitude columns (useful for Google Maps)
        gdf["lat"] = gdf_wgs.geometry.y
        gdf["lon"] = gdf_wgs.geometry.x

        # Remove the shapely geometry column to keep CSV tidy
        gdf = gdf.drop(columns=["geometry"])

        # --- Write snapshot -------------------------------------------------
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        outfile = os.path.join(SNAPSHOT_DIR, f"parkhaeuser_{timestamp}.csv")
        gdf.to_csv(outfile, index=False)
        print(f"[{timestamp}] Saved {len(gdf)} records → {outfile}")
    except Exception as exc:
        print(f"[{timestamp}] ERROR: {exc}")


def main() -> None:
    print("Starting 5‑minute polling of Hamburg parking WFS (Ctrl+C to stop)…")
    while True:
        fetch_and_save()
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
