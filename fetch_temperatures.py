#!/usr/bin/env python3
"""
Henter vejtemperaturer fra Trafikkort (Vejdirektoratet), konverterer koordinater til WGS84,
og gemmer data i tre CSV-filer:
- vej_temp_1.csv (første 500)
- vej_temp_2.csv (resten)
- vejtemp_udvalgte.csv (45 repræsentative punkter fordelt over DK)
"""

from __future__ import annotations
import requests
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import KMeans
import numpy as np

URL = "https://storage.googleapis.com/trafikkort-data/geojson/25832/temperatures.point.json"

# EPSG:25832 -> WGS84
transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

def fetch_geojson(url: str) -> dict:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

def parse_features(geojson: dict) -> list[dict]:
    rows = []
    id_counter = 1

    for feat in geojson.get("features", []):
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        coords = geom.get("coordinates") if geom else None

        lon, lat = (None, None)
        if coords and len(coords) >= 2:
            lon, lat = transformer.transform(coords[0], coords[1])

        rows.append({
            "ID": id_counter,
            "NAME": id_counter,
            "Latitude": lat,
            "Longitude": lon,
            "Vej_temp": props.get("roadSurfaceTemperature"),
            "Luft_temp": props.get("airTemperature"),
        })

        id_counter += 1

    return rows

def pick_representative_points(df: pd.DataFrame, k: int = 45) -> pd.DataFrame:
    """
    Udvælg k punkter fordelt over Danmark vha. k-means clustering.
    Vi finder cluster-centroid, og vælger det punkt der ligger tættest på centret.
    """

    coords = df[["Latitude", "Longitude"]].to_numpy()

    # KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_

    selected_indices = []

    for i in range(k):
        cluster_points = coords[labels == i]
        cluster_indices = df.index[labels == i]

        if len(cluster_points) == 0:
            continue

        # find punktet tættest på centroid
        center = centers[i]
        dists = np.linalg.norm(cluster_points - center, axis=1)
        nearest_idx = cluster_indices[np.argmin(dists)]
        selected_indices.append(nearest_idx)

    return df.loc[selected_indices].sort_values("ID")

def main():
    geojson = fetch_geojson(URL)
    rows = parse_features(geojson)
    df = pd.DataFrame(rows)

    # Split i to filer á maks 500
    df_1 = df.iloc[:500]
    df_2 = df.iloc[500:]

    df_1.to_csv("vej_temp_1.csv", index=False)
    df_2.to_csv("vej_temp_2.csv", index=False)

    print(f"Gemte {len(df_1)} rækker i vej_temp_1.csv")
    print(f"Gemte {len(df_2)} rækker i vej_temp_2.csv")

    # Udvælg 45 repræsentative punkter
    df_selected = pick_representative_points(df, k=45)
    df_selected.to_csv("vejtemp_udvalgte.csv", index=False)

    print(f"Gemte {len(df_selected)} repræsentative punkter i vejtemp_udvalgte.csv")

if __name__ == "__main__":
    main()
