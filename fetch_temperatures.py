#!/usr/bin/env python3
"""
Henter vejtemperaturer fra Trafikkort, konverterer til WGS84,
og gemmer data i tre CSV-filer:
- vej_temp_1.csv (første 500)
- vej_temp_2.csv (resten)
- vejtemp_udvalgte.csv (45 faste repræsentative punkter)
"""

from __future__ import annotations
import requests
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import KMeans
import numpy as np
import json
import os

URL = "https://storage.googleapis.com/trafikkort-data/geojson/25832/temperatures.point.json"
SELECTED_IDS_FILE = "selected_ids.json"

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

FIXED_IDS = [
    27,28,76,103,118,148,150,177,178,187,195,209,215,225,237,
    253,259,273,274,286,329,363,370,381,396,423,433,440,454,
    457,464,498,508,523,526,530,542,547,561,570,577,584,615,
    618,658
]

def pick_representative_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Brug de 45 faste stationer (FIXED_IDS) til vejtemp_udvalgte.csv.
    """
    df_sorted = df.sort_values("ID").reset_index(drop=True)
    df_selected = df_sorted[df_sorted["ID"].isin(FIXED_IDS)].sort_values("ID")
    return df_selected


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

    # Udvælg 45 faste repræsentative punkter
    df_selected = pick_representative_points(df)
    df_selected.to_csv("vejtemp_udvalgte.csv", index=False)
    print(f"Gemte {len(df_selected)} rækker i vejtemp_udvalgte.csv (faste repræsentative punkter)")


if __name__ == "__main__":
    main()
