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

FIXED_IDS = [213, 485, 143, 540, 237, 46, 415, 521, 157, 504, 191, 245, 509, 238, 577, 18, 590, 262, 565, 351, 480, 182, 468, 429, 515, 235, 631, 194, 502, 418]

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
