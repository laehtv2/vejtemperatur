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

FIXED_IDS = [1, 3, 7, 11, 36, 38, 56, 93, 95, 103, 127, 137, 141, 144, 153, 154, 156, 163, 171, 181, 196, 198, 201, 204, 209, 213, 231, 235, 243, 244, 247, 252, 255, 258, 259, 262, 267, 270, 283, 288, 306, 314, 323, 328, 336, 340, 341, 345, 366, 371, 373, 379, 390, 396, 399, 404, 410, 417, 424, 425, 427, 431, 436, 440, 446, 451, 457, 467, 468, 474, 478, 484, 499, 500, 505, 507, 510, 514, 516, 519, 522, 523, 527, 535, 540, 560, 563, 565, 573, 574, 586, 593, 597, 603, 606, 612, 620, 630, 637, 643]

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
