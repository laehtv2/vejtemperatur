#!/usr/bin/env python3
"""
Henter vejtemperaturer fra Trafikkort (Vejdirektoratet), konverterer koordinater til WGS84,
og gemmer data i to CSV-filer:
vej_temp_1.csv og vej_temp_2.csv
"""

from __future__ import annotations
import requests
import pandas as pd
from pyproj import Transformer

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
            lon, lat = transformer.transform(coords[0], coords[1])  # til WGS84

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

if __name__ == "__main__":
    main()
