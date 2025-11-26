#!/usr/bin/env python3
"""
Henter vejtemperaturer fra Trafikkort (Vejdirektoratet), konverterer koordinater til WGS84,
og gemmer CSV direkte i repo.
"""
from __future__ import annotations
import json
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
    for feat in geojson.get("features", []):
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        coords = geom.get("coordinates") if geom else None
        lon, lat = (None, None)
        if coords and len(coords) >= 2:
            lon, lat = coords[0], coords[1]
            lon, lat = transformer.transform(lon, lat)  # Konverter til WGS84

        rows.append({
            "lat": lat,
            "lon": lon,
            "road_temperature": props.get("roadSurfaceTemperature"),
            "air_temperature": props.get("airTemperature"),
        })
    return rows

def main():
    geojson = fetch_geojson(URL)
    rows = parse_features(geojson)
    df = pd.DataFrame(rows)
    df.to_csv("temperatures.csv", index=False)
    print(f"Gemte {len(df)} rækker til temperatures.csv")

if __name__ == "__main__":
    main()
