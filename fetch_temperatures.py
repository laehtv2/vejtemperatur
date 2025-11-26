#!/usr/bin/env python3
"""
Henter vejtemperaturer fra Trafikkort (Vejdirektoratet) GeoJSON, konverterer koordinater til WGS84, og gemmer som CSV.
"""
from __future__ import annotations
import argparse
import json
from typing import Any
import requests
import pandas as pd
from pyproj import Transformer

URL = (
    "https://storage.googleapis.com/trafikkort-data/geojson/25832/temperatures.point.json"
)

transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

def fetch_geojson(url: str) -> dict[str, Any]:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

def parse_features(geojson: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    features = geojson.get("features", [])

    for feat in features:
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}

        coords = geom.get("coordinates") if geom else None
        lon, lat = (None, None)
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            lon, lat = coords[0], coords[1]
            lon, lat = transformer.transform(lon, lat)  # Konverter til WGS84

        row = {
            "station_id": props.get("featureId"),
            "road_temperature": props.get("roadSurfaceTemperature"),
            "air_temperature": props.get("airTemperature"),
            "lon_wgs84": lon,
            "lat_wgs84": lat,
            "timestamp": props.get("updated") or props.get("timestamp"),
            "properties_json": json.dumps(props, ensure_ascii=False),
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Hent vejtemperaturer og gem som CSV.")
    parser.add_argument("--url", default=URL, help="GeoJSON endpoint (default: Trafikkort)")
    parser.add_argument("--output", default="temperatures.csv", help="Output CSV filnavn")
    args = parser.parse_args()

    print(f"Henter GeoJSON fra: {args.url}")
    geojson = fetch_geojson(args.url)
    rows = parse_features(geojson)

    if not rows:
        print("Ingen features fundet i GeoJSON. Tjek URL eller indhold.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Gemte {len(df)} rækker til {args.output}")

if __name__ == "__main__":
    main()
