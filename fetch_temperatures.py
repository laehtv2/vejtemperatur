#!/usr/bin/env python3
"""
Henter vejtemperaturer fra Trafikkort (Vejdirektoratet) GeoJSON og gemmer som CSV.

Brugsvejledning:
    python fetch_temperatures.py --output temperatures.csv

Scriptet bruger `requests` til at hente GeoJSON og `pandas` til at skrive CSV.
"""

from __future__ import annotations
import argparse
import json
from typing import Any
import requests
import pandas as pd
from datetime import datetime

URL = (
    "https://storage.googleapis.com/trafikkort-data/geojson/25832/temperatures.point.json"
)


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

        # Forsøg at finde et tydeligt id / temperatur / tid
        station_id = props.get("id") or props.get("stationId") or props.get("station_id")
        # temperaturfelter kan hedde temperature, temp, value etc.
        temperature = (
            props.get("temperature")
            if props.get("temperature") is not None
            else props.get("temp")
            if props.get("temp") is not None
            else props.get("value")
        )

        # Tid/updated
        timestamp = (
            props.get("updated")
            or props.get("timestamp")
            or props.get("time")
            or props.get("last_update")
            or props.get("lastUpdated")
        )

        # Normaliser timestamp hvis muligt
        if isinstance(timestamp, (int, float)):
            # epoch seconds?
            try:
                timestamp = datetime.utcfromtimestamp(int(timestamp)).isoformat() + "Z"
            except Exception:
                timestamp = str(timestamp)
        elif isinstance(timestamp, str):
            # Lad det være som-streng (antag ISO-format)
            timestamp = timestamp

        row = {
            "station_id": station_id,
            "temperature": temperature,
            "lon": lon,
            "lat": lat,
            "timestamp": timestamp,
            # Gem hele properties som JSON hvis du vil bruge senere
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
    # Sortér gerne for nemheds skyld
    if "timestamp" in df.columns:
        try:
            df["_ts_sort"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values("_ts_sort", ascending=False).drop(columns=["_ts_sort"])
        except Exception:
            pass

    df.to_csv(args.output, index=False)
    print(f"Gemte {len(df)} rækker til {args.output}")


if __name__ == "__main__":
    main()
