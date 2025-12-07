#!/usr/bin/env python3
"""
Henter vejtemperaturer fra Trafikkort og matcher med DMI dugpunkt via NÆRMESTE PUNKT.
- Kræver at DMI har koordinater i deres output (hvilket MetObs endpoint gør).
"""

from __future__ import annotations
import requests
import pandas as pd
from pyproj import Transformer
import numpy as np
import json
import time
import datetime
import geopandas as gpd
from shapely.geometry import Point
from typing import Optional, Any

# ---------------------------
# KONFIGURATION
# ---------------------------
URL_VEJTEMP = "https://storage.googleapis.com/trafikkort-data/geojson/25832/temperatures.point.json"
DMI_BASE = "https://opendataapi.dmi.dk/v2/metObs/collections/observation/items"
DMI_DATETIME_WINDOW = "now-PT60M/now" 

# EPSG:25832 -> WGS84
transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

SELECTED_IDS = [284, 237, 203, 145, 216, 214, 509, 226, 642, 188, 631, 213, 494, 361, 395, 40, 208, 584, 608, 565, 42, 46, 207, 590, 545, 540]

# ---------------------------
# DMI HENTNING (OPPDATERET TIL ALLE STATIONER)
# ---------------------------

def fetch_all_dmi_dewpoints(parameter_id: str) -> pd.DataFrame:
    """
    Henter dugpunkt for ALLE tilgængelige DMI MetObs stationer og returnerer en GeoDataFrame.
    """
    params = {
        "parameterId": parameter_id,
        "datetime": DMI_DATETIME_WINDOW,
    }
    
    try:
        r = requests.get(DMI_BASE, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Fejl ved hentning af ALLE DMI data: {e}")
        return gpd.GeoDataFrame()

    dmi_rows = []
    
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        
        # Geometri er typisk i WGS84
        lon, lat = geom.get("coordinates") if geom and geom.get("coordinates") else (None, None)
        val = props.get("value")
        
        if lat is not None and lon is not None and val is not None:
            dmi_rows.append({
                "Dewpoint": float(val),
                "Latitude": lat,
                "Longitude": lon,
                "geometry": Point(lon, lat)
            })

    # Opret GeoDataFrame fra DMI data
    dmi_gdf = gpd.GeoDataFrame(dmi_rows, crs="EPSG:4326")
    
    # Fjern duplikater for at sikre én observation pr. station (baseret på unikke koordinater)
    dmi_gdf = dmi_gdf.drop_duplicates(subset=["Latitude", "Longitude"], keep='first')
    
    print(f"Hentet {len(dmi_gdf)} unikke DMI dugpunkt observationer.")
    return dmi_gdf

# ---------------------------
# VEJTEMP HENTNING & PARSING (uændret)
# ---------------------------

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

        # StationID er ikke længere nødvendig til matching
        station_id = props.get("device_id") 

        rows.append({
            "ID": id_counter,
            "StationID": station_id if station_id else f"Vejtemp_{id_counter}",
            "Latitude": lat,
            "Longitude": lon,
            "Vej_temp": props.get("roadSurfaceTemperature"),
            "Luft_temp": props.get("airTemperature"),
        })
        id_counter += 1
    return rows

def pick_representative_points(df: pd.DataFrame) -> pd.DataFrame:
    """ Bruger den faste liste SELECTED_IDS til at vælge repræsentative stationer. """
    df_selected = df[df["ID"].isin(SELECTED_IDS)].sort_values("ID")
    return df_selected


def main():
    geojson = fetch_geojson(URL_VEJTEMP)
    df = pd.DataFrame(parse_features(geojson))

    # --- Hent DMI data for alle stationer ---
    dmi_gdf = fetch_all_dmi_dewpoints("temp_dew")
    
    if dmi_gdf.empty:
        print("ADVARSEL: Kunne ikke hente DMI dugpunkt data. Fortsætter med Luft_temp som fallback.")
        df["Dewpoint"] = df["Luft_temp"]
    else:
        # --- Matche dugpunkt til vejstationer (Nærmeste Nabo) ---
        
        # 1. Konverter vejtemp DF til GeoDataFrame
        vejtemp_gdf = gpd.GeoDataFrame(
            df.copy(), 
            geometry=gpd.points_from_xy(df.Longitude, df.Latitude), 
            crs="EPSG:4326"
        )
        
        # 2. Match Nærmeste Nabo (DMI -> Vejtemp)
        # Dette udfører en spatial join for at finde den nærmeste DMI-station til hver vejstation.
        # Vi matcher i WGS84 for at holde det simpelt, men en projiceret CRS ville være mere præcis.
        
        # Find index for nærmeste DMI station til hver vejtemp station
        nearest_dmi_idx = vejtemp_gdf.geometry.apply(lambda point: dmi_gdf.geometry.distance(point).idxmin())
        
        # Kopier dugpunktsværdien over
        df["Dewpoint"] = dmi_gdf.loc[nearest_dmi_idx, "Dewpoint"].values


    # --- Gem opdaterede CSV-filer ---
    
    # Fjern 'Precip' kolonnen, hvis den eksisterer
    if "Precip" in df.columns:
        df = df.drop(columns=["Precip"])
    
    df_1 = df.iloc[:500]
    df_2 = df.iloc[500:]

    df_1.to_csv("vej_temp_1.csv", index=False)
    df_2.to_csv("vej_temp_2.csv", index=False)

    print(f"Gemte {len(df_1)} rækker i vej_temp_1.csv (inkl. geografisk matchet dugpunkt)")
    print(f"Gemte {len(df_2)} rækker i vej_temp_2.csv (inkl. geografisk matchet dugpunkt)")

    df_selected = pick_representative_points(df)
    df_selected.to_csv("vejtemp_udvalgte.csv", index=False)
    print(f"Gemte {len(df_selected)} rækker i vejtemp_udvalgte.csv (manuelt udvalgte punkter, inkl. dugpunkt)")


if __name__ == "__main__":
    # For at dette script kan køre, skal du sikre, at GeoPandas, Shapely og Requests er installeret.
    main()
