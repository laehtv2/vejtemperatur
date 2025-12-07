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

# VIGTIGT: Denne liste SKAL indeholde de STABILE StationID'er (device_id) 
# fra API'en, IKKE de dynamiske ID'er (1, 2, 3, ...).
# Du skal køre koden ÉN gang for at finde de korrekte StationID'er (se Trin 2).
SELECTED_STATION_IDS: list[str] = [
    # Indsæt de 30 stabile StationID'er her efter du har fundet dem.
] 

# ---------------------------
# DMI HENTNING (Uændret)
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
# VEJTEMP HENTNING & PARSING (Rettet til at gemme StationID)
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

        station_id = props.get("device_id") # Dette er den STABILE ID!

        rows.append({
            "ID": id_counter,
            "StationID": station_id if station_id else f"Vejtemp_{id_counter}", # Brug StationID til udvælgelse
            "Latitude": lat,
            "Longitude": lon,
            "Vej_temp": props.get("roadSurfaceTemperature"),
            "Luft_temp": props.get("airTemperature"),
        })
        id_counter += 1
    return rows

def pick_representative_points(df: pd.DataFrame) -> pd.DataFrame:
    """ Bruger den faste liste SELECTED_STATION_IDS til at vælge repræsentative stationer. """
    if not SELECTED_STATION_IDS:
        print("\nADVARSEL: SELECTED_STATION_IDS er tom. Returnerer alle stationer. Find de stabile ID'er først.")
        return pd.DataFrame() # Retur tom DF, hvis ID'er mangler
        
    # Vælger nu baseret på StationID, som er den STABILE ID.
    df_selected = df[df["StationID"].astype(str).isin(map(str, SELECTED_STATION_IDS))].sort_values("StationID")
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
        nearest_dmi_idx = vejtemp_gdf.geometry.apply(lambda point: dmi_gdf.geometry.distance(point).idxmin())
        
        # Kopier dugpunktsværdien over
        df["Dewpoint"] = dmi_gdf.loc[nearest_dmi_idx, "Dewpoint"].values


    # --- Gem opdaterede CSV-filer ---
    
    # Fjern 'Precip' kolonnen, hvis den eksisterer
    if "Precip" in df.columns:
        df = df.drop(columns=["Precip"])
        
    # VIGTIGT: Gem StationID i CSV'erne
    df_save = df.drop(columns=["ID", "NAME"]) # Drop de ustabile kolonner
    df_1 = df_save.iloc[:500]
    df_2 = df_save.iloc[500:]

    df_1.to_csv("vej_temp_1.csv", index=False)
    df_2.to_csv("vej_temp_2.csv", index=False)

    print(f"Gemte {len(df_1)} rækker i vej_temp_1.csv (inkl. geografisk matchet dugpunkt og STABIL StationID)")
    print(f"Gemte {len(df_2)} rækker i vej_temp_2.csv (inkl. geografisk matchet dugpunkt og STABIL StationID)")

    df_selected = pick_representative_points(df)
    if not df_selected.empty:
        df_selected_final = df_selected.drop(columns=["ID", "NAME"]) # Drop de ustabile kolonner
        df_selected_final.to_csv("vejtemp_udvalgte.csv", index=False)
        print(f"Gemte {len(df_selected_final)} rækker i vejtemp_udvalgte.csv (stabil udvælgelse)")


if __name__ == "__main__":
    # For at dette script kan køre, skal du sikre, at GeoPandas, Shapely og Requests er installeret.
    main()
