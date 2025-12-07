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
from pandas.errors import EmptyDataError 

# ---------------------------
# KONFIGURATION
# ---------------------------
URL_VEJTEMP = "https://storage.googleapis.com/trafikkort-data/geojson/25832/temperatures.point.json"
DMI_BASE = "https://opendataapi.dmi.dk/v2/metObs/collections/observation/items"
DMI_DATETIME_WINDOW = "now-PT60M/now" 

# EPSG:25832 -> WGS84
transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

# VIGTIGT: Denne liste SKAL opdateres med de STABILE StationID'er (device_id)
# som matcher de 30 bedst fordelte koordinater.
# BEMÆRK: Disse ID'er er eksempler og skal erstattes efter Trin 3.
SELECTED_STATION_IDS: list[str] = [
    '5056', '7126', '6024', '1400', '6588', '7025', '6042', '6048', '6055', 
    '6445', '7140', '5998', '7162', '7192', '7170', '7188', '7154', '7116', 
    '5057', '6147', '7060', '7061', '6049', '5935', '5944', '5926', '5917', 
    '5937', '5951', '5977'
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
    except requests.exceptions.HTTPError as e:
        print(f"Fejl ved hentning af ALLE DMI data: {e}")
        return gpd.GeoDataFrame()
    except Exception as e:
        print(f"Generel fejl ved hentning af ALLE DMI data: {e}")
        return gpd.GeoDataFrame()

    dmi_rows = []
    
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        
        lon, lat = geom.get("coordinates") if geom and geom.get("coordinates") else (None, None)
        val = props.get("value")
        
        if lat is not None and lon is not None and val is not None:
            dmi_rows.append({
                "Dewpoint": float(val),
                "Latitude": lat,
                "Longitude": lon,
                "geometry": Point(lon, lat)
            })

    dmi_gdf = gpd.GeoDataFrame(dmi_rows, crs="EPSG:4326")
    dmi_gdf = dmi_gdf.drop_duplicates(subset=["Latitude", "Longitude"], keep='first')
    
    print(f"Hentet {len(dmi_gdf)} unikke DMI dugpunkt observationer.")
    return dmi_gdf

# ---------------------------
# VEJTEMP HENTNING & PARSING (Rettet: Bevarer ID og NAME for header-kompatibilitet)
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

        # VIGTIGT: device_id er den STABILE ID!
        station_id = props.get("device_id") 

        rows.append({
            "ID": id_counter, # Bevarer den ustabile ID for output-kompatibilitet
            "NAME": str(id_counter), # Laver et dummy NAME, der matcher det ustabile ID for output-kompatibilitet
            "StationID": str(station_id) if station_id is not None else f"Vejtemp_{id_counter}", # Den STABILE ID, der bruges til udvælgelse
            "Latitude": lat,
            "Longitude": lon,
            "Vej_temp": props.get("roadSurfaceTemperature"),
            "Luft_temp": props.get("airTemperature"),
        })
        id_counter += 1
    return rows

def pick_representative_points(df: pd.DataFrame) -> pd.DataFrame:
    """ Bruger den faste liste SELECTED_STATION_IDS til at vælge repræsentative stationer (baseret på StationID). """
    if not SELECTED_STATION_IDS:
        print("\nADVARSEL: SELECTED_STATION_IDS er tom. Fortsæt til Trin 3 for at finde de stabile ID'er.")
        return pd.DataFrame() 
        
    df_selected = df[df["StationID"].isin(map(str, SELECTED_STATION_IDS))].sort_values("StationID")
    return df_selected


def main():
    try:
        geojson = fetch_geojson(URL_VEJTEMP)
    except Exception as e:
        print(f"Fejl ved hentning af Vejtemp API: {e}. Kan ikke fortsætte.")
        return

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
        
    # Nu bevares ID og NAME, som ønsket.
    df_1 = df.iloc[:500].copy()
    df_2 = df.iloc[500:].copy()

    df_1.to_csv("vej_temp_1.csv", index=False)
    df_2.to_csv("vej_temp_2.csv", index=False)

    print(f"Gemte {len(df_1)} rækker i vej_temp_1.csv (bevarer ID, NAME, Latitude, Longitude)")
    print(f"Gemte {len(df_2)} rækker i vej_temp_2.csv (bevarer ID, NAME, Latitude, Longitude)")

    df_selected = pick_representative_points(df)
    if not df_selected.empty:
        df_selected.to_csv("vejtemp_udvalgte.csv", index=False)
        print(f"Gemte {len(df_selected)} rækker i vejtemp_udvalgte.csv (stabil udvælgelse via StationID)")


if __name__ == "__main__":
    main()
