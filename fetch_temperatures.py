#!/usr/bin/env python3
"""
Henter vejtemperaturer fra Trafikkort og matcher med DMI dugpunkt via NÆRMESTE PUNKT.
- Udvalgte stationer hentes nu via matchende STABILE koordinater fra ekstern CSV.
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

# Navn på filen med de stabile koordinater
STABLE_COORDS_FILE = "vejtemp_30_centralt_fordelte_stationer_STABIL.csv"

# EPSG:25832 -> WGS84
transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

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
# VEJTEMP HENTNING & PARSING
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

        station_id = props.get("device_id") 

        rows.append({
            "ID": id_counter, # Til outputfil: Ustabilt, men nødvendigt
            "NAME": str(id_counter), # Til outputfil: Ustabilt, men nødvendigt
            "Latitude": lat, # Til outputfil
            "Longitude": lon, # Til outputfil
            "StationID": str(station_id) if station_id is not None else f"Vejtemp_{id_counter}", # Den STABILE ID
            "Vej_temp": props.get("roadSurfaceTemperature"),
            "Luft_temp": props.get("airTemperature"),
        })
        id_counter += 1
    return rows

def pick_representative_points(df_api: pd.DataFrame) -> pd.DataFrame:
    """ 
    Udvælger repræsentative stationer ved at matche API-data mod
    STABILE koordinater fra filen.
    """
    try:
        # Læs den stabile koordinatfil
        df_stable = pd.read_csv(STABLE_COORDS_FILE)
        
        # Omdøb kolonner for at matche den stabile fil's struktur
        df_stable.rename(columns={'STABLE_ID': 'StationID', 
                                  'LATITUDE': 'Latitude_Stable', 
                                  'LONGITUDE': 'Longitude_Stable'}, inplace=True)
        
    except FileNotFoundError:
        print(f"\nFATAL FEJL: Kunne ikke finde den stabile koordinatfil '{STABLE_COORDS_FILE}'. Sørg for at den ligger i dit repo.")
        return pd.DataFrame()

    # 1. Konverter API-data (vejtemp) til GeoDataFrame
    vejtemp_gdf = gpd.GeoDataFrame(
        df_api.copy(), 
        geometry=gpd.points_from_xy(df_api.Longitude, df_api.Latitude), 
        crs="EPSG:4326"
    )

    # 2. Konverter de stabile referencekoordinater til GeoDataFrame
    stable_gdf = gpd.GeoDataFrame(
        df_stable.copy(), 
        geometry=gpd.points_from_xy(df_stable.Longitude_Stable, df_stable.Latitude_Stable), 
        crs="EPSG:4326"
    )

    # 3. Match Nærmeste Nabo (Stabil reference -> Vejtemp API)
    # Find index i vejtemp_gdf (API data) af den nærmeste station for HVER stabil reference
    nearest_api_idx = stable_gdf.geometry.apply(lambda stable_point: vejtemp_gdf.geometry.distance(stable_point).idxmin())

    # 4. Filter for at sikre, at matchet er præcist (tolerance)
    # Vi checker afstanden mellem det stabile punkt og det matchede punkt i API-dataen
    min_distance = stable_gdf.geometry.apply(lambda stable_point: vejtemp_gdf.geometry.distance(stable_point).min())
    
    # Tolerance på 0.0001 grader (~10-15 meter) sikrer, at vi kun matcher de korrekte stationer
    acceptable_matches = min_distance < 0.0001 

    # 5. Udvælg de matchede rækker fra API-dataen
    # Vi bruger kun de indices, hvor matchet var acceptabelt.
    selected_indices = nearest_api_idx[acceptable_matches].unique()

    df_selected = vejtemp_gdf.loc[selected_indices].copy()
    
    # Drop geometry column for final CSV output
    if 'geometry' in df_selected.columns:
        df_selected = df_selected.drop(columns=['geometry'])
        
    return df_selected.sort_values('StationID')


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
        
    # Sørg for at kolonnerne er i den ønskede rækkefølge til WSI Max: ID, NAME, Latitude, Longitude, ...
    cols = ["ID", "NAME", "Latitude", "Longitude", "StationID", "Vej_temp", "Luft_temp", "Dewpoint"]
    
    # Filtrer kolonner der findes i df (Dewpoint kan mangle ved DMI fejl)
    cols_to_use = [c for c in cols if c in df.columns]
    df = df[cols_to_use]

    df_1 = df.iloc[:500].copy()
    df_2 = df.iloc[500:].copy()

    df_1.to_csv("vej_temp_1.csv", index=False)
    df_2.to_csv("vej_temp_2.csv", index=False)

    print(f"Gemte {len(df_1)} rækker i vej_temp_1.csv (Korrekt kolonneorden for WSI Max)")
    print(f"Gemte {len(df_2)} rækker i vej_temp_2.csv (Korrekt kolonneorden for WSI Max)")

    # >>> HER BRUGES DEN NYE LOGIK <<<
    df_selected = pick_representative_points(df)
    
    if not df_selected.empty:
        df_selected.to_csv("vejtemp_udvalgte.csv", index=False)
        print(f"Gemte {len(df_selected)} rækker i vejtemp_udvalgte.csv (Stabil udvælgelse via koordinater)")


if __name__ == "__main__":
    main()
