#!/usr/bin/env python3
"""
Henter vejtemperaturer fra Trafikkort og matcher med DMI dugpunkt via NÆRMESTE PUNKT.
- Udvalgte stationer hentes nu via matchende STABILE koordinater fra ekstern CSV
  ved hjælp af robust SciPy/NumPy matching (løser GeoPandas ValueError).
"""

from __future__ import annotations
import requests
import pandas as pd
from pyproj import Transformer
import numpy as np
from scipy.spatial.distance import cdist # Nyt, stabilt matching værktøj
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
            "ID": id_counter,
            "NAME": str(id_counter),
            "Latitude": lat,
            "Longitude": lon,
            "StationID": str(station_id) if station_id is not None else f"Vejtemp_{id_counter}",
            "Vej_temp": props.get("roadSurfaceTemperature"),
            "Luft_temp": props.get("airTemperature"),
        })
        id_counter += 1
    return rows

def pick_representative_points(df_api: pd.DataFrame) -> pd.DataFrame:
    """ 
    Udvælger repræsentative stationer ved at matche API-data mod
    STABILE koordinater fra filen ved hjælp af SciPy/NumPy Cdist.
    """
    try:
        # Læs den stabile koordinatfil
        df_stable = pd.read_csv(STABLE_COORDS_FILE)
    except FileNotFoundError:
        print(f"\nFATAL FEJL: Kunne ikke finde den stabile koordinatfil '{STABLE_COORDS_FILE}'. Sørg for at den ligger i dit repo.")
        return pd.DataFrame()

    # 1. Forbered koordinater som NumPy arrays
    stable_coords = df_stable[['LATITUDE', 'LONGITUDE']].values
    api_coords = df_api[['Latitude', 'Longitude']].values

    # 2. Beregn afstandsmatrix (Euclidean distance on Lat/Lon)
    # Dette er den robuste erstatning for GeoPandas apply()
    distances = cdist(stable_coords, api_coords, metric='euclidean')
    
    # 3. Find indexen i API-dataen for det tætteste match
    # indices: index i df_api for den nærmeste station til hver stabil koordinat
    indices = np.argmin(distances, axis=1)

    # 4. Tjek for tæt match (Validering af at stationen faktisk er der)
    min_distances = np.min(distances, axis=1)
    MAX_DISTANCE_TOLERANCE = 0.0001 # ~10-15 meter
    
    # Filtrer indices, hvor matchet var tæt nok
    acceptable_indices = indices[min_distances < MAX_DISTANCE_TOLERANCE]
    
    # Sørg for unikke stationer
    selected_indices = np.unique(acceptable_indices)

    # 5. Udtræk de matchede rækker fra API-dataen
    df_selected = df_api.iloc[selected_indices].copy()
        
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
        
    # Sørg for at kolonnerne er i den ønskede rækkefølge til WSI Max
    cols = ["ID", "NAME", "Latitude", "Longitude", "StationID", "Vej_temp", "Luft_temp", "Dewpoint"]
    
    cols_to_use = [c for c in cols if c in df.columns]
    df = df[cols_to_use]

    df_1 = df.iloc[:500].copy()
    df_2 = df.iloc[500:].copy()

    df_1.to_csv("vej_temp_1.csv", index=False)
    df_2.to_csv("vej_temp_2.csv", index=False)

    print(f"Gemte {len(df_1)} rækker i vej_temp_1.csv (Korrekt kolonneorden for WSI Max)")
    print(f"Gemte {len(df_2)} rækker i vej_temp_2.csv (Korrekt kolonneorden for WSI Max)")

    # >>> HER BRUGES DEN NYE, STABILE OG ROBUSTE MATCHING <<<
    df_selected = pick_representative_points(df)
    
    if not df_selected.empty:
        df_selected.to_csv("vejtemp_udvalgte.csv", index=False)
        print(f"Gemte {len(df_selected)} rækker i vejtemp_udvalgte.csv (Stabil udvælgelse via koordinater)")


if __name__ == "__main__":
    main()
