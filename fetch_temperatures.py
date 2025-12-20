#!/usr/bin/env python3
"""
Henter vejtemperaturer fra Trafikkort.
Outputter 3 filer:
1. vej_temp_1.csv (Station 1-500)
2. vej_temp_2.csv (Station 500-slut)
3. vejtemp_udvalgte.csv (30 udvalgte faste punkter med data fra nærmeste nabo)
"""

from __future__ import annotations
import requests
import pandas as pd
from pyproj import Transformer
import numpy as np
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import Point
import io

# ---------------------------
# KONFIGURATION
# ---------------------------
URL_VEJTEMP = "https://storage.googleapis.com/trafikkort-data/geojson/25832/temperatures.point.json"
DMI_BASE = "https://opendataapi.dmi.dk/v2/metObs/collections/observation/items"
DMI_DATETIME_WINDOW = "now-PT60M/now"

# EPSG:25832 -> WGS84
transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

# Dine 30 faste "ankre"
STABLE_STATIONS_CSV = """ID,NAME,Latitude,Longitude,Region_Beskrivelse
1,Skagen,57.729397,10.548962,Nordjylland_Top
2,Hjoerring,57.453903,10.532840,Nordjylland_Oest
3,Thisted,56.911575,8.391812,Nordjylland_Vest
4,Aalborg,57.040035,9.916249,Nordjylland_City
5,Viborg,56.458168,9.989520,Midtjylland_Nord
6,Randers,56.469730,9.915512,Midtjylland_Oest
7,Holstebro,56.372086,8.586160,Midtjylland_Vest
8,Herning,56.152554,8.987697,Midtjylland_Center
9,Aarhus,56.193900,10.226004,Aarhus_Omraad
10,Ringkoebing,56.004220,8.255941,Vestjylland_Kyst
11,Horsens_Vejle,55.706036,9.573479,Oestjylland_Syd
12,Esbjerg,55.514550,8.628857,Sydvestjylland
13,Ribe,55.298500,8.908380,Sydjylland_Vest
14,Kolding,55.480820,9.422792,Trekantomraadet
15,Haderslev,55.253166,9.304873,Soenderjylland_Oest
16,Padborg,54.837070,9.339673,Graensen
17,Middelfart,55.518660,9.749365,Fyn_Vest
18,Odense,55.374600,10.333482,Fyn_Center
19,Svendborg,55.060246,10.598031,Fyn_Syd
20,Nyborg,55.296710,10.805489,Storebaelt
21,Kalundborg,55.676304,11.070842,Sjaelland_Vest_Nord
22,Slagelse,55.382904,11.375092,Sjaelland_Vest_Syd
23,Naestved,55.213960,11.765656,Sjaelland_Syd
24,Lolland,54.744896,11.451069,Lolland
25,Falster,54.938538,11.967156,Falster_Moen
26,Roskilde,55.643200,12.039816,Sjaelland_Midt
27,Koege,55.457657,12.182324,Koege_Bugt
28,Kobenhavn_N,55.802920,12.533508,Storkoebenhavn_Nord
29,Amager,55.629807,12.617974,Storkoebenhavn_Syd
30,Bornholm,55.163532,14.982273,Bornholm
"""

# ---------------------------
# HENT DMI DATA
# ---------------------------
def fetch_all_dmi_dewpoints(parameter_id: str) -> pd.DataFrame:
    params = {"parameterId": parameter_id, "datetime": DMI_DATETIME_WINDOW}
    try:
        r = requests.get(DMI_BASE, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Fejl ved DMI hentning: {e}")
        return gpd.GeoDataFrame()

    dmi_rows = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        lon, lat = geom.get("coordinates") if geom else (None, None)
        val = props.get("value")
        if lat and lon and val is not None:
            dmi_rows.append({"Dewpoint": float(val), "geometry": Point(lon, lat)})

    dmi_gdf = gpd.GeoDataFrame(dmi_rows, crs="EPSG:4326")
    return dmi_gdf.drop_duplicates(subset=["geometry"])

# ---------------------------
# HENT OG PARSE VEJTEMP
# ---------------------------
def fetch_and_parse_vejtemp() -> pd.DataFrame:
    try:
        resp = requests.get(URL_VEJTEMP, timeout=20)
        resp.raise_for_status()
        geojson = resp.json()
    except Exception as e:
        print(f"Fejl ved Vejtemp API: {e}")
        return pd.DataFrame()

    rows = []
    id_counter = 1
    for feat in geojson.get("features", []):
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        coords = geom.get("coordinates")
        
        lon, lat = (None, None)
        if coords and len(coords) >= 2:
            lon, lat = transformer.transform(coords[0], coords[1])

        # Gem både som 'Latitude' (til output) og 'Live_Lat' (til beregning)
        rows.append({
            "ID": id_counter,
            "NAME": str(id_counter),
            "Latitude": lat,
            "Longitude": lon,
            "StationID": props.get("device_id") or f"Vejtemp_{id_counter}",
            "Vej_temp": props.get("roadSurfaceTemperature"),
            "Luft_temp": props.get("airTemperature"),
        })
        id_counter += 1
    return pd.DataFrame(rows)

# ---------------------------
# MATCHING LOGIK (30 STABILE)
# ---------------------------
def create_stable_dataset(df_live: pd.DataFrame) -> pd.DataFrame:
    df_stable = pd.read_csv(io.StringIO(STABLE_STATIONS_CSV))
    
    if df_live.empty:
        df_stable["Vej_temp"] = np.nan
        df_stable["Luft_temp"] = np.nan
        return df_stable

    stable_coords = df_stable[['Latitude', 'Longitude']].values
    live_coords = df_live[['Latitude', 'Longitude']].values

    # Find nærmeste nabo
    distances = cdist(stable_coords, live_coords, metric='euclidean')
    nearest_indices = np.argmin(distances, axis=1)

    # Hent data fra de fundne live-indekser
    matched_data = df_live.iloc[nearest_indices].reset_index(drop=True)

    df_stable["Vej_temp"] = matched_data["Vej_temp"]
    df_stable["Luft_temp"] = matched_data["Luft_temp"]
    
    # Sørg for ID og StationID format
    df_stable["StationID"] = "Vejtemp_" + df_stable["ID"].astype(str)
    
    return df_stable

# ---------------------------
# MAIN
# ---------------------------
def main():
    print("--- Starter script ---")
    
    # 1. Hent Vejtemperaturer (Alle stationer)
    df = fetch_and_parse_vejtemp()
    if df.empty:
        print("Ingen vejdata fundet. Stopper.")
        return

    # 2. Hent DMI data
    dmi_gdf = fetch_all_dmi_dewpoints("temp_dew")
    
    # Hjælpefunktion til at finde dugpunkt
    def add_dewpoint(target_df):
        if dmi_gdf.empty:
            target_df["Dewpoint"] = target_df["Luft_temp"] # Fallback
            return target_df
        
        # Lav GeoDataFrame til matching
        gdf = gpd.GeoDataFrame(
            target_df, 
            geometry=gpd.points_from_xy(target_df.Longitude, target_df.Latitude),
            crs="EPSG:4326"
        )
        # Find nærmeste DMI måling
        nearest_vals = gdf.geometry.apply(
            lambda g: dmi_gdf.loc[dmi_gdf.geometry.distance(g).idxmin(), "Dewpoint"]
        )
        target_df["Dewpoint"] = nearest_vals.values
        return target_df

    # 3. Behandl DE STORE FILER (Alle data)
    print(f"Behandler {len(df)} rå målinger...")
    df = add_dewpoint(df) # Tilføj dugpunkt til alle
    
    # Gem raw data split (WSI Max format)
    cols = ["ID", "NAME", "Latitude", "Longitude", "StationID", "Vej_temp", "Luft_temp", "Dewpoint"]
    df_out = df[cols] # Sorter kolonner
    
    df_1 = df_out.iloc[:500].copy()
    df_2 = df_out.iloc[500:].copy()
    
    df_1.to_csv("vej_temp_1.csv", index=False)
    df_2.to_csv("vej_temp_2.csv", index=False)
    print(f"-> Gemte 'vej_temp_1.csv' ({len(df_1)} rækker)")
    print(f"-> Gemte 'vej_temp_2.csv' ({len(df_2)} rækker)")

    # 4. Behandl DE 30 STABILE STATIONER
    print("Behandler 30 stabile stationer (Nærmeste nabo)...")
    df_stable = create_stable_dataset(df) # Matcher vej-data
    df_stable = add_dewpoint(df_stable)   # Matcher DMI-data til de faste punkter
    
    # Gem stabil fil
    df_stable_out = df_stable[cols]
    df_stable_out.to_csv("vejtemp_udvalgte.csv", index=False)
    print(f"-> Gemte 'vejtemp_udvalgte.csv' ({len(df_stable_out)} rækker)")
    print("--- Færdig ---")

if __name__ == "__main__":
    main()
