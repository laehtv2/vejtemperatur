#!/usr/bin/env python3
"""
Henter vejtemperaturer fra Trafikkort.
Outputter 3 filer:
1. vej_temp_1.csv (Station 1-500)
2. vej_temp_2.csv (Station 500-slut)
3. vejtemp_udvalgte.csv (30 faste punkter -> Viser LAVESTE temp fra de 5 nærmeste målere)

Opdatering: Tilføjer '°' symbol til temperaturværdierne i outputtet.
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

# Hvor mange nabo-stationer skal vi tjekke for at finde den koldeste?
SEARCH_NEIGHBORS = 5 

# EPSG:25832 -> WGS84
transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

# Dine 30 faste "ankre"
STABLE_STATIONS_CSV = """ID,NAME,Latitude,Longitude,Region_Beskrivelse
1,Skagen,57.72540330309609,10.579238916078774,Nordjylland_Top
2,Hjoerring,57.45770035155239,9.994979640604203,Nordjylland_Oest
3,Thisted,56.95923282275056,8.700091644977993,Nordjylland_Vest
4,Aalborg,57.043675769295795,9.92528877646127,Nordjylland_City
5,Viborg,56.45297138538134,9.398296561781704,Midtjylland_Nord
6,Randers,56.46055972507178,10.03825016268686,Midtjylland_Oest
7,Holstebro,56.35951074837025,8.621013855470943,Midtjylland_Vest
8,Herning,56.1382467390902,8.96982975689481,Midtjylland_Center
9,Aarhus,56.167004750185406,10.190152028068606,Aarhus_Omraad
10,Ringkoebing,56.098014035831014,8.30623339262066,Vestjylland_Kyst
11,Horsens_Vejle,55.79358726687867,9.700333182852141,Oestjylland_Syd
12,Esbjerg,55.48371236383134,8.467712132801912,Sydvestjylland
13,Ribe,55.32785444149237,8.796052237837126,Sydjylland_Vest
14,Kolding,55.49286134097332,9.468880321925678,Trekantomraadet
15,Haderslev,55.25429669134515,9.490410820616512,Soenderjylland_Oest
16,Padborg,54.83488912134988,9.33969733820579,Graensen
17,Middelfart,55.50200818453559,9.786455186040588,Fyn_Vest
18,Odense,55.40433218492701,10.40007439872935,Fyn_Center
19,Svendborg,55.082131253176975,10.615379385637684,Fyn_Syd
20,Nyborg,55.32785443244441,10.787623375164356,Storebaelt
21,Kalundborg,55.678430102348926,11.132111417023047,Sjaelland_Vest_Nord
22,Slagelse,55.404150925906904,11.350696483230257,Sjaelland_Vest_Syd
23,Naestved,55.2270828936444,11.768146425954058,Sjaelland_Syd
24,Lolland,54.75588445571971,11.453693750653583,Lolland
25,Falster,54.86405991160968,11.897788640140478,Falster_Moen
26,Roskilde,55.64148895313767,12.078647866931744,Sjaelland_Midt
27,Koege,55.445905747800005,12.148787888346185,Koege_Bugt
28,Hilleroed,55.930236083032625,12.294077940665346,Nordsjaelland
29,Koebenhavn,55.67540416204908,12.51952800949747,Koebenhavn
30,Bornholm,55.12063248266432,14.926833853544782,Bornholm
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
    
    # Konverter til tal og håndter manglende data
    df = pd.DataFrame(rows)
    df["Vej_temp"] = pd.to_numeric(df["Vej_temp"], errors='coerce')
    df["Luft_temp"] = pd.to_numeric(df["Luft_temp"], errors='coerce')
    return df

# ---------------------------
# NY MATCHING LOGIK (FIND KOLDESTE I OMRÅDET)
# ---------------------------
def create_stable_dataset(df_live: pd.DataFrame) -> pd.DataFrame:
    df_stable = pd.read_csv(io.StringIO(STABLE_STATIONS_CSV))
    
    if df_live.empty:
        df_stable["Vej_temp"] = np.nan
        df_stable["Luft_temp"] = np.nan
        return df_stable

    # Fjern rækker hvor Vej_temp er NaN
    df_live_valid = df_live.dropna(subset=["Vej_temp"]).copy()
    
    if df_live_valid.empty:
        print("Advarsel: Ingen gyldige vejtemperaturer fundet.")
        return df_stable

    stable_coords = df_stable[['Latitude', 'Longitude']].values
    live_coords = df_live_valid[['Latitude', 'Longitude']].values

    # 1. Beregn afstande
    distances = cdist(stable_coords, live_coords, metric='euclidean')
    
    vej_temps = []
    luft_temps = []

    # 2. Loop gennem hver fast station
    for i in range(len(df_stable)):
        row_dists = distances[i]
        
        # Find indexene på de N nærmeste stationer
        closest_indices = np.argsort(row_dists)[:SEARCH_NEIGHBORS]
        
        # Udvælg disse rækker
        candidate_rows = df_live_valid.iloc[closest_indices]
        
        # 3. Find rækken med den LAVESTE vejtemperatur
        coldest_idx = candidate_rows["Vej_temp"].idxmin()
        coldest_row = candidate_rows.loc[coldest_idx]
        
        # Gem værdierne
        vej_temps.append(coldest_row["Vej_temp"])
        luft_temps.append(coldest_row["Luft_temp"])

    df_stable["Vej_temp"] = vej_temps
    df_stable["Luft_temp"] = luft_temps
    
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
        
        gdf = gpd.GeoDataFrame(
            target_df, 
            geometry=gpd.points_from_xy(target_df.Longitude, target_df.Latitude),
            crs="EPSG:4326"
        )
        nearest_vals = gdf.geometry.apply(
            lambda g: dmi_gdf.loc[dmi_gdf.geometry.distance(g).idxmin(), "Dewpoint"]
        )
        target_df["Dewpoint"] = nearest_vals.values
        return target_df
        
    # Hjælpefunktion til formatering (Tilføj °)
    def format_temperatures(target_df, cols):
        for c in cols:
            if c in target_df.columns:
                # Konverter til string og tilføj gradtegn, hvis værdien ikke er tom
                target_df[c] = target_df[c].apply(lambda x: f"{x}°" if pd.notnull(x) else "")
        return target_df

    # 3. Behandl DE STORE FILER (Alle data)
    print(f"Behandler {len(df)} rå målinger...")
    df = add_dewpoint(df) # Tilføj dugpunkt til alle
    
    cols = ["ID", "NAME", "Latitude", "Longitude", "StationID", "Vej_temp", "Luft_temp", "Dewpoint"]
    valid_cols = [c for c in cols if c in df.columns]
    
    # Kopier data til output dataframe
    df_out = df[valid_cols].copy()
    
    # >>> HER FORMATERER VI DE STORE FILER <<<
    df_out = format_temperatures(df_out, ["Vej_temp", "Luft_temp", "Dewpoint"])
    
    df_1 = df_out.iloc[:500].copy()
    df_2 = df_out.iloc[500:].copy()
    
    df_1.to_csv("vej_temp_1.csv", index=False)
    df_2.to_csv("vej_temp_2.csv", index=False)
    print("Gemte vej_temp_1.csv og vej_temp_2.csv med gradtegn.")


    # 4. Behandl DE 30 STABILE STATIONER
    print(f"Behandler 30 stabile stationer (Finder laveste temp blandt {SEARCH_NEIGHBORS} naboer)...")
    df_stable = create_stable_dataset(df) # Matcher vej-data (Worst-case)
    df_stable = add_dewpoint(df_stable)   # Matcher DMI-data til de faste punkter
    
    # Klargør stabil output
    valid_cols_stable = [c for c in cols if c in df_stable.columns]
    df_stable_out = df_stable[valid_cols_stable].copy()
    
    # >>> HER FORMATERER VI DEN LILLE FIL <<<
    df_stable_out = format_temperatures(df_stable_out, ["Vej_temp", "Luft_temp", "Dewpoint"])
    
    df_stable_out.to_csv("vejtemp_udvalgte.csv", index=False)
    print("Gemte vejtemp_udvalgte.csv med gradtegn.")

if __name__ == "__main__":
    main()
