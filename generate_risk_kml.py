#!/usr/bin/env python3
"""
generate_risk_kml.py

Henter vejtemperaturer (Trafikkort) og dugpunkt + nedbør (DMI MetObs),
beregner glatføre-risiko (Model A) og skriver:
- risiko_glatfoere.kml
- risiko_glatfoere.kmz

Kør: python generate_risk_kml.py
"""

from __future__ import annotations
import requests
import pandas as pd
from pyproj import Transformer
import simplekml
import zipfile
import io
import datetime
import time
from typing import Tuple, Optional

# ---------------------------
# KONFIG
# ---------------------------
DMI_API_KEY = "DIN_DMI_API_NOEGLE_HER"  # <-- indsæt din DMI API-nøgle
DMI_BASE = "https://opendataapi.dmi.dk/v2/metObs/collections/observation/items"

TRAFIKKORT_URL = "https://storage.googleapis.com/trafikkort-data/geojson/25832/temperatures.point.json"

# tidvindue for DMI-hentning (fx sidste 60 minutter)
DMI_DATETIME_WINDOW = "now-PT60M/now"

# EPSG:25832 -> WGS84
transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)

# Farver (blå skala) i KML-format aabbggrr (alpha, blue, green, red)
COLOR_MAP = {
    1: "7fe6d8ad",  # lys blå  (RGB AD D8 E6)
    2: "7fff9966",  # mellem blå (RGB 66 99 FF)
    3: "7fff6633",  # mørk blå (RGB 33 66 FF)
    4: "7f8b0000",  # dyb blå (navy RGB 00 00 8B)
}

# ---------------------------
# HENT & PARS TRAFIKKORT
# ---------------------------
def fetch_trafikkort(url: str = TRAFIKKORT_URL) -> dict:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

def parse_trafikkort(geojson: dict) -> pd.DataFrame:
    rows = []
    id_counter = 1
    for feat in geojson.get("features", []):
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        # coords kommer i EPSG:25832 (x,y) - transformer expects x,y -> lon,lat in WGS84
        x, y = coords[0], coords[1]
        lon, lat = transformer.transform(x, y)
        rows.append({
            "ID": id_counter,
            "NAME": str(id_counter),
            "Latitude": lat,
            "Longitude": lon,
            "Vej_temp": props.get("roadSurfaceTemperature"),
            "Luft_temp": props.get("airTemperature"),
            # Trafikkort stationId kan være brugbar til DMI-match
            "StationID": props.get("stationId") or props.get("station_id") or None
        })
        id_counter += 1
    return pd.DataFrame(rows)

# ---------------------------
# HENT DMI OBSERVATIONER PER STATION
# ---------------------------
def fetch_dmi_for_station(station_id: str, parameters: list[str]) -> dict:
    """
    Hent observationer for en station. 
    parameters: e.g. ["temp_dew","precip_past1h","temp_dry"]
    Returnerer dict param -> (value, observed_timestamp) for seneste value i tidsvinduet.
    """
    params = {
        "stationId": station_id,
        "parameterId": ",".join(parameters),
        "datetime": DMI_DATETIME_WINDOW,
        "limit": 1000  # rimeligt højt for at hente seneste flere parametre
    }
    if DMI_API_KEY:
        params["api-key"] = DMI_API_KEY

    try:
        r = requests.get(DMI_BASE, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        # fallback: tomt
        # print(f"Warning: DMI call failed for station {station_id}: {e}")
        return {}

    features = data.get("features", [])
    latest_by_param = {}
    for feat in features:
        props = feat.get("properties", {})
        param = props.get("parameterId")
        val = props.get("value")
        observed = props.get("observed")  # ISO timestamp
        try:
            observed_ts = datetime.datetime.fromisoformat(observed.replace("Z", "+00:00")) if observed else None
        except Exception:
            observed_ts = None

        if param is None:
            continue
        # keep latest (most recent observed) per parameter
        cur = latest_by_param.get(param)
        if cur is None or (observed_ts is not None and cur[1] is not None and observed_ts > cur[1]):
            latest_by_param[param] = (val, observed_ts)
    return latest_by_param

# ---------------------------
# RISIKO-MODEL (Model A)
# ---------------------------
def compute_risk(vej_temp: Optional[float], dew: Optional[float], precip: Optional[float]) -> int:
    """Returnerer score 0..4 iflg Model A regler."""
    if vej_temp is None:
        return 0

    # Sørg for floats hvis muligt
    try:
        Troad = float(vej_temp)
    except Exception:
        return 0

    dew_val = None
    if dew is not None:
        try:
            dew_val = float(dew)
        except Exception:
            dew_val = None

    precip_val = None
    if precip is not None:
        try:
            precip_val = float(precip)
        except Exception:
            precip_val = None

    delta = None
    if dew_val is not None:
        delta = Troad - dew_val

    # Niveau 4 - ekstrem
    if Troad < 0.0 and delta is not None and delta < 0.5 and precip_val and precip_val > 0.0:
        return 4

    # Niveau 3 - høj
    if (Troad < 0.5 and delta is not None and delta < 1.0) or (Troad < 1.0 and precip_val and precip_val > 0.1):
        return 3

    # Niveau 2 - moderat
    if Troad < 1.5 and delta is not None and delta < 2.0:
        return 2

    # Niveau 1 - lav
    if Troad < 2.5 and delta is not None and delta < 3.0:
        return 1

    return 0

# ---------------------------
# GENERER KML & KMZ
# ---------------------------
def generate_kml(df: pd.DataFrame, out_kml: str = "risiko_glatfoere.kml", out_kmz: str = "risiko_glatfoere.kmz") -> None:
    kml = simplekml.Kml()
    # Opret et dokument navn
    kml.document.name = "Glatføre risiko"

    for _, r in df.iterrows():
        risk = int(r.get("Risk", 0) or 0)
        if risk <= 0:
            continue  # transparent = ikke tegnet

        # Beskrivelse med relevante værdier
        desc = (
            f"ID: {r.get('ID')}<br/>"
            f"Vejtemp: {r.get('Vej_temp')} °C<br/>"
            f"Dugpunkt: {r.get('Dewpoint')} °C<br/>"
            f"Nedbør (1h): {r.get('Precip')} mm<br/>"
            f"Risk: {risk}"
        )

        p = kml.newpoint(name=f"ID {r.get('ID')} - Risk {risk}", coords=[(float(r["Longitude"]), float(r["Latitude"]))])
        p.description = desc
        color = COLOR_MAP.get(risk, "7f000000")
        p.style.iconstyle.color = color
        p.style.iconstyle.scale = 1.2
        p.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
        # label style (valgfri)
        p.style.labelstyle.color = color
        p.style.labelstyle.scale = 1.0

    kml.save(out_kml)
    print(f"✔ KML gemt: {out_kml}")

    # Lav KMZ (zip med doc.kml)
    # simplekml har ikke altid savekmz portable, så vi zippe manuelt
    with open(out_kml, "rb") as f:
        kml_bytes = f.read()
    with zipfile.ZipFile(out_kmz, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_bytes)
    print(f"✔ KMZ gemt: {out_kmz}")

# ---------------------------
# MAIN
# ---------------------------
def main():
    # 1) Hent trafikkort
    print("Henter Trafikkort data...")
    gj = fetch_trafikkort()
    df = parse_trafikkort(gj)
    print(f"Antal punkter fundet: {len(df)} (max 672 forventet)")

    # 2) Split i to csv (som du allerede bruger)
    df.iloc[:500].to_csv("vej_temp_1.csv", index=False)
    df.iloc[500:].to_csv("vej_temp_2.csv", index=False)
    print("✔ Gemt: vej_temp_1.csv, vej_temp_2.csv")

    # 3) For hver station hent DMI parametre (temp_dew, precip_past1h)
    params_to_get = ["temp_dew", "precip_past1h", "temp_dry"]
    dew_list = []
    precip_list = []
    tempdry_list = []

    print("Henter DMI-observationer pr. station (dette kan tage et øjeblik)...")
    # Tryk for rate-limiting: vi gør små sleep mellem kald, især hvis mange stationer
    for idx, station_id in enumerate(df["StationID"]):
        if station_id is None:
            dew_list.append(None)
            precip_list.append(None)
            tempdry_list.append(None)
            continue

        # station_id kan være tal eller streng - sørg for str
        sid = str(station_id)
        try:
            obs = fetch_dmi_for_station(sid, params_to_get)
        except Exception:
            obs = {}

        # extract values
        dew_val = None
        precip_val = None
        tempdry_val = None
        if "temp_dew" in obs:
            dew_val = obs["temp_dew"][0]
        # nogle param navne kan hedde temp_dry eller temp_dry - tjek keys
        if "temp_dry" in obs:
            tempdry_val = obs["temp_dry"][0]
        if "precip_past1h" in obs:
            precip_val = obs["precip_past1h"][0]

        dew_list.append(dew_val)
        precip_list.append(precip_val)
        tempdry_list.append(tempdry_val)

        # mild rate-limit (tilpas ved behov)
        time.sleep(0.05)

    df["Dewpoint"] = dew_list
    df["Precip"] = precip_list
    df["TempDry"] = tempdry_list

    # 4) Beregn risiko per punkt
    df["Risk"] = df.apply(lambda r: compute_risk(r.get("Vej_temp"), r.get("Dewpoint"), r.get("Precip")), axis=1)

    # 5) Gem også en CSV af resultater (valgfrit, nyttigt)
    df.to_csv("risiko_glatfoere.csv", index=False)
    print("✔ Gemt CSV med risici: risiko_glatfoere.csv")

    # 6) Generer KML og KMZ
    generate_kml(df)

if __name__ == "__main__":
    if not DMI_API_KEY or DMI_API_KEY == "DIN_DMI_API_NOEGLE_HER":
        print("OBS: Indsæt din DMI API-nøgle i variablen DMI_API_KEY i toppen af scriptet.")
        print("Scriptet fortsætter, men DMI-kald vil fejle uden gyldig nøgle.")
    main()
