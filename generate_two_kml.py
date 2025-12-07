#!/usr/bin/env python3
"""
generate_two_kml.py

Genererer to KML-filer baseret på vejtemperaturer:
1. vejtemp_only.kml : markerer punkter med vejtemp < 0°C
2. vejtemp_dugpunkt.kml : markerer risiko for glatføre baseret på vejtemp og dugpunkt
"""

import pandas as pd
import simplekml

# ---------------------------
# Farver (aabbggrr)
# ---------------------------
COLOR_TEMP = "7f66ccff"   # lys blå for vejtemp < 0
COLOR_RISK_LOW = "7f66ccff"
COLOR_RISK_MED = "7fff9966"
COLOR_RISK_HIGH = "7f8b0000"

# ---------------------------
# Læs CSV
# ---------------------------
df1 = pd.read_csv("vej_temp_1.csv")
df2 = pd.read_csv("vej_temp_2.csv")
df = pd.concat([df1, df2], ignore_index=True)

# ---------------------------
# KML 1: Vejtemperatur < 0°C
# ---------------------------
kml_temp = simplekml.Kml()
for _, r in df.iterrows():
    try:
        t = float(r["Vej_temp"])
    except:
        continue
    if t < 0:
        p = kml_temp.newpoint(
            name=f"ID {r['ID']} Vejtemp {t}°C",
            coords=[(r["Longitude"], r["Latitude"])]
        )
        p.style.iconstyle.color = COLOR_TEMP
        p.style.iconstyle.scale = 1.2
        p.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"

kml_temp.save("vejtemp_only.kml")
print("✔ KML gemt: vejtemp_only.kml")

# ---------------------------
# KML 2: Risiko for glatføre (vejtemp + dugpunkt)
# ---------------------------
kml_risk = simplekml.Kml()
for _, r in df.iterrows():
    try:
        t = float(r["Vej_temp"])
        dew = float(r["Luft_temp"])
    except:
        continue
    # simpel risiko-logik: hvis vejtemp < dugpunkt → risiko
    risk = 0
    if t < 0 and dew is not None:
        delta = t - dew
        if delta < 0:
            risk = 3
        elif delta < 1:
            risk = 2
        else:
            risk = 1
    if risk > 0:
        color = COLOR_RISK_LOW
        if risk == 2:
            color = COLOR_RISK_MED
        elif risk == 3:
            color = COLOR_RISK_HIGH
        p = kml_risk.newpoint(
            name=f"ID {r['ID']} Risk {risk}",
            coords=[(r["Longitude"], r["Latitude"])]
        )
        p.style.iconstyle.color = color
        p.style.iconstyle.scale = 1.2
        p.style.iconstyle.icon.href = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"

kml_risk.save("vejtemp_dugpunkt.kml")
print("✔ KML gemt: vejtemp_dugpunkt.kml")
