#!/usr/bin/env python3
"""
Genererer KML med cirkler omkring observationer under grænseværdier
og et testbillede
"""

import pandas as pd
import numpy as np
import simplekml
import matplotlib.pyplot as plt

# ---------------------------
# Parametre
# ---------------------------
CIRCLE_RADIUS_DEG = 0.03  # cirka 3 km radius
VEJTEMP_THRESHOLD = 6

COLOR_TEMP = "7f66ccff"
COLOR_RISK_LOW = "7f66ccff"
COLOR_RISK_MED = "7fff9966"
COLOR_RISK_HIGH = "7f8b0000"

# ---------------------------
# Læs CSV
# ---------------------------
df1 = pd.read_csv("vej_temp_1.csv")
df2 = pd.read_csv("vej_temp_2.csv")
df = pd.concat([df1, df2], ignore_index=True)

lons = df["Longitude"].to_numpy()
lats = df["Latitude"].to_numpy()
vejtemp = df["Vej_temp"].to_numpy()
dugpunkt = df["Luft_temp"].to_numpy()

# ---------------------------
# Gem testbillede
# ---------------------------
plt.figure(figsize=(8,10))
plt.scatter(lons, lats, c=vejtemp, cmap="Blues", s=200, alpha=0.6)
plt.colorbar(label="Vejtemperatur (°C)")
plt.title("Vejtemperatur observationer")
plt.savefig("test_vejtemp.png", dpi=200)
plt.close()
print("✔ Testbillede gemt: test_vejtemp.png")

# ---------------------------
# Funktion til at tegne cirkler i KML under grænse
# ---------------------------
def add_circles_to_kml_under_threshold(kml_obj, lons, lats, values, radius_deg, threshold, color_func):
    for lon, lat, val in zip(lons, lats, values):
        if np.isnan(val):
            continue
        if val >= threshold:  # spring punkter over, der ikke er under grænsen
            continue
        pol = kml_obj.newpolygon()
        # Cirkel som polygon med 20 punkter
        circle_coords = [
            (lon + radius_deg*np.cos(theta), lat + radius_deg*np.sin(theta))
            for theta in np.linspace(0, 2*np.pi, 20)
        ]
        circle_coords.append(circle_coords[0])
        pol.outerboundaryis.coords = circle_coords
        pol.style.polystyle.color = color_func(val)
        pol.style.polystyle.fill = 1
        pol.style.polystyle.outline = 0

# ---------------------------
# KML: Vejtemperatur (kun under threshold)
# ---------------------------
kml_temp = simplekml.Kml()
add_circles_to_kml_under_threshold(
    kml_temp,
    lons, lats, vejtemp,
    radius_deg=CIRCLE_RADIUS_DEG,
    threshold=VEJTEMP_THRESHOLD,
    color_func=lambda val: COLOR_TEMP
)
kml_temp.save("vejtemp_only.kml")
print("✔ KML gemt: vejtemp_only.kml")

# ---------------------------
# KML: Risiko for glatføre (kun negative værdier)
# ---------------------------
kml_risk = simplekml.Kml()
risk_vals = []
for t, dew in zip(vejtemp, dugpunkt):
    if np.isnan(t) or np.isnan(dew) or t >= 0:
        risk_vals.append(np.nan)
    else:
        risk_vals.append(t - dew)

add_circles_to_kml_under_threshold(
    kml_risk,
    lons, lats, risk_vals,
    radius_deg=CIRCLE_RADIUS_DEG,
    threshold=0,  # kun negative værdier
    color_func=lambda delta: COLOR_RISK_HIGH if delta < 0 else (COLOR_RISK_MED if delta < 1 else COLOR_RISK_LOW)
)
kml_risk.save("vejtemp_dugpunkt.kml")
print("✔ KML gemt: vejtemp_dugpunkt.kml")
