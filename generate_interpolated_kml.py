#!/usr/bin/env python3
"""
Genererer to KML-filer med interpolerede områder fra vejtemperaturer:

1. vejtemp_only.kml : områder med vejtemp < 0°C
2. vejtemp_dugpunkt.kml : områder med risiko for glatføre (vejtemp + dugpunkt)
"""

import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import simplekml

# ---------------------------
# Parametre
# ---------------------------
GRID_SIZE = 200   # grid resolution
VEJTEMP_THRESHOLD = 0  # grader C
RISK_LOW = 1
RISK_MED = 2
RISK_HIGH = 3

# Farver (aabbggrr)
COLOR_TEMP = "7f66ccff"   # lys blå
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
# Opret grid
# ---------------------------
grid_lon = np.linspace(lons.min(), lons.max(), GRID_SIZE)
grid_lat = np.linspace(lats.min(), lats.max(), GRID_SIZE)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# Interpoler vejtemperatur
grid_vejtemp = griddata(points=(lons, lats), values=vejtemp,
                        xi=(grid_lon, grid_lat), method="linear")

# Interpoler dugpunkt
grid_dug = griddata(points=(lons, lats), values=dugpunkt,
                    xi=(grid_lon, grid_lat), method="linear")

# ---------------------------
# Funktion til at lave polygoner
# ---------------------------
def create_polygons_from_grid(kml_obj, grid_vals, grid_lon, grid_lat, color_func, threshold_func):
    rows, cols = grid_vals.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            val = grid_vals[i, j]
            if np.isnan(val):
                continue
            if not threshold_func(val):
                continue
            coords = [
                (grid_lon[i,j], grid_lat[i,j]),
                (grid_lon[i,j+1], grid_lat[i,j+1]),
                (grid_lon[i+1,j+1], grid_lat[i+1,j+1]),
                (grid_lon[i+1,j], grid_lat[i+1,j]),
                (grid_lon[i,j], grid_lat[i,j])
            ]
            pol = kml_obj.newpolygon()
            pol.outerboundaryis.coords = coords
            pol.style.polystyle.color = color_func(val)
            pol.style.polystyle.fill = 1
            pol.style.polystyle.outline = 0

# ---------------------------
# KML 1: Vejtemperatur < 0°C
# ---------------------------
kml_temp = simplekml.Kml()
create_polygons_from_grid(
    kml_temp,
    grid_vals=grid_vejtemp,
    grid_lon=grid_lon,
    grid_lat=grid_lat,
    color_func=lambda val: COLOR_TEMP,
    threshold_func=lambda val: val < VEJTEMP_THRESHOLD
)
kml_temp.save("vejtemp_only.kml")
print("✔ KML gemt: vejtemp_only.kml")

# ---------------------------
# KML 2: Risiko for glatføre (vejtemp + dugpunkt)
# ---------------------------
def risk_color(t, dew):
    # Simpel risiko-logik
    delta = t - dew
    if delta < 0:
        return COLOR_RISK_HIGH
    elif delta < 1:
        return COLOR_RISK_MED
    else:
        return COLOR_RISK_LOW

kml_risk = simplekml.Kml()
grid_risk = np.full_like(grid_vejtemp, np.nan)

# Beregn risiko for hvert gridpunkt
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        t = grid_vejtemp[i,j]
        dew = grid_dug[i,j]
        if np.isnan(t) or np.isnan(dew):
            continue
        if t >= 0:
            continue
        grid_risk[i,j] = t - dew  # brug delta til farve

create_polygons_from_grid(
    kml_risk,
    grid_vals=grid_risk,
    grid_lon=grid_lon,
    grid_lat=grid_lat,
    color_func=lambda delta: COLOR_RISK_HIGH if delta < 0 else (COLOR_RISK_MED if delta < 1 else COLOR_RISK_LOW),
    threshold_func=lambda delta: not np.isnan(delta)
)

kml_risk.save("vejtemp_dugpunkt.kml")
print("✔ KML gemt: vejtemp_dugpunkt.kml")
