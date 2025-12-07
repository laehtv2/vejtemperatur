#!/usr/bin/env python3
"""
IDW (Inverse Distance Weighting) interpolering for at generere KML-områder
for "kolde" vejtemperaturområder og glatførerisiko.

- Genererer et grid over Danmark.
- Interpolerer vejtemperatur og vejtemp - dugpunkt (delta) på grid'et.
- Maskerer grid'et for områder, hvor interpoleret temp < VEJTEMP_THRESHOLD.
- Klippes til Danmark (naturalearth highres) og bounding box.

Output:
    vejtemp_only.kml (interpolerede områder hvor temp < threshold)
    vejtemp_dugpunkt.kml (interpolerede glatføre-risiko områder: delta < 0)
    test_vejtemp.png (visuelt kort)
"""
import os
import math
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPoint, box
from shapely.ops import unary_union
import geopandas as gpd
import simplekml
import matplotlib.pyplot as plt

# ---------------------------
# Parametre
# ---------------------------
VEJTEMP_THRESHOLD = 8             # Tærskel for "kold" vejtemp (°C)
GRID_RESOLUTION = 200             # Opløsning af interpolationsgrid (f.eks. 200x200)
IDW_POWER = 2                     # Inverse Distance Weighting Power (typisk 2)

# Bounding box over Danmark (nu udvidet til Bornholm)
LON_MIN, LON_MAX = 8.0, 15.5      # Udvidelse til 15.5 inkluderer Bornholm
LAT_MIN, LAT_MAX = 54.5, 57.9

# Filnavne
KML_VEJTEMP = "vejtemp_only.kml"
KML_RISK = "vejtemp_dugpunkt.kml"
PNG_TEST = "test_vejtemp.png"
NE_HIGHRES_FILE = "ne_10m_admin_0_countries.shp" # Lokal fil hentet af workflowet

# Farver (AABBGGRR hex-format for simplekml)
COLOR_TEMP = "7fffc040"           # Transparent (7f) Blå-Turkis
COLOR_RISK_HIGH = "7f0000ff"      # Transparent (7f) Rød

# ---------------------------
# Interpolationsfunktioner
# ---------------------------

def idw_interpolation(stations: np.ndarray, values: np.ndarray, grid_points: np.ndarray, power: int = 2) -> np.ndarray:
    """
    Inverse Distance Weighting (IDW) interpolation.
    Beregner værdier på grid_points baseret på de målte stationer og værdier.
    """
    
    # Beregn euklidisk afstand mellem hver grid-punkt og hver station
    # grid_points.shape = (M, 2), stations.shape = (N, 2)
    # Distances.shape = (M, N)
    distances = np.sqrt(
        (grid_points[:, None, 0] - stations[None, :, 0])**2 +
        (grid_points[:, None, 1] - stations[None, :, 1])**2
    )
    
    # Håndter division med nul: Hvis afstanden er 0 (grid-punkt er en station), sæt til lille værdi.
    distances[distances == 0] = 1e-6 
    
    # Beregn vægte: 1 / afstand^power
    weights = 1.0 / (distances ** power)
    
    # Beregn interpolerede værdier: (værdi * vægt) / (sum af vægte)
    interpolated_values = np.sum(values * weights, axis=1) / np.sum(weights, axis=1)
    
    return interpolated_values

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)

def create_interpolated_union(interpolated_values: np.ndarray, grid_points_m: np.ndarray, threshold: float, threshold_type: str, cell_buffer: float) -> gpd.GeoSeries | None:
    """
    Maskerer gridet og samler de overskydende punkter til et samlet polygon.
    """
    if threshold_type == 'less_than':
        mask_grid = interpolated_values < threshold
    elif threshold_type == 'greater_than':
        mask_grid = interpolated_values > threshold
    else:
        raise ValueError("Ukendt threshold_type")

    gdf_m = gpd.GeoDataFrame(
        {'value': interpolated_values[mask_grid]},
        geometry=[Point(x, y) for x, y in grid_points_m[mask_grid]],
        crs='EPSG:3857'
    )

    if gdf_m.empty:
        return None
    else:
        # Buffer punkter for at skabe sammenhængende celler og derefter union
        polygons = [geom.buffer(cell_buffer, join_style=2) for geom in gdf_m.geometry]
        return unary_union(polygons)

# ---------------------------
# Hovedlogik
# ---------------------------

# Load data
df1 = pd.read_csv("vej_temp_1.csv")
df2 = pd.read_csv("vej_temp_2.csv")
df = pd.concat([df1, df2], ignore_index=True)
df = df.dropna(subset=["Longitude","Latitude", "Vej_temp", "Luft_temp"])

lons = df["Longitude"].to_numpy()
lats = df["Latitude"].to_numpy()
vejtemp = df["Vej_temp"].to_numpy()

# Forbered glatførerisiko (delta = Vejtemp - Dugpunkt)
# Vi sætter delta til NaN, hvis Vejtemp >= 0, da risikoen er lav/ikke eksisterende over frysepunktet.
risk_vals = df.apply(
    lambda row: row['Vej_temp'] - row['Luft_temp'] if row['Vej_temp'] < 0 else np.nan,
    axis=1
).to_numpy()

# Konverter til GeoDataFrame og projicer til meter (EPSG:3857)
gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
gdf_m = gdf.to_crs(epsg=3857)  # meter projection
points_m = np.vstack([gdf_m.geometry.x.values, gdf_m.geometry.y.values]).T

# ---------------------------
# Grid generation for IDW (i meter EPSG:3857)
# ---------------------------

# Definer grid-området baseret på dataens udstrækning
x_min, x_max = gdf_m.bounds.minx.min(), gdf_m.bounds.maxx.max()
y_min, y_max = gdf_m.bounds.miny.min(), gdf_m.bounds.maxy.max()

x_range = np.linspace(x_min, x_max, GRID_RESOLUTION)
y_range = np.linspace(y_min, y_max, GRID_RESOLUTION)

X, Y = np.meshgrid(x_range, y_range)
grid_points_m = np.vstack([X.ravel(), Y.ravel()]).T
grid_spacing_m = x_range[1] - x_range[0]
cell_buffer = grid_spacing_m * 0.7  # Buffer for at skabe kontinuerligt polygon

# ---------------------------
# IDW Interpolering
# ---------------------------

# 1. Vejtemp
print("Starter IDW interpolering for Vejtemp...")
interpolated_temp = idw_interpolation(points_m, vejtemp, grid_points_m, IDW_POWER)
union_cold = create_interpolated_union(interpolated_temp, grid_points_m, VEJTEMP_THRESHOLD, 'less_than', cell_buffer)
print(f"✔ Vejtemp områder oprettet via IDW (under {VEJTEMP_THRESHOLD}°C).")

# 2. Risiko (Delta < 0)
# Fjern NaN fra risk_vals og tilsvarende punkter for kun at interpolere, hvor data eksisterer
valid_risk_mask = ~np.isnan(risk_vals)
if np.sum(valid_risk_mask) > 1:
    print("Starter IDW interpolering for Glatførerisiko (delta < 0)...")
    interpolated_risk_delta = idw_interpolation(points_m[valid_risk_mask], risk_vals[valid_risk_mask], grid_points_m, IDW_POWER)
    union_risk = create_interpolated_union(interpolated_risk_delta, grid_points_m, 0, 'less_than', cell_buffer)
    print("✔ Risiko områder oprettet via IDW (delta < 0).")
else:
    union_risk = None
    print("Ikke nok data til at interpolere glatførerisiko.")

# ---------------------------
# Begræns til Danmark (Highres)
# ---------------------------
try:
    world = gpd.read_file(NE_HIGHRES_FILE)
except Exception:
    raise SystemExit(f"Fejl: Kunne ikke læse {NE_HIGHRES_FILE}. Sikr dig, at filen er hentet og unzipped i workflowet.")

# Find Danmark baseret på de nye kolonnenavne for highres data
denmark = world[world["SOVEREIGNT"] == "Denmark"]
if denmark.empty:
    denmark = world[world["ADM0_A3"] == "DNK"]
if denmark.empty:
    denmark = world[world["ADMIN"] == "Denmark"]

if denmark.empty:
    raise SystemExit("Fejl: Kunne ikke finde 'Denmark' i high-res datasættet.")

# Opret og klip med Bounding Box
bbox_wgs = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
gdf_bbox = gpd.GeoDataFrame({"geometry":[bbox_wgs]}, crs="EPSG:4326")
bbox_m = gdf_bbox.to_crs(epsg=3857).geometry.iloc[0]

# Intersect denmark geometry med bbox
denmark_m = denmark.to_crs(epsg=3857).geometry.unary_union
denmark_m = denmark_m.intersection(bbox_m)

# Clip union_cold og union_risk til denmark_m
def clip_to_land(geom):
    if geom is None or geom.is_empty:
        return None
    return geom.intersection(denmark_m)

union_cold_clipped = clip_to_land(union_cold)
union_risk_clipped = clip_to_land(union_risk)

# ---------------------------
# Gem KML-filer
# ---------------------------
# ... (resten af KML-gemme funktionerne er uændret) ...

def geom_to_lonlat_coords(geom):
    """
    Konverter shapely geometrier i EPSG:3857 til (lon,lat) koordinater.
    """
    if geom is None or geom.is_empty:
        return []
    g = gpd.GeoSeries([geom], crs="EPSG:3857").to_crs(epsg=4326).geometry.iloc[0]
    polys = []
    if g.geom_type == "Polygon":
        polys.append(list(g.exterior.coords))
    elif g.geom_type == "MultiPolygon":
        for p in g.geoms:
            polys.append(list(p.exterior.coords))
    return polys

ensure_dir(KML_VEJTEMP)
ensure_dir(KML_RISK)

# Vejtemp KML
kml_temp = simplekml.Kml()
polys = geom_to_lonlat_coords(union_cold_clipped)
for coords in polys:
    pol = kml_temp.newpolygon()
    pol.outerboundaryis.coords = coords
    pol.style.polystyle.color = COLOR_TEMP
    pol.style.polystyle.fill = 1
    pol.style.polystyle.outline = 0
kml_temp.save(KML_VEJTEMP)
print(f"✔ KML gemt: {KML_VEJTEMP} (interpolerede områder for temp < {VEJTEMP_THRESHOLD}°C)")

# Risiko KML
kml_risk = simplekml.Kml()
polys_r = geom_to_lonlat_coords(union_risk_clipped)
for coords in polys_r:
    pol = kml_risk.newpolygon()
    pol.outerboundaryis.coords = coords
    pol.style.polystyle.color = COLOR_RISK_HIGH
    pol.style.polystyle.fill = 1
    pol.style.polystyle.outline = 0
kml_risk.save(KML_RISK)
print(f"✔ KML gemt: {KML_RISK} (interpolerede områder for glatføre delta < 0)")

# ---------------------------
# Gem testbillede (WGS84)
# ---------------------------
fig, ax = plt.subplots(figsize=(8,10))
# Plot Denmark land (bbox-limited)
if not denmark.empty:
    den_wgs = denmark.geometry.iloc[0].intersection(bbox_wgs)
    gpd.GeoSeries([den_wgs]).plot(ax=ax, color="lightgray", edgecolor="k")

# Plot cold areas (interpoleret)
if union_cold_clipped is not None and not union_cold_clipped.is_empty:
    poly_gs = gpd.GeoSeries([union_cold_clipped], crs="EPSG:3857").to_crs(epsg=4326)
    # Brug samme farve som KML
    # AA  BB  GG  RR
    # 7f ff c0 40 -> plt format #40c0ff (RGB) med alpha 0.5
    poly_gs.plot(ax=ax, color="#40c0ff", alpha=0.5)

# Plot station points colored by temp
sc = ax.scatter(lons, lats, c=vejtemp, cmap="RdYlBu_r", s=40, edgecolor='k', zorder=5)
plt.colorbar(sc, ax=ax, label="Vejtemperatur (°C)")
ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_title(f"IDW Interpolation: Områder for temp < {VEJTEMP_THRESHOLD}°C")
plt.savefig(PNG_TEST, dpi=200)
plt.close()
print(f"✔ Testbillede gemt: {PNG_TEST}")
