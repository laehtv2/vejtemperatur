#!/usr/bin/env python3
"""
IDW (Inverse Distance Weighting) interpolering for at generere KML-områder
for "kolde" vejtemperaturområder og glatførerisiko.

- Genererer to testbilleder: Et for Vejtemperatur og et for Risiko.
- Skaber KML-polygon(er) med flydende farveskala for temperatur.
- Klippes til Danmark (naturalearth highres) og bounding box.

Output:
    vejtemp_only.kml (interpolerede områder med flydende farveskala 0°C til -10°C)
    vejtemp_dugpunkt.kml (interpolerede glatføre-risiko områder: delta < 0)
    vejtemp_map.png (visuelt kort for vejtemperatur) <--- NYT FILNAVN
    risk_map.png (visuelt kort for glatførerisiko) <--- NYT FILNAVN
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
import matplotlib.cm as cm
import matplotlib.colors as colors

# ---------------------------
# Parametre
# ---------------------------
VEJTEMP_THRESHOLD = 0.0           # Tærskel for "kold" vejtemp (°C) - vi farver under denne.
MIN_TEMP_COLOR = 0.0              # Temperatur for den lyseste farve (lyseblå)
MAX_TEMP_COLOR = -10.0            # Temperatur for den mørkeste farve (mørkeblå)

GRID_RESOLUTION = 200             # Opløsning af interpolationsgrid (f.eks. 200x200)
IDW_POWER = 2                     # Inverse Distance Weighting Power (typisk 2)

# Bounding box over Danmark (nu udvidet til Bornholm)
LON_MIN, LON_MAX = 8.0, 15.5      
LAT_MIN, LAT_MAX = 54.5, 57.9

# Filnavne
KML_VEJTEMP = "vejtemp_only.kml"
KML_RISK = "vejtemp_dugpunkt.kml"
PNG_TEMP_MAP = "vejtemp_map.png"      # NYT FILNAVN
PNG_RISK_MAP = "risk_map.png"         # NYT FILNAVN
NE_HIGHRES_FILE = "ne_10m_admin_0_countries.shp" 

# Farver (AABBGGRR hex-format for simplekml)
COLOR_RISK_HIGH = "7f0000ff"      # Transparent (7f) Rød

# ---------------------------
# Interpolationsfunktioner
# ---------------------------

def idw_interpolation(stations: np.ndarray, values: np.ndarray, grid_points: np.ndarray, power: int = 2) -> np.ndarray:
    """ Inverse Distance Weighting (IDW) interpolation. """
    
    distances = np.sqrt(
        (grid_points[:, None, 0] - stations[None, :, 0])**2 +
        (grid_points[:, None, 1] - stations[None, :, 1])**2
    )
    
    distances[distances == 0] = 1e-6 
    weights = 1.0 / (distances ** power)
    
    interpolated_values = np.sum(values * weights, axis=1) / np.sum(weights, axis=1)
    
    return interpolated_values

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)

def temp_to_kml_color(temp: float, vmin: float, vmax: float) -> str:
    """ Oversætter temperatur til en KML farvestreng (AABBGGRR). """
    
    norm = colors.Normalize(vmin=vmax, vmax=vmin) # vmin=-10, vmax=0
    scaled_temp = norm(temp)

    cmap = cm.get_cmap('YlGnBu_r') 
    rgb = cmap(scaled_temp)[:3]  
    
    rr = int(rgb[0] * 255)
    gg = int(rgb[1] * 255)
    bb = int(rgb[2] * 255)
    alpha = 127 # 7f i hex for 50% gennemsigtighed
    
    kml_color = f"{alpha:02x}{bb:02x}{gg:02x}{rr:02x}"
    return kml_color

def create_interpolated_polygons(interpolated_values: np.ndarray, x_range: np.ndarray, y_range: np.ndarray, threshold: float, threshold_type: str) -> list[tuple[Polygon, float]]:
    """ Genererer en liste af Shapely Polygoner (grid-celler) og deres interpolerede værdier. """
    
    if threshold_type == 'less_than':
        mask_grid = interpolated_values < threshold
    else:
        return []

    polygons_with_values = []
    
    num_cols = len(x_range)
    num_rows = len(y_range)
    
    for i in range(num_rows - 1): 
        for j in range(num_cols - 1): 
            flat_index = i * num_cols + j
            
            if mask_grid[flat_index]:
                x_ll, y_ll = x_range[j], y_range[i]
                x_ur, y_ur = x_range[j+1], y_range[i+1]
                
                cell_poly = box(x_ll, y_ll, x_ur, y_ur)
                value = interpolated_values[flat_index]
                polygons_with_values.append((cell_poly, value))
                
    return polygons_with_values

def create_single_union_polygon(interpolated_values_input, grid_points_m_input, threshold_input, threshold_type_input, cell_buffer_input):
    """ Genererer et enkelt, samlet polygon (bruges til risiko). """
    if threshold_type_input == 'less_than':
        mask_grid_input = interpolated_values_input < threshold_input
    elif threshold_type_input == 'greater_than':
        mask_grid_input = interpolated_values_input > threshold_input
    else:
        return None

    gdf_m_input = gpd.GeoDataFrame(
        {'value': interpolated_values_input[mask_grid_input]},
        geometry=[Point(x, y) for x, y in grid_points_m_input[mask_grid_input]],
        crs='EPSG:3857'
    )

    if gdf_m_input.empty:
        return None
    else:
        polygons_input = [geom.buffer(cell_buffer_input, join_style=2) for geom in gdf_m_input.geometry]
        return unary_union(polygons_input)

def geom_to_lonlat_coords(geom):
    """ Konverter shapely geometrier i EPSG:3857 til (lon,lat) koordinater. """
    if geom is None or geom.is_empty:
        return []
    
    # Håndter MultiPolygoner
    if geom.geom_type == "MultiPolygon":
        polys_coords = []
        for p in geom.geoms:
            g = gpd.GeoSeries([p], crs="EPSG:3857").to_crs(epsg=4326).geometry.iloc[0]
            if g.geom_type == "Polygon":
                polys_coords.append(list(g.exterior.coords))
        return polys_coords
    else: # Antag Polygon
        g = gpd.GeoSeries([geom], crs="EPSG:3857").to_crs(epsg=4326).geometry.iloc[0]
        if g.geom_type == "Polygon":
            return [list(g.exterior.coords)]
        return []

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
risk_vals = df.apply(
    lambda row: row['Vej_temp'] - row['Luft_temp'] if row['Vej_temp'] < 0 else np.nan,
    axis=1
).to_numpy()

# Konverter til GeoDataFrame og projicer til meter (EPSG:3857)
gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
gdf_m = gdf.to_crs(epsg=3857)  # meter projection
points_m = np.vstack([gdf_m.geometry.x.values, gdf_m.geometry.y.values]).T

# ---------------------------
# Grid generation & Interpolering
# ---------------------------

# Definer grid-området baseret på den udvidede bounding box
bbox_wgs_for_grid = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
gdf_bbox_for_grid = gpd.GeoDataFrame({"geometry":[bbox_wgs_for_grid]}, crs="EPSG:4326")
bbox_m_for_grid = gdf_bbox_for_grid.to_crs(epsg=3857).geometry.iloc[0]

x_min_grid, y_min_grid, x_max_grid, y_max_grid = bbox_m_for_grid.bounds

x_range = np.linspace(x_min_grid, x_max_grid, GRID_RESOLUTION)
y_range = np.linspace(y_min_grid, y_max_grid, GRID_RESOLUTION)

X, Y = np.meshgrid(x_range, y_range)
grid_points_m = np.vstack([X.ravel(), Y.ravel()]).T

# 1. Vejtemp Interpolering
print("Starter IDW interpolering for Vejtemp...")
interpolated_temp = idw_interpolation(points_m, vejtemp, grid_points_m, IDW_POWER)
cold_polygons_w_values = create_interpolated_polygons(
    interpolated_temp, x_range, y_range, VEJTEMP_THRESHOLD, 'less_than'
)
print(f"✔ Vejtemp områder oprettet via IDW ({len(cold_polygons_w_values)} grid-celler under {VEJTEMP_THRESHOLD}°C).")


# 2. Risiko Interpolering
valid_risk_mask = ~np.isnan(risk_vals)
if np.sum(valid_risk_mask) > 1:
    print("Starter IDW interpolering for Glatførerisiko (delta < 0)...")
    interpolated_risk_delta = idw_interpolation(points_m[valid_risk_mask], risk_vals[valid_risk_mask], grid_points_m, IDW_POWER)
    
    grid_spacing_m = x_range[1] - x_range[0] 
    cell_buffer = grid_spacing_m * 0.7 
    union_risk = create_single_union_polygon(interpolated_risk_delta, grid_points_m, 0, 'less_than', cell_buffer)
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
    raise SystemExit(f"Fejl: Kunne ikke læse {NE_HIGHRES_FILE}. Sikr dig, at filen er hentet og committet i repoet.")

# Find Danmark
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

# Funktion til at klippe geometri
def clip_to_land(geom):
    if geom is None or geom.is_empty:
        return None
    return geom.intersection(denmark_m)

union_risk_clipped = clip_to_land(union_risk)

# ---------------------------
# Gem KML-filer
# ---------------------------

ensure_dir(KML_VEJTEMP)
ensure_dir(KML_RISK)

# Vejtemp KML (DYNAMISK FARVE)
kml_temp = simplekml.Kml()
for poly_m, temp_val in cold_polygons_w_values:
    clipped_poly = clip_to_land(poly_m)
    if clipped_poly is None or clipped_poly.is_empty:
        continue
        
    polys_coords_lonlat = geom_to_lonlat_coords(clipped_poly)
    kml_color = temp_to_kml_color(temp_val, MIN_TEMP_COLOR, MAX_TEMP_COLOR)
    
    for coords in polys_coords_lonlat:
        pol = kml_temp.newpolygon()
        pol.outerboundaryis.coords = coords
        pol.style.polystyle.color = kml_color 
        pol.style.polystyle.fill = 1
        pol.style.polystyle.outline = 0
        
kml_temp.save(KML_VEJTEMP)
print(f"✔ KML gemt: {KML_VEJTEMP} (farvet interpoleret map {MIN_TEMP_COLOR}°C til {MAX_TEMP_COLOR}°C)")

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
# Gem testbilleder (To filer)
# ---------------------------

# Konverter grid-koordinater til WGS84 for plotting
grid_points_wgs = gpd.GeoDataFrame(geometry=[Point(x,y) for x,y in grid_points_m], crs="EPSG:3857").to_crs(epsg=4326)
X_wgs = grid_points_wgs.geometry.x.to_numpy().reshape((GRID_RESOLUTION, GRID_RESOLUTION))
Y_wgs = grid_points_wgs.geometry.y.to_numpy().reshape((GRID_RESOLUTION, GRID_RESOLUTION))
Z_temp = interpolated_temp.reshape((GRID_RESOLUTION, GRID_RESOLUTION))
Z_risk = interpolated_risk_delta.reshape((GRID_RESOLUTION, GRID_RESOLUTION)) if np.sum(valid_risk_mask) > 1 else None
den_wgs = denmark.geometry.iloc[0].intersection(bbox_wgs)


### PLOT 1: VEJTEMPERATUR (MAP 0°C TIL -10°C)
fig, ax = plt.subplots(figsize=(8,10))
gpd.GeoSeries([den_wgs]).plot(ax=ax, color="lightgray", edgecolor="k")

if cold_polygons_w_values:
    # Konturskalaen er sat fra -10 til 0
    levels = np.linspace(MAX_TEMP_COLOR, MIN_TEMP_COLOR, 11) 
    
    # Konturfyld
    cp = ax.contourf(X_wgs, Y_wgs, Z_temp, levels=levels, cmap='YlGnBu_r', extend='min', alpha=0.7, zorder=2)
    ax.contour(X_wgs, Y_wgs, Z_temp, levels=levels, colors='k', linewidths=0.2, alpha=0.5, zorder=3)

    plt.colorbar(cp, ax=ax, label="Interpoleret Vejtemperatur (°C)", orientation='vertical', pad=0.02)
    
ax.scatter(lons, lats, c=vejtemp, cmap="RdYlBu_r", s=40, edgecolor='k', zorder=5)
ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_title(f"IDW Interpolation: Vejtemperatur {MIN_TEMP_COLOR}°C til {MAX_TEMP_COLOR}°C")
plt.savefig(PNG_TEMP_MAP, dpi=200)
plt.close()
print(f"✔ Testbillede gemt: {PNG_TEMP_MAP}")


### PLOT 2: GLATFØRERISIKO (DELTA < 0)
fig, ax = plt.subplots(figsize=(8,10))
gpd.GeoSeries([den_wgs]).plot(ax=ax, color="lightgray", edgecolor="k")

if union_risk_clipped is not None and not union_risk_clipped.is_empty:
    poly_gs = gpd.GeoSeries([union_risk_clipped], crs="EPSG:3857").to_crs(epsg=4326)
    # Brug den røde farve (fra COLOR_RISK_HIGH = "7f0000ff" -> #ff0000 i Matplotlib RGB)
    poly_gs.plot(ax=ax, color="#ff0000", alpha=0.5, zorder=2)
    
# Definer scatterplot-variabelen her (f.eks. sc_risk)
# Her plotter vi vejstationerne farvet efter glatførerisiko (risk_vals)
sc_risk = ax.scatter(lons, lats, c=risk_vals, cmap="Reds", s=40, edgecolor='k', zorder=5) # <--- NY DEFINITION
plt.colorbar(sc_risk, ax=ax, label="Vejtemp - Dugpunkt Delta (°C)", orientation='vertical', pad=0.02) # <--- BRUG sc_risk

ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_title("IDW Risiko: Områder med Glatførerisiko (Delta < 0)")
plt.savefig(PNG_RISK_MAP, dpi=200)
plt.close()
print(f"✔ Testbillede gemt: {PNG_RISK_MAP}")
