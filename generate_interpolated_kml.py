#!/usr/bin/env python3
"""
IDW (Inverse Distance Weighting) interpolering for at generere KML-områder
for "kolde" vejtemperaturområder og glatførerisiko.

- Vejtemp interpolering: Bruger rå Vejtemp.
- Risiko interpolering: Bruger Delta (Vejtemp - DMI Dugpunkt).
- Risiko KML: Viser områder, hvor (Interpoleret Vejtemp < 0°C) OG (Interpoleret Delta < 0°C).
- Interpolerer KUN over land.
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
VEJTEMP_THRESHOLD = 7.0           # Tærskel for farveskala (lyseblå ved 7.0)
MIN_TEMP_COLOR = 7.0              # Start for farveskalaen
MAX_TEMP_COLOR = -3.0             # Slut for farveskalaen (mørkeblå ved -3.0)

RISK_TEMP_THRESHOLD = 0.0         # Vejtemp skal være under 0°C for at have risiko
RISK_DELTA_THRESHOLD = 0.0        # Delta (T_vej - T_dug) skal være under 0°C for at have risiko

GRID_RESOLUTION = 200             
IDW_POWER = 2                     

# Bounding box over Danmark
LON_MIN, LON_MAX = 8.0, 15.5      
LAT_MIN, LAT_MAX = 54.5, 57.9

# Filnavne
KML_VEJTEMP = "vejtemp_only.kml"
KML_RISK = "vejtemp_dugpunkt.kml"
PNG_TEMP_MAP = "vejtemp_map.png"      
PNG_RISK_MAP = "risk_map.png"         
NE_HIGHRES_FILE = "ne_10m_admin_0_countries.shp" 

# Farver (AABBGGRR hex-format for simplekml)
COLOR_RISK_HIGH = "7f0000ff"      # Transparent (7f) Rød

# ---------------------------
# Hjælpefunktioner
# ---------------------------

def idw_interpolation(stations: np.ndarray, values: np.ndarray, grid_points: np.ndarray, power: int = 2) -> np.ndarray:
    """ Inverse Distance Weighting (IDW) interpolation. """
    distances = np.sqrt(
        (grid_points[:, None, 0] - stations[None, :, 0])**2 +
        (grid_points[:, None, 1] - stations[None, :, 1])**2
    )
    distances[distances == 0] = 1e-6  
    weights = 1.0 / (distances ** power)
    return np.sum(values * weights, axis=1) / np.sum(weights, axis=1)

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)

def temp_to_kml_color(temp: float, vmin: float, vmax: float) -> str:
    """ Oversætter temperatur til en KML farvestreng (AABBGGRR). """
    norm = colors.Normalize(vmin=vmax, vmax=vmin) 
    scaled_temp = norm(temp)
    cmap = cm.get_cmap('YlGnBu_r') 
    rgb = cmap(scaled_temp)[:3]  
    
    rr = int(rgb[0] * 255)
    gg = int(rgb[1] * 255)
    bb = int(rgb[2] * 255)
    alpha = 127
    return f"{alpha:02x}{bb:02x}{gg:02x}{rr:02x}"

def create_interpolated_polygons(interpolated_values: np.ndarray, x_range: np.ndarray, y_range: np.ndarray, mask_grid: np.ndarray) -> list[tuple[Polygon, float]]:
    """ Genererer Shapely Polygoner (grid-celler) KUN for punkter i masken. """
    
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

def create_single_union_polygon(grid_points_m_input, mask_grid_input, cell_buffer_input):
    """ Genererer et enkelt, samlet polygon baseret på maske (bruges til risiko). """
    
    gdf_m_input = gpd.GeoDataFrame(
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
    
    if geom.geom_type == "MultiPolygon":
        polys_coords = []
        for p in geom.geoms:
            g = gpd.GeoSeries([p], crs="EPSG:3857").to_crs(epsg=4326).geometry.iloc[0]
            if g.geom_type == "Polygon": polys_coords.append(list(g.exterior.coords))
        return polys_coords
    else:
        g = gpd.GeoSeries([geom], crs="EPSG:3857").to_crs(epsg=4326).geometry.iloc[0]
        if g.geom_type == "Polygon": return [list(g.exterior.coords)]
        return []
        
# ---------------------------
# MAIN LOGIK
# ---------------------------

# Load data
df1 = pd.read_csv("vej_temp_1.csv")
df2 = pd.read_csv("vej_temp_2.csv")
df = pd.concat([df1, df2], ignore_index=True)

# Vigtigt: Drop NaN i Vej_temp og den nye Dewpoint
df = df.dropna(subset=["Longitude","Latitude", "Vej_temp", "Dewpoint"])

lons = df["Longitude"].to_numpy()
lats = df["Latitude"].to_numpy()
vejtemp = df["Vej_temp"].to_numpy()
dewpoint = df["Dewpoint"].to_numpy() # NYT: Bruger DMI Dugpunkt

# Beregn Delta (Vejtemp - Dugpunkt) for interpolering
risk_deltas = vejtemp - dewpoint
valid_data_mask = ~np.isnan(risk_deltas) # Maske for stationer der har valid data (ingen NaN)

# Konverter til GeoDataFrame og projicer til meter (EPSG:3857)
gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
gdf_m = gdf.to_crs(epsg=3857)  # meter projection
points_m_valid = np.vstack([gdf_m.geometry.x.values, gdf_m.geometry.y.values]).T[valid_data_mask]
vejtemp_valid = vejtemp[valid_data_mask]
risk_deltas_valid = risk_deltas[valid_data_mask]

# --- Filter station data ned til kun de gyldige datapunkter ---
df_valid = df[valid_data_mask].copy() # Bruges til plotting af punkter
lons_valid = df_valid["Longitude"].to_numpy()
lats_valid = df_valid["Latitude"].to_numpy()


# ---------------------------
# Forbered Danmark og Grid
# ---------------------------
# Hent Danmark (Landmaske)
try:
    world = gpd.read_file(NE_HIGHRES_FILE)
except Exception:
    raise SystemExit(f"Fejl: Kunne ikke læse {NE_HIGHRES_FILE}. Sikr dig, at filen er hentet og committet i repoet.")

denmark = world[world["SOVEREIGNT"] == "Denmark"]
if denmark.empty: denmark = world[world["ADM0_A3"] == "DNK"]
if denmark.empty: denmark = world[world["ADMIN"] == "Denmark"]
if denmark.empty: raise SystemExit("Fejl: Kunne ikke finde 'Denmark' i high-res datasættet.")

# Opret og klip med Bounding Box
bbox_wgs = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
gdf_bbox = gpd.GeoDataFrame({"geometry":[bbox_wgs]}, crs="EPSG:4326")
bbox_m = gdf_bbox.to_crs(epsg=3857).geometry.iloc[0]

# Intersect denmark geometry med bbox for at få den endelige landmaske
denmark_m = denmark.to_crs(epsg=3857).geometry.unary_union
denmark_m = denmark_m.intersection(bbox_m)

# Definer grid-området baseret på bounding box
x_min_grid, y_min_grid, x_max_grid, y_max_grid = bbox_m.bounds 
x_range = np.linspace(x_min_grid, x_max_grid, GRID_RESOLUTION)
y_range = np.linspace(y_min_grid, y_max_grid, GRID_RESOLUTION)
X, Y = np.meshgrid(x_range, y_range)
grid_points_m = np.vstack([X.ravel(), Y.ravel()]).T

# --- Masker Grid-punkter der ligger over HAV (til interpolation) ---
grid_points_gdf = gpd.GeoDataFrame(geometry=[Point(x,y) for x,y in grid_points_m], crs="EPSG:3857")
grid_over_land_mask = grid_points_gdf.intersects(denmark_m).to_numpy()
points_over_land = grid_points_m[grid_over_land_mask]

# ---------------------------
# IDW Interpolering af Vejtemp og Delta (over land)
# ---------------------------

# 1. Vejtemp Interpolering
print("Starter IDW interpolering for Vejtemp over land...")
interpolated_temp_land = idw_interpolation(points_m_valid, vejtemp_valid, points_over_land, IDW_POWER)
interpolated_temp_full = np.full(grid_points_m.shape[0], np.nan)
interpolated_temp_full[grid_over_land_mask] = interpolated_temp_land

# 2. Delta (Risiko) Interpolering
print("Starter IDW interpolering for Delta (Vejtemp - Dugpunkt) over land...")
interpolated_risk_delta_land = idw_interpolation(points_m_valid, risk_deltas_valid, points_over_land, IDW_POWER)
interpolated_risk_delta_full = np.full(grid_points_m.shape[0], np.nan)
interpolated_risk_delta_full[grid_over_land_mask] = interpolated_risk_delta_land

# ---------------------------
# Generering af KML-masker
# ---------------------------

# 1. Vejtemp KML Maske (Flydende Farver)
temp_color_mask_full = (interpolated_temp_full < VEJTEMP_THRESHOLD) & grid_over_land_mask
cold_polygons_w_values = create_interpolated_polygons(
    interpolated_temp_full, x_range, y_range, temp_color_mask_full
)
print(f"✔ Vejtemp områder oprettet via IDW ({len(cold_polygons_w_values)} grid-celler under {VEJTEMP_THRESHOLD}°C).")

# 2. Risiko KML Maske (Overlappende logik)
# Risiko = (Vejtemp < 0°C) AND (Delta < 0°C) AND (Over Land)
final_risk_mask = (interpolated_temp_full < RISK_TEMP_THRESHOLD) & \
                  (interpolated_risk_delta_full < RISK_DELTA_THRESHOLD) & \
                  grid_over_land_mask

grid_spacing_m = x_range[1] - x_range[0]  
cell_buffer = grid_spacing_m * 0.7 
# RETTET FUNKTIONSKALD: Fjerner overflødigt argument
union_risk = create_single_union_polygon(grid_points_m, final_risk_mask, cell_buffer)

# Funktion til at klippe geometri (bruges til at klippe den samlede risiko polygon)
def clip_to_land(geom):
    if geom is None or geom.is_empty: return None
    return geom.intersection(denmark_m)

union_risk_clipped = clip_to_land(union_risk)
print("✔ Risiko områder oprettet via IDW (Temp < 0°C OG Delta < 0°C).")

# ---------------------------
# Gem KML-filer
# ---------------------------

ensure_dir(KML_VEJTEMP)
ensure_dir(KML_RISK)

# Vejtemp KML (DYNAMISK FARVE)
kml_temp = simplekml.Kml()
for poly_m, temp_val in cold_polygons_w_values:
    # Klip hver celle til den fine landmaske
    clipped_poly = poly_m.intersection(denmark_m) 
    if clipped_poly is None or clipped_poly.is_empty: continue
        
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
print(f"✔ KML gemt: {KML_RISK} (interpolerede områder for glatføre T<0 & Delta<0)")

# ---------------------------
# Gem testbilleder (To filer - Sikker Gemning)
# ---------------------------

# Konverter grid-koordinater til WGS84 for plotting
grid_points_wgs = gpd.GeoDataFrame(geometry=[Point(x,y) for x,y in grid_points_m], crs="EPSG:3857").to_crs(epsg=4326)
X_wgs = grid_points_wgs.geometry.x.to_numpy().reshape((GRID_RESOLUTION, GRID_RESOLUTION))
Y_wgs = grid_points_wgs.geometry.y.to_numpy().reshape((GRID_RESOLUTION, GRID_RESOLUTION))
Z_temp = interpolated_temp_full.reshape((GRID_RESOLUTION, GRID_RESOLUTION))
Z_risk_delta = interpolated_risk_delta_full.reshape((GRID_RESOLUTION, GRID_RESOLUTION)) 

den_wgs = denmark.geometry.iloc[0].intersection(bbox_wgs)


# --- PLOT 1: VEJTEMPERATUR ---
fig, ax = plt.subplots(figsize=(8,10))
gpd.GeoSeries([den_wgs]).plot(ax=ax, color="lightgray", edgecolor="k")

# Hvis der er data at plotte (Konturfyld)
if not np.all(np.isnan(Z_temp)):
    levels = np.linspace(MAX_TEMP_COLOR, MIN_TEMP_COLOR, 11)
    cp = ax.contourf(X_wgs, Y_wgs, Z_temp, levels=levels, cmap='YlGnBu_r', extend='min', alpha=0.7, zorder=2)
    ax.contour(X_wgs, Y_wgs, Z_temp, levels=levels, colors='k', linewidths=0.2, alpha=0.5, zorder=3)
    plt.colorbar(cp, ax=ax, label="Interpoleret Vejtemperatur (°C)", orientation='vertical', pad=0.02)
    
# Brug lons_valid og vejtemp_valid, da de har Dewpoint data
sc_temp = ax.scatter(lons_valid, lats_valid, c=vejtemp_valid, cmap="RdYlBu_r", s=40, edgecolor='k', zorder=5) 
ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_title(f"IDW Interpolation: Vejtemperatur {MIN_TEMP_COLOR}°C til {MAX_TEMP_COLOR}°C")
plt.savefig(PNG_TEMP_MAP, dpi=200) # Gemmes uanset hvad
plt.close()
print(f"✔ Testbillede gemt: {PNG_TEMP_MAP}")


# --- PLOT 2: GLATFØRERISIKO ---
fig, ax = plt.subplots(figsize=(8,10))
gpd.GeoSeries([den_wgs]).plot(ax=ax, color="lightgray", edgecolor="k")

# Vis de samlede risiko-områder
if union_risk_clipped is not None and not union_risk_clipped.is_empty:
    poly_gs = gpd.GeoSeries([union_risk_clipped], crs="EPSG:3857").to_crs(epsg=4326)
    poly_gs.plot(ax=ax, color="#ff0000", alpha=0.5, zorder=2)
    
# Scatterplot og Colorbar for station data (delta)
sc_risk = ax.scatter(lons_valid, lats_valid, c=risk_deltas_valid, cmap="Reds", s=40, edgecolor='k', zorder=5) 
plt.colorbar(sc_risk, ax=ax, label="Vejtemp - Dugpunkt Delta (°C)", orientation='vertical', pad=0.02) 

ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_title(f"IDW Risiko: Overlap (T_vej < {RISK_TEMP_THRESHOLD}°C OG Delta < {RISK_DELTA_THRESHOLD}°C)")
plt.savefig(PNG_RISK_MAP, dpi=200) # Gemmes uanset hvad
plt.close()
print(f"✔ Testbillede gemt: {PNG_RISK_MAP}")
