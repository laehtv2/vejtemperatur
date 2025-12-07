#!/usr/bin/env python3
"""
Voronoi + buffer metode for at generere KML-områder for "kolde" stationer.

- Cold station: vejtemp < VEJTEMP_THRESHOLD
- Hver cold station får:
    region = Voronoi_cell(upon all stations)  UNION  buffer(10 km)
- Union af alle cold-station-regioner => samlet area (kan være flere adskilte områder)
- Klippes til Danmark (naturalearth via geopandas) og bounding box (LON_MIN..LON_MAX, LAT_MIN..LAT_MAX)
- Output: vejtemp_only.kml (områder hvor temp < threshold)
          vejtemp_dugpunkt.kml (glatføre-risiko: delta < 0, behandles analogt)
- Genererer også test_vejtemp.png
"""
import os
import math
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPoint, box
from shapely.ops import unary_union
import geopandas as gpd
from scipy.spatial import Voronoi
import simplekml
import matplotlib.pyplot as plt

# ---------------------------
# Parametre
# ---------------------------
VEJTEMP_THRESHOLD = 6           # nemt at ændre senere (fx 0)
BUFFER_M = 10000                # 10 km buffer (meter)
N_VOR_POLY_VERTS = 100          # bruges i finite polygon konstruktion (ikke kritisk)

# Bounding box over Danmark (bruges til at trimme Voronoi)
LON_MIN, LON_MAX = 8.0, 12.7
LAT_MIN, LAT_MAX = 54.5, 57.9

# Filnavne (som du ønskede)
KML_VEJTEMP = "vejtemp_only.kml"
KML_RISK = "vejtemp_dugpunkt.kml"
PNG_TEST = "test_vejtemp.png"

# Farver (simplekml bruger aabbggrr hex-format)
COLOR_TEMP = "7f66ccff"
COLOR_RISK_LOW = "7f66ccff"
COLOR_RISK_MED = "7fff9966"
COLOR_RISK_HIGH = "7f8b0000"

# ---------------------------
# Hjælpefunktioner
# ---------------------------

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Convert scipy Voronoi regions to finite polygons.
    Return list of polygons as list of vertex arrays (in voronoi coordinate space).
    Code adapted from common recipes (scipy docs / gist).
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Map containing all ridges for a point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if -1 not in vertices:
            # finite region
            new_regions.append([v for v in vertices])
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v != -1]

        for p2, v1, v2 in ridges:
            if v2 < 0 or v1 < 0:
                v = v1 if v1 >= 0 else v2
                # direction vector from point p1 to p2
                t = vor.points[p2] - vor.points[p1]
                t = t / np.linalg.norm(t)
                # normal
                n = np.array([-t[1], t[0]])
                # far point
                far_point = vor.vertices[v] + n * radius
                new_vertices.append(far_point.tolist())
                new_region.append(len(new_vertices) - 1)

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)

# ---------------------------
# Load data
# ---------------------------
df1 = pd.read_csv("vej_temp_1.csv")
df2 = pd.read_csv("vej_temp_2.csv")
df = pd.concat([df1, df2], ignore_index=True)

# Expect columns "Longitude", "Latitude", "Vej_temp", "Luft_temp"
if not {"Longitude","Latitude","Vej_temp","Luft_temp"}.issubset(df.columns):
    raise SystemExit("CSV skal indeholde kolonnerne: Longitude, Latitude, Vej_temp, Luft_temp")

# Drop rows with missing coords
df = df.dropna(subset=["Longitude","Latitude"])

lons = df["Longitude"].to_numpy()
lats = df["Latitude"].to_numpy()
vejtemp = df["Vej_temp"].to_numpy()
dugpunkt = df["Luft_temp"].to_numpy()

# ---------------------------
# Lav GeoDataFrame projiceret i meter (EPSG:3857) for korrekt buffer i meter
# ---------------------------
gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326")
gdf_m = gdf.to_crs(epsg=3857)  # meter projection

points_m = np.vstack([gdf_m.geometry.x.values, gdf_m.geometry.y.values]).T

# ---------------------------
# Voronoi i meter
# ---------------------------
if len(points_m) < 2:
    raise SystemExit("Skal være mindst 2 stationer for Voronoi")

vor = Voronoi(points_m)
regions, vertices = voronoi_finite_polygons_2d(vor, radius=1e6)  # stor radius for sikkerhed

# Lav shapely polygons for hver region og associer med station index
vor_polys = []
for region in regions:
    poly_coords = vertices[region]
    poly = Polygon(poly_coords)
    vor_polys.append(poly)

# Map index -> vor poly (clipped later)
# Note: vor.regions order matches points order via point_region mapping
point_region_map = {}
for idx in range(len(points_m)):
    point_region_map[idx] = vor_polys[idx]

# ---------------------------
# Buffer (10 km) omkring hver station i meter
# ---------------------------
buffers = [geom.buffer(BUFFER_M) for geom in gdf_m.geometry]

# ---------------------------
# For hver station: hvis cold (vejtemp < threshold), tag region = vor_cell U buffer
# ---------------------------
cold_mask = (vejtemp < VEJTEMP_THRESHOLD)

cold_regions = []
for idx, is_cold in enumerate(cold_mask):
    if not is_cold:
        continue
    vor_poly = point_region_map[idx]
    buf = buffers[idx]
    region = vor_poly.union(buf)
    cold_regions.append(region)

if len(cold_regions) == 0:
    print("Ingen stationer under tærskel — ingen KML genereret (men PNG gemmes).")
    union_cold = None
else:
    union_cold = unary_union(cold_regions)

# ---------------------------
# Håndter glatførerisiko (delta = t - dew), vi ønsker negative deltas
# For risiko laver vi samme fremgangsmåde: stations med delta < 0 betragtes 'cold' for glatføre
# ---------------------------
risk_vals = []
for t, dew in zip(vejtemp, dugpunkt):
    if np.isnan(t) or np.isnan(dew) or t >= 0:
        risk_vals.append(np.nan)
    else:
        risk_vals.append(t - dew)
risk_vals = np.array(risk_vals)
risk_mask = ~np.isnan(risk_vals)  # True for points with risk (delta computed) ; we will require delta < 0 in practice

# For risk we want delta < 0
risk_mask = (risk_vals < 0)

risk_regions = []
for idx, is_risk in enumerate(risk_mask):
    if not is_risk:
        continue
    vor_poly = point_region_map[idx]
    buf = buffers[idx]
    region = vor_poly.union(buf)
    risk_regions.append(region)

union_risk = unary_union(risk_regions) if len(risk_regions) > 0 else None

# ---------------------------
# Begræns til Danmark: brug geopandas' naturalearth 'lowres' data (ingen ekstern fil nødvendig)
# ---------------------------
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
# Naturalearth kan indeholde Grønland mv. Vi begrænser via bbox + country name to be safe.
denmark = world[world["name"] == "Denmark"]
if denmark.empty:
    # Fallback: prøv iso_a3 == 'DNK'
    denmark = world[world["iso_a3"] == "DNK"]

# Opret bbox (i WGS84) og transform til meter CRS
bbox_wgs = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
gdf_bbox = gpd.GeoDataFrame({"geometry":[bbox_wgs]}, crs="EPSG:4326")
bbox_m = gdf_bbox.to_crs(epsg=3857).geometry.iloc[0]

if not denmark.empty:
    denmark_m = denmark.to_crs(epsg=3857).geometry.unary_union
    # Intersect denmark geometry with bbox to exclude Greenland/Faroe if present
    denmark_m = denmark_m.intersection(bbox_m)
else:
    denmark_m = bbox_m  # fallback: blot brug bbox

# Clip union_cold og union_risk til denmark_m
def clip_to_land(geom):
    if geom is None or geom.is_empty:
        return None
    return geom.intersection(denmark_m)

union_cold_clipped = clip_to_land(union_cold)
union_risk_clipped = clip_to_land(union_risk)

# ---------------------------
# Gem KML-filer (konverter tilbage til lon/lat)
# ---------------------------
def geom_to_lonlat_coords(geom):
    """
    Convert a shapely geometry in EPSG:3857 to lon/lat list(s)
    Returns list of polygons; each polygon is list of (lon,lat) tuples
    """
    if geom is None or geom.is_empty:
        return []
    g = gpd.GeoSeries([geom], crs="EPSG:3857").to_crs(epsg=4326).geometry.iloc[0]
    polys = []
    if g.geom_type == "Polygon":
        polys.append(list(g.exterior.coords))
        for interior in g.interiors:
            # Optionally handle holes by adding as separate polygons with opposite winding
            pass
    elif g.geom_type == "MultiPolygon":
        for p in g.geoms:
            polys.append(list(p.exterior.coords))
    else:
        # Other types, ignore
        pass
    # Convert (lon,lat) coords ready for simplekml
    return polys

# ensure output dir exists (current dir usually fine)
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
print(f"✔ KML gemt: {KML_VEJTEMP} (områder for temp < {VEJTEMP_THRESHOLD}°C)")

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
print(f"✔ KML gemt: {KML_RISK} (områder for glatføre delta < 0)")

# ---------------------------
# Gem testbillede (WGS84)
# ---------------------------
fig, ax = plt.subplots(figsize=(8,10))
# Plot Denmark land (bbox-limited)
if not denmark.empty:
    den_wgs = denmark.geometry.iloc[0].intersection(bbox_wgs)
    gpd.GeoSeries([den_wgs]).plot(ax=ax, color="lightgray", edgecolor="k")

# Plot cold areas
if union_cold_clipped is not None and not union_cold_clipped.is_empty:
    poly_gs = gpd.GeoSeries([union_cold_clipped], crs="EPSG:3857").to_crs(epsg=4326)
    poly_gs.plot(ax=ax, color="#66ccff", alpha=0.6)

# Plot station points colored by temp
sc = ax.scatter(lons, lats, c=vejtemp, cmap="RdYlBu_r", s=40, edgecolor='k', zorder=5)
plt.colorbar(sc, ax=ax, label="Vejtemperatur (°C)")
ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_title(f"Stationer og områder for temp < {VEJTEMP_THRESHOLD}°C")
plt.savefig(PNG_TEST, dpi=200)
plt.close()
print(f"✔ Testbillede gemt: {PNG_TEST}")
