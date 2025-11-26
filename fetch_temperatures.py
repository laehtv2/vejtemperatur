#!/usr/bin/env python3


# Tid/updated
timestamp = (
props.get("updated")
or props.get("timestamp")
or props.get("time")
or props.get("last_update")
or props.get("lastUpdated")
)


# Normaliser timestamp hvis muligt
if isinstance(timestamp, (int, float)):
# epoch seconds?
try:
timestamp = datetime.utcfromtimestamp(int(timestamp)).isoformat() + "Z"
except Exception:
timestamp = str(timestamp)
elif isinstance(timestamp, str):
# Lad det være som-streng (antag ISO-format)
timestamp = timestamp


row = {
"station_id": station_id,
"temperature": temperature,
"lon": lon,
"lat": lat,
"timestamp": timestamp,
# Gem hele properties som JSON hvis du vil bruge senere
"properties_json": json.dumps(props, ensure_ascii=False),
}
rows.append(row)
return rows




def main() -> None:
parser = argparse.ArgumentParser(description="Hent vejtemperaturer og gem som CSV.")
parser.add_argument("--url", default=URL, help="GeoJSON endpoint (default: Trafikkort)")
parser.add_argument("--output", default="temperatures.csv", help="Output CSV filnavn")
args = parser.parse_args()


print(f"Henter GeoJSON fra: {args.url}")
geojson = fetch_geojson(args.url)
rows = parse_features(geojson)


if not rows:
print("Ingen features fundet i GeoJSON. Tjek URL eller indhold.")
return


df = pd.DataFrame(rows)
# Sortér gerne for nemheds skyld
if "timestamp" in df.columns:
try:
df["_ts_sort"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
df = df.sort_values("_ts_sort", ascending=False).drop(columns=["_ts_sort"])
except Exception:
pass


df.to_csv(args.output, index=False)
print(f"Gemte {len(df)} rækker til {args.output}")




if __name__ == "__main__":
main()
