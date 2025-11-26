# Trafikinfo Vejtemperatur Export


Et enkelt Python-script der henter vejtemperaturer (GeoJSON) fra Trafikkort (Vejdirektoratet) og gemmer resultatet som en CSV-fil.


## Hvad scriptet gør


- Henter GeoJSON fra:
`https://storage.googleapis.com/trafikkort-data/geojson/25832/temperatures.point.json`
- Parserer hvert feature og udtrækker:
- station_id (hvis tilgængelig)
- temperatur
- koordinater (lon, lat)
- tidspunkt / sidste opdatering
- Gemmer output som CSV (f.eks. `temperatures.csv`).


## Krav


- Python 3.10+
- Afhængigheder i `requirements.txt` (requests, pandas)


## Installation


```bash
python -m venv venv
source venv/bin/activate # macOS / Linux
venv\Scripts\activate # Windows
pip install -r requirements.txt
