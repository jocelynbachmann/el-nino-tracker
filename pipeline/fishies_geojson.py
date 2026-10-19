#!/usr/bin/env python3
# converts csv to geojson (supported by maplibre)

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "fish_data" / "herring-spawning.csv"
OUT = ROOT / "client" / "public" / "herring-spawn.geojson"


def num(v):
    return None if v in ("", "NA") else float(v)


features = []
with open(SRC, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        lon, lat = num(r["Longitude"]), num(r["Latitude"])
        if lon is None or lat is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(lon, 4), round(lat, 4)],
                },
                "properties": {
                    "year": int(r["Year"]),
                    "name": r["LocationName"],
                    "start": None if r["StartDate"] == "NA" else r["StartDate"],
                    "end": None if r["EndDate"] == "NA" else r["EndDate"],
                },
            }
        )

OUT.write_text(
    json.dumps(
        {"type": "FeatureCollection", "features": features}, separators=(",", ":")
    )
)
print(f"{len(features)} points -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
