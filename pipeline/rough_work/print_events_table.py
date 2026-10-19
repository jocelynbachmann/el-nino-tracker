import json, s3fs, pandas as pd
fs = s3fs.S3FileSystem(profile="hack-team")   # on the hub: s3fs.S3FileSystem()
ev = json.load(fs.open("s3://ocean-hackathon-2026/teams/team-elnino/processed/analysis/events.json"))
rows = []
for e in ev["events"]:
    for v, x in e["effects"].items():
        if x.get("too_early"):
            continue
        mo = lambda d: None if d is None else round(d / 30.4)
        rows.append({"event": f"{e['label']} {'El Niño' if e['phase'] == 'el_nino' else 'La Niña'}",
                     "variable": v, "% affected": x["pct_points_changed"], "direction": x["main_direction"],
                     "onset (mo)": mo(x["median_onset_days"]), "lasted (mo)": mo(x["median_duration_days"]),
                     "peak SD": x["median_peak_sd"], "anomaly": x["median_anomaly"], "units": x["units"]})
print(pd.DataFrame(rows).to_string(index=False))