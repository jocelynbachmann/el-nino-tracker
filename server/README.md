# Server (Flask)

Serves the processed ocean data, ENSO labels and the MLD forecast as JSON.
It does no drawing: the client turns these numbers into maps and charts.

## Run

```bash
cd server
# Python 3.11+ (3.11–3.13 on Windows); see the main README for Windows and the full-repo venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python setup_aws.py            # once per machine: paste the team's AWS key (same as hub Step 1)
flask --app app run --debug --port 5001
```

Data is streamed from `s3://ocean-hackathon-2026/teams/team-elnino/processed` using the
`hack-team` AWS profile. Nothing is downloaded up front: each request reads only the chunks it
needs (a map month is ~1–3 MB; a clicked point's 1993–2026 series takes 1–3 s the first time,
then it's cached). Rerunning the pipeline or model notebook updates the app automatically
(restart the server to pick up new ocean data; the forecast is re-read on every `/forecast`).

Overrides: `ELNINO_DATA=/some/local/folder` to read a local copy instead, `AWS_PROFILE=name`
for another profile. Set `PRELOAD=1` to load all ocean data into memory (~400 MB) at startup
for faster point queries.

## Endpoints

Base URL when running locally: `http://localhost:5001`

| Endpoint | Returns |
|---|---|
| `GET /health` | `{status, months, forecast}` |
| `GET /metadata` | Variables (label, units, `range`, `anomaly_range`, `colormap`, `has_map`, `has_forecast`), `observed_months`, `forecast_months`, `grid` (start/step/count for latitude and longitude), `bounds`, `model_region` |
| `GET /el-nino` | `[{month, oni, phase, provisional}]`. ONI is NOAA's Oceanic Niño Index; `phase` is `el_nino`, `la_nina` or `neutral` |
| `GET /maps/<variable>/<YYYY-MM>` | One month's map: `latitudes` (south→north), `longitudes` (west→east), `values[row][col]` (`null` = land), `source` (`observed`/`forecast`), `color_range`. Options: `?anomaly=true` (difference from normal), `?area=model-region` (crop to 46–52°N, 134–120°W) |
| `GET /timeseries/location?lat=48.5&lon=-126&variable=mixed-layer-depth` | Monthly series at the nearest grid cell: `[{month, value, anomaly}]`, plus `is_ocean` |
| `GET /timeseries/region/<variable>` | Same, averaged over the model region. Also accepts the wind variables |
| `GET /forecast` | Contents of `forecast_mld.json`: per month `mld`, `mld_anomaly`, `normal`, `lower`/`upper`, `source` |

Variables: `temperature`, `salinity`, `sea-surface-height`, `mixed-layer-depth`,
`current-east`, `current-north`. Regional time series also: `wind-speed`, `wind-north`,
`wind-east`, `wind-stress`.

Future months: only `mixed-layer-depth` has maps (normal MLD for that month + the model's
regional forecast anomaly, `source: "forecast"`), and only for months in `forecast_mld.json`.

A full-area map is about 42,000 values (~300 KB of JSON); `area=model-region` is about 5,000.

## Forecast file contract (written by the model notebook)

Save as `forecast_mld.json` in the processed folder:

```json
{
  "issued": "2026-10",
  "model": "linear_regression_v1",
  "months": [
    {"month": "2026-10", "mld": 42.1, "mld_anomaly": -3.2, "lower": 35.0, "upper": 49.0, "source": "model"},
    {"month": "2027-05", "mld": 18.0, "mld_anomaly": 0.0, "lower": 9.0, "upper": 27.0, "source": "climatology"}
  ]
}
```

`mld` is the regional mean in metres (normal + anomaly); `lower`/`upper` is the uncertainty band;
`source` is `model` for months driven by Copernicus forecasts (1–6) and `climatology` beyond that.
`/forecast` re-reads the file on every call, so a new model run shows up without a restart.
