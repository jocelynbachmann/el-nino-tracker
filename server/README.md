# Server (Flask)

Serves the processed ocean data, El Niño history analysis, ENSO labels and the MLD forecast as JSON.
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
needs (a map is ~1–3 MB; a clicked point's 1993–2026 series takes 1–3 s the first time,
then it's cached). Rerunning the pipeline or model updates the app automatically
(restart the server to pick up new ocean data or analysis; the forecasts are re-read on every
`/forecast`).

Overrides: `ELNINO_DATA=/some/local/folder` to read a local copy instead, `AWS_PROFILE=name`
for another profile. `PRELOAD_DAILY=1` loads the daily maps into memory at startup so
scrubbing the timeline is fast; `PRELOAD=1` does the same for the monthly data (~400 MB) for
faster point queries.

## Endpoints

Base URL when running locally: `http://localhost:5001`

| Endpoint | Returns |
|---|---|
| `GET /health` | `{status, months, forecast}` |
| `GET /metadata` | Variables (label, units, `range`, `anomaly_range`, `colormap`, `has_map`, `has_forecast`), `time_step` (`day` once the daily data exists), `observed_days`, `observed_months`, `forecast_months`, `grid`, `bounds`, `model_region` |
| `GET /el-nino` | `[{month, oni, phase, provisional}]`. ONI is NOAA's Oceanic Niño Index; `phase` is `el_nino`, `la_nina` or `neutral` |
| `GET /maps/<variable>/<YYYY-MM-DD>` | One day's map (daily data, what the app uses). Same shape and options as the monthly map below |
| `GET /maps/<variable>/<YYYY-MM>` | One month's map: `latitudes` (south→north), `longitudes` (west→east), `values[row][col]` (`null` = land), `source` (`observed`/`forecast`), `color_range`. Options: `?anomaly=true` (difference from normal), `?area=model-region` (crop to 46–52°N, 134–120°W) |
| `GET /timeseries/location?lat=48.5&lon=-126&variable=mixed-layer-depth` | Monthly series at the nearest grid cell: `[{month, value, anomaly}]`, plus `is_ocean` |
| `GET /timeseries/region/<variable>` | Same, averaged over the model region. Also accepts the wind variables |
| `GET /forecast` | Contents of `forecast_mld.json` (older region-average forecast), plus `grid`: the months that have per-point maps in `forecast_mld_grid.json` |
| `GET /forecast/maps/<YYYY-MM>` | Per-point MLD forecast for one month (what the Predictions switch shows), with `lower`/`upper` (10th–90th percentile). `?anomaly=true` for difference from normal |
| `GET /events` | Every El Niño and La Niña since 2002 and what each did here, per variable; region-wide unusual spells (from `pipeline/analyze_events.py`) |
| `GET /events/<event-id>/maps/<variable>/<field>` | One event's effect at every point. `field`: `onset` (months after the event began), `duration` (months), `strength` (in SD). Also `no_change`: points checked with no clear change |
| `GET /typical/<phase>/maps/<variable>` | `el-nino` or `la-nina`: how consistently each point changed the same way across past events (−1..1) |

Variables: `temperature`, `salinity`, `sea-surface-height`, `mixed-layer-depth`,
`current-east`, `current-north`. Regional time series also: `wind-speed`, `wind-north`,
`wind-east`, `wind-stress`.

Future months: only mixed layer depth has predictions, from `/forecast/maps/<YYYY-MM>`.

A full-area map is about 42,000 values (~300 KB of JSON); `area=model-region` is about 5,000.

## Forecast files

- `forecast_mld_grid.json` (used by the app): written by `model/train_mld_forecast.py`. One
  entry per month with `mld`, `lower`, `upper` and `normal` grids on the model's 1/4° grid
  (south→north, west→east, `null` = land), plus `latitudes`, `longitudes`, `issued`, `model`.
  The server reads it from the processed folder, or from `model/output/` if that copy is missing.
- `forecast_mld.json` (older): region-average forecast from an archived script, served as is by
  `/forecast` but not shown in the app.

Both are re-read on every `/forecast` call, so a new model run shows up without a restart.
