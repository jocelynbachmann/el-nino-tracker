# El Niño Tracker

A web map of how El Niño and La Niña have played out in the waters off Washington and
Vancouver Island, in plain language for fishers, coastal communities and everyone else.
It shows daily ocean conditions since 2002, what each El Niño and La Niña since then did
here (where, when it started, how long it lasted), DFO herring spawning records, and an
experimental mixed layer depth (MLD) forecast through March 2027.

Built at the Victoria Ocean Hackathon 2026.

## How it fits together

```
pipeline/  and  model/          S3 team bucket                 Your laptop
(run once, mostly on the hub)   ──────────────                 ───────────
prepare data, analyse events,  ──►  teams/team-elnino/   ──►  server/ (Flask API)  ──►  client/ (React map)
train the MLD model                 processed/                 streams from S3
```

1. **Pipeline** (`pipeline/`): cleans the raw ocean data, analyses every El Niño and La Niña,
   and writes small, app-ready files to `s3://ocean-hackathon-2026/teams/team-elnino/processed/`.
2. **Model** (`model/`): trains the MLD model and turns the seasonal forecast into
   per-point MLD predictions, uploaded to the same folder.
3. **Server** (`server/`): Flask API that streams those files from S3 as JSON. No drawing.
4. **Client** (`client/`): React + MapLibre app that draws the map, timeline and cards.

`rough_work/` folders (at the root and inside `pipeline/` and `model/`) hold exploration and
earlier versions that the app doesn't use.

## Repo layout

| Path | What |
|---|---|
| `pipeline/preprocess.ipynb` | Monthly ARMOR3D ocean data and wind: crops to our area, computes the normals, El Niño index labels (`enso.json`) and `meta.json` |
| `pipeline/analyze_events.py` | The El Niño history analysis on the daily data: anomalies, change points, unusual spells, each event's effect at every point |
| `pipeline/build_daily_store.py` | Daily data in a map-friendly format for the server, plus each point's normal |
| `pipeline/herring/fishies_geojson.py` | DFO herring spawn records (`herring-spawning.csv`) to `client/public/herring-spawn.geojson` |
| `model/train_mld_forecast.py` | Trains the MLD model and writes the per-point forecast the Predictions switch shows |
| `model/forecast_inputs/` | The ECMWF seasonal forecast the model runs on, its download script and `merge_to_csv.py` (see its README) |
| `server/` | Flask API (see [server/README.md](server/README.md) for endpoints) |
| `client/` | React frontend (see [client/README.md](client/README.md)) |
| `requirements.txt` | Python packages for the pipeline, model and server |

## Processed data (in `s3://ocean-hackathon-2026/teams/team-elnino/processed/`)

| File | Made by | Contents |
|---|---|---|
| `ocean_monthly.zarr`, `climatology.zarr` | `preprocess.ipynb` | Monthly temperature, salinity, sea surface height, MLD, currents since 1993, and the normal for each calendar month |
| `wind_monthly.zarr`, `wind_climatology.zarr` | `preprocess.ipynb` | Monthly wind and its normals |
| `enso.json`, `meta.json` | `preprocess.ipynb` | NOAA El Niño index (ONI) per month with its phase; variables, units and ranges |
| `daily/…_2002-2026_combined.nc` | (downloaded) | Daily ARMOR3D surface data, 2002–2026: input to the analysis |
| `analysis/events.json`, `analysis/event_effects.zarr`, … | `analyze_events.py` | Every El Niño / La Niña and its effect at each point; patterns and spells |
| `ocean_daily.zarr`, `daily_normal.zarr`, `meta_daily.json` | `build_daily_store.py` | Daily maps and each point's normal, for the server |
| `forecast_mld_grid.json` | `train_mld_forecast.py` | Per-point MLD forecast, Oct 2026 – Mar 2027 (the Predictions switch) |
| `forecast_mld.json` | (archived `train_v2.py`) | Older region-average MLD forecast; still served by `/forecast`, not shown in the app |

## Running locally

Use **Python 3.11 or newer**. On **Windows, stick to 3.11–3.13**: `numcodecs` (needed by
zarr) has no prebuilt wheels for newer versions, so pip has to compile it. That works on macOS
and Linux with a C compiler (Xcode command-line tools on Mac) but usually fails on Windows.
The client needs **Node.js 22** or newer and **pnpm**.

1. **Create one virtual environment at the repo root** and install everything:
   ```bash
   # macOS / Linux
   python3 -m venv .venv && source .venv/bin/activate
   # Windows (PowerShell)
   py -3.11 -m venv .venv; .venv\Scripts\Activate.ps1

   pip install -r requirements.txt
   ```
   This covers the pipeline, the model and the server. (The server alone only needs
   `server/requirements.txt`.)
2. **Save the team AWS key on your laptop** (once): `python server/setup_aws.py`.
   It asks for the same Access key ID and Secret access key as the hub's Step 1 and stores
   them in `~/.aws/credentials` under the profile `hack-team`, never in the repo. The pipeline,
   the model and the server all read data through that profile.
3. **Start the server.**
   ```bash
   cd server
   PRELOAD_DAILY=1 flask --app app run --debug --port 5001
   ```
   Check http://localhost:5001/metadata. `PRELOAD_DAILY=1` is optional: it loads the daily
   maps into memory at startup so scrubbing the timeline is fast (Windows PowerShell:
   `$env:PRELOAD_DAILY=1` first).
4. **Start the client** in a second terminal:
   ```bash
   cd client
   pnpm install
   pnpm dev
   ```
   Open the address it prints (usually http://localhost:5173). It talks to the server on
   port 5001.

## Regenerating the data

Only needed when the data changes; the app reads whatever is already in the bucket. Run in
this order (the first three are best run on the hub):

1. `pipeline/preprocess.ipynb`
2. `python pipeline/analyze_events.py`
3. `python pipeline/build_daily_store.py`
4. Forecast: in `model/forecast_inputs/`, `python download_forecast_inputs.py` (needs a CDS
   API key in `~/.cdsapirc`), then `python merge_to_csv.py`, then
   `python ../train_mld_forecast.py`, which uploads `forecast_mld_grid.json`
5. Herring layer: `python pipeline/herring/fishies_geojson.py`

## Model

A gradient-boosted tree model (scikit-learn `HistGradientBoostingRegressor`) predicts each
point's monthly MLD anomaly (difference from that point's normal for the month) over
46–52°N, 134–120°W. Inputs: same-month anomalies of sea surface temperature, salinity, sea
surface height and wind, each point's normals, the month, location, the El Niño index (ONI,
current and lagged) and the Pacific Decadal Oscillation. It is trained on months before 2010,
with El Niño months weighted 3x.

For the forecast it runs on all 51 members of ECMWF's SEAS5 seasonal forecast issued
1 October 2026 (October 2026 to March 2027); 2 m air temperature stands in for sea surface
temperature. The map shows the members' mean; the 10th–90th percentile range is in the file.
It is experimental: the inputs are not bias-corrected and the script does not yet report a
held-out accuracy score.

## Data sources

- **Ocean observations:** Copernicus Marine ARMOR3D, `MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012` (daily and monthly)
- **Wind:** Copernicus Marine `WIND_GLO_PHY_CLIMATE_L4_MY_012_003`
- **El Niño index:** NOAA Climate Prediction Center, Oceanic Niño Index (ONI)
- **Pacific Decadal Oscillation:** NOAA ERSST v5 PDO index
- **Forecasts:** Copernicus Climate Change Service (C3S), ECMWF SEAS5 seasonal forecasts
- **Herring spawn:** Fisheries and Oceans Canada (Open Government Licence – Canada)

## Rules

- Never commit AWS keys or anything from `~/.aws/`.
- Never commit data (`*.zarr`, downloads). Data lives in the team bucket.
- Clear notebook outputs before committing large plots.
