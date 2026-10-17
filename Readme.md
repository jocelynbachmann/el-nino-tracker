# El Niño Tracker

A web app showing how El Niño shows up in the waters off Washington and Vancouver Island,
for shellfish farmers and anyone else who works on the coast. It maps monthly ocean
conditions since 1993 and forecasts mixed layer depth (MLD) up to 2 years ahead.

Built at the Victoria Ocean Hackathon 2026.

## How it fits together

```
JupyterHub (pipeline/)          S3 team bucket                 Your laptop
─────────────────────           ──────────────                 ───────────
preprocess.ipynb        ──►     teams/team-elnino/     ──►     server/ (Flask API)  ──►  client/ (React)
model notebook          ──►       processed/                   streams from S3
```

1. **Pipeline** (on JupyterHub): cleans the raw data and trains the MLD model. Writes small,
   app-ready files to `s3://ocean-hackathon-2026/teams/team-elnino/processed/`.
2. **Server** (`server/`): Flask API that streams the processed data from S3 and serves it as
   JSON. No drawing.
3. **Client** (`client/`): React app that draws the map, time slider and charts.

## Repo layout

| Path | What |
|---|---|
| `pipeline/preprocess.ipynb` | Stitches ARMOR3D 1993–2024 + 2025–present, crops to 40–62°N 150–120°W, computes climatology, ENSO labels, wind, `meta.json` |
| `model/` | MLD model notebooks (`model1.ipynb`, `model2.ipynb`) |
| `cds_api.py` | Downloads C3S seasonal forecasts (needs a CDS API key in `~/.cdsapirc`) |
| `server/` | Flask API (see [server/README.md](server/README.md) for endpoints) |
| `requirements.txt` | Python packages for the notebooks, scripts and server |
| `client/` | React frontend (coming) |

## Processed data (in `s3://ocean-hackathon-2026/teams/team-elnino/processed/`)

| File | Contents |
|---|---|
| `ocean_monthly.zarr` | Temperature, salinity, sea surface height, MLD, currents. Monthly, Jan 1993–Aug 2026, 1/8° |
| `climatology.zarr` | Normal value per calendar month, 1993–2018 |
| `wind_monthly.zarr` | Wind speed, east/north wind, wind stress. Monthly, Jul 1994–May 2026, 1/4° |
| `wind_climatology.zarr` | Normal wind per calendar month, 1994–2018 |
| `enso.json` | NOAA ONI per month with `el_nino` / `la_nina` / `neutral` (+ `provisional` for an ongoing event) |
| `meta.json` | Variables, units, value ranges, date range. Drives the client |
| `forecast_mld.json` | MLD forecast from the model notebook (format in server/README.md) |

## Running locally

Use **Python 3.11 or newer**. On **Windows, stick to 3.11–3.13**: `numcodecs` (needed by
zarr) has no prebuilt wheels for newer versions, so pip has to compile it. That works on macOS
and Linux with a C compiler (Xcode command-line tools on Mac) but usually fails on Windows.

1. **Create one virtual environment at the repo root** and install everything:
   ```bash
   # macOS / Linux
   python3 -m venv .venv && source .venv/bin/activate
   # Windows (PowerShell)
   py -3.11 -m venv .venv; .venv\Scripts\Activate.ps1

   pip install -r requirements.txt
   ```
   This covers the notebooks, `cds_api.py` and the server. (The server alone only needs
   `server/requirements.txt`.)
2. **Save the team AWS key on your laptop** (once): `python server/setup_aws.py`.
   It asks for the same Access key ID and Secret access key as the hub's Step 1 and stores
   them in `~/.aws/credentials` under the profile `hack-team`, never in the repo. The model
   notebooks and the server both read data through that profile.
3. **Run the notebooks.** Open `model/model2.ipynb` (or any notebook) in Jupyter or VS Code and
   pick the `.venv` kernel at the repo root.
4. **Start the server.**
   ```bash
   cd server
   flask --app app run --debug --port 5001
   ```
   Check http://localhost:5001/metadata
5. **Start the client.** (coming)

## Model

Predicts monthly MLD over 46–52°N, 134–120°W from same-month sea surface temperature,
salinity, sea surface height, wind speed and north wind. Trained on ARMOR3D + wind (1994–2012),
validated on 2013–2018, tested on 2019–2026. For the future, it is fed Copernicus C3S seasonal
forecasts of those predictors (months 1–6); later months fall back to normal conditions and
are shown as lower confidence.

## Data sources

- **Ocean observations:** Copernicus Marine ARMOR3D, `MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012`
- **Wind:** Copernicus Marine `WIND_GLO_PHY_CLIMATE_L4_MY_012_003`
- **El Niño index:** NOAA Climate Prediction Center, Oceanic Niño Index (ONI)
- **Forecasts:** Copernicus Climate Change Service (C3S) seasonal forecasts

## Rules

- Never commit AWS keys or anything from `~/.aws/`.
- Never commit data (`*.zarr`, downloads). Data lives in the team bucket.
- Clear notebook outputs before committing large plots.
