# Daily ERA5 10 m wind for the NE Pacific, fetched year by year from the Climate Data Store
# and appended to a Zarr in our S3 folder. Run on the hub. Temp files live on the hub only.
#
# One-time setup:
#   1. CDS account -> open the dataset page -> "Download" tab -> accept the licence (requests fail until you do)
#   2. In a hub TERMINAL, create ~/.cdsapirc (never in a notebook or GitHub):
#        url: https://cds.climate.copernicus.eu/api
#        key: <your personal access token from your CDS profile page>
#   3. pip install cdsapi

# %% SETTINGS
import os, glob, shutil, tempfile, time, zipfile
from concurrent.futures import ThreadPoolExecutor
import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
import s3fs

START_YEAR = 1993                      # match the ocean file
AREA = [52, -134, 46, -120]            # N, W, S, E
DATASET = "derived-era5-single-levels-daily-statistics"
OUT = ("s3://ocean-hackathon-2026/teams/team-elnino/processed/"
       f"era5_wind_daily_{START_YEAR}-present_46-52N_134-120W.zarr")
# Hub: key is in env vars -> {}. Laptop: key is in the "hack-team" profile (server/setup_aws.py).
import os
STORAGE = ({} if os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE")
           else {"profile": os.environ.get("AWS_PROFILE", "hack-team")})

WORKERS = 6                            # requests in flight at once (CDS queues extras per user)
fs = s3fs.S3FileSystem(**STORAGE)

def request(year, months):
    return {
        "product_type": "reanalysis",
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": str(year),
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],   # CDS skips dates that don't exist
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": AREA,
    }

def fetch(year, months, tmp):
    """Download one request, return it as a tidy Dataset (time, latitude, longitude)."""
    target = os.path.join(tmp, f"{year}_{months[0]:02d}")
    cdsapi.Client(quiet=True).retrieve(DATASET, request(year, months)).download(target)   # one client per thread
    if zipfile.is_zipfile(target):                  # several variables can arrive as a zip of .nc files
        with zipfile.ZipFile(target) as z:
            z.extractall(target + "_x")
        files = sorted(glob.glob(target + "_x/*.nc"))
    else:
        files = [target]
    ds = xr.merge([xr.open_dataset(f) for f in files]).load()
    ds = ds.rename({k: "time" for k in ("valid_time", "date") if k in ds.dims})
    ds = ds.drop_vars([v for v in ds.coords if v not in ("time", "latitude", "longitude")])
    ds["time"] = pd.to_datetime(ds.time.values).floor("D")
    return ds.sortby("latitude")

# %% DOWNLOAD (resumable: re-running skips years already in the store)
done = set()
if fs.exists(OUT):
    done = set(pd.to_datetime(xr.open_zarr(OUT, storage_options=STORAGE).time.values).year)
    print("Already have:", sorted(done))

today = pd.Timestamp.today()
for year in range(START_YEAR, today.year + 1):
    if year in done and year != today.year:
        continue
    months = list(range(1, 13 if year < today.year else today.month))   # current year: full months only
    tmp = tempfile.mkdtemp()
    try:
        t0 = time.time()                             # a full year exceeds the CDS cost limit; one month fits,
        with ThreadPoolExecutor(WORKERS) as pool:    # so send the year's months in parallel
            parts = list(pool.map(lambda m: fetch(year, [m], tmp), months))
        ds = xr.concat(parts, dim="time").sortby("time")
        print(f"  {year}: {len(months)} months in {(time.time() - t0) / 60:.1f} min")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    ds = ds.rename({"u10": "wind_east", "v10": "wind_north"}).astype("float32")
    # Speed of the DAILY-MEAN wind. Underestimates average speed on days the wind swings around,
    # but it's consistent through time, which is what anomalies and change points need.
    ds["wind_speed"] = np.hypot(ds.wind_east, ds.wind_north)
    for v in ds.variables:
        ds[v].encoding = {}
    ds = ds.chunk({"time": -1, "latitude": -1, "longitude": -1})

    if year == today.year and year in done:          # refresh the partial current year
        old = xr.open_zarr(OUT, storage_options=STORAGE)
        keep = old.sel(time=old.time.dt.year < year).load()
        ds = xr.concat([keep, ds], dim="time").chunk({"time": 365})
        ds.to_zarr(OUT, mode="w", consolidated=True, storage_options=STORAGE)
    elif not fs.exists(OUT):
        ds.to_zarr(OUT, mode="w", consolidated=True, storage_options=STORAGE)
    else:
        ds.to_zarr(OUT, append_dim="time", consolidated=True, storage_options=STORAGE)
    print(f"{year}: {ds.sizes['time']} days written")

# %% VERIFY
chk = xr.open_zarr(OUT, storage_options=STORAGE)
print(chk)
t = pd.to_datetime(chk.time.values)
print("Steps:", pd.Series(np.diff(t)).value_counts().head().to_dict())
print("Days:", len(t), t[0].date(), "->", t[-1].date(),
      "| missing:", len(pd.date_range(t[0], t[-1], freq="D").difference(t)))
print("Mean wind speed (m/s):", float(chk.wind_speed.mean()))   # expect roughly 5-9 offshore
