# STEP 2 - CONNECT TO THE DATA  (run every time you open the notebook)
import boto3, s3fs
import xarray as xr

DATA_FOLDER = "elnino"       # your team's data folder - no need to change this

PROFILE = "hack-team"            # use the key you saved in Step 1 (the hub's own AWS login can't see this data)
BUCKET  = "ocean-hackathon-2026"
fs = s3fs.S3FileSystem(profile=PROFILE)
opts = {"profile": PROFILE}

# Ask AWS which team this key belongs to, so you never have to type your team name
who = boto3.Session(profile_name=PROFILE).client("sts").get_caller_identity()["Arn"]
TEAM = who.split("/")[-1]                      # e.g. "team-sound"
TEAM_DIR = f"{BUCKET}/teams/{TEAM}"            # your team's workspace (you can save files here)
DATA_DIR = f"{BUCKET}/data/{DATA_FOLDER}"      # your raw data (read-only)
OUT_DIR  = f"{TEAM_DIR}/processed"

print("Connected as:", TEAM)
print("Raw data:    ", "s3://" + DATA_DIR)
print("Your outputs:", "s3://" + TEAM_DIR)
print(fs.ls(OUT_DIR))

import zarr
print(zarr.__version__)

ocean     = xr.open_zarr(f"s3://{OUT_DIR}/ocean_monthly.zarr",     storage_options=opts)
clim      = xr.open_zarr(f"s3://{OUT_DIR}/climatology.zarr",       storage_options=opts)
wind      = xr.open_zarr(f"s3://{OUT_DIR}/wind_monthly.zarr",      storage_options=opts)
wind_clim = xr.open_zarr(f"s3://{OUT_DIR}/wind_climatology.zarr",  storage_options=opts)

stores = {"ocean": ocean, "clim": clim, "wind": wind, "wind_clim": wind_clim}

for name, ds in stores.items():
    print(f"\n=== {name} ===  dims: {dict(ds.sizes)}")
    for v, da in ds.data_vars.items():
        units = da.attrs.get("units", "")
        print(f"  {v:24s} {str(da.dims):35s} {units}")



import numpy as np
import pandas as pd

def coord_summary(da, name):
    v = da.values
    if v.size == 0:
        return f"  {name:10s} (empty)"
    if np.issubdtype(v.dtype, np.datetime64):
        t = pd.to_datetime(v)
        steps = pd.Series(t).diff().dropna()
        freq = pd.infer_freq(t) if len(t) >= 3 else None
        return (f"  {name:10s} n={len(t):4d}  {t.min():%Y-%m-%d} → {t.max():%Y-%m-%d}"
                f"  freq={freq}  step(min/max)={steps.min()} / {steps.max()}")
    d = np.diff(v) if v.size > 1 else np.array([np.nan])
    return (f"  {name:10s} n={v.size:4d}  {v.min():9.3f} → {v.max():9.3f}"
            f"  step={np.nanmedian(d):.4f} (min {np.nanmin(d):.4f}, max {np.nanmax(d):.4f})"
            f"  {'ascending' if d[0] > 0 else 'descending'}")

for name, ds in stores.items():
    print(f"\n=== {name} ===  dims: {dict(ds.sizes)}")
    for c in ds.coords:
        if ds[c].ndim == 1:
            print(coord_summary(ds[c], c))
    # chunking of the first variable, useful for planning reads
    v0 = next(iter(ds.data_vars))
    print(f"  chunks ({v0}): {ds[v0].encoding.get('chunks') or ds[v0].chunks}")

# ---- Area cutout in plain words ----
lat, lon = ocean.latitude.values, ocean.longitude.values
print("\n=== Area cutout (ocean grid) ===")
print(f"  lat {lat.min():.2f}° to {lat.max():.2f}°, lon {lon.min():.2f}° to {lon.max():.2f}°")
print(f"  lon convention: {'0–360' if lon.max() > 180 else '-180–180'}")
print(f"  resolution ≈ {abs(np.median(np.diff(lat))):.3f}° lat × {abs(np.median(np.diff(lon))):.3f}° lon")

# ---- Missing months in the time series ----
for name in ["ocean", "wind"]:
    t = pd.to_datetime(stores[name].time.values)
    full = pd.date_range(t.min().to_period("M").to_timestamp(),
                         t.max().to_period("M").to_timestamp(), freq="MS")
    missing = sorted(set(full.to_period("M")) - set(t.to_period("M")))
    print(f"\n{name}: {len(t)} steps, expected {len(full)} monthly; missing: {missing[:10]}{' …' if len(missing) > 10 else ''}")

# ---- Do ocean and wind grids match? ----
print("\n=== Grid alignment ===")
for c in ["latitude", "longitude"]:
    a, b = ocean[c].values, wind[c].values
    same = a.shape == b.shape and np.allclose(a, b)
    print(f"  {c}: ocean {a.shape} vs wind {b.shape} → {'identical' if same else 'DIFFERENT, needs interp/reindex'}")
t_o, t_w = pd.to_datetime(ocean.time.values), pd.to_datetime(wind.time.values)
overlap = t_o.intersection(t_w)
print(f"  time overlap: {len(overlap)} steps"
      + (f", {overlap.min():%Y-%m} → {overlap.max():%Y-%m}" if len(overlap) else ""))

# ---- How much of the box is land (NaN) ----
frac_nan = float(ocean["to"].isel(time=0).isnull().mean())
print(f"\nLand/NaN fraction in first ocean timestep: {frac_nan:.1%}")