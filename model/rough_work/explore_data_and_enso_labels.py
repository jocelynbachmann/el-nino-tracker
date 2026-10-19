"""
Data checks + El Niño feature building. No model is trained here (see model/train_mld_forecast.py).

Loads the ocean/wind data and runs sanity checks on it, then turns the ONI file into
El Niño labels, model inputs and forecast targets.

Terms:
  ONI      Oceanic Niño Index: tropical Pacific temperature vs normal, in °C (3-month average).
  El Niño  ONI >= +0.5 for 5+ months in a row. La Niña: ONI <= -0.5 for 5+ months.
  Feature  A model input column.   Target: the value the model learns to predict.
  Lag      Value from N months ago. Lead: how many months ahead we predict.
"""

import json

import boto3
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr


# --- 1. Connect to the data -------------------------------------------------------
DATA_FOLDER = "elnino"
PROFILE = "hack-team"           # AWS profile saved during setup
BUCKET  = "ocean-hackathon-2026"

fs = s3fs.S3FileSystem(profile=PROFILE)
opts = {"profile": PROFILE}

# Work out the team name from the AWS key
who = boto3.Session(profile_name=PROFILE).client("sts").get_caller_identity()["Arn"]
TEAM = who.split("/")[-1]
TEAM_DIR = f"{BUCKET}/teams/{TEAM}"             # our workspace (writable)
DATA_DIR = f"{BUCKET}/data/{DATA_FOLDER}"       # raw data (read-only)
OUT_DIR  = f"{TEAM_DIR}/processed"

print("Connected as:", TEAM)
print("Raw data:    ", "s3://" + DATA_DIR)
print("Your outputs:", "s3://" + TEAM_DIR)
print(fs.ls(OUT_DIR))
print("zarr version:", zarr.__version__)

# Opened lazily: values are only downloaded when used.
# *_clim = the long-term average ("normal") for each calendar month.
ocean     = xr.open_zarr(f"s3://{OUT_DIR}/ocean_monthly.zarr",     storage_options=opts)
clim      = xr.open_zarr(f"s3://{OUT_DIR}/climatology.zarr",       storage_options=opts)
wind      = xr.open_zarr(f"s3://{OUT_DIR}/wind_monthly.zarr",      storage_options=opts)
wind_clim = xr.open_zarr(f"s3://{OUT_DIR}/wind_climatology.zarr",  storage_options=opts)

stores = {"ocean": ocean, "clim": clim, "wind": wind, "wind_clim": wind_clim}


# --- 2. What's in each dataset ----------------------------------------------------
for name, ds in stores.items():
    print(f"\n=== {name} ===  dims: {dict(ds.sizes)}")
    for v, da in ds.data_vars.items():
        units = da.attrs.get("units", "")
        print(f"  {v:24s} {str(da.dims):35s} {units}")


def coord_summary(da, name):
    """One line per coordinate: count, first → last, and spacing."""
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
    # Storage chunk sizes, useful for planning how much to load at once
    v0 = next(iter(ds.data_vars))
    print(f"  chunks ({v0}): {ds[v0].encoding.get('chunks') or ds[v0].chunks}")


# --- 3. Sanity checks -------------------------------------------------------------
lat, lon = ocean.latitude.values, ocean.longitude.values
print("\n=== Area cutout (ocean grid) ===")
print(f"  lat {lat.min():.2f}° to {lat.max():.2f}°, lon {lon.min():.2f}° to {lon.max():.2f}°")
print(f"  lon convention: {'0–360' if lon.max() > 180 else '-180–180'}")
print(f"  resolution ≈ {abs(np.median(np.diff(lat))):.3f}° lat × {abs(np.median(np.diff(lon))):.3f}° lon")

# Missing months
for name in ["ocean", "wind"]:
    t = pd.to_datetime(stores[name].time.values)
    full = pd.date_range(t.min().to_period("M").to_timestamp(),
                         t.max().to_period("M").to_timestamp(), freq="MS")
    missing = sorted(set(full.to_period("M")) - set(t.to_period("M")))
    print(f"\n{name}: {len(t)} steps, expected {len(full)} monthly; "
          f"missing: {missing[:10]}{' …' if len(missing) > 10 else ''}")

# Ocean and wind must share grid points and months before they can be combined
print("\n=== Grid alignment ===")
for c in ["latitude", "longitude"]:
    a, b = ocean[c].values, wind[c].values
    same = a.shape == b.shape and np.allclose(a, b)
    print(f"  {c}: ocean {a.shape} vs wind {b.shape} → {'identical' if same else 'DIFFERENT, needs interp/reindex'}")
t_o, t_w = pd.to_datetime(ocean.time.values), pd.to_datetime(wind.time.values)
overlap = t_o.intersection(t_w)
print(f"  time overlap: {len(overlap)} steps"
      + (f", {overlap.min():%Y-%m} → {overlap.max():%Y-%m}" if len(overlap) else ""))

# Land points have no ocean values (NaN)
frac_nan = float(ocean["to"].isel(time=0).isnull().mean())
print(f"\nLand/NaN fraction in first ocean timestep: {frac_nan:.1%}")


# --- 4. Time features ---------------------------------------------------------------
# Time is given to the model in two ways:
#   t_years  continuous clock (years since start), so it can pick up long-term trends.
#            Tree models can't extrapolate it: later months get the trend as of the
#            last training year.
#   month    1-12 as a category, so it can learn seasons without treating 12 as "far" from 1
T0 = pd.to_datetime(ocean.time.values).min()

def add_time_features(df, time_col="time", t0=T0):
    t = pd.to_datetime(df[time_col])
    df = df.copy()
    df["t_years"] = (t - t0).dt.total_seconds() / (365.25 * 24 * 3600)
    df["month"] = pd.Categorical(t.dt.month, categories=range(1, 13))
    return df

def time_design_matrix(df):
    """t_years + month as 11 one-hot (1/0) columns, January being all zeros.
    Only needed for models that require plain numbers (e.g. linear regression);
    tree models like the one in model/train_mld_forecast.py take `month` directly."""
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True, dtype=float)
    return pd.concat([df[["t_years"]], month_dummies], axis=1)

time_df = add_time_features(pd.DataFrame({"time": ocean.time.values}))
print("\n=== Time features ===")
print(time_df.head(13))
print(time_design_matrix(time_df).head(10))


# --- 5. Load ONI --------------------------------------------------------------------
ONI_PATH = "ocean-hackathon-2026/teams/team-elnino/processed/enso.json"

def load_oni(path=ONI_PATH):
    """Read enso.json -> DataFrame[time, oni, phase, provisional], one row per month.

    File format: [{"month": "1993-01", "oni": 0.13, "phase": "neutral", "provisional": false}, ...]
    `provisional` marks recent values NOAA may still revise.
    """
    with fs.open(path) as f:
        oni = pd.DataFrame(json.load(f))
    oni["time"] = pd.to_datetime(oni["month"], format="%Y-%m")
    oni["oni"] = oni["oni"].astype(float)
    oni["provisional"] = oni["provisional"].astype(bool)
    return (oni[["time", "oni", "phase", "provisional"]]
            .drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True))

oni_raw = load_oni()
print(f"\n=== ONI data ===  {len(oni_raw)} months, {oni_raw.time.min():%Y-%m} → {oni_raw.time.max():%Y-%m}")
print(oni_raw.head(12))


# --- 6. Label El Niño / La Niña events --------------------------------------------
# NOAA's definition: the threshold must hold for 5+ consecutive months. A short warm
# spell is not an event, so a simple `oni >= 0.5` check isn't enough.
EL_NINO_THRESHOLD = 0.5
LA_NINA_THRESHOLD = -0.5
MIN_EVENT_MONTHS = 5

def event_ids(condition, min_months):
    """Number each run of True lasting >= min_months as 1, 2, 3...; 0 elsewhere."""
    stretch = (condition != condition.shift()).cumsum()            # id per run of equal values
    stretch_length = condition.groupby(stretch).transform("size")
    in_event = condition & (stretch_length >= min_months)
    return stretch.where(in_event).rank(method="dense").fillna(0).astype(int)

def streak_length(condition):
    """Consecutive months `condition` has been True, up to and including each row."""
    stretch = (condition != condition.shift()).cumsum()
    return condition.astype(int).groupby(stretch).cumsum()

def strength_category(peak_oni):
    a = abs(peak_oni)
    return "very strong" if a >= 2.0 else "strong" if a >= 1.5 else "moderate" if a >= 1.0 else "weak"

# Reindex to a full monthly calendar so shifts/lags below are always whole months
oni_df = oni_raw.set_index("time").sort_index()
oni_df = oni_df.reindex(pd.date_range(oni_df.index.min(), oni_df.index.max(), freq="MS"))
oni_df.index.name = "time"
oni = oni_df["oni"]

oni_df["el_nino_event_id"] = event_ids(oni >= EL_NINO_THRESHOLD, MIN_EVENT_MONTHS)
oni_df["la_nina_event_id"] = event_ids(oni <= LA_NINA_THRESHOLD, MIN_EVENT_MONTHS)
oni_df["el_nino"] = (oni_df["el_nino_event_id"] > 0).astype(float)
oni_df["la_nina"] = (oni_df["la_nina_event_id"] > 0).astype(float)

# A spell still running at the end of the data, shorter than 5 months, may still become an
# event. Mark it unknown (NaN) rather than "no event", or we'd train on a wrong label.
for col, condition in (("el_nino", oni >= EL_NINO_THRESHOLD), ("la_nina", oni <= LA_NINA_THRESHOLD)):
    months_so_far = int(streak_length(condition).iloc[-1])
    if 0 < months_so_far < MIN_EVENT_MONTHS:
        oni_df.loc[oni_df.index[-months_so_far:], col] = np.nan

# +1 El Niño, 0 neutral, -1 La Niña, NaN unknown
oni_df["enso_state"] = np.select([oni_df["el_nino"] == 1, oni_df["la_nina"] == 1], [1, -1], 0)
oni_df.loc[oni_df["el_nino"].isna() | oni_df["la_nina"].isna(), "enso_state"] = np.nan

el_nino_events = (oni_df[oni_df["el_nino_event_id"] > 0].reset_index()
                  .groupby("el_nino_event_id")
                  .agg(start=("time", "min"), end=("time", "max"), months=("oni", "size"),
                       peak_oni=("oni", "max")))
el_nino_events["strength"] = el_nino_events["peak_oni"].map(strength_category)

print("\n=== El Niño events (ONI >= 0.5 for >= 5 months) ===")
print(el_nino_events)
# Our labels vs the file's `phase` column - should agree except the latest unknown months
print(pd.crosstab(oni_df["phase"], oni_df["enso_state"], dropna=False))


# --- 7. Features (past data only) -------------------------------------------------
# A model must only see what would have been known at prediction time; otherwise it scores
# well in testing and fails in real use ("leakage").
#  - Each month's ONI averages the month before, the month itself AND the month after,
#    so it isn't final until the next month ends. Features therefore start at last month,
#    whose ONI includes this month: a row's features are known at the END of its month.
#  - The event labels above need future months to decide (does the spell last 5 months?),
#    so they are only ever used as targets, never as features.
prev = oni.shift(1)                                         # last month's ONI

oni_feats = pd.DataFrame(index=oni_df.index)
for lag in (1, 2, 3, 6, 12):
    oni_feats[f"oni_lag{lag}"] = oni.shift(lag)
oni_feats["oni_trend_3m"] = prev - oni.shift(4)                      # warming or cooling
oni_feats["oni_mean_12m"] = prev.rolling(12, min_periods=6).mean()
oni_feats["warm_streak_months"] = streak_length(prev >= EL_NINO_THRESHOLD)
oni_feats["cold_streak_months"] = streak_length(prev <= LA_NINA_THRESHOLD)


# --- 8. Targets: what happens 1, 3 and 6 months later --------------------------------
# Example, row for March 2023 (features = ONI up to Feb 2023, which needs data to end of March):
#   el_nino_in_1m = 0             April 2023 was not El Niño yet
#   el_nino_in_3m = 1             June 2023 was El Niño
#   el_nino_onset_within_3m = 1   an El Niño started during April-June (May 2023, if the
#                                 file labels each 3-month average by its middle month)
# Leads count from the end of the row's month. A month's ONI already includes the month
# before it, so the 1-month target is partly known when the prediction is made.
# The last few rows have no future data yet, so their targets are NaN.
LEADS = (1, 3, 6)

starts = ((oni_df["el_nino"] == 1) & (oni_df["el_nino"].shift(fill_value=0) == 0)).astype(float)
starts[oni_df["el_nino"].isna()] = np.nan      # unknown label = unknown whether one started

oni_targets = pd.DataFrame(index=oni_df.index)
for h in LEADS:
    oni_targets[f"el_nino_in_{h}m"]    = oni_df["el_nino"].shift(-h)
    oni_targets[f"la_nina_in_{h}m"]    = oni_df["la_nina"].shift(-h)
    oni_targets[f"enso_state_in_{h}m"] = oni_df["enso_state"].shift(-h)
    oni_targets[f"oni_in_{h}m"]        = oni.shift(-h)      # predicting the value also gives strength
    # Early warning: does a new El Niño start in any of the next h months?
    next_h_months = pd.concat([starts.shift(-k) for k in range(1, h + 1)], axis=1)
    oni_targets[f"el_nino_onset_within_{h}m"] = next_h_months.max(axis=1, skipna=False)
    oni_targets[f"target_provisional_{h}m"] = oni_df["provisional"].shift(-h).astype("boolean")


# --- 9. Assemble the training table -------------------------------------------------
# Only ~1 in 4 months is El Niño, so a model could look accurate by always answering "no".
# Class weights make El Niño months count more, so both outcomes matter equally overall.
# Provisional months count half because their values may still change; NaN targets count 0.
PROVISIONAL_WEIGHT = 0.5

def sample_weights(y, provisional=None, provisional_weight=PROVISIONAL_WEIGHT):
    y = pd.Series(y).astype(float)
    w = pd.Series(0.0, index=y.index)
    known = y.notna()
    share_el_nino = (y[known] == 1).mean()
    if 0 < share_el_nino < 1:
        w[known] = np.where(y[known] == 1, 0.5 / share_el_nino, 0.5 / (1 - share_el_nino))
    else:
        w[known] = 1.0
    if provisional is not None:
        w[pd.Series(provisional, index=y.index).fillna(False).astype(bool)] *= provisional_weight
    return w

# One row per month: time features + ONI features. Target (y) and weights (w) kept separate.
month_start = pd.to_datetime(time_df["time"]).dt.to_period("M").dt.to_timestamp()
model_df = time_df.set_index(month_start).join(oni_feats, how="left")
y = oni_targets.reindex(model_df.index)["el_nino_in_3m"]
w = sample_weights(y, oni_targets.reindex(model_df.index)["target_provisional_3m"])
print(model_df.tail(3))
