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

# ---- Time features for the model ----
# t_years: continuous time (fractional years since the first timestep), so the model
#          sees the date as one ordered number and can pick up long-term trends.
# month:   separate categorical variable (1-12), so the model can still learn seasonality.
T0 = pd.to_datetime(ocean.time.values).min()

def add_time_features(df, time_col="time", t0=T0):
    t = pd.to_datetime(df[time_col])
    df = df.copy()
    df["t_years"] = (t - t0).dt.total_seconds() / (365.25 * 24 * 3600)
    df["month"] = pd.Categorical(t.dt.month, categories=range(1, 13))
    return df

def time_design_matrix(df):
    """Numeric model inputs: continuous t_years + one-hot month (January is the baseline)."""
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True, dtype=float)
    return pd.concat([df[["t_years"]], month_dummies], axis=1)

time_df = add_time_features(pd.DataFrame({"time": ocean.time.values}))
print("\n=== Time features ===")
print(time_df.head(13))
print(time_design_matrix(time_df).head(3))

# ---- El Niño / La Niña events from the ONI file ----
# ONI = 3-month running mean of Niño-3.4 SST anomaly. NOAA's event rule: ONI >= +0.5
# (El Niño) or <= -0.5 (La Niña) for at least 5 consecutive overlapping seasons.
#
# How ONI enters the model (keeps it from cheating):
#   * Event labels look ahead (a run must last 5 months), so they are TARGETS only,
#     shifted forward by the lead time. Never use them as inputs.
#   * The ONI value for month t is centred on t, so it includes month t+1. Only ONI from
#     t-1 and earlier is known at the end of month t, so every ONI input is lagged >= 1.
#   * Provisional ONI values may still be revised, so they get a lower sample weight.
ONI_PATH = f"ocean-hackathon-2026/teams/team-elnino/processed/enso.json"
EL_NINO_THRESHOLD = 0.5
LA_NINA_THRESHOLD = -0.5
MIN_EVENT_MONTHS = 5
LEADS = (1, 3, 6)                # months ahead the model forecasts
PROVISIONAL_WEIGHT = 0.5

def load_oni(path=ONI_PATH):
    """STUB - implement later.

    Read the file with columns month, oni, phase, provisional and return a DataFrame with:
      time        Timestamp, first day of the month (if `month` is a season code like
                  "DJF", use its centre month: DJF -> Jan, JFM -> Feb, ...)
      oni         float
      phase       str ("El Nino" / "La Nina" / "Neutral", as given in the file)
      provisional bool
    sorted by time, one row per month.
    """
    raise NotImplementedError("load_oni: ONI file import not implemented yet")

def _event_runs(mask, min_len):
    """Number each run of True values lasting >= min_len months (1, 2, ...); 0 = not in an event."""
    run = (mask != mask.shift()).cumsum()
    run_len = mask.groupby(run).transform("size")
    in_event = mask & (run_len >= min_len)
    ids = run.where(in_event)
    return ids.map({r: i + 1 for i, r in enumerate(ids.dropna().unique())}).fillna(0).astype(int)

def _months_in_streak(mask):
    """Months the condition has held up to and including each row, counting backwards only."""
    run = (mask != mask.shift()).cumsum()
    return mask.astype(int).groupby(run).cumsum()

def strength_category(peak_oni):
    a = abs(peak_oni)
    return "very strong" if a >= 2.0 else "strong" if a >= 1.5 else "moderate" if a >= 1.0 else "weak"

def build_oni_features(oni_df, leads=LEADS):
    df = oni_df.set_index("time").sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="MS"))  # find missing months
    df.index.name = "time"
    oni = df["oni"]

    # Event labels (look ahead, so targets only)
    df["el_nino_event_id"] = _event_runs(oni >= EL_NINO_THRESHOLD, MIN_EVENT_MONTHS)
    df["la_nina_event_id"] = _event_runs(oni <= LA_NINA_THRESHOLD, MIN_EVENT_MONTHS)
    df["el_nino"] = (df["el_nino_event_id"] > 0).astype(int)
    df["la_nina"] = (df["la_nina_event_id"] > 0).astype(int)
    df["enso_state"] = np.select([df["el_nino"] == 1, df["la_nina"] == 1], [1, -1], 0)  # +1 / 0 / -1
    onset = (df["el_nino"] == 1) & (df["el_nino"].shift(fill_value=0) == 0)

    # Causal inputs: only ONI from month t-1 and earlier
    feats = pd.DataFrame(index=df.index)
    for lag in (1, 2, 3, 6, 12):
        feats[f"oni_lag{lag}"] = oni.shift(lag)
    feats["oni_trend_3m"] = oni.shift(1) - oni.shift(4)                 # warming or cooling
    feats["oni_mean_12m"] = oni.shift(1).rolling(12, min_periods=6).mean()
    feats["warm_streak_months"] = _months_in_streak(oni.shift(1) >= EL_NINO_THRESHOLD)
    feats["cold_streak_months"] = _months_in_streak(oni.shift(1) <= LA_NINA_THRESHOLD)

    # Targets at each lead: event flag, onset within the window, and the ONI value itself.
    # Train the classifier on `el_nino_in_{h}m`; predicting `oni_in_{h}m` as well also gives strength.
    targets = pd.DataFrame(index=df.index)
    for h in leads:
        targets[f"el_nino_in_{h}m"] = df["el_nino"].shift(-h)
        targets[f"la_nina_in_{h}m"] = df["la_nina"].shift(-h)
        targets[f"enso_state_in_{h}m"] = df["enso_state"].shift(-h)
        targets[f"oni_in_{h}m"] = oni.shift(-h)
        # 1 if a new El Niño starts in months t+1 .. t+h (early-warning target)
        onset_h = onset[::-1].rolling(h, min_periods=1).max()[::-1].shift(-1).astype(float)
        onset_h.iloc[-h:] = np.nan                                      # window runs past the data
        targets[f"el_nino_onset_within_{h}m"] = onset_h
        targets[f"target_provisional_{h}m"] = df["provisional"].shift(-h).astype("boolean")

    # One row per event, for checking results by event and strength
    events = (df[df["el_nino_event_id"] > 0].reset_index()
              .groupby("el_nino_event_id")
              .agg(start=("time", "min"), end=("time", "max"), months=("oni", "size"),
                   peak_oni=("oni", "max")))
    events["strength"] = events["peak_oni"].map(strength_category)

    return df, feats, targets, events

def sample_weights(y, provisional=None, provisional_weight=PROVISIONAL_WEIGHT):
    """Balance the classes so the rarer El Niño months count as much as the rest; down-weight provisional labels."""
    y = pd.Series(y).astype(float)
    w = pd.Series(0.0, index=y.index)
    ok = y.notna()
    pos = (y[ok] == 1).mean()
    if 0 < pos < 1:
        w[ok] = np.where(y[ok] == 1, 0.5 / pos, 0.5 / (1 - pos))
    else:
        w[ok] = 1.0
    if provisional is not None:
        w[pd.Series(provisional, index=y.index).fillna(False).astype(bool)] *= provisional_weight
    return w

try:
    oni_raw = load_oni()
except NotImplementedError as e:
    print(f"\nSkipping ONI events: {e}")
else:
    oni_df, oni_feats, oni_targets, el_nino_events = build_oni_features(oni_raw)
    print("\n=== El Niño events (ONI >= 0.5 for >= 5 months) ===")
    print(el_nino_events)
    # Check: our labels vs the file's own `phase` column
    print(pd.crosstab(oni_df["phase"], oni_df["enso_state"], dropna=False))

    # Model table: time features + ONI inputs, joined on month; targets kept separate
    month_start = pd.to_datetime(time_df["time"]).dt.to_period("M").dt.to_timestamp()
    model_df = time_df.set_index(month_start).join(oni_feats, how="left")
    y = oni_targets.reindex(model_df.index)["el_nino_in_3m"]
    w = sample_weights(y, oni_targets.reindex(model_df.index)["target_provisional_3m"])
    print(model_df.tail(3))