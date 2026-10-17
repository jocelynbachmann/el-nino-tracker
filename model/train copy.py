# STEP 2 - CONNECT TO THE DATA  (run every time you open the notebook)
import boto3, s3fs, json
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

import numpy as np, pandas as pd, xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cut the box [N, W, S, E] = [52, -134, 46, -120]
cut = dict(latitude=slice(46, 52), longitude=slice(-134, -120))
oc, cl = ocean.sel(**cut).load(), clim.sel(**cut).load()
wd = wind.sel(**cut).load()

# 2. Ocean onto the wind grid (2x2 average)
def regrid(ds):
    ds = ds.coarsen(latitude=2, longitude=2, boundary="exact").mean()
    return ds.assign_coords(latitude=wd.latitude, longitude=wd.longitude)
oc, cl = regrid(oc), regrid(cl)

# 3. Months present in both datasets
T = np.intersect1d(oc.time.values, wd.time.values)
oc, wd = oc.sel(time=T), wd.sel(time=T)

# 4. Features: SAME month, raw values (no anomalies, no lagged MLD)
def level(da, z):
    """Pick a depth level if the variable has depth; otherwise return as-is."""
    if "depth" in da.dims:
        return da.sel(depth=z, method="nearest").drop_vars("depth")
    return da

X = {
    "sst":   level(oc.to, 0),
    "sss":   level(oc.so, 0),
    "u":     wd.eastward_wind,
    "v":     wd.northward_wind,
    "speed": np.hypot(wd.eastward_wind, wd.northward_wind),
    "tau":   wd.wind_stress_magnitude,
}
# Subsurface T/S tell the model about stratification, which strongly controls MLD
if "depth" in oc.to.dims:
    for z in [50, 100, 200]:
        X[f"t_{z}"] = level(oc.to, z)
        X[f"s_{z}"] = level(oc.so, z)
        X[f"dT_0_{z}"] = X["sst"] - X[f"t_{z}"]

X["y"] = oc.mlotst
df = xr.Dataset(X).to_dataframe().reset_index().dropna()
df["month"] = df.time.dt.month
feats = [c for c in df.columns if c not in ("time", "y")]   # includes month, lat, lon

# 5. Climatology from TRAINING years only (fair baseline, no peeking at 2020+)
keys = ["month", "latitude", "longitude"]
vars_ = ["y", "sst", "sss", "tau", "u", "v", "speed"]
clim_tr = (df[df.time < "2020"].groupby(keys)[vars_].mean()
           .add_suffix("_clim").reset_index())
df = df.merge(clim_tr, on=keys, how="left").dropna()

# Inputs as departures from normal (removes drift), target = MLD minus normal MLD
for c in vars_[1:]:
    df[c + "_anom"] = df[c] - df[c + "_clim"]
df["target"] = df.y - df.y_clim

# 5b. Continuous time: fractional years since the first month, so the date is one ordered
#     number. `month` stays its own 1-12 column and is treated as a category (see step 7).
T0 = df.time.min()
df["t_years"] = (df.time - T0).dt.total_seconds() / (365.25 * 24 * 3600)

# 5c. El Niño state from the ONI file (month, oni, phase, provisional)
# NOAA rule: El Niño = ONI >= +0.5 (La Niña <= -0.5) for at least 5 months in a row.
# Each month's ONI is a 3-month mean centred on that month, so it already includes the next
# month. Model inputs therefore only use ONI from t-1 and earlier. The event label looks
# ahead, so it is only used to weight and score the model, never as an input.
ONI_PATH = "ocean-hackathon-2026/teams/team-elnino/processed/enso.json"
EL_NINO_THRESHOLD, LA_NINA_THRESHOLD, MIN_EVENT_MONTHS = 0.5, -0.5, 5
EL_NINO_WEIGHT = 3.0            # how much more an El Niño month counts in training

def load_oni(path=ONI_PATH):
    """Read the ONI file from S3 and return one row per month, sorted by time:
      time        Timestamp, first day of the month
      oni         float
      phase       str ("el_nino" / "la_nina" / "neutral", as given in the file)
      provisional bool

    The file is a JSON list of {"month": "1993-01", "oni": 0.13, "phase": "neutral",
    "provisional": false}.
    """
    with fs.open(path) as f:
        oni = pd.DataFrame(json.load(f))
    oni["time"] = pd.to_datetime(oni["month"], format="%Y-%m")
    oni["oni"] = oni["oni"].astype(float)
    oni["provisional"] = oni["provisional"].astype(bool)
    return (oni[["time", "oni", "phase", "provisional"]]
            .drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True))

def _event_runs(mask, min_len):
    """True where the month is inside a run of True lasting >= min_len months."""
    run = (mask != mask.shift()).cumsum()
    return mask & (mask.groupby(run).transform("size") >= min_len)

def _months_in_streak(mask):
    """Months the condition has held up to and including each row, counting backwards only."""
    run = (mask != mask.shift()).cumsum()
    return mask.astype(int).groupby(run).cumsum()

def oni_monthly(oni_df):
    """One row per month: lagged ONI inputs + that month's ENSO state (+1 El Niño, 0, -1 La Niña)."""
    s = oni_df.set_index("time").sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
    oni, prev = s["oni"], s["oni"].shift(1)
    out = pd.DataFrame(index=s.index)
    for lag in (1, 2, 3, 6, 12):
        out[f"oni_lag{lag}"] = oni.shift(lag)
    out["oni_trend_3m"] = prev - oni.shift(4)                     # warming or cooling
    out["oni_mean_12m"] = prev.rolling(12, min_periods=6).mean()
    out["warm_streak_months"] = _months_in_streak(prev >= EL_NINO_THRESHOLD)
    out["cold_streak_months"] = _months_in_streak(prev <= LA_NINA_THRESHOLD)
    out["enso_state"] = np.select([_event_runs(oni >= EL_NINO_THRESHOLD, MIN_EVENT_MONTHS),
                                   _event_runs(oni <= LA_NINA_THRESHOLD, MIN_EVENT_MONTHS)], [1, -1], 0)
    # A run still going at the end of the record may yet reach 5 months: unknown, not neutral
    for mask in (oni >= EL_NINO_THRESHOLD, oni <= LA_NINA_THRESHOLD):
        tail = int(_months_in_streak(mask).iloc[-1])
        if 0 < tail < MIN_EVENT_MONTHS:
            out.loc[out.index[-tail:], "enso_state"] = np.nan
    return out.rename_axis("ym").reset_index()

ONI_FEATS = ["oni_lag1", "oni_lag2", "oni_lag3", "oni_lag6", "oni_lag12",
             "oni_trend_3m", "oni_mean_12m", "warm_streak_months", "cold_streak_months"]
oni_m = oni_monthly(load_oni())
df["ym"] = df.time.dt.to_period("M").dt.to_timestamp()
df = df.merge(oni_m, on="ym", how="left").drop(columns="ym")

feats = ["sst_anom", "sss_anom", "tau_anom", "u_anom", "v_anom", "speed_anom",
         "y_clim", "sst_clim", "month", "t_years", "latitude", "longitude"] + ONI_FEATS

# 6. Split by time
train, test = df[df.time < "2020"], df[df.time >= "2020"].copy()

# 7. Fit on the departure, then add the normal MLD back
#    month is categorical (no false order like "December is far from January");
#    El Niño months get extra weight so the model works hardest to get them right.
model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                      min_samples_leaf=200, l2_regularization=1.0,
                                      categorical_features=["month"], random_state=0)
w = np.where(train.enso_state == 1, EL_NINO_WEIGHT, 1.0)
model.fit(train[feats], train.target, sample_weight=w)
test["pred"] = test.y_clim + model.predict(test[feats])   # real MLD in metres

rmse = lambda p: mean_squared_error(test.y, p) ** 0.5
print(f"rows: train {len(train):,}  test {len(test):,}")
print(f"climatology RMSE (train-years only): {rmse(test.y_clim):.2f} m")
print(f"model RMSE:                          {rmse(test.pred):.2f} m   (R² = {r2_score(test.y, test.pred):.2f})")

# Error by month - winter is usually where it's hardest
by_m = test.groupby("month").apply(lambda g: pd.Series({
    "clim": mean_squared_error(g.y, g.y_clim) ** 0.5,
    "model": mean_squared_error(g.y, g.pred) ** 0.5}))
print(by_m.round(1))

# Error by ENSO state - the El Niño row is the one to watch
state = test.enso_state.map({1: "El Niño", 0: "Neutral", -1: "La Niña"}).fillna("unknown")
by_enso = test.groupby(state).apply(lambda g: pd.Series({
    "months": g.time.nunique(),
    "clim": mean_squared_error(g.y, g.y_clim) ** 0.5,
    "model": mean_squared_error(g.y, g.pred) ** 0.5}))
print(by_enso.round(1))

# 8. Which inputs matter
sample = test.sample(min(5000, len(test)), random_state=0)
imp = permutation_importance(model, sample[feats], sample.target, n_repeats=5, random_state=0)
print(pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False).round(3))

# 9. Predict a full MLD map for any month
def predict_month(when):                       # e.g. "2023-01"
    sub = df[df.time.dt.to_period("M") == pd.Period(when)].copy()
    sub["mld_pred"] = sub.y_clim + model.predict(sub[feats])
    return sub.set_index(["latitude", "longitude"])[["y", "mld_pred"]].to_xarray()