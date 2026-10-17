"""
Mixed-layer depth (MLD) model for the NE Pacific box off Washington / Vancouver Island.

MLD is how deep the wind-stirred surface layer of the ocean reaches: deep in winter,
shallow in summer. We predict it from the same month's ocean and wind conditions plus
recent El Niño state, train on years before 2020 and test on 2020 onwards.

Terms:
  Normal / climatology  Average value for a calendar month at a grid point (training years).
  Anomaly               Actual minus normal, e.g. +1.5 °C = warmer than usual for that month.
  ONI                   El Niño index. El Niño = ONI >= +0.5 for 5+ months in a row.
  Feature / target      Model input column / the value the model predicts.
  RMSE                  Typical prediction error, in metres. Lower is better.
  R²                    Share of the variation the model explains (1 = perfect, 0 = none).
"""

import json

import boto3
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score


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

ocean = xr.open_zarr(f"s3://{OUT_DIR}/ocean_monthly.zarr", storage_options=opts)
wind  = xr.open_zarr(f"s3://{OUT_DIR}/wind_monthly.zarr",  storage_options=opts)


# --- 2. Cut out our box: 46-52°N, 134-120°W -------------------------------------
cut = dict(latitude=slice(46, 52), longitude=slice(-134, -120))
oc = ocean.sel(**cut).load()
wd = wind.sel(**cut).load()


# --- 3. Align ocean and wind ------------------------------------------------------
# The ocean grid is twice as fine as the wind grid: average 2x2 ocean cells so each
# ocean value sits on a wind grid point.
def regrid(ds):
    ds = ds.coarsen(latitude=2, longitude=2, boundary="exact").mean()
    return ds.assign_coords(latitude=wd.latitude, longitude=wd.longitude)

oc = regrid(oc)

T = np.intersect1d(oc.time.values, wd.time.values)      # months present in both
oc, wd = oc.sel(time=T), wd.sel(time=T)


# --- 4. Flatten to a table: one row per (month, grid point) -------------------------
def level(da, z):
    """Value at depth z metres, if the variable has a depth dimension."""
    if "depth" in da.dims:
        return da.sel(depth=z, method="nearest").drop_vars("depth")
    return da

columns = {
    "sst":   level(oc.to, 0),                                    # sea surface temperature
    "sss":   level(oc.so, 0),                                    # sea surface salinity
    "u":     wd.eastward_wind,
    "v":     wd.northward_wind,
    "speed": np.hypot(wd.eastward_wind, wd.northward_wind),
    "tau":   wd.wind_stress_magnitude,                           # how hard wind pushes the surface
}
# Subsurface temperature/salinity. Strongly layered water is hard to mix, which keeps the
# mixed layer shallow. Built but not yet used as features - worth trying.
if "depth" in oc.to.dims:
    for z in [50, 100, 200]:
        columns[f"t_{z}"] = level(oc.to, z)
        columns[f"s_{z}"] = level(oc.so, z)
        columns[f"dT_0_{z}"] = columns["sst"] - columns[f"t_{z}"]

columns["y"] = oc.mlotst                                         # MLD in metres

df = xr.Dataset(columns).to_dataframe().reset_index().dropna()   # dropna removes land points
df["month"] = df.time.dt.month


# --- 5. Normals and anomalies -----------------------------------------------------
# MLD swings by tens of metres with the seasons; El Niño shifts it by a few. Predicting
# "MLD minus its normal for that month" lets the model focus on the unusual part instead
# of relearning the seasons. Inputs get the same treatment: "+1.5 °C vs normal" means the
# same thing in any month, "12 °C" doesn't. The normals are also passed in as features,
# so the model still knows the absolute levels.
#
# Normals use training years (pre-2020) only, so no test-year data leaks into training.
keys = ["month", "latitude", "longitude"]
values = ["y", "sst", "sss", "tau", "u", "v", "speed"]
normals = (df[df.time < "2020"].groupby(keys)[values].mean()
           .add_suffix("_clim").reset_index())
df = df.merge(normals, on=keys, how="left").dropna()

for c in values[1:]:
    df[c + "_anom"] = df[c] - df[c + "_clim"]
df["target"] = df.y - df.y_clim


# --- 6a. Time features ------------------------------------------------------------
# t_years: continuous clock, so the model can pick up long-term trends.
# month (from step 4): treated as a category in step 9, so seasons are learned without
# implying 12 is "far" from 1.
T0 = df.time.min()
df["t_years"] = (df.time - T0).dt.total_seconds() / (365.25 * 24 * 3600)


# --- 6b. El Niño features from ONI --------------------------------------------------
# Leakage rules (the model may only use what was known at the time):
#  - Each month's ONI averages the month before, the month itself AND the month after, so
#    features only use ONI from last month or earlier.
#  - Whether a month is "El Niño" depends on the following months (5-month rule), so that
#    label is never a feature. It's used only to weight training and to score results.
ONI_PATH = "ocean-hackathon-2026/teams/team-elnino/processed/enso.json"
EL_NINO_THRESHOLD = 0.5
LA_NINA_THRESHOLD = -0.5
MIN_EVENT_MONTHS = 5
EL_NINO_WEIGHT = 3.0            # El Niño months count 3x in training - the case we care most about

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

def in_event(condition, min_months):
    """True inside runs where `condition` held for >= min_months in a row."""
    stretch = (condition != condition.shift()).cumsum()             # id per run of equal values
    return condition & (condition.groupby(stretch).transform("size") >= min_months)

def streak_length(condition):
    """Consecutive months `condition` has been True, up to and including each row."""
    stretch = (condition != condition.shift()).cumsum()
    return condition.astype(int).groupby(stretch).cumsum()

def oni_monthly(oni_df):
    """One row per month: ONI features + that month's El Niño state (+1 / 0 / -1)."""
    s = oni_df.set_index("time").sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))   # no gaps, so shifts = months
    oni, prev = s["oni"], s["oni"].shift(1)                        # prev = last month's ONI
    out = pd.DataFrame(index=s.index)
    for lag in (1, 2, 3, 6, 12):
        out[f"oni_lag{lag}"] = oni.shift(lag)
    out["oni_trend_3m"] = prev - oni.shift(4)                      # warming or cooling
    out["oni_mean_12m"] = prev.rolling(12, min_periods=6).mean()
    out["warm_streak_months"] = streak_length(prev >= EL_NINO_THRESHOLD)
    out["cold_streak_months"] = streak_length(prev <= LA_NINA_THRESHOLD)

    out["enso_state"] = np.select([in_event(oni >= EL_NINO_THRESHOLD, MIN_EVENT_MONTHS),
                                   in_event(oni <= LA_NINA_THRESHOLD, MIN_EVENT_MONTHS)], [1, -1], 0)
    # A spell still running at the end of the data, under 5 months, may yet become an event:
    # mark it unknown rather than neutral.
    for condition in (oni >= EL_NINO_THRESHOLD, oni <= LA_NINA_THRESHOLD):
        months_so_far = int(streak_length(condition).iloc[-1])
        if 0 < months_so_far < MIN_EVENT_MONTHS:
            out.loc[out.index[-months_so_far:], "enso_state"] = np.nan
    return out.rename_axis("ym").reset_index()

ONI_FEATS = ["oni_lag1", "oni_lag2", "oni_lag3", "oni_lag6", "oni_lag12",
             "oni_trend_3m", "oni_mean_12m", "warm_streak_months", "cold_streak_months"]

# ONI is one value per month for the whole Pacific, so every grid point in a month gets the same values
oni_m = oni_monthly(load_oni())
df["ym"] = df.time.dt.to_period("M").dt.to_timestamp()
df = df.merge(oni_m, on="ym", how="left").drop(columns="ym")


# --- 7. Features and train/test split ---------------------------------------------
feats = ["sst_anom", "sss_anom", "tau_anom", "u_anom", "v_anom", "speed_anom",  # vs normal
         "y_clim", "sst_clim",                                                  # the normals
         "month", "t_years",
         "latitude", "longitude"] + ONI_FEATS

# Split by date, not randomly: neighbouring months are nearly identical, so a random split
# lets the model see near-copies of the test data and overstates accuracy.
train, test = df[df.time < "2020"], df[df.time >= "2020"].copy()


# --- 8. Inspect the model input -----------------------------------------------------
print(f"\n=== Model input ===  train {train[feats].shape}, test {test[feats].shape}  "
      f"(rows, columns), target = 'target'")
with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
    print(train[["time"] + feats + ["target"]].head().T)          # transposed: one line per column
    print(train[feats].dtypes.value_counts())

# Per-month distribution of each numeric feature and the target (all years, all grid points)
num_cols = [c for c in feats if c not in ("month", "t_years", "latitude", "longitude")] + ["target"]
g = df.groupby("month")[num_cols]
month_stats = pd.concat({"min": g.min(), "Q1": g.quantile(0.25), "mean": g.mean(),
                         "median": g.median(), "Q3": g.quantile(0.75), "max": g.max()}, axis=1)
with pd.option_context("display.max_rows", None, "display.width", 200):
    for c in num_cols:
        print(f"\n--- {c} by month ---")
        print(month_stats.xs(c, axis=1, level=1).round(3))


# --- 9. Train ---------------------------------------------------------------------
# Gradient boosting = many small decision trees, each correcting the previous ones' errors.
# Handles non-linear effects and interactions (e.g. warm water matters more in winter)
# without hand-crafting them. min_samples_leaf and l2_regularization stop it memorising
# noise; a low learning_rate with many trees makes it steadier.
model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                      min_samples_leaf=200, l2_regularization=1.0,
                                      categorical_features=["month"], random_state=0)

w = np.where(train.enso_state == 1, EL_NINO_WEIGHT, 1.0)
model.fit(train[feats], train.target, sample_weight=w)

# The model predicts the anomaly; add the normal back to get MLD in metres
test["pred"] = test.y_clim + model.predict(test[feats])


# --- 10. Evaluate -----------------------------------------------------------------
# Baseline = "just predict the normal MLD for that month". The model has to beat this.
rmse = lambda p: mean_squared_error(test.y, p) ** 0.5
print(f"rows: train {len(train):,}  test {len(test):,}")
print(f"climatology RMSE (train-years only): {rmse(test.y_clim):.2f} m")
print(f"model RMSE:                          {rmse(test.pred):.2f} m   (R² = {r2_score(test.y, test.pred):.2f})")

# Error by calendar month (winter is usually hardest)
by_m = test.groupby("month").apply(lambda g: pd.Series({
    "clim": mean_squared_error(g.y, g.y_clim) ** 0.5,
    "model": mean_squared_error(g.y, g.pred) ** 0.5}))
print(by_m.round(1))

# Error by El Niño state - the El Niño row is the one we care most about
state = test.enso_state.map({1: "El Niño", 0: "Neutral", -1: "La Niña"}).fillna("unknown")
by_enso = test.groupby(state).apply(lambda g: pd.Series({
    "months": g.time.nunique(),
    "clim": mean_squared_error(g.y, g.y_clim) ** 0.5,
    "model": mean_squared_error(g.y, g.pred) ** 0.5}))
print(by_enso.round(1))


# --- 11. Feature importance -------------------------------------------------------
# Shuffle one feature at a time and measure how much worse the model gets.
# Big drop = the model relies on it; ~0 = barely used.
sample = test.sample(min(5000, len(test)), random_state=0)
imp = permutation_importance(model, sample[feats], sample.target, n_repeats=5, random_state=0)
print(pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False).round(3))


# --- 12. Predict an MLD map for one month ---------------------------------------------
def predict_month(when):
    """Actual vs predicted MLD per grid point, e.g. predict_month("2023-01")."""
    sub = df[df.time.dt.to_period("M") == pd.Period(when)].copy()
    sub["mld_pred"] = sub.y_clim + model.predict(sub[feats])
    return sub.set_index(["latitude", "longitude"])[["y", "mld_pred"]].to_xarray()
