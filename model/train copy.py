"""
Mixed-layer depth (MLD) model for the NE Pacific box off Washington / Vancouver Island.

MLD is how deep the wind-stirred surface layer of the ocean reaches: deep in winter,
shallow in summer. We predict it from the same month's ocean and wind conditions plus
the state of the tropical Pacific (El Niño, ONI) and the North Pacific (PDO), train on
years before 2020 and test on 2020 onwards.

In use, the ocean and wind inputs come from other models' forecasts (e.g. SEAS5). Here we
train and test on observed values, so the test score assumes perfect input forecasts: real
forecasts will do worse, by however much error the input forecasts carry.

Terms:
  Normal / climatology  Average value for a calendar month at a grid point (training years).
  Anomaly               Actual minus normal, e.g. +1.5 °C = warmer than usual for that month.
  ONI                   El Niño index. El Niño = ONI >= +0.5 for 5+ months in a row.
  PDO                   Pacific Decadal Oscillation: North Pacific temperature pattern.
                        Positive = warm along our coast. Shifts phase every few years-decades.
  Feature / target      Model input column / the value the model predicts.
  RMSE                  Typical prediction error, in metres. Lower is better.
  R²                    Share of the variation the model explains (1 = perfect, 0 = none).
"""

import json
import warnings

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
    # Check the averaged cells really sit on the wind points before relabelling them:
    # an offset or a flipped axis would otherwise silently shift the ocean data.
    for c in ["latitude", "longitude"]:
        offset = np.abs(ds[c].values - wd[c].values).max()
        if offset > 0.01:
            warnings.warn(f"regrid: ocean {c} is up to {offset:.3f}° off the wind grid")
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

# These six are the inputs predict_mld() takes (step 13), so the model is trained on exactly them
columns = {
    "sst":   level(oc.to, 0),                                    # sea surface temperature
    "sss":   level(oc.so, 0),                                    # sea surface salinity
    "ssh":   oc.zo,                                              # sea surface height
    "u":     wd.eastward_wind,
    "v":     wd.northward_wind,
    "speed": wd.wind_speed,
}
# Subsurface temperature/salinity. Strongly layered water is hard to mix, which keeps the
# mixed layer shallow. Built but not yet used as features - worth trying. They are empty
# (NaN) wherever the sea floor is shallower than z, e.g. on the continental shelf.
if "depth" in oc.to.dims:
    for z in [50, 100, 200]:
        columns[f"t_{z}"] = level(oc.to, z)
        columns[f"s_{z}"] = level(oc.so, z)
        columns[f"dT_0_{z}"] = columns["sst"] - columns[f"t_{z}"]

columns["y"] = oc.mlotst                                         # MLD in metres

# Drop land points only. Checking just the surface columns keeps shelf points, whose
# subsurface columns are empty.
SURFACE = ["sst", "sss", "ssh", "u", "v", "speed", "y"]
df = xr.Dataset(columns).to_dataframe().reset_index().dropna(subset=SURFACE)
df["month"] = df.time.dt.month


# --- 5. Normals and anomalies -----------------------------------------------------
# MLD swings by tens of metres with the seasons; El Niño shifts it by a few. Predicting
# "MLD minus its normal for that month" lets the model focus on the unusual part instead
# of relearning the seasons. Inputs get the same treatment: "+1.5 °C vs normal" means the
# same thing in any month, "12 °C" doesn't. The normals are also passed in as features,
# so the model still knows the absolute levels.
#
# Normals use training years (pre-2020) only, so no test-year data leaks into training.
TEST_START = "2010"
keys = ["month", "latitude", "longitude"]
values = ["y", "sst", "sss", "ssh", "u", "v", "speed"]
normals = (df[df.time < TEST_START].groupby(keys)[values].mean()
           .add_suffix("_clim").reset_index())
df = df.merge(normals, on=keys, how="left").dropna(subset=[c + "_clim" for c in values])

for c in values[1:]:
    df[c + "_anom"] = df[c] - df[c + "_clim"]
df["target"] = df.y - df.y_clim


# --- 6a. Time features ------------------------------------------------------------
# t_years: continuous clock, so the model can pick up long-term trends (e.g. MLD slowly
# shoaling as the ocean warms). Trees can't extrapolate it, so every month after the
# training years gets the trend as it stood at the end of 2019 - better than no trend.
# month (from step 4): treated as a category in step 9, so seasons are learned without
# implying 12 is "far" from 1.
T0 = df.time.min()
df["t_years"] = (df.time - T0).dt.total_seconds() / (365.25 * 24 * 3600)


# --- 6b. El Niño features from ONI --------------------------------------------------
# El Niño reaches our coast mostly through the atmosphere (stronger Gulf of Alaska winter
# storms) within weeks, so this month's ONI matters most; the lags cover the slower ocean route.
#  - oni_now is this month's ONI, which averages the month before, the month itself AND the
#    month after. Like the ocean and wind inputs, in use it comes from a forecast (NOAA CPC /
#    IRI ENSO outlook, or SEAS5 Niño 3.4), so the test score assumes a perfect ONI forecast.
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
    out["oni_now"] = oni
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

ONI_FEATS = ["oni_now", "oni_lag1", "oni_lag2", "oni_lag3", "oni_lag6", "oni_lag12",
             "oni_trend_3m", "oni_mean_12m", "warm_streak_months", "cold_streak_months"]

# ONI is one value per month for the whole Pacific, so every grid point in a month gets the same values
oni_obs = load_oni()
oni_m = oni_monthly(oni_obs)
df["ym"] = df.time.dt.to_period("M").dt.to_timestamp()
df = df.merge(oni_m, on="ym", how="left")


# --- 6c. North Pacific state from the PDO --------------------------------------------
# ONI describes the tropical Pacific; the PDO describes the North Pacific itself, which our
# box responds to more directly (the 2014-15 "Blob" heatwave wasn't El Niño). It's also a
# candidate for the jump in MLD around 2000-2002: the PDO changed phase around 1998-99.
# One monthly value for the whole region, like ONI. Not a 3-month average, so this month's
# value only uses this month.
# Read from our S3 folder if a copy was uploaded there, else downloaded from NOAA; if
# neither works the model runs without PDO.
PDO_SOURCES = [f"{OUT_DIR}/ersst.v5.pdo.dat",
               "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"]

def read_pdo(src):
    """NOAA ERSST v5 PDO file (title line, then year + 12 monthly values; 99.99 = missing)
    -> Series of PDO indexed by month start."""
    if src.startswith("http"):
        table = pd.read_csv(src, sep=r"\s+", skiprows=1)
    else:
        with fs.open(src) as f:
            table = pd.read_csv(f, sep=r"\s+", skiprows=1)
    table = table.iloc[:, :13].apply(pd.to_numeric, errors="coerce")
    table.columns = ["year"] + list(range(1, 13))
    long = table.dropna(subset=["year"]).melt(id_vars="year", var_name="month", value_name="pdo")
    long["time"] = pd.to_datetime(long[["year", "month"]].astype(int).assign(day=1))
    long = long[long.pdo.abs() < 90]
    return long.set_index("time").pdo.sort_index()

def pdo_monthly(pdo):
    """One row per month: this month's PDO and its 12-month mean (the slow background state)."""
    s = pdo.reindex(pd.date_range(pdo.index.min(), pdo.index.max(), freq="MS"))
    return (pd.DataFrame({"pdo": s, "pdo_mean_12m": s.rolling(12, min_periods=6).mean()})
            .rename_axis("ym").reset_index())

pdo_obs = None
for src in PDO_SOURCES:
    try:
        pdo_obs = read_pdo(src)
        print(f"PDO: {len(pdo_obs)} months from {src}, ends {pdo_obs.index.max():%Y-%m}")
        break
    except Exception as e:
        print(f"PDO: couldn't read {src} ({type(e).__name__}: {e})")
if pdo_obs is None:
    warnings.warn("No PDO data: training without it. Upload ersst.v5.pdo.dat from NOAA to "
                  f"s3://{PDO_SOURCES[0]} to use it.")
PDO_FEATS = ["pdo", "pdo_mean_12m"] if pdo_obs is not None else []
if PDO_FEATS:
    df = df.merge(pdo_monthly(pdo_obs), on="ym", how="left")
df = df.drop(columns="ym")

# Months without full ONI/PDO history (the first year of the files, or months past their
# end) are dropped, so the model never learns from missing values and can't misread them later.
missing = df[ONI_FEATS + PDO_FEATS].isna().any(axis=1)
print(f"Dropping {df.time[missing].nunique()} months without full ONI/PDO features: "
      f"{sorted(df.time[missing].dt.strftime('%Y-%m').unique())}")
df = df[~missing]


# --- 7. Features and train/test split ---------------------------------------------
feats = ["sst_anom", "sss_anom", "ssh_anom", "u_anom", "v_anom", "speed_anom",  # vs normal
         "y_clim", "sst_clim",                                                  # the normals
         "month", "t_years",
         "latitude", "longitude"] + ONI_FEATS + PDO_FEATS

# Split by date, not randomly: neighbouring months are nearly identical, so a random split
# lets the model see near-copies of the test data and overstates accuracy.
train, test = df[df.time < TEST_START], df[df.time >= TEST_START].copy()


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
# R² is scored on the anomaly (MLD minus normal): on raw MLD the seasons alone would
# give a high R², even for the baseline.
rmse = lambda p: mean_squared_error(test.y, p) ** 0.5
print(f"rows: train {len(train):,}  test {len(test):,}")
print(f"climatology RMSE (train-years only): {rmse(test.y_clim):.2f} m")
print(f"model RMSE:                          {rmse(test.pred):.2f} m")
print(f"skill vs climatology:                {1 - rmse(test.pred) / rmse(test.y_clim):.2f}"
      "   (0 = no better than the normals, 1 = perfect)")
print(f"anomaly R²:                          {r2_score(test.target, test.pred - test.y_clim):.2f}"
      "   (share of the unusual part explained)")
print(f"raw-MLD R²: model {r2_score(test.y, test.pred):.2f}, "
      f"climatology alone {r2_score(test.y, test.y_clim):.2f}   (the gap is what the model adds)")
# Error by calendar month (winter is usually hardest)
by_m = test.groupby("month").apply(lambda g: pd.Series({
    "clim": mean_squared_error(g.y, g.y_clim) ** 0.5,
    "model": mean_squared_error(g.y, g.pred) ** 0.5}))
print(by_m.round(1))

# Error by El Niño state - the El Niño row is the one we care most about. The test years
# hold only one El Niño (2023-24), so that row is one case study, not a general result.
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

# --- 11b. Long-term drift ------------------------------------------------------------
# t_years shows ~0 in the importance list whatever it does: every test month sits in the same
# "latest years" branch, so shuffling it among them changes nothing. These show it directly.
# 1. In the data: average MLD minus its 1993-2019 normal, by year (negative = shallower than normal)
print(df.groupby(df.time.dt.year).target.mean().round(2))

# 2. In the model: the same test rows predicted as if they came from each year
s = test.sample(5000, random_state=0)
effect = {}
for yr in range(1994, 2010):
    X = s[feats].copy()
    X["t_years"] = (yr - T0.year) + X["t_years"] % 1      # keep the month, change the year
    effect[yr] = model.predict(X).mean()
effect["2010+"] = model.predict(s[feats]).mean()
print(pd.Series(effect).round(2))

# --- 12. Predict an MLD map for one month ---------------------------------------------
def predict_month(when):
    """Actual vs predicted MLD per grid point, e.g. predict_month("2023-01")."""
    if pd.Period(when) < pd.Period(TEST_START, "M"):
        warnings.warn(f"{when} is a training month: the model has seen it, so the fit will look too good")
    sub = df[df.time.dt.to_period("M") == pd.Period(when)].copy()
    sub["mld_pred"] = sub.y_clim + model.predict(sub[feats])
    return sub.set_index(["latitude", "longitude"])[["y", "mld_pred"]].to_xarray()


# --- 13. Predict MLD from raw measurements --------------------------------------------
# The model works on anomalies, so raw inputs are first compared with the normal for that
# month and grid point (step 5). The ONI features need ONI for the month of `date` and the
# 12 months before it. Past the end of the ONI file, pass forecast ONI values in
# `oni_forecast`; observed values always take priority over forecast ones. PDO works the
# same way through `pdo_forecast`, except that missing months fall back to the last known
# PDO (it changes slowly), with a warning.
def predict_mld(temp, salinity, wind_speed, surface_height, north_wind, east_wind,
                date, latitude, longitude, oni_forecast=None, pdo_forecast=None):
    """Predicted mixed-layer depth in metres for one month at one location.

    temp             sea surface temperature, °C
    salinity         sea surface salinity (same units as the data, ~32-34)
    wind_speed       wind speed at 10 m, m/s
    surface_height   sea surface height, m
    north_wind       northward wind at 10 m, m/s
    east_wind        eastward wind at 10 m, m/s
    date             any date in the month, e.g. "2024-01"
    latitude, longitude  inside the box (46-52°N, 134-120°W); snapped to the nearest grid point
    oni_forecast     optional {"YYYY-MM": oni} for months past the end of the ONI file,
                     up to and including the month of `date`
    pdo_forecast     optional {"YYYY-MM": pdo}, the same for the PDO file

    Inputs from a forecast model (e.g. SEAS5) should be bias-corrected to this data first:
    any offset between that model and the training data reads as an anomaly.

    Example: predict_mld(10.5, 32.4, 9.0, 0.35, 2.0, 5.5, "2024-01", 48.5, -128.0)
             predict_mld(..., "2027-01", 48.5, -128.0,
                         oni_forecast={"2026-10": 0.9, "2026-11": 1.1, "2026-12": 1.2,
                                       "2027-01": 1.2})
    """
    when = pd.Timestamp(date)
    lats, lons = normals.latitude.unique(), normals.longitude.unique()
    lat, lon = lats[np.abs(lats - latitude).argmin()], lons[np.abs(lons - longitude).argmin()]
    if abs(lat - latitude) > 0.5 or abs(lon - longitude) > 0.5:
        raise ValueError(f"({latitude}, {longitude}) is outside the model's area")
    norm = normals[(normals.month == when.month) & (normals.latitude == lat) & (normals.longitude == lon)]
    if norm.empty:
        raise ValueError(f"No ocean data at grid point ({lat}, {lon}) - probably land")
    norm = norm.iloc[0]

    raw = {"sst": temp, "sss": salinity, "ssh": surface_height,
           "u": east_wind, "v": north_wind, "speed": wind_speed}
    x = {f"{k}_anom": v - norm[f"{k}_clim"] for k, v in raw.items()}
    x.update(y_clim=norm.y_clim, sst_clim=norm.sst_clim, month=when.month,
             t_years=(when - T0).total_seconds() / (365.25 * 24 * 3600),
             latitude=lat, longitude=lon)

    oni_table = oni_m
    if oni_forecast:
        extra = pd.DataFrame({"time": pd.to_datetime(list(oni_forecast), format="%Y-%m"),
                              "oni": [float(v) for v in oni_forecast.values()]})
        series = pd.concat([oni_obs[["time", "oni"]], extra]).drop_duplicates("time", keep="first")
        oni_table = oni_monthly(series)
    month_start = when.to_period("M").to_timestamp()
    oni_row = oni_table[oni_table.ym == month_start]
    if oni_row.empty or oni_row[ONI_FEATS].isna().values.any():
        raise ValueError(f"Not enough ONI to predict {month_start:%Y-%m}: observed ONI ends "
                         f"{oni_obs.time.max():%Y-%m}; pass the later months in oni_forecast")
    for c in ONI_FEATS:
        x[c] = oni_row[c].iloc[0]

    if PDO_FEATS:
        pdo = pdo_obs
        if pdo_forecast:
            extra = pd.Series({pd.Timestamp(k): float(v) for k, v in pdo_forecast.items()})
            pdo = pd.concat([pdo, extra[~extra.index.isin(pdo.index)]]).sort_index()
        last = pdo.index.max()
        if month_start > last:
            warnings.warn(f"No PDO for {month_start:%Y-%m}: holding the last known value "
                          f"({last:%Y-%m}) - pass pdo_forecast to override")
            gap = pd.date_range(last + pd.offsets.MonthBegin(1), month_start, freq="MS")
            pdo = pd.concat([pdo, pd.Series(pdo.iloc[-1], index=gap)])
        pdo_row = pdo_monthly(pdo)
        pdo_row = pdo_row[pdo_row.ym == month_start]
        if pdo_row.empty or pdo_row[PDO_FEATS].isna().values.any():
            raise ValueError(f"Not enough PDO to predict {month_start:%Y-%m}: "
                             "check pdo_forecast for gaps")
        for c in PDO_FEATS:
            x[c] = pdo_row[c].iloc[0]

    anomaly = model.predict(pd.DataFrame([x])[feats])[0]
    return float(norm.y_clim + anomaly)
