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

feats = ["sst_anom", "sss_anom", "tau_anom", "u_anom", "v_anom", "speed_anom",
         "y_clim", "sst_clim", "month", "latitude", "longitude"]

# 6. Split by time
train, test = df[df.time < "2020"], df[df.time >= "2020"].copy()

# 7. Fit on the departure, then add the normal MLD back
model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                      min_samples_leaf=200, l2_regularization=1.0,
                                      random_state=0)
model.fit(train[feats], train.target)
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

# 8. Which inputs matter
sample = test.sample(min(5000, len(test)), random_state=0)
imp = permutation_importance(model, sample[feats], sample.target, n_repeats=5, random_state=0)
print(pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False).round(3))

# 9. Predict a full MLD map for any month
def predict_month(when):                       # e.g. "2023-01"
    sub = df[df.time.dt.to_period("M") == pd.Period(when)].copy()
    sub["mld_pred"] = sub.y_clim + model.predict(sub[feats])
    return sub.set_index(["latitude", "longitude"])[["y", "mld_pred"]].to_xarray()