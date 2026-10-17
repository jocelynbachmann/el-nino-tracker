# %% ------------------------------------------------------------------
# DAILY VERSION of the MLD model
# Cells are marked with "# %%" - paste each one into its own notebook cell.
# --------------------------------------------------------------------

# %% STEP 2 - CONNECT TO THE DATA  (unchanged, run every time)
import boto3, s3fs
import numpy as np, pandas as pd, xarray as xr

DATA_FOLDER = "elnino"
PROFILE = "hack-team"
BUCKET  = "ocean-hackathon-2026"
fs = s3fs.S3FileSystem(profile=PROFILE)
opts = {"profile": PROFILE}

who = boto3.Session(profile_name=PROFILE).client("sts").get_caller_identity()["Arn"]
TEAM = who.split("/")[-1]
TEAM_DIR = f"{BUCKET}/teams/{TEAM}"
DATA_DIR = f"{BUCKET}/data/{DATA_FOLDER}"
OUT_DIR  = f"{TEAM_DIR}/processed"

print("Connected as:", TEAM)
print("Raw data:    ", "s3://" + DATA_DIR)
print("Your outputs:", "s3://" + TEAM_DIR)

# The box [N, W, S, E] = [52, -134, 46, -120]
cut = dict(latitude=slice(46, 52), longitude=slice(-134, -120))


# %% STEP 3 - BUILD DAILY STORES FROM THE RAW DATA  (run ONCE, then skip)
# The processed/*.zarr stores are monthly, so daily data has to come from the raw folder.
# Look at the listing below and set the two patterns to match the ocean and wind files.
for p in fs.ls(DATA_DIR):
    print(p)

OCEAN_GLOB = "ocean/*"     # <- adjust to the raw ocean files (zarr or netcdf)
WIND_GLOB  = "wind/*"      # <- adjust to the raw wind files
DEPTHS     = [0, 50, 100, 200]   # keep a few levels; full profiles are huge at daily res

RENAME = {"lat": "latitude", "lon": "longitude", "thetao": "to"}

def open_raw(pattern):
    paths = sorted(fs.glob(f"{DATA_DIR}/{pattern}"))
    if not paths:
        raise FileNotFoundError(f"nothing matches {DATA_DIR}/{pattern}")
    if paths[0].rstrip("/").endswith(".zarr"):
        parts = [xr.open_zarr(f"s3://{p}", storage_options=opts) for p in paths]
    else:
        parts = [xr.open_dataset(fs.open(p), engine="h5netcdf", chunks={}) for p in paths]
    ds = parts[0] if len(parts) == 1 else xr.combine_by_coords(parts, combine_attrs="override")
    ds = ds.rename({k: v for k, v in RENAME.items() if k in ds.variables})
    if float(ds.longitude.max()) > 180:                      # 0..360 -> -180..180
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
    return ds.sortby("latitude").sortby("longitude")

def save(ds, name):
    for v in ds.variables:
        ds[v].encoding = {}                                  # drop source chunk settings
    ds.chunk({"time": 365}).to_zarr(f"s3://{OUT_DIR}/{name}", mode="w",
                                    storage_options=opts)
    print("saved", name, dict(ds.sizes))

oc_raw = open_raw(OCEAN_GLOB).sel(**cut)[["to", "so", "mlotst"]]
if "depth" in oc_raw.dims:
    oc_raw = oc_raw.sel(depth=DEPTHS, method="nearest")
wd_raw = open_raw(WIND_GLOB).sel(**cut)[["eastward_wind", "northward_wind",
                                         "wind_stress_magnitude"]]

# Check the raw time step: daily is needed, finer (e.g. hourly wind) is fine
for name, ds in [("ocean", oc_raw), ("wind", wd_raw)]:
    print(name, "time step:", pd.Series(ds.time.values).diff().median())

# resample to 1 day: averages hourly data and puts both on midnight timestamps,
# so ocean and wind days line up exactly
save(oc_raw.resample(time="1D").mean(), "ocean_daily.zarr")
save(wd_raw.resample(time="1D").mean(), "wind_daily.zarr")


# %% STEP 4 - DAILY MLD MODEL
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score

oc = xr.open_zarr(f"s3://{OUT_DIR}/ocean_daily.zarr", storage_options=opts).sel(**cut).load()
wd = xr.open_zarr(f"s3://{OUT_DIR}/wind_daily.zarr",  storage_options=opts).sel(**cut).load()

# 2. Ocean onto the wind grid (2x2 average)
oc = oc.coarsen(latitude=2, longitude=2, boundary="exact").mean()
oc = oc.assign_coords(latitude=wd.latitude, longitude=wd.longitude)

def level(da, z):
    if "depth" in da.dims:
        return da.sel(depth=z, method="nearest").drop_vars("depth")
    return da

# 3. Wind features. At daily scale the mixed layer responds to wind built up over
#    the previous days, so add trailing means of stress (today + past N-1 days).
#    Done BEFORE trimming so the rolling windows see a continuous record.
tau = wd.wind_stress_magnitude
W = xr.Dataset({
    "u":       wd.eastward_wind,
    "v":       wd.northward_wind,
    "speed":   np.hypot(wd.eastward_wind, wd.northward_wind),
    "tau":     tau,
    "tau_3d":  tau.rolling(time=3).mean(),
    "tau_7d":  tau.rolling(time=7).mean(),
    "tau_14d": tau.rolling(time=14).mean(),
})

# Days present in both datasets
T = np.intersect1d(oc.time.values, W.time.values)
oc, W = oc.sel(time=T), W.sel(time=T)

F = W.assign(sst=level(oc.to, 0), sss=level(oc.so, 0), y=oc.mlotst)

# 4. Smoothed DAILY climatology from training years only.
#    A raw day-of-year mean from a few decades is noisy, so take a circular
#    31-day running mean of it (Dec wraps into Jan). Feb 29 is folded into day 365.
doy = np.minimum(F.time.dt.dayofyear, 365).rename("doy")
F = F.assign_coords(doy=doy)

clim = F.sel(time=F.time < np.datetime64("2020-01-01")).groupby("doy").mean()
pad = 15
clim = xr.concat([clim.isel(doy=slice(-pad, None)), clim, clim.isel(doy=slice(None, pad))], "doy")
clim = clim.rolling(doy=2 * pad + 1, center=True).mean().isel(doy=slice(pad, -pad))
clim_t = clim.sel(doy=F.doy).drop_vars("doy")   # normal value for every day
F = F.drop_vars("doy")

# 5. Inputs as departures from normal; target = MLD minus normal MLD
vars_ = ["sst", "sss", "tau", "u", "v", "speed", "tau_3d", "tau_7d", "tau_14d"]
D = xr.Dataset({f"{v}_anom": F[v] - clim_t[v] for v in vars_})
D["y"], D["y_clim"], D["sst_clim"] = F.y, clim_t.y, clim_t.sst

df = D.to_dataframe().reset_index().dropna()
num = df.select_dtypes("float64").columns
df[num] = df[num].astype("float32")                  # halves memory - daily is ~30x more rows
df["doy"] = np.minimum(df.time.dt.dayofyear, 365)
df["month"] = df.time.dt.month                       # only used for the error table
df["target"] = df.y - df.y_clim

feats = [f"{v}_anom" for v in vars_] + ["y_clim", "sst_clim", "doy", "latitude", "longitude"]

# 6. Split by time
train, test = df[df.time < "2020"], df[df.time >= "2020"].copy()
# If fitting is too slow / runs out of memory, subsample training rows:
# train = train.sample(frac=0.3, random_state=0)

# 7. Fit on the departure, then add the normal MLD back
model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                      min_samples_leaf=1000, l2_regularization=1.0,
                                      early_stopping=True, validation_fraction=0.1,
                                      random_state=0)
model.fit(train[feats], train.target)
test["pred"] = test.y_clim + model.predict(test[feats])

rmse = lambda p: mean_squared_error(test.y, p) ** 0.5
print(f"rows: train {len(train):,}  test {len(test):,}")
print(f"daily climatology RMSE (train-years only): {rmse(test.y_clim):.2f} m")
print(f"model RMSE:                                {rmse(test.pred):.2f} m   "
      f"(R² = {r2_score(test.y, test.pred):.2f})")

# Error by month - winter storms are where daily MLD jumps around most
by_m = test.groupby("month").apply(lambda g: pd.Series({
    "clim":  mean_squared_error(g.y, g.y_clim) ** 0.5,
    "model": mean_squared_error(g.y, g.pred) ** 0.5}))
print(by_m.round(1))

# 8. Which inputs matter
sample = test.sample(min(20000, len(test)), random_state=0)
imp = permutation_importance(model, sample[feats], sample.target, n_repeats=5, random_state=0)
print(pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False).round(3))

# 9. Predict a full MLD map for any day
def predict_day(when):                       # e.g. "2023-01-15"
    sub = df[df.time == pd.Timestamp(when)].copy()
    sub["mld_pred"] = sub.y_clim + model.predict(sub[feats])
    return sub.set_index(["latitude", "longitude"])[["y", "mld_pred"]].to_xarray()

# Time series at one grid point, to see how well it tracks individual storms
def predict_point(lat, lon):
    sub = test[(np.isclose(test.latitude, lat, atol=0.13)) &
               (np.isclose(test.longitude, lon, atol=0.13))]
    return sub.set_index("time")[["y", "y_clim", "pred"]]