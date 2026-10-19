# Daily sea-surface wind for the NE Pacific from Copernicus Marine (satellite-based L4, hourly),
# streamed (no queue) and averaged to daily, then written to our S3 folder year by year.
# Same login as download_daily.py:  pip install copernicusmarine && copernicusmarine login
#
# Three hourly datasets are stitched onto ONE 0.25 deg grid so the record is consistent:
#   1994-06 .. 2007-10  old 0.25 deg (MY)
#   2007-11 .. MY end   0.125 deg (MY), interpolated to 0.25
#   after MY end        0.125 deg (NRT), interpolated to 0.25

# %% SETTINGS
import os, time
import copernicusmarine as cm
import numpy as np
import pandas as pd
import xarray as xr
import s3fs

LAT, LON = (46, 52), (-134, -120)
OLD = "cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1H_202406"
NEW = "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_202211"
NRT = "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H_202207"
SWITCH = pd.Timestamp("2007-11-01")      # first day taken from the 0.125 deg MY dataset

OUT = ("s3://ocean-hackathon-2026/teams/team-elnino/processed/"
       "wind_daily_cmems_1994-present_46-52N_134-120W.zarr")
# Hub: key is in env vars -> {}. Laptop: key is in the "hack-team" profile (server/setup_aws.py).
STORAGE = ({} if os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE")
           else {"profile": os.environ.get("AWS_PROFILE", "hack-team")})
fs = s3fs.S3FileSystem(**STORAGE)

# %% OPEN (lazy) and pick the wind variables
def open_hourly(dataset_id):
    return cm.open_dataset(dataset_id=dataset_id,
                           minimum_latitude=LAT[0], maximum_latitude=LAT[1],
                           minimum_longitude=LON[0], maximum_longitude=LON[1])

old, new, nrt = open_hourly(OLD), open_hourly(NEW), open_hourly(NRT)
print("Variables:", list(old.data_vars))
U = next(v for v in old.data_vars if v in ("eastward_wind", "uwnd", "u10"))
V = next(v for v in old.data_vars if v in ("northward_wind", "vwnd", "v10"))
print(f"Using {U} / {V}")
for name, ds in [("old", old), ("new", new), ("nrt", nrt)]:
    print(f"{name}: {str(ds.time.values[0])[:10]} -> {str(ds.time.values[-1])[:10]}")
NEW_END = pd.Timestamp(new.time.values[-1]).floor("D")   # last full-coverage day of the MY 0.125 set
GRID = dict(latitude=old.latitude, longitude=old.longitude)  # target: the 0.25 deg grid

def daily(ds, start, end):
    """Hourly -> daily means on the 0.25 deg grid. Speed is averaged from hourly speeds (not from mean u/v)."""
    h = ds[[U, V]].sel(time=slice(start, end)).load()
    if h.sizes["time"] == 0:
        return None
    h["wind_speed"] = np.hypot(h[U], h[V])
    d = h.resample(time="1D").mean()
    if not np.array_equal(d.latitude.values, GRID["latitude"].values) or \
       not np.array_equal(d.longitude.values, GRID["longitude"].values):
        d = d.interp(**GRID)
    return d.rename({U: "wind_east", V: "wind_north"}).astype("float32")

# %% SEAM CHECK - old and new overlap Nov 2007 - Oct 2009; a big offset would look like a fake change point
a = daily(old, "2008-01-01", "2008-12-31")
b = daily(new, "2008-01-01", "2008-12-31")
for v in ["wind_east", "wind_north", "wind_speed"]:
    d = float((b[v] - a[v]).mean())
    r = float(xr.corr(a[v].mean(["latitude", "longitude"]), b[v].mean(["latitude", "longitude"])))
    print(f"{v:>11}: mean diff (new - old) {d:+.2f} m/s   daily correlation {r:.3f}")

# %% DOWNLOAD (resumable: skips years already in the store)
done = set()
if fs.exists(OUT):
    done = set(pd.to_datetime(xr.open_zarr(OUT, storage_options=STORAGE).time.values).year)
    print("Already have:", sorted(done))

this_year = pd.Timestamp.today().year
for year in range(1994, this_year + 1):
    if year in done and year != this_year:
        continue
    t0 = time.time()
    y0, y1 = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31 23:59")
    parts = []
    if y0 < SWITCH:
        parts.append(daily(old, y0, min(y1, SWITCH - pd.Timedelta(hours=1))))
    if y1 >= SWITCH and y0 <= NEW_END:
        parts.append(daily(new, max(y0, SWITCH), min(y1, NEW_END + pd.Timedelta(hours=23))))
    if y1 > NEW_END:
        parts.append(daily(nrt, max(y0, NEW_END + pd.Timedelta(days=1)), y1))
    parts = [p for p in parts if p is not None]
    if not parts:
        continue
    ds = xr.concat(parts, dim="time")
    ds = ds.sel(time=~ds.get_index("time").duplicated())
    for v in ds.variables:
        ds[v].encoding = {}
    ds = ds.chunk({"time": -1, "latitude": -1, "longitude": -1})

    if year in done:                                   # refresh the partial current year
        keep = xr.open_zarr(OUT, storage_options=STORAGE)
        keep = keep.sel(time=keep.time.dt.year < year).load()
        xr.concat([keep, ds], dim="time").chunk({"time": 365}).to_zarr(
            OUT, mode="w", consolidated=True, storage_options=STORAGE)
    elif not fs.exists(OUT):
        ds.to_zarr(OUT, mode="w", consolidated=True, storage_options=STORAGE)
    else:
        ds.to_zarr(OUT, append_dim="time", consolidated=True, storage_options=STORAGE)
    print(f"{year}: {ds.sizes['time']} days in {(time.time() - t0) / 60:.1f} min")

# %% VERIFY
chk = xr.open_zarr(OUT, storage_options=STORAGE)
print(chk)
t = pd.to_datetime(chk.time.values)
print("Steps:", pd.Series(np.diff(t)).value_counts().head().to_dict())
print("Days:", len(t), t[0].date(), "->", t[-1].date(),
      "| missing:", len(pd.date_range(t[0], t[-1], freq="D").difference(t)))
print("Mean wind speed (m/s):", round(float(chk.wind_speed.mean()), 2), "(expect roughly 6-9 offshore)")
