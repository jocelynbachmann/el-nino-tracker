# Daily ARMOR3D (surface) for the NE Pacific, streamed from Copernicus Marine into our S3 folder.
# Run on the hub, cell by cell. Nothing is downloaded to a laptop.
#
# One-time setup in a hub TERMINAL (not a notebook, so the password never lands in a file):
#   pip install copernicusmarine
#   copernicusmarine login          # free Copernicus Marine account; saves creds in ~/.copernicusmarine

# %% SETTINGS
import copernicusmarine as cm
import numpy as np
import pandas as pd
import xarray as xr

START = "1993-01-01"   # 1993 keeps the 1997-98 El Nino. MLD before ~2002 is less reliable (pre-Argo): flag it, don't drop it.
LAT = (46, 52)
LON = (-134, -120)
VARS = ["to", "so", "zo", "mlotst", "ugo", "vgo"]   # temperature, salinity, sea level, MLD, currents E/N

MY = "cmems_obs-mob_glo_phy_my_0.125deg_P1D-m"    # reprocessed, 1993-2024
NRT = "cmems_obs-mob_glo_phy_nrt_0.125deg_P1D-m"  # near-real-time, Sep 2024 -> now
MY_END = "2024-12-31"                             # use MY wherever it exists, NRT after

OUT = (f"s3://ocean-hackathon-2026/teams/team-elnino/processed/"
       f"armor3d_daily_surface_{START[:4]}-present_46-52N_134-120W.zarr")
# Hub: key is in env vars -> {}. Laptop: key is in the "hack-team" profile (server/setup_aws.py).
import os
STORAGE = ({} if os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE")
           else {"profile": os.environ.get("AWS_PROFILE", "hack-team")})

# %% OPEN (lazy - nothing is pulled yet)
def open_part(dataset_id, start, end=None):
    ds = cm.open_dataset(
        dataset_id=dataset_id, variables=VARS,
        minimum_latitude=LAT[0], maximum_latitude=LAT[1],
        minimum_longitude=LON[0], maximum_longitude=LON[1],
        minimum_depth=0, maximum_depth=1,          # surface level only
        start_datetime=start, end_datetime=end,
    )
    if "depth" in ds.dims:
        ds = ds.isel(depth=0, drop=True)
    ds["time"] = pd.to_datetime(ds.time.values).floor("D")
    return ds

my = open_part(MY, START, MY_END)
nrt = open_part(NRT, "2024-09-01")
print(my)
print("MY :", my.time.values[0], "->", my.time.values[-1])
print("NRT:", nrt.time.values[0], "->", nrt.time.values[-1])

# %% SEAM CHECK - MY and NRT overlap Sep-Dec 2024; they should roughly agree
ov = slice(str(nrt.time.values[0])[:10], MY_END)
for v in VARS:
    a = my[v].sel(time=ov).mean().compute().item()
    b = nrt[v].sel(time=ov).mean().compute().item()
    print(f"{v:>7}: MY {a:9.3f}   NRT {b:9.3f}   diff {b - a:+.3f}")

# %% STITCH + WRITE (streams a year at a time; takes a while)
nrt_after = nrt.sel(time=slice(pd.Timestamp(MY_END) + pd.Timedelta(days=1), None))
ds = xr.concat([my, nrt_after], dim="time").astype("float32")
ds = ds.chunk({"time": 365, "latitude": -1, "longitude": -1})
for v in ds.variables:
    ds[v].encoding = {}
ds.attrs.update(source=f"{MY} (to {MY_END}) + {NRT} (after)", region="46-52N, 134-120W, surface")
print(f"Size: {ds.nbytes / 1e9:.2f} GB")
ds.to_zarr(OUT, mode="w", consolidated=True, storage_options=STORAGE)
print("Saved:", OUT)

# %% VERIFY - really daily, with real day-to-day variation?
chk = xr.open_zarr(OUT, storage_options=STORAGE)
t = pd.to_datetime(chk.time.values)
print("Steps:", pd.Series(np.diff(t)).value_counts().head().to_dict())
print("Days:", len(t), t[0].date(), "->", t[-1].date(),
      "| missing:", len(pd.date_range(t[0], t[-1], freq="D").difference(t)))
for v in chk.data_vars:
    x = chk[v].isel(time=slice(0, 120)).mean(["latitude", "longitude"]).values
    print(f"{v:>7}: {np.mean(np.isclose(np.diff(x), 0)):.0%} of day-to-day changes are zero (want ~0%)")
