"""
merge_to_csv.py - merge the SEAS5 ocean (ZIP of NetCDF) and wind (GRIB) forecasts
into clean CSV files.

Usage (PowerShell, in the `future` folder):
    pip install xarray cfgrib eccodes netcdf4 pandas numpy
    python merge_to_csv.py
    # or with explicit paths:
    python merge_to_csv.py --ocean ocean_2026-10.grib --wind wind_ocean_2026-10.grib --out clean

Writes to ./clean/:
    seas5_merged_full.csv      every member x month x grid cell (main table)
    seas5_ensemble_stats.csv   ensemble mean / std / p10 / p90 per month x grid cell
    seas5_area_mean.csv        ocean-cell average per member x month (one row = one forecast)
    seas5_area_summary.csv     ensemble summary of the area mean per month (for the tracker)
"""

import argparse
import os
import re
import tempfile
import warnings
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

KEY = ["init_date", "month", "member", "lat", "lon"]


# ---------------------------------------------------------------------------
# Ocean: ZIP (or folder / single file) of per-member NetCDFs
# ---------------------------------------------------------------------------
def ocean_files(path):
    """Return a list of .nc paths, unzipping if needed."""
    with open(path, "rb") as f:
        head = f.read(4)
    if head == b"PK\x03\x04":
        out = tempfile.mkdtemp(prefix="seas5_ocean_")
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
        files = [os.path.join(r, n) for r, _, ns in os.walk(out) for n in ns if n.endswith(".nc")]
    elif os.path.isdir(path):
        files = [os.path.join(path, n) for n in os.listdir(path) if n.endswith(".nc")]
    else:
        files = [path]
    if not files:
        raise SystemExit(f"No .nc files found in {path}")
    return sorted(files)


def load_ocean(path):
    files = ocean_files(path)
    print(f"Ocean: {len(files)} NetCDF file(s)")
    frames = []
    for fp in files:
        ds = xr.open_dataset(fp)

        # member number: from the realization attr/coord or filename (r40i00p00 -> 40)
        real = ds["realization"].values.item() if "realization" in ds else os.path.basename(fp)
        if isinstance(real, bytes):
            real = real.decode()
        m = re.search(r"r(\d+)i\d+p\d+", str(real)) or re.search(r"_r(\d+)i", os.path.basename(fp))
        member = int(m.group(1))

        # forecast month: start of each averaging period
        if "time_bnds" in ds:
            start = pd.to_datetime(ds["time_bnds"].values[:, 0])
        else:
            start = pd.to_datetime(ds["time"].values)
        months = start.to_period("M").to_timestamp()

        init = pd.to_datetime(ds["reftime"].values.item()) if "reftime" in ds else pd.NaT

        data_vars = [v for v, da in ds.data_vars.items()
                     if "bnds" not in v and da.ndim == 3 and "leadtime" in da.dims]
        for var in data_vars:
            da = ds[var].assign_coords(month=("leadtime", months)).swap_dims({"leadtime": "month"})
            df = da.drop_vars([c for c in da.coords if c not in ("month", "lat", "lon")]) \
                   .to_dataframe(name=var).reset_index()
            df["member"] = member
            df["init_date"] = init
            frames.append(df.set_index(KEY)[[var]])
        ds.close()

    # combine members (concat) and variables (columns)
    ocean = pd.concat(frames).groupby(level=KEY, dropna=False).first()
    ocean = ocean.reset_index()
    ocean["lon"] = np.where(ocean["lon"] > 180, ocean["lon"] - 360, ocean["lon"])
    print(f"  variables: {[c for c in ocean.columns if c not in KEY]}")
    print(f"  members: {ocean['member'].nunique()}  months: {sorted(ocean['month'].dt.strftime('%Y-%m').unique())}")
    return ocean


# ---------------------------------------------------------------------------
# Wind / atmosphere: GRIB
# ---------------------------------------------------------------------------
def load_wind(path):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    init = pd.to_datetime(ds["time"].values.item())
    # monthly mean is labelled by its END (valid_time = 1st of next month) -> step back 1 day
    vt = pd.to_datetime(ds["valid_time"].values) - pd.Timedelta(days=1)
    months = vt.to_period("M").to_timestamp()
    ds = ds.assign_coords(month=("step", months)).swap_dims({"step": "month"})
    ds = ds.drop_vars([c for c in ds.coords if c not in ("number", "month", "latitude", "longitude")])
    df = ds.to_dataframe().reset_index().rename(
        columns={"number": "member", "latitude": "lat", "longitude": "lon"})
    df["init_date"] = init
    print(f"Wind: variables {[c for c in df.columns if c not in KEY]}")
    print(f"  members: {df['member'].nunique()}  months: {sorted(df['month'].dt.strftime('%Y-%m').unique())}")
    return df


# ---------------------------------------------------------------------------
# Clean + merge
# ---------------------------------------------------------------------------
def clean(df):
    df["lat"] = df["lat"].astype(float).round(2)
    df["lon"] = df["lon"].astype(float).round(2)
    df["member"] = df["member"].astype(int)
    df["init_date"] = pd.to_datetime(df["init_date"]).dt.normalize()
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ocean", default="ocean_2026-10.grib")
    p.add_argument("--wind", default="wind_ocean_2026-10.grib")
    p.add_argument("--out", default="clean")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ocean = clean(load_ocean(a.ocean))
    wind = clean(load_wind(a.wind))

    # init dates should match; if one side is missing, take the other's
    if ocean["init_date"].isna().all():
        ocean["init_date"] = wind["init_date"].iloc[0]

    # inner join: only months/cells present in both (drops the ocean's extra April)
    df = wind.merge(ocean, on=KEY, how="inner", validate="one_to_one")
    lost_w, lost_o = len(wind) - len(df), len(ocean) - len(df)
    print(f"\nMerged rows: {len(df):,}  (wind rows not matched: {lost_w:,}, ocean rows not matched: {lost_o:,})")
    if len(df) == 0:
        raise SystemExit("Nothing matched - check lat/lon/month alignment above.")

    # derived columns
    if "t2m" in df:
        df["t2m_c"] = df["t2m"] - 273.15
    if {"u10", "v10"} <= set(df.columns):
        # meteorological direction: where the wind blows FROM, degrees clockwise from north
        df["wind_dir_deg"] = (np.degrees(np.arctan2(-df["u10"], -df["v10"])) % 360)
    ocean_vars = [c for c in ocean.columns if c not in KEY]
    df["is_ocean"] = df[ocean_vars].notna().any(axis=1)

    value_cols = [c for c in df.columns if c not in KEY + ["is_ocean"]]
    df = df[KEY + ["is_ocean"] + value_cols].sort_values(KEY).reset_index(drop=True)
    df["init_date"] = df["init_date"].dt.strftime("%Y-%m-%d")
    df["month"] = df["month"].dt.strftime("%Y-%m")

    # 1. full table
    f1 = os.path.join(a.out, "seas5_merged_full.csv")
    df.to_csv(f1, index=False, float_format="%.4f")

    # directions can't be averaged as numbers (mean of 350 and 10 is not 180),
    # so aggregate u/v and recompute direction from the averaged components
    agg_cols = [c for c in value_cols if c != "wind_dir_deg"]

    def add_dir(t, suffix):
        u, v = f"u10_{suffix}", f"v10_{suffix}"
        if u in t and v in t:
            t[f"wind_dir_deg_{suffix}"] = np.degrees(np.arctan2(-t[u], -t[v])) % 360
        return t

    # 2. ensemble stats per cell x month
    g = df.groupby(["init_date", "month", "lat", "lon", "is_ocean"])[agg_cols]
    stats = pd.concat({
        "mean": g.mean(), "std": g.std(),
        "p10": g.quantile(0.10), "p90": g.quantile(0.90),
    }, axis=1)
    stats.columns = [f"{v}_{s}" for s, v in stats.columns]
    stats = add_dir(stats.reset_index(), "mean")
    f2 = os.path.join(a.out, "seas5_ensemble_stats.csv")
    stats.to_csv(f2, index=False, float_format="%.4f")

    # 3. ocean-only area mean per member x month
    sea = df[df["is_ocean"]]
    area = sea.groupby(["init_date", "month", "member"])[agg_cols].mean().reset_index()
    if {"u10", "v10"} <= set(area.columns):
        area["wind_dir_deg"] = np.degrees(np.arctan2(-area["u10"], -area["v10"])) % 360
    area["n_cells"] = sea.groupby(["init_date", "month", "member"]).size().values
    f3 = os.path.join(a.out, "seas5_area_mean.csv")
    area.to_csv(f3, index=False, float_format="%.4f")

    # 4. ensemble summary of the area mean per month
    ga = area.groupby(["init_date", "month"])[agg_cols]
    summ = pd.concat({"mean": ga.mean(), "std": ga.std(),
                      "min": ga.min(), "max": ga.max()}, axis=1)
    summ.columns = [f"{v}_{s}" for s, v in summ.columns]
    summ = add_dir(summ.reset_index(), "mean")
    f4 = os.path.join(a.out, "seas5_area_summary.csv")
    summ.to_csv(f4, index=False, float_format="%.4f")

    print("\nWrote:")
    for f, d in [(f1, df), (f2, stats), (f3, area), (f4, summ)]:
        print(f"  {f:40s} {len(d):>7,} rows x {d.shape[1]} cols")
    print("\nColumns in seas5_merged_full.csv:\n  " + ", ".join(df.columns))
    print("\nPreview (area summary):")
    show = ["month"] + [c for c in summ.columns if c.endswith("_mean")]
    print(summ[show].to_string(index=False))


if __name__ == "__main__":
    main()