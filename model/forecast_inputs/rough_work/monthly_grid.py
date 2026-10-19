"""
monthly_grid.py - one CSV: 6 forecast months x every grid cell, ensemble-mean values.

Needs merge_to_csv.py in model/forecast_inputs/ (reuses its loaders). Archived: the app uses seas5_merged_full.csv instead.

Usage (PowerShell, in model/forecast_inputs):
    python monthly_grid.py
    python monthly_grid.py --ocean ocean_2026-10.grib --wind wind_ocean_2026-10.grib --out clean

Output: clean/seas5_monthly_grid.csv

Columns
    month                  forecast month (YYYY-MM)
    lat, lon               cell centre (degrees, lon in -180..180)
    is_ocean               False for land cells (ocean fields are empty there)
    sea_surface_height_m   SEAS5 zos, sea surface height above geoid
    surface_temp_c         SEAS5 sea surface temperature if downloaded, else 2 m air temp (proxy)
    wind_speed_ms          10 m wind speed (mean of member speeds, not speed of the mean wind)
    north_wind_ms          10 m V component, + = blowing toward north
    east_wind_ms           10 m U component, + = blowing toward east
    salinity_psu           SEAS5 sea surface salinity if downloaded, else climatology placeholder
    *_spread               ensemble standard deviation (forecast uncertainty)
    surface_temp_source    "model_sst" or "proxy_air_temp_2m"
    salinity_source        "model_sss" or "approx_climatology"
"""

import argparse
import os

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # merge_to_csv.py lives in model/forecast_inputs/
from merge_to_csv import KEY, clean, load_ocean, load_wind

# Names SEAS5 / C3S use for each ocean field
SST_NAMES = ["tos", "sst", "thetao", "sea_surface_temperature"]
SSS_NAMES = ["sos", "so", "sss", "sea_surface_salinity"]
SSH_NAMES = ["zos"]


def pick(cols, names):
    for n in names:
        if n in cols:
            return n
    return None


def salinity_climatology(lat, lon):
    """Rough NE Pacific surface salinity (PSU) for 46-52N, 134-120W.

    Open ocean here is about 32.4-32.6 PSU, freshening toward the coast from
    river runoff (Columbia, Fraser) and coastal rain. This is a placeholder
    gradient, not data: download `sea_surface_salinity` for real values.
    """
    dist_to_coast_deg = np.clip(lon - (-133.5), 0, None)  # 0 offshore edge -> ~10 near coast
    return 32.6 - 0.08 * dist_to_coast_deg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ocean", default="ocean_2026-10.grib")
    p.add_argument("--wind", default="wind_ocean_2026-10.grib")
    p.add_argument("--out", default="clean")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ocean = clean(load_ocean(a.ocean))
    wind = clean(load_wind(a.wind))
    if ocean["init_date"].isna().all():
        ocean["init_date"] = wind["init_date"].iloc[0]

    df = wind.merge(ocean, on=KEY, how="inner")
    if df.empty:
        raise SystemExit("Nothing matched between ocean and wind data.")

    ssh = pick(df.columns, SSH_NAMES)
    sst = pick(df.columns, SST_NAMES)
    sss = pick(df.columns, SSS_NAMES)
    ocean_vars = [c for c in ocean.columns if c not in KEY]
    print(f"\nOcean variables found: {ocean_vars}")
    print(f"  sea surface height : {ssh or 'MISSING'}")
    print(f"  surface temperature: {sst or 'missing -> 2 m air temperature proxy'}")
    print(f"  salinity           : {sss or 'missing -> climatology placeholder'}")

    out = pd.DataFrame({k: df[k] for k in KEY})
    out["is_ocean"] = df[ocean_vars].notna().any(axis=1)
    out["sea_surface_height_m"] = df[ssh] if ssh else np.nan

    if sst:
        t = df[sst]
        out["surface_temp_c"] = np.where(t > 200, t - 273.15, t)  # K or degC
        temp_src = "model_sst"
    else:
        out["surface_temp_c"] = df["t2m"] - 273.15
        temp_src = "proxy_air_temp_2m"

    out["wind_speed_ms"] = df["si10"] if "si10" in df else np.hypot(df["u10"], df["v10"])
    out["north_wind_ms"] = df["v10"]
    out["east_wind_ms"] = df["u10"]

    if sss:
        out["salinity_psu"] = df[sss]
        sal_src = "model_sss"
    else:
        out["salinity_psu"] = np.where(out["is_ocean"],
                                       salinity_climatology(out["lat"], out["lon"]), np.nan)
        sal_src = "approx_climatology"

    # ensemble mean + spread per month x cell
    vals = ["sea_surface_height_m", "surface_temp_c", "wind_speed_ms",
            "north_wind_ms", "east_wind_ms", "salinity_psu"]
    g = out.groupby(["month", "lat", "lon"])
    res = g[vals].mean()
    spread = g[vals].std().add_suffix("_spread")
    res = pd.concat([res, spread], axis=1)
    res.insert(0, "is_ocean", g["is_ocean"].any())
    res = res.reset_index()

    # land cells: ocean-only fields stay empty
    for c in ["sea_surface_height_m", "salinity_psu",
              "sea_surface_height_m_spread", "salinity_psu_spread"]:
        res.loc[~res["is_ocean"], c] = np.nan
    if temp_src == "model_sst":
        res.loc[~res["is_ocean"], ["surface_temp_c", "surface_temp_c_spread"]] = np.nan
    if sal_src == "approx_climatology":
        res["salinity_psu_spread"] = np.nan  # a placeholder has no ensemble spread

    res["surface_temp_source"] = temp_src
    res["salinity_source"] = sal_src
    res["month"] = pd.to_datetime(res["month"]).dt.strftime("%Y-%m")
    res = res.sort_values(["month", "lat", "lon"], ascending=[True, False, True])

    f = os.path.join(a.out, "seas5_monthly_grid.csv")
    res.to_csv(f, index=False, float_format="%.4f")

    print(f"\nWrote {f}: {len(res)} rows "
          f"({res['month'].nunique()} months x {len(res) // res['month'].nunique()} cells, "
          f"{int(res['is_ocean'].sum() / res['month'].nunique())} ocean cells per month)")
    print("\nMonthly averages over ocean cells:")
    sea = res[res["is_ocean"]]
    print(sea.groupby("month")[vals].mean().round(3).to_string())


if __name__ == "__main__":
    main()