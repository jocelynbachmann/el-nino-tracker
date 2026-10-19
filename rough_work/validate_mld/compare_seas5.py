"""Compare ECMWF SEAS5's mixed layer depth forecast (Oct 2026 start, 51 runs) with our forecast
and with ARMOR3D, on the SAME 1-degree ocean cells.

    cd validate_mld
    pip install h5netcdf          # once, so xarray can read the SEAS5 .nc files
    python compare_seas5.py d397c0727b1f874927b577e0f835427b.zip

Important: SEAS5 is another model's forecast, not a measurement. Nothing has measured
Oct 2026 - Mar 2027 yet, so this is a model-vs-model comparison (a sanity check / benchmark),
not validation. Real validation = compare forecasts made in the past with what ARMOR3D
later observed.

Also: SEAS5's MLD uses a 0.01 kg/m3 density threshold ("mlotstheta001"), ARMOR3D a looser one,
so SEAS5 tends to be somewhat shallower for the same ocean. Compare patterns and changes, and
treat absolute differences of a few metres with care.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from mld_table import ENS_NAMES, LAT_NAMES, LON_NAMES, extract, pick, valid_months

PROFILE = "hack-team"
OUT_DIR = "ocean-hackathon-2026/teams/team-elnino/processed"
opts = {"profile": PROFILE}


def load_seas5(zip_path):
    """All ensemble members -> DataArray (member, month, lat, lon) in metres."""
    members, months = [], None
    for f in sorted(p for p in extract(zip_path) if p.endswith(".nc")):
        ds = xr.open_dataset(f)
        da = ds[list(ds.data_vars)[0]]
        lat, lon = pick(da.dims, LAT_NAMES), pick(da.dims, LON_NAMES)
        if (da[lon] > 180).any():
            da = da.assign_coords({lon: ((da[lon] + 180) % 360) - 180})
        ens = pick(da.dims, ENS_NAMES)
        if ens:
            da = da.isel({ens: 0}, drop=True)
        other = [d for d in da.dims if d not in (lat, lon)]
        lead = next((d for d in ("forecastMonth", "leadtime_month", "forecast_period") if d in other),
                    max(other, key=lambda d: da.sizes[d]))
        # Label lead months from the forecast START date in the file name (e.g. "_S2026100100" =
        # started 1 Oct 2026): lead 1 = the start month itself. The files' own time stamps mark
        # the END of each monthly average, which would shift every label one month late.
        start = re.search(r"_S(\d{8})", os.path.basename(f))
        if start:
            t0 = pd.Timestamp(start.group(1))
            months = [(t0 + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(da.sizes[lead])]
        else:
            months = valid_months(ds, lead)
        if not members:
            print(f"lead dimension '{lead}' values: {np.atleast_1d(da[lead].values)[:8]}"
                  f" -> labelled {months}")
        for d in [d for d in da.dims if d not in (lat, lon, lead)]:
            da = da.isel({d: 0}, drop=True)
        da = da.sortby(lead)
        da = (da.rename({lat: "latitude", lon: "longitude", lead: "month"})
                .assign_coords(month=months).sortby(["latitude", "longitude"]))
        members.append(da.load())
    s5 = xr.concat(members, dim="member")
    print(f"SEAS5: {s5.sizes['member']} runs, months {list(s5.month.values)}, "
          f"{s5.sizes['latitude']} x {s5.sizes['longitude']} cells, units: {s5.attrs.get('units', '?')}")
    return s5


def armor_on_seas5_grid(da, s5):
    """Average ARMOR3D's 1/8-degree cells inside each 1-degree SEAS5 cell."""
    out = []
    for la in s5.latitude.values:
        row = []
        for lo in s5.longitude.values:
            box = da.sel(latitude=slice(la - 0.5, la + 0.5), longitude=slice(lo - 0.5, lo + 0.5))
            row.append(box.mean(["latitude", "longitude"]))
        out.append(xr.concat(row, dim=pd.Index(s5.longitude.values, name="longitude")))
    return xr.concat(out, dim=pd.Index(s5.latitude.values, name="latitude"))


def main(zip_path):
    s5 = load_seas5(zip_path)
    s5_mean = s5.mean("member")

    ocean = xr.open_zarr(f"s3://{OUT_DIR}/ocean_monthly.zarr", storage_options=opts)
    clim = xr.open_zarr(f"s3://{OUT_DIR}/climatology.zarr", storage_options=opts)
    cut = dict(latitude=slice(45.9, 52.1), longitude=slice(-134.1, -119.9))
    clim_1deg = armor_on_seas5_grid(clim.mlotst.sel(**cut).load(), s5)          # (month 1-12, lat, lon)
    obs_1deg = armor_on_seas5_grid(ocean.mlotst.sel(**cut).load(), s5)          # (time, lat, lon)

    # Only cells that are ocean in BOTH datasets, so averages cover the same water
    mask = (s5_mean.isel(month=0, drop=True).notnull()
            & clim_1deg.isel(month=0, drop=True).notnull())
    print(f"cells compared: {int(mask.sum())} (ocean in both)")

    fs = s3fs.S3FileSystem(profile=PROFILE)
    ours = {}
    if fs.exists(f"{OUT_DIR}/forecast_mld.json"):
        with fs.open(f"{OUT_DIR}/forecast_mld.json") as f:
            fc = json.load(f)
        ours = {m["month"]: m for m in fc["months"]}
        print(f"our forecast: {fc.get('model')} issued {fc.get('issued')}")

    def area_mean(da):
        return float(da.where(mask).mean(["latitude", "longitude"]))

    rows = []
    for m in s5.month.values:
        cal = int(m[5:7])
        runs = s5.sel(month=m).where(mask).mean(["latitude", "longitude"]).values   # one value per run
        recent = [area_mean(obs_1deg.sel(time=f"{y}-{m[5:7]}").isel(time=0))
                  for y in range(int(m[:4]) - 3, int(m[:4]))
                  if f"{y}-{m[5:7]}" in pd.DatetimeIndex(obs_1deg.time.values).strftime("%Y-%m")]
        o = ours.get(m, {})
        rows.append({
            "month": m,
            "SEAS5 mean": round(float(np.mean(runs)), 1),
            "SEAS5 10-90%": f"{np.percentile(runs, 10):.1f}-{np.percentile(runs, 90):.1f}",
            "ARMOR normal (same cells)": round(area_mean(clim_1deg.sel(month=cal)), 1),
            "ARMOR last 3 yrs, same month": round(float(np.mean(recent)), 1) if recent else None,
            "ours (forecast_mld.json)": o.get("mld"),
            "ours 80% band": f"{o['lower']:.1f}-{o['upper']:.1f}" if o else None,
            "ours source": o.get("source"),
        })
    table = pd.DataFrame(rows)
    print("\nRegional average mixed layer depth (m). SEAS5 and ARMOR columns use the same 1-degree cells;")
    print("'ours' is the regional average from forecast_mld.json (1/8-degree cells, same region).\n")
    print(table.to_string(index=False))

    # Treat SEAS5 as the reference ("truth") and score our forecast against it
    scored = table.dropna(subset=["ours (forecast_mld.json)"]).copy()
    if len(scored):
        lo_hi = scored["ours 80% band"].str.split("-", expand=True).astype(float)
        scored["ours - SEAS5 (m)"] = (scored["ours (forecast_mld.json)"] - scored["SEAS5 mean"]).round(1)
        scored["SEAS5 inside our band?"] = ((scored["SEAS5 mean"] >= lo_hi[0]) &
                                            (scored["SEAS5 mean"] <= lo_hi[1])).map({True: "yes", False: "no"})
        print("\nOur forecast scored against SEAS5 (SEAS5 treated as correct):")
        print(scored[["month", "SEAS5 mean", "ours (forecast_mld.json)", "ours - SEAS5 (m)",
                      "SEAS5 inside our band?", "ours source"]].to_string(index=False))
        d = scored["ours - SEAS5 (m)"]
        print(f"  average difference {d.mean():+.1f} m | typical size of difference (RMS) "
              f"{np.sqrt((d ** 2).mean()):.1f} m | SEAS5 inside our band in "
              f"{(scored['SEAS5 inside our band?'] == 'yes').sum()} of {len(scored)} months")
        print("  Note: 'ours' covers 1/8-degree cells and SEAS5 1-degree cells over the same region;")
        print("  differences of 1-2 m can come from that alone.")

    print("\nHow SEAS5 compares with the ARMOR normal, per month (SEAS5 minus normal, m):")
    print("  " + "  ".join(f"{r['month']}: {r['SEAS5 mean'] - r['ARMOR normal (same cells)']:+.1f}"
                           for r in rows))

    print("\nDecember 2026, cell by cell (SEAS5 mean / ARMOR normal), north at top:")
    dec = [m for m in s5.month.values if m.endswith("-12")]
    if dec:
        a = s5_mean.sel(month=dec[0]).where(mask).to_pandas().sort_index(ascending=False).round(0)
        b = clim_1deg.sel(month=12).where(mask).to_pandas().sort_index(ascending=False).round(0)
        print((a.astype("Int64").astype(str) + "/" + b.astype("Int64").astype(str))
              .replace("<NA>/<NA>", "  -  ").to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "d397c0727b1f874927b577e0f835427b.zip")
