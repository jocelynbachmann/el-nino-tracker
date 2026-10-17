"""
Print monthly mixed-layer-depth tables from a CDS 'seasonal-monthly-ocean' download.

Usage:
    python mld_table.py d397c0727b1f874927b577e0f835427b.zip            # all lead months
    python mld_table.py d397c0727b1f874927b577e0f835427b.zip --inspect  # just show file structure
    python mld_table.py d397c0727b1f874927b577e0f835427b.zip --month 2026-12

Needs: pip install xarray cfgrib eccodes pandas numpy
"""
import argparse
import glob
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon")
ENS_NAMES = ("member", "number", "realization")


def extract(path):
    """Return a list of GRIB/NetCDF files, unzipping if needed."""
    if zipfile.is_zipfile(path):
        out = tempfile.mkdtemp(prefix="mld_")
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
        files = [f for f in glob.glob(os.path.join(out, "**", "*"), recursive=True)
                 if os.path.isfile(f) and not f.endswith(".idx")]
    else:
        files = [path]  # CDS sometimes names a plain GRIB file .zip
    return files


def open_any(f):
    if f.endswith((".nc", ".nc4")):
        return xr.open_dataset(f)
    # indexpath="" stops cfgrib writing .idx files next to the data
    return xr.open_dataset(f, engine="cfgrib", backend_kwargs={"indexpath": ""})


def pick(names, candidates):
    for c in candidates:
        if c in names:
            return c
    return None


def inspect(ds, f):
    print(f"\n##### {os.path.basename(f)}")
    print(ds)
    for v in ds.data_vars:
        da = ds[v]
        print(f"\nvariable '{v}': dims={da.dims} shape={da.shape}")
        for k in ("long_name", "units", "GRIB_shortName", "GRIB_paramId"):
            if k in da.attrs:
                print(f"  {k}: {da.attrs[k]}")
    for c in ds.coords:
        vals = np.atleast_1d(ds[c].values)
        print(f"coord {c}: {vals[:8]}{' ...' if vals.size > 8 else ''}")


def valid_months(ds, lead_dim):
    """Label each lead step with its valid month (YYYY-MM)."""
    if "valid_time" in ds.coords:
        vt = pd.to_datetime(np.atleast_1d(ds["valid_time"].values).ravel())
        if len(vt) == ds.sizes[lead_dim]:
            return [t.strftime("%Y-%m") for t in vt]
    ref = pick(ds.coords, ("time", "forecast_reference_time", "indexing_time"))
    start = pd.Timestamp(np.atleast_1d(ds[ref].values)[0]) if ref else None
    leads = np.atleast_1d(ds[lead_dim].values)
    if start is None:
        print(f"  [warning] no start-date coordinate found; showing raw '{lead_dim}' values")
        return [str(x) for x in leads]
    if lead_dim == "forecastMonth":
        # CDS convention: leadtime_month 1 = the start month itself
        return [(start + pd.DateOffset(months=int(m) - 1)).strftime("%Y-%m") for m in leads]
    if np.issubdtype(leads.dtype, np.timedelta64):
        return [(start + pd.Timedelta(s)).strftime("%Y-%m") for s in leads]
    # plain numbers (e.g. forecast_period / leadtime): hours since the start date
    return [(start + pd.Timedelta(hours=float(h))).strftime("%Y-%m") for h in leads]


def print_table(da2d, label, lat, lon):
    df = da2d.to_pandas()  # rows = lat, cols = lon
    if df.index.name != lat:
        df = df.T
    df = df.sort_index(ascending=False)          # north at top
    df = df.reindex(sorted(df.columns), axis=1)  # west to east
    df.index.name, df.columns.name = "lat", "lon"
    df.index = [f"{x:.1f}" for x in df.index]
    df.columns = [f"{x:.1f}" for x in df.columns]
    df.index.name, df.columns.name = "lat", "lon"

    shown = df.apply(lambda col: col.map(lambda x: "land" if pd.isna(x) else f"{x:.1f}"))
    print(f"=== MLD forecast {label} (metres) ===")
    print(shown.to_string(justify="right"))

    vals = df.to_numpy(dtype=float)
    ocean = vals[~np.isnan(vals)]
    # Simple average over ocean cells, matching the example output.
    print(f"Area average: {ocean.mean():.1f} m  ({ocean.size} ocean cells)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--inspect", action="store_true", help="print file structure and stop")
    ap.add_argument("--month", help="only this valid month, e.g. 2026-12")
    a = ap.parse_args()

    files = sorted(extract(a.path))
    if a.inspect:
        print(f"{len(files)} file(s) in archive")
        inspect(open_any(files[0]), files[0])
        return

    # The CDS NetCDF download is one file per ensemble member (..._r01i00p00, _r02..., etc.).
    # Stack them along a 'member' dimension so they can be averaged together.
    datasets = [open_any(f) for f in files]
    ds = datasets[0]
    var = list(ds.data_vars)[0]
    if len(datasets) > 1:
        da = xr.concat([d[var] for d in datasets], dim="member", coords="minimal",
                       compat="override", join="override")
    else:
        da = ds[var]

    lat, lon = pick(da.dims, LAT_NAMES), pick(da.dims, LON_NAMES)
    ens = pick(da.dims, ENS_NAMES)
    n_members = da.sizes[ens] if ens else 1
    print(f"Variable '{var}': ensemble mean of {n_members} member(s)\n")
    if ens:
        da = da.mean(ens, skipna=True)
    # convert 0-360 longitudes to -180..180 if needed
    if (da[lon] > 180).any():
        da = da.assign_coords({lon: ((da[lon] + 180) % 360) - 180})

    other = [d for d in da.dims if d not in (lat, lon)]
    if not other:
        print_table(da, "(single field)", lat, lon)
        return

    lead = "forecastMonth" if "forecastMonth" in other else other[0]
    labels = valid_months(ds, lead)
    for d in other:
        if d != lead:
            da = da.isel({d: 0})
    for i, label in enumerate(labels):
        if a.month and label != a.month:
            continue
        print_table(da.isel({lead: i}), label, lat, lon)

# python mld_table.py d397c0727b1f874927b577e0f835427b.zip --month 2026-12

if __name__ == "__main__":
    main()