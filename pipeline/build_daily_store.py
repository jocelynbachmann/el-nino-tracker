# Make the daily data map-friendly for the server. Run once (and again whenever the daily file is updated).
#
# Reads the same daily ARMOR3D file as analyze_events.py and writes, next to the other processed outputs:
#   ocean_daily.zarr     every variable, one chunk per ~month of days with the whole map in it,
#                        so the server reads one small chunk per map request
#   daily_normal.zarr    each point's "normal" as a handful of coefficients (seasonal cycle + trend), fitted
#                        exactly as in analyze_events.py. The server rebuilds the normal for any day from these,
#                        so anomaly maps on the site match the El Nino analysis.
#   meta_daily.json      days covered, map bounds, labels, units and colour ranges per variable
#
# Best run on the hub (it uploads ~1 GB). On a laptop it uses the hack-team AWS profile.
#   python build_daily_store.py

# %% SETTINGS
import json, time
import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from analyze_events import (BASE_YEARS, HARMONICS, NORMAL_ORIGIN, OUT_DIR, STORAGE, TREND_ORIGIN_YEARS,
                            VARS as ANALYSIS_VARS, fit_normal, load_oni, normal_design, normal_terms,
                            oni_on_days, open_daily)

PROCESSED = OUT_DIR.rsplit("/analysis", 1)[0]      # .../team-elnino/processed
LABELS = {  # code -> (label, units, colormap hint), same as preprocess.ipynb
    "to": ("Sea surface temperature", "°C", "thermal"),
    "so": ("Sea surface salinity", "PSU", "haline"),
    "zo": ("Sea surface height", "m", "balance"),
    "mlotst": ("Mixed layer depth", "m", "deep"),
    "ugo": ("Geostrophic current (east)", "m/s", "balance"),
    "vgo": ("Geostrophic current (north)", "m/s", "balance"),
}
DETREND = {c: ANALYSIS_VARS[c]["detrend"] if c in ANALYSIS_VARS else False for c in LABELS}
DAYS_PER_CHUNK = 32


def robust(a, sym=False):
    lo, hi = np.nanpercentile(a, [2, 98])
    if sym:
        m = max(abs(lo), abs(hi))
        return [round(-float(m), 4), round(float(m), 4)]
    return [round(float(lo), 4), round(float(hi), 4)]


def main():
    on_s3 = PROCESSED.startswith("s3://")
    fs = s3fs.S3FileSystem(**STORAGE) if on_s3 else None
    opts = {"storage_options": STORAGE} if on_s3 else {}
    t0 = time.time()

    ds = open_daily(fs)
    t = pd.DatetimeIndex(ds.time.values)
    codes = [c for c in LABELS if c in ds.data_vars]
    print(f"Daily file: {len(t)} days {t[0].date()} -> {t[-1].date()}, {ds.sizes['latitude']} x "
          f"{ds.sizes['longitude']}, variables {codes}")
    oni_d = oni_on_days(load_oni(fs), t)

    store = f"{PROCESSED}/ocean_daily.zarr"
    terms = normal_terms(HARMONICS)
    normal = xr.Dataset(coords=dict(term=terms, latitude=ds.latitude, longitude=ds.longitude))
    meta_vars = {}
    for k, code in enumerate(codes):
        t1 = time.time()
        da = ds[code].astype("float32").transpose("time", "latitude", "longitude").load()
        Y = da.values.reshape(len(t), -1)
        ocean = np.isfinite(Y).mean(0) > 0.9

        # 1. daily values, chunked for map reads
        out = da.to_dataset(name=code).chunk({"time": DAYS_PER_CHUNK, "latitude": -1, "longitude": -1})
        for v in out.variables:
            out[v].encoding = {}
        out.to_zarr(store, mode="w" if k == 0 else "a", consolidated=True, zarr_format=2, **opts)

        # 2. normal coefficients (same fit as the analysis)
        coef = np.full((len(terms), Y.shape[1]), np.nan, np.float32)
        coef[:, ocean] = fit_normal(Y[:, ocean], t, DETREND[code], oni_d)
        normal[code] = (("term", "latitude", "longitude"), coef.reshape(len(terms), *da.shape[1:]))

        # 3. colour ranges from every 7th day
        sample = slice(None, None, 7)
        anom = Y[sample] - normal_design(t[sample]) @ np.nan_to_num(coef)
        anom[:, ~ocean] = np.nan
        label, units, cmap = LABELS[code]
        meta_vars[code] = dict(label=label, units=units, cmap=cmap, range=robust(Y[sample]),
                               anomaly_range=robust(anom, sym=True), detrended=DETREND[code])
        print(f"  {code:>7}: written + normal fitted  [{time.time() - t1:.0f}s]")
        del da, Y, anom

    normal.attrs.update(
        description="Each point's normal = design(day) @ coefficients, design = [1, cos(2pi k y), sin(2pi k y) "
                    f"for k=1..{HARMONICS}, y - {TREND_ORIGIN_YEARS}], y = (day - {NORMAL_ORIGIN.date()}) / "
                    "365.2425 days. Fitted on the base years with the ONI as an extra (unused) regressor; same "
                    "as pipeline/analyze_events.py.",
        harmonics=HARMONICS, base_years=list(BASE_YEARS), origin=str(NORMAL_ORIGIN.date()),
        days_per_year=365.2425, trend_origin_years=TREND_ORIGIN_YEARS)
    for v in normal.variables:
        normal[v].encoding = {}
    normal.to_zarr(f"{PROCESSED}/daily_normal.zarr", mode="w", consolidated=True, zarr_format=2, **opts)

    lat, lon = ds.latitude.values, ds.longitude.values
    dlat, dlon = float(lat[1] - lat[0]), float(lon[1] - lon[0])
    meta = dict(
        first_day=f"{t[0]:%Y-%m-%d}", last_day=f"{t[-1]:%Y-%m-%d}", n_days=len(t),
        bounds=dict(west=float(lon[0] - dlon / 2), east=float(lon[-1] + dlon / 2),
                    south=float(lat[0] - dlat / 2), north=float(lat[-1] + dlat / 2)),
        normal=dict(base_years=list(BASE_YEARS), detrended=[c for c in codes if DETREND[c]]),
        variables=meta_vars)
    path = f"{PROCESSED}/meta_daily.json"
    if on_s3:
        with fs.open(path, "w") as f:
            json.dump(meta, f, indent=1)
    else:
        with open(path, "w") as f:
            json.dump(meta, f, indent=1)

    # sanity check: read one day back
    chk = xr.open_zarr(store, **opts)
    print(f"\nWrote {store} ({chk.sizes['time']} days), {PROCESSED}/daily_normal.zarr, {path}")
    print("Spot check 2015-08-15 temperature mean:",
          round(float(chk["to"].sel(time="2015-08-15").mean()), 2), "°C")
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
