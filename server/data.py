"""Read the processed outputs written by preprocess.ipynb (and the model notebook).

Data is streamed from the team's S3 folder by default:
  s3://ocean-hackathon-2026/teams/team-elnino/processed
using the "hack-team" AWS profile in ~/.aws (run `python setup_aws.py` once per machine).
Nothing is downloaded up front; each request reads only the chunks it needs.

Overrides (environment variables):
  ELNINO_DATA   another location, e.g. a local folder copy of processed/
  AWS_PROFILE   another profile name
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  used instead of a profile if set

Expected files (any missing optional file just disables that feature):
  ocean_monthly.zarr, climatology.zarr, meta.json        required
  enso.json                                              optional - ENSO phase strip
  wind_monthly.zarr, wind_climatology.zarr              optional - wind charts
  forecast_mld.json                                      optional - written by the model notebook
"""
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_DATA = "s3://ocean-hackathon-2026/teams/team-elnino/processed"
DATA_DIR = os.environ.get("ELNINO_DATA", DEFAULT_DATA).rstrip("/")
IS_S3 = DATA_DIR.startswith("s3://")

# The box the MLD model is trained on; region time series and forecasts refer to this.
MODEL_REGION = {"south": 46.0, "north": 52.0, "west": -134.0, "east": -120.0}

# Readable names used in URLs and responses -> variable names inside the data files.
OCEAN_VARIABLES = {
    "temperature": "to",
    "salinity": "so",
    "sea-surface-height": "zo",
    "mixed-layer-depth": "mlotst",
    "current-east": "ugo",
    "current-north": "vgo",
}
WIND_VARIABLES = {  # regional time series only (no maps)
    "wind-speed": ("wind_speed", "Wind speed", "m/s"),
    "wind-north": ("northward_wind", "North wind", "m/s"),
    "wind-east": ("eastward_wind", "East wind", "m/s"),
    "wind-stress": ("wind_stress_magnitude", "Wind stress", "N/m2"),
}
CODE_TO_NAME = {**{c: n for n, c in OCEAN_VARIABLES.items()},
                **{c: n for n, (c, _, _) in WIND_VARIABLES.items()}}


def _path(name):
    return f"{DATA_DIR}/{name}"


def _storage_options():
    """Local folder: none. S3: key env vars if set, otherwise the hack-team profile."""
    if not IS_S3:
        return None
    if os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE"):
        return {}
    return {"profile": os.environ.get("AWS_PROFILE", "hack-team")}


def _exists(name):
    if IS_S3:
        import s3fs
        return s3fs.S3FileSystem(**_storage_options()).exists(_path(name))
    return os.path.exists(_path(name))


def _read_json(name, default=None):
    if not _exists(name):
        return default
    if IS_S3:
        import s3fs
        with s3fs.S3FileSystem(**_storage_options()).open(_path(name)) as f:
            return json.load(f)
    with open(_path(name)) as f:
        return json.load(f)


def _open_zarr(name):
    return xr.open_zarr(_path(name), storage_options=_storage_options())


def _clean(x, nd=3):
    """float -> rounded float, NaN -> None (JSON null)."""
    x = float(x)
    return None if np.isnan(x) else round(x, nd)


def month_key(ts):
    return pd.Timestamp(ts).strftime("%Y-%m")


class DataStore:
    def __init__(self):
        if not _exists("ocean_monthly.zarr"):
            raise FileNotFoundError(
                f"No ocean_monthly.zarr in {DATA_DIR}. Set ELNINO_DATA to the processed folder.")
        self.ocean = _open_zarr("ocean_monthly.zarr")
        if os.environ.get("PRELOAD") == "1":          # ~400 MB; faster point queries
            self.ocean = self.ocean.load()
        self.clim = _open_zarr("climatology.zarr").load()   # 12 months, small
        self.meta = _read_json("meta.json", {})
        self.enso = _read_json("enso.json", [])
        self.wind = _open_zarr("wind_monthly.zarr") if _exists("wind_monthly.zarr") else None
        self.wind_clim = (_open_zarr("wind_climatology.zarr").load()
                          if _exists("wind_climatology.zarr") else None)
        self.reload_forecast()

        self.lats = self.ocean.latitude.values
        self.lons = self.ocean.longitude.values
        self.months = pd.DatetimeIndex(self.ocean.time.values)
        self.month_index = {month_key(t): i for i, t in enumerate(self.months)}

    # ---------- forecast (written by the model notebook) ----------
    def reload_forecast(self):
        """forecast_mld.json format agreed with the model notebook:
        {"issued": "2026-10", "model": "...", "region": {...},
         "months": [{"month": "2026-10", "mld": 42.1, "mld_anomaly": -3.2,
                     "lower": 35.0, "upper": 49.0, "source": "model" | "climatology"}]}"""
        self.forecast = _read_json("forecast_mld.json", None)
        self.forecast_by_month = {m["month"]: m for m in (self.forecast or {}).get("months", [])}

    # ---------- helpers ----------
    def resolve_var(self, name, include_wind=False):
        """Readable name (or file code) -> code in the data files, or None if unknown."""
        code = OCEAN_VARIABLES.get(name, name)
        if code in self.meta.get("variables", {}):
            return code
        if include_wind and self.wind is not None:
            code = WIND_VARIABLES.get(name, (name,))[0]
            if code in self.wind:
                return code
        return None

    def var_meta(self, var):
        return self.meta["variables"][var]

    def nearest_cell(self, lat, lon):
        i = int(np.abs(self.lats - lat).argmin())
        j = int(np.abs(self.lons - lon).argmin())
        return i, j

    def in_bounds(self, lat, lon):
        dlat = float(self.lats[1] - self.lats[0])
        dlon = float(self.lons[1] - self.lons[0])
        return (self.lats[0] - dlat <= lat <= self.lats[-1] + dlat
                and self.lons[0] - dlon <= lon <= self.lons[-1] + dlon)

    # ---------- map grids ----------
    def grid(self, var, month, anomaly=False, model_region_only=False):
        """One month as (lats, lons, 2D values, source). source is 'observed' or 'forecast'.
        model_region_only crops to MODEL_REGION. Returns None if there's no data for that month."""
        cal_month = int(month[5:7])
        clim = self.clim[var].sel(month=cal_month)
        if month in self.month_index:
            da = self.ocean[var].isel(time=self.month_index[month])
            if anomaly:
                da = da - clim
            source = "observed"
        else:
            fc = self.forecast_by_month.get(month)
            if var != "mlotst" or fc is None or fc.get("mld_anomaly") is None:
                return None
            # The model forecasts one regional anomaly: shown as the normal pattern for that
            # month shifted by the forecast anomaly (land stays NaN).
            anom = clim * 0 + fc["mld_anomaly"]
            da = anom if anomaly else clim + anom
            source = "forecast"
        if model_region_only:
            r = MODEL_REGION
            da = da.sel(latitude=slice(r["south"], r["north"]), longitude=slice(r["west"], r["east"]))
        return da.latitude.values, da.longitude.values, da.values, source

    # ---------- time series ----------
    def point_series(self, var, lat, lon):
        i, j = self.nearest_cell(lat, lon)
        vals = self.ocean[var].isel(latitude=i, longitude=j).values
        clim = self.clim[var].isel(latitude=i, longitude=j).values      # 12 values
        anom = vals - clim[self.months.month.to_numpy() - 1]
        series = [{"month": month_key(t), "value": _clean(v), "anomaly": _clean(a)}
                  for t, v, a in zip(self.months, vals, anom)]
        return {"lat": float(self.lats[i]), "lon": float(self.lons[j]),
                "is_ocean": bool(np.isfinite(vals).any()), "series": series}

    @lru_cache(maxsize=16)
    def region_series(self, var):
        """Mean over MODEL_REGION ocean cells, per month (+ anomaly)."""
        r = MODEL_REGION
        sel = dict(latitude=slice(r["south"], r["north"]), longitude=slice(r["west"], r["east"]))
        src, clim_src = (self.ocean, self.clim) if var in self.ocean else (self.wind, self.wind_clim)
        if src is None or var not in src:
            return None
        ts = src[var].sel(**sel).mean(["latitude", "longitude"]).values
        cl = clim_src[var].sel(**sel).mean(["latitude", "longitude"]).values
        times = pd.DatetimeIndex(src.time.values)
        anom = ts - cl[times.month.to_numpy() - 1]
        return [{"month": month_key(t), "value": _clean(v), "anomaly": _clean(a)}
                for t, v, a in zip(times, ts, anom)]

    def region_climatology(self, var):
        r = MODEL_REGION
        sel = dict(latitude=slice(r["south"], r["north"]), longitude=slice(r["west"], r["east"]))
        cl = self.clim[var].sel(**sel).mean(["latitude", "longitude"]).values
        return [_clean(v) for v in cl]
