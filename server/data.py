"""Read the processed outputs written by preprocess.ipynb (and the model notebook).

Data is streamed from the team's S3 folder by default:
  s3://ocean-hackathon-2026/teams/team-elnino/processed
using the "hack-team" AWS profile in ~/.aws (run `python setup_aws.py` once per machine).
Nothing is downloaded up front; each request reads only the chunks it needs.

Overrides (environment variables):
  ELNINO_DATA   another location, e.g. a local folder copy of processed/
  PRELOAD_DAILY=1  load all daily data into memory at startup (~1 GB; slower start, every map instant)
  AWS_PROFILE   another profile name
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  used instead of a profile if set

Expected files (any missing optional file just disables that feature):
  ocean_monthly.zarr, climatology.zarr, meta.json        required
  enso.json                                              optional - ENSO phase strip
  wind_monthly.zarr, wind_climatology.zarr              optional - wind charts
  forecast_mld.json                                      optional - written by the model notebook
  analysis/events.json                                   optional - El Nino history (pipeline/analyze_events.py)
  analysis/event_effects.zarr                            optional - per-event maps (start, duration, typical)
  ocean_daily.zarr, daily_normal.zarr, meta_daily.json   optional - daily maps (pipeline/build_daily_store.py);
                                                         when present the site works in days, not months
"""
import json
import os
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_DATA = "s3://ocean-hackathon-2026/teams/team-elnino/processed"
DATA_DIR = os.environ.get("ELNINO_DATA", DEFAULT_DATA).rstrip("/")
IS_S3 = DATA_DIR.startswith("s3://")
# Written by model/train copy.py next to the SEAS5 input; used when the data folder has no copy
LOCAL_FORECAST_GRID = Path(__file__).resolve().parent.parent / "future" / "clean" / "forecast_mld_grid.json"

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


# Fail with an error instead of hanging forever when S3 doesn't answer
S3_TIMEOUTS = {"config_kwargs": {"connect_timeout": 10, "read_timeout": 30,
                                 "retries": {"max_attempts": 3, "mode": "standard"}}}


def _storage_options():
    """Local folder: none. S3: key env vars if set, otherwise the hack-team profile."""
    if not IS_S3:
        return None
    if os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE"):
        return dict(S3_TIMEOUTS)
    return {"profile": os.environ.get("AWS_PROFILE", "hack-team"), **S3_TIMEOUTS}


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
        self.events = _read_json("analysis/events.json", None)   # El Nino history analysis
        self.effects = (_open_zarr("analysis/event_effects.zarr").load()   # ~10 MB: per-event maps
                        if _exists("analysis/event_effects.zarr") else None)

        # Daily maps (preferred when present). The normal is stored as coefficients and rebuilt per day,
        # exactly as in the El Nino analysis, so anomaly maps match it.
        self.daily = _open_zarr("ocean_daily.zarr") if _exists("ocean_daily.zarr") else None
        if self.daily is not None:
            self.daily_normal = _open_zarr("daily_normal.zarr").load()        # a few MB
            self.meta_daily = _read_json("meta_daily.json", {})
            self.days = pd.DatetimeIndex(self.daily.time.values)
            self.day_index = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(self.days)}
            self.daily_lats = self.daily.latitude.values
            self.daily_lons = self.daily.longitude.values
            if os.environ.get("PRELOAD_DAILY") == "1":
                print("Loading all daily data into memory (PRELOAD_DAILY=1)...", flush=True)
                self.daily = self.daily.load()
            # Each S3 read fetches a whole chunk (~a month of days). Keep recent chunks in memory so
            # neighbouring days and switching back to a variable don't go back to S3.
            self._blocks = OrderedDict()
            self._blocks_lock = threading.Lock()

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
        # Per-point forecast maps from model/train copy.py (SEAS5 grid): forecast_mld_grid.json
        # Falls back to the local copy the model writes, so it can be tried before uploading.
        self.forecast_grid = _read_json("forecast_mld_grid.json", None)
        if self.forecast_grid is None and LOCAL_FORECAST_GRID.exists():
            with open(LOCAL_FORECAST_GRID) as f:
                self.forecast_grid = json.load(f)
        self.forecast_grid_by_month = {m["month"]: m for m in (self.forecast_grid or {}).get("months", [])}

    def forecast_grid_map(self, month, anomaly=False):
        """One month of the per-point MLD forecast as (lats, lons, values, lower, upper), or None.
        anomaly=True gives forecast minus that point's normal (the band is shifted the same way)."""
        m = self.forecast_grid_by_month.get(month)
        if m is None:
            return None
        arr = lambda k: np.array(m[k], dtype="float64")       # None -> NaN
        values, lower, upper = arr("mld"), arr("lower"), arr("upper")
        if anomaly:
            normal = arr("normal")
            values, lower, upper = values - normal, lower - normal, upper - normal
        fg = self.forecast_grid
        return fg["latitudes"], fg["longitudes"], values, lower, upper

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

    def var_meta(self, var, daily=False):
        if daily and self.daily is not None and var in self.meta_daily.get("variables", {}):
            return self.meta_daily["variables"][var]
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

    # ---------- El Nino analysis maps (pipeline/analyze_events.py) ----------
    DAYS_PER_MONTH = 30.44

    def effect_grid(self, event_id, slug, field):
        """One event's effect at every point for one variable, or None.
        field: onset (months after the event began, <0 = already under way), duration (months),
        strength (peak, in SD of that point's own variability). Points with no clear change are NaN in
        values and 1 in the returned no_change mask."""
        ds = self.effects
        if ds is None or event_id not in ds.event.values or slug not in ds.variable.values:
            return None
        sel = ds.sel(event=event_id, variable=slug)
        if field == "onset":
            v = sel["onset_days"].values / self.DAYS_PER_MONTH
        elif field == "duration":
            v = sel["duration_days"].values / self.DAYS_PER_MONTH
        elif field == "strength":
            v = sel["peak_z"].values
        else:
            return None
        det = sel["detected"].values
        v = np.where(det == 1, v, np.nan)
        no_change = np.where(det == 0, 1.0, np.nan)   # ocean that was checked: no clear change (NaN = land)
        return ds.latitude.values, ds.longitude.values, v, no_change

    def typical_grid(self, phase, slug):
        """How consistently each point changed across all past events of a phase (el_nino / la_nina):
        +1 = went up (warmer/saltier/higher/deeper) in every one, -1 = down in every one, 0 = no pattern."""
        ds = self.effects
        if ds is None or slug not in ds.variable.values or f"{phase}_mean_z" not in ds:
            return None, 0
        n = int(ds.attrs.get(f"n_{phase}", 0))
        if n == 0:
            return None, 0
        sel = ds.sel(variable=slug)
        mean_z = sel[f"{phase}_mean_z"].values
        v = np.sign(mean_z) * sel[f"{phase}_n_same_sign"].values / n
        return (ds.latitude.values, ds.longitude.values, v), n

    # ---------- daily maps ----------
    def daily_normal_map(self, var, day):
        """The normal for one day at every point: design(day) @ coefficients (see build_daily_store.py)."""
        a = self.daily_normal.attrs
        y = (pd.Timestamp(day) - pd.Timestamp(a["origin"])).days / float(a["days_per_year"])
        x = [1.0]
        for k in range(1, int(a["harmonics"]) + 1):
            x += [np.cos(2 * np.pi * k * y), np.sin(2 * np.pi * k * y)]
        x.append(y - float(a["trend_origin_years"]))
        return np.tensordot(np.asarray(x), self.daily_normal[var].values, axes=1)

    MAX_BLOCKS = 120  # ~0.7 MB each

    def _daily_block(self, var, i):
        """The stored chunk of days containing day index i, as a numpy array (days, lat, lon)."""
        da = self.daily[var]
        size = da.chunks[0][0] if da.chunks else len(self.days)       # days per stored chunk
        b = i // size
        key = (var, b)
        with self._blocks_lock:
            if key in self._blocks:
                self._blocks.move_to_end(key)
                return self._blocks[key], i - b * size
        block = da.isel(time=slice(b * size, (b + 1) * size)).values   # one S3 read
        with self._blocks_lock:
            self._blocks[key] = block
            while len(self._blocks) > self.MAX_BLOCKS:
                self._blocks.popitem(last=False)
        return block, i - b * size

    def grid_day(self, var, day, anomaly=False, model_region_only=False):
        """One day as (lats, lons, 2D values, 'observed'), or None if that day isn't in the data."""
        if self.daily is None or day not in self.day_index or var not in self.daily:
            return None
        block, k = self._daily_block(var, self.day_index[day])
        values = block[k]
        if anomaly:
            values = values - self.daily_normal_map(var, day)
        lats, lons = self.daily_lats, self.daily_lons
        if model_region_only:
            r = MODEL_REGION
            i = (lats >= r["south"]) & (lats <= r["north"])
            j = (lons >= r["west"]) & (lons <= r["east"])
            lats, lons, values = lats[i], lons[j], values[np.ix_(i, j)]
        return lats, lons, values, "observed"

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
