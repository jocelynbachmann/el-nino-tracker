"""
predict.py - predicted mixed layer depth (MLD) for a given day, from the saved model
and the cleaned seasonal forecast grid.

    from predict import predict, predict_month, predict_all_months
    predict("2026-12-15")      # one day       -> per-cell MLD
    predict_month("2026-12")   # daily preds averaged over the month -> per-cell MLD
    predict_all_months()       # area-mean monthly MLD for every forecast month
    # per-cell columns: latitude, longitude, mld (m), mld_p10, mld_p90

    python predict.py            # table of all months
    python predict.py 2026-12    # one month, per cell

Default files (relative to this script's folder):
    model    : mld_gbm.joblib
    forecast : ../future/clean/seas5_monthly_grid.csv

Forecast columns used as model inputs (rough stand-ins):
    surface_temp_c        -> to   (currently 2 m air temp, see surface_temp_source)
    salinity_psu          -> so
    sea_surface_height_m  -> zo
    east_wind_ms  x wind_to_current -> ugo   (wind ~ surface current)
    north_wind_ms x wind_to_current -> vgo

Uncertainty: each input is jittered by its *_spread column (treated as a standard
deviation across forecast members), and the 10th/90th percentiles of the resulting
MLD predictions are returned as mld_p10 / mld_p90.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "mld_gbm.joblib"
DEFAULT_FORECAST = HERE / ".." / "future" / "clean" / "seas5_monthly_grid.csv"

# forecast column -> model input
COLUMN_MAP = {
    "surface_temp_c": "to",
    "salinity_psu": "so",
    "sea_surface_height_m": "zo",
    "east_wind_ms": "ugo",
    "north_wind_ms": "vgo",
}


@lru_cache(maxsize=None)
def _load_model(path):
    return joblib.load(path)


@lru_cache(maxsize=None)
def _load_forecast(path):
    fc = pd.read_csv(path)
    fc = fc[fc["is_ocean"].astype(str) == "True"].copy()
    fc["month"] = pd.to_datetime(fc["month"]).dt.to_period("M")
    return fc


def _build_inputs(df, bundle, doy, el_nino):
    k = bundle["wind_to_current"]
    X = pd.DataFrame({
        "latitude": df["lat"].values,
        "longitude": df["lon"].values,
        "to": df["surface_temp_c"].values,
        "so": df["salinity_psu"].values,
        "zo": df["sea_surface_height_m"].values,
        "ugo": df["east_wind_ms"].values * k,
        "vgo": df["north_wind_ms"].values * k,
    })
    X["speed"] = np.hypot(X["ugo"], X["vgo"])
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["el_nino"] = bundle["el_nino_default"] if el_nino is None else int(el_nino)
    return X


def _run(bundle, X):
    pred = bundle["model"].predict(X[bundle["features"]])
    return np.expm1(pred) if bundle.get("target_transform") == "log1p" else pred


def _mean_over_days(bundle, sel, doys, el_nino):
    """Feed the same monthly inputs in once per day (only the season changes),
    predict daily MLD, and return the per-cell average over those days."""
    n = len(sel)
    X = pd.concat([_build_inputs(sel, bundle, d, el_nino) for d in doys], ignore_index=True)
    daily = _run(bundle, X).reshape(len(doys), n)      # (days, cells)
    return daily.mean(axis=0)


def _predict(sel, doys, bundle, uncertainty, n_samples, el_nino, seed):
    out = pd.DataFrame({"latitude": sel["lat"].values, "longitude": sel["lon"].values})
    out["mld"] = _mean_over_days(bundle, sel, doys, el_nino)

    if uncertainty:
        rng = np.random.default_rng(seed)
        draws = np.empty((n_samples, len(sel)))
        for i in range(n_samples):
            jit = sel.copy()
            for col in COLUMN_MAP:
                spread = f"{col}_spread"
                if spread in jit:
                    jit[col] = jit[col] + rng.normal(0, 1, len(jit)) * jit[spread].fillna(0)
            draws[i] = _mean_over_days(bundle, jit, doys, el_nino)
        out["mld_p10"] = np.percentile(draws, 10, axis=0)
        out["mld_p90"] = np.percentile(draws, 90, axis=0)
    return out


def _month_rows(fc, period):
    sel = fc[fc["month"] == period]
    if sel.empty:
        months = sorted(str(m) for m in fc["month"].unique())
        raise ValueError(f"No forecast for {period}. Available months: {months}")
    return sel


def predict(day, forecast_csv=DEFAULT_FORECAST, model_path=DEFAULT_MODEL,
            uncertainty=True, n_samples=50, el_nino=None, seed=0):
    """
    Predict MLD (metres) for every ocean grid cell on one `day`, using that month's
    average forecast inputs.

    Returns a DataFrame: latitude, longitude, mld [, mld_p10, mld_p90]
    """
    day = pd.Timestamp(day)
    bundle = _load_model(str(model_path))
    sel = _month_rows(_load_forecast(str(forecast_csv)), day.to_period("M"))
    return _predict(sel, [day.dayofyear], bundle, uncertainty, n_samples, el_nino, seed)


def predict_month(month, forecast_csv=DEFAULT_FORECAST, model_path=DEFAULT_MODEL,
                  uncertainty=True, n_samples=50, el_nino=None, seed=0):
    """
    Monthly MLD (metres) for every ocean grid cell.

    The monthly-average forecast inputs are fed into the daily model once for every
    day of the month (only the time-of-year changes), and the daily predictions are
    averaged back to a monthly value per cell.

    month : "2026-12" (any date inside the month also works)

    Returns a DataFrame: latitude, longitude, mld [, mld_p10, mld_p90]
    """
    period = pd.Timestamp(month).to_period("M")
    bundle = _load_model(str(model_path))
    sel = _month_rows(_load_forecast(str(forecast_csv)), period)
    days = pd.date_range(period.start_time, period.end_time.normalize(), freq="D")
    return _predict(sel, list(days.dayofyear), bundle, uncertainty, n_samples, el_nino, seed)


def predict_all_months(**kwargs):
    """Area-weighted monthly MLD for every month in the forecast."""
    fc = _load_forecast(str(kwargs.get("forecast_csv", DEFAULT_FORECAST)))
    rows = []
    for period in sorted(fc["month"].unique()):
        res = predict_month(str(period), **kwargs)
        w = np.cos(np.deg2rad(res["latitude"]))
        row = {"month": str(period), "mld": np.average(res["mld"], weights=w)}
        if "mld_p10" in res:
            row["mld_p10"] = np.average(res["mld_p10"], weights=w)
            row["mld_p90"] = np.average(res["mld_p90"], weights=w)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        m = sys.argv[1]
        res = predict_month(m)
        print(res.head(10).round(2))
        w = np.cos(np.deg2rad(res["latitude"]))
        print(f"\n{len(res)} ocean cells | area-mean MLD for {m}: "
              f"{np.average(res['mld'], weights=w):.1f} m "
              f"(p10-p90 {np.average(res['mld_p10'], weights=w):.1f}-"
              f"{np.average(res['mld_p90'], weights=w):.1f} m)")
    else:
        print(predict_all_months().round(1).to_string(index=False))


