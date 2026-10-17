import importlib.util
import sys

import numpy as np
import pandas as pd

MODEL_FILE = "train copy.py"            # the file with predict_mld() in it
SEAS5_CSV = "seas5_monthly_grid.csv"

_model = None


def load_model():
    """Import the model file once; it trains on import."""
    global _model
    if _model is None:
        spec = importlib.util.spec_from_file_location("mld_model", MODEL_FILE)
        _model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_model)
    return _model


def load_seas5(path=SEAS5_CSV):
    df = pd.read_csv(path)
    df["month"] = df["month"].astype(str).str[:7]
    df["is_ocean"] = df["is_ocean"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def forecast_month(month, seas5=None):
    """MLD forecast for one month (e.g. "2027-01").

    Returns (grid, average):
      grid     DataFrame, rows = latitude (south -> north), columns = longitude (west -> east),
               MLD in metres, NaN = land or outside the model's ocean grid
      average  mean MLD over all ocean cells that got a prediction, in metres
    """
    seas5 = load_seas5() if seas5 is None else seas5
    rows = seas5[seas5.month == month]
    if rows.empty:
        months = ", ".join(sorted(seas5.month.unique()))
        raise ValueError(f"No SEAS5 data for {month}. Available: {months}")

    predict_mld = load_model().predict_mld
    out = []
    for r in rows.itertuples(index=False):
        mld = np.nan
        if r.is_ocean:
            try:
                mld = predict_mld(temp=r.surface_temp_c, salinity=r.salinity_psu,
                                  wind_speed=r.wind_speed_ms, surface_height=r.sea_surface_height_m,
                                  north_wind=r.north_wind_ms, east_wind=r.east_wind_ms,
                                  date=month, latitude=r.lat, longitude=r.lon)
            except ValueError:
                pass            # land or outside the box in the model's own grid
        out.append({"lat": r.lat, "lon": r.lon, "mld": mld})

    grid = (pd.DataFrame(out).pivot(index="lat", columns="lon", values="mld")
            .sort_index().sort_index(axis=1))
    average = float(np.nanmean(grid.values)) if grid.notna().any().any() else np.nan
    return grid, average


def main(args):
    save = "--save" in args
    months = [a for a in args if not a.startswith("--")]
    if not months:
        sys.exit(__doc__)
    seas5 = load_seas5()
    for month in months:
        grid, average = forecast_month(month, seas5)
        print(f"\n=== MLD forecast {month} (metres) ===")
        print(grid.round(1).iloc[::-1].to_string(na_rep="  land"))   # north at the top
        print(f"Area average: {average:.1f} m  ({grid.notna().sum().sum()} ocean cells)")
        if save:
            name = f"mld_forecast_{month}.csv"
            grid.round(2).to_csv(name)
            print(f"Saved {name}")


if __name__ == "__main__":
    main(sys.argv[1:])