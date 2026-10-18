"""El Niño tracker API. Serves data only; the client does all the drawing.

Run locally:
    cd server
    flask --app app run --debug --port 5001        then open http://localhost:5001/metadata

Endpoints (all GET, all JSON):
    /health
    /metadata                                   variables, value ranges, grid, months, forecast months
    /el-nino                                    monthly El Niño index (ONI) and phase
    /maps/<variable>/<YYYY-MM-DD>               one day's map values (daily data, preferred)
    /maps/<variable>/<YYYY-MM>                  one month's map values (older monthly data / forecast)
                                                  ?anomaly=true          difference from normal
                                                  ?area=model-region     crop to 46-52N, 134-120W
    /timeseries/location?lat=..&lon=..&variable=..   monthly series at the nearest grid cell
    /timeseries/region/<variable>               monthly series averaged over the model region
    /forecast                                   mixed layer depth forecast (empty until the model runs);
                                                  "grid" lists the months with per-point maps
    /forecast/maps/<YYYY-MM>                    per-point MLD forecast map (model/train copy.py on SEAS5)
                                                  ?anomaly=true          difference from normal
    /events                                     past El Ninos / La Ninas and what each did here
                                                  (from pipeline/analyze_events.py)
    /events/<event-id>/maps/<variable>/<field>  one event's effect at every point; field = onset
                                                  (months after it began), duration (months), strength (SD)
    /typical/<phase>/maps/<variable>            el-nino | la-nina: how consistently each point changed
                                                  the same way across all past events (-1..1)

Variables: temperature, salinity, sea-surface-height, mixed-layer-depth, current-east,
current-north. /timeseries/region also takes wind-speed, wind-north, wind-east, wind-stress.
"""
import re
import time
from functools import lru_cache

import numpy as np
from flask import Flask, jsonify, request

from data import CODE_TO_NAME, MODEL_REGION, OCEAN_VARIABLES, WIND_VARIABLES, DataStore

app = Flask(__name__)
app.json.sort_keys = False
store = DataStore()

MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
DAY_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")


# ---------- plumbing ----------
@app.after_request
def add_cors(resp):
    # The React dev server runs on another port; allow it to call us.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def error(msg, status):
    return jsonify(error=msg), status


def get_variable(name, include_wind=False):
    code = store.resolve_var(name, include_wind)
    if code is None:
        known = list(OCEAN_VARIABLES) + (list(WIND_VARIABLES) if include_wind else [])
        return None, error(f"Unknown variable {name!r}. Known: {', '.join(known)}", 404)
    return code, None


def flag(name):
    return request.args.get(name, "false").lower() in ("1", "true", "yes")


def axis_info(values):
    values = np.asarray(values, dtype="float64")
    return {"start": round(float(values[0]), 5), "step": round(float(values[1] - values[0]), 5),
            "count": int(len(values))}


def variable_info(code, daily=False):
    vm = store.var_meta(code, daily)
    return {"label": vm["label"], "units": vm["units"]}


# ---------- endpoints ----------
@app.get("/health")
def health():
    return jsonify(status="ok", months=len(store.months), forecast=bool(store.forecast_by_month))


@app.get("/metadata")
def metadata():
    meta = store.meta
    daily = store.daily is not None
    var_source = store.meta_daily.get("variables", {}) if daily else meta.get("variables", {})
    variables = {}
    for code, vm in var_source.items():
        variables[CODE_TO_NAME.get(code, code)] = {
            "label": vm["label"], "units": vm["units"],
            "range": vm["range"], "anomaly_range": vm["anomaly_range"],
            "colormap": vm["cmap"], "has_map": True, "has_forecast": vm.get("forecast", False),
        }
    for name, (code, label, units) in WIND_VARIABLES.items():
        if store.wind is not None and code in store.wind:
            variables[name] = {"label": label, "units": units, "has_map": False, "has_forecast": False}
    md = store.meta_daily if daily else {}
    return jsonify({
        # "day" when the daily store exists: maps are requested as /maps/<variable>/<YYYY-MM-DD>
        "time_step": "day" if daily else "month",
        "observed_days": {"start": md.get("first_day"), "end": md.get("last_day")} if daily else None,
        "normal": md.get("normal") if daily else {"climatology_years": meta.get("climatology_years")},
        "first_month": meta.get("first_month"),
        "last_observed_month": meta.get("last_observed_month"),
        "observed_months": list(store.month_index),
        "forecast_months": list(store.forecast_by_month),
        "forecast_issued": (store.forecast or {}).get("issued"),
        "climatology_years": meta.get("climatology_years"),
        "bounds": md.get("bounds") if daily else meta.get("bounds"),
        "grid": ({"latitude": axis_info(store.daily_lats), "longitude": axis_info(store.daily_lons)} if daily
                 else {"latitude": axis_info(store.lats), "longitude": axis_info(store.lons)}),
        "model_region": MODEL_REGION,
        "variables": variables,
    })


@app.get("/el-nino")
def el_nino():
    return jsonify(store.enso)


@lru_cache(maxsize=512)
def map_payload(code, when, anomaly, model_region_only):
    daily = DAY_RE.match(when) is not None
    out = (store.grid_day(code, when, anomaly, model_region_only) if daily
           else store.grid(code, when, anomaly, model_region_only))
    if out is None:
        return None
    lats, lons, values, source = out
    vm = store.var_meta(code, daily)
    rows = np.round(np.asarray(values, dtype="float64"), 3)
    rows = [[None if np.isnan(v) else float(v) for v in row] for row in rows]
    return {
        "variable": CODE_TO_NAME.get(code, code), **variable_info(code, daily),
        ("day" if daily else "month"): when, "anomaly": anomaly, "source": source,   # observed | forecast
        "color_range": vm["anomaly_range"] if anomaly else vm["range"],  # same for every month
        "latitudes": [round(float(x), 5) for x in lats],           # south -> north (rows)
        "longitudes": [round(float(x), 5) for x in lons],          # west -> east (columns)
        "values": rows,                                            # values[row][col]; null = land
    }


@app.get("/maps/<variable>/<when>")
def maps(variable, when):
    code, err = get_variable(variable)
    if err:
        return err
    if not (DAY_RE.match(when) or MONTH_RE.match(when)):
        return error("Use a day like 2015-08-15 (or a month like 2015-08)", 400)
    area = request.args.get("area", "full")
    if area not in ("full", "model-region"):
        return error("area must be 'full' (default) or 'model-region'", 400)
    t0 = time.perf_counter()
    payload = map_payload(code, when, flag("anomaly"), area == "model-region")
    ms = (time.perf_counter() - t0) * 1000
    if ms > 300:
        print(f"slow map: {variable} {when} took {ms:.0f} ms (data read)", flush=True)
    if payload is None:
        return error(f"No {variable} data or forecast for {when}", 404)
    resp = jsonify(payload)
    resp.headers["Server-Timing"] = f"data;dur={ms:.0f}"     # visible in the browser's Network tab
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


@app.get("/timeseries/location")
def timeseries_location():
    name = request.args.get("variable", "temperature")
    code, err = get_variable(name)
    if err:
        return err
    try:
        lat, lon = float(request.args["lat"]), float(request.args["lon"])
    except (KeyError, ValueError):
        return error("lat and lon are required numbers, e.g. ?lat=48.5&lon=-125.2", 400)
    if not store.in_bounds(lat, lon):
        return error("Location is outside the data area", 404)
    out = store.point_series(code, lat, lon)
    return jsonify(variable=CODE_TO_NAME.get(code, code), **variable_info(code), **out)


@app.get("/timeseries/region/<variable>")
def timeseries_region(variable):
    code, err = get_variable(variable, include_wind=True)
    if err:
        return err
    series = store.region_series(code)
    if series is None:
        return error(f"No regional series for {variable!r}", 404)
    if code in store.meta.get("variables", {}):
        info = variable_info(code)
    else:
        _, label, units = WIND_VARIABLES[CODE_TO_NAME[code]]
        info = {"label": label, "units": units}
    return jsonify(variable=CODE_TO_NAME.get(code, code), **info, region=MODEL_REGION, series=series)


@app.get("/events")
def events():
    """El Nino / La Nina events since the daily record starts, each with a per-variable summary of
    its effect on this region, plus the region-wide unusual spells and main patterns."""
    if store.events is None:
        return error("No analysis/events.json yet - run pipeline/analyze_events.py", 404)
    resp = jsonify(store.events)
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


# Colour ranges and colormap hints for the analysis maps (same JSON shape as /maps)
ANALYSIS_FIELDS = {
    "onset": {"label": "When the change started", "units": "months after the event began",
              "color_range": [-6, 18], "colormap": "plasma"},
    "duration": {"label": "How long it lasted", "units": "months", "color_range": [0, 30],
                 "colormap": "blues_visible"},
    "strength": {"label": "How unusual it got", "units": "SD of that spot's normal variability",
                 "color_range": [-4, 4], "colormap": "rdbu_r"},
}
ANALYSIS_VARIABLES = ("temperature", "salinity", "sea-surface-height", "mixed-layer-depth")


def grid_json(lats, lons, values, **extra):
    rows = np.round(np.asarray(values, dtype="float64"), 3)
    return {**extra,
            "latitudes": [round(float(x), 5) for x in lats],
            "longitudes": [round(float(x), 5) for x in lons],
            "values": [[None if np.isnan(v) else float(v) for v in row] for row in rows]}


@app.get("/events/<event_id>/maps/<variable>/<field>")
def event_map(event_id, variable, field):
    if store.effects is None:
        return error("No analysis/event_effects.zarr yet - run pipeline/analyze_events.py", 404)
    if variable not in ANALYSIS_VARIABLES:
        return error(f"Only {', '.join(ANALYSIS_VARIABLES)} were analysed", 404)
    if field not in ANALYSIS_FIELDS:
        return error(f"field must be one of {', '.join(ANALYSIS_FIELDS)}", 400)
    out = store.effect_grid(event_id, variable, field)
    if out is None:
        return error(f"No event {event_id!r}", 404)
    lats, lons, values, no_change = out
    payload = grid_json(lats, lons, values, variable=variable, event=event_id, field=field, **ANALYSIS_FIELDS[field])
    # 1 where the spot was checked but had no clear change (drawn grey), null for land / changed spots
    payload["no_change"] = [[None if np.isnan(v) else 1 for v in row] for row in no_change]
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = "no-cache"   # small; always fetch the latest analysis
    return resp


@app.get("/typical/<phase>/maps/<variable>")
def typical_map(phase, variable):
    phase_key = {"el-nino": "el_nino", "la-nina": "la_nina"}.get(phase)
    if phase_key is None:
        return error("phase must be el-nino or la-nina", 400)
    if variable not in ANALYSIS_VARIABLES:
        return error(f"Only {', '.join(ANALYSIS_VARIABLES)} were analysed", 404)
    out, n = store.typical_grid(phase_key, variable)
    if out is None:
        return error("No analysis/event_effects.zarr yet - run pipeline/analyze_events.py", 404)
    resp = jsonify(grid_json(*out, variable=variable, phase=phase, n_events=n,
                             label=f"Same change in past {phase.replace('-', ' ')}s",
                             units="share of past events", color_range=[-1, 1], colormap="rdbu_r"))
    resp.headers["Cache-Control"] = "no-cache"   # small; always fetch the latest analysis
    return resp


@app.get("/forecast")
def forecast():
    store.reload_forecast()                    # pick up a new model run without restarting
    map_payload.cache_clear()
    fc = store.forecast or {"months": []}
    fg = store.forecast_grid
    grid = ({"issued": fg.get("issued"), "model": fg.get("model"), "inputs": fg.get("inputs"),
             "band": fg.get("band"), "months": list(store.forecast_grid_by_month)} if fg else None)
    # Each month already carries its own "normal" (same 1993-2018 baseline as every other endpoint)
    return jsonify({**fc, "variable": "mixed-layer-depth", "region": MODEL_REGION, "grid": grid})


@app.get("/forecast/maps/<month>")
def forecast_map(month):
    """Per-point MLD forecast for one month (model/train copy.py on SEAS5), same shape as /maps.
    ?anomaly=true gives the difference from each point's normal."""
    if not MONTH_RE.match(month):
        return error("Use a month like 2027-01", 400)
    anomaly = flag("anomaly")
    out = store.forecast_grid_map(month, anomaly)
    if out is None:
        months = ", ".join(store.forecast_grid_by_month) or "none - run model/train copy.py"
        return error(f"No per-point MLD forecast for {month}. Available: {months}", 404)
    lats, lons, values, lower, upper = out
    vm = store.var_meta("mlotst", store.daily is not None)     # same colours as the observed MLD layer
    clean = lambda a: grid_json(lats, lons, a)["values"]
    fg = store.forecast_grid
    return jsonify(grid_json(lats, lons, values,
                             variable="mixed-layer-depth", label="Mixed layer depth forecast",
                             units=vm["units"], month=month, anomaly=anomaly, source="forecast",
                             issued=fg.get("issued"), model=fg.get("model"), band=fg.get("band"),
                             color_range=vm["anomaly_range"] if anomaly else vm["range"],
                             colormap=vm["cmap"]) | {"lower": clean(lower), "upper": clean(upper)})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
