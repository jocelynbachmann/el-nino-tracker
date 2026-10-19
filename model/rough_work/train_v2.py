"""Mixed layer depth (MLD) model, version 2.

Run on JupyterHub:   %run train_v2.py      (or paste each "STEP" block into its own cell)

What it does
  1. Builds a table: one row per ocean grid cell per month (46-52N, 134-120W, 1/4 degree),
     with that month's sea surface temperature, salinity, height and wind, and the MLD.
  2. Turns everything into anomalies (difference from that cell's normal for that calendar month).
  3. Compares 4 input sets x 2 model types with rolling validation (train on everything before
     each block of years, score the block: 2010-12, 2013-15, 2016-18) and keeps the best average.
     Only Argo-era data (2005 onward) is used; see START below.
  4. Scores it once on 2019-2026 against "just predict normal" (climatology), and prints a
     per-year table of errors and input anomalies to show any drift.
  5. Retrains on all years and forecasts 24 months ahead:
       - months with Copernicus C3S forecast inputs -> the model's prediction
       - months without (or if no C3S file exists yet) -> normal MLD for that month
  6. Writes forecast_mld.json to the team bucket for the server's /forecast endpoint.

Every input must also exist in the Copernicus C3S seasonal forecasts, in the same form,
because those forecasts replace the observed inputs when predicting the future.
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ======================================================================================
# STEP 1 - Settings
# ======================================================================================
PROFILE = "hack-team"
OUT_DIR = "ocean-hackathon-2026/teams/team-elnino/processed"

LAT = (46, 52)            # model region, degrees north
LON = (-134, -120)        # degrees east (negative = west)

# Argo era only. ARMOR3D's mixed layer before ~2002 is several metres shallower than after
# (estimated from far fewer below-surface measurements before the Argo float network), so the
# model trains on 2005 onward, when Argo coverage was good, and "normal" is 2005-2018 too.
START = "2005-01-01"
CLIM_YEARS = ("2005", "2018")   # train + validation years only, so the test years are never seen

# Time splits
VAL_END = "2019-01-01"    # before this: training + rolling validation (see FOLDS below)
                          # test: 2019 - May 2026 (scored once, at the end)

HORIZON = 24              # months to forecast
C3S_MONTHS = 6            # Copernicus seasonal forecasts reach ~6 months
EL_NINO_WEIGHT = 3.0      # El Niño months count 3x in training (rare but most important)
BAND_Z = 1.2816           # +/- this many standard deviations = 80% uncertainty band

# Inputs. Each has a matching Copernicus C3S forecast variable:
#   sst   <- sea surface temperature              (seasonal-monthly-single-levels)
#   speed <- 10m wind speed                        (seasonal-monthly-single-levels)
#   u, v  <- 10m u / v component of wind           (seasonal-monthly-single-levels)
#   sss   <- sea surface salinity                  (seasonal-monthly-ocean)
#   ssh   <- sea surface height above geoid        (seasonal-monthly-ocean)
# Not used: wind stress magnitude (C3S only gives averaged stress components, which cancel
# out when the wind changes direction, so it would not match), currents (no forecast).
ANOM_VARS = ["sst", "sss", "ssh", "speed", "u", "v"]
BASE_FEATS = ["mld_clim", "sst_clim", "month", "latitude", "longitude"]

# Input sets compared in STEP 4. "ssh_slow_removed" = sea height minus its own average over the
# previous 5 years, which removes slow sea-level drift but keeps El Niño-scale changes.
INPUT_SETS = {
    "all":              ["sst_anom", "sss_anom", "ssh_anom", "speed_anom", "u_anom", "v_anom"],
    "ssh_slow_removed": ["sst_anom", "sss_anom", "ssh_hp_anom", "speed_anom", "u_anom", "v_anom"],
    "no_ssh":           ["sst_anom", "sss_anom", "speed_anom", "u_anom", "v_anom"],
    "no_ssh_no_sss":    ["sst_anom", "speed_anom", "u_anom", "v_anom"],
}
# Rolling validation: train on everything before each block, score on the block.
# All blocks are before 2019, so the test years stay untouched until the very end.
FOLDS = [("2010-01-01", "2013-01-01"), ("2013-01-01", "2016-01-01"), ("2016-01-01", "2019-01-01")]
SUBSAMPLE_FOR_COMPARISON = True   # use every 2nd grid cell while comparing (4x faster); final model uses all


def feats_for(input_set):
    return INPUT_SETS[input_set] + BASE_FEATS


def years_since_2000(t):
    """Works for a single date, a column of dates or a date index."""
    return (pd.to_datetime(t) - pd.Timestamp("2000-01-01")) / pd.Timedelta(days=365.25)


# ======================================================================================
# STEP 2 - Load and build the table (one row per ocean cell per month)
# ======================================================================================
def connect():
    import s3fs
    return s3fs.S3FileSystem(profile=PROFILE)


def build_table(fs):
    """Returns (df, last_observed_month). Ocean data is averaged 2x2 onto the 1/4 degree
    wind grid so every row has both ocean and wind values for the same patch of sea."""
    import xarray as xr
    opts = {"profile": PROFILE}
    ocean = xr.open_zarr(f"s3://{OUT_DIR}/ocean_monthly.zarr", storage_options=opts)
    wind = xr.open_zarr(f"s3://{OUT_DIR}/wind_monthly.zarr", storage_options=opts)
    last_observed = pd.Timestamp(ocean.time.values[-1])

    cut = dict(latitude=slice(*LAT), longitude=slice(*LON))
    oc = ocean[["to", "so", "zo", "mlotst"]].sel(**cut).load()
    wd = wind[["wind_speed", "eastward_wind", "northward_wind"]].sel(**cut).load()

    oc = oc.coarsen(latitude=2, longitude=2, boundary="trim").mean()
    assert np.allclose(oc.latitude, wd.latitude, atol=0.01) and \
        np.allclose(oc.longitude, wd.longitude, atol=0.01), "ocean and wind grids don't line up"
    oc = oc.assign_coords(latitude=wd.latitude.values, longitude=wd.longitude.values)

    times = np.intersect1d(oc.time.values, wd.time.values)
    times = times[times >= np.datetime64(START)]                     # 2005 - May 2026
    ds = xr.Dataset({
        "mld": oc.mlotst.sel(time=times), "sst": oc.to.sel(time=times),
        "sss": oc.so.sel(time=times), "ssh": oc.zo.sel(time=times),
        "speed": wd.wind_speed.sel(time=times),                      # average speed, like C3S
        "u": wd.eastward_wind.sel(time=times), "v": wd.northward_wind.sel(time=times),
    })
    df = ds.to_dataframe().reset_index().dropna()                    # drops land cells
    df["time"] = pd.to_datetime(df["time"])
    print(f"table: {len(df):,} rows, {df.groupby(['latitude', 'longitude']).ngroups} ocean cells, "
          f"{df.time.min():%Y-%m} to {df.time.max():%Y-%m}")

    # Normals over CLIM_YEARS (Argo era), computed here from the monthly data.
    # NOTE: the app's maps still use the pipeline's 1993-2018 normal until preprocess.ipynb's
    # climatology years are changed to 2005-2018 as well.
    period = slice(f"{CLIM_YEARS[0]}-01-01", f"{CLIM_YEARS[1]}-12-31")
    oc_fine = ocean[["to", "so", "zo", "mlotst"]].sel(**cut, time=period).load()
    oc_clim_fine = oc_fine.groupby("time.month").mean("time")
    regional_normal = {int(m): float(v) for m, v in zip(
        oc_clim_fine.month.values, oc_clim_fine.mlotst.mean(["latitude", "longitude"]).values)}
    occ = oc_clim_fine.coarsen(latitude=2, longitude=2, boundary="trim").mean()
    occ = occ.assign_coords(latitude=wd.latitude.values, longitude=wd.longitude.values)
    wdc = wd.sel(time=period).groupby("time.month").mean("time")
    clim = xr.Dataset({
        "mld_clim": occ.mlotst, "sst_clim": occ.to, "sss_clim": occ.so, "ssh_clim": occ.zo,
        "speed_clim": wdc.wind_speed, "u_clim": wdc.eastward_wind, "v_clim": wdc.northward_wind,
    }).to_dataframe().reset_index().dropna()
    clim["month"] = clim["month"].astype(int)
    return df, last_observed, clim, regional_normal


def load_enso(fs):
    with fs.open(f"{OUT_DIR}/enso.json") as f:
        e = pd.DataFrame(json.load(f))
    e["time"] = pd.to_datetime(e["month"], format="%Y-%m")
    return e[["time", "oni", "phase"]]


# ======================================================================================
# STEP 3 - Normals (climatology) and anomalies
# ======================================================================================
def add_anomalies(df, clim=None):
    """Normal = that cell's average for that calendar month over CLIM_YEARS (2005-2018).
    Inputs and target become differences from normal. The model predicts the MLD anomaly,
    and the normal is added back to get MLD in metres."""
    df = df.copy()
    df["month"] = df.time.dt.month
    keys = ["month", "latitude", "longitude"]
    if clim is None:   # only for offline testing without the climatology files
        clim = (df[df.time < VAL_END].groupby(keys)[["mld"] + ANOM_VARS].mean()
                .add_suffix("_clim").reset_index())
    df = df.merge(clim, on=keys, how="left").dropna()
    for v in ANOM_VARS:
        df[f"{v}_anom"] = df[v] - df[f"{v}_clim"]
    df["target"] = df.mld - df.mld_clim

    # Sea level rises ~3-4 mm/yr everywhere, so recent SSH anomalies drift upward for reasons
    # unrelated to mixing. Remove a straight-line trend fitted on the normal-period years.
    region_ssh = df.groupby("time").ssh_anom.mean()
    fit_on = region_ssh[region_ssh.index < VAL_END]
    slope, intercept = np.polyfit(years_since_2000(fit_on.index), fit_on.values, 1)
    df["ssh_anom"] -= slope * years_since_2000(df.time) + intercept
    # Also: SSH minus its regional average over the previous 5 years (past months only, so the
    # same thing can be computed at forecast time from the last observed months).
    region_ssh = df.groupby("time").ssh_anom.mean()
    slow = region_ssh.shift(1).rolling(60, min_periods=12).mean().fillna(0.0)
    df["ssh_hp_anom"] = df.ssh_anom - df.time.map(slow)
    ssh_trend = {"slope_m_per_year": float(slope), "intercept_m": float(intercept),
                 "recent_slow_m": float(region_ssh.iloc[-60:].mean())}
    print(f"sea level trend removed from SSH: {slope * 1000:.1f} mm/yr")
    return df, clim, ssh_trend


def label_enso(df, enso):
    """El Niño label per month - used only for training weights and scoring, never as an input
    (the label is decided using months that come later)."""
    df = df.merge(enso[["time", "phase"]], on="time", how="left")
    df["phase"] = df["phase"].fillna("unknown")
    return df


# ======================================================================================
# STEP 4 - Models and scoring
# ======================================================================================
def make_models(feats):
    numeric = [f for f in feats if f != "month"]
    linear = make_pipeline(
        ColumnTransformer([("month", OneHotEncoder(handle_unknown="ignore"), ["month"]),
                           ("num", StandardScaler(), numeric)]),
        Ridge(alpha=1.0))
    boosted = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, min_samples_leaf=200, l2_regularization=1.0,
        categorical_features=["month"], random_state=0)
    return {"linear": linear, "boosted": boosted}


def fit(model, rows, feats, el_nino_weight=EL_NINO_WEIGHT):
    w = np.where(rows.phase == "el_nino", el_nino_weight, 1.0)
    step = model.steps[-1][0] if hasattr(model, "steps") else None
    kw = {f"{step}__sample_weight": w} if step else {"sample_weight": w}
    return model.fit(rows[feats], rows.target, **kw)


def score(rows, pred_anom):
    """RMSE in metres, skill vs climatology (>0 = better than predicting normal), and R² of the
    anomaly (how much of the departure from normal the model explains)."""
    err, err_clim = rows.target - pred_anom, rows.target
    mse, mse_clim = (err ** 2).mean(), (err_clim ** 2).mean()
    var = ((rows.target - rows.target.mean()) ** 2).mean()
    return {"rmse_m": round(float(np.sqrt(mse)), 2),
            "climatology_rmse_m": round(float(np.sqrt(mse_clim)), 2),
            "skill_vs_climatology": round(float(1 - mse / mse_clim), 3),
            "anomaly_r2": round(float(1 - mse / var), 3)}


def regional(rows, pred_anom):
    """Average over the region per month - what the app actually shows."""
    r = rows.assign(pred_anom=pred_anom).groupby("time").agg(
        target=("target", "mean"), pred_anom=("pred_anom", "mean"),
        month=("month", "first"), phase=("phase", "first"))
    return r


def report(name, rows, pred_anom):
    s = score(rows, pred_anom)
    reg = regional(rows, pred_anom)
    s_reg = score(reg, reg.pred_anom)
    print(f"\n== {name} ==")
    print(f"  per cell : {s}")
    print(f"  regional : {s_reg}")
    print("  regional, by ENSO phase:")
    for phase, g in reg.groupby("phase"):
        if len(g) > 2:
            print(f"    {phase:9s} ({len(g):3d} months): {score(g, g.pred_anom)}")
    return {"per_cell": s, "regional": s_reg}


def yearly_drift(rows, pred_anom):
    """Per year: average model error and average input anomalies over the region.
    A steady non-zero error that tracks an input's drift = that input moved outside the
    range the model learned from."""
    r = rows.assign(err=pred_anom - rows.target)
    t = r.groupby(r.time.dt.year).agg(error_m=("err", "mean"), mld_anom=("target", "mean"),
                                      sst_anom=("sst_anom", "mean"), sss_anom=("sss_anom", "mean"),
                                      ssh_anom_cm=("ssh_anom", "mean"), ssh_hp_cm=("ssh_hp_anom", "mean"),
                                      speed_anom=("speed_anom", "mean"))
    t["ssh_anom_cm"] *= 100
    t["ssh_hp_cm"] *= 100
    return t.round(2)


def diagnostics(df, pre, test, feats, model_name, pred_test):
    """Checks that tell apart the possible reasons for a test-year failure."""
    print("\n" + "=" * 88 + "\nDIAGNOSTICS (test years 2019 - May 2026, regional average unless noted)\n" + "=" * 88)
    reg_all = df.groupby("time").agg(target=("target", "mean"))          # observed regional anomaly
    reg_test = regional(test, pred_test)

    # 1. Baselines vs model
    rows = []
    rows.append(("normal (1993-2018 climatology)", score(reg_test, 0 * reg_test.target)))
    rows.append((f"model ({model_name})", score(reg_test, reg_test.pred_anom)))
    # recent normal, updated monthly: normal + average anomaly of the previous 36 observed months
    rolling_offset = reg_all.target.shift(1).rolling(36, min_periods=24).mean()
    rows.append(("normal + last 36 months' offset (updated monthly)",
                 score(reg_test, rolling_offset.reindex(reg_test.index).values)))
    # recent normal, frozen at the start of the test period (like a long-lead forecast would be)
    frozen = float(reg_all.target[(reg_all.index >= "2016-01-01") & (reg_all.index < VAL_END)].mean())
    rows.append((f"normal + 2016-2018 offset ({frozen:+.2f} m, frozen)",
                 score(reg_test, np.full(len(reg_test), frozen))))
    # same model, no El Niño weighting
    m_nw = fit(make_models(feats)[model_name], pre, feats, el_nino_weight=1.0)
    pred_nw = m_nw.predict(test[feats])
    reg_nw = regional(test, pred_nw)
    rows.append((f"model ({model_name}) without El Niño weighting", score(reg_nw, reg_nw.pred_anom)))
    print("\n1. Model vs simple baselines:")
    print(pd.DataFrame({name: sc for name, sc in rows}).T.to_string())

    # 2. Does the bias depend on an input? Error by low / middle / high third of each input
    #    (thirds defined on the years before 2019). Per grid cell.
    print("\n2. Average error (prediction - actual, m) by input range. A large difference between")
    print("   low and high = the model's bias is tied to that input:")
    err = pred_test - test.target.values
    out = {}
    for f in [c for c in feats if c.endswith("_anom")]:
        lo, hi = pre[f].quantile([1 / 3, 2 / 3])
        bins = pd.cut(test[f], [-np.inf, lo, hi, np.inf], labels=["low", "middle", "high"])
        g = pd.Series(err, index=test.index).groupby(bins, observed=False)
        out[f] = {**g.mean().round(2).to_dict(),
                  "share_high_%": round(100 * float((bins == "high").mean()), 0)}
    print(pd.DataFrame(out).T.to_string())
    print("   (share_high_% is how often test months fall in the top third; ~33 = no shift)")

    # 3. Reprocessed (2019-2024) vs near-real-time (2025-2026) ARMOR3D
    print("\n3. Reprocessed vs near-real-time data:")
    for label, sel in (("2019-2024 (reprocessed)", reg_test.index < "2025-01-01"),
                       ("2025-2026 (near-real-time)", reg_test.index >= "2025-01-01")):
        r = reg_test[sel]
        print(f"   {label:27s} model {score(r, r.pred_anom)}  | mean actual anomaly "
              f"{r.target.mean():+.2f} m, mean model error {(r.pred_anom - r.target).mean():+.2f} m")

    # 4. Regional anomaly by year across the whole record (is the deep shift real/long-running?)
    yearly = reg_all.target.groupby(reg_all.index.year).mean().round(2)
    print("\n4. Regional MLD anomaly by year, whole record (m, + = deeper than 1993-2018 normal):")
    print("   " + "  ".join(f"{y}:{v:+.1f}" for y, v in yearly.items()))


def compare_and_test(df):
    pre = df[df.time < VAL_END]
    test = df[df.time >= VAL_END]
    if SUBSAMPLE_FOR_COMPARISON:
        lat_i = (pre.latitude.rank(method="dense") % 2 == 0)
        lon_i = (pre.longitude.rank(method="dense") % 2 == 0)
        pre_cmp = pre[lat_i & lon_i]
    else:
        pre_cmp = pre
    print(f"\nrows: before 2019 {len(pre):,} (comparison uses {len(pre_cmp):,})  test {len(test):,}")

    # 1. Rolling validation: every input set x model type, on every fold
    results = []
    for set_name in INPUT_SETS:
        feats = feats_for(set_name)
        for model_name in ("linear", "boosted"):
            fold_skill = []
            for start, end in FOLDS:
                tr = pre_cmp[pre_cmp.time < start]
                va = pre_cmp[(pre_cmp.time >= start) & (pre_cmp.time < end)]
                model = fit(make_models(feats)[model_name], tr, feats)
                reg = regional(va, model.predict(va[feats]))
                fold_skill.append(score(reg, reg.pred_anom)["skill_vs_climatology"])
            results.append({"inputs": set_name, "model": model_name,
                            **{f"{a[:4]}-{int(b[:4]) - 1}": v for (a, b), v in zip(FOLDS, fold_skill)},
                            "mean": round(float(np.mean(fold_skill)), 3),
                            "worst": round(float(np.min(fold_skill)), 3)})
            print(f"  done: {set_name:17s} {model_name}")
    table = pd.DataFrame(results).sort_values("mean", ascending=False)
    print("\n== Rolling validation: regional skill vs climatology (>0 beats 'predict normal') ==")
    print(table.to_string(index=False))

    best = table.iloc[0]
    set_name, model_name = best["inputs"], best["model"]
    feats = feats_for(set_name)
    print(f"\n-> chosen: inputs={set_name}, model={model_name} (best average across folds)")

    # 2. Refit on all years before 2019 (all cells), score ONCE on the untouched test years
    model = fit(make_models(feats)[model_name], pre, feats)
    pred_test = model.predict(test[feats])
    test_scores = report(f"TEST 2019-2026: {set_name} / {model_name}", test, pred_test)

    print("\nPer year (test): average error (prediction - actual) and average input anomalies")
    print(yearly_drift(test, pred_test).to_string())
    print("\nSame for 2013-2018 with a model trained before 2013 (for comparison):")
    m_early = fit(make_models(feats)[model_name], pre[pre.time < "2013-01-01"], feats)
    late = pre[pre.time >= "2013-01-01"]
    print(yearly_drift(late, m_early.predict(late[feats])).to_string())

    diagnostics(df, pre, test, feats, model_name, pred_test)

    sample = test.sample(min(5000, len(test)), random_state=0)
    imp = permutation_importance(model, sample[feats], sample.target, n_repeats=5, random_state=0)
    print("\ninput importance on test years (how much worse the model gets without each one):")
    print(pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False).round(3).to_string())

    reg = regional(test, pred_test)
    return (set_name, model_name), table, test_scores, reg


# ======================================================================================
# STEP 5 - Uncertainty bands
# ======================================================================================
def band_sigmas(df, reg_test):
    """Spread (standard deviation) of regional errors per calendar month, smoothed over the
    neighbouring months so each value is based on more than a handful of years.
      model: test-year errors of the model fed with OBSERVED inputs (a best case - real
             forecast inputs add their own error, so the band widens with lead time below)
      clim:  how far real MLD strays from normal in the normal-period years (the honest
             spread once there is no forecast information)"""
    def smooth(series_by_month):
        out = {}
        for m in range(1, 13):
            nb = [(m - 2) % 12 + 1, m, m % 12 + 1]
            vals = series_by_month[series_by_month.index.isin(nb)]
            out[m] = float(np.sqrt((vals ** 2).mean()))
        return out

    err = pd.Series((reg_test.target - reg_test.pred_anom).values, index=reg_test.month.values)
    sig_model = smooth(err)
    clim_reg = df[df.time < VAL_END].groupby("time").agg(target=("target", "mean"),
                                                          month=("month", "first"))
    sig_clim = smooth(pd.Series(clim_reg.target.values, index=clim_reg.month.values))
    return sig_model, sig_clim


# ======================================================================================
# STEP 6 - Forecast 24 months ahead
# ======================================================================================
def load_forecast_inputs(fs):
    """Copernicus C3S forecast inputs, prepared by the (separate) C3S download notebook and
    saved as processed/forecast_inputs.zarr with:
        dims  time (month start), latitude, longitude  on this model's 1/4 degree grid
        vars  sst_anom, sss_anom, ssh_anom, speed_anom, u_anom, v_anom
    Anomalies are C3S forecast minus C3S's own hindcast normal for that month and lead
    (SST in K or degC is the same as an anomaly). Returns None until that file exists."""
    path = f"{OUT_DIR}/forecast_inputs.zarr"
    if not fs.exists(path):
        print("\nNo forecast_inputs.zarr yet: every forecast month uses normal conditions.")
        return None
    import xarray as xr
    ds = xr.open_zarr(f"s3://{path}", storage_options={"profile": PROFILE}).load()
    inputs = ds.to_dataframe().reset_index()
    inputs["time"] = pd.to_datetime(inputs["time"])
    return inputs


def forecast(model, feats, clim, regional_normal, ssh_trend, last_observed, inputs, sig_model, sig_clim):
    months = pd.date_range(last_observed + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
    cells = clim[["latitude", "longitude"]].drop_duplicates()
    out = []
    for lead, t in enumerate(months, start=1):
        m = t.month
        rows = cells.assign(time=t, month=m).merge(clim, on=["month", "latitude", "longitude"])
        normal = regional_normal[m]          # same regional normal the server shows
        have_c3s = (inputs is not None and lead <= C3S_MONTHS and (inputs.time == t).any())
        if have_c3s:
            rows = rows.merge(inputs[inputs.time == t].drop(columns="time"),
                              on=["latitude", "longitude"], how="left")
            rows[[f"{v}_anom" for v in ANOM_VARS]] = rows[[f"{v}_anom" for v in ANOM_VARS]].fillna(0.0)
            rows["ssh_anom"] -= ssh_trend["slope_m_per_year"] * years_since_2000(t) + ssh_trend["intercept_m"]
            rows["ssh_hp_anom"] = rows["ssh_anom"] - ssh_trend["recent_slow_m"]
            anom = float(model.predict(rows[feats]).mean())
            # Band starts at the model's own error and widens toward the climatology spread
            frac = (lead - 1) / C3S_MONTHS
            sigma = sig_model[m] + (sig_clim[m] - sig_model[m]) * frac
            source = "model"
        else:
            anom, sigma, source = 0.0, sig_clim[m], "climatology"
        mld = normal + anom
        out.append({"month": t.strftime("%Y-%m"), "lead_months": lead,
                    "mld": round(mld, 2), "mld_anomaly": round(anom, 2),
                    "lower": round(max(mld - BAND_Z * sigma, 0.0), 2),
                    "upper": round(mld + BAND_Z * sigma, 2),
                    "normal": round(normal, 2), "source": source})
    return out


# ======================================================================================
# STEP 7 - Run everything and save
# ======================================================================================
def main():
    fs = connect()
    df, last_observed, clim, regional_normal = build_table(fs)
    df, clim, ssh_trend = add_anomalies(df, clim)
    df = label_enso(df, load_enso(fs))

    (set_name, model_name), comparison, test_scores, reg_test = compare_and_test(df)
    sig_model, sig_clim = band_sigmas(df, reg_test)

    # Final model for the live forecast: all years, chosen inputs and model type
    feats = feats_for(set_name)
    final = fit(make_models(feats)[model_name], df, feats)
    months = forecast(final, feats, clim, regional_normal, ssh_trend, last_observed,
                      load_forecast_inputs(fs), sig_model, sig_clim)

    result = {
        "issued": pd.Timestamp.today().strftime("%Y-%m"),
        "model": f"mld_{model_name}_{set_name}_v2",
        "inputs": [f for f in feats if f.endswith("_anom")],
        "region": {"south": LAT[0], "north": LAT[1], "west": LON[0], "east": LON[1]},
        "last_observed_month": last_observed.strftime("%Y-%m"),
        "band": "80% range",
        "skill_test_2019_2026": test_scores,
        "rolling_validation": comparison.to_dict(orient="records"),
        "months": months,
    }
    target = f"{OUT_DIR}/forecast_mld.json"
    if fs.exists(target):   # never silently replace someone else's run: keep a dated copy
        backup = f"{OUT_DIR}/backups/forecast_mld_{pd.Timestamp.now():%Y%m%d-%H%M%S}.json"
        fs.copy(target, backup)
        print(f"Existing forecast_mld.json backed up to {backup}")
    with fs.open(target, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {OUT_DIR}/forecast_mld.json")
    print(pd.DataFrame(months).to_string(index=False))
    return result


if __name__ == "__main__":
    main()
