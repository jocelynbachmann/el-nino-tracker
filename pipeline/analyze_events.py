# Historical El Nino analysis on the DAILY ARMOR3D surface data (2002-2026, 46-52N, 134-120W).
#
# For every map point, each variable is compared only with its OWN history:
#   1. anomalies     value minus that point's normal seasonal cycle (and minus its linear trend for sea level)
#   2. change points where the point's weekly anomaly series shifts to a new level (PELT, squared-error cost -
#                    the same method as ruptures' Pelt(model="l2"), written in numpy so there's nothing to install)
#   3. spells        runs of segments that sit well above / below normal ("a warm spell from Jun 2015 to Sep 2016")
#   4. event effects for each El Nino / La Nina (NOAA ONI): did this point have a spell, when did it start
#                    relative to the event, how long did it last, how unusual did it get
#   5. patterns      EOFs (PCA) of monthly anomalies across the region, and how each pattern lines up with the ONI;
#                    plus a per-point lag map (how many months after the ONI each point responds most)
#
# Outputs (OUT_DIR):
#   events.json              ONI events + regional summaries per variable + main patterns
#   spells.csv               every unusual spell at every point (start, end, size, which ENSO event it overlaps)
#   event_effects.zarr       per event x variable x point: onset, duration, peak, mean anomaly, typical El Nino maps
#   patterns.zarr            EOF maps, PC time series, ONI lag-correlation maps
#   anomalies_weekly.zarr    weekly anomaly series per point (for the click-a-point chart)
#
# Run on the hub or a laptop with the hack-team AWS profile:  python analyze_events.py
# Needs: numpy pandas xarray h5netcdf s3fs zarr

# %% SETTINGS
import json, os, time, warnings
import numpy as np
import pandas as pd

INPUT = os.environ.get("ELNINO_DAILY",
    "s3://ocean-hackathon-2026/teams/team-elnino/processed/daily/cmems_obs-mob_glo_phy_P1D-m_2002-2026_combined.nc")
OUT_DIR = os.environ.get("ELNINO_ANALYSIS",
    "s3://ocean-hackathon-2026/teams/team-elnino/processed/analysis")
ENSO_JSON = os.environ.get("ELNINO_ENSO",
    "s3://ocean-hackathon-2026/teams/team-elnino/processed/enso.json")      # written by preprocess.ipynb
ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"       # fallback if enso.json is missing

# Hub: key is in env vars -> {}. Laptop: key is in the "hack-team" profile (server/setup_aws.py).
STORAGE = ({} if os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE")
           else {"profile": os.environ.get("AWS_PROFILE", "hack-team")})

REGION = dict(latitude=(46, 52), longitude=(-134, -120))
VARS = {   # code in file -> slug used by the API, units, remove linear trend?
    "to":     dict(slug="temperature",        units="°C",  detrend=False),
    "so":     dict(slug="salinity",           units="PSU", detrend=False),
    "zo":     dict(slug="sea-surface-height", units="m",   detrend=True),   # sea-level rise would look like one long "spell"
    "mlotst": dict(slug="mixed-layer-depth",  units="m",   detrend=False),
}
BASE_YEARS = (2003, 2022)    # "normal" = these 20 full years
HARMONICS = 3                # seasonal cycle = mean + 3 annual harmonics (smooth, no day-to-day noise)

CPD_PENALTY = 2.0            # x log(n) on unit-variance weekly anomalies. Higher -> fewer, bigger change points
CPD_MIN_WEEKS = 8            # shortest segment between change points (~2 months)
UNUSUAL_SD = 0.75            # a segment whose mean is beyond +/-0.75 SD of the point's own variability is "unusual"
MAX_RESPONSE_LAG = 6         # each point's event window = the event's months shifted by that point's own ONI lag
                             # (best lag 0..6 months from the lag map), since the ocean here responds late
EVENT_RANK = 0.10            # affected = the 6 months around the (lagged) peak are more extreme than 90% of
PEAK_WINDOW_WEEKS = 26       #   this point's 6-month periods in ENSO-neutral times
CLEAR_EFFECT = 0.20          # an event "clearly" affected a variable if >= 20% of points changed the same way
                             # (by chance ~1 point in 10 passes the test, so less than this is noise)
N_MODES = 3                  # EOF patterns to keep
MAX_LAG = 12                 # months of ONI lead tested in the lag maps


# %% CORE (pure numpy/pandas - no I/O)
def pelt_l2(x, pen, min_size):
    """Optimal change points for a piecewise-constant mean (PELT, squared-error cost).
    Returns segment end indices, last one = len(x). Same result as ruptures.Pelt(model="l2", min_size, jump=1)."""
    n = len(x)
    if n < 2 * min_size:
        return [n]
    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(x * x)])
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    last = np.zeros(n + 1, dtype=np.int64)
    R = np.array([0], dtype=np.int64)              # candidate start points
    K = np.array([n + 1], dtype=np.int64)          # last t each candidate may still be needed for
    for t in range(min_size, n + 1):
        alive = K >= t
        R, K = R[alive], K[alive]
        rd = t - R >= min_size
        ready = R[rd]
        cost = F[ready] + (s2[t] - s2[ready]) - (s1[t] - s1[ready]) ** 2 / (t - ready)
        k = int(np.argmin(cost))
        F[t] = cost[k] + pen
        last[t] = ready[k]
        # PELT pruning: a start that already loses at t can't win once a full segment fits after t,
        # i.e. from t + min_size on. Keep it until then (dropping it at t is wrong with a minimum size).
        Kr = K[rd]
        Kr[cost > F[t]] = np.minimum(Kr[cost > F[t]], t + min_size - 1)
        K[rd] = Kr
        R, K = np.concatenate([R, [t]]), np.concatenate([K, [n + 1]])
    bkps, t = [], n
    while t > 0:
        bkps.append(int(t))
        t = int(last[t])
    return bkps[::-1]


def spells_from_segments(z, bkps, thresh):
    """Merge consecutive unusual segments of the same sign into spells. z is unit-variance."""
    out, cur, s = [], None, 0
    for e in bkps:
        mu = float(np.mean(z[s:e]))
        sign = 1 if mu >= thresh else -1 if mu <= -thresh else 0
        if sign and cur is not None and cur["sign"] == sign:
            cur["end"] = e
        elif sign:
            cur = dict(start=s, end=e, sign=sign)
            out.append(cur)
        else:
            cur = None
        s = e
    for sp in out:
        seg = z[sp["start"]:sp["end"]]
        sp["mean_z"] = float(seg.mean())
        sp["peak"] = sp["start"] + int(np.argmax(sp["sign"] * seg))
        sp["peak_z"] = float(z[sp["peak"]])
    return out


NORMAL_ORIGIN = pd.Timestamp("2000-01-01")   # time axis of the normal's design matrix (years since)
TREND_ORIGIN_YEARS = 12.0                    # trend term is (years since 2000) - 12, i.e. centred on 2012


def normal_terms(harmonics=HARMONICS):
    return ["mean"] + [f"{f}{k}" for k in range(1, harmonics + 1) for f in ("cos", "sin")] + ["trend"]


def normal_design(t, harmonics=HARMONICS):
    """Columns of the 'normal': mean, annual harmonics, linear trend. Shared with the server, which
    rebuilds the normal for any day as normal_design(day) @ coefficients."""
    yrs = ((t - NORMAL_ORIGIN).days.values / 365.2425).astype(np.float64)
    cols = [np.ones_like(yrs)]
    for k in range(1, harmonics + 1):
        cols += [np.cos(2 * np.pi * k * yrs), np.sin(2 * np.pi * k * yrs)]
    cols.append(yrs - TREND_ORIGIN_YEARS)
    return np.stack(cols, 1)


def fit_normal(Y, t, detrend, oni_daily=None, base_years=BASE_YEARS, harmonics=HARMONICS):
    """Fit each point's normal (seasonal cycle, + linear trend if detrend) on the base years.
    The ONI (at several lags) is fitted alongside but is NOT part of the normal, so El Ninos and
    La Ninas that happen to fall early or late in the base years don't leak into the trend or the
    seasonal cycle. Returns coefficients (terms, points); the trend row is 0 when detrend is False."""
    X = normal_design(t, harmonics)
    if not detrend:
        X = X[:, :-1]
    n_keep = X.shape[1]
    if oni_daily is not None:
        X = np.concatenate([X, np.stack(list(oni_daily), 1)], 1)
    base = (t.year >= base_years[0]) & (t.year <= base_years[1]) & np.isfinite(X).all(1)
    rows = base & (np.isfinite(Y).mean(1) >= 0.99)                # skip days with gaps across the map
    Yb = Y[rows].astype(np.float64)
    Yb = np.where(np.isfinite(Yb), Yb, np.nanmean(Yb, 0))        # the odd missing point -> its mean
    coef, *_ = np.linalg.lstsq(X[rows], Yb, rcond=None)
    coef = coef[:n_keep]
    if not detrend:
        coef = np.vstack([coef, np.zeros((1, coef.shape[1]))])
    return coef


def seasonal_anomalies(Y, t, detrend, oni_daily=None, base_years=BASE_YEARS, harmonics=HARMONICS):
    """Y (time, points) -> (anomalies from each point's normal, trend per decade or None)."""
    coef = fit_normal(Y, t, detrend, oni_daily, base_years, harmonics)
    A = (Y - normal_design(t, harmonics) @ coef).astype(np.float32)
    return A, (coef[-1] * 10 if detrend else None)


def oni_on_days(oni_monthly, t, lags=(0, 3, 6)):
    """Monthly ONI -> daily series at several lags (months), for the seasonal fit."""
    out = []
    for L in lags:
        src = (t - pd.DateOffset(months=L)).to_period("M").to_timestamp()
        out.append(oni_monthly.reindex(src).to_numpy(np.float64))
    return out


def window_means(Z, length):
    """Means of every run of `length` consecutive rows (rows, points)."""
    c = np.concatenate([np.zeros((1, Z.shape[1])), np.cumsum(Z, 0)])
    return (c[length:] - c[:-length]) / length


def block_mean(A, keys):
    """Mean of rows sharing a key (keys sorted, contiguous). NaN-aware. Returns (unique keys, means)."""
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    fin = np.isfinite(A)
    s = np.add.reduceat(np.where(fin, A, 0), starts, axis=0)
    c = np.add.reduceat(fin, starts, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
    m[c < np.maximum(1, 0.5 * np.diff(np.r_[starts, len(keys)]))[:, None]] = np.nan   # need half the days
    return keys[starts], m.astype(np.float32)


def fill_gaps(Z):
    return pd.DataFrame(Z).interpolate(limit_direction="both", axis=0).to_numpy(np.float64)


def standardize(M):
    mu = np.nanmean(M, 0)
    sd = np.nanstd(M, 0)
    return (M - mu) / np.where(sd > 0, sd, np.nan), sd


def eof(Mz, lat_w, n_modes):
    """EOFs of unit-variance monthly anomalies (months, points), area-weighted. Returns patterns as
    regression maps (SD of the point per SD of the PC), standardized PCs and variance fractions."""
    F = np.nan_to_num(Mz) * lat_w[None, :]
    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    frac = S ** 2 / np.sum(S ** 2)
    pcs = U[:, :n_modes] * S[:n_modes]
    pcs = (pcs - pcs.mean(0)) / pcs.std(0)
    pats = np.nan_to_num(Mz).T @ pcs / len(pcs)
    return pats.T, pcs.T, frac[:n_modes]


def lag_corr(Mz, oni, max_lag):
    """Correlation of each point (months, points) with the ONI L months earlier, L = 0..max_lag."""
    out = np.full((max_lag + 1, Mz.shape[1]), np.nan)
    for L in range(max_lag + 1):
        a = Mz[L:] if L else Mz
        b = oni[:len(oni) - L] if L else oni
        ok = np.isfinite(b)
        a, b = a[ok], b[ok]
        a = (a - np.nanmean(a, 0)) / np.nanstd(a, 0)
        b = (b - b.mean()) / b.std()
        out[L] = np.nanmean(a * b[:, None], 0)
    return out


def find_events(oni, thresh=0.5, run=5):
    """NOAA ONI events: >= +0.5 (El Nino) or <= -0.5 (La Nina) for >= 5 consecutive overlapping seasons.
    A run still going at the end of the record is kept as 'ongoing' (and 'provisional' if < 5 seasons)."""
    events = []
    for sign, phase in ((1, "el_nino"), (-1, "la_nina")):
        hit = (sign * oni) >= thresh
        grp = (hit != hit.shift()).cumsum()
        for _, months in oni[hit].groupby(grp[hit]):
            ongoing = months.index[-1] == oni.index[-1]
            if len(months) < run and not ongoing:
                continue
            pk = (sign * months).idxmax()
            a = abs(float(months[pk]))
            y0, y1 = months.index[0].year, months.index[-1].year
            events.append(dict(
                id=f"{phase.replace('_', '-')}-{y0}", phase=phase,
                label=(f"{y0}" if y0 == y1 else f"{y0}–{str(y1)[2:]}") + (" (ongoing)" if ongoing else ""),
                start=months.index[0], end=months.index[-1] + pd.offsets.MonthEnd(0),
                peak_month=pk, peak_oni=round(float(months[pk]), 2),
                strength="very strong" if a >= 2 else "strong" if a >= 1.5 else "moderate" if a >= 1 else "weak",
                ongoing=bool(ongoing), provisional=bool(ongoing and len(months) < run)))
    return sorted(events, key=lambda e: e["start"])


def analyse_variable(Y, t, lat_of_point, detrend, events, oni_monthly):
    """Everything for one variable. Y: (days, ocean points). Returns a dict of arrays / tables."""
    A, trend = seasonal_anomalies(Y, t, detrend, oni_on_days(oni_monthly, t))

    # weekly anomalies (for change points and the point chart)
    wk = (t - t[0]).days.values // 7
    wkeys, W = block_mean(A, wk)
    wdates = t[0] + pd.to_timedelta(wkeys * 7, unit="D")
    Wz, sd_w = standardize(W)
    Wf = fill_gaps(Wz)

    # monthly anomalies (for patterns and lag maps)
    mkeys, M = block_mean(A, (t.year * 12 + t.month - 1).values)
    mdates = pd.DatetimeIndex(pd.to_datetime(dict(year=mkeys // 12, month=mkeys % 12 + 1, day=1)))
    Mz, _ = standardize(M)

    # change points + spells at every point
    n = len(Wf)
    pen = CPD_PENALTY * np.log(n)
    spells = []
    for j in range(Wf.shape[1]):
        bk = pelt_l2(Wf[:, j], pen, CPD_MIN_WEEKS)
        for sp in spells_from_segments(Wf[:, j], bk, UNUSUAL_SD):
            sp["point"] = j
            spells.append(sp)

    # regional mean (area-weighted) - one series for the whole box
    w = np.cos(np.deg2rad(lat_of_point))
    reg = np.nansum(W * w, 1) / np.sum(np.isfinite(W) * w, 1)
    reg_z = fill_gaps(standardize(reg[:, None])[0])[:, 0]
    reg_spells = spells_from_segments(reg_z, pelt_l2(reg_z, pen, CPD_MIN_WEEKS), UNUSUAL_SD)

    # how many months after the ONI does each point respond most? (lag map, ONI leading by 0..MAX_LAG months)
    oni_m = oni_monthly.reindex(mdates).to_numpy(np.float64)
    lc_map = lag_corr(Mz, oni_m, MAX_LAG)
    best = np.nanargmax(np.abs(np.nan_to_num(lc_map, nan=0)), 0)
    resp_lag = np.nanargmax(np.abs(np.nan_to_num(lc_map[:MAX_RESPONSE_LAG + 1], nan=0)), 0)

    # event effects. A point counts as affected by an event when BOTH:
    #   (a) its mean anomaly over the 6 months around the event's peak (shifted by the point's response lag)
    #       is more extreme than (1 - EVENT_RANK) of its own 6-month periods in ENSO-neutral times, and
    #   (b) a change-point spell with the same sign overlaps the event's (lagged) months - this gives
    #       the onset, end, duration and peak.
    P = Wf.shape[1]
    E = len(events)
    eff = {k: np.full((E, P), np.nan, np.float32)
           for k in ("detected", "sign", "onset_days", "duration_days", "peak_z", "mean_anom",
                     "event_mean_z", "rank_pct")}
    by_point = {}
    for sp in spells:
        by_point.setdefault(sp["point"], []).append(sp)
    wend = wdates + pd.Timedelta(days=6)
    nw, H = len(wdates), PEAK_WINDOW_WEEKS
    lags = sorted(set(resp_lag.tolist()))
    for L in lags:
        pts = np.flatnonzero(resp_lag == L)
        allw = window_means(Wf[:, pts], H)                               # every 6-month window
        centre = np.arange(len(allw)) + H // 2
        src = (wdates[centre] - pd.DateOffset(months=L)).to_period("M").to_timestamp()
        neutral = np.abs(oni_monthly.reindex(src).to_numpy()) < 0.5        # ONI neutral L months earlier
        null = np.abs(allw[neutral])
        if len(null) < 20:                                                 # not enough neutral periods to compare
            continue
        for e, ev in enumerate(events):
            if L == lags[0]:
                eff["detected"][e] = 0
            pk = ev["peak_month"] + pd.DateOffset(months=L)
            c = int(np.searchsorted(wdates, pk + pd.Timedelta(days=14)))
            i0 = min(max(c - H // 2, 0), nw - H)
            if c >= nw:                                                     # peak (+lag) not in the data yet:
                eff["detected"][e, pts] = np.nan                            # too early to tell
                continue
            m = allw[i0]
            rank = (null < np.abs(m)).mean(0) * 100
            eff["event_mean_z"][e, pts] = m
            eff["rank_pct"][e, pts] = rank
            w0 = ev["start"] + pd.DateOffset(months=L)
            w1 = ev["end"] + pd.DateOffset(months=L)
            for jj in np.flatnonzero(rank >= 100 * (1 - EVENT_RANK)):
                j = pts[jj]
                sg = 1 if m[jj] > 0 else -1
                best_sp, score = None, 0
                for sp in by_point.get(j, []):
                    if sp["sign"] == sg and wdates[sp["start"]] <= w1 and wend[sp["end"] - 1] >= w0:
                        sc = abs(sp["mean_z"]) * (sp["end"] - sp["start"])
                        if sc > score:
                            best_sp, score = sp, sc
                if best_sp is None:
                    continue
                eff["detected"][e, j] = 1
                eff["sign"][e, j] = sg
                eff["onset_days"][e, j] = (wdates[best_sp["start"]] - ev["start"]).days
                eff["duration_days"][e, j] = 7 * (best_sp["end"] - best_sp["start"])
                eff["peak_z"][e, j] = best_sp["peak_z"]
                eff["mean_anom"][e, j] = np.nanmean(W[best_sp["start"]:best_sp["end"], j])

    typical = {}
    for phase in ("el_nino", "la_nina"):
        idx = [i for i, ev in enumerate(events) if ev["phase"] == phase and not ev["ongoing"]]
        mean_z = np.full(P, np.nan)
        for L in lags:
            pts = np.flatnonzero(resp_lag == L)
            mask = np.zeros(len(wdates), bool)
            for i in idx:
                mask |= (wdates >= events[i]["start"] + pd.DateOffset(months=L)) & \
                        (wdates <= events[i]["end"] + pd.DateOffset(months=L))
            if mask.any():
                mean_z[pts] = np.nanmean(Wz[mask][:, pts], 0)
        same = np.zeros(P)
        for i in idx:
            same += (eff["detected"][i] == 1) & (eff["sign"][i] == np.sign(mean_z))
        typical[phase] = dict(mean_z=mean_z.astype(np.float32), n_same_sign=same.astype(np.float32), n_events=len(idx))

    # patterns
    pats, pcs, frac = eof(Mz, np.sqrt(np.cos(np.deg2rad(lat_of_point))), N_MODES)
    modes = []
    for k in range(len(frac)):
        lc = lag_corr(pcs[k][:, None], oni_m, MAX_LAG)[:, 0]
        L = int(np.nanargmax(np.abs(lc)))
        if lc[L] < 0:                                   # orient so the pattern goes "with" El Nino
            pats[k], pcs[k], lc = -pats[k], -pcs[k], -lc
        pc_spells = spells_from_segments(pcs[k], pelt_l2(pcs[k], CPD_PENALTY * np.log(len(pcs[k])), 3), UNUSUAL_SD)
        modes.append(dict(mode=k + 1, variance_pct=round(100 * float(frac[k]), 1),
                          oni_corr=round(float(lc[L]), 2), oni_lag_months=L,
                          shifts=[dict(start=mdates[s["start"]].strftime("%Y-%m"),
                                       end=mdates[s["end"] - 1].strftime("%Y-%m"),
                                       sign="above" if s["sign"] > 0 else "below",
                                       mean_sd=round(s["mean_z"], 2)) for s in pc_spells]))
    return dict(A=A, trend=trend, W=W, Wz=Wz, sd_w=sd_w, wdates=wdates, mdates=mdates,
                spells=spells, reg_spells=reg_spells, eff=eff, typical=typical,
                pats=pats, pcs=pcs, modes=modes, lag_map=lc_map,
                best_lag=best.astype(np.float32), best_corr=lc_map[best, np.arange(lc_map.shape[1])],
                resp_lag=resp_lag.astype(np.float32))


def load_oni(fs=None):
    """Monthly ONI: our enso.json (written by preprocess.ipynb), or NOAA directly if that's missing."""
    try:
        if ENSO_JSON.startswith("s3://"):
            with fs.open(ENSO_JSON) as f:
                rec = json.load(f)
        else:
            with open(ENSO_JSON) as f:
                rec = json.load(f)
        print("ONI from", ENSO_JSON)
        return pd.Series({pd.Timestamp(r["month"] + "-01"): r["oni"] for r in rec}).sort_index()
    except Exception as e:
        print(f"enso.json not readable ({e}); reading NOAA")
        raw = pd.read_csv(ONI_URL, sep=r"\s+")
        mid = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
               "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
        raw["time"] = pd.to_datetime(dict(year=raw["YR"], month=raw["SEAS"].map(mid), day=1))
        return raw.set_index("time")["ANOM"].sort_index()


def open_daily(fs=None):
    """The daily ARMOR3D file, cropped to REGION, sorted, one record per day."""
    import xarray as xr
    if INPUT.startswith("s3://"):
        fobj = fs.open(INPUT, "rb", block_size=32 * 2 ** 20, cache_type="blockcache")
        ds = xr.open_dataset(fobj, engine="h5netcdf")
    else:
        ds = xr.open_dataset(INPUT)
    ds = ds.rename({k: v for k, v in {"lat": "latitude", "lon": "longitude"}.items() if k in ds.dims})
    if "depth" in ds.dims:
        ds = ds.isel(depth=0, drop=True)
    ds = ds.sortby("latitude").sortby("longitude")
    ds = ds.sel(latitude=slice(*REGION["latitude"]), longitude=slice(*REGION["longitude"]))
    ds = ds.assign_coords(time=pd.to_datetime(ds.time.values).floor("D"))
    return ds.sel(time=~ds.get_index("time").duplicated(keep="last")).sortby("time")


# %% I/O + RUN
def main():
    import s3fs
    import xarray as xr
    warnings.filterwarnings("ignore", message=".*consolidated.*")
    on_s3 = lambda p: p.startswith("s3://")
    fs = s3fs.S3FileSystem(**STORAGE) if any(map(on_s3, (INPUT, OUT_DIR, ENSO_JSON))) else None

    def put_json(obj, name):
        path = f"{OUT_DIR}/{name}"
        if on_s3(path):
            with fs.open(path, "w") as f:
                json.dump(obj, f, indent=1, default=str)
        else:
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(path, "w") as f:
                json.dump(obj, f, indent=1, default=str)

    def put_csv(df, name):
        path = f"{OUT_DIR}/{name}"
        if on_s3(path):
            with fs.open(path, "w") as f:
                df.to_csv(f, index=False)
        else:
            os.makedirs(OUT_DIR, exist_ok=True)
            df.to_csv(path, index=False)

    def put_zarr(ds, name):
        path = f"{OUT_DIR}/{name}"
        for v in ds.variables:
            ds[v].encoding = {}
        ds.to_zarr(path, mode="w", consolidated=True, zarr_format=2,
                   **({"storage_options": STORAGE} if on_s3(path) else {}))
        print("  wrote", path)

    # ---- open the daily file
    t0 = time.time()
    ds = open_daily(fs)
    t = pd.DatetimeIndex(ds.time.values)
    steps = pd.Series(np.diff(t)).value_counts()
    print(f"Input: {ds.sizes['time']} days {t[0].date()} -> {t[-1].date()}, "
          f"{ds.sizes['latitude']} x {ds.sizes['longitude']} points")
    print("  time steps:", {str(k): int(v) for k, v in steps.head(3).items()})
    print("  missing days:", len(pd.date_range(t[0], t[-1], freq="D").difference(t)))
    codes = [c for c in VARS if c in ds.data_vars]
    missing = [c for c in VARS if c not in ds.data_vars]
    print("  variables:", codes, "| not found:", missing, "| also in file:", [v for v in ds.data_vars if v not in VARS])
    if steps.index[0] != pd.Timedelta(days=1):
        raise SystemExit("This doesn't look like daily data - stopping.")

    # ---- ENSO
    oni = load_oni(fs)
    events = [e for e in find_events(oni) if e["end"] >= t[0] and e["start"] <= t[-1]]
    print(f"ONI to {oni.index[-1]:%Y-%m}; events in the data period:")
    for ev in events:
        print(f"  {ev['id']:>16}  {ev['start']:%Y-%m} -> {ev['end']:%Y-%m}  peak {ev['peak_oni']:+.1f} ({ev['strength']})"
              + ("  ONGOING" if ev["ongoing"] else ""))

    lat, lon = ds.latitude.values, ds.longitude.values
    ny, nx = len(lat), len(lon)
    LAT2 = np.repeat(lat, nx)
    results, ocean_masks = {}, {}
    for code in codes:
        cfg = VARS[code]
        t1 = time.time()
        Y = ds[code].astype("float32").transpose("time", "latitude", "longitude").values.reshape(len(t), -1)
        ocean = np.isfinite(Y).mean(0) > 0.9
        r = analyse_variable(Y[:, ocean], t, LAT2[ocean], cfg["detrend"], events, oni)
        results[code], ocean_masks[code] = r, ocean
        msg = f"{cfg['slug']:>19}: {ocean.sum()} ocean points, {len(r['spells'])} spells, " \
              f"EOF1 {r['modes'][0]['variance_pct']}% (ONI r={r['modes'][0]['oni_corr']}, lag {r['modes'][0]['oni_lag_months']} mo)"
        if r["trend"] is not None:
            msg += f", trend removed {np.nanmedian(r['trend']) * 100:.1f} cm/decade"
        print(msg + f"  [{time.time() - t1:.0f}s]")
        del Y

    # ---- grids
    def grid(vec, code):
        g = np.full(ny * nx, np.nan, np.float32)
        g[ocean_masks[code]] = vec
        return g.reshape(ny, nx)

    slugs = [VARS[c]["slug"] for c in codes]
    ev_ids = [e["id"] for e in events]
    coords = dict(variable=slugs, latitude=lat, longitude=lon)

    # ---- event_effects.zarr
    fields = ["detected", "sign", "onset_days", "duration_days", "peak_z", "mean_anom", "event_mean_z", "rank_pct"]
    eff = {f: (("variable", "event", "latitude", "longitude"),
               np.stack([np.stack([grid(results[c]["eff"][f][e], c) for e in range(len(events))]) for c in codes]))
           for f in fields}
    for phase in ("el_nino", "la_nina"):
        eff[f"{phase}_mean_z"] = (("variable", "latitude", "longitude"),
                                  np.stack([grid(results[c]["typical"][phase]["mean_z"], c) for c in codes]))
        eff[f"{phase}_n_same_sign"] = (("variable", "latitude", "longitude"),
                                       np.stack([grid(results[c]["typical"][phase]["n_same_sign"], c) for c in codes]))
    eff_ds = xr.Dataset(eff, coords=dict(coords, event=ev_ids))
    eff_ds.attrs.update(
        description="Each point's event window is shifted by its own ONI response lag (patterns.zarr response_lag_months). "
                    "event_mean_z = mean anomaly over the 6 months around the (lagged) ONI peak; rank_pct = % of that "
                    "point's 6-month periods in ENSO-neutral times that were less extreme. detected = rank_pct >= "
                    f"{100 * (1 - EVENT_RANK):.0f} AND a same-direction change-point spell overlaps the window; onset, "
                    "duration and peak come from that spell. onset_days < 0 means the point was already unusual "
                    "before the event began. z / sd = SD of that point's own weekly anomalies.",
        n_el_nino=results[codes[0]]["typical"]["el_nino"]["n_events"],
        n_la_nina=results[codes[0]]["typical"]["la_nina"]["n_events"],
        units=json.dumps({VARS[c]["slug"]: VARS[c]["units"] for c in codes}))
    put_zarr(eff_ds, "event_effects.zarr")

    # ---- patterns.zarr
    months = results[codes[0]]["mdates"]
    pat_ds = xr.Dataset(dict(
        eof=(("variable", "mode", "latitude", "longitude"),
             np.stack([np.stack([grid(p, c) for p in results[c]["pats"]]) for c in codes])),
        pc=(("variable", "mode", "time"), np.stack([results[c]["pcs"] for c in codes]).astype(np.float32)),
        variance_pct=(("variable", "mode"), np.array([[m["variance_pct"] for m in results[c]["modes"]] for c in codes])),
        oni_corr=(("variable", "mode"), np.array([[m["oni_corr"] for m in results[c]["modes"]] for c in codes])),
        oni_lag_months=(("variable", "mode"), np.array([[m["oni_lag_months"] for m in results[c]["modes"]] for c in codes])),
        oni_lag_corr=(("variable", "lag", "latitude", "longitude"),
                      np.stack([np.stack([grid(results[c]["lag_map"][L], c) for L in range(MAX_LAG + 1)]) for c in codes])),
        best_lag_months=(("variable", "latitude", "longitude"), np.stack([grid(results[c]["best_lag"], c) for c in codes])),
        best_lag_corr=(("variable", "latitude", "longitude"), np.stack([grid(results[c]["best_corr"], c) for c in codes])),
        response_lag_months=(("variable", "latitude", "longitude"), np.stack([grid(results[c]["resp_lag"], c) for c in codes])),
    ), coords=dict(coords, mode=np.arange(1, N_MODES + 1), time=months.values, lag=np.arange(MAX_LAG + 1)))
    pat_ds.attrs["description"] = ("EOFs of monthly anomalies (each point scaled by its own SD, area-weighted). "
                                   "eof = SD change at each point per +1 SD of the pattern's time series (pc). "
                                   "Patterns are oriented so they correlate positively with the ONI.")
    put_zarr(pat_ds, "patterns.zarr")

    # ---- anomalies_weekly.zarr
    wdates = results[codes[0]]["wdates"]
    an_ds = xr.Dataset(dict(
        anomaly=(("variable", "time", "latitude", "longitude"),
                 np.stack([np.stack([grid(row, c) for row in results[c]["W"]]) for c in codes])),
        sd=(("variable", "latitude", "longitude"), np.stack([grid(results[c]["sd_w"], c) for c in codes])),
    ), coords=dict(coords, time=wdates.values))
    an_ds.attrs.update(description=f"Weekly mean anomaly from each point's {BASE_YEARS[0]}-{BASE_YEARS[1]} seasonal "
                                   "cycle (sea level also detrended). sd = SD of that point's weekly anomalies.",
                       units=json.dumps({VARS[c]["slug"]: VARS[c]["units"] for c in codes}))
    put_zarr(an_ds.chunk({"time": -1, "latitude": 8, "longitude": 8}), "anomalies_weekly.zarr")

    # ---- spells.csv
    rows = []
    for c in codes:
        r = results[c]
        pts = np.flatnonzero(ocean_masks[c])
        wd, wend = r["wdates"], r["wdates"] + pd.Timedelta(days=6)
        for sp in r["spells"]:
            flat = pts[sp["point"]]
            s0, s1 = wd[sp["start"]], wend[sp["end"] - 1]
            L = int(r["resp_lag"][sp["point"]])
            overlaps = [ev["id"] for ev in events
                        if s0 <= ev["end"] + pd.DateOffset(months=L) and s1 >= ev["start"] + pd.DateOffset(months=L)]
            rows.append(dict(variable=VARS[c]["slug"], lat=float(lat[flat // nx]), lon=float(lon[flat % nx]),
                             start=s0.date(), end=s1.date(), days=(s1 - s0).days + 1,
                             direction="above" if sp["sign"] > 0 else "below",
                             mean_sd=round(sp["mean_z"], 2), peak_sd=round(sp["peak_z"], 2),
                             peak_week=wd[sp["peak"]].date(),
                             mean_anomaly=round(float(np.nanmean(r["W"][sp["start"]:sp["end"], sp["point"]])), 4),
                             units=VARS[c]["units"], enso_events=";".join(overlaps),
                             ongoing=bool(sp["end"] == len(wd))))
    spells_df = pd.DataFrame(rows)
    put_csv(spells_df, "spells.csv")
    print(f"  wrote {OUT_DIR}/spells.csv ({len(spells_df)} spells)")

    # ---- events.json
    def fmt_spell(s, wd):
        return dict(start=f"{wd[s['start']]:%Y-%m-%d}", end=f"{wd[s['end'] - 1] + pd.Timedelta(days=6):%Y-%m-%d}",
                    direction="above" if s["sign"] > 0 else "below", mean_sd=round(s["mean_z"], 2),
                    peak_sd=round(s["peak_z"], 2))

    out_events = []
    for e, ev in enumerate(events):
        effects = {}
        for c in codes:
            r = results[c]
            d = r["eff"]["detected"][e] == 1
            sg = r["eff"]["sign"][e][d]
            maj = 1 if (sg > 0).sum() >= (sg < 0).sum() else -1
            same = d & (r["eff"]["sign"][e] == maj)
            L = int(np.median(r["resp_lag"]))
            w0, w1 = ev["start"] + pd.DateOffset(months=L), ev["end"] + pd.DateOffset(months=L)
            reg = [s for s in r["reg_spells"]
                   if r["wdates"][s["start"]] <= w1 and r["wdates"][s["end"] - 1] >= w0]
            reg = max(reg, key=lambda s: abs(s["mean_z"]) * (s["end"] - s["start"])) if reg else None
            med = lambda a: None if not same.any() else round(float(np.nanmedian(a[same])), 1)
            if np.isnan(r["eff"]["detected"][e]).all():
                effects[VARS[c]["slug"]] = dict(too_early=True, units=VARS[c]["units"])
                continue
            effects[VARS[c]["slug"]] = dict(
                too_early=False,
                pct_points_changed=round(100 * float(d.mean()), 1),
                clear_effect=bool(same.mean() >= CLEAR_EFFECT),       # well above the ~10% false-alarm rate
                main_direction=None if not d.any() else "above" if maj > 0 else "below",
                pct_points_main_direction=round(100 * float(same.mean()), 1),
                median_onset_days=med(r["eff"]["onset_days"][e]),
                median_duration_days=med(r["eff"]["duration_days"][e]),
                median_peak_sd=med(r["eff"]["peak_z"][e]),
                median_anomaly=None if not same.any() else round(float(np.nanmedian(r["eff"]["mean_anom"][e][same])), 4),
                median_rank_pct=round(float(np.nanmedian(r["eff"]["rank_pct"][e])), 1),
                typical_response_lag_months=int(np.median(r["resp_lag"])),
                units=VARS[c]["units"],
                regional_spell=fmt_spell(reg, r["wdates"]) if reg else None)
        out_events.append(dict({k: (f"{v:%Y-%m}" if isinstance(v, pd.Timestamp) else v) for k, v in ev.items()},
                               effects=effects))
    put_json(dict(
        generated=pd.Timestamp.now().isoformat(timespec="seconds"),
        source=INPUT, data_start=f"{t[0]:%Y-%m-%d}", data_end=f"{t[-1]:%Y-%m-%d}",
        method=dict(base_years=list(BASE_YEARS), seasonal_harmonics=HARMONICS,
                    detrended=[VARS[c]["slug"] for c in codes if VARS[c]["detrend"]],
                    change_points="PELT (squared error) on weekly anomalies in SD units",
                    penalty=f"{CPD_PENALTY} x log(n)", min_segment_weeks=CPD_MIN_WEEKS,
                    unusual_threshold_sd=UNUSUAL_SD,
                    event_window=f"event months shifted by each point's ONI response lag (0-{MAX_RESPONSE_LAG} months)",
                    clear_effect_if=f">= {100 * CLEAR_EFFECT:.0f}% of points changed in the same direction",
                    affected_if=f"6 months around the lagged peak more extreme than {100 * (1 - EVENT_RANK):.0f}% of the "
                                "point's own ENSO-neutral 6-month periods, AND a same-direction spell overlaps the event"),
        events=out_events,
        regional_spells={VARS[c]["slug"]: [fmt_spell(s, results[c]["wdates"]) for s in results[c]["reg_spells"]]
                         for c in codes},
        patterns={VARS[c]["slug"]: results[c]["modes"] for c in codes},
    ), "events.json")
    print(f"  wrote {OUT_DIR}/events.json")

    # ---- quick read-out
    print("\nRegion-wide spells per variable (whole box):")
    for c in codes:
        print(f"  {VARS[c]['slug']}:")
        for s in results[c]["reg_spells"]:
            f = fmt_spell(s, results[c]["wdates"])
            print(f"    {f['start']} -> {f['end']}  {f['direction']:>5}  mean {f['mean_sd']:+.1f} SD")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
