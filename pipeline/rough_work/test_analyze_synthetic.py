# Synthetic check of analyze_events.py (no S3, no xarray needed): plants a lagged El Nino signal of
# increasing strength across 600 fake points and checks that detection, lag, trend and EOF1 recover it.
# Run: python test_analyze_synthetic.py   (~20 s)
import numpy as np, pandas as pd, time
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # analyze_events.py lives in pipeline/
from analyze_events import analyse_variable, find_events
rng=np.random.default_rng(1)
# synthetic ONI with real-ish events
months=pd.date_range("1993-01-01","2026-09-01",freq="MS")
B=[(2002.8,1.3),(2004.8,.7),(2006.9,.9),(2007.9,-1.6),(2009.9,1.6),(2010.9,-1.6),(2011.9,-1),(2015.8,2.6),(2017.9,-.9),(2018.9,.9),(2020.9,-1.3),(2021.9,-1),(2022.8,-1),(2023.9,2),(2026.7,.8)]
tt=months.year+(months.month-.5)/12
oni=pd.Series(sum(a*np.exp(-(tt-c)**2/.35) for c,a in B)+rng.normal(0,.05,len(months)),index=months)
events=[e for e in find_events(oni) if e["end"]>=pd.Timestamp("2002-01-01")]
for e in events: print(e["id"],e["label"],e["start"].date(),e["end"].date(),e["peak_oni"],e["strength"],e["ongoing"],e["provisional"])
# daily data, 600 points; El Nino signal lagged 3 months scaled by g, + seasonal cycle + trend + AR noise
t=pd.date_range("2002-01-01","2026-08-01",freq="D"); nt=len(t); P=600
lat=np.linspace(46,52,P)
g=np.linspace(0,1.5,P)                       # first points: no ENSO effect; last: strong
oni_d=oni.reindex(t-pd.DateOffset(months=3),method="ffill").to_numpy()
doy=t.dayofyear.values
seas=8+4*np.sin(2*np.pi*(doy-120)/365.25)
Y=np.empty((nt,P),np.float32); noise=np.zeros(P)
for i in range(nt):
    noise=0.97*noise+rng.normal(0,.15,P)
    Y[i]=seas[i]+g*0.6*oni_d[i]+noise+0.02*(t[i].year-2012)
Y[5000:5003]=np.nan                          # gap
t0=time.time()
r=analyse_variable(Y,t,lat,True,events,oni)
print(f"analyse_variable: {time.time()-t0:.0f}s, {len(r['spells'])} spells, trend median {np.nanmedian(r['trend']):.3f}/decade (true 0.2)")
print("modes:",[(m['variance_pct'],m['oni_corr'],m['oni_lag_months']) for m in r['modes']])
print("EOF1 loading weak-end vs strong-end:", r['pats'][0][:50].mean().round(2), r['pats'][0][-50:].mean().round(2))
print("best lag (strong points):", np.median(r['best_lag'][-100:]), " corr:", np.median(r['best_corr'][-100:]).round(2))
ix=[i for i,e in enumerate(events) if e['id']=='el-nino-2015'][0]
d=r['eff']['detected'][ix]
print("2015 rank strong-end", np.median(r["eff"]["rank_pct"][ix][-100:]), "weak-end", np.median(r["eff"]["rank_pct"][ix][:100])); print("2015 El Nino detected: weak-end", d[:100].mean().round(2), " strong-end", d[-100:].mean().round(2))
print("  onset days (strong):", np.nanmedian(r['eff']['onset_days'][ix][-100:]), " duration:", np.nanmedian(r['eff']['duration_days'][ix][-100:]), " sign:", np.nanmean(r['eff']['sign'][ix][-100:]))
ty=r['typical']['el_nino']; print("typical El Nino mean z strong-end:", ty['mean_z'][-100:].mean().round(2), " same-sign:", ty['n_same_sign'][-100:].mean().round(1),"/",ty['n_events'])
print("regional spells:",[(str(r['wdates'][s['start']].date()),str(r['wdates'][s['end']-1].date()),round(s['mean_z'],2)) for s in r['reg_spells']])
for i,e in enumerate(events):
    print(f"{e['id']:>14} detected strong {np.nanmean(r['eff']['detected'][i][-100:]):.2f} weak {np.nanmean(r['eff']['detected'][i][:100]):.2f}  rank strong {np.nanmedian(r['eff']['rank_pct'][i][-100:]):.0f}  sign {np.nanmean(r['eff']['sign'][i][-100:]):+.2f}")
