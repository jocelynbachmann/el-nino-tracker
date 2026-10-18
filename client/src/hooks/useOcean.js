/*
 * @file @/hooks/useOcean.js
 * @description hooks for getting
 */
import { useEffect, useMemo, useState } from "react";
import {
    getElNino,
    getForecast,
    getForecastMap,
    getMap,
    getMetadata,
    normalizeVariables,
    toBounds,
    toMonths,
    toDays,
    toRange,
} from "@/lib/oceanApi";
import { buildLut, getColourmap } from "@/lib/colourmap";
import { gridToImage } from "@/lib/gridImage";

// Metadata, ENSO labels and forecast, loaded once
export function useOceanMeta() {
    const [state, setState] = useState({ loading: true, error: null });

    useEffect(() => {
        let cancelled = false;

        Promise.all([
            getMetadata(),
            getElNino().catch(() => []), // optional extras: don't fail the app over them
            getForecast().catch(() => null),
        ])
            .then(([meta, elNino, forecast]) => {
                if (cancelled) return;
                setState({
                    loading: false,
                    error: null,
                    meta,
                    variables: normalizeVariables(meta).filter((v) => v.hasMap),
                    // "day" once the daily store exists: maps are fetched per day, not per month
                    timeStep: meta.time_step ?? "month",
                    observedDays: toDays(meta.observed_days),
                    observedMonths: toMonths(meta.observed_months),
                    forecastMonths: toMonths(meta.forecast_months),
                    bounds: toBounds(meta.bounds, meta.grid),
                    elNinoByMonth: new Map(
                        (elNino ?? []).map((e) => [e.month?.slice(0, 7), e]),
                    ),
                    forecastByMonth: new Map(
                        (forecast?.months ?? []).map((f) => [
                            f.month?.slice(0, 7),
                            f,
                        ]),
                    ),
                    forecastInfo: forecast
                        ? { issued: forecast.issued, model: forecast.model }
                        : null,
                    // Per-point MLD forecast maps: { issued, model, band, months: ["2026-10", ...] }
                    forecastGrid: forecast?.grid?.months?.length
                        ? forecast.grid
                        : null,
                });
            })
            .catch((err) => {
                if (!cancelled) setState({ loading: false, error: err });
            });

        return () => {
            cancelled = true;
        };
    }, []);

    return state;
}
const imageCache = new Map();
const MAX_CACHED_IMAGES = 400; // days add up quickly when scrubbing; drop the oldest

function loadOverlay(variable, month, anomaly) {
    const key = `${variable.id}|${month}|${anomaly}`;
    if (imageCache.has(key)) return imageCache.get(key);
    if (imageCache.size >= MAX_CACHED_IMAGES)
        imageCache.delete(imageCache.keys().next().value);

    const p = getMap(variable.id, month, { anomaly }).then((grid) => {
        const range =
              toRange(grid.color_range) ??
              toRange(anomaly ? variable.anomaly_range : variable.range) ??
              null; // null -> computed from the data
        const lut = buildLut(getColourmap(variable.colormap, { anomaly }));
        const img = gridToImage(grid, lut, range, { extend: 2 });
        return {
            overlay: { url: img.url, coordinates: img.coordinates },
            range: img.range,
            source: grid.source ?? "observed",
        };
    });

    imageCache.set(key, p);
    p.catch(() => imageCache.delete(key)); // don't cache failures
    return p;
}

// Several variables for one month. Returns { [variableId]: { overlay, range, source, loading, error } }
export function useOceanOverlays({ variables, month, anomaly }) {
    const [results, setResults] = useState({});
    const ids = variables.map((v) => v.id).join(",");

    useEffect(() => {
        if (!month || !variables.length) {
            setResults({});
            return;
        }
        let cancelled = false;

        // Keep each layer's previous image while the new month loads (no flicker),
        // and drop layers that were unchecked
        setResults((prev) => {
            const next = {};
            for (const v of variables)
                next[v.id] = { ...prev[v.id], loading: true, error: null };
            return next;
        });

        for (const v of variables) {
            loadOverlay(v, month, anomaly)
                .then((r) => {
                    if (!cancelled) {
                        setResults((prev) => ({
                            ...prev,
                            [v.id]: { ...r, loading: false, error: null },
                        }));
                    }
                })
                .catch((err) => {
                    // e.g. a forecast month for a variable that only has observations
                    if (!cancelled) {
                        setResults((prev) => ({
                            ...prev,
                            [v.id]: {
                                overlay: null,
                                loading: false,
                                error: err,
                            },
                        }));
                    }
                });
        }

        return () => {
            cancelled = true; // ignore late responses while scrubbing the slider
        };
    }, [ids, month, anomaly]); // eslint-disable-line react-hooks/exhaustive-deps

    return results;
}

// Per-point MLD forecast for one month, coloured like the observed MLD layer (`variable`).
// Returns { overlay, range, loading, error }; empty when month is null (layer off).
export function useForecastOverlay({ month, anomaly, variable }) {
    const [state, setState] = useState({});

    useEffect(() => {
        if (!month) {
            setState({});
            return;
        }
        let cancelled = false;
        setState((s) => ({ ...s, loading: true, error: null }));

        const key = `forecast|${month}|${anomaly}`;
        if (!imageCache.has(key)) {
            const p = getForecastMap(month, { anomaly }).then((grid) => {
                const lut = buildLut(
                    getColourmap(grid.colormap ?? variable?.colormap, { anomaly }),
                );
                const img = gridToImage(grid, lut, toRange(grid.color_range));
                return {
                    overlay: { url: img.url, coordinates: img.coordinates },
                    range: img.range,
                };
            });
            imageCache.set(key, p);
            p.catch(() => imageCache.delete(key));
        }
        imageCache
            .get(key)
            .then((r) => !cancelled && setState({ ...r, loading: false, error: null }))
            .catch((err) => !cancelled && setState({ loading: false, error: err }));

        return () => {
            cancelled = true;
        };
    }, [month, anomaly]); // eslint-disable-line react-hooks/exhaustive-deps

    return state;
}

// Observed months, plus forecast months if any selected variable has them
export function useMonths(meta, variables) {
    return useMemo(() => {
        if (!meta?.observedMonths) return [];
        if (!variables.some((v) => v.hasForecast)) return meta.observedMonths;
        const seen = new Set(meta.observedMonths);
        return [
            ...meta.observedMonths,
            ...meta.forecastMonths.filter((m) => !seen.has(m)),
        ];
    }, [meta, variables]);
}
