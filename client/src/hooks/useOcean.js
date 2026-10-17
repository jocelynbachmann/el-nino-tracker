import { useEffect, useMemo, useState } from "react";
import {
    getElNino,
    getForecast,
    getMap,
    getMetadata,
    normalizeVariables,
    toBounds,
    toMonths,
    toRange,
} from "@/lib/oceanApi";
import { buildLut, getColormap } from "@/lib/colourmap";
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

// One month of one variable, rendered to an image for the map
export function useOceanOverlay({
    variable,
    month,
    anomaly,
    area,
    enabled = true,
}) {
    const [state, setState] = useState({
        overlay: null,
        loading: false,
        error: null,
    });

    useEffect(() => {
        if (!enabled || !variable?.id || !month) return;
        let cancelled = false;
        setState((s) => ({ ...s, loading: true, error: null }));

        getMap(variable.id, month, { anomaly, area })
            .then((grid) => {
                if (cancelled) return;
                const range =
                    toRange(grid.color_range) ??
                    toRange(
                        anomaly ? variable.anomaly_range : variable.range,
                    ) ??
                    null; // null -> computed from the data
                const lut = buildLut(
                    getColormap(variable.colormap, { anomaly }),
                );
                const img = gridToImage(grid, lut, range);
                setState({
                    loading: false,
                    error: null,
                    overlay: { url: img.url, coordinates: img.coordinates },
                    range: img.range,
                    source: grid.source ?? "observed",
                });
            })
            .catch((err) => {
                if (!cancelled)
                    setState((s) => ({ ...s, loading: false, error: err }));
            });

        return () => {
            cancelled = true; // ignore late responses while scrubbing the slider
        };
    }, [variable, month, anomaly, area, enabled]);

    return state;
}

// The month list for a variable: observed months, plus forecast months if it has them
export function useMonths(meta, variable) {
    return useMemo(() => {
        if (!meta?.observedMonths) return [];
        if (!variable?.hasForecast) return meta.observedMonths;
        const seen = new Set(meta.observedMonths);
        return [
            ...meta.observedMonths,
            ...meta.forecastMonths.filter((m) => !seen.has(m)),
        ];
    }, [meta, variable]);
}
