/*
 * @file @/hooks/useAnalysisMap.js
 * @description Map overlay from the El Niño analysis for the selected event:
 *   onset (when each spot's change started), duration, or typical (same change in past events).
 */
import { useEffect, useState } from "react";
import { getEventMap, getTypicalMap, toRange } from "@/lib/oceanApi";
import { buildLut, getColourmap } from "@/lib/colourmap";
import { gridToImage } from "@/lib/gridImage";

const cache = new Map();

function load(mode, event, variable) {
    const key = `${mode}|${event.id}|${variable}`;
    if (cache.has(key)) return cache.get(key);
    const req =
        mode === "typical"
            ? getTypicalMap(event.phase.replace("_", "-"), variable)
            : getEventMap(event.id, variable, mode);
    const p = req.then((grid) => {
        const lut = buildLut(getColourmap(grid.colormap));
        const img = gridToImage(grid, lut, toRange(grid.color_range));
        const anyValue = grid.values.some((row) => row.some((v) => v != null));
        return {
            overlay: { url: img.url, coordinates: img.coordinates },
            range: toRange(grid.color_range),
            colormap: grid.colormap,
            nEvents: grid.n_events,
            empty: !anyValue, // e.g. no clear change anywhere, or too early to tell
        };
    });
    cache.set(key, p);
    p.catch(() => cache.delete(key));
    return p;
}

// mode: "day" (no analysis overlay) | "onset" | "duration" | "typical"
export function useAnalysisMap({ mode, event, variable }) {
    const [state, setState] = useState({});
    const active = event && mode && mode !== "day";

    useEffect(() => {
        if (!active) {
            setState({});
            return;
        }
        let cancelled = false;
        setState((s) => ({ ...s, loading: true, error: null }));
        load(mode, event, variable)
            .then((r) => !cancelled && setState({ ...r, loading: false, error: null }))
            .catch((err) => !cancelled && setState({ loading: false, error: err }));
        return () => {
            cancelled = true;
        };
    }, [active, mode, event?.id, variable]); // eslint-disable-line react-hooks/exhaustive-deps

    return state;
}
