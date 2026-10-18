/*
 * @file @/hooks/useAnalysisMap.js
 * @description Map overlay from the El Niño analysis for the selected event:
 *   onset (when each spot's change started), duration, or typical (same change in past events).
 */
import { useEffect, useState } from "react";
import { getEventMap, getTypicalMap, toRange } from "@/lib/oceanApi";
import { buildLut, getColourmap } from "@/lib/colourmap";
import { gridToImage, gridToPixels } from "@/lib/gridImage";

const cache = new Map();
export const NO_CHANGE_RGB = [120, 113, 108]; // grey stripes for "checked, no clear change"

// Diagonal grey stripes over the cells marked 1: "checked, no clear change" (plain grey looks like water)
function hatchedImage(grid) {
    const px = gridToPixels(grid, buildLut(() => NO_CHANGE_RGB), [0, 1], { scale: 8 });
    for (let r = 0; r < px.height; r++) {
        for (let c = 0; c < px.width; c++) {
            if ((r + c) % 9 >= 3) px.data[(r * px.width + c) * 4 + 3] = 0; // keep 3 of every 9 diagonals
        }
    }
    const canvas = document.createElement("canvas");
    canvas.width = px.width;
    canvas.height = px.height;
    canvas.getContext("2d").putImageData(new ImageData(px.data, px.width, px.height), 0, 0);
    return { url: canvas.toDataURL(), coordinates: px.coordinates };
}

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
        // spots that were checked but had no clear change: flat grey, drawn underneath
        const overlays = [];
        if (grid.no_change?.some((row) => row.some((v) => v != null))) {
            const grey = hatchedImage({ latitudes: grid.latitudes, longitudes: grid.longitudes, values: grid.no_change });
            overlays.push({ id: "analysis-none", url: grey.url, coordinates: grey.coordinates, opacity: 0.9 });
        }
        overlays.push({ id: "analysis", url: img.url, coordinates: img.coordinates, opacity: 0.9 });
        return {
            overlays,
            hasNoChange: overlays.length > 1,
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
