/**
 * @file '@/lib/oceanApi.js'
 * @description fetches data from Flask, caches it
 */
// "/ocean" is proxied to http://localhost:5001 in vite.config.js
const BASE = import.meta.env?.VITE_OCEAN_API ?? "/ocean";
const cache = new Map(); // caching so going back to a month is instant

function get(path, { cached = false } = {}) {
    const url = BASE + path;
    if (cached && cache.has(url)) return cache.get(url);

    const p = fetch(url).then(async (res) => {
        if (!res.ok) {
            let reason = res.statusText;
            try {
                const body = await res.json();
                reason = body.error ?? body.message ?? reason;
            } catch {
                /* not JSON */
                throw new Error("Error in response body.");
            }
            throw new Error(`${res.status} ${reason} (${path})`);
        }
        return res.json();
    });

    if (cached) {
        cache.set(url, p);
        p.catch(() => cache.delete(url)); // don't cache failures
    }
    return p;
}

export const getHealth = () => get("/health");
export const getMetadata = () => get("/metadata", { cached: true });
export const getElNino = () => get("/el-nino", { cached: true });
export const getForecast = () => get("/forecast"); // re-read by the server every call
export const getEvents = () => get("/events", { cached: true }); // El Niño history analysis

// when: "YYYY-MM-DD" (daily data) or "YYYY-MM" (monthly data / forecast)
export function getMap(variable, when, { anomaly = false, area } = {}) {
    const params = new URLSearchParams();
    if (anomaly) params.set("anomaly", "true");
    if (area) params.set("area", area);
    const qs = params.toString();
    return get(`/maps/${variable}/${when}${qs ? `?${qs}` : ""}`, {
        cached: true,
    });
}

export function getLocationSeries(lat, lon, variable) {
    const params = new URLSearchParams({ lat, lon, variable });
    return get(`/timeseries/location?${params}`, { cached: true });
}

export const getRegionSeries = (variable) =>
    get(`/timeseries/region/${variable}`, { cached: true });

// variables: object keyed by id, or an array of {id|name|key, ...}
export function normalizeVariables(meta) {
    const vars = meta?.variables;
    if (!vars) return [];
    const list = Array.isArray(vars)
        ? vars.map((v) =>
              typeof v === "string"
                  ? { id: v }
                  : { id: v.id ?? v.name ?? v.key, ...v },
          )
        : Object.entries(vars).map(([id, v]) => ({ id, ...v }));
    return list.map((v) => ({
        ...v,
        label: v.label ?? v.id,
        hasMap: v.has_map ?? true,
        hasForecast: v.has_forecast ?? false,
    }));
}

// [min, max] or {min, max}
export function toRange(r) {
    if (!r) return null;
    if (Array.isArray(r) && r.length >= 2) return [Number(r[0]), Number(r[1])];
    if (typeof r === "object" && "min" in r)
        return [Number(r.min), Number(r.max)];
    return null;
}

const ym = (y, m) => `${y}-${String(m).padStart(2, "0")}`;

function expandRange(start, end) {
    const [y0, m0] = start.split("-").map(Number);
    const [y1, m1] = end.split("-").map(Number);
    const out = [];
    for (
        let y = y0, m = m0;
        y < y1 || (y === y1 && m <= m1);
        m === 12 ? (y++, (m = 1)) : m++
    ) {
        out.push(ym(y, m));
    }
    return out;
}

// {start: "2002-01-01", end: "2026-10-11"} -> every day in between, as "YYYY-MM-DD"
export function toDays(value) {
    if (!value?.start || !value?.end) return [];
    const out = [];
    const d = new Date(`${value.start}T00:00:00Z`);
    const end = new Date(`${value.end}T00:00:00Z`);
    while (d <= end) {
        out.push(d.toISOString().slice(0, 10));
        d.setUTCDate(d.getUTCDate() + 1);
    }
    return out;
}

export function toMonths(value) {
    if (!value) return [];
    if (Array.isArray(value)) {
        return value
            .map((v) => (typeof v === "string" ? v : v?.month))
            .filter(Boolean)
            .map((s) => s.slice(0, 7));
    }
    if (typeof value === "object" && value.start && value.end) {
        return expandRange(value.start.slice(0, 7), value.end.slice(0, 7));
    }
    return [];
}

// Map bounds as [[west, south], [east, north]]
export function toBounds(b, grid) {
    if (b) {
        if (Array.isArray(b) && b.length === 2 && Array.isArray(b[0])) return b;
        if (Array.isArray(b) && b.length === 4)
            return [
                [b[0], b[1]],
                [b[2], b[3]],
            ];
        if (typeof b === "object") {
            const w = b.west ?? b.lon_min ?? b.min_lon ?? b.lonMin;
            const s = b.south ?? b.lat_min ?? b.min_lat ?? b.latMin;
            const e = b.east ?? b.lon_max ?? b.max_lon ?? b.lonMax;
            const n = b.north ?? b.lat_max ?? b.max_lat ?? b.latMax;
            if ([w, s, e, n].every((x) => typeof x === "number"))
                return [
                    [w, s],
                    [e, n],
                ];
        }
    }
    // Fall back to the grid description: {latitude: {start, step, count}, longitude: {...}}
    const lat = grid?.latitude;
    const lon = grid?.longitude;
    if (lat && lon) {
        const lat1 = lat.start + lat.step * (lat.count - 1);
        const lon1 = lon.start + lon.step * (lon.count - 1);
        return [
            [Math.min(lon.start, lon1), Math.min(lat.start, lat1)],
            [Math.max(lon.start, lon1), Math.max(lat.start, lat1)],
        ];
    }
    return null;
}

export const PHASE_LABEL = {
    el_nino: "El Niño",
    la_nina: "La Niña",
    neutral: "Neutral",
};
