/**
 * @file '@/lib/eventText.js'
 * @description Turns the El Niño analysis (/events) into plain-language text.
 */

// How each variable is described to a non-expert. scale converts the file's units for display.
export const VARIABLE_WORDS = {
    temperature: { label: "Water temperature", up: "warmer", down: "colder", units: "°C", scale: 1, digits: 1 },
    salinity: { label: "Salinity", up: "saltier", down: "fresher", units: "PSU", scale: 1, digits: 2 },
    "sea-surface-height": { label: "Sea level", up: "higher", down: "lower", units: "cm", scale: 100, digits: 0 },
    "mixed-layer-depth": { label: "Mixing depth", up: "deeper", down: "shallower", units: "m", scale: 1, digits: 0 },
};

// By chance about 1 point in 10 passes the per-point test, so an event only "clearly"
// changed a variable when at least this share of the region moved the same way.
export const CLEAR_EFFECT_PCT = 20;

export const PHASE_NAME = { el_nino: "El Niño", la_nina: "La Niña" };

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function formatYm(ym) {
    if (!ym) return "";
    const [y, m] = ym.split("-").map(Number);
    return `${MONTHS[m - 1]} ${y}`;
}

// "YYYY-MM" -> "Nov 2015", "YYYY-MM-DD" -> "15 Nov 2015"
export function formatTime(t) {
    if (!t) return "";
    if (t.length <= 7) return formatYm(t);
    const [y, m, d] = t.split("-").map(Number);
    return `${d} ${MONTHS[m - 1]} ${y}`;
}

export function addMonths(ym, n) {
    const [y, m] = ym.split("-").map(Number);
    const t = y * 12 + (m - 1) + n;
    return `${Math.floor(t / 12)}-${String((t % 12) + 1).padStart(2, "0")}`;
}

const toMonths = (days) => (days == null ? null : Math.round(days / 30.4));

export function isClear(effect) {
    if (!effect || effect.too_early) return false;
    if (typeof effect.clear_effect === "boolean") return effect.clear_effect;
    return (effect.pct_points_main_direction ?? 0) >= CLEAR_EFFECT_PCT; // files written before the flag
}

/**
 * One variable's effect during one event, as a short headline and a sentence.
 * Returns { status: "clear" | "none" | "early", direction: 1 | -1 | 0, headline, detail }
 */
export function describeEffect(slug, effect, phase, { ongoing = false } = {}) {
    const w = VARIABLE_WORDS[slug];
    const name = PHASE_NAME[phase] ?? "the event";
    if (!w || !effect) return { status: "none", direction: 0, headline: "No data", detail: "" };
    if (effect.too_early) {
        return {
            status: "early",
            direction: 0,
            headline: "Too early to tell",
            detail: `${name}'s peak hasn't reached this region yet.`,
        };
    }
    if (!isClear(effect)) {
        return {
            status: "none",
            direction: 0,
            headline: ongoing ? "No clear sign yet" : "No clear effect",
            detail: `Only ${Math.round(effect.pct_points_changed ?? 0)}% of the region changed, about what chance alone gives.`,
        };
    }

    const up = effect.main_direction === "above";
    const word = up ? w.up : w.down;
    const pct = Math.round(effect.pct_points_main_direction ?? effect.pct_points_changed);
    const amount =
        effect.median_anomaly == null
            ? null
            : Math.abs(effect.median_anomaly * w.scale).toFixed(w.digits);

    const onset = toMonths(effect.median_onset_days);
    const lasted = toMonths(effect.median_duration_days);
    let when = "";
    if (onset != null) {
        if (onset <= -2) when = `already under way ${-onset} months before ${name} began`;
        else if (onset <= 1) when = `starting about when ${name} began`;
        else when = `starting about ${onset} months after ${name} began`;
    }

    // the headline already says which way and by how much; the detail says where and when
    const parts = [`Across ${pct}% of the region`];
    if (when) parts.push(when);
    if (lasted) parts.push(`lasting about ${lasted} months`);

    return {
        status: "clear",
        direction: up ? 1 : -1,
        headline: `${word[0].toUpperCase()}${word.slice(1)}${amount != null ? ` · ${up ? "+" : "−"}${amount} ${w.units}` : ""}`,
        detail: `${parts.join(", ")}.`,
    };
}

// The month to show on the map for an event: its ONI peak, shifted by how long the ocean
// here typically takes to respond (from the analysis; 3 months if unknown).
export function eventMapMonth(event) {
    const lag = event?.effects?.temperature?.typical_response_lag_months ?? 3;
    return addMonths(event.peak_month, lag);
}

// Same, as a day (mid-month) when the map works in days
export function eventMapTime(event, timeStep = "month") {
    const m = eventMapMonth(event);
    return timeStep === "day" ? `${m}-15` : m;
}
