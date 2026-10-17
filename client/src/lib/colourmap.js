// Colour maps for the ocean grids, no dependencies.
// Each scheme is a list of evenly spaced hex stops, interpolated linearly.
// Names follow matplotlib; cmocean names map to the closest scheme here.

const SCHEMES = {
    viridis: [
        "#440154",
        "#482878",
        "#3e4989",
        "#31688e",
        "#26828e",
        "#1f9e89",
        "#35b779",
        "#6ece58",
        "#b5de2b",
        "#fde725",
    ],
    plasma: [
        "#0d0887",
        "#46039f",
        "#7201a8",
        "#9c179e",
        "#bd3786",
        "#d8576b",
        "#ed7953",
        "#fb9f3a",
        "#fdca26",
        "#f0f921",
    ],
    inferno: [
        "#000004",
        "#1b0c41",
        "#4a0c6b",
        "#781c6d",
        "#a52c60",
        "#cf4446",
        "#ed6925",
        "#fb9b06",
        "#f7d13d",
        "#fcffa4",
    ],
    magma: [
        "#000004",
        "#180f3d",
        "#440f76",
        "#721f81",
        "#9e2f7f",
        "#cd4071",
        "#f1605d",
        "#fd9668",
        "#feca8d",
        "#fcfdbf",
    ],
    cividis: [
        "#00224e",
        "#123570",
        "#3b496c",
        "#575d6d",
        "#707173",
        "#8a8678",
        "#a59c74",
        "#c3b369",
        "#e1cc55",
        "#fee838",
    ],
    turbo: [
        "#30123b",
        "#4662d7",
        "#36aaf9",
        "#1ae4b6",
        "#72fe5e",
        "#c8ef34",
        "#faba39",
        "#f66b19",
        "#ca2a04",
        "#7a0403",
    ],
    rdbu: [
        "#67001f",
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#f7f7f7",
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",
        "#053061",
    ],
    rdylbu: [
        "#a50026",
        "#d73027",
        "#f46d43",
        "#fdae61",
        "#fee090",
        "#ffffbf",
        "#e0f3f8",
        "#abd9e9",
        "#74add1",
        "#4575b4",
        "#313695",
    ],
    spectral: [
        "#9e0142",
        "#d53e4f",
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#ffffbf",
        "#e6f598",
        "#abdda4",
        "#66c2a5",
        "#3288bd",
        "#5e4fa2",
    ],
    blues: [
        "#f7fbff",
        "#deebf7",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#08519c",
        "#08306b",
    ],
    reds: [
        "#fff5f0",
        "#fee0d2",
        "#fcbba1",
        "#fc9272",
        "#fb6a4a",
        "#ef3b2c",
        "#cb181d",
        "#a50f15",
        "#67000d",
    ],
    ylgnbu: [
        "#ffffd9",
        "#edf8b1",
        "#c7e9b4",
        "#7fcdbb",
        "#41b6c4",
        "#1d91c0",
        "#225ea8",
        "#253494",
        "#081d58",
    ],
    ylgn: [
        "#ffffe5",
        "#f7fcb9",
        "#d9f0a3",
        "#addd8e",
        "#78c679",
        "#41ab5d",
        "#238443",
        "#006837",
        "#004529",
    ],
};

const ALIASES = {
    thermal: "inferno",
    haline: "viridis",
    deep: "ylgnbu",
    dense: "plasma",
    speed: "ylgn",
    amp: "reds",
    ice: "blues",
    tempo: "ylgnbu",
    balance: "rdbu_r",
    coolwarm: "rdbu_r",
    bwr: "rdbu_r",
    seismic: "rdbu_r",
    curl: "spectral",
    delta: "spectral",
    diff: "rdbu_r",
};

const hexToRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

function interpolator(stops) {
    const rgb = stops.map(hexToRgb);
    const n = rgb.length - 1;
    return (t) => {
        const x = Math.min(1, Math.max(0, t)) * n;
        const i = Math.min(n - 1, Math.floor(x));
        const f = x - i;
        const a = rgb[i];
        const b = rgb[i + 1];
        return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * f));
    };
}

function resolve(name) {
    if (!name) return null;
    let n = String(name)
        .toLowerCase()
        .replace(/^cmo\./, "");
    n = ALIASES[n] ?? n;
    let flip = false;
    if (n.endsWith("_r")) {
        flip = true;
        n = n.slice(0, -2);
    }
    n = ALIASES[n] ?? n; // e.g. "balance_r"
    if (n.endsWith("_r")) {
        flip = !flip;
        n = n.slice(0, -2);
    }
    const stops = SCHEMES[n];
    if (!stops) return null;
    return interpolator(flip ? [...stops].reverse() : stops);
}

// Returns t (0..1) -> [r, g, b]. Anomalies always use blue (below normal) -> red (above).
export function getColormap(name, { anomaly = false } = {}) {
    if (anomaly) return resolve("rdbu_r");
    return resolve(name) ?? resolve("viridis");
}

// 256-entry RGB lookup table for fast per-pixel colouring
export function buildLut(fn) {
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) lut.set(fn(i / 255), i * 3);
    return lut;
}

// CSS gradient for the legend
export function cssGradient(fn, stops = 12) {
    const parts = Array.from({ length: stops }, (_, i) => {
        const t = i / (stops - 1);
        const [r, g, b] = fn(t);
        return `rgb(${r}, ${g}, ${b}) ${Math.round(t * 100)}%`;
    });
    return `linear-gradient(to right, ${parts.join(", ")})`;
}
