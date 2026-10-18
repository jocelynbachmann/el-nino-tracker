/**
 * @file '@/lib/gridImage.js'
 * @description Converts response from /map endpoint with MapLibre compatibility.
 */

// resampling to fit web mercator prejection
const toMercY = (lat) =>
    Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
const fromMercY = (y) =>
    ((2 * Math.atan(Math.exp(y)) - Math.PI / 2) * 180) / Math.PI;

export function autoRange(values) {
    let min = Infinity;
    let max = -Infinity;
    for (const row of values) {
        for (const v of row) {
            if (v == null || Number.isNaN(v)) continue;
            if (v < min) min = v;
            if (v > max) max = v;
        }
    }
    return Number.isFinite(min) ? [min, max] : [0, 1];
}

const valid = (v) => v != null && !Number.isNaN(v);

// Copy ocean values into neighbouring land cells, `passes` cells deep, so the
// colour reaches the real coastline once a detailed land layer is drawn on top
function extendIntoLand(values, passes) {
    let cur = values;
    for (let n = 0; n < passes; n++) {
        const next = cur.map((row) => row.slice());
        for (let i = 0; i < cur.length; i++) {
            for (let j = 0; j < cur[i].length; j++) {
                if (valid(cur[i][j])) continue;
                let sum = 0;
                let k = 0;
                for (let di = -1; di <= 1; di++) {
                    for (let dj = -1; dj <= 1; dj++) {
                        const v = cur[i + di]?.[j + dj];
                        if (valid(v)) {
                            sum += v;
                            k++;
                        }
                    }
                }
                if (k) next[i][j] = sum / k;
            }
        }
        cur = next;
    }
    return cur;
}
// options.scale:  output pixels per grid cell in each direction. The data is
//                 only as detailed as the grid, but upsampling with bilinear
//                 interpolation removes the blocky cells.
// options.smooth: false = one flat colour per cell (shows the raw grid).
// options.extend: cells to spread ocean values onto land (default 0). Only use
//                 with a land layer drawn on top, or the colour shows on land.
export function gridToPixels(
    grid,
    lut,
    range,
    { scale = 6, smooth = true, extend = 0 } = {},
) {
    const { latitudes: lats, longitudes: lons } = grid;
    const values = extend ? extendIntoLand(grid.values, extend) : grid.values;
    const nLat = lats.length;
    const nLon = lons.length;

    // Grid cell size; latitudes/longitudes are cell centres, south->north, west->east
    const dLat = nLat > 1 ? (lats[nLat - 1] - lats[0]) / (nLat - 1) : 1;
    const dLon = nLon > 1 ? (lons[nLon - 1] - lons[0]) / (nLon - 1) : 1;
    const south = lats[0] - dLat / 2;
    const north = lats[nLat - 1] + dLat / 2;
    const west = lons[0] - dLon / 2;
    const east = lons[nLon - 1] + dLon / 2;

    const [min, max] = range ?? autoRange(grid.values);
    const span = max - min || 1;

    const width = Math.round(nLon * scale);
    const height = Math.round(nLat * scale);
    const yTop = toMercY(north);
    const yBottom = toMercY(south);
    const data = new Uint8ClampedArray(width * height * 4); // transparent by default

    const cell = (i, j) =>
        i >= 0 && i < nLat && j >= 0 && j < nLon ? values[i]?.[j] : undefined;

    for (let r = 0; r < height; r++) {
        // Row centre -> latitude, in Mercator space so rows line up with the basemap
        const lat = fromMercY(yTop + ((r + 0.5) / height) * (yBottom - yTop));
        // Fractional row index (0 = first cell centre), clamped so the outer
        // half-cell repeats the edge values instead of fading out
        const fy = Math.min(nLat - 1, Math.max(0, (lat - lats[0]) / dLat));

        for (let c = 0; c < width; c++) {
            const lon = west + ((c + 0.5) / width) * (east - west);
            const fx = Math.min(nLon - 1, Math.max(0, (lon - lons[0]) / dLon));

            let v;
            if (smooth) {
                // Bilinear blend of the surrounding ocean cell centres for the colour,
                // but draw the pixel whenever its nearest cell is ocean, so the coastline
                // follows the grid cells and no coastal cells are dropped.
                const i0 = Math.floor(fy);
                const j0 = Math.floor(fx);
                const ty = fy - i0;
                const tx = fx - j0;
                const nearest = cell(Math.round(fy), Math.round(fx));
                if (!valid(nearest)) continue;
                let sum = 0;
                let w = 0;
                for (const [di, dj, wt] of [
                    [0, 0, (1 - ty) * (1 - tx)],
                    [0, 1, (1 - ty) * tx],
                    [1, 0, ty * (1 - tx)],
                    [1, 1, ty * tx],
                ]) {
                    const cv = cell(i0 + di, j0 + dj);
                    if (valid(cv) && wt > 0) {
                        sum += cv * wt;
                        w += wt;
                    }
                }
                // if (w < 0.5) continue; // mostly land: transparent
                // v = sum / w;
                v = w > 0 ? sum / w : nearest;
            } else {
                v = cell(Math.round(fy), Math.round(fx));
                if (!valid(v)) continue;
            }

            const t = Math.min(1, Math.max(0, (v - min) / span));
            const k = Math.round(t * 255) * 3;
            const p = (r * width + c) * 4;
            data[p] = lut[k];
            data[p + 1] = lut[k + 1];
            data[p + 2] = lut[k + 2];
            data[p + 3] = 255;
        }
    }

    return {
        width,
        height,
        data,
        range: [min, max],
        // MapLibre image source order: top-left, top-right, bottom-right, bottom-left
        coordinates: [
            [west, north],
            [east, north],
            [east, south],
            [west, south],
        ],
    };
}

// Browser part: pixels -> data URL for a MapLibre image source
export function gridToImage(grid, lut, range, options) {
    const px = gridToPixels(grid, lut, range, options);
    const canvas = document.createElement("canvas");
    canvas.width = px.width;
    canvas.height = px.height;
    canvas
        .getContext("2d")
        .putImageData(new ImageData(px.data, px.width, px.height), 0, 0);
    return {
        url: canvas.toDataURL(),
        coordinates: px.coordinates,
        range: px.range,
    };
}
