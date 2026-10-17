// Turns a /maps response into an image MapLibre can place on the map.
//
// The server's grid is regular in latitude, but MapLibre stretches an image
// source linearly in Web Mercator. Rows are therefore resampled in Mercator
// space; otherwise the data would drift north/south against the coastline.

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

// Pure part: RGBA pixels + corner coordinates (no DOM, so it can be tested in node)
export function gridToPixels(grid, lut, range) {
    const { latitudes: lats, longitudes: lons, values } = grid;
    const nLat = lats.length;
    const nLon = lons.length;

    // Grid cell size; latitudes/longitudes are cell centres, south->north, west->east
    const dLat = nLat > 1 ? (lats[nLat - 1] - lats[0]) / (nLat - 1) : 1;
    const dLon = nLon > 1 ? (lons[nLon - 1] - lons[0]) / (nLon - 1) : 1;
    const south = lats[0] - dLat / 2;
    const north = lats[nLat - 1] + dLat / 2;
    const west = lons[0] - dLon / 2;
    const east = lons[nLon - 1] + dLon / 2;

    const [min, max] = range ?? autoRange(values);
    const span = max - min || 1;

    const width = nLon;
    const height = nLat * 2; // oversample rows so the Mercator resampling stays smooth
    const yTop = toMercY(north);
    const yBottom = toMercY(south);
    const data = new Uint8ClampedArray(width * height * 4); // transparent by default

    for (let r = 0; r < height; r++) {
        const lat = fromMercY(yTop + ((r + 0.5) / height) * (yBottom - yTop));
        const i = Math.min(
            nLat - 1,
            Math.max(0, Math.floor((lat - south) / dLat)),
        );
        const row = values[i];
        if (!row) continue;

        for (let c = 0; c < width; c++) {
            const v = row[c];
            if (v == null || Number.isNaN(v)) continue; // land stays transparent
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
export function gridToImage(grid, lut, range) {
    const px = gridToPixels(grid, lut, range);
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
