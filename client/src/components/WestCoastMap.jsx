/**
 * @file WestCoastMap.jsx
 * @description Map component, focused on west coast of canada, static.
 */
import { useState, useEffect, useRef } from "react";
import { Map as MapLibreMap, setWorkerUrl } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "maplibre-gl/dist/maplibre-gl.css";
import { fetchCurrentWeather } from "@/lib/weather";

const RADAR_URL =
    "https://geo.weather.gc.ca/geomet?service=WMS&version=1.3.0&request=GetMap" +
    "&layers=RADAR_1KM_RRAI&styles=&format=image/png&transparent=true" +
    "&crs=EPSG:3857&width=256&height=256&bbox={bbox-epsg-3857}";

// Maplibre's web worker
setWorkerUrl(workerUrl);

// [[west, south], [east, north]]
// [-134, 46],
// [-120, 53],
const COAST_BOUNDS = [
    [-128.175, 46.17],
    [-122.8, 51.89],
];

const WEATHER_LAYER_IDS = ["weather-circles", "weather-temps", "weather-names"];
const REFRESH_MS = 15 * 60 * 1000;
const OCEAN_PREFIX = "ocean-"; // each data layer starts with ocean-
const HERRING_LAYER_IDS = ["herring-spawn", "herring-count"];
const EMPTY = { type: "FeatureCollection", features: [] };

function addWeatherLayers(map) {
    map.addSource("weather", {
        type: "geojson",
        data: EMPTY,
        attribution:
            '<a href="https://open-meteo.com/">Weather data by Open-Meteo.com</a>',
    });

    map.addLayer({
        id: "weather-circles",
        type: "circle",
        source: "weather",
        paint: {
            "circle-radius": 30,
            "circle-color": [
                "interpolate",
                ["linear"],
                ["get", "temp"],
                -10,
                "#1d4ed8",
                0,
                "#60a5fa",
                10,
                "#eab308",
                20,
                "#f97316",
                30,
                "#dc2626",
            ],
            "circle-stroke-color": "#ffffff",
            "circle-stroke-width": 2,
            "circle-opacity": 0.9,
        },
    });

    map.addLayer({
        id: "weather-temps",
        type: "symbol",
        source: "weather",
        layout: {
            "text-field": ["concat", ["to-string", ["get", "temp"]], "°"],
            "text-font": ["Noto Sans Bold"],
            "text-size": 12,
            "text-allow-overlap": true,
            "text-ignore-placement": true,
        },
        paint: {
            "text-color": "#ffffff",
            "text-halo-color": "rgba(0,0,0,0.35)",
            "text-halo-width": 1,
        },
    });

    map.addLayer({
        id: "weather-names",
        type: "symbol",
        source: "weather",
        layout: {
            "text-field": [
                "concat",
                ["get", "name"],
                "\n",
                ["get", "condition"],
            ],
            "text-font": ["Noto Sans Regular"],
            "text-size": 11,
            "text-anchor": "top",
            "text-offset": [0, 1.6],
            "text-optional": true,
        },
        paint: {
            "text-color": "#334155",
            "text-halo-color": "#ffffff",
            "text-halo-width": 1.5,
        },
    });
}

function addHerringLayer(map) {
    map.addSource("herring", {
        type: "geojson",
        data: EMPTY, // filled per year by an effect
        cluster: true,
        clusterRadius: 30, // px: points closer than this merge into one circle
        attribution:
            "Herring spawn: Fisheries and Oceans Canada (Open Government Licence – Canada)",
    });

    map.addLayer({
        id: "herring-spawn",
        type: "circle",
        source: "herring",
        layout: { visibility: "none" },
        paint: {
            // single spawns have no point_count, so treat them as 1
            "circle-radius": [
                "interpolate",
                ["linear"],
                ["coalesce", ["get", "point_count"], 1],
                1,
                4,
                10,
                10,
                50,
                18,
                150,
                28,
            ],
            // Soft magenta with a white ring: a colour no data layer uses, so it stands out on every
            // colour scale and the dark basemap without being mistaken for red/blue anomalies
            // or the amber predictions
            "circle-color": "#db5fa3",
            "circle-opacity": 0.85,
            "circle-stroke-color": "#ffffff",
            "circle-stroke-width": 1.5,
        },
    });

    map.addLayer({
        id: "herring-count",
        type: "symbol",
        source: "herring",
        filter: ["has", "point_count"], // only label clusters
        layout: {
            visibility: "none",
            "text-field": ["get", "point_count_abbreviated"],
            "text-font": ["Noto Sans Bold"],
            "text-size": 11,
            "text-allow-overlap": true,
        },
        paint: { "text-color": "#ffffff" },
    });
}

// // place ocean raster under basemap labels
// function firstSymbolLayerId(map) {
//     return map
//         .getStyle()
//         .layers.find((l) => l.type === "symbol" && !l.id.startsWith("weather"))
//         ?.id;
// }
//

// first basemap layer above the water, so roads and labels stay on top of the overlay
function firstAboveWater(map) {
    const layers = map.getStyle().layers;
    const last = layers.findLastIndex((l) => l["source-layer"] === "water");
    return layers[last + 1]?.id;
}

// Detailed land drawn over the ocean rasters, so their edge is the real coastline
function addLandMask(map) {
    const bg = map.getStyle().layers.find((l) => l.type === "background");
    map.addSource("land", {
        type: "geojson",
        data: "/land.geojson",
        attribution: "© OpenStreetMap contributors",
    });
    map.addLayer(
        {
            id: "land-mask",
            type: "fill",
            source: "land",
            paint: {
                "fill-color": map.getPaintProperty(bg.id, "background-color"),
            },
        },
        firstAboveWater(map),
    );
}

// Basemap polygons above the water (parks, landcover, landuse) reach into the sea
// and cover the ocean overlay, so hide them
function hideBasemapFills(map) {
    const layers = map.getStyle().layers;
    const above = layers.slice(
        layers.findIndex((l) => l.id === "land-mask") + 1,
    );
    const hidden = new Set(
        above
        .filter((l) => l.type.startsWith("fill"))
        .map((l) => l["source-layer"]),
    );
    for (const l of above) {
        if (
            l.type.startsWith("fill") ||
                (l.type === "line" && hidden.has(l["source-layer"]))
        )
            map.setLayoutProperty(l.id, "visibility", "none");
    }
}

/**
 * Map component using maplibre
 * @param {object} props - component properties
 * @param {className} props.className - tailwind
 */
export default function WestCoastMap({
    className = "h-svh w-full",
    bounds = COAST_BOUNDS,
    overlays = [],
    showWeather = false,
    showHerring = false,
    herringYear = null,
}) {
    const containerRef = useRef(null);
    const mapRef = useRef(null);
    const boundsRef = useRef(bounds);
    const urlsRef = useRef(new Map()); // layer id
    const [loaded, setLoaded] = useState(false);
    const [herring, setHerring] = useState(null);

    useEffect(() => {
        fetch("/herring-spawn.geojson")
                             .then((r) => r.json())
                             .then(setHerring)
                             .catch((err) => console.error("Herring data failed to load:", err));
    }, []);
    // Create the map once
    useEffect(() => {
        const map = new MapLibreMap({
            container: containerRef.current,
            style: "https://tiles.openfreemap.org/styles/fiord",
            bounds: boundsRef.current,
            fitBoundsOptions: { padding: 20 },
            interactive: false,
            attributionControl: { compact: true },
        });
        mapRef.current = map;
        window.map = map; // debugging
        const urls = urlsRef.current;

        const fit = (why) => {
            const el = containerRef.current;
            map.fitBounds(boundsRef.current, { padding: 20, animate: false });
        };

        // This is what already worked when resizing the window
        map.on("resize", () => fit("resize"));

        map.on("load", () => {
            addLandMask(map);
            hideBasemapFills(map);
            addHerringLayer(map);
            addWeatherLayers(map);
            setLoaded(true);
        });

        // Once everything has loaded and drawn for the first time, frame it again.
        // This catches anything that moved the camera during loading.
        map.once("idle", () => {
            map.resize();
            fit("first idle");
        });

        return () => {
            map.remove();
            mapRef.current = null;
            urls.clear();
            setLoaded(false);
        };
    }, []);

    // Re-frame when the data bounds arrive from /metadata
    const boundsKey = JSON.stringify(bounds);
    useEffect(() => {
        boundsRef.current = bounds;
        const map = mapRef.current;
        if (map) {
            console.log("fit: bounds changed", bounds);
            map.fitBounds(bounds, { padding: 20, animate: false });
        }
    }, [boundsKey]); // eslint-disable-line react-hooks/exhaustive-deps

    // Ocean data overlay
    useEffect(() => {
        const map = mapRef.current;
        if (!loaded || !map) return;
        const urls = urlsRef.current;
        const wanted = new Set(overlays.map((o) => OCEAN_PREFIX + o.id));
        for (const layer of map.getStyle().layers) {
            if (layer.id.startsWith(OCEAN_PREFIX) && !wanted.has(layer.id)) {
                map.removeLayer(layer.id);
                map.removeSource(layer.id);
                urls.delete(layer.id);
            }
        }

        // const beforeId = firstSymbolLayerId(map);
        const beforeId = "land-mask";

        for (const o of overlays) {
            const id = OCEAN_PREFIX + o.id;

            if (!map.getSource(id)) {
                map.addSource(id, {
                    type: "image",
                    url: o.url,
                    coordinates: o.coordinates,
                });
                map.addLayer(
                    {
                        id,
                        type: "raster",
                        source: id,
                        paint: {
                            "raster-opacity": o.opacity,
                            "raster-fade-duration": 0,
                            "raster-resampling": "linear",
                        },
                    },
                    beforeId,
                );
                urls.set(id, o.url);
            } else if (urls.get(id) !== o.url) {
                map.getSource(id).updateImage({
                    url: o.url,
                    coordinates: o.coordinates,
                });
                urls.set(id, o.url);
            }

            map.setPaintProperty(id, "raster-opacity", o.opacity);
            // Moving each one just under the labels, in order, leaves the last one on top
            map.moveLayer(id, beforeId);
        }

        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [loaded, overlays]);

    useEffect(() => {
        if (!loaded || !showWeather) return;
        let cancelled = false;

        const update = async () => {
            try {
                const geojson = await fetchCurrentWeather();
                if (!cancelled)
                    mapRef.current?.getSource("weather")?.setData(geojson);
            } catch (err) {
                console.error("Weather fetch failed:", err);
            }
        };

        update();
        const id = setInterval(update, REFRESH_MS);
        return () => {
            cancelled = true;
            clearInterval(id);
        };
    }, [loaded, showWeather]);

    useEffect(() => {
        const map = mapRef.current;
        if (!loaded || !map) return;
        for (const id of WEATHER_LAYER_IDS) {
            if (map.getLayer(id)) {
                map.setLayoutProperty(
                    id,
                    "visibility",
                    showWeather ? "visible" : "none",
                );
            }
        }
    }, [loaded, showWeather]);

    useEffect(() => {
        const map = mapRef.current;
        if (!loaded || !map?.getSource("herring")) return;

        const features =
              herring && showHerring
                  ? herring.features.filter(
                      (f) => f.properties.year === herringYear,
                  )
                : [];
        map.getSource("herring").setData({
            type: "FeatureCollection",
            features,
        });

        for (const id of HERRING_LAYER_IDS) {
            map.setLayoutProperty(
                id,
                "visibility",
                showHerring ? "visible" : "none",
            );
        }
    }, [loaded, herring, showHerring, herringYear]);

    return <div ref={containerRef} className={className} />;
}
