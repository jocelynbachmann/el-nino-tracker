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

function addWeatherLayers(map) {
    map.addSource("weather", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
        attribution:
            '<a href="https://open-meteo.com/">Weather data by Open-Meteo.com</a>',
    });

    map.addLayer({
        id: "weather-circles",
        type: "circle",
        source: "weather",
        paint: {
            "circle-radius": 15,
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
        data: "/herring-spawn.geojson",
        attribution:
            "Herring spawn: Fisheries and Oceans Canada (Open Government Licence – Canada)",
    });

    map.addLayer({
        id: "herring-spawn",
        type: "circle",
        source: "herring",
        layout: { visibility: "none" },
        filter: ["==", ["get", "year"], -1], // nothing until a year is set
        paint: {
            "circle-radius": 3,
            "circle-color": "#7c3aed",
            "circle-stroke-color": "#ffffff",
            "circle-stroke-width": 0.5,
            "circle-opacity": 0.9,
        },
    });
}

// place ocean raster under basemap labels
function firstSymbolLayerId(map) {
    return map
        .getStyle()
        .layers.find((l) => l.type === "symbol" && !l.id.startsWith("weather"))
        ?.id;
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
    // Create the map once
    useEffect(() => {
        const map = new MapLibreMap({
            container: containerRef.current,
            style: "https://tiles.openfreemap.org/styles/positron",
            bounds: boundsRef.current,
            fitBoundsOptions: { padding: 20 },
            interactive: false,
            attributionControl: { compact: true },
        });
        mapRef.current = map;
        const urls = urlsRef.current;

        const fit = (why) => {
            const el = containerRef.current;
            console.log(
                "fit:",
                why,
                el?.clientWidth,
                el?.clientHeight,
                boundsRef.current,
            );
            map.fitBounds(boundsRef.current, { padding: 20, animate: false });
        };

        // This is what already worked when resizing the window
        map.on("resize", () => fit("resize"));

        map.on("load", () => {
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

        const beforeId = firstSymbolLayerId(map);

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
        if (!loaded || !map?.getLayer("herring-spawn")) return;
        map.setFilter("herring-spawn", [
            "==",
            ["get", "year"],
            herringYear ?? -1,
        ]);
        map.setLayoutProperty(
            "herring-spawn",
            "visibility",
            showHerring ? "visible" : "none",
        );
    }, [loaded, showHerring, herringYear]);

    return <div ref={containerRef} className={className} />;
}
