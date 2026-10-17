/**
 * @file WestCoastMap.jsx
 * @description Map component, focused on west coast of canada, static.
 */
import { useState, useEffect, useRef } from "react";
import { Map, setWorkerUrl } from "maplibre-gl";
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
const COAST_BOUNDS = [
    [-134, 46],
    [-120, 53],
];

const WEATHER_LAYER_IDS = ["weather-circles", "weather-temps", "weather-names"];
const REFRESH_MS = 15 * 60 * 1000;

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

// place ocean raster under basemap labels
function firstSymbolLayerId(map) {
    return map.getStyle().layers.find((l) => l.type === "symbol")?.id;
}

/**
 * Map component using maplibre
 * @param {object} props - component properties
 * @param {className} props.className - tailwind
 */
export default function WestCoastMap({
    className = "h-svh w-full",
    bounds = COAST_BOUNDS,
    overlay = null,
    overlayOpacity = 0.85,
    layers = {},
}) {
    const containerRef = useRef(null);
    const mapRef = useRef(null);
    const boundsRef = useRef(bounds);
    const [loaded, setLoaded] = useState(false);

    useEffect(() => {
        const map = new Map({
            container: containerRef.current,
            style: "https://tiles.openfreemap.org/styles/positron",
            bounds: COAST_BOUNDS,
            fitBoundsOptions: { padding: 20 },
            interactive: false, // static map
            attributionControl: { compact: true },
        });

        // const addRadar = () => {
        //     if (map.getSource("radar")) return; // protect against double-adding
        //     map.addSource("radar", {
        //         type: "raster",
        //         tiles: [RADAR_URL],
        //         tileSize: 256,
        //         attribution: "Environment and Climate Change Canada",
        //     });
        //     map.addLayer({
        //         id: "radar-layer",
        //         type: "raster",
        //         source: "radar",
        //         paint: { "raster-opacity": 0.7 },
        //     });
        // };
        mapRef.current = map;

        map.on("load", () => {
            addWeatherLayers(map);
            setLoaded(true);
        });

        // Resizing to map always fits
        const onResize = () =>
            map.fitBounds(COAST_BOUNDS, { padding: 20, animate: false });
        map.on("resize", onResize);

        return () => {
            map.off("resize", onResize);
            map.remove();
        }; // cleanup
    }, []);

    // refreand when bounds change (will be affected by metadata)
    const boundsKey = JSON.stringify(bounds);
    useEffect(() => {
        boundsRef.current = bounds;
        mapRef.current?.fitBounds(bounds, { padding: 20, animate: false });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [boundsKey]);

    // Ocean data overlay
    useEffect(() => {
        const map = mapRef.current;
        if (!loaded || !map || !overlay) return;

        const source = map.getSource("ocean");
        if (source) {
            source.updateImage({
                url: overlay.url,
                coordinates: overlay.coordinates,
            });
        } else {
            map.addSource("ocean", {
                type: "image",
                url: overlay.url,
                coordinates: overlay.coordinates,
            });
            map.addLayer(
                {
                    id: "ocean",
                    type: "raster",
                    source: "ocean",
                    paint: {
                        "raster-opacity": overlayOpacity,
                        "raster-fade-duration": 0,
                        "raster-resampling": "linear",
                    },
                },
                firstSymbolLayerId(map),
            );
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [loaded, overlay]);

    useEffect(() => {
        const map = mapRef.current;
        if (!loaded || !map?.getLayer("ocean")) return;
        map.setPaintProperty("ocean", "raster-opacity", overlayOpacity);
    }, [loaded, overlay, overlayOpacity]);

    // Weather: fetch once the map is ready, then refresh (only while it's shown)
    useEffect(() => {
        if (!loaded || !layers.weather) return;
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
    }, [loaded, layers.weather]);

    // Layer toggles
    useEffect(() => {
        const map = mapRef.current;
        if (!loaded || !map) return;
        const vis = (on) => (on ? "visible" : "none");
        for (const id of WEATHER_LAYER_IDS) {
            map.setLayoutProperty(id, "visibility", vis(layers.weather));
        }
        if (map.getLayer("ocean")) {
            map.setLayoutProperty(
                "ocean",
                "visibility",
                vis(layers.ocean !== false),
            );
        }
    }, [loaded, overlay, layers.weather, layers.ocean]);

    return <div ref={containerRef} className={className} />;
}
