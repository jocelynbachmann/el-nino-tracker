/**
 * @file WestCoastMap.jsx
 * @description Map component, focused on west coast of canada, static.
 */
import { useEffect, useRef } from "react";
import { Map, setWorkerUrl } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "maplibre-gl/dist/maplibre-gl.css";

// Maplibre's web worker
setWorkerUrl(workerUrl);

// [[west, south], [east, north]]
const COAST_BOUNDS = [
    [-134, 46],
    [-120, 53],
];

/**
 * Map component using maplibre
 * @param {object} props - component properties
 * @param {className} props.className - tailwind
 */
export default function WestCoastMap({ className = "h-svh w-full" }) {
    const containerRef = useRef(null);

    useEffect(() => {
        const map = new Map({
            container: containerRef.current,
            style: "https://tiles.openfreemap.org/styles/positron",
            bounds: COAST_BOUNDS,
            fitBoundsOptions: { padding: 20 },
            interactive: false, // static map
            attributionControl: { compact: true },
        });

        // Resizing to map always fits
        const onResize = () =>
            map.fitBounds(COAST_BOUNDS, { padding: 20, animate: false });
        map.on("resize", onResize);

        return () => {
            map.off("resize", onResize);
            map.remove();
        };
    }, []);

    return <div ref={containerRef} className={className} />;
}
