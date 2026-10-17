// Current weather for BC coast towns via Open-Meteo (no API key, non-commercial use)

export const STATIONS = [
    { name: "Victoria", lat: 48.4284, lon: -123.3656 },
    { name: "Vancouver", lat: 49.2827, lon: -123.1207 },
    { name: "Squamish", lat: 49.7016, lon: -123.1558 },
    { name: "Nanaimo", lat: 49.1659, lon: -123.9401 },
    { name: "Port Alberni", lat: 49.2339, lon: -124.8055 },
    { name: "Tofino", lat: 49.153, lon: -125.9066 },
    { name: "Powell River", lat: 49.8352, lon: -124.5247 },
    { name: "Campbell River", lat: 50.0244, lon: -125.2475 },
    { name: "Port Hardy", lat: 50.7224, lon: -127.4969 },
    { name: "Bella Bella", lat: 52.1606, lon: -128.1456 },
    { name: "Bella Coola", lat: 52.372, lon: -126.754 },
    { name: "Kitimat", lat: 54.0524, lon: -128.6534 },
    { name: "Prince Rupert", lat: 54.315, lon: -130.3208 },
    { name: "Daajing Giids", lat: 53.254, lon: -132.08 },
    { name: "Masset", lat: 54.0118, lon: -132.1478 },
];

const WMO = {
    0: "Clear",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    56: "Freezing drizzle",
    57: "Freezing drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Freezing rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Showers",
    81: "Showers",
    82: "Heavy showers",
    85: "Snow showers",
    86: "Snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm",
    97: "Thunderstorm",
    99: "Thunderstorm",
};

// One request for all stations; returns a GeoJSON FeatureCollection
export async function fetchCurrentWeather(stations = STATIONS) {
    const params = new URLSearchParams({
        latitude: stations.map((s) => s.lat).join(","),
        longitude: stations.map((s) => s.lon).join(","),
        current:
            "temperature_2m,weather_code,wind_speed_10m,wind_direction_10m",
        timezone: "America/Vancouver",
    });

    const res = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`);
    if (!res.ok) throw new Error(`Open-Meteo ${res.status}`);

    const data = await res.json();
    const results = Array.isArray(data) ? data : [data]; // single location → object

    return {
        type: "FeatureCollection",
        features: results.map((r, i) => {
            const s = stations[i];
            const c = r.current;
            return {
                type: "Feature",
                // use the town's coords, not the model grid cell, so dots sit on the town
                geometry: { type: "Point", coordinates: [s.lon, s.lat] },
                properties: {
                    name: s.name,
                    temp: Math.round(c.temperature_2m),
                    condition: WMO[c.weather_code] ?? "",
                    wind: Math.round(c.wind_speed_10m),
                    windDir: c.wind_direction_10m,
                    time: c.time,
                },
            };
        }),
    };
}
