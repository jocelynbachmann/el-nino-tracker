/*
 * @file @/App.jsx
 * TODO: a lot of this can be moved to a component
 */
import { useEffect, useMemo, useState } from "react";
import WestCoastMap from "@/components/WestCoastMap";
import EventTimeline from "@/components/EventTimeline";
import EventCard from "@/components/EventCard";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import { useMonths, useOceanMeta, useOceanOverlays } from "@/hooks/useOcean";
import { useEvents } from "@/hooks/useEvents";
import { eventMapTime, formatTime } from "@/lib/eventText";
import { cssGradient, getColourmap } from "@/lib/colourmap";
import { PHASE_LABEL } from "@/lib/oceanApi";

const DEFAULT_LAYERS = ["mixed-layer-depth"];
const DEFAULT_OPACITY = 85;

const sliderValue = (v) => (Array.isArray(v) ? v[0] : v); // Radix passes arrays, Base UI may not

// Waits until a value stops changing (e.g. while dragging the slider) before passing it on
function useDebounced(value, ms) {
    const [v, setV] = useState(value);
    useEffect(() => {
        const id = setTimeout(() => setV(value), ms);
        return () => clearTimeout(id);
    }, [value, ms]);
    return v;
}

function formatMonth(m) {
    if (!m) return "";
    if (m.length > 7) return formatTime(m); // a day: "15 Nov 2015"
    const [y, mo] = m.split("-").map(Number);
    return new Date(y, mo - 1).toLocaleString("en-CA", {
        month: "short",
        year: "numeric",
    });
}

function formatValue(v, span) {
    if (v == null || !Number.isFinite(v)) return "–";
    const digits = span >= 50 ? 0 : span >= 5 ? 1 : 2;
    return v.toFixed(digits);
}

function Toggle({ id, label, checked, onChange, hint }) {
    return (
        <div className="flex items-center justify-between gap-3">
            <div className="grid gap-0.5">
                <Label htmlFor={id} className="font-normal">
                    {label}
                </Label>
                {hint && (
                    <span className="text-xs text-muted-foreground">
                        {hint}
                    </span>
                )}
            </div>
            <Switch id={id} checked={checked} onCheckedChange={onChange} />
        </div>
    );
}

function LegendScale({ range, units, anomaly }) {
    if (!range) return null;
    const [min, max] = range;
    const span = Math.abs(max - min);
    const u = units ? ` ${units}` : "";
    return (
        <div className="flex justify-between text-xs tabular-nums text-muted-foreground">
            <span>
                {formatValue(min, span)}
                {u}
            </span>
            {anomaly && <span>normal</span>}
            <span>
                {anomaly && max > 0 ? "+" : ""}
                {formatValue(max, span)}
                {u}
            </span>
        </div>
    );
}

function LayerRow({
    variable,
    checked,
    onCheckedChange,
    opacity,
    onOpacityChange,
    result,
    anomaly,
    onTop,
}) {
    const id = `layer-${variable.id}`;
    const gradient = cssGradient(getColourmap(variable.colormap, { anomaly }));

    return (
        <div
            className={`rounded-md px-2 py-1.5 transition-colors ${checked ? "bg-accent" : "hover:bg-accent/60"}`}
        >
            <Label
                htmlFor={id}
                className="flex cursor-pointer items-center gap-3 font-normal"
            >
                <Checkbox
                    id={id}
                    checked={checked}
                    onCheckedChange={(c) => onCheckedChange(c === true)}
                />
                <span className="flex-1 truncate">{variable.label}</span>
                {checked && onTop && (
                    <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        top
                    </span>
                )}
                {result?.loading && (
                    <span className="text-xs text-muted-foreground">…</span>
                )}
            </Label>

            {checked && (
                <div className="mt-2 space-y-2 pl-7">
                    <div
                        className="h-2 w-full rounded-sm"
                        style={{ background: gradient, opacity: opacity / 100 }}
                    />
                    <LegendScale
                        range={result?.range}
                        units={variable.units}
                        anomaly={anomaly}
                    />

                    {result?.error ? (
                        <p className="text-xs text-destructive">
                            No data for this date
                        </p>
                    ) : (
                        <div className="flex items-center gap-3">
                            <Slider
                                value={[opacity]}
                                onValueChange={(v) =>
                                    onOpacityChange(sliderValue(v))
                                }
                                min={10}
                                max={100}
                                step={5}
                                aria-label={`${variable.label} opacity`}
                            />
                            <span className="w-9 text-right text-xs tabular-nums text-muted-foreground">
                                {opacity}%
                            </span>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
export default function App() {
    const meta = useOceanMeta();
    const history = useEvents();

    const bounds = meta.bounds ?? undefined;
    const [selected, setSelected] = useState(DEFAULT_LAYERS);
    const [opacities, setOpacities] = useState({});
    const [anomaly, setAnomaly] = useState(false);
    const [pickedMonth, setPickedMonth] = useState(null);
    const [showWeather, setShowWeather] = useState(false);
    const [selectedEventId, setSelectedEventId] = useState(null);
    const selectedVariables = useMemo(
        () =>
            selected
                .map((id) => meta.variables?.find((v) => v.id === id))
                .filter(Boolean),
        [selected, meta.variables],
    );
    const allMonths = useMonths(meta, selectedVariables);

    // The whole record is available (the history timeline is for browsing past events), in days
    // when the server has the daily data, otherwise in months. The latest one is the default.
    // ("month" below holds whichever: "YYYY-MM-DD" or "YYYY-MM".)
    const daily = meta.timeStep === "day" && meta.observedDays?.length > 0;
    const months = daily ? meta.observedDays : allMonths;
    const monthPos = useMemo(() => new Map(months.map((m, i) => [m, i])), [months]);
    const defaultMonth = daily
        ? months.at(-1)
        : (meta.observedMonths?.at(-1) ?? months.at(-1) ?? null);
    const month = monthPos.has(pickedMonth) ? pickedMonth : defaultMonth;
    const mapMonth = useDebounced(month, 150); // don't fetch every day the slider passes over

    const results = useOceanOverlays({
        variables: selectedVariables,
        month: mapMonth,
        anomaly,
    });

    const opacityOf = (id) => opacities[id] ?? DEFAULT_OPACITY;

    // El Niño history: selecting an event jumps to when it hit hardest here, as an anomaly map
    const ensoList = useMemo(
        () => Array.from(meta.elNinoByMonth?.values() ?? []),
        [meta.elNinoByMonth],
    );
    const selectedEvent = history.events.find((e) => e.id === selectedEventId) ?? null;
    const clampMonth = (m) =>
        monthPos.has(m) ? m : m > months.at(-1) ? months.at(-1) : months[0];
    const eventMonth = selectedEvent
        ? clampMonth(eventMapTime(selectedEvent, daily ? "day" : "month"))
        : null;
    const selectEvent = (id) => {
        setSelectedEventId(id);
        const ev = history.events.find((e) => e.id === id);
        if (ev) {
            setPickedMonth(clampMonth(eventMapTime(ev, daily ? "day" : "month")));
            setAnomaly(true);
        }
    };

    const overlays = useMemo(
        () =>
            selected
                .filter((id) => results[id]?.overlay)
                .map((id) => ({
                    id,
                    ...results[id].overlay,
                    opacity: opacityOf(id) / 100,
                })),
        [selected, results, opacities], // eslint-disable-line react-hooks/exhaustive-deps
    );

    const toggleLayer = (id, on) =>
        setSelected((prev) => {
            const rest = prev.filter((x) => x !== id);
            return on ? [...rest, id] : rest; // newly checked layers go on top
        });

    const monthIndex = monthPos.get(month) ?? 0;
    const isForecast =
        !daily && month && meta.observedMonths && !meta.observedMonths.includes(month);
    const enso = meta.elNinoByMonth?.get(month?.slice(0, 7));
    const forecast = isForecast ? meta.forecastByMonth?.get(month) : null;
    const loading =
        meta.loading || Object.values(results).some((r) => r.loading);

    return (
        <div className="relative h-svh w-full">
            <WestCoastMap
                className="h-full w-full"
                overlays={overlays}
                bounds={bounds}
                showWeather={showWeather}
            />

            <div className="absolute left-4 top-4 z-10 max-h-[calc(100svh-2rem)] w-72 space-y-4 overflow-y-auto rounded-lg bg-background/90 p-4 shadow-md backdrop-blur">
                <div className="flex items-baseline justify-between">
                    <h2 className="text-sm font-medium">Ocean data</h2>
                    {loading && (
                        <span className="text-xs text-muted-foreground">
                            Loading…
                        </span>
                    )}
                </div>

                {meta.error ? (
                    <p className="text-sm text-destructive">
                        Can't reach the ocean server.
                        <span className="mt-1 block text-xs text-muted-foreground">
                            {meta.error.message}
                        </span>
                    </p>
                ) : (
                    <>
                        {/* Month */}
                        <div className="space-y-3">
                            <div className="flex items-center justify-between text-sm">
                                <span className="font-medium">
                                    {formatMonth(month)}
                                </span>
                                {isForecast && (
                                    <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-xs font-medium text-amber-700 dark:text-amber-400">
                                        Forecast
                                    </span>
                                )}
                            </div>
                            <Slider
                                value={[monthIndex]}
                                onValueChange={(v) =>
                                    setPickedMonth(months[sliderValue(v)])
                                }
                                min={0}
                                max={Math.max(0, months.length - 1)}
                                step={1}
                                disabled={months.length < 2}
                                aria-label={daily ? "Date" : "Month"}
                            />
                            <div className="flex justify-between text-xs text-muted-foreground">
                                <span>{formatMonth(months[0])}</span>
                                <span>{formatMonth(months.at(-1))}</span>
                            </div>

                            {/* NOTE: even needed? */}
                            {/*
                            {enso && (
                                <p className="text-xs text-muted-foreground">
                                    ENSO:{" "}
                                    <span className="text-foreground">
                                        {PHASE_LABEL[enso.phase] ?? enso.phase}
                                    </span>
                                    {enso.oni != null &&
                                        ` (ONI ${Number(enso.oni).toFixed(1)})`}
                                    {enso.provisional && " · provisional"}
                                </p>
                            )}
                              */}

                            {forecast && (
                                <p className="text-xs text-muted-foreground">
                                    Regional MLD{" "}
                                    <span className="text-foreground">
                                        {forecast.mld?.toFixed(1)} m
                                    </span>
                                    {forecast.lower != null &&
                                        ` (${forecast.lower.toFixed(0)}–${forecast.upper.toFixed(0)} m)`}
                                    {" · "}
                                    {forecast.source === "model"
                                        ? "model forecast"
                                        : "climatology"}
                                </p>
                            )}
                        </div>

                        <Toggle
                            id="anomaly"
                            label="Show anomaly"
                            hint={
                                daily
                                    ? "Difference from each spot's normal for that time of year"
                                    : "Difference from the normal for that month"
                            }
                            checked={anomaly}
                            onChange={setAnomaly}
                        />

                        <Separator />

                        {/* Layers */}
                        <div className="space-y-1">
                            <h2 className="mb-2 text-sm font-medium">Layers</h2>
                            {meta.variables?.map((v) => (
                                <LayerRow
                                    key={v.id}
                                    variable={v}
                                    checked={selected.includes(v.id)}
                                    onCheckedChange={(on) =>
                                        toggleLayer(v.id, on)
                                    }
                                    opacity={opacityOf(v.id)}
                                    onOpacityChange={(o) =>
                                        setOpacities((prev) => ({
                                            ...prev,
                                            [v.id]: o,
                                        }))
                                    }
                                    result={results[v.id]}
                                    anomaly={anomaly}
                                    onTop={
                                        selected.length > 1 &&
                                        selected.at(-1) === v.id
                                    }
                                />
                            ))}
                        </div>
                    </>
                )}
                {/*

  <Separator />


                        <Toggle
                        id="layer-weather"
                        label="Current weather"
                        hint="Test data from Open-Meteo"
                        checked={showWeather}
                        onChange={setShowWeather}
                        />


                  */}
            </div>

            {selectedEvent && (
                <EventCard
                    className="absolute right-4 top-4 z-10 max-h-[calc(100svh-10rem)] w-80 overflow-y-auto"
                    event={selectedEvent}
                    peakTime={eventMonth}
                    mapTime={month}
                    onShowOnMap={setPickedMonth}
                    onClose={() => setSelectedEventId(null)}
                />
            )}

            {!meta.error && months.length > 0 && (
                <EventTimeline
                    className="absolute bottom-4 left-80 right-4 z-10"
                    enso={ensoList}
                    events={history.events}
                    times={months}
                    time={month}
                    onTimeChange={setPickedMonth}
                    selectedId={selectedEventId}
                    onSelectEvent={selectEvent}
                />
            )}
        </div>
    );
}
