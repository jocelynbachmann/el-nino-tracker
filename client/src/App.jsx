/*
 * @file @/App.jsx
 * TODO: a lot of this can be moved to a component
 */
import { useEffect, useMemo, useState } from "react";
import WestCoastMap from "@/components/WestCoastMap";
import EventTimeline from "@/components/EventTimeline";
import EventCard from "@/components/EventCard";
import MinimizeButton from "@/components/MinimizeButton";
import { OceanPanelHelp, PredictionsHelp, VariableInfo } from "@/components/HelpText";
import EventList from "@/components/EventList";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Separator } from "@/components/ui/separator";
import {
    useForecastOverlay,
    useMonths,
    useOceanMeta,
    useOceanOverlays,
} from "@/hooks/useOcean";
import { useEvents } from "@/hooks/useEvents";
import { useAnalysisMap } from "@/hooks/useAnalysisMap";
import { eventMapTime, formatTime, VARIABLE_WORDS } from "@/lib/eventText";
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
    disabled = false, // past the last observation: no prediction for this variable
    prediction = null, // { month, issued }: showing the model's prediction instead of observations
}) {
    const id = `layer-${variable.id}`;
    const gradient = cssGradient(getColourmap(variable.colormap, { anomaly }));

    return (
        <div
            className={`rounded-md px-2 py-1.5 transition-colors ${checked ? "bg-accent" : disabled ? "opacity-50" : "hover:bg-accent/60"}`}
        >
            <div className="flex items-center gap-1.5">
                <Label
                    htmlFor={id}
                    className={`flex min-w-0 flex-1 items-center gap-3 font-normal ${disabled || prediction ? "cursor-default" : "cursor-pointer"}`}
                >
                    <Checkbox
                        id={id}
                        checked={checked}
                        disabled={disabled || !!prediction}
                        onCheckedChange={(c) => onCheckedChange(c === true)}
                    />
                    <span className="flex-1 truncate">{variable.label}</span>
                    {prediction && (
                        <span className="rounded-full bg-amber-500/15 px-1.5 text-[10px] uppercase tracking-wide text-amber-700 dark:text-amber-400">
                            predicted
                        </span>
                    )}
                    {disabled && (
                        <span className="text-[10px] text-muted-foreground">
                            no prediction
                        </span>
                    )}
                    {checked && onTop && !prediction && (
                        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                            top
                        </span>
                    )}
                    {result?.loading && (
                        <span className="text-xs text-muted-foreground">…</span>
                    )}
                </Label>
                {/* outside the label so clicking it doesn't toggle the layer */}
                <VariableInfo id={variable.id} />
            </div>

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
                            {prediction
                                ? "No prediction for this month"
                                : "No data for this date"}
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
                    {prediction && (
                        <p className="text-xs text-muted-foreground">
                            Model prediction for {formatMonth(prediction.month)}{" "}
                            (monthly average), from the SEAS5 seasonal forecast
                            of {formatMonth(prediction.issued?.slice(0, 7))}.
                            Rough: treat as a guide.
                        </p>
                    )}
                </div>
            )}
        </div>
    );
}

const PREDICTED_VARIABLE = "mixed-layer-depth"; // the only variable the model predicts

// Times past the last observation that the MLD prediction covers, in the same form as `times`:
// every day (daily data) or every month, up to the end of the last predicted month
function predictionTimes(grid, lastObserved, daily) {
    if (!grid?.months?.length || !lastObserved) return [];
    const months = new Set(grid.months);
    if (!daily) return grid.months.filter((m) => m > lastObserved);
    const [y, m] = grid.months.at(-1).split("-").map(Number);
    const end = new Date(Date.UTC(y, m, 0)); // last day of the last predicted month
    const out = [];
    const d = new Date(`${lastObserved}T00:00:00Z`);
    d.setUTCDate(d.getUTCDate() + 1);
    for (; d <= end; d.setUTCDate(d.getUTCDate() + 1)) {
        const day = d.toISOString().slice(0, 10);
        if (months.has(day.slice(0, 7))) out.push(day);
    }
    return out;
}

export default function App() {
    const meta = useOceanMeta();
    const history = useEvents();

    const [selected, setSelected] = useState(DEFAULT_LAYERS);
    const [opacities, setOpacities] = useState({});
    const [anomaly, setAnomaly] = useState(false);
    const [pickedMonth, setPickedMonth] = useState(null);
    const [showWeather, setShowWeather] = useState(false);
    const [selectedEventId, setSelectedEventId] = useState(null);
    const [mapMode, setMapMode] = useState("day"); // for a selected event: day | onset | duration | typical
    const [showHerring, setShowHerring] = useState(false);
    const [panelOpen, setPanelOpen] = useState(true); // left "Ocean data" panel

    // Height of the history strip, so the event card can stop above it (changes when minimized)
    const [stripEl, setStripEl] = useState(null);
    const [stripHeight, setStripHeight] = useState(180);
    useEffect(() => {
        if (!stripEl) return;
        const ro = new ResizeObserver(() =>
                                     setStripHeight(stripEl.getBoundingClientRect().height),
        );
        ro.observe(stripEl);
        return () => ro.disconnect();
    }, [stripEl]);
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
    const observedTimes = daily ? meta.observedDays : allMonths;

    // Predictions: the MLD model's monthly forecast extends the timeline past the last observation
    const forecastGrid = meta.forecastGrid;
    const lastObserved = daily
        ? meta.observedDays.at(-1)
        : (meta.observedMonths?.at(-1) ?? null);
    const futureTimes = useMemo(
        () => predictionTimes(forecastGrid, lastObserved, daily),
        [forecastGrid, lastObserved, daily],
    );
    const [showPredictions, setShowPredictions] = useState(false);
    const months = useMemo(
        () =>
            showPredictions && futureTimes.length
                ? [...observedTimes, ...futureTimes]
                : observedTimes,
        [showPredictions, futureTimes, observedTimes],
    );
    const monthPos = useMemo(
        () => new Map(months.map((m, i) => [m, i])),
        [months],
    );
    const defaultMonth = daily
        ? months.at(-1)
        : (meta.observedMonths?.at(-1) ?? months.at(-1) ?? null);
    const month = monthPos.has(pickedMonth) ? pickedMonth : defaultMonth;
    const herringYear = month ? Number(month.slice(0, 4)) : null;
    const mapMonth = useDebounced(month, 150); // don't fetch every day the slider passes over

    // Past the last observation only the MLD prediction exists: the other layers pause
    const inFuture = !!(lastObserved && month && month > lastObserved);
    const mapInFuture = !!(lastObserved && mapMonth && mapMonth > lastObserved);

    const results = useOceanOverlays({
        variables: mapInFuture ? [] : selectedVariables,
        month: mapMonth,
        anomaly,
    });

    const opacityOf = (id) => opacities[id] ?? DEFAULT_OPACITY;

    const forecastResult = useForecastOverlay({
        month: mapInFuture ? mapMonth.slice(0, 7) : null,
        anomaly,
        variable: meta.variables?.find((v) => v.id === PREDICTED_VARIABLE),
    });

    const togglePredictions = (on) => {
        setShowPredictions(on);
        if (on && futureTimes.length) setPickedMonth(futureTimes[0]); // show it straight away
    };

    // The history strip's region-wide spells follow the top map layer (if it was analysed)
    const spellVariable =
        [...selected].reverse().find((id) => VARIABLE_WORDS[id]) ??
        "temperature";

    // El Niño history: selecting an event jumps to when it hit hardest here, as an anomaly map
    const ensoList = useMemo(
        () => Array.from(meta.elNinoByMonth?.values() ?? []),
        [meta.elNinoByMonth],
    );
    const selectedEvent =
        history.events.find((e) => e.id === selectedEventId) ?? null;
    const clampMonth = (m) =>
        monthPos.has(m) ? m : m > months.at(-1) ? months.at(-1) : months[0];
    const eventMonth = selectedEvent
        ? clampMonth(eventMapTime(selectedEvent, daily ? "day" : "month"))
        : null;
    // jump: go to when the event hit hardest (from the list); a click on the timeline keeps the
    // date that was clicked instead
    const selectEvent = (id, { jump = true } = {}) => {
        setSelectedEventId(id);
        const ev = history.events.find((e) => e.id === id);
        if (ev && jump) {
            setPickedMonth(
                clampMonth(eventMapTime(ev, daily ? "day" : "month")),
            );
            // "Show anomaly" stays as the user set it (it used to switch on with every event opened)
        }
    };

    // Analysis maps for the selected event (start / duration / past events) replace the day's layers
    const analysisMode = selectedEvent ? mapMode : "day";
    const analysis = useAnalysisMap({
        mode: analysisMode,
        event: selectedEvent,
        variable: spellVariable,
    });

    const overlays = useMemo(() => {
        if (analysisMode !== "day") return analysis.overlays ?? [];
        if (mapInFuture) {
            return forecastResult.overlay
                ? [
                      {
                          id: "mld-prediction",
                          ...forecastResult.overlay,
                          opacity: opacityOf(PREDICTED_VARIABLE) / 100,
                      },
                  ]
                : [];
        }
        return selected
            .filter((id) => results[id]?.overlay)
            .map((id) => ({
                id,
                ...results[id].overlay,
                opacity: opacityOf(id) / 100,
            }));
    }, [selected, results, opacities, analysisMode, analysis.overlays, mapInFuture, forecastResult.overlay]); // eslint-disable-line react-hooks/exhaustive-deps

    const toggleLayer = (id, on) =>
        setSelected((prev) => {
            const rest = prev.filter((x) => x !== id);
            return on ? [...rest, id] : rest; // newly checked layers go on top
        });

    const monthIndex = monthPos.get(month) ?? 0;
    const isForecast =
        inFuture ||
        (!daily &&
            month &&
            meta.observedMonths &&
            !meta.observedMonths.includes(month));
    const enso = meta.elNinoByMonth?.get(month?.slice(0, 7));
    const forecast = isForecast ? meta.forecastByMonth?.get(month) : null;
    const loading =
        meta.loading ||
        Object.values(results).some((r) => r.loading) ||
        (mapInFuture && forecastResult.loading);

    return (
        <div className="relative h-svh w-full">
            <WestCoastMap
                className="h-full w-full"
                overlays={overlays}
                showWeather={showWeather}
                showHerring={showHerring}
                herringYear={herringYear}
            />

            <div className="absolute right-4 top-4 z-10 max-h-[calc(100svh-2rem)] w-72 space-y-4 overflow-y-auto rounded-lg bg-background/90 p-4 shadow-md backdrop-blur">
                <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5">
                        <h2 className="text-sm font-medium">Ocean data</h2>
                        <OceanPanelHelp />
                    </div>
                    <div className="flex items-center gap-2">
                        {loading && (
                            <span className="text-xs text-muted-foreground">
                                Loading…
                            </span>
                        )}
                        {!panelOpen && month && (
                            <span className="text-xs text-muted-foreground">
                                {formatMonth(month)}
                            </span>
                        )}
                        <MinimizeButton
                            open={panelOpen}
                            onToggle={() => setPanelOpen((o) => !o)}
                            label="ocean data panel"
                        />
                    </div>
                </div>

                {!panelOpen ? null : meta.error ? (
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
                                        {inFuture ? "Prediction" : "Forecast"}
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

                            {futureTimes.length > 0 && (
                                <div className="flex items-center justify-between gap-3">
                                    <div className="grid gap-0.5">
                                        <div className="flex items-center gap-1.5">
                                            <Label
                                                htmlFor="predictions"
                                                className="font-normal"
                                            >
                                                Predictions
                                            </Label>
                                            <PredictionsHelp
                                                from={futureTimes[0]}
                                                until={futureTimes.at(-1)}
                                                issued={forecastGrid?.issued}
                                            />
                                        </div>
                                        <span className="text-xs text-muted-foreground">
                                            Mixed layer depth to{" "}
                                            {formatMonth(futureTimes.at(-1).slice(0, 7))}
                                        </span>
                                    </div>
                                    <Switch
                                        id="predictions"
                                        checked={showPredictions}
                                        onCheckedChange={(on) =>
                                            togglePredictions(on === true)
                                        }
                                    />
                                </div>
                            )}

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

                        <EventList
                            events={history.events}
                            selectedId={selectedEventId}
                            onSelect={(id) =>
                                id ? selectEvent(id) : setSelectedEventId(null)
                            }
                        />

                        <Separator />

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
                            {inFuture && (
                                <p className="mb-1 text-xs text-muted-foreground">
                                    Past {formatMonth(lastObserved)} there are
                                    only predictions, for mixed layer depth.
                                </p>
                            )}
                            {meta.variables?.map((v) => {
                                const predicted =
                                    inFuture && v.id === PREDICTED_VARIABLE;
                                return (
                                    <LayerRow
                                        key={v.id}
                                        variable={v}
                                        checked={
                                            inFuture
                                                ? predicted
                                                : selected.includes(v.id)
                                        }
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
                                        result={
                                            predicted
                                                ? forecastResult
                                                : results[v.id]
                                        }
                                        anomaly={anomaly}
                                        onTop={
                                            !inFuture &&
                                            selected.length > 1 &&
                                            selected.at(-1) === v.id
                                        }
                                        disabled={inFuture && !predicted}
                                        prediction={
                                            predicted
                                                ? {
                                                      month: month.slice(0, 7),
                                                      issued: forecastGrid?.issued,
                                                  }
                                                : null
                                        }
                                    />
                                );
                            })}
                        </div>

                        <Separator />

                        <Toggle
                            id="layer-herring"
                            label="Herring spawn"
                            hint={`DFO, (${herringYear ?? "–"})`}
                            checked={showHerring}
                            onChange={setShowHerring}
                        />
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
                    className="absolute left-4 top-4 z-10 w-80 overflow-y-auto"
                    style={{
                        maxHeight: `calc(100svh - ${Math.round(stripHeight) + 48}px)`,
                    }}
                    event={selectedEvent}
                    peakTime={eventMonth}
                    mapTime={month}
                    onShowOnMap={setPickedMonth}
                    onClose={() => {
                        setSelectedEventId(null);
                        setMapMode("day");
                    }}
                    mapMode={mapMode}
                    onMapModeChange={setMapMode}
                    analysisVariable={spellVariable}
                    analysis={analysis}
                />
            )}

            {!meta.error && months.length > 0 && (
                <div
                    ref={setStripEl}
                    className="absolute bottom-4 right-80 left-4 z-10"
                >
                    <EventTimeline
                        enso={ensoList}
                        events={history.events}
                        times={months}
                        time={month}
                        onTimeChange={setPickedMonth}
                        selectedId={selectedEventId}
                        onSelectEvent={(id) => selectEvent(id, { jump: false })}
                        spellVariable={spellVariable}
                        regionalSpells={
                            history.regionalSpells?.[spellVariable] ?? []
                        }
                        predictions={
                            futureTimes.length
                                ? { on: showPredictions, from: futureTimes[0] }
                                : null
                        }
                    />
                </div>
            )}
        </div>
    );
}
