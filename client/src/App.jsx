import { useEffect, useMemo, useState } from "react";
import WestCoastMap from "@/components/WestCoastMap";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { useMonths, useOceanMeta, useOceanOverlay } from "@/hooks/useOcean";
import { cssGradient, getColormap } from "@/lib/colourmap";
import { PHASE_LABEL, toBounds } from "@/lib/oceanApi";

const DEFAULT_VARIABLE = "temperature";

const sliderValue = (v) => (Array.isArray(v) ? v[0] : v); // Radix passes arrays, Base UI may not

function formatMonth(m) {
    if (!m) return "";
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

function Legend({ variable, anomaly, range }) {
    if (!variable || !range) return null;
    const fn = getColormap(variable.colormap, { anomaly });
    const [min, max] = range;
    const span = Math.abs(max - min);
    const units = variable.units ? ` ${variable.units}` : "";

    return (
        <div className="space-y-1">
            <div
                className="h-2.5 w-full rounded-sm"
                style={{ background: cssGradient(fn) }}
            />
            <div className="flex justify-between text-xs tabular-nums text-muted-foreground">
                <span>
                    {formatValue(min, span)}
                    {units}
                </span>
                {anomaly && <span>normal</span>}
                <span>
                    {anomaly && max > 0 ? "+" : ""}
                    {formatValue(max, span)}
                    {units}
                </span>
            </div>
        </div>
    );
}

export default function App() {
    const meta = useOceanMeta();

    const [variableId, setVariableId] = useState(DEFAULT_VARIABLE);
    const [anomaly, setAnomaly] = useState(false);
    const [modelRegion, setModelRegion] = useState(false);
    const [month, setMonth] = useState(null);
    const [opacity, setOpacity] = useState(85);
    const [layers, setLayers] = useState({ ocean: true, weather: false });

    const variable = useMemo(
        () =>
            meta.variables?.find((v) => v.id === variableId) ??
            meta.variables?.[0],
        [meta.variables, variableId],
    );
    const months = useMonths(meta, variable);

    // Default to the latest observed month; keep the current month if the new variable has it
    useEffect(() => {
        if (!months.length) return;
        if (!month || !months.includes(month)) {
            setMonth(meta.observedMonths.at(-1) ?? months.at(-1));
        }
    }, [months]); // eslint-disable-line react-hooks/exhaustive-deps

    const ocean = useOceanOverlay({
        variable,
        month,
        anomaly,
        area: modelRegion ? "model-region" : undefined,
        enabled: layers.ocean,
    });

    const bounds = useMemo(() => {
        if (modelRegion) {
            return (
                toBounds(meta.meta?.model_region) ?? [
                    [-134, 46],
                    [-120, 52],
                ]
            );
        }
        return meta.bounds ?? undefined;
    }, [modelRegion, meta.meta, meta.bounds]);

    const monthIndex = Math.max(0, months.indexOf(month));
    const isForecast =
        ocean.source === "forecast" ||
        (month && !meta.observedMonths?.includes(month));
    const enso = meta.elNinoByMonth?.get(month);
    const forecast = isForecast ? meta.forecastByMonth?.get(month) : null;

    return (
        <div className="relative h-svh w-full">
            <WestCoastMap
                className="h-full w-full"
                bounds={bounds}
                overlay={ocean.overlay}
                overlayOpacity={opacity / 100}
                layers={layers}
            />

            <div className="absolute left-4 top-4 z-10 max-h-[calc(100svh-2rem)] w-72 space-y-4 overflow-y-auto rounded-lg bg-background/90 p-4 shadow-md backdrop-blur">
                <div className="flex items-baseline justify-between">
                    <h2 className="text-sm font-medium">Ocean data</h2>
                    {(meta.loading || ocean.loading) && (
                        <span className="text-xs text-muted-foreground">
                            Loading…
                        </span>
                    )}
                </div>

                {meta.error ? (
                    <p className="text-sm text-destructive">
                        Can't reach the ocean server. Is Flask running on port
                        5001?
                        <span className="mt-1 block text-xs text-muted-foreground">
                            {meta.error.message}
                        </span>
                    </p>
                ) : (
                    <>
                        {/* Variable */}
                        <div className="grid gap-1.5">
                            <Label htmlFor="variable" className="font-normal">
                                Variable
                            </Label>
                            <select
                                id="variable"
                                value={variable?.id ?? ""}
                                onChange={(e) => setVariableId(e.target.value)}
                                disabled={!meta.variables?.length}
                                className="h-9 w-full rounded-md border border-input bg-transparent px-2 text-sm shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                            >
                                {meta.variables?.map((v) => (
                                    <option key={v.id} value={v.id}>
                                        {v.label}
                                        {v.units ? ` (${v.units})` : ""}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <Toggle
                            id="anomaly"
                            label="Show anomaly"
                            hint="Difference from the normal for that month"
                            checked={anomaly}
                            onChange={setAnomaly}
                        />
                        <Toggle
                            id="model-region"
                            label="Model region only"
                            hint="46–52°N, 134–120°W"
                            checked={modelRegion}
                            onChange={setModelRegion}
                        />

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
                                    setMonth(months[sliderValue(v)])
                                }
                                min={0}
                                max={Math.max(0, months.length - 1)}
                                step={1}
                                disabled={months.length < 2}
                            />
                            <div className="flex justify-between text-xs text-muted-foreground">
                                <span>{formatMonth(months[0])}</span>
                                <span>{formatMonth(months.at(-1))}</span>
                            </div>

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

                        <Legend
                            variable={variable}
                            anomaly={anomaly}
                            range={ocean.range}
                        />

                        {ocean.error && (
                            <p className="text-xs text-destructive">
                                {ocean.error.message}
                            </p>
                        )}
                    </>
                )}

                <Separator />

                <div className="space-y-3">
                    <h2 className="text-sm font-medium">Layers</h2>
                    <Toggle
                        id="layer-ocean"
                        label="Ocean data"
                        checked={layers.ocean}
                        onChange={(on) =>
                            setLayers((l) => ({ ...l, ocean: on }))
                        }
                    />
                    <Toggle
                        id="layer-weather"
                        label="Current weather"
                        hint="Test data from Open-Meteo"
                        checked={layers.weather}
                        onChange={(on) =>
                            setLayers((l) => ({ ...l, weather: on }))
                        }
                    />

                    <div className="space-y-2 pt-1">
                        <div className="flex justify-between text-sm">
                            <span>Opacity</span>
                            <span className="tabular-nums text-muted-foreground">
                                {opacity}%
                            </span>
                        </div>
                        <Slider
                            value={[opacity]}
                            onValueChange={(v) => setOpacity(sliderValue(v))}
                            min={10}
                            max={100}
                            step={5}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
}
