/**
 * @file EventCard.jsx
 * @description Plain-language summary of one El Niño / La Niña: when it ran and what it did to
 * each variable across the region (from /events).
 */
import { describeEffect, formatTime, formatYm, PHASE_NAME, VARIABLE_WORDS } from "@/lib/eventText";
import { cssGradient, getColourmap } from "@/lib/colourmap";
import { useState } from "react";
import MinimizeButton from "@/components/MinimizeButton";
import { EventHelp } from "@/components/HelpText";

const ORDER = ["temperature", "sea-surface-height", "salinity", "mixed-layer-depth"];
const DOT = { 1: "#b2182b", "-1": "#2166ac", 0: "var(--muted-foreground)" };

// What the map can show for an event (besides the ocean on the chosen day)
function modesFor(event) {
    const plural = event.phase === "la_nina" ? "La Niñas" : "El Niños";
    return [
        { id: "day", label: "On the day" },
        { id: "onset", label: "Start" },
        { id: "duration", label: "Duration" },
        { id: "typical", label: `Past ${plural}` },
    ];
}

function MapLegend({ mode, event, variable, analysis }) {
    if (mode === "day" || !analysis) return null;
    const w = VARIABLE_WORDS[variable];
    const name = PHASE_NAME[event.phase];
    const plural = event.phase === "la_nina" ? "La Niñas" : "El Niños";
    const gradient = analysis.colormap ? cssGradient(getColourmap(analysis.colormap)) : null;
    const text = {
        onset: {
            title: `When ${w.label.toLowerCase()} changed at each spot`,
            left: "−6 mo",
            mid: "begins",
            right: "+18 mo",
            note: `Months from when ${name} began.`,
        },
        duration: {
            title: `How long the change in ${w.label.toLowerCase()} lasted`,
            left: "0",
            mid: "",
            right: "30+ months",
            note: "",
        },
        typical: {
            title: `${w.label}: same change in past ${plural}?`,
            left: `${w.down} in all`,
            mid: "no pattern",
            right: `${w.up} in all`,
            note: analysis.nEvents ? `Based on the ${analysis.nEvents} completed ${plural} since 2002.` : "",
        },
    }[mode];
    // where "event begins" sits on the onset scale (-6..18 months)
    const midPos = mode === "onset" ? "25%" : "50%";

    return (
        <div className="mt-2 space-y-1">
            <p className="text-[11px] font-medium">{text.title}</p>
            {analysis.loading && <p className="text-[11px] text-muted-foreground">Loading…</p>}
            {analysis.error && <p className="text-[11px] text-destructive">Couldn't load this map.</p>}
            {gradient && (
                <>
                    <div className="h-2 w-full rounded-sm" style={{ background: gradient }} />
                    <div className="relative h-4 text-[10px] text-muted-foreground">
                        <span className="absolute left-0">{text.left}</span>
                        {text.mid && (
                            <span className="absolute -translate-x-1/2" style={{ left: midPos }}>
                                {text.mid}
                            </span>
                        )}
                        <span className="absolute right-0">{text.right}</span>
                    </div>
                </>
            )}
            {analysis.hasNoChange && (
                <p className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
                    <span
                        className="inline-block h-3 w-4 rounded-sm border border-border"
                        style={{
                            background:
                                "repeating-linear-gradient(45deg, rgb(120,113,108) 0 1.5px, transparent 1.5px 4.5px)",
                        }}
                    />
                    Striped: checked, no clear change there
                </p>
            )}
            {analysis.empty ? (
                <p className="text-[11px] text-muted-foreground">
                    {event.ongoing ? "Too early to tell for this event." : "No clear change anywhere for this variable."}
                </p>
            ) : (
                text.note && <p className="text-[10px] text-muted-foreground">{text.note}</p>
            )}
        </div>
    );
}

// peakTime: the day (or month) to jump to: ONI peak + the region's response lag, clamped to the data
// mapMode / onMapModeChange: what the map shows (see modesFor); analysis: useAnalysisMap() state
export default function EventCard({
    event,
    peakTime,
    mapTime,
    onShowOnMap,
    onClose,
    mapMode = "day",
    onMapModeChange,
    analysisVariable = "temperature",
    analysis,
    className = "",
    style,
}) {
    const [open, setOpen] = useState(true);
    if (!event) return null;
    const name = PHASE_NAME[event.phase] ?? "Event";
    const showingPeak = mapTime === peakTime;

    return (
        <div className={`rounded-lg bg-background/90 p-3 shadow-md backdrop-blur ${className}`} style={style}>
            <div className="flex items-start justify-between gap-3">
                <div>
                    <h2 className="text-sm font-semibold">
                        {event.label.replace(" (ongoing)", "")} {name}
                    </h2>
                    <p className="text-[11px] text-muted-foreground">
                        {formatYm(event.start)} – {event.ongoing ? "now" : formatYm(event.end)} · {event.strength}
                        {event.ongoing ? " so far" : ""} · {event.ongoing ? "highest" : "peak"} ONI{" "}
                        {event.peak_oni > 0 ? "+" : ""}
                        {event.peak_oni} in {formatYm(event.peak_month)}
                    </p>
                </div>
                <div className="-mr-1 -mt-1 flex items-center gap-0.5">
                    <EventHelp
                        phaseName={name}
                        plural={event.phase === "la_nina" ? "La Niñas" : "El Niños"}
                    />
                    <MinimizeButton open={open} onToggle={() => setOpen((o) => !o)} label="event card" />
                <button
                    type="button"
                    onClick={onClose}
                    aria-label="Close"
                    className="rounded px-2 py-0.5 text-lg leading-none text-muted-foreground hover:bg-muted hover:text-foreground"
                >
                    ×
                </button>
                </div>
            </div>

            {open && (
                <>
            {event.ongoing && (
                <p className="mt-2 rounded bg-amber-500/15 px-2 py-1 text-[11px] text-amber-800 dark:text-amber-300">
                    Still under way. These are early signs, not the full picture.
                </p>
            )}

            <ul className="mt-2 space-y-1.5">
                {ORDER.filter((slug) => event.effects?.[slug]).map((slug) => {
                    const d = describeEffect(slug, event.effects[slug], event.phase, { ongoing: event.ongoing });
                    return (
                        <li key={slug} className="text-[13px]">
                            <div className="flex items-baseline gap-2">
                                <span
                                    className="inline-block size-2 shrink-0 translate-y-[-1px] rounded-full"
                                    style={{ background: DOT[d.direction] }}
                                />
                                <span className="font-medium">{VARIABLE_WORDS[slug].label}</span>
                                <span className={d.status === "clear" ? "" : "text-muted-foreground"}>{d.headline}</span>
                            </div>
                            {d.status === "clear" && (
                                <p className="mt-0.5 pl-4 text-[11px] leading-snug text-muted-foreground">{d.detail}</p>
                            )}
                        </li>
                    );
                })}
            </ul>

            <div className="mt-3 border-t border-border pt-2">
                <p className="mb-1.5 text-[11px] text-muted-foreground">
                    Map shows{mapMode !== "day" && ` · ${VARIABLE_WORDS[analysisVariable]?.label.toLowerCase()}`}
                </p>
                <div className="grid grid-cols-4 gap-1" role="radiogroup" aria-label="What the map shows">
                    {modesFor(event).map((m) => (
                        <button
                            key={m.id}
                            type="button"
                            role="radio"
                            aria-checked={mapMode === m.id}
                            onClick={() => onMapModeChange?.(m.id)}
                            className={`rounded-md border px-1.5 py-1 text-[10px] font-medium leading-tight ${
                                mapMode === m.id
                                    ? "border-foreground bg-foreground text-background"
                                    : "border-border hover:bg-muted"
                            }`}
                        >
                            {m.label}
                        </button>
                    ))}
                </div>
                <MapLegend mode={mapMode} event={event} variable={analysisVariable} analysis={analysis} />
            </div>

            <div className="mt-3 flex items-center justify-between gap-3">
                <button
                    type="button"
                    onClick={() => {
                        onMapModeChange?.("day");
                        onShowOnMap?.(peakTime);
                    }}
                    disabled={showingPeak && mapMode === "day"}
                    className="rounded-md border border-border px-3 py-1.5 text-[11px] font-medium hover:bg-muted disabled:opacity-50"
                >
                    {showingPeak && mapMode === "day"
                        ? `Showing ${formatTime(peakTime)}`
                        : `Show ${formatTime(peakTime)} on map`}
                </button>
                <span className="text-right text-[10px] leading-tight text-muted-foreground">
                    Each spot vs its own
                    <br />
                    2003–2022 normal
                </span>
            </div>
                </>
            )}
        </div>
    );
}
