/**
 * @file EventCard.jsx
 * @description Plain-language summary of one El Niño / La Niña: when it ran and what it did to
 * each variable across the region (from /events).
 */
import { describeEffect, formatTime, formatYm, PHASE_NAME, VARIABLE_WORDS } from "@/lib/eventText";

const ORDER = ["temperature", "sea-surface-height", "salinity", "mixed-layer-depth"];
const DOT = { 1: "#b2182b", "-1": "#2166ac", 0: "var(--muted-foreground)" };

// peakTime: the day (or month) to jump to: ONI peak + the region's response lag, clamped to the data
export default function EventCard({ event, peakTime, mapTime, onShowOnMap, onClose, className = "" }) {
    if (!event) return null;
    const name = PHASE_NAME[event.phase] ?? "Event";
    const showingPeak = mapTime === peakTime;

    return (
        <div className={`rounded-lg bg-background/90 p-4 shadow-md backdrop-blur ${className}`}>
            <div className="flex items-start justify-between gap-3">
                <div>
                    <h2 className="text-base font-medium">
                        {event.label.replace(" (ongoing)", "")} {name}
                    </h2>
                    <p className="text-xs text-muted-foreground">
                        {formatYm(event.start)} – {event.ongoing ? "now" : formatYm(event.end)} · {event.strength}
                        {event.ongoing ? " so far" : ""} · {event.ongoing ? "highest" : "peak"} ONI{" "}
                        {event.peak_oni > 0 ? "+" : ""}
                        {event.peak_oni} in {formatYm(event.peak_month)}
                    </p>
                </div>
                <button
                    type="button"
                    onClick={onClose}
                    aria-label="Close"
                    className="-mr-1 -mt-1 rounded px-2 py-0.5 text-lg leading-none text-muted-foreground hover:bg-muted hover:text-foreground"
                >
                    ×
                </button>
            </div>

            {event.ongoing && (
                <p className="mt-2 rounded bg-amber-500/15 px-2 py-1 text-xs text-amber-800 dark:text-amber-300">
                    Still under way. These are early signs, not the full picture.
                </p>
            )}

            <ul className="mt-3 space-y-2.5">
                {ORDER.filter((slug) => event.effects?.[slug]).map((slug) => {
                    const d = describeEffect(slug, event.effects[slug], event.phase, { ongoing: event.ongoing });
                    return (
                        <li key={slug} className="text-sm">
                            <div className="flex items-baseline gap-2">
                                <span
                                    className="inline-block size-2 shrink-0 translate-y-[-1px] rounded-full"
                                    style={{ background: DOT[d.direction] }}
                                />
                                <span className="font-medium">{VARIABLE_WORDS[slug].label}</span>
                                <span className={d.status === "clear" ? "" : "text-muted-foreground"}>{d.headline}</span>
                            </div>
                            {d.status === "clear" && (
                                <p className="mt-0.5 pl-4 text-xs leading-relaxed text-muted-foreground">{d.detail}</p>
                            )}
                        </li>
                    );
                })}
            </ul>

            <div className="mt-4 flex items-center justify-between gap-3">
                <button
                    type="button"
                    onClick={() => onShowOnMap?.(peakTime)}
                    disabled={showingPeak}
                    className="rounded-md border border-border px-3 py-1.5 text-xs font-medium hover:bg-muted disabled:opacity-50"
                >
                    {showingPeak ? `Showing ${formatTime(peakTime)}` : `Show ${formatTime(peakTime)} on map`}
                </button>
                <span className="text-right text-[11px] leading-tight text-muted-foreground">
                    Each spot vs its own
                    <br />
                    2003–2022 normal
                </span>
            </div>
        </div>
    );
}
