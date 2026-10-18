/**
 * @file EventList.jsx
 * @description Collapsible list of past El Niños / La Niñas for the Ocean data panel.
 * Clicking one opens its event card (same selection as clicking its period on the history strip).
 */
import { useMemo, useState } from "react";
import MinimizeButton from "@/components/MinimizeButton";
import { PHASE_NAME } from "@/lib/eventText";

const COLOURS = { el_nino: "#b2182b", la_nina: "#2166ac" };
const FILTERS = [
    { id: "all", label: "All" },
    { id: "el_nino", label: "El Niño" },
    { id: "la_nina", label: "La Niña" },
];

export default function EventList({ events = [], selectedId, onSelect }) {
    const [open, setOpen] = useState(true);
    const [filter, setFilter] = useState("all");
    const shown = useMemo(
        () => [...events].reverse().filter((e) => filter === "all" || e.phase === filter), // newest first
        [events, filter],
    );
    if (!events.length) return null;

    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
                <h2 className="text-sm font-medium">Past El Niños and La Niñas</h2>
                <MinimizeButton open={open} onToggle={() => setOpen((o) => !o)} label="event list" />
            </div>
            {open && (
                <>
                    <div className="grid grid-cols-3 gap-1" role="radiogroup" aria-label="Show events">
                        {FILTERS.map((f) => (
                            <button
                                key={f.id}
                                type="button"
                                role="radio"
                                aria-checked={filter === f.id}
                                onClick={() => setFilter(f.id)}
                                className={`rounded-md border px-2 py-0.5 text-[11px] font-medium ${
                                    filter === f.id
                                        ? "border-foreground bg-foreground text-background"
                                        : "border-border hover:bg-muted"
                                }`}
                            >
                                {f.label}
                            </button>
                        ))}
                    </div>
                    <ul className="max-h-56 space-y-0.5 overflow-y-auto pr-1">
                        {shown.map((ev) => {
                            const sel = ev.id === selectedId;
                            return (
                                <li key={ev.id}>
                                    <button
                                        type="button"
                                        onClick={() => onSelect?.(sel ? null : ev.id)}
                                        aria-pressed={sel}
                                        className={`flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-sm ${
                                            sel ? "bg-accent font-medium" : "hover:bg-accent/60"
                                        }`}
                                    >
                                        <span
                                            className="inline-block size-2 shrink-0 rounded-full"
                                            style={{ background: COLOURS[ev.phase] }}
                                        />
                                        <span className="flex-1 truncate">
                                            {ev.label.replace(" (ongoing)", "")} {PHASE_NAME[ev.phase]}
                                        </span>
                                        <span className="shrink-0 text-[11px] font-normal text-muted-foreground">
                                            {ev.ongoing ? "now" : ev.strength}
                                        </span>
                                    </button>
                                </li>
                            );
                        })}
                    </ul>
                </>
            )}
        </div>
    );
}
