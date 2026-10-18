/**
 * @file ElNinoStats.jsx
 * @description Buttons for picking a past El Niño year to view on the map.
 */

/**
 * @file ElNinoStats.jsx
 * @description Buttons for picking a past El Niño event to view on the map.
 */
import { Button } from "@/components/ui/button";

/**
 * El Niño event picker
 * @param {object} props - component properties
 * @param {{id: string, label: string, start: string, end: string, peakMonth: string, peakOni: number|null}[]} props.events
 * @param {string|null} props.selectedId
 * @param {(event: object|null) => void} props.onSelect
 */
export default function ElNinoStats({ events = [], selectedId, onSelect }) {
    if (!events.length) return null;

    return (
        <div className="space-y-2">
            <div className="flex items-baseline justify-between">
                <h2 className="text-sm font-medium">El Niño stats</h2>
                {selectedId != null && (
                    <Button
                        variant="ghost"
                        size="xs"
                        onClick={() => onSelect(null)}
                    >
                        Clear
                    </Button>
                )}
            </div>
            <p className="text-xs text-muted-foreground">
                Pick an event to jump to its peak
            </p>
            <div className="flex flex-wrap gap-1.5">
                {events.map((ev) => (
                    <Button
                        key={ev.id}
                        size="xs"
                        variant={ev.id === selectedId ? "default" : "outline"}
                        aria-pressed={ev.id === selectedId}
                        title={`${ev.start} to ${ev.end} · peak ${ev.peakMonth}${ev.peakOni != null ? ` (ONI ${ev.peakOni.toFixed(1)})` : ""}`}
                        onClick={() => onSelect(ev)}
                    >
                        {ev.label}
                    </Button>
                ))}
            </div>
        </div>
    );
}
