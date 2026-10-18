/**
 * @file EventTimeline.jsx
 * @description History strip under the map: the ONI (El Niño index) month by month, with the
 * El Niño / La Niña events from the analysis on top. Click or drag anywhere to pick a time
 * (a day with daily data, a month with monthly data); click an event to select it.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { formatTime, PHASE_NAME } from "@/lib/eventText";

const COLOURS = { el_nino: "#b2182b", la_nina: "#2166ac", neutral: "#a8a29e" };
const H = 92; // svg height
const PAD_X = 8;
const EVENT_Y = 4;
const EVENT_H = 18;
const ZERO_Y = 50;
const BAR_SCALE = 9; // px per ONI unit
const AXIS_Y = 88;

// "YYYY-MM" or "YYYY-MM-DD" -> ms (UTC). Months start on the 1st.
const toMs = (s) => {
    const [y, m, d] = s.split("-").map(Number);
    return Date.UTC(y, m - 1, d || 1);
};
const nextMonthMs = (ym) => {
    const [y, m] = ym.split("-").map(Number);
    return Date.UTC(y, m, 1);
};

function useWidth(ref) {
    const [w, setW] = useState(0);
    useEffect(() => {
        if (!ref.current) return;
        const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
        ro.observe(ref.current);
        return () => ro.disconnect();
    }, [ref]);
    return w;
}

// Nearest entry of a sorted ms array
function nearest(arr, v) {
    let lo = 0;
    let hi = arr.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid] < v) lo = mid + 1;
        else hi = mid;
    }
    return lo > 0 && v - arr[lo - 1] < arr[lo] - v ? lo - 1 : lo;
}

export default function EventTimeline({
    enso = [],
    events = [],
    times = [],
    time,
    onTimeChange,
    selectedId,
    onSelectEvent,
    className = "",
}) {
    const boxRef = useRef(null);
    const width = useWidth(boxRef);
    const [hover, setHover] = useState(null);
    const dragging = useRef(false);

    const ms = useMemo(() => times.map(toMs), [times]);
    const monthly = useMemo(
        () =>
            enso
                .map((e) => ({ ...e, month: e.month?.slice(0, 7) }))
                .filter((e) => e.month && e.oni != null),
        [enso],
    );
    const ensoByMonth = useMemo(() => new Map(monthly.map((e) => [e.month, e])), [monthly]);

    const n = times.length;
    const t0 = ms[0];
    const t1 = n ? Math.max(ms[n - 1], times[n - 1].length === 7 ? nextMonthMs(times[n - 1]) : ms[n - 1]) : 0;
    const inner = Math.max(1, width - 2 * PAD_X);
    const x = (v) => PAD_X + (inner * (v - t0)) / Math.max(1, t1 - t0);
    const inRange = (a, b) => b > t0 && a < t1;

    // Consecutive months in the same phase (older events are drawn faded behind the analysed ones)
    const runs = useMemo(() => {
        const out = [];
        let cur = null;
        for (const e of monthly) {
            if (cur && cur.phase === e.phase && toMs(e.month) === cur.end) {
                cur.end = nextMonthMs(e.month);
            } else {
                cur = { phase: e.phase, start: toMs(e.month), end: nextMonthMs(e.month) };
                if (e.phase !== "neutral") out.push(cur);
            }
        }
        return out;
    }, [monthly]);

    const timeAt = (clientX) => {
        const r = boxRef.current.getBoundingClientRect();
        const v = t0 + ((clientX - r.left - PAD_X) / inner) * (t1 - t0);
        return times[nearest(ms, v)];
    };

    const years = useMemo(() => {
        if (!n) return [];
        const step = width < 700 ? 10 : 5;
        const out = [];
        const y0 = new Date(t0).getUTCFullYear();
        const y1 = new Date(t1).getUTCFullYear();
        for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) {
            const v = Date.UTC(y, 0, 1);
            if (v >= t0 && v <= t1) out.push({ y, v });
        }
        return out;
    }, [n, t0, t1, width]);

    const shown = hover ?? time;
    const info = shown ? ensoByMonth.get(shown.slice(0, 7)) : null;

    if (!n) return null;

    return (
        <div className={`rounded-lg bg-background/90 p-3 shadow-md backdrop-blur ${className}`}>
            <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 text-xs">
                <div className="flex items-baseline gap-3">
                    <h2 className="text-sm font-medium">El Niño history</h2>
                    <span className="flex items-center gap-1 text-muted-foreground">
                        <span className="inline-block size-2 rounded-full" style={{ background: COLOURS.el_nino }} />
                        El Niño
                    </span>
                    <span className="flex items-center gap-1 text-muted-foreground">
                        <span className="inline-block size-2 rounded-full" style={{ background: COLOURS.la_nina }} />
                        La Niña
                    </span>
                </div>
                <span className="tabular-nums text-muted-foreground">
                    {shown && (
                        <>
                            <span className="text-foreground">{formatTime(shown)}</span>
                            {info?.oni != null && ` · ONI ${info.oni > 0 ? "+" : ""}${Number(info.oni).toFixed(1)}`}
                            {info?.phase && info.phase !== "neutral" && ` · ${PHASE_NAME[info.phase]}`}
                        </>
                    )}
                </span>
            </div>

            <div ref={boxRef} className="w-full touch-none select-none">
                {width > 0 && (
                    <svg
                        width={width}
                        height={H}
                        role="img"
                        aria-label="El Niño index by month with El Niño and La Niña events. Click to choose a date."
                        className="cursor-crosshair"
                        onPointerDown={(e) => {
                            dragging.current = true;
                            e.currentTarget.setPointerCapture(e.pointerId);
                            onTimeChange?.(timeAt(e.clientX));
                        }}
                        onPointerMove={(e) => {
                            const t = timeAt(e.clientX);
                            setHover(t);
                            if (dragging.current) onTimeChange?.(t);
                        }}
                        onPointerUp={() => (dragging.current = false)}
                        onPointerLeave={() => setHover(null)}
                    >
                        {/* all ENSO phases, faded (events before the analysis starts stay like this) */}
                        {runs
                            .filter((r) => inRange(r.start, r.end))
                            .map((r) => (
                                <rect
                                    key={`run-${r.start}`}
                                    x={x(Math.max(r.start, t0))}
                                    y={EVENT_Y + 4}
                                    width={Math.max(2, x(Math.min(r.end, t1)) - x(Math.max(r.start, t0)))}
                                    height={EVENT_H - 8}
                                    rx={2}
                                    fill={COLOURS[r.phase]}
                                    opacity={0.18}
                                />
                            ))}

                        {/* analysed events: clickable */}
                        {events.map((ev) => {
                            const a = toMs(ev.start);
                            const b = nextMonthMs(ev.end);
                            if (!inRange(a, b)) return null;
                            const xa = x(Math.max(a, t0));
                            const w = Math.max(3, x(Math.min(b, t1)) - xa);
                            const sel = ev.id === selectedId;
                            const strong = Math.abs(ev.peak_oni) >= 1.5;
                            return (
                                <g
                                    key={ev.id}
                                    className="cursor-pointer"
                                    onPointerDown={(e) => {
                                        e.stopPropagation();
                                        onSelectEvent?.(sel ? null : ev.id);
                                    }}
                                >
                                    <title>{`${ev.label} ${PHASE_NAME[ev.phase]} (${ev.strength}), peak ONI ${ev.peak_oni > 0 ? "+" : ""}${ev.peak_oni}`}</title>
                                    <rect
                                        x={xa}
                                        y={EVENT_Y}
                                        width={w}
                                        height={EVENT_H}
                                        rx={3}
                                        fill={COLOURS[ev.phase]}
                                        opacity={sel ? 1 : strong ? 0.8 : 0.5}
                                        stroke={sel ? "var(--foreground)" : "none"}
                                        strokeWidth={2}
                                    />
                                    {w > 40 && (
                                        <text
                                            x={xa + w / 2}
                                            y={EVENT_Y + 13}
                                            textAnchor="middle"
                                            fontSize={11}
                                            fill="#fff"
                                            style={{ pointerEvents: "none" }}
                                        >
                                            {ev.label.replace(" (ongoing)", "")}
                                        </text>
                                    )}
                                </g>
                            );
                        })}

                        {/* ONI bars, one per month */}
                        <line x1={PAD_X} x2={width - PAD_X} y1={ZERO_Y} y2={ZERO_Y} stroke="var(--border)" />
                        {monthly.map((e) => {
                            const a = toMs(e.month);
                            const b = nextMonthMs(e.month);
                            if (!inRange(a, b)) return null;
                            const v = Math.max(-3, Math.min(3, e.oni));
                            const xa = x(Math.max(a, t0));
                            return (
                                <rect
                                    key={e.month}
                                    x={xa}
                                    y={v > 0 ? ZERO_Y - v * BAR_SCALE : ZERO_Y}
                                    width={Math.max(0.6, x(Math.min(b, t1)) - xa - 0.3)}
                                    height={Math.max(0.5, Math.abs(v) * BAR_SCALE)}
                                    fill={COLOURS[e.phase] ?? COLOURS.neutral}
                                    opacity={e.phase === "neutral" ? 0.45 : 0.85}
                                />
                            );
                        })}

                        {years.map(({ y, v }) => (
                            <text key={y} x={x(v)} y={AXIS_Y} textAnchor="middle" fontSize={11} fill="var(--muted-foreground)">
                                {y}
                            </text>
                        ))}

                        {/* current time */}
                        {time && (
                            <g style={{ pointerEvents: "none" }}>
                                <line x1={x(toMs(time))} x2={x(toMs(time))} y1={EVENT_Y} y2={AXIS_Y - 11} stroke="var(--foreground)" strokeWidth={1.5} />
                                <path d={`M${x(toMs(time)) - 4},${AXIS_Y - 11} h8 l-4,-5 z`} fill="var(--foreground)" />
                            </g>
                        )}
                    </svg>
                )}
            </div>
        </div>
    );
}
