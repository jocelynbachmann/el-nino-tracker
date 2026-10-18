/**
 * @file EventTimeline.jsx
 * @description History strip under the map: the ONI (El Niño index) month by month, with each
 * El Niño / La Niña period shaded behind it, and a row of region-wide unusual spells below.
 * Drag to scrub the date. A click inside a shaded period selects that event (once it's selected,
 * clicks inside it pick dates). Events can also be picked from the list in the Ocean data panel.
 * With predictions on, the strip runs past the last observation into the MLD model's forecast
 * (hatched amber).
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { formatTime, PHASE_NAME, VARIABLE_WORDS } from "@/lib/eventText";
import MinimizeButton from "@/components/MinimizeButton";
import { HistoryHelp } from "@/components/HelpText";

const PREDICTION_COLOUR = "#d97706"; // amber, like the "Prediction" badge
const COLOURS = { el_nino: "#b2182b", la_nina: "#2166ac", neutral: "#a8a29e" };
const SPELL_COLOURS = { above: "#d6604d", below: "#4393c3" }; // same ends as the anomaly map
const H = 100; // svg height
const PAD_X = 8;
const BAND_Y = 2; // event shading spans the ONI area
const BAND_H = 62;
const ZERO_Y = 33;
const BAR_SCALE = 9; // px per ONI unit
const SPELL_Y = 72; // region-wide unusual spells row
const SPELL_H = 12;
const AXIS_Y = 97;
const CLICK_PX = 4; // a press that moves less than this is a click, not a drag
const MIN_PREDICTION_SHARE = 0.12; // predictions get at least this share of the width, so they're easy to scrub

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
    regionalSpells = [], // [{start, end, direction: "above"|"below", mean_sd}] for spellVariable
    spellVariable = "temperature",
    // { on, from }: when on, the strip runs on past the observations to the end of the MLD
    // prediction (from = first predicted time). The switch itself is in the Ocean data panel.
    predictions = null,
    className = "",
}) {
    const boxRef = useRef(null);
    const width = useWidth(boxRef);
    const [hover, setHover] = useState(null); // time under the pointer
    const [hoverSpell, setHoverSpell] = useState(null);
    const [open, setOpen] = useState(true);
    const press = useRef(null); // {x, moved}

    const ms = useMemo(() => times.map(toMs), [times]);
    const monthly = useMemo(
        () =>
            enso
                .map((e) => ({ ...e, month: e.month?.slice(0, 7) }))
                .filter((e) => e.month && e.oni != null),
        [enso],
    );
    const ensoByMonth = useMemo(() => new Map(monthly.map((e) => [e.month, e])), [monthly]);
    const spans = useMemo(
        () => events.map((ev) => ({ ev, a: toMs(ev.start), b: nextMonthMs(ev.end) })),
        [events],
    );

    const n = times.length;
    const t0 = ms[0];
    const t1 = n ? Math.max(ms[n - 1], times[n - 1].length === 7 ? nextMonthMs(times[n - 1]) : ms[n - 1]) : 0;
    const inner = Math.max(1, width - 2 * PAD_X);
    // With predictions on, the predicted months are stretched to at least MIN_PREDICTION_SHARE of
    // the width (they're a few months on a 25-year strip); the observed years share the rest.
    const predFrom = predictions?.on && predictions.from ? toMs(predictions.from) : null;
    const split = predFrom != null && predFrom > t0 && predFrom < t1 ? predFrom : null;
    const share = split == null ? 0 : Math.max(MIN_PREDICTION_SHARE, (t1 - split) / Math.max(1, t1 - t0));
    const x = (v) => {
        if (split == null) return PAD_X + (inner * (v - t0)) / Math.max(1, t1 - t0);
        if (v <= split) return PAD_X + (inner * (1 - share) * (v - t0)) / (split - t0);
        return PAD_X + inner * (1 - share) + (inner * share * (v - split)) / Math.max(1, t1 - split);
    };
    const inRange = (a, b) => b > t0 && a < t1;

    // Consecutive months in the same phase: shading for periods the analysis doesn't cover (pre-2002)
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

    const msAt = (clientX) => {
        const r = boxRef.current.getBoundingClientRect();
        const f = (clientX - r.left - PAD_X) / inner; // 0..1 along the strip
        if (split == null) return t0 + f * (t1 - t0);
        return f <= 1 - share
            ? t0 + (f / (1 - share)) * (split - t0)
            : split + ((f - (1 - share)) / share) * (t1 - split);
    };
    const timeAt = (clientX) => times[nearest(ms, msAt(clientX))];
    const eventAt = (v) => spans.find((s) => v >= s.a && v < s.b)?.ev ?? null;

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

    const words = VARIABLE_WORDS[spellVariable];
    const spellText = (sp) =>
        `Whole region ${sp.direction === "above" ? words?.up : words?.down} than usual: ` +
        `${formatTime(sp.start)} to ${formatTime(sp.end)} (average ${sp.mean_sd > 0 ? "+" : ""}${sp.mean_sd} SD)`;

    const shown = hover ?? time;
    const shownPredicted = predFrom != null && shown && toMs(shown) >= predFrom;
    const info = shown ? ensoByMonth.get(shown.slice(0, 7)) : null;
    const hoverEvent = hover ? eventAt(toMs(hover)) : null;

    if (!n) return null;

    return (
        <div className={`rounded-lg bg-background/90 p-3 shadow-md backdrop-blur ${className}`}>
            <div className={`${open ? "mb-1" : ""} flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-xs`}>
                <div className="flex items-center gap-3">
                    <h2 className="text-sm font-medium">El Niño history</h2>
                    <HistoryHelp />
                    <span className="flex items-center gap-1 text-muted-foreground">
                        <span className="inline-block h-2.5 w-3 rounded-sm" style={{ background: COLOURS.el_nino, opacity: 0.35 }} />
                        El Niño
                    </span>
                    <span className="flex items-center gap-1 text-muted-foreground">
                        <span className="inline-block h-2.5 w-3 rounded-sm" style={{ background: COLOURS.la_nina, opacity: 0.35 }} />
                        La Niña
                    </span>
                </div>
                <span className="ml-auto tabular-nums text-muted-foreground">
                    {hoverSpell ? (
                        <span className="text-foreground">{spellText(hoverSpell)}</span>
                    ) : (
                        shown && (
                            <>
                                <span className="text-foreground">{formatTime(shown)}</span>
                                {shownPredicted && (
                                    <span className="text-amber-700 dark:text-amber-400"> · predicted mixed layer depth</span>
                                )}
                                {info?.oni != null && ` · ONI ${info.oni > 0 ? "+" : ""}${Number(info.oni).toFixed(1)}`}
                                {hoverEvent
                                    ? ` · ${hoverEvent.label.replace(" (ongoing)", "")} ${PHASE_NAME[hoverEvent.phase]}${
                                          hoverEvent.id === selectedId ? "" : " (click to open)"
                                      }`
                                    : info?.phase && info.phase !== "neutral" && ` · ${PHASE_NAME[info.phase]}`}
                            </>
                        )
                    )}
                </span>
                <MinimizeButton open={open} onToggle={() => setOpen((o) => !o)} label="El Niño history" />
            </div>

            <div ref={boxRef} className={`w-full touch-none select-none ${open ? "" : "hidden"}`}>
                {width > 0 && (
                    <svg
                        width={width}
                        height={H}
                        role="img"
                        aria-label="El Niño index by month, with El Niño and La Niña periods shaded. Drag to choose a date; click a shaded period to open it."
                        className={hoverEvent && hoverEvent.id !== selectedId ? "cursor-pointer" : "cursor-crosshair"}
                        onPointerDown={(e) => {
                            press.current = { x: e.clientX, moved: false };
                            e.currentTarget.setPointerCapture(e.pointerId);
                            onTimeChange?.(timeAt(e.clientX));
                        }}
                        onPointerMove={(e) => {
                            const t = timeAt(e.clientX);
                            setHover(t);
                            const p = press.current;
                            if (!p) return;
                            if (Math.abs(e.clientX - p.x) >= CLICK_PX) p.moved = true;
                            if (p.moved) onTimeChange?.(t);
                        }}
                        onPointerUp={(e) => {
                            const p = press.current;
                            press.current = null;
                            if (!p || p.moved) return;
                            // a click (not a drag) inside an event that isn't open yet opens it
                            const ev = eventAt(msAt(e.clientX));
                            if (ev && ev.id !== selectedId) onSelectEvent?.(ev.id);
                        }}
                        onPointerLeave={() => setHover(null)}
                    >
                        {/* predictions: hatched amber from the first predicted day to the end */}
                        {predFrom != null && predFrom < t1 && (
                            <g style={{ pointerEvents: "none" }}>
                                <defs>
                                    <pattern id="prediction-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                                        <rect width="6" height="6" fill={PREDICTION_COLOUR} opacity={0.08} />
                                        <line x1="0" y1="0" x2="0" y2="6" stroke={PREDICTION_COLOUR} strokeWidth="2" opacity={0.35} />
                                    </pattern>
                                </defs>
                                <rect
                                    x={x(predFrom)}
                                    y={BAND_Y}
                                    width={Math.max(2, x(t1) - x(predFrom))}
                                    height={SPELL_Y + SPELL_H - BAND_Y}
                                    fill="url(#prediction-hatch)"
                                />
                                <line x1={x(predFrom)} x2={x(predFrom)} y1={BAND_Y} y2={SPELL_Y + SPELL_H} stroke={PREDICTION_COLOUR} strokeDasharray="3 2" />
                                <text x={(x(predFrom) + x(t1)) / 2} y={AXIS_Y} textAnchor="middle" fontSize={11} fill={PREDICTION_COLOUR}>
                                    Predicted
                                </text>
                            </g>
                        )}

                        {/* older ENSO periods (before the analysis starts): light, not clickable */}
                        {runs
                            .filter((r) => inRange(r.start, r.end) && !spans.some((s) => s.a < r.end && s.b > r.start))
                            .map((r) => (
                                <rect
                                    key={`run-${r.start}`}
                                    x={x(Math.max(r.start, t0))}
                                    y={BAND_Y}
                                    width={Math.max(1, x(Math.min(r.end, t1)) - x(Math.max(r.start, t0)))}
                                    height={BAND_H}
                                    fill={COLOURS[r.phase]}
                                    opacity={0.06}
                                />
                            ))}

                        {/* analysed El Niño / La Niña periods: shaded, click to open */}
                        {spans
                            .filter(({ a, b }) => inRange(a, b))
                            .map(({ ev, a, b }) => {
                                const xa = x(Math.max(a, t0));
                                const w = Math.max(2, x(Math.min(b, t1)) - xa);
                                const sel = ev.id === selectedId;
                                const hov = hoverEvent?.id === ev.id;
                                return (
                                    <rect
                                        key={ev.id}
                                        x={xa}
                                        y={BAND_Y}
                                        width={w}
                                        height={BAND_H}
                                        rx={3}
                                        fill={COLOURS[ev.phase]}
                                        opacity={sel ? 0.22 : hov ? 0.18 : 0.1}
                                        stroke={sel ? COLOURS[ev.phase] : "none"}
                                        strokeWidth={1.5}
                                        strokeOpacity={1}
                                    />
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
                                    style={{ pointerEvents: "none" }}
                                />
                            );
                        })}

                        {/* region-wide unusual spells (change points on the box-average series) */}
                        <text x={PAD_X} y={SPELL_Y - 3} fontSize={10} fill="var(--muted-foreground)">
                            {`Whole region: ${words?.label?.toLowerCase() ?? spellVariable}`}
                        </text>
                        <rect x={PAD_X} y={SPELL_Y} width={inner} height={SPELL_H} rx={2} fill="var(--muted)" opacity={0.6} />
                        {regionalSpells.map((sp) => {
                            const a = Math.max(toMs(sp.start), t0);
                            const b = Math.min(toMs(sp.end) + 86400000, t1);
                            if (b <= a) return null;
                            const strength = Math.min(1, 0.45 + Math.abs(sp.mean_sd) / 4);
                            return (
                                <rect
                                    key={`${sp.start}-${sp.direction}`}
                                    x={x(a)}
                                    y={SPELL_Y}
                                    width={Math.max(2, x(b) - x(a))}
                                    height={SPELL_H}
                                    rx={2}
                                    fill={SPELL_COLOURS[sp.direction]}
                                    opacity={strength}
                                    className="cursor-pointer"
                                    onPointerEnter={() => setHoverSpell(sp)}
                                    onPointerLeave={() => setHoverSpell(null)}
                                    onPointerDown={(e) => {
                                        // jump to the middle of the spell
                                        e.stopPropagation();
                                        const mid = (toMs(sp.start) + toMs(sp.end)) / 2;
                                        onTimeChange?.(times[nearest(ms, mid)]);
                                    }}
                                >
                                    <title>{spellText(sp)}</title>
                                </rect>
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
                                <line x1={x(toMs(time))} x2={x(toMs(time))} y1={BAND_Y} y2={AXIS_Y - 11} stroke="var(--foreground)" strokeWidth={1.5} />
                                <path d={`M${x(toMs(time)) - 4},${AXIS_Y - 11} h8 l-4,-5 z`} fill="var(--foreground)" />
                            </g>
                        )}
                    </svg>
                )}
            </div>
        </div>
    );
}
