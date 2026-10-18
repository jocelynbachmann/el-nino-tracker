/**
 * @file InfoTip.jsx
 * @description Small "i" button that opens a short explanation of a feature. The popover is drawn
 * in a portal (fixed position) so scrolling cards don't clip it. Closes on outside click or Esc.
 */
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

const GAP = 6;

export default function InfoTip({ title, children, side = "bottom", align = "start", width = 300 }) {
    const btnRef = useRef(null);
    const popRef = useRef(null);
    const [pos, setPos] = useState(null); // null = closed

    const open = () => {
        const r = btnRef.current.getBoundingClientRect();
        const w = Math.min(width, window.innerWidth - 16);
        const left = Math.min(
            Math.max(8, align === "start" ? r.left : r.right - w),
            window.innerWidth - w - 8,
        );
        setPos(
            side === "top"
                ? { left, bottom: window.innerHeight - r.top + GAP, width: w, maxHeight: r.top - 16 }
                : { left, top: r.bottom + GAP, width: w, maxHeight: window.innerHeight - r.bottom - 16 },
        );
    };

    useEffect(() => {
        if (!pos) return;
        const close = () => setPos(null);
        const onDown = (e) => {
            if (!popRef.current?.contains(e.target) && !btnRef.current?.contains(e.target)) close();
        };
        const onKey = (e) => e.key === "Escape" && close();
        document.addEventListener("pointerdown", onDown);
        document.addEventListener("keydown", onKey);
        window.addEventListener("resize", close);
        return () => {
            document.removeEventListener("pointerdown", onDown);
            document.removeEventListener("keydown", onKey);
            window.removeEventListener("resize", close);
        };
    }, [pos]);

    return (
        <>
            <button
                ref={btnRef}
                type="button"
                onClick={() => (pos ? setPos(null) : open())}
                aria-expanded={!!pos}
                aria-label={`How this works: ${title}`}
                title="How this works"
                className={`inline-flex size-5 shrink-0 items-center justify-center rounded-full border text-[11px] font-semibold leading-none ${
                    pos
                        ? "border-foreground bg-foreground text-background"
                        : "border-border text-muted-foreground hover:bg-muted hover:text-foreground"
                }`}
            >
                i
            </button>
            {pos &&
                createPortal(
                    <div
                        ref={popRef}
                        role="dialog"
                        aria-label={title}
                        style={{ position: "fixed", zIndex: 50, overflowY: "auto", ...pos }}
                        className="rounded-lg border border-border bg-background p-3 text-xs leading-relaxed text-foreground shadow-lg"
                    >
                        <p className="mb-2 text-sm font-medium">{title}</p>
                        <div className="space-y-2 text-muted-foreground [&_b]:font-medium [&_b]:text-foreground">
                            {children}
                        </div>
                    </div>,
                    document.body,
                )}
        </>
    );
}
