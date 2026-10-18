/**
 * @file MinimizeButton.jsx
 * @description Small chevron button that collapses / expands a floating card.
 */
export default function MinimizeButton({ open, onToggle, label = "panel", className = "" }) {
    return (
        <button
            type="button"
            onClick={onToggle}
            aria-expanded={open}
            aria-label={`${open ? "Minimize" : "Expand"} ${label}`}
            title={open ? "Minimize" : "Expand"}
            className={`inline-flex size-6 shrink-0 items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground ${className}`}
        >
            <svg
                viewBox="0 0 16 16"
                width="14"
                height="14"
                aria-hidden="true"
                style={{ transform: open ? "rotate(0deg)" : "rotate(180deg)", transition: "transform 150ms" }}
            >
                <path d="M4 10l4-4 4 4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
        </button>
    );
}
