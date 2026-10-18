/**
 * @file HelpText.jsx
 * @description The explanations shown by the "i" buttons, kept in one place so they're easy to edit.
 */
import InfoTip from "@/components/InfoTip";
import { formatTime } from "@/lib/eventText";

const Swatch = ({ colour }) => (
    <span className="mr-1 inline-block size-2 translate-y-[-1px] rounded-full" style={{ background: colour }} />
);

export function OceanPanelHelp() {
    return (
        <InfoTip title="Ocean data">
            <p>
                <b>Date.</b> Drag the slider to choose a day (2002 to now). You can also click or drag on the El
                Niño history strip at the bottom.
            </p>
            <p>
                <b>Layers.</b> Tick a variable to colour the map with it. The last one you tick is drawn on top;
                its slider sets how see-through it is.
            </p>
            <p>
                <b>Past El Niños and La Niñas.</b> Click one to open a summary of what it did to our coast and
                jump the map to when it hit hardest. Filter to just El Niños or La Niñas with the buttons.
            </p>
            <p>
                <b>Show anomaly.</b> Instead of raw values, colours show how different each spot was from its own
                normal for that time of year (2003–2022): <Swatch colour="#d6604d" />
                above normal, <Swatch colour="#4393c3" />
                below.
            </p>
            <p>Use the chevron to tuck this panel away.</p>
        </InfoTip>
    );
}

export function HistoryHelp() {
    return (
        <InfoTip title="El Niño history" side="top" width={340}>
            <p>
                <b>Shaded periods.</b> Each <Swatch colour="#b2182b" />
                red or <Swatch colour="#2166ac" />
                blue band is an El Niño or La Niña, by NOAA's definition (the ONI stays above +0.5 or below −0.5
                for five overlapping seasons). <b>Click one</b> to see what it did to our coast, or pick it from
                the list in the Ocean data panel. Very faint bands are before 2002, which the analysis doesn't
                cover.
            </p>
            <p>
                <b>Bars.</b> The ONI month by month: how much warmer or cooler than normal the tropical
                Pacific was. This is what defines El Niño and La Niña.
            </p>
            <p>
                <b>Whole region (bottom row).</b> When our whole area (46–52°N, off BC and Washington) ran unusually
                high or low for the variable on top of the map, found automatically from the data. Hover for
                dates, click to jump there. Compare it with the shaded periods: not every El Niño reaches us.
            </p>
            <p>
                <b>Drag</b> along the strip to change the date on the map. Once an event is open, clicking
                inside its period picks a date too.
            </p>
        </InfoTip>
    );
}

// The "Predictions" switch on the history strip
export function PredictionsHelp({ from, until, issued }) {
    return (
        <InfoTip title="Predictions" width={320}>
            <p>
                Extends the date slider and the El Niño history strip past the last observation,{" "}
                {from ? `from ${formatTime(from)} ` : ""}to {until ? formatTime(until) : "the end of the forecast"}.
                On the strip, predicted dates are striped amber.
            </p>
            <p>
                <b>Mixed layer depth only.</b> That's the one variable our model predicts, so past the last
                observation the other layers switch off and come back when you go back in time.
            </p>
            <p>
                The model turns the ECMWF SEAS5 seasonal forecast{issued ? ` (issued ${formatTime(issued)})` : ""}{" "}
                and the El Niño index into a mixed layer depth for each spot, one value per month (the average
                of 51 forecast runs). Every day in a month shows that month's prediction. It's experimental:
                treat it as a rough guide.
            </p>
        </InfoTip>
    );
}

export function EventHelp({ phaseName = "El Niño", plural = "El Niños" }) {
    return (
        <InfoTip title={`Reading this ${phaseName}`} align="end" width={330}>
            <p>
                <b>Each line</b> says what happened to that variable across our region during this {phaseName}.
                A change only counts as clear when at least 20% of the region shifted the same way. Chance alone
                gives about 10%, so smaller numbers are noise.
            </p>
            <p>
                Every spot is compared only with <b>its own history</b> (its normal for that time of year,
                2003–2022), so a small change at a usually calm spot can count as unusual.
            </p>
            <p>
                <b>Map shows</b> (for the variable on top of the map):
            </p>
            <ul className="list-disc space-y-1 pl-4">
                <li>
                    <b>On the day</b>: the ocean on the chosen date.
                </li>
                <li>
                    <b>Start</b>: when each spot's change began, compared with when {phaseName} began.
                </li>
                <li>
                    <b>Duration</b>: how many months the change lasted at each spot.
                </li>
                <li>
                    <b>Past {plural}</b>: whether each spot changed the same way in every past {phaseName}, or
                    only sometimes.
                </li>
            </ul>
            <p>Striped spots were checked but had no clear change.</p>
        </InfoTip>
    );
}

// What each map variable is, in plain words (the "i" next to each layer)
const VARIABLE_INFO = {
    temperature: {
        title: "Sea surface temperature",
        text: "How warm the top few metres of the ocean are (°C). Warm water holds less oxygen and can stress shellfish, salmon and kelp. Marine heatwaves like the 2014–16 \"Blob\" show up here.",
    },
    salinity: {
        title: "Sea surface salinity",
        text: "How salty the surface water is (PSU, about grams of salt per kilogram of water). Lower means fresher, usually from rain, snowmelt and rivers like the Fraser and Columbia.",
    },
    "sea-surface-height": {
        title: "Sea surface height",
        text: "How high the sea surface sits (metres). Warm water expands and winds pile water against the coast, so El Niño years often raise coastal sea level. Differences in height also drive currents.",
    },
    "mixed-layer-depth": {
        title: "Mixed layer depth",
        text: "How deep the well-mixed surface layer reaches (metres). Winter storms and cooling stir it deeper; in summer it is shallow, keeping heat and plankton near the surface. A shallower layer can mean fewer nutrients brought up from below.",
    },
    "current-east": {
        title: "Current (east)",
        text: "Speed of the surface current flowing east (positive) or west (negative), in m/s, worked out from differences in sea surface height.",
    },
    "current-north": {
        title: "Current (north)",
        text: "Speed of the surface current flowing north (positive) or south (negative), in m/s, worked out from differences in sea surface height.",
    },
};

export function VariableInfo({ id }) {
    const info = VARIABLE_INFO[id];
    if (!info) return null;
    return (
        <InfoTip title={info.title} icon="i" width={280}>
            <p>{info.text}</p>
            <p>Satellite and ocean-float data (Copernicus ARMOR3D), daily, about 14 km grid.</p>
        </InfoTip>
    );
}
