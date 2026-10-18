/*
 * @file @/hooks/useEvents.js
 * @description El Niño / La Niña history from /events (pipeline/analyze_events.py), loaded once.
 */
import { useEffect, useState } from "react";
import { getEvents } from "@/lib/oceanApi";

export function useEvents() {
    const [state, setState] = useState({ loading: true, error: null, events: [] });

    useEffect(() => {
        let cancelled = false;
        getEvents()
            .then((data) => {
                if (cancelled) return;
                setState({
                    loading: false,
                    error: null,
                    events: data.events ?? [],
                    dataStart: data.data_start,
                    dataEnd: data.data_end,
                    method: data.method,
                    regionalSpells: data.regional_spells,
                    patterns: data.patterns,
                });
            })
            .catch((err) => {
                // optional feature: the map still works without it
                if (!cancelled) setState({ loading: false, error: err, events: [] });
            });
        return () => {
            cancelled = true;
        };
    }, []);

    return state;
}
