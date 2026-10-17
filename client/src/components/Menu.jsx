/**
 * @file Menu.jsx
 * @description Menu component to select timeline, layers, etc.
 */

import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { FieldGroup, Field, FieldLabel } from "@/components/ui/field";

/**
 * Menu component to select map options
 */
export default function Menu() {
    return (
        <>
            <FieldGroup className="text-sm gap-y-1">
                <Field orientation="horizontal">
                    <Switch id={1} />
                    <FieldLabel>Layer 1 </FieldLabel>
                </Field>
                <Field orientation="horizontal">
                    <Switch id={2} />
                    <FieldLabel>Layer 2 </FieldLabel>
                </Field>
                <Field orientation="horizontal">
                    <Switch id={3} />
                    <FieldLabel>Layer 3 </FieldLabel>
                </Field>
            </FieldGroup>

            <div className="flex flex-col gap-y-1">
                {/* TODO: change scale */}
                <Slider
                    defaultValue={[2026]}
                    max={2028}
                    min={2023}
                    steps={1}
                    className="mx-auto w-full max-w-xs"
                />
                <div className="flex flex-row justify-between text-xs">
                    <p>2023</p>
                    <p>2024</p>
                    <p>2025</p>
                    <p>2026</p>
                    <p>2027</p>
                    <p>2028</p>
                </div>
            </div>
        </>
    );
}
