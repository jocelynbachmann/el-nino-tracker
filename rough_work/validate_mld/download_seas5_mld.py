"""
Archived: downloads ECMWF's own SEAS5 mixed layer depth forecast, used only to compare
against our model (validate_mld). Not part of the app. Needs ~/.cdsapirc and cdsapi.
"""
import cdsapi

dataset = "seasonal-monthly-ocean"
request = {
    "originating_centre": "ecmwf",
    "system": "51",
    "variable": ["mixed_layer_depth_0_01"],
    "forecast_type": ["forecast"],
    "year": ["2026"],
    "month": ["10"],
        "leadtime_month": [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6"
    ],
    "data_format": "grib",
    "area": [52, -134, 46, -120]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
