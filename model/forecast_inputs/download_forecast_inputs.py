"""
Downloads the 6-month weather and ocean forecast that train_mld_forecast.py turns into the
app's mixed layer depth predictions: ECMWF's SEAS5 seasonal forecast, issued 1 Oct 2026,
from the Copernicus Climate Data Store.

Needs a CDS account and API key in ~/.cdsapirc (see the root README), and: pip install cdsapi
Run in this folder:  python download_forecast_inputs.py
Then:                python merge_to_csv.py   and   python ../train_mld_forecast.py
To use a newer forecast, change "year" / "month" below (and the file names to match).
"""
import cdsapi

client = cdsapi.Client()

# Ocean: surface salinity, sea level, upper-300 m temperature
dataset = "seasonal-monthly-ocean"
request = {
    "originating_centre": "ecmwf",
    "system": "51",
    "variable": [
        "sea_surface_salinity",
        "sea_surface_height_above_geoid",
        "depth_average_potential_temperature_of_upper_300m"
    ],
    "forecast_type": ["forecast"],
    "year": ["2026"],
    "month": ["10"],
    "leadtime_month": ["1", "2", "3", "4", "5", "6"],
    "area": [52, -134, 46, -120]
}
ocean_request = request
client.retrieve(dataset, ocean_request).download("ocean_2026-10.grib")

# Atmosphere: 10 m wind (u, v, speed) and 2 m air temperature (stands in for sea surface temperature)
dataset = "seasonal-monthly-single-levels"
request = {
    "originating_centre": "ecmwf",
    "system": "51",
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "10m_wind_speed",
        "2m_temperature"
    ],
    "product_type": ["monthly_mean"],
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
wind_request = request
client.retrieve(dataset, wind_request).download("wind_ocean_2026-10.grib")
