import cdsapi

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

#client = cdsapi.Client()

#client.retrieve(dataset, request).download("ocean_2026-10.grib")


import cdsapi

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

#client = cdsapi.Client()
#client.retrieve(dataset, request).download("wind_ocean_2026-10.grib")


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
