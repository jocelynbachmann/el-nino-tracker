# Forecast inputs

The 6-month weather and ocean forecast that `../train_mld_forecast.py` turns into the app's
mixed layer depth predictions (the Predictions switch in the Ocean data panel).

- `ocean_2026-10.grib`, `wind_ocean_2026-10.grib`: raw forecast issued 1 Oct 2026, covering
  Oct 2026 to Mar 2027, 51 ensemble members. From ECMWF's SEAS5 seasonal forecast system
  (Copernicus Climate Data Store), downloaded by `download_forecast_inputs.py`.
- `merge_to_csv.py`: turns the two files into one table, `clean/seas5_merged_full.csv`, which
  the model reads.
- `rough_work/`: by-products and scripts the app doesn't use.

To refresh the predictions: update the year/month in `download_forecast_inputs.py` and run it,
then `python merge_to_csv.py` here,
then `python ../train_mld_forecast.py` (it uploads the result to S3 for the app).
