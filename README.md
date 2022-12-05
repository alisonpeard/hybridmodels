# Hybrid Coastal Flood Models
Work in progress, developing code to:
1. Scrape coastal flood-related remote-sensing data from Google Earth Engine and other sources
2. Process into a dataset of 32 x 32 km grid squares 
3. Train machine learning models to predict flood maps from the data

## Contents
* Python script `get_data_parallel.py` with code to generate dataset. Requirements:
  1. CSV `current_datasets.csv` and `event_dates.csv`: csvs containing storm name, region name, number of subregions (32x32 grid squares) for each region, storm start date, end date, landfall date (from IBTrACs data), satellite acquisition date of the flood map, and a geojson string of the subregion polygons. These should be in `./hybridmodels/data/csv` directory.
  2. a folder with name following the format `<storm name>_<region>` and containing a flood polygon (`flood.gpkg`) and area of interest (`areaOfInterest.gpkg`) polygon in GeoPackage format. These should be in `./hybridmodels/data/<storm name>_<region>` directory.
  3. The hybridmodels conda environment (create from `hybridmodels.yml` contained in envs folder).
  4. Googel Earth Engine Service account with gcloud key saved in hidden folder in the directory (not uploaded here).
