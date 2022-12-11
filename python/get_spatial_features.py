# env: hybridmodels-modelbuild
"""Add intermediate features to feature_stats GPKGs"""
from os.path import join, dirname
import logging
import traceback
import logging.handlers

import pandas as pd
import geopandas as gpd

import model_utils
import data_utils

# Shapely deprecation warning
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# environment
bd = dirname(__file__)
wd = join(bd, "..", "data")
log_file_path = join(bd, 'logfiles')
log_name = 'spatial_features'

# logger
logger = logging.getLogger(log_name)
fh = logging.FileHandler(join(log_file_path, f"{log_name}.log"), 'a')
f = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s: %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(f)
fh.setFormatter(f)
fh.setLevel(logging.INFO)
sh.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(sh)

# settings
start_on = 0  # sometimes don't want to do all subregions
recalculate = True
recalculate_neighbours = True
verbose = True
temporal = False
binary = True

def main():
    # load and parse data
    df = pd.read_csv(join(wd, "csvs", "current_datasets.csv"))
    df = df[df.to_process == "yes"]  # only process selected events
    rows = [row for _, row in df.iterrows()]
    rows = [(f"{row.event}_{row.region}", row.nsubregions) for row in rows]

    for storm, nsubregions in rows:
        for subregion in range(start_on, 6):  #int(nsubregions)):
            event = f"{storm}_{subregion}"
            logger.info(f"Add spatial features to {event}.")
            try:
                gdf, features, columns = model_utils.load_raw_data(wd, data_utils.default_features, temporal, binary, subset=event)
                data_utils.add_spatial_features(gdf, [event], data_utils.spatial_features, wd, recalculate_neighbours=recalculate_neighbours, recalculate=recalculate, verbose=verbose)
                gdf = model_utils.load_spatial_data(wd, subset=event)
                data_utils.add_intermediate_features(gdf, [event], data_utils.intermediate_features, wd, recalculate=recalculate, verbose=verbose)
            except Exception:
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
