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
warnings.filterwarnings("ignore", category=RuntimeWarning)  # suppress intersection warning for non-intersecting Polygons (bug)

# environment
bd = dirname(__file__)
wd = join(bd, "..", "data")
outdir = join(wd, "feature_stats")
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
START_ON = 0  # sometimes don't want to do all subregions
RECALCULATE = True
RECALCULATE_NEIGHBOURS = False # turn off in contingency matrices already present
VERBOSE = True
TEMPORAL = False
BINARY = True
KEYWORD = 'yes'

def main():
    # load and parse data
    df = pd.read_csv(join(wd, "csvs", "current_datasets.csv"))
    df = df[df.to_process == KEYWORD]  # only process selected events
    rows = [row for _, row in df.iterrows()]
    rows = [(f"{row.event}_{row.region}", row.nsubregions) for row in rows]

    for storm, nsubregions in rows:
        for subregion in range(START_ON, int(nsubregions)):  #int(nsubregions)):
            event = f"{storm}_{subregion}"
            indir = join(wd, 'storm_events', storm)
            outdir =join(wd, 'feature_stats')
            logger.info(f"Add spatial features to {event}.")
            try:
                gdf, _, _ = model_utils.load_raw_data(indir, data_utils.default_features, TEMPORAL, BINARY, subset=event)
                print(f"gdf length afer loading: {len(gdf)} cells\n")
                gdf = data_utils.add_spatial_features(gdf, [event], data_utils.spatial_features, outdir, recalculate_neighbours=RECALCULATE_NEIGHBOURS, recalculate=RECALCULATE, verbose=VERBOSE)
                print(f"gdf length after adding spatial features: {len(gdf)} cells\n")
                gdf = data_utils.add_intermediate_features(gdf, [event], data_utils.intermediate_features, outdir, recalculate=RECALCULATE, verbose=VERBOSE)
                print(f"gdf length after adding intermediate features: {len(gdf)} cells\n")

            except Exception:
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
