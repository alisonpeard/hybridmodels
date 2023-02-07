"""
Use:
----
>>> conda activate hybridmodels
>>> cd ~/Documents/DPhil/hybridmodels/python
>>> python get_data_parallel.py
"""

# settings
storm = "roanu"
region = "chittagong"
subregion = 0
nsubregions = 4
feature_list = None                # None for all features or else ['feature1', 'feature2']
recalculate_all = True            # False to just append to files
recalculate_features = True

# imports
from os.path import join, dirname
import logging
import logging.handlers
import multiprocessing
import traceback
import pandas as pd
from event import Event
from multiprocessing_logging import listener_configurer, worker_configurer, listener_process

# environment
global bd, wd, log_file_path, log_name
bd = dirname(__file__)
wd = join(bd, "..", "data")
log_file_path = join(bd, 'logfiles')
log_name = 'data_collection'

# function for workers
def main(storm, region, subregion, nsubregions):

    logger = logging.getLogger(f"data_collection.{storm}")
    fh = logging.FileHandler(join(log_file_path, f"data_collection.{storm}.log"), 'a')
    f = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s: %(message)s')
    fh.setFormatter(f)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(f)
    logger.addHandler(fh)
    logger.addHandler(sh)

    try:
        event = Event(storm, region, nsubregions, wd, bd)
        logger.info(f"Setting up Storm {storm.capitalize()} Event instance for "\
                f"{region.capitalize()} with {nsubregions} subregions.")
        event.make_grids()
        if feature_list is not None:
            for feature in feature_list:
                getattr(event, f"get_{feature}")(subregion, recalculate_features)
        else:
            event.get_all_features(subregion, recalculate_all, recalculate_features, feature_list)
    except Exception:
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main(storm, region, subregion, nsubregions)
