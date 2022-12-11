"""
Get data, using multiple cores at once.

Loads list of events from ../data/csvs/current_datasets.csv and processes those which have 'yes' in
the to_process column.

Before Running:
---------------
Check the contents of current_datasets.csv is correct and alter the settings below.
TODO: Change this to command line flags later.

Settings:
---------
TEST_RUN : bool
    Test multiprocessing and logging working as expected without processing data.
feature_list : list or None
    Only calculate features in feature_list, or None to calculate all features.
recalculate_all : bool
    Recalculate the entire dataset from scratch. If False only recalculates or appends
    features in features_list, depending on recalculate_features.
recalculate_features : bool
    Recalculate all the features in feature_list. If False only features which haven't
    already been calculated are appended. If True everything is recalculated.

Use:
----
>>> conda activate hybridmodels
>>> cd ~/Documents/DPhil/hybridmodels/python
>>> python get_data_parallel.py
"""

# settings
TEST_RUN = False             # check logging and multiprocessing behaving correctly (currently failing)
feature_list = ['exclusion_mask'] # ['deltares']          # None for all features or else ['feature1', 'feature2']
recalculate_all = False      # false to just append to files
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
bd = dirname(__file__)
wd = join(bd, "..", "data")
log_file_path = join(bd, 'logfiles')
log_name = 'data_collection'

# function for workers
def process_events(row, queue, configurer):
    storm = row['event']
    region = row['region']
    nsubregions = int(row['nsubregions'])

    configurer(queue)
    logger = logging.getLogger(f"data_collection.{storm}")
    fh = logging.FileHandler(join(log_file_path, f"data_collection.{storm}.log"), 'a')
    f = logging.Formatter('%(asctime)s %(name)s %(levelname)-8s: %(message)s')
    fh.setFormatter(f)
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(f)
    logger.addHandler(fh)
    logger.addHandler(sh)

    try:
        event = Event(storm, region, nsubregions, wd, bd)
        if not TEST_RUN:
            logger.info(f"Setting up Storm {storm.capitalize()} Event instance for "\
                    f"{region.capitalize()} with {nsubregions} subregions.")
            event.make_grids()
            event.process_all_subregions(recalculate_all, recalculate_features, feature_list)
            # event.get_all_features(2, recalculate_all, recalculate_features, feature_list)  # just one event
            # event.get_era5(0)  # just one field for one event
        else:
            logger.info(event)
    except Exception:
        logger.error(traceback.format_exc())


def main():
    # load and parse data
    df = pd.read_csv(join(wd, "csvs", "current_datasets.csv"))
    df = df[df.to_process == "yes"]  # only process selected events
    events = [row for _, row in df.iterrows()]

    # start the listener for pool
    manager = multiprocessing.Manager()
    queue = manager.Queue(200)
    listener = multiprocessing.Process(target=listener_process, args=(queue, listener_configurer, log_name, log_file_path))
    listener.start()

    # start the pool of workers
    npools = multiprocessing.cpu_count() - 1
    with multiprocessing.Pool(npools) as pool:
        params = []
        for event in events:
            params.append((event, queue, worker_configurer))
        pool.starmap(process_events, params)

    # close the queue and listener
    queue.put_nowait(None)
    listener.join()


if __name__ == '__main__':
    main()
