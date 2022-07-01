from os.path import join, dirname
import logging
import logging.handlers
import multiprocessing
import traceback
import pandas as pd
from event import Event

bd = dirname(__file__)
wd = join(bd, "..", "data")

log_file_path = join(bd, 'logfiles') # Wherever your log files live
log_name = 'data_collection'

# next three functions enable logging with multiprocessing
def listener_configurer():
    root = logging.getLogger()
    fh = logging.FileHandler(join(log_file_path, f"{log_name}.log"), 'w')
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    fh.setFormatter(f)
    root.addHandler(fh)

def worker_configurer(queue):
    root = logging.getLogger()
    fh = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(sh)
    root.setLevel(logging.INFO)

def listener_process(queue, configurer):
    """
    This is the listener process top-level loop: wait for logging events
    (LogRecords)on the queue and handle them, quit when you get a None for a
    LogRecord.
    """
    configurer()
    while True:
        try:
            record = queue.get()  # pop latest item from the queue
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

# main function of script
def process_events(row, queue, configurer):
    storm = row['event']
    region = row['region']
    nsubregions = row['nsubregions']

    configurer(queue)
    logger = logging.getLogger(f"data_collection.{storm}")
    log_file_path = join(bd, 'logfiles') # Wherever your log files live
    fh = logging.FileHandler(join(log_file_path, f"data_collection.{storm}.log"), 'w')
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    fh.setFormatter(f)
    logger.addHandler(fh)

    try:
        event = Event(storm, region, nsubregions, wd, bd)
        event.process_all_subregions()
    except Exception:
        pass
        logger.error(traceback.format_exc())


def main():
    # start a listener process for the queue
    queue = multiprocessing.Queue(-1)
    listener = multiprocessing.Process(target=listener_process,
                                       args=(queue, listener_configurer))
    listener.start()

    # load and parse data
    df = pd.read_csv(join(wd, "current_datasets.csv"))
    events = [row for _, row in df.iterrows()]

    # start the workers processing the events
    workers = []
    for event in events:
        worker = multiprocessing.Process(target=process_events,
                                         args=(event, queue, worker_configurer))
        workers.append(worker)
        worker.start()
    for w in workers:
        w.join()

    # close the queue and listener
    queue.put_nowait(None)
    listener.join()


if __name__ == '__main__':
    main()
