from os.path import join, dirname
import logging
import logging.handlers
import multiprocessing
import traceback


bd = dirname(__file__)
wd = join(bd, "..", "data")

log_file_path = join(bd, 'logfiles') # Wherever your log files live
log_name = 'mfe'

# next three functions enable logging with multiprocessing
def listener_configurer():
    # get root logger
    root = logging.getLogger()

    # set up logging to file
    fh = logging.FileHandler(join(log_file_path, f"{log_name}.log"), 'w')
    fh.setLevel(logging.INFO)
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    fh.setFormatter(f)

    # set up logging to terminal
    sh = logging.StreamHandler()
    sh.setLevel(logging.ERROR)

    # add file logging to root logger
    root.addHandler(fh)


def worker_configurer(queue):
    # get root logger
    root = logging.getLogger()

    # set up logging to provided queue
    qh = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    qh.setLevel(logging.INFO)

    # set up logging to terminal
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    # set up logging to file
    fh = logging.FileHandler(join(log_file_path, f"{log_name}.log"), 'a')
    fh.setLevel(logging.INFO)

    # add these to the root logger
    root.addHandler(qh)
    root.addHandler(fh)
    root.addHandler(sh)
    root.setLevel(logging.INFO)


def listener_process(queue, configurer):
    """
    Listener process top-level loop: wait for logging events (LogRecords)
    in the queue and handles them, quits when it gets a None for a
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
            print('Failure in listener_process:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


# main function of script
def process_events(i, queue, configurer):

    # worker_configurer
    configurer(queue)

    # get or create a logger with given name
    logger = logging.getLogger(f"{log_name}")

    # set up log-to-file functionality and add to this logger
    fh = logging.FileHandler(join(log_file_path, f"{log_name}.log"), 'a')
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    fh.setFormatter(f)
    logger.addHandler(fh)

    try:
        logger.info(f"logged {i}")
    except Exception:
        logger.error(traceback.format_exc())


def main():
    # start the listener for pool
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    listener = multiprocessing.Process(target=listener_process, args=(queue, listener_configurer))
    listener.start()

    # start the pool of workers
    npools = multiprocessing.cpu_count() - 1
    with multiprocessing.Pool(npools) as pool:
        params = []
        for i in range(10):
            params.append((i, queue, worker_configurer))
        pool.starmap(process_events, params)

    # close the queue and listener
    queue.put_nowait(None)
    listener.join()


if __name__ == '__main__':
    main()
