"""
Code to enable logging while using multiprocessing.

Steps:
-----
1. Start a listener process with the a queue object and the listener configurer function as args.
2. Pass the queue to the function for the workers, with the worker configurer
3. Workers will pass messages to the queue.
"""


from os.path import join
import logging
import logging.handlers
import multiprocessing


def listener_configurer(log_name, log_file_path):
    root = logging.getLogger()
    fh = logging.FileHandler(join(log_file_path, f"{log_name}.log"), 'w')
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    f = logging.Formatter('%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    fh.setFormatter(f)
    root.addHandler(fh)

def worker_configurer(queue):
    root = logging.getLogger()
    fh = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    fh.setLevel(logging.INFO)
    root.addHandler(fh)
    root.setLevel(logging.INFO)

def listener_process(queue, configurer, log_name, log_file_path):
    """
    Listener process top-level loop: wait for logging events.
    (LogRecords)on the queue and handle them, quit when you get a None for a
    LogRecord.
    """
    configurer(log_name, log_file_path)
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
            break
