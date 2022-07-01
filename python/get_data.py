from os.path import join, dirname
import logging
import traceback
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from event import Event

def main():
    bd = dirname(__file__)
    wd = join(bd, "..", "data")
    df = pd.read_csv(join(wd, "current_datasets.csv"))
    logging.basicConfig(filename=join(bd, 'logfiles', 'data_collection.log'),
                        filemode='a', level=logging.INFO)
    logging.info(f"\n\n\nNew run at time: {datetime.now()}")

    for i, row in tqdm(df.iterrows()):
        try:
            event = row['event']
            region = row['region']
            nsubregions = row['nsubregions']
            storm = Event(event, region, nsubregions, wd, bd)
            storm.process_all_subregions()
        except Exception:
            logging.error(traceback.format_exc())
            continue


if __name__ == "__main__":
    main()
