from os.path import join
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from event import Event

wd = join("..", "data", "indata_new")
df = pd.read_csv(join(wd, "current_datasets.csv"))

for i, row in tqdm(df.iterrows()):
    try:
        event = row['event']
        region = row['region']
        nsubregions = row['nsubregions']
        print(f'Caculating {event}, {region} with {nsubregions} subregions')
        storm = Event(event, region, nsubregions)
        storm.process_all_subregions(recalculate=True)
    except Exception as e:
        with open('get_data_log.txt', 'a') as f:
            f.write(f'{str(datetime.now())}: issue with {event}, {region}: {e}\n')
            print(f"Error {e}\n")
            print(f"Logged, continuing...\n\n")
            continue