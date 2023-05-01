# TODO: make this show variable names
import tracemalloc
import linecache

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

tracemalloc.start()

# settings
FEATURE_LIST = ['permwater']         # None for all features or else ['feature1', 'feature2']
RECALCULATE_ALL = False               # false to just append to files
RECALCULATE_FEATURES = False
KEYWORD = "yes"


from os.path import join, dirname
import pandas as pd
from event import Event

# environment
bd = dirname(__file__)
datadir = join(bd, "..", "data")
df = pd.read_csv(join(datadir, "csvs", "current_datasets.csv"))

snapshot = tracemalloc.take_snapshot()
print("========== SNAPSHOT =============")
for stat in snapshot.statistics("lineno")[:10]:
    print(stat)
    print(stat.traceback.format())
    print('\n')
print('\n')

for stat in snapshot.statistics("lineno")[:10]:
    if "pandas" in stat.traceback._frames[0].f_globals["__file__"]:
        print(stat)
        print('\n')

# print("[ Top 10 ]")
# for stat in top_stats[:10]:
#     print(stat)



# from os.path import join, dirname
# import pandas as pd

# from event import Event

# environment
# bd = dirname(__file__)
# datadir = join(bd, "..", "data")

# def main():
#     # load and parse data
#     df = pd.read_csv(join(datadir, "csvs", "current_datasets.csv"))
#     df = df[df.to_process == KEYWORD]  # only process selected events
#     events = [row for _, row in df.iterrows()]
#     for event in events:
#         storm = event['event']
#         region = event['region']
#         nsubregions = int(event['nsubregions'])
#         try:
#             pass
#             wd = join(datadir, "storm_events", f"{storm}_{region}")
#             event = Event(storm, region, nsubregions, wd, datadir)
#             event.process_all_subregions(RECALCULATE_ALL, RECALCULATE_FEATURES, FEATURE_LIST)
#         except Exception as e:
#             print(e)
#             pass



# main()

