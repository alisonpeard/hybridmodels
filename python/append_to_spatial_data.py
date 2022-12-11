import pandas as pd
import geopandas as gpd

import data_utils
import model_utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from os.path import join, dirname

# environment
bd = dirname(__file__)
wd = join(bd, "..", "data")

features_to_add = ['soiltemp1', 'soiltemp2']

def main():
    gdfs = model_utils.load_all_gdfs(wd)
    gdf = pd.concat(gdfs)
    columns = [*gdf.columns]
    gdf, columns = model_utils.format_event_col(gdf, columns)
    events = [*gdf.event.unique()]
    # events = ['batsirai_menabe_0', 'batsirai_menabe_1']

    for event in events:
        try:
            gdf_event = gdf[gdf.event == event]
            data_utils.add_features_to_spatial_data(wd, event, gdf_event, features_to_add, recalculate=True)
        except Exception as e:
            print(e)



if __name__ == "__main__":
    main()
