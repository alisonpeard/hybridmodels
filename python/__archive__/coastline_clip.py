from os.path import join
import numpy as np
from multiprocessing import Pool, cpu_count
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# start with unprocessed Idai IBTrACs data
indir = join("..", "beira", "indata")
outdir = join("..", "beira", "outdata")

# load IBTrACs data and calculate bounding box
wind_df = pd.read_csv(join(indir, "ibtracs.csv"), header=0)[1:]
wind_df = wind_df.dropna(subset="WMO WIND")
wind_gdf = gpd.GeoDataFrame(wind_df, geometry=gpd.points_from_xy(wind_df.LON, wind_df.LAT))
bbox = box(*wind_gdf.total_bounds)

# load coastline data
print("Loading coastline data...")
coast_gdf = gpd.read_file(join(indir, "osm_processed", "osm_coastline_final.shp"))
print("Finished loading osm_coastline.shp...\n")

if __name__ == '__main__':

    # multiprocessing
    print("Starting multiprocessing...")
    def clip(gdf):
        clipped = gpd.clip(gdf, bbox)
        return clipped

    def parallelize_dataframe(gdf, func, n_cores=4):
        gdf_split = np.array_split(gdf, n_cores)
        pool = Pool(n_cores)
        gdf = pd.concat(pool.map(func, gdf_split))
        pool.close()
        pool.join()
        return gdf

    # output results
    clipped_gdf = parallelize_dataframe(coast_gdf, clip)
    clipped_gdf.to_file(join(outdir, "osm_clipped.shp"))
    print("Fini!")
