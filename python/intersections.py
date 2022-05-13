"""
Command line utility to calculate overlap for user input grid and polygon ESRI shapefile.

References:
-----------
..[1] Muckley (2020)

Examples:
--------
>>> python3 intersections.py --g '../beira/indata/grid.shp' \
     -p '../beira/indata/trueflood_pm.shp' -o '../beira/outdata'
"""


from os.path import join
import argparse, os
import geopandas as gpd
from utils import get_grid_intersects

def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

# command line input
parser = argparse.ArgumentParser(description='Calculate overlap between grid and polygon.')
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gridpath", dest="gridpath", required=True, type=validate_file,
                        help="input grid path", metavar="FILE")
parser.add_argument("-p", "--polypath", dest="polypath", required=True, type=validate_file,
                        help="input polygon path", metavar="FILE")
parser.add_argument("-o", "--outpath", dest="outpath", required=True, type=validate_file,
                        help="output file path", metavar="FILE")
args = parser.parse_args()

# load files
grid_gdf = gpd.read_file(args.gridpath)
poly_gdf = gpd.read_file(args.polypath)

# calculate intersections and output
print("'\nCalculating intersections...\n")
gdf = get_grid_intersects(poly_gdf, grid_gdf)
gdf.to_file(join(args.outpath, "intersect.shp"))

# print summary statistics
print(f"\n\nFinished calculating intersections.\nOutput to {args.outpath}/intersect.shp")
print(f"Average intersection: {gdf.floodfrac.mean()}")
print(f"Maximum intersection: {gdf.floodfrac.max()}")
print(f"Minimum intersection: {gdf.floodfrac.min()}")
print(f"Standard deviation of intersections: {gdf.floodfrac.std()}\n\n\n")
