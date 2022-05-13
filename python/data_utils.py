from os.path import join
from pathlib import Path
import requests
import geopandas as gpd
import rtree
from tqdm import tqdm
import numpy as np
from math import ceil, floor
from shapely.geometry import Polygon, box
import rasterio, rasterstats
from rasterio import plot
from rasterio.merge import merge
import matplotlib.pyplot as plt

# GENERAL
def make_grid(xmin, ymin, xmax, ymax, length=1000, wide=1000):
    """ 
    Function for creating a grid using polygon objects.
    
    Must be in a metres coordinate reference system. Adapted from [1].
    Parameters
    ----------
    xmin : float, geo-coordinate
    ymin : float, geo-coordinate
    xmax : float, geo-coordinate
    ymax : float, geo-coordinate

    Returns
    -------
    grid : GeoDataFrame

    References:
    ..[1] Muckley (2020)
    """

    cols = list(np.arange(xmin, xmax, wide))
    rows = list(np.arange(ymin, ymax, length))
    rows = rows[::-1]

    polygons = []
    for x in cols:
        for y in rows:
            polygons.append( Polygon([(x,y), (x+wide, y), (x+wide, y-length), (x, y-length)]) )

    grid = gpd.GeoDataFrame({'geometry':polygons}).set_crs("EPSG:3857")
    return grid 

#--------------------------------------------------------------------------------------------------------
# Flood maps
def get_grid_intersects(gdf, grid):
    """ 
    Function for calculating the fraction of polygon in each gridcell.
    
    Each entry ranges in [0, 1]. Adapted from Muckley (2020) [1].
    
    Parameters
    ----------
    gdf  : GeoDataFrame
    grid : GeoDataFrame
    
    Returns
    -------
    df   : DataFrame
           The resuling DataFrame will contain the fraction of flooding,
           for each of the original coordinates.
           
    Reference:
    ----------
    ..[1] Muckley (2020)
            https://github.com/leomuckley/Multi-Input-ConvLSTM
    """
    poly = gpd.sjoin(grid, gdf, how='left')
    total_list = []
    for grd in tqdm(grid.geometry):
        total = 0
        for ply in gdf.geometry:
            if ply.intersects(grd):
                poly = (ply.intersection(grd)).area / grd.area
                total = total + poly
        # print(total)
        total_list.append(total)
    grid.loc[:, 'floodfrac'] = total_list
    
    return grid


#--------------------------------------------------------------------------------------------------------
# FABDEM
def download_fabdem(aoi, save_dir, new=False):
    """
    Download FABDEM file covering area of interest (AOI).
    
    Note: if AOI extends between two FABDEM ranges this needs to be run on each. This
    can be a little slower than downloading manually. If you select new=True it will
    just give you the link to the download file which is marginally faster.
    
    Parameters:
    ----------
    aoi : shapely.Polygon
        The area of interest in EPSG:4326 coordinate reference system
    save_dir : str
        Directory to save the zipfile in
    new : bool (default=False)
        Download the file. When new=False just returns the download urls.
    
    References:
    -----------
    ..[1] https://dx.doi.org/10.1088/1748-9326/ac4d4f
    """
    
    def roundup(x):
        if x >= 0:
            return int((x + 9) // 10 * 10)
        else: 
            return int(ceil((x + 9) / 10) * 10)

    def rounddown(x):
        return int((x) // 10 * 10)

    xmin, ymin, xmax, ymax = aoi.total_bounds
    xmin = rounddown(xmin)
    ymin = rounddown(ymin)
    xmax = roundup(xmax)
    ymax = roundup(ymax)
    
    def format_coords(ymin, ymax, xmin, xmax):
        ymin = f"S{-ymin:02d}" if ymin < 0 else f"N{ymin:02d}"
        ymax = f"S{-ymax:02d}" if ymax < 0 else f"N{ymax:02d}"

        xmin = f"W{-xmin:03d}" if xmin < 0 else f"E{xmin:03d}"
        xmax = f"W{-xmax:03d}" if xmax < 0 else f"E{xmax:03d}"

        filestr = f"{ymin}{xmin}-{ymax}{xmax}_FABDEM_V1-0.zip"
        fabdem_url = f"https://data.bris.ac.uk/datasets/25wfy0f9ukoge2gs7a5mqpq2j7/{filestr}"

        return fabdem_url
    

    fabdemurls = []
    if abs(ymin - ymax) <= 10 and abs(xmin - xmax) <= 10:
        fabdemurls += format(ymin, ymax, xmin, xmax)
    elif abs(ymin - ymax) > 10 and abs(xmin - xmax) <= 10:
        print("This AOI requires two fabdem zip files....")
        ymid = ymax - 10
        fabdemurls.append(format_coords(ymin, ymid, xmin, xmax))
        fabdemurls.append(format_coords(ymid, ymax, xmin, xmax))
    elif abs(xmin - xmax) > 10 and abs(ymin - ymax) <= 10:
        print("This AOI requires two fabdem zip files...")
        xmid = xmax - 10
        fabdemurls.append(format_coords(ymin, ymax, xmin, xmid))
        fabdemurls.append(format_coords(ymin, ymax, xmid, xmax))
    elif abs(xmin - xmax) > 10 and abs(ymin - ymax) > 10:
        print("This AOI requires four fabdem zip files...")
        fabdemurls.append(format_coords(ymin, ymid, xmin, xmid))
        fabdemurls.append(format_coords(ymin, ymid, xmid, xmax))
        fabdemurls.append(format_coords(ymid, ymax, xmin, xmid))
        fabdemurls.append(format_coords(ymid, ymax, xmid, xmax))
        
    print("All urls:", *fabdemurls)

    if new:
        for url in fabdemurls:
            print(f"Requesting {url}")
            r = requests.get(url)
            name = url.split('/')[-1]
            with open(join(save_dir, name), "wb") as f:
                f.write(r.content)
            print(f"Succesfully downloaded {name} to {save_dir}\n")

    return fabdemurls

def get_subfile(aoi, save_dir, fabdemurl):
    """
    Extract subfile covering AOI from a folder of .tifs.
    
    Note: if AOI extends between two FABDEM ranges this needs to be run for each.
    
    Parameters:
    ----------
    aoi : shapely.Polygon
        The area of interest in EPSG:4326 coordinate reference system
    save_dir : str
        Directory to save the zipfile in
    fabdemurl : str
        Result of download_fabdem() with target file name.
    
    References:
    -----------
    ..[1] https://dx.doi.org/10.1088/1748-9326/ac4d4f
    """
    
    filestr = fabdemurl.split('/')[-1]
    xmin, _, _, ymax = aoi.total_bounds
    xmin = floor(xmin)
    ymax = ceil(ymax)

    # get paths
    paths = []
    for path in Path(join(save_dir, filestr[:-4])).glob("*"):
        paths.append(str(path))

    
    # get min yval>ymax and highest xval<xmin
    lats = []
    lons = []
    for ip, path in enumerate(paths):

        path = path.split("/")[-1]
        lat = int(path[1:3])
        lon = int(path[4:7])
        lat = lat * (-1) if path[0]=="S" else lat
        lon = lon * (-1) if path[0]=="W" else lon

        lats.append(lat)
        lons.append(lon)
        
    latmax = max(lats)
    lonmin = min(lons)

    while (lonmin < xmin) and (len(lons) > 1):
        lons.remove(lonmin)
        lonmin = min(lons)

    while (latmax >= ymax) and (len(lats) > 1):  # was >= 13/05/2022
        lats.remove(latmax)
        latmax = max(lats)
    
    # FABDEM file syntax
    y = f"S{-latmax:02d}" if latmax < 0 else f"N{latmax:02d}"
    x = f"W{-lonmin:03d}" if lonmin < 0 else f"E{lonmin:03d}"

    filepath = join(save_dir, filestr[:-4], f"{y}{x}_FABDEM_V1-0.tif")
    return filepath