"""
Functions to help processing data.
"""

from os.path import join
import logging
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
    gdf = gdf.unary_union
    overlap_list = []
    for grd in tqdm(grid.geometry):
        total = 0
        if gdf.intersects(grd):
            frac = (gdf.intersection(grd)).area / grd.area
            total += frac
            assert total <= 1  # sanity check
        overlap_list.append(total)
    grid.loc[:, 'floodfrac'] = overlap_list

    return grid

#--------------------------------------------------------------------------------------------------------
# IBTRaCs wind fields

# using this to derive 10 min wind factors
# based on Yu Mo's work and climada source code: https://climada-python.readthedocs.io/en/stable/_modules/climada/hazard/tc_tracks.html
IBTRACS_AGENCY_10MIN_WIND_FACTOR = {
    "usa": [1.0, 0.0],
    "tokyo": [1.0, 0.0],
    "newdelhi": [0.88, 0.0],  # MSW3==MSW1 in Kruk paper
    "reunion": [1.0, 0.0],
    "bom": [1.0, 0.0],
    "nadi": [1.0, 0.0],
    "wellington": [1.0, 0.0],
    'cma': [1.0, 0.0],
    'hko': [1.0, 0.0],
    'ds824': [1.0, 0.0],
    'td9636': [1.0, 0.0],
    'td9635': [1.0, 0.0],
    'neumann': [1.0, 0.0],
    'mlc': [1.0, 0.0],
    'hurdat_atl' : [0.88, 0.0],
    'hurdat_epa' : [0.88, 0.0],
    'atcf' : [0.88, 0.0],
    'cphc': [0.88, 0.0]
}

WIND_COLS = ['WMO_WIND',
 'USA_WIND',
 'CMA_WIND',
 'HKO_WIND',
 'NEWDELHI_WIND',
 'REUNION_WIND',
 'BOM_WIND',
 'NADI_WIND',
 'WELLINGTON_WIND',
 'DS824_WIND',
 'TD9636_WIND',
 'TD9635_WIND',
 'NEUMANN_WIND',
 'MLC_WIND']

DEF_ENV_PRESSURE = 1010
BASIN_ENV_PRESSURE = {
    '': DEF_ENV_PRESSURE,
    'EP': 1010, 'NA': 1010, 'SA': 1010,
    'NI': 1005, 'SI': 1005, 'WP': 1005,
    'SP': 1004,
}


def knots_to_mps(x):
    return x * 0.514


def nmile_to_km(x):
    return x * 1.852


def process_ibtracs(ibtracs_df):
    """Process IBTrACS wind and radius of max wind data."""
    # set up geometry
    ibtracs_gdf = ibtracs_df
    for col in ["LAT", "LON"]: ibtracs_gdf[col] = pd.to_numeric(ibtracs_gdf[col])
    ibtracs_gdf["geometry"] = gpd.points_from_xy(ibtracs_gdf.LON, ibtracs_gdf.LAT)
    del ibtracs_gdf["LAT"]
    del ibtracs_gdf["LON"]
    ibtracs_gdf = ibtracs_gdf.set_crs("EPSG:4326")
    assert not ibtracs_gdf.BASIN.isna().any(), "BASIN has NaN values"

    # grab most-recorded wind speed if WMO not available
    if ibtracs_gdf.WMO_WIND.isna().all():
        wind_col = ibtracs_gdf[WIND_COLS].notna().sum().idxmax()
        agency = wind_col.split("_")[0]
        pressure_col = f"{agency}_PRES"
        rmw_col = f"{agency}_RMW"

        # rescale wind speed to MSW10
        scale, shift = IBTRACS_AGENCY_10MIN_WIND_FACTOR[agency]
        ibtracs_gdf[f'{agency}_wind'.upper()] = pd.to_numeric(ibtracs_gdf[f'{agency}_wind'.upper()])
        ibtracs_gdf[f'{agency}_wind'.upper()] *= scale
        ibtracs_gdf[f'{agency}_wind'.upper()] += shift
    else:
        wind_col = "WMO_WIND"
        pressure_col = "WMO_PRES"
        agency = ibtracs_gdf["WMO_AGENCY"].mode()
        rmw_col = f"{agency}_RMW".upper() # double-check this

        # rescale wind speed to MSW10
        scale, shift = IBTRACS_AGENCY_10MIN_WIND_FACTOR[agency]
        ibtracs_gdf[f'{agency}_wind'.upper()] = pd.to_numeric(ibtracs_gdf[f'{agency}_wind'.upper()])
        ibtracs_gdf[f'{agency}_wind'.upper()] *= scale
        ibtracs_gdf[f'{agency}_wind'.upper()] += shift

    logging.info(f"Agency: {agency}")
    logging.info(f"RMW column: {rmw_col}, wind column: {wind_col}, pressure column {pressure_col}")
    logging.info(f"Scale: {scale}")
    logging.info(f"Shift: {shift}")

    # fix timestamps formatting
    newtimes = []
    for time in ibtracs_gdf["ISO_TIME"]:
        if len(time) > 8:
            date = time[:10]
            newtimes.append(time)
        else:
            newtime = f"{date} {time}"
            newtimes.append(newtime)

    ibtracs_gdf['ISO_TIME'] = newtimes
    ibtracs_gdf = ibtracs_gdf.dropna(subset=wind_col).reset_index()
    return ibtracs_gdf



def haversine(lon1, lat1, lon2_lst, lat2_lst):
    """Code from J. Verschuur. Haversine distance in km."""
    lon2_arr = np.array(lon2_lst)
    lat2_arr = np.array(lat2_lst)

    # convert degrees to radians
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2_arr = np.deg2rad(lon2_arr)
    lat2_arr = np.deg2rad(lat2_arr)

    # formula
    dlon = lon2_arr - lon1
    dlat = lat2_arr - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2_arr) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r_e = 6371
    return c * r_e

def holland_wind_field(r, wind, pressure, pressure_env, distance, lat):
    """Code from J. Verschuur. Uses different rho-value to Holland (1980).

    Parameters:
    -----------
    r : float
        radius of maximum winds (km).
    wind : float
        wind speed (m / s)
    pressure : float
        central pressure (mb == hPa)
    pressure_env : float
        ambient pressure (mb == hPa), often taken as value of first anticyclonic isobar
        Holland (1980).
    distance : float
        distance from point to storm centre (km)
    latitude : float

    """

    rho = 1.15  # Holland (1980), 1.10 Verschuur
    lat *= np.pi / 180
    distance *= 1000
    r *= 1000
    f = np.abs(1.45842300 * 10 ** -4 * np.sin(lat))
    e = 2.71828182846
    # p_drop = 2*wind**2
    p_drop = (pressure_env - pressure) * 100


    B = rho * e * wind ** 2 / p_drop
    Vg = (
        np.sqrt(
            ((r / distance) ** B) * (wind ** 2) * np.exp(1 - (r / distance) ** B)
            + (r ** 2) * (f ** 2) / 4
        )
        - (r * f) / 2
    )
    return Vg


def get_wind_fields(ibtracs_gdf,feature_gdf):
    # get centroids
    centroids = feature_gdf.to_crs("EPSG:3857").centroid.to_crs("EPSG:4326")
    wind_tracks = ibtracs_gdf.geometry
    lats = [*wind_tracks.y]
    lons = [*wind_tracks.x]

    # haversine distances
    h_distances = []
    for centroid in centroids:
        h_distances.append(haversine(centroid.x, centroid.y, lons, lats))
    h_distances = np.array(h_distances)
    assert len(ibtracs_gdf) == h_distances.shape[1],\
        "Number of haversine distances calculates did not match number of centroids"

    # calculate wind field for each time stamp
    timestamps = []
    for time in range(len(ibtracs_gdf)):

        # inputs for holland function
        h_dists = [x[time] for x in h_distances]
        basin = ibtracs_gdf["BASIN"][time]
        pressure_env = BASIN_ENV_PRESSURE[basin]
        pressure = float(ibtracs_gdf[pressure_col][time])
        lat = float(ibtracs_gdf["LAT"][time])

        if abs(pressure_env - pressure) > 0:
            # radius of maximum winds in km
            if units_df[rmw_col][0] == "nmile":
                r = nmile_to_km(float(ibtracs_gdf[rmw_col][time]))
            else:
                r = float(ibtracs_gdf[rmw_col][time])

            # maximum wind speed in mps
            if units_df[wind_col][0] == "kts":
                wind = knots_to_mps(float(ibtracs_gdf[wind_col][time]))
            else:
                wind = ibtracs_gdf[wind_col][time]

            # calculate wind field
            wind_field = []
            for distance in h_dists:
                wind_speed = holland_wind_field(r, wind, pressure, pressure_env, distance, lat)
                wind_field.append(wind_speed)

            # reformat time string
            iso_time = ibtracs_gdf['ISO_TIME'][time]
            date, time = iso_time.split(" ")
            date = date[5:].replace("-", "")
            time = time[:2]

            # if non-neglible wind, append to dataframe
            if sum(wind_field) > 0:
                feature_gdf[f"wnd{date}_{time}"] = wind_field
        else:
            logging.warning(f"No pressure drop for time {time}, skipping wind speed calculation.")

        return feature_gdf

#--------------------------------------------------------------------------------------------------------
# FABDEM (no longer using below this)
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
