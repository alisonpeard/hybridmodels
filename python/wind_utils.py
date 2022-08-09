"""
Functions and variables for IBTrACs wind fields.

Modified from CLIMADA source code:
    https://climada-python.readthedocs.io/en/stable/_modules/climada/hazard/tc_tracks.html
"""
from os.path import join
import logging
from pathlib import Path
import requests
import pandas as pd
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


# {agency: [scale, shift]}
# MSW10 = scale * MSW1 + shift
IBTRACS_AGENCY_10MIN_WIND_FACTOR = {"usa": [1.0, 0.0], "tokyo": [1.0, 0.0],
                                    "newdelhi": [0.88, 0.0], "reunion": [1.0, 0.0],
                                    "bom": [1.0, 0.0], "nadi": [1.0, 0.0],
                                    "wellington": [1.0, 0.0], 'cma': [1.0, 0.0],
                                    'hko': [1.0, 0.0], 'ds824': [1.0, 0.0],
                                    'td9636': [1.0, 0.0], 'td9635': [1.0, 0.0],
                                    'neumann': [1.0, 0.0], 'mlc': [1.0, 0.0],
                                    'hurdat_atl' : [0.88, 0.0], 'hurdat_epa' : [0.88, 0.0],
                                    'atcf' : [0.88, 0.0],     'cphc': [0.88, 0.0]
}

# wind columns in IBTrACS data
WIND_COLS = ['WMO_WIND', 'USA_WIND', 'CMA_WIND', 'HKO_WIND', 'NEWDELHI_WIND',
             'REUNION_WIND', 'BOM_WIND', 'NADI_WIND', 'WELLINGTON_WIND', 'DS824_WIND',
             'TD9636_WIND', 'TD9635_WIND', 'NEUMANN_WIND', 'MLC_WIND']

# basin environmental pressures
DEF_ENV_PRESSURE = 1010
BASIN_ENV_PRESSURE = {
    '': DEF_ENV_PRESSURE,
    'EP': 1010, 'NA': 1010, 'SA': 1010,
    'NI': 1005, 'SI': 1005, 'WP': 1005,
    'SP': 1004,
}


# knots to m/s
def knots_to_mps(x):
    return x * 0.514

# nautical miles to km
def nmile_to_km(x):
    return x * 1.852


def process_ibtracs(ibtracs_df, storm):
    """
    Process IBTrACS wind and radius of max wind data.

    Note: when using WMO winds uses USA RMW for radius of max winds.
    """

    logger = logging.getLogger(f"data_collection.{storm}")

    # set up geometry
    ibtracs_gdf = ibtracs_df
    for col in ["LAT", "LON"]: ibtracs_gdf[col] = pd.to_numeric(ibtracs_gdf[col])
    ibtracs_gdf["geometry"] = gpd.points_from_xy(ibtracs_gdf.LON, ibtracs_gdf.LAT)
    ibtracs_gdf = ibtracs_gdf.set_crs("EPSG:4326")
    assert not ibtracs_gdf.BASIN.isna().any(), "BASIN has NaN values"

    # grab most-recorded wind speed if WMO not available
    if ibtracs_gdf.WMO_WIND.isna().all():
        logger.info("Not using WMO Winds.")
        wind_col = ibtracs_gdf[WIND_COLS].notna().sum().idxmax()
        agency = wind_col.split("_")[0]
        logger.info(f"Agency: {agency}")
        pressure_col = f"{agency}_PRES"
        rmw_col = f"{agency}_RMW"

        # rescale wind speed to MSW10
        scale, shift = IBTRACS_AGENCY_10MIN_WIND_FACTOR[agency.lower()]
        ibtracs_gdf[f'{agency}_wind'.upper()] = pd.to_numeric(ibtracs_gdf[f'{agency}_wind'.upper()])
        ibtracs_gdf[f'{agency}_wind'.upper()] *= scale
        ibtracs_gdf[f'{agency}_wind'.upper()] += shift
    else:
        logger.info("Using WMO Winds.")
        wind_col = "WMO_WIND"
        pressure_col = "WMO_PRES"
        agency = ibtracs_gdf["WMO_AGENCY"].mode()[0]
        logger.info(f"Agency: {agency}")
        rmw_col = f"USA_RMW"  # possibly use different later

        # rescale wind speed to MSW10
        scale, shift = IBTRACS_AGENCY_10MIN_WIND_FACTOR[agency]
        ibtracs_gdf[wind_col] = pd.to_numeric(ibtracs_gdf[wind_col])
        ibtracs_gdf[wind_col] *= scale
        ibtracs_gdf[wind_col] += shift

    logger.info(f"RMW column: {rmw_col}, wind column: {wind_col}, pressure column {pressure_col}")
    logger.info(f"Scale: {scale}")
    logger.info(f"Shift: {shift}")

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
    return ibtracs_gdf, wind_col, pressure_col, rmw_col


def haversine(lon1, lat1, lon2_lst, lat2_lst):
    """Calculate Haversine distance in km.

    Code: J. Verschuur
    """

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
    """
    Calculate the gradient wind at a given radius from the storm centre.

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

    Notes:
    ------
    Code from J. Verschuur. Uses different rho-value to Holland (1980).
    """

    rho = 1.15  # Holland (1980) air density
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


def get_wind_field(ibtracs_gdf, feature_gdf, units_df, wind_col, pressure_col, rmw_col):
    """
    Calculate wind fields from IBTrACS data for a grid GeoDataFrame.
    """

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
            date = date[5:]
            time = time[:2]

            # if non-neglible wind, append to dataframe
            if sum(wind_field) > 0:
                feature_gdf[f"wnd{date}_{time}"] = wind_field
        else:
            logging.warning(f"No pressure drop for time {time}, skipping wind speed calculation.")

    return feature_gdf
