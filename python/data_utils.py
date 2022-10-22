"""
Functions to help in data acquisition.
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
from tqdm import tqdm
from shapely.geometry import Point, LineString

from wind_utils import *

# flood threshold for binarising
floodthresh = 0
pwater_thresh = 0.99

# maintain a list of binary features
binary_keywords = ['lulc', 'aqueduct', 'deltares']

# default features (wind depends on whether temporal or not)
default_features = ["elevation", "jrc_permwa", "slope_pw", "dist_pw", "precip",
                    "soilcarbon", "mangrove", "ndvi", "aqueduct", "lulc", "deltares"]
# all features
all_features = ["elevation", "jrc_permwa", "slope_pw", "dist_pw", "precip",
                'mslp', 'sp', 'u10_u', 'u10_v', "soilcarbon", "mangrove", "ndvi", "aqueduct",
                "lulc", "deltares"]

def split_features_binary_continuous(binary_keywords, features):
    '''Split list of all features into binary and continuous according to binary keywords.'''
    binary_features = []
    continuous_features = []
    for keyword in binary_keywords:
        binary_features += [x for x in features if (keyword in x)]
    continuous_features = [x for x in features if x not in binary_features]
    return binary_features, continuous_features


# set up
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


# for flood maps
def get_grid_intersects(gdf, grid, floodcol='floodfrac'):
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
           The resulting DataFrame will contain the fraction of flooding,
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
    grid.loc[:, floodcol] = overlap_list

    return grid


# for adding auxilliary features
features_to_process = {'floodfrac': np.mean,
                       'soilcarbon': np.mean,
                       'mangrove': np.mean,
                       'ndvi': np.mean,
                       'elevation': max
                      }

def add_intermediate_features(feature_gdf, features_to_process, thresh):
    """Calculate averages along grid cells connecting dry and wet cell.
    
    This is only approximate as it is applied to the gridded dataframe rather than the
    original data.
    """

    feature_gdf_pm = feature_gdf.to_crs("EPSG:3857").reset_index(drop=True)

    # calculate max / mean of intersecting points
    for feature, func in features_to_process.items():
        feature_gdf_pm[f'{feature}_to_pw'] = [np.nan] * len(feature_gdf_pm)

    # fill gdf with empty lines
    feature_gdf_pm['line_to_pw'] = [LineString([Point(0, 0), Point(0,0)])] * len(feature_gdf_pm)

    wet_points = feature_gdf_pm[feature_gdf_pm['jrc_permwa'] > thresh].index
    dry_points = feature_gdf_pm[feature_gdf_pm['jrc_permwa'] <= thresh].index
    wet_points = feature_gdf_pm['geometry'].iloc[wet_points]
    dry_points = feature_gdf_pm['geometry'].iloc[dry_points]

    for dry_index in tqdm([*dry_points.index]):
        wet_index = wet_points.sindex.nearest(dry_points[dry_index])[1][0]  # take first point (random order)

        # make line between the points
        dry_point = dry_points.loc[dry_index].centroid
        wet_point = wet_points.iloc[wet_index].centroid
        shortest_line = LineString([dry_point, wet_point])

        # append to geodataframe for now
        feature_gdf_pm.loc[dry_index, 'line_to_pw'] = shortest_line

        # find all grid cells intersecting this linestring
        intersecting_cells = [*feature_gdf_pm[feature_gdf_pm.intersects(shortest_line)].index]

        # calculate max / mean of intersecting points
        for feature, func in features_to_process.items():
            feature_gdf_pm.loc[dry_index, f'{feature}_to_pw'] = func(feature_gdf_pm.loc[intersecting_cells, feature])
            
    del feature_gdf_pm['line_to_pw']
    return feature_gdf_pm.to_crs(4326)