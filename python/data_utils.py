"""
Functions to help in data acquisition and post-processing.
"""
from os.path import join, exists
import logging
from pathlib import Path
import requests
from tqdm.notebook import tqdm, tnrange
import numpy as np
from math import ceil, floor
import networkx as nx
from pysal.lib import weights
from scipy import sparse
import pandas as pd
import geopandas as gpd
import rtree
from shapely.geometry import Polygon, box
import rasterio, rasterstats
from rasterio import plot
from rasterio.merge import merge
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Point, LineString
from wind_utils import *


"""Global settings."""
floodthresh = 0
pwater_thresh = 98
binary_keywords = ['lulc', 'aqueduct', 'deltares']
default_features = ["elevation", "jrc_permwa", "slope_pw", "dist_pw", "precip",
                    "soilcarbon", "mangrove", "ndvi", "aqueduct", "lulc", "deltares"]
all_features = ["elevation", "jrc_permwa", "slope_pw", "dist_pw", "precip",
                'mslp', 'sp', 'u10_u', 'u10_v', "soilcarbon", "mangrove", "ndvi", "aqueduct",
                "lulc", "deltares"]


"""Data acquisition functions."""
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


"""Functions for adding spatial features."""
spatial_features = ['dist_pw', 'lulc__90', 'deltares', 'lulc__20', 'aqueduct', 'lulc__80', 'lulc__30', 'soilcarbon', 'slope_pw',
                    'precip', 'jrc_permwa', 'mangrove', 'lulc__40', 'ndvi', 'lulc__60', 'lulc__10', 'lulc__50', 'elevation',
                    'lulc__95']
intermediate_features = {'soilcarbon': np.mean, 'mangrove': np.mean, 'ndvi': np.mean, 'elevation': max}

def process_continuous_neighbours(gdf, feature, neighbours):
    """Calculate mean of a group of cell neighbours."""
    gdf[feature] = gdf[feature].replace('', np.nan) # catch any erroneous strings
    gdf[feature] = gdf[feature].astype(float)       # temporary cast to float type
    return gdf[feature][neighbours].mean()


def process_binary_neighbours(gdf, feature, neighbours):
    """Check if any list of neighbour cells are True."""
    gdf[feature] = gdf[feature].replace('', np.nan) # catch any erroneous strings
    gdf[feature] = gdf[feature].astype(float)       # temporary cast to float type
    return int((gdf[feature][neighbours] > 0).any())


def split_features_binary_continuous(binary_keywords, features):
    '''Split list of all features into binary and continuous according to binary keywords.'''
    binary_features = []
    continuous_features = []
    for keyword in binary_keywords:
        binary_features += [x for x in features if (keyword in x)]
    continuous_features = [x for x in features if x not in binary_features]
    return binary_features, continuous_features


def add_spatial_features(gdf, events, features, wd, recalculate_neighbours=False, recalculate=False, verbose=True):
    """Should really change postfix to _neighbours..."""
    # sort binary and continuous features
    features_binary, features_continuous = split_features_binary_continuous(binary_keywords, features)

    # cycle through events
    for event in (pbar1 := tqdm(events, desc='events', total=len(events))):
        if recalculate or (not exists(join(wd, 'feature_stats_spatial', f'{event}.gpkg'))):
            if verbose: print(f"Getting spatial features for {event}.")
            if verbose: pbar1.set_description(f"Getting spatial features for {event}...")
            gdf_event = gdf[gdf.event == event]
            gdf_event = gdf_event.reset_index(drop=True)
            features_surrounding = {feature: [] for feature in features}

            # calculate neighbours
            if recalculate_neighbours or (not exists(join(wd, 'feature_stats_spatial', f'{event}_contiguity.npz'))):
                if verbose: print(f"Calculating node neighbours.")
                w = weights.Queen.from_dataframe(gdf_event)
                W, _ = weights.full(w)
                ids = [*gdf_event.index]  # 0 -> 4096
                np.savez(join(wd, 'feature_stats_spatial', f"{event}_contiguity"), W=W, ids=ids)

            # load weight matrices and create graph
            dat = np.load(join(wd, 'feature_stats_spatial', f"{event}_contiguity.npz"))
            W, ids = dat['W'], dat['ids']
            id_dict = {old: new for old, new in zip(range(4096), ids)}
            G = nx.from_numpy_array(W)
            G = nx.relabel_nodes(G, id_dict, copy=True)

            # cycle through node's neighbours for each event
            for node in tqdm(ids, desc=f'neighbours {event}', total=len(ids)):
                neighbours = [*G.neighbors(node)]

                # cycle through continuous features
                for feature in features_continuous:
                    mean = process_continuous_neighbours(gdf, feature, neighbours)
                    features_surrounding[feature].append(mean)

                # cycle through binary features
                for feature in features_binary:
                    presence = process_binary_neighbours(gdf, feature, neighbours)
                    features_surrounding[feature].append(presence)
            
            # append to new dataframe
            gdf_tosave = pd.DataFrame.from_dict(features_surrounding, orient='columns')
            gdf_tosave.set_index(ids, drop=True, inplace=True)
            gdf_tosave = gdf_event.merge(gdf_tosave, left_index=True, right_index=True, suffixes=('', '_spatial'))
            gdf_tosave.to_file(join(wd, 'feature_stats_spatial', f'{event}.gpkg'), driver='GPKG')


def add_intermediate_features(feature_gdf, events, intermediate_features, wd, thresh=pwater_thresh, recalculate=False, verbose=True):
    """Calculate averages along grid cells connecting dry and wet cell.
    
    This is only approximate as it is applied to the gridded dataframe rather than the
    original data.
    """
    for event in (pbar1 := tqdm(events, desc='events', total=len(events))):
        if recalculate:
            if verbose: pbar1.set_description(f"Getting intermediate features for {event}...")
            gdf_event = feature_gdf[feature_gdf.event == event]
            feature_gdf_pm = gdf_event.to_crs("EPSG:3857").reset_index(drop=True)

            # calculate max / mean of intersecting points
            for feature, func in intermediate_features.items():
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
                for feature, func in intermediate_features.items():
                    feature_gdf_pm.loc[dry_index, f'{feature}_to_pw'] = func(feature_gdf_pm.loc[intersecting_cells, feature])

            del feature_gdf_pm['line_to_pw']
            gdf_tosave = feature_gdf_pm.to_crs(4326)
            gdf_tosave.to_file(join(wd, 'feature_stats_spatial', f'{event}.gpkg'), driver='GPKG')