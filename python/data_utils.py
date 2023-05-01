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
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.geometry import Point, LineString
from wind_utils import *


"""Global settings."""
floodthresh = 0
pwater_thresh = 20  # changed from 90 on 2023-03- (water present for >44 months)
binary_keywords = ['lulc', 'aqueduct', 'deltares', 'mangrove', 'exclusion_mask']

all_features = ["elevation", "jrc_permwa", "dist_pw", "dist_coast", "slope_coast",
                'mslp', 'sp', 'u10_u', 'u10_v',
                "precip", #  "wind_avg", "pressure_avg" not included because get processed
                "mangrove", "evi_anom", "evi", "lulc",
                "soilcarbon", "soiltemp1", "soiltemp2", "soiltemp1_anom", "soiltemp2_anom",
                "aqueduct", "deltares",
                "exclusion_mask"] 

default_features = ["elevation", "jrc_permwa", "slope_coast", "dist_pw", "dist_coast",
                    "precip",  #  "wind_avg", "pressure_avg" not included because get processed
                    "mangrove", "evi_anom", "evi", "lulc",
                    "soilcarbon", "soiltemp2", "soiltemp2_anom",
                    "aqueduct", "deltares",
                    "exclusion_mask"]

spatial_features = ['elevation', 'jrc_permwa', 'evi', 'mangrove', 'soilcarbon']
intermediate_features = {'soilcarbon': np.mean, 'mangrove': np.mean, 'evi': np.mean, 'elevation': max}


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



def get_grid_intersects(gdf, grid, col='floodfrac'):
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
            intersection = (gdf.intersection(grd)).area
            frac = intersection / grd.area
            total += frac
            assert total <= 1  # sanity check
        overlap_list.append(total)


    grid.loc[:, col] = overlap_list

    return grid


"""Functions for adding spatial features."""

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
    # sort binary and continuous features
    features_binary, features_continuous = split_features_binary_continuous(binary_keywords, features)

    # cycle through events
    for event in (pbar1 := tqdm(events, desc='events', total=len(events))):
        if recalculate or (not exists(join(wd, f'{event}.parquet'))):
            if verbose: print(f"Getting spatial features for {event}.")
            if verbose: pbar1.set_description(f"Getting spatial features for {event}...")
            gdf_event = gdf[gdf.event == event]
            gdf_event = gdf_event.reset_index(drop=True)
            features_surrounding = {feature: [] for feature in features}

            # calculate neighbours
            if recalculate_neighbours or (not exists(join(wd, 'contiguity_mats', f'{event}_contiguity.npz'))):
                if verbose: print(f"Calculating node neighbours.")
                w = weights.Queen.from_dataframe(gdf_event)
                W, _ = weights.full(w)
                ids = [*gdf_event.index]  # 0 -> 4096
                np.savez(join(wd, 'contiguity_mats', f"{event}_contiguity"), W=W, ids=ids)

            # load weight matrices and create graph
            dat = np.load(join(wd, 'contiguity_mats', f"{event}_contiguity.npz"))
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
            assert len(gdf_tosave) == 4096, "GeoDataFrame is no longer correct size."
            gdf_tosave.to_parquet(join(wd, 'dataframes', f'{event}.parquet'))
            return gdf_tosave


def add_intermediate_features(feature_gdf, events, intermediate_features, wd, thresh=pwater_thresh, recalculate=False, verbose=True, save=True):
    """Calculate averages along shortest line between dry cells and permanent water.

    This is approximate as it is applied to the gridded dataframe rather than the
    original data.

    Parameters:
    -----------
    feature_gdf : Geopandas.GeoDataFrame
    events : list
    intermediate_features : dict
        Dictionary features as keys and aggregation functions as values
    wd : str
        working directory
    thresh : float
        threshold for JRC permanent water occurance value to classify a cell as wet/dry
    recalculate : bool, default=False
    verbose : bool, default=True
    """
    for event in (pbar1 := tqdm(events, desc='events', total=len(events))):
        if recalculate:
            if verbose: pbar1.set_description(f"Getting intermediate features for {event}...")
            assert event in feature_gdf['event'].unique(), f"Event {event} not in dataframe"
            gdf_event = feature_gdf[feature_gdf.event == event].copy()
            feature_gdf_pm = gdf_event.to_crs("EPSG:3857").reset_index(drop=True)
            for feature, _ in intermediate_features.items():
                feature_gdf_pm[f'{feature}_to_pw'] = [0] * len(feature_gdf_pm)

            # get index for all wet and dry points
            wet_points = feature_gdf_pm[feature_gdf_pm['jrc_permwa'] > thresh].index
            dry_points = feature_gdf_pm[feature_gdf_pm['jrc_permwa'] <= thresh].index
            wet_points = feature_gdf_pm['geometry'].iloc[wet_points]
            dry_points = feature_gdf_pm['geometry'].iloc[dry_points]
            assert len(wet_points) > 0, f"No permanent water found with thresh {thresh}"
            assert len(dry_points) > 0, "No dry land found with thresh {thresh}"

            # iterate through all the dry points
            for dry_index in tqdm([*dry_points.index]):
                # find closest wet point
                wet_index = wet_points.sindex.nearest(dry_points[dry_index])[1][0]  # take first point (random order)

                # construct LineString between the points
                dry_point = dry_points.loc[dry_index].centroid
                wet_point = wet_points.iloc[wet_index].centroid
                shortest_line = LineString([dry_point, wet_point])

                # find all grid cells intersecting this linestring
                intersecting_cells = [*feature_gdf_pm[feature_gdf_pm.intersects(shortest_line)].index]

                # calculate aggregate function of intersecting points for each feature
                for feature, aggfunc in intermediate_features.items():
                    feature_gdf_pm.loc[dry_index, f'{feature}_to_pw'] = aggfunc(feature_gdf_pm.loc[intersecting_cells, feature])
                    assert len(feature_gdf_pm) == 4096, f"Error occured adding intermediate features for {feature},"\
                                                        "GeoDataFrame is no longer correct size."
                    
            for feature, _ in intermediate_features.items():
                assert feature_gdf_pm[feature].notnull().all(), "NaNs created"

            gdf_tosave = feature_gdf_pm.to_crs(4326)
            if save:
                for feature, _ in intermediate_features.items():
                    assert feature_gdf_pm[feature].notnull().all(), "NaNs created"
                assert len(gdf_tosave) == 4096, "GeoDataFrame is no longer correct size."
                gdf_tosave.to_parquet(join(wd, 'dataframes', f'{event}.parquet'))
            return gdf_tosave


def add_features_to_spatial_data(wd, event, gdf, features, recalculate=True):
    """
    gdf : GeoPandas.GeoDataFrame
        Feature GeoDataFrame without spatial features and containing features to be added.
    features : list
        Features to be added to spatial data.
    """
    if exists(join(wd, 'feature_stats_spatial', f'{event}.gpkg')):
        gdf_spatial = gpd.read_file(join(wd, 'feature_stats_spatial', f'{event}.gpkg'))
        assert len(gdf) == len(gdf_spatial), f"len(gdf)={len(gdf)} and len(gdf_spatial)={len(gdf_spatial)} for {event}."
        for feature in features:
            if (feature not in [*gdf_spatial.columns]) or recalculate:
                gdf_spatial[feature] = gdf[feature].replace('', np.nan).astype(float).reset_index(drop=True)
                print(f"Added {feature} to {event} GeoDataFrame")
            else:
                print(f"{feature} already in spatial GeoDataFrame for {event}")
        
        assert len(gdf_spatial) == 4096, "GeoDataFrame is no longer correct size."
        gdf_spatial.to_file(join(wd, f'{event}.gpkg'), driver='GPKG')

    else:
        print(f"No spatial data for {event}.")
