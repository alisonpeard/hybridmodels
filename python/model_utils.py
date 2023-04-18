"""
Functions to help post-processing and modelling data.
"""
from os.path import join, basename
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from itertools import compress
from datetime import datetime, timedelta

import data_utils

"""Final features to use in model."""
test_events = ['gombe_mossuril_1', 'emnati_madagascar_3_0', 'batsirai_menabe_0',
               'noul_vietnam_0', 'roanu_satkania_0', 'megi_westernvisayas_0',
               'iota_loweraguanbasin_0', 'irma_tampasouth_0']

model_features = ["elevation", "elevation_spatial", "elevation_to_pw",
                 "jrc_permwa", "jrc_permwa_spatial", "slope_pw", "dist_pw",
                 "precip", "wind_avg", "pressure_avg", "wind_max", "pressure_min",
                 "mangrove", "mangrove_spatial", "mangrove_to_pw",
                 "evi", "evi_anom", "evi_spatial", "evi_to_pw",
                 "built_up", "bare_soil",
                 "soilcarbon", "soilcarbon_spatial", "soilcarbon_to_pw",
                 "soiltemp2", "soiltemp2_anom",
                 "aqueduct", "deltares",
                 "exclusion_mask"] 

lulc_categories = {'built_up': ['lulc__50'],
                   'bare_soil': ['lulc__60', 'lulc__70'],
                   'vegetated': ['lulc__10', 'lulc__20', 'lulc__30', 'lulc__40',
                                 'lulc__90', 'lulc__95', 'lulc_100'],
                   'water': ['lulc__80']
                  }
lulc_cols = ['lulc__20', 'lulc__30', 'lulc__40', 'lulc__50', 'lulc__60', 'lulc__70', 'lulc__80', 'lulc__90', 'lulc__95', 'lulc_100']

## MAIN DATA-LOADING FUNCTIONS
def load_all_gdfs(wd, subset=''):
    """Return all gdfs in a folder (and filter by subset string)."""

    files = [filename for filename in glob.glob(join(wd, "*.geoparquet"))]

    # make sure subset strings in files
    if not subset == '':
        subset = [f'{file}.geoparquet' for file in [subset]]
        filemask = [basename(file) in subset for file in files]
        files = list(compress(files, filemask))
    gdfs = [gpd.read_file(filename, SHAPE_RESTORE_SHX='YES') for filename in files]
    return gdfs


def load_raw_data(wd, features, temporal=False, binary=True, subset=''):
    """Load data from feature_stats: needs some processing to be usable.
    
    Parameters:
    -----------
    wd : str
        path to working directory
    features : list
        list of feature names to load (exclusing wind and pressure data)
    temporal : bool (default=False)
        whether to generate temporal or temporally-aggregated wind and pressure features
    binary : bool (default=True)
        whether to binarise flooded fraction to binary presence / absence
    subset : list (default='')
        subset of events to load, use '' for all events

    """
    # Load list of gdfs
    columns = features + ['storm', 'region', 'subregion', "geometry", "floodfrac"]
    gdfs = load_all_gdfs(wd, subset=subset)
    gdfs, features, columns = process_winds(gdfs, temporal, features, columns)
    # one big GeoDataFrame
    gdf = pd.concat(gdfs)
    gdf.attrs['transforms'] = {}
    gdf, columns = format_event_col(gdf, columns)
    
    if binary:
        gdf, features, columns = binarise_feature(gdf, 'floodfrac', 'floodfrac', data_utils.floodthresh, features, columns)
        gdf.attrs['transforms']['floodfrac'] = 'binarised flood fraction'

    gdf = gdf.replace("", np.nan)
    gdf = clean_slope(gdf)
    gdf.attrs['transforms']['slope'] = 'tidied-up slope'
    features_binary, _ = data_utils.split_features_binary_continuous(data_utils.binary_keywords, features)

    if 'lulc' in features:
        gdf, features, _, columns, _ = one_hot_encode_feature(gdf, 'lulc', features_binary, features, columns)
        for lulc_col in lulc_cols:
            if lulc_col not in [*gdf.columns]:
                gdf[lulc_col] = [0] * len(gdf)
                features.append(lulc_col)
                columns.append(lulc_col)

    gdf = gdf.reset_index(drop=True)
    return gdf, features, columns


def load_spatial_data(wd, subset='', intermediate_features=True):
    """Load data with all features from feature_stats_spatial."""
    gdfs = load_all_gdfs(wd, 'feature_stats_spatial', subset)
    gdf = pd.concat(gdfs)

    # fixes for issues with spatial data (can get rid of this after re-run)
    if intermediate_features:
        intermediate_features = [*data_utils.intermediate_features.keys()]
        intermediate_features = [f"{feat}_to_pw" for feat in intermediate_features]
        gdf[intermediate_features] = gdf[intermediate_features].replace(np.nan, 0.0)

    gdf['exclusion_mask'] = gdf['exclusion_mask'].replace(np.nan, 0.0)

    return gdf


def format_event_col(gdf, columns):
    gdf["event"] = gdf["storm"] + "_" + gdf["region"] + "_" + gdf["subregion"].astype(str)
    gdf = gdf.drop(["storm", "region", "subregion"], axis=1).reset_index(drop=True)
    for feature in ["storm", "region", "subregion"]:
        columns.remove(feature)
    columns += ['event']
    return gdf, list(set(columns))


def process_winds(gdfs, temporal, features, columns):
    """Process winds for a list of GeoDataFrames and add to feature and column lists."""
    if temporal:
        templist = []
        for gdf in gdfs:
            try:
                templist.append(get_wind_range(gdf, columns))
            except Exception as e:
                print(f"Error for {gdf.region[0]}:\n{e}")
        gdfs = templist
        columns = list(set(columns + ['Tm6', 'Tm3', 'T']))
        features = list(set(features + ['Tm6', 'Tm3', 'T']))

    else:
        climate_aggregates = ['wind_avg', 'wind_max', 'pressure_avg', 'pressure_min']
        columns = list(set(columns + climate_aggregates))
        templist = []

        for gdf in gdfs:
            try:
                templist.append(gdf.loc[:, columns])
            except Exception as e:
                print(f"Error for {gdf.region[0]}:\n{e}")
        gdfs = templist
        features = list(set(features + climate_aggregates))

    return gdfs, features, columns


def binarise_feature(gdf, old_feature, new_feature, thresh, features, columns):
    print(f"\nBinarising {old_feature} as {new_feature}...\n")
    assert (thresh <= 1) and (thresh >=0), "threshold must be a fraction"
    gdf[new_feature] = gdf[old_feature].apply(lambda x: 1 if x > thresh else 0)
    gdf[new_feature] = gdf[new_feature].astype('Int64')
    features += [new_feature]
    columns += [new_feature]
    return gdf, list(set(features)), list(set(columns))


def clean_slope(gdf):
    """Sort out the messy slope field."""
    gdf['slope_pw'] = gdf['slope_pw'].replace(np.inf, 0.0)
    gdf['slope_pw'] = gdf['slope_pw'].replace('inf', 0.0)
    gdf['slope_pw'] = gdf['slope_pw'].replace(np.nan, 0.0)
    gdf['slope_pw'] = gdf['slope_pw'].apply(lambda x: float(x) if type(x) == str else x)
    return gdf



## NOT LOADING STUFF
def confusion_label(y_test, y_pred):
    """Assign confusion label to a (true, predicted) pair."""
    if y_pred + y_test == 2:
        return 'TP'
    elif y_pred + y_test == 0:
        return 'TN'
    elif (y_test == 0) and (y_pred == 1):
        return 'FP'
    elif (y_test == 1) and (y_pred == 0):
        return 'FN'
    else:
        print(f"ISSUE: {y_test}, {y_pred}")
        return ''

def get_rows_and_cols(df, gridsize=500):
    """
    Calculate no. rows and cols in grid.

    Parameters:
    -----------
    gridsize : float
        length of one side of a square gridcell in grid.
    """
    from shapely.geometry import box
    df_pm = df.to_crs("EPSG:3857")
    bbounds = df_pm.total_bounds
    width = bbounds[2] - bbounds[0]
    height = bbounds[3] - bbounds[1]
    nrows = int(np.round(height / gridsize, 0))
    ncols = int(np.round(width  / gridsize, 0))
    return nrows, ncols


def get_wind_range(gdf, columns):
    # NOTE: working dir hardcoded here
    df = pd.read_csv(join("..", "data", "csvs", "current_datasets.csv"))

    # extract storm
    assert gdf["storm"].nunique() == 1, "One storm per gdf"
    storm = gdf["storm"][0]
    region = gdf["region"][0]
    subregion = gdf["subregion"][0]

    # get landfall/acquisition date and time
    # landfall = df[(df["event"]==storm) and (df["region"]==region)].landfall_time.reset_index(drop=True)
    landfall = df[(df["event"]==storm) & (df["region"]==region)].acquisition_time.reset_index(drop=True)
    assert landfall.nunique() == 1, f"landfall must be same for event {storm}."
    landfall = landfall[0]
    landfall_date, landfall_time = landfall.split(" ")

    # get columns for time window around landfall time
    lf = datetime.strptime(landfall, '%Y-%m-%d %H:%M')
    timedeltas = [-6, -3, 0]
    window = [lf + timedelta(hours=x) for x in timedeltas]
    wind_cols = [f"wnd{x.strftime('%m-%d_%H')}" for x in window]
    pressure_cols = [f"pressure{x.strftime('%m-%d_%H')}" for x in window]

    # create new columns and names and subset gdf
    new_cols = columns + wind_cols + pressure_cols
    new_col_names = columns + ['Wm6', 'Wm3', 'W'] + ['Pm6', 'Pm3', 'P']

    # add zeros column if any time isn't included
    for col in new_cols:
        if col not in gdf.columns:
            warnings.warn(f"{col.capitalize()} not in DataFrame for {storm.capitalize()}, {region.capitalize()}, {subregion}. "\

                          f"It's been replaced with all zeros. "\
                          f"Note this and check it's correct.\n\n")
            gdf[col] = [0.0] * len(gdf)

    gdf = gdf[new_cols]
    gdf.columns = new_col_names

    return gdf


def reshape_df(df, nrows, ncols, features):
    df = df.drop("geometry", axis=1)
    feature_mat = np.empty(shape=(nrows, ncols, len(features)), dtype="float")
    feature_key = {}
    for i, feature in enumerate(features):
        feature_arr = df[feature].values.reshape(nrows, ncols, order="F")
        feature_mat[:, :, i] = feature_arr
        feature_key[i] = feature
    return feature_mat, feature_key


def get_matrices(gdf, gridsize, features):
    """Helper function for reshaping GeoDataframe."""
    nrows, ncols = get_rows_and_cols(gdf, gridsize=gridsize)
    X, feature_key = reshape_df(gdf, nrows, ncols, features)
    y, _ = reshape_df(gdf, nrows, ncols, ["floodfrac"])

    # clip to 64x64 images
    X = X[:64, :64, :]
    y = y[:64, :64, 0]

    a, b, c = X.shape
    Xmat = np.zeros((64, 64, c), dtype=float)
    ymat = np.zeros((64, 64), dtype=float)
    Xmat[:a, :b, :] = X
    ymat[:a, :b] = y
    return Xmat, ymat, feature_key


def plot_history(history, metric='root_mean_squared_error'):
    if not type(metric)==str:
        metric = metric.__name__.lower()
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.plot(history.epoch, np.array(history.history[metric]),
           label='train')
    plt.plot(history.epoch, np.array(history.history[f'val_{metric}']),
           label = 'valid')
    plt.legend()
    plt.ylim([0, max(history.history[metric])])


def plot_prediction(test_labels, y_pred, title):
    plt.figure()
    plt.title(title)
    plt.scatter(test_labels, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([-100, 100],[-100,100])

    plt.figure()
    error = y_pred - test_labels
    plt.hist(error, bins = 50)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")



## FORMATTING DATA
def cut_feature(gdf: gpd.GeoDataFrame, feature: str, cutoff=0.001):
    """Cuts outliers past cutoff quintile from dataframe

    >>> gdf = cut_feature(gdf, 'elevation')
    >>> print(gdf.attrs['transforms']['elevation']
    """
    gdf = gdf.copy(deep=True)

    if 'transforms' not in gdf.attrs:
        gdf.attrs['transforms'] = {}

    if feature not in gdf.attrs['transforms']:
        gdf.attrs['transforms'][feature] = []

    if f"{cutoff}_cutoff" not in gdf.attrs['transforms'][feature]:
        mincut = gdf[feature].quantile(cutoff)
        maxcut = gdf[feature].quantile(1 - cutoff)

        gdf = gdf[(gdf[feature] > mincut) & (gdf[feature] < maxcut)]
        gdf.attrs['transforms'][feature].append(f"{cutoff}_cutoff")
        return gdf

    else:
        print(f'The {feature} has already had outliers cut.')
        return None


def log_feature(gdf: gpd.GeoDataFrame, feature: str, shift=None):
    """Return dataframe with log taken of selected feature.

    >>> gdf = log_feature(gdf, 'elevation')
    >>> gdf.attrs['transforms']
    """
    gdf = gdf.copy(deep=True)

    if "transforms" not in gdf.attrs:
        gdf.attrs['transforms'] = {}

    if feature not in gdf.attrs['transforms']:
        gdf.attrs['transforms'][feature] = []

    if "log" not in gdf.attrs['transforms'][feature]:
        if shift is None:
            shift = gdf[feature].min()
            shift = abs(shift) + 0.01 if shift <= 0 else 0
            print(f"shift: {shift}")

        gdf[feature] = gdf[feature].apply(lambda x: np.log(x + shift))
        gdf.attrs['transforms'][feature].append("log")
        gdf.attrs['transforms'][feature].append(f"log_shift: {shift}")
        return gdf
    else:
        print(f"The {feature} is already log-normalised.")
        return None


def normalise_per_subset(gdf, features, subset):
    """Normalise gdf on a per-subset basis."""
    gdf_ssets = []
    for sset in [*gdf[subset].unique()]:
        gdf_sset = gdf[gdf[subset] == sset]
        gdf_ssets.append(normalise_gdf(gdf_sset, features))
    gdf_norm = pd.concat(gdf_ssets)
    return gdf_norm


def normalise_gdf(gdf, features):
    """normalise and fit a PCA on all the data"""
    gdf_norm = gdf.copy(deep=True)
    for feature in features:
        gdf_norm = normalise_feature(gdf_norm, feature)
    return gdf_norm


def normalise_feature(gdf: gpd.GeoDataFrame, feature: str):
    """Return dataframe with norm taken of selected feature.

    >>> gdf = log_feature(gdf, 'elevation')
    >>> print(gdf.attrs['transforms']['elevation']
    """
    gdf = gdf.copy(deep=True)
    gdf[feature] = (gdf[feature] - gdf[feature].mean()) / gdf[feature].std()
    return gdf


def one_hot_encode_feature(gdf, feature, features_binary, features, columns, notes=[]):
    """One-hot encode a binary features, update binary features lists."""

    # cast all to string to make sure no duplicate columns
    gdf[feature] = gdf[feature].astype(str)
    onehot = pd.get_dummies(gdf[feature], prefix=f'{feature}_')
    cols = [*onehot.columns]
    gdf = pd.concat([gdf, onehot], axis=1)
    gdf = gdf.drop(columns=[feature])
    features.remove(feature)
    features_binary.remove(feature)
    features += cols
    features_binary += cols
    columns.remove(feature)
    columns += cols
    notes.append(f'One-hot encoded {feature}.')
    return gdf, list(set(features)), list(set(features_binary)), list(set(columns)), notes


"""Functions for pre-processing the data."""
def remove_feature(gdf, to_remove, columns, notes):
    gdf = gdf.drop(columns=to_remove)
    columns = [*gdf.columns]
    notes.append(f"removed {to_remove}")
    return gdf, columns, notes


def summarise_lulc(gdf, columns, features_binary, features_continuous, notes, spatial=False):
    """Summarise LULC into four categories."""

    # set up lulc dict
    lulc_summary = {value: key for key, values in lulc_categories.items() for value in values}
    if spatial:
        lulc_summary_spatial = {f'{key}_spatial': f'{value}_spatial' for key, value in lulc_summary.items()}
        lulc_summary = lulc_summary | lulc_summary_spatial

    # change gdf
    lulc_cols = [key for key in lulc_categories.keys()]
    if spatial:
        lulc_cols += [f'{key}_spatial' for key in lulc_cols]
    gdf = gdf.rename(columns=lulc_summary)
    gdf_lulc = gdf[lulc_cols].groupby(level=0, axis=1).any().astype(float)
    gdf = gdf.drop(columns=lulc_cols)# .join(gdf_lulc)
    gdf[lulc_cols] = gdf_lulc[lulc_cols]

    # change corresponding variables
    columns = [*set(lulc_summary[x] if x in lulc_summary.keys() else x for x in columns)]
    features_binary = [*set(lulc_summary[x] if x in lulc_summary.keys() else x for x in features_binary)]
    features_continuous = [*set(lulc_summary[x] if x in lulc_summary.keys() else x for x in features_continuous)]
    nfeatures = len(features_binary) + len(features_continuous)

    # make note of changes
    if "transforms" not in gdf.attrs:
        gdf.attrs['transforms'] = {}
    if 'lulc' not in gdf.attrs['transforms']:
        gdf.attrs['transforms']['lulc'] = []
    gdf.attrs['transforms']['lulc'].append('summarised into three categories')
    notes.append('summarised lulc columns into three categories')

    return gdf, columns, features_binary, features_continuous, nfeatures, notes


def random_undersample(gdf, n0, n1, notes, columns, SEED):

    gdf_flood = gdf[gdf.floodfrac==1]
    gdf_noflood = gdf[gdf.floodfrac==0]
    gdf_noflood = gdf_noflood.sample(n=n1, random_state=SEED)
    gdf_undersampled = pd.concat([gdf_flood, gdf_noflood])
    gdf_undersampled = gdf_undersampled[columns]

    # make note of changes
    if "transforms" not in gdf.attrs:
        gdf.attrs['transforms'] = {}
    gdf_undersampled.attrs = gdf.attrs
    if 'general' not in gdf_undersampled.attrs['transforms']:
        gdf_undersampled.attrs['transforms']['general'] = []
    gdf_undersampled.attrs['transforms']['general'].append('undersampled to same number of flood and non-flood.')
    notes.append('undersampled to same number of flood and non-flood pixels')

    return gdf_undersampled, notes
