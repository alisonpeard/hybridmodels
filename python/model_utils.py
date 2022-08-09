"""
Functions to help modelling data.
"""
from os.path import join
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from itertools import compress
from datetime import datetime, timedelta


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


def get_data(wd, features, temporal, binary, storm=""):
    columns = features + ['storm', 'region', 'subregion', "geometry", "floodfrac"]

    # list of files to use
    files = [filename for filename in glob.glob(join(wd, "feature_stats", "*.gpkg"))]
    filemask = [storm in file for file in files]
    files = list(compress(files, filemask))

    # generate the GeoDataFrame
    gdf, columms = format_gdf(files, columns, temporal=temporal, binary=binary)
    gdf = gdf.replace("", np.nan)

    # fill null slope values
    gdf['slope_pw'] = gdf['slope_pw'].replace(np.inf, 0)
    gdf['slope_pw'] = gdf['slope_pw'].replace(np.nan, 0)

    gdf = gdf.dropna().reset_index(drop=True)

    features = features + ['Tm6', 'Tm3', 'T'] if temporal else features + ['wind_avg']
    
    return gdf, features


def format_gdf(filelist, columns, temporal=True, binary=True, thresh=0.5):
    """Format GeoDataframe from list of files."""
    gdfs = [gpd.read_file(filename, SHAPE_RESTORE_SHX='YES') for filename in filelist]

    # process winds
    if temporal:
        gdfs = [get_wind_range(gdf, columns) for gdf in gdfs]
    else:
        columns = columns + ["wind_avg"]
        gdfs = [gdf[columns] for gdf in gdfs]

    gdf = pd.concat(gdfs, axis=0)

    print("Number of storms:", gdf["storm"].nunique())
    print("Number of regions:", gdf["region"].nunique())

    gdf["event"] = gdf["storm"] + "_" + gdf["region"] + "_" + gdf["subregion"].astype(str)
    gdf = gdf.drop(["storm", "region", "subregion"], axis=1).reset_index(drop=True)
    for feature in ["storm", "region", "subregion"]:
        columns.remove(feature)

    if binary:
        print("\nBinarising floodfrac...\n")
        assert (thresh <= 1) and (thresh >=0), "thresh must be a fraction"
        gdf['floodfrac'] = gdf['floodfrac'].apply(lambda x: 1 if x > thresh else 0)

    return gdf, columns


def get_wind_range(gdf, columns):
    # NOTE: working dir hardcoded here
    df = pd.read_csv(join("..", "data", "csvs", "current_datasets.csv"))

    # extract storm
    assert gdf["storm"].nunique() == 1, "One storm per gdf"
    storm = gdf["storm"][0]
    region = gdf["region"][0]
    subregion = gdf["subregion"][0]

    # get landfall date and time
    landfall = df[df["event"]==storm].landfall_time.reset_index(drop=True)
    assert landfall.nunique() == 1, f"landfall must be same for event {storm}."
    landfall = landfall[0]
    landfall_date, landfall_time = landfall.split(" ")

    # get columns for time window around landfall time
    lf = datetime.strptime(landfall, '%Y-%m-%d %H:%M')
    timedeltas = [-6, -3, 0]
    window = [lf + timedelta(hours=x) for x in timedeltas]
    wind_cols = [f"wnd{x.strftime('%m-%d_%H')}" for x in window]

    # create new columns and names and subset gdf
    new_cols = columns + wind_cols
    new_col_names = columns + ['Tm6', 'Tm3', 'T']

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
    plt.ylim([0, max(history.history['loss'])])


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
