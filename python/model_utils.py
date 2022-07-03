"""
Functions to help modelling data.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

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

def reshape_df(df, nrows, ncols, features):
    df = df.drop("geometry", axis=1)
    feature_mat = np.empty(shape=(nrows, ncols, len(features)), dtype="float")
    for i, feature in enumerate(features):
        feature_arr = df[feature].values.reshape(nrows, ncols, order="F")
        feature_mat[:, :, i] = feature_arr
    return feature_mat

def format_gdf(filelist, columns, binary=True, thresh=0.5, storm=None):
    """Format GeoDataframe from list of files."""
    l = [gpd.read_file(filename, SHAPE_RESTORE_SHX='YES') for filename in filelist]
    gdf = pd.concat(l, axis=0)
    gdf = gdf[columns]
    
    print("Number of storms:", gdf["storm"].nunique())
    print("Number of regions:", gdf["region"].nunique())
    
    if storm:
        print(f"\nExtracting storm {storm}...\n")
        gdf = gdf[gdf["storm"]==storm]
    
    gdf["event"] = gdf["storm"] + "_" + gdf["region"] + "_" + gdf["subregion"].astype(str)
    gdf = gdf.drop(["storm", "region", "subregion"], axis=1).reset_index(drop=True)
    for feature in ["storm", "region", "subregion"]:
        columns.remove(feature)
    
    if binary:
        print("\nBinarising floodfrac...\n")
        assert (thresh <= 1) and (thresh >=0), "thresh must be a fraction"
        gdf['floodfrac'] = gdf['floodfrac'].apply(lambda x: 1 if x > thresh else 0)
        
    return gdf, columns

def get_matrices(gdf, gridsize, features):
    """Helper function for reshaping GeoDataframe."""
    nrows, ncols = get_rows_and_cols(gdf, gridsize=gridsize)
    X = reshape_df(gdf, nrows, ncols, features)
    y = reshape_df(gdf, nrows, ncols, ["floodfrac"])
    
    # clip to 64x64 images
    X = X[:64, :64, :]
    y = y[:64, :64, 0]
    
    a, b, c = X.shape
    Xmat = np.zeros((64, 64, c), dtype=float)
    ymat = np.zeros((64, 64), dtype=float)
    Xmat[:a, :b, :] = X
    ymat[:a, :b] = y
    return Xmat, ymat

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