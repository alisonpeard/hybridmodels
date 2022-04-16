import geopandas as gpd
import rtree
from tqdm import tqdm

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