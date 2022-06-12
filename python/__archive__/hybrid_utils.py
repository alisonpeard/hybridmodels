from model_utils import reshape_df, get_rows_and_cols
import numpy as np
import rasterio
from rasterio.crs import CRS

def gdf_to_tiff(gdf, outdir, crs_epsg=4326):
    """Doesn't preserve geographic info.
    
    TODO: See geocube for this.
    """

    nrows, ncols = get_rows_and_cols(gdf)
    mat = reshape_df(gdf, nrows, ncols, ["elevation"])[:, :, 0]
    mat = mat[np.newaxis, :, :]

    out_meta = {"driver": "GTiff",
                     "height": nrows,
                     "width": ncols,
                     "crs": CRS.from_epsg(crs_epsg),
                     "count": 1,
                     "dtype": 'float32'
                    }

    with rasterio.open(outdir, "w", **out_meta) as dest:
        dest.write(mat)
    print(f"Saved GeoTIFF to {outdir}.\n")
      
    src = rasterio.open(outdir, "r")
    im = src.read(1)
    
    return im