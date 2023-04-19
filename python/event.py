import os
os.environ['USE_PYGEOS'] = '0'  # should solve runtime warnings
from os.path import join, exists
from os import remove
import warnings
from ast import literal_eval
import logging
from numbers import Number
import pystac_client

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, shape
from shapely.ops import nearest_points

# for deltares
import dask.distributed
import xarray as xr
# import pystac_client  # issue on old Macbook
import rasterio
import rioxarray

import ee
import geemap  # need later

from data_utils import *
from model_utils import *

# Connection rest by peer error
import httplib2shim
httplib2shim.patch()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# NOTE: correspond to processing functions and not final features
default_features = ["floodfrac", "elevation", "permwater", "pw_dists", "coast_dists", "precipitation",
                    "era5", "soilcarbon","soiltemp", "mangroves", "evi", "wind_fields", "aqueduct",
                    "lulc", "deltares", "exclusion_mask"]

class Event:
    """
    Event class for holding storm events and extracting features.


    Attributes:
    ----------
    storm : string
        Name of storm as in IBTrACS data.
    region : string
        Name of region of interest as in current_datasets.csv.
    nsubregions : int
        Number of subregions RoI is subdivided into in event_geojsons.csv.
    bd : filepath
        Base directory
    wd : filepath
        Working directory
    gridsize : int
        Size of grid cells each subregion will be divided into.


    Methods:
    ------
    make_grids(self)
        Creates a list of grid attributes representing the grid for each subregion.
    process_all_subregions(self, recalculate_all=False, recalculate_features=False, feature_list=None)
        Iterate through subregions in the event and calculate all features in feature_list for each. if
        recalculate_all is True, the data is recalculated from scratch. If recalculate_all is False but
        recalculate_features is True, then only the features in feature_list are recalculated from scratch.
        If feature_list is None, the global default_features are calculated/recalculated. If it is a list,
        only those in the list are calculated/recalculated.
    get_all_features(self, subregion, recalculate_all=False, recalculate_features=False, feature_list=None)
        Calculate all features for a subregin. if
        recalculate_all is True, the data is recalculated from scratch. If recalculate_all is False but
        recalculate_features is True, then only the features in feature_list are recalculated from scratch.
        If feature_list is None, the global default_features are calculated/recalculated. If it is a list,
        only those in the list are calculated/recalculated.


    Examples:
    ---------
    To calculate all features for a new event.
    >>> event = Event(storm, region, nsubregions, wd, bd)
    >>> event.make_grids()
    >>> event.process_all_subregions(recalculate_all=False, recalculate_features=False, feature_list=None)
    """


    def __init__(self, storm, region, nsubregions, wd, bd, gridsize=500):
        """Set up instance of the storm Event."""

        # set up logging
        logger = logging.getLogger(f"data_collection.{storm}")
        self.logger = logger

        # set up attributes
        self.storm = storm
        self.region = region
        self.gridsize = gridsize
        self.nsubregions = nsubregions
        self.subregions = [x for x in range(nsubregions)]

        self.wd = wd                                     # working dir
        self.bd = bd                                     # base dir

        self.startdate, self.enddate = [*pd.read_csv(join(self.bd, "csvs", "event_dates.csv"),
                                   index_col="storm").loc[storm]]
        
        current_dataset  = pd.read_csv(join(self.bd, 'csvs', 'current_datasets.csv'),
                                            index_col=['event', 'region']).loc[(self.storm, self.region)]
        self.acquisition_time = current_dataset['acquisition_time']
        self.year = int(self.enddate[:4])
        self.sid = current_dataset['sid']

        self.aoi_pm = [None] * nsubregions
        self.aoi_lonlat = [None] * nsubregions
        self.grid_pm = [None] * nsubregions
        self.grid_lonlat = [None] * nsubregions
        self._feature_gdf = [None] * nsubregions
        self.aoi_ee = [None] * nsubregions
        self.location = [None] * nsubregions
        self.grid_ee = [None] * nsubregions

        self._aoi_event = None
        self._flood = None
        self._oceanmask = None
        self._coastline = None

        self._connected_to_gee = -1

        self.get_flood()
        self.get_aoi()


    def __repr__(self):
        string = f"\nStorm {self.storm.capitalize()} in {self.region.capitalize()}:\n"\
                "-------------------------------\n"

        for subregion, feature_gdf in enumerate(self._feature_gdf):
            if feature_gdf is not None:
                field_info = f"subregion {subregion} fields: {[*feature_gdf.columns]}\n"
                string += field_info
            else:
                string += f"subregion {subregion} not yet processed.\n"

        return string


    def get_gdf(self, subregion, recalculate=False):
        """Load pre-existing gdf or create one if recalculating or doesn't exist."""
        grid_lonlat = self.grid_lonlat[subregion]
        assert grid_lonlat is not None, "Need to call Event.make_grids() first."

        if recalculate:
            self.logger.info("Recalculating shapefile...")
            feature_gdf = gpd.GeoDataFrame(grid_lonlat)
            feature_gdf["storm"] = [self.storm] * len(feature_gdf)
            feature_gdf["region"] = [self.region] * len(feature_gdf)
            feature_gdf["subregion"] = [subregion] * len(feature_gdf)

            feature_gdf = feature_gdf.set_crs("EPSG:4326")
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)
        else:
            file = join(self.wd, f'feature_stats_{subregion}.parquet')
            self.logger.info(f"Looking for {file}")
            try:
                self._feature_gdf[subregion] = gpd.read_parquet(file)
                self.logger.info(f"Loaded existing shapefile {file}...")
            except:
                self.logger.info("No shapefile exists, creating new one...")
                feature_gdf = gpd.GeoDataFrame(grid_lonlat)
                feature_gdf["storm"] = [self.storm] * len(feature_gdf)
                feature_gdf["region"] = [self.region] * len(feature_gdf)
                feature_gdf["subregion"] = [subregion] * len(feature_gdf)

                feature_gdf = feature_gdf.set_crs("EPSG:4326")
                self._feature_gdf[subregion] = feature_gdf
                self.save_gdf(subregion)
    

    def save_gdf(self, subregion):
        """Save/update the GeoDataFrame as a GeoParquet file."""
        feature_gdf = self.get_feature_gdf(subregion)
        assert len(feature_gdf) == 64 * 64, f"Grid not of size 64x64"
        file = join(self.wd, f'feature_stats_{subregion}.parquet')
        feature_gdf.to_parquet(file)
        self.logger.info(f"Saved as {file}...\n")


    def get_grid(self, subregion):
        """Make spatial grid from geoJSON file for given subregion."""

        # grab approx. aois from geoJSON file
        geoJSON = literal_eval(pd.read_csv(join(self.bd, "csvs", "current_datasets.csv"),
                                           index_col="region").loc[self.region]['subregion_geojsons'])
        poly = [shape(x['geometry']) for x in geoJSON['features']]
        poly = gpd.GeoDataFrame({"geometry": poly[subregion]}, index=[0])
        poly = poly.set_crs("EPSG:4326").to_crs("EPSG:3857").geometry[0]

        # construct aoi polygon of standard 32x32 km size
        (x, y) = (poly.centroid.x, poly.centroid.y)
        dx = 16000
        dy = 16000
        poly = box(x - dx, y - dy, x + dx, y + dy)
        aoi_pm = gpd.GeoDataFrame({"geometry": poly}, index=[0], crs="EPSG:3857")
        aoi_lonlat = aoi_pm.to_crs("EPSG:4326")

        # create a grid over the aoi
        grid_pm = make_grid(*aoi_pm.total_bounds, length=self.gridsize, wide=self.gridsize)
        grid_lonlat = grid_pm.to_crs("EPSG:4326")

        # redefine aoi to make sure aoi and grid are totally aligned
        aoi_pm = gpd.GeoDataFrame({'geometry': grid_pm.unary_union}, index=[0], crs='EPSG:3857')
        aoi_lonlat = aoi_pm.to_crs("EPSG:4326")

        # save the grid
        grid_lonlat.to_parquet(join(self.wd, f'grid_lonlat_{subregion}.parquet'))

        # set all instance variables
        self.aoi_pm[subregion] = aoi_pm
        self.aoi_lonlat[subregion] = aoi_lonlat
        self.grid_pm[subregion] = grid_pm
        self.grid_lonlat[subregion] = grid_lonlat


    def make_grids(self):
        """Calculate the grid for all subregions of an event."""
        for n in self.subregions:
            self.get_grid(n)

    def get_feature_gdf(self, subregion):
        if self._feature_gdf[subregion] is None:
            self.get_gdf(subregion)
        return self._feature_gdf[subregion]
    
    def get_flood(self):
        """Load Copernicus EMS flood polygon."""
        if self._flood is None:
            flood = gpd.read_file(join(self.wd, "flood.gpkg")).to_crs('EPSG:4326')
            if exists(join(self.wd, "hydrographyA.gpkg")):
                pwater = gpd.read_file(join(self.wd, "hydrographyA.gpkg"))
                trueflood = gpd.overlay(flood, pwater, how="difference")
            else:
                trueflood = flood
            self._flood = trueflood
        return self._flood


    def get_aoi(self):
        if self._aoi_event is None:
            self._aoi_event = gpd.read_file(join(self.wd, "areaOfInterest.gpkg")).to_crs('EPSG:4326')
        return self._aoi_event


    def get_oceanmask(self):
        """Get ocean mask for the area of interest."""
        if self._oceanmask is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                aoi = self.get_aoi()
                self._oceanmask = gpd.read_parquet(join(self.bd, 'openstreetmap', 'water-osm.parquet')).to_crs(4326).clip(mask=aoi)
        return self._oceanmask
        
    def get_coastline(self):
            """Get coastline for the area of interest."""
            if self._coastline is None:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    aoi = self.get_aoi()
                    self._coastline = gpd.read_parquet(join(self.bd, 'openstreetmap', 'coastlines-osm.parquet')).to_crs(4326).clip(mask=aoi)
            return self._coastline
    

    def get_floodfrac(self, subregion, recalculate=False, cols=['det_method', 'obj_desc']):
        """Calculate flood fraction per gridcell."""

        feature_gdf = self.get_feature_gdf(subregion)

        if "floodfrac" not in feature_gdf or recalculate:
            # calculate flood fraction over the grid cells in Pseudo-Mercator
            self.logger.info("Calculating flood fractions...")
            aoi_lonlat  = self.aoi_lonlat[subregion]
            grid_pm = self.grid_pm[subregion]
            flood = self.get_flood()

            assert aoi_lonlat.crs == flood.crs,\
                    f"Flood file and subregion geometries have different crs {subregion} for {self.region.upper()}."

            flood = function_wrapper(flood, aoi_lonlat, gpd.overlay, **{'how': 'intersection'})

            assert len(flood) > 0,\
                    f"Flood file does not intersect subregion {subregion} for {self.region.upper()}."

            flood_pm = flood.to_crs("EPSG:3857")
            floodfrac_gdf = function_wrapper(flood_pm, grid_pm, get_grid_intersects)
            feature_gdf["floodfrac"] = floodfrac_gdf["floodfrac"]

            try:
                # get extra data on flood detection
                grid_flooded = function_wrapper(feature_gdf.reset_index(drop=False), flood, gpd.overlay, **{'how': 'intersection', 'keep_geom_type': False})
                grid_flooded = grid_flooded[['index'] + cols].groupby('index').agg(pd.Series.mode)
                feature_gdf  = pd.merge(feature_gdf, grid_flooded, left_index=True, right_index=True, how='left').fillna("")
            except Exception as e:
                self.logger.info(f"{self.storm}_{self.region}_{subregion}: error adding extra flood info:\n{e}\nCreating empty fields")
                for col in cols:
                    feature_gdf[col] =  [np.nan] * len(feature_gdf)

            # save the feature_stats shapefile with flood fraction
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def start_gee(self, subregion):
        """Initialize Google Earth Engine with service account."""
        if self._connected_to_gee != subregion:
            # workaround to solve conflict with collections
            self.logger.info("Connecting to Google Earth Engine...\n")
            import collections
            collections.Callable = collections.abc.Callable

            # initialize GEE
            try:
                ee.Initialize()
            except:
                service_account = "hybrid-models@hybridmodels-354115.iam.gserviceaccount.com"
                credentials = ee.ServiceAccountCredentials(service_account, join("gcloud_keys", ".hybridmodels-354115-e71f122c7f06.json"))
                ee.Initialize(credentials)

            self._connected_to_gee = subregion

            aoi_lonlat = self.aoi_lonlat[subregion]
            grid_lonlat = self.grid_lonlat[subregion]

            # convert aoi to a GEE Feature Collection
            aoi_ee = ee.Geometry.Polygon(aoi_lonlat.geometry[0].__geo_interface__["coordinates"],
                                        proj=ee.Projection('EPSG:4326'))
            location = aoi_ee.centroid().coordinates().getInfo()[::-1]

            # convert grid to a GEE Feature Collection
            features = []
            for geom in grid_lonlat.geometry:
                poly = ee.Geometry.Polygon(geom.__geo_interface__['coordinates'],
                                    proj=ee.Projection('EPSG:4326'))
                features.append(poly)

            grid_ee = ee.FeatureCollection(features)
            self.logger.info(f"Grid size: {grid_ee.size().getInfo()}")

            self.aoi_ee[subregion] = aoi_ee
            self.location[subregion] = location
            self.grid_ee[subregion] = grid_ee
    
    
    def get_gebco(self, subregion, recalculate=False):
        """Download GEBCO raster from GEE.

        GEBCO bathymetry data: 15 arcseconds (approx. 0.5km)
        """
        feature_gdf = self.get_feature_gdf(subregion)
        if "gebco" not in feature_gdf or recalculate:
            self.logger.info("Calculating GEBCO bathymetry...")
            try:
                self.start_gee(subregion)
                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]
                gebco = ee.Image(ee.ImageCollection("projects/sat-io/open-datasets/gebco/gebco_grid")
                                  .filterBounds(aoi_ee)
                                  .select("b1")
                                  .median()
                                  .clip(aoi_ee)
                                  .unmask(999))
                ocean = gebco.lte(0)
                mean_gebco = gebco.mask(ocean).unmask(0).reduceRegions(collection=grid_ee,
                                                                       reducer=ee.Reducer.mean(),
                                                                       scale=self.gridsize)
                gebco_list = mean_gebco.getInfo()
                gebco_list = [feat['properties'].get('mean', np.nan) for feat in gebco_list['features']]
                feature_gdf["gebco"] = gebco_list
            except Exception as e:
                self.logger.warning(f"Error for GEBCO for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty fields.")
                feature_gdf["gebco"] = [np.nan] * len(feature_gdf)
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_aqueduct(self, subregion, recalculate=False):
        """Download Aqueduct flood hazard maps.

        Chooses map with most correlation to flood...
        """
        feature_gdf = self.get_feature_gdf(subregion)

        # get data from GEE
        if f"aqueduct" not in feature_gdf or recalculate:
            try:
                self.start_gee(subregion)
                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]
                # cycle through all return periods
                rps = ['25', '50', '100']
                corrs = {}
                for rp in rps:
                    try:
                        # get GEE image collection
                        self.logger.info(f"Calculating Aqueduct for RP 1-in-{rp} year...")
                        flood_depths = ee.Image(ee.Image(f"projects/hybridmodels-2022/assets/inuncoast_historical_nosub_hist_rp{rp.zfill(4)}_0")
                                          # .filterBounds(aoi_ee)
                                          # .select("rp")
                                          .clip(aoi_ee)
                                          .unmask(0))
                        flood_present = flood_depths.gt(0)
                        # NOTE: mean() works here because frac is sum(pixels) / len(pixels), same as mean()
                        floodfrac = flood_present.reduceRegions(collection=grid_ee, reducer=ee.Reducer.mean(), scale=self.gridsize)
                        aqueduct_list = floodfrac.aggregate_array('mean').getInfo()
                        feature_gdf[f"aqueduct_{rp}"] = aqueduct_list
                        feature_gdf[f"aqueduct_{rp}"] = feature_gdf[f"aqueduct_{rp}"].replace(np.nan, 0.0)
                        corr = feature_gdf['floodfrac'].corr(feature_gdf[f"aqueduct_{rp}"])
                        assert not np.isnan(corr), f"Got a NaN correlation for RP {rp}."
                        corrs[rp] = corr
                    except Exception as e:
                        self.logger.error(f"{self.storm}, {self.region}, {subregion}:\n{e}")

                allzero = True
                for rp in rps:
                    if feature_gdf[f"aqueduct_{rp}"].sum() > 0:
                        allzero = False
                # choose rp most correlated to floodfrac
                if not allzero:
                    best_rp = max(corrs, key=corrs.get)
                    feature_gdf['aqueduct'] = feature_gdf[f"aqueduct_{best_rp}"]
                    self.logger.info(f"Chose return period {best_rp} for {self.storm}, {self.region}, {subregion}\n"
                                     f"with correlation {max(corrs.values()):.4f}")
                else:
                    feature_gdf['aqueduct'] = [0] * len(feature_gdf)

            except Exception as e:
                self.logger.error(f"Error for Aqueduct data for {self.storm}, {self.region}, {subregion}"\
                                    f"\n{e}\nCreating empty fields.")
                feature_gdf["aqueduct"] = [np.nan] * len(feature_gdf)

            # save results
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_deltares(self, subregion, recalculate=False):
        """Download Deltares flood data.

        Deltares coastal flood hazard data.
        """
        feature_gdf = self.get_feature_gdf(subregion)

        # get data from Deltares
        if "deltares" not in feature_gdf or recalculate:
            try:
                rp = 100
                slr = 2018
                self.logger.info(f"Calculating Deltares flood map RP{rp}.")
                minx, miny, maxx, maxy = feature_gdf.unary_union.bounds

                # load deltares data
                client = dask.distributed.Client(processes=False)
                catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1/")
                search = catalog.search(
                    collections=["deltares-floods"],
                    query={
                        "deltares:dem_name": {"eq": "NASADEM"},
                        "deltares:sea_level_year": {"eq": slr},
                        "deltares:return_period": {"eq": rp},
                    },
                )
                item = next(search.get_items())
                url = item.assets["index"].href
                ds = xr.open_dataset(f"reference::{url}", engine="zarr", consolidated=False, chunks={})
                ds_aoi = ds.sel(lat=slice(miny, maxy), lon=slice(minx, maxx), time=ds.time[0])
                flooded = ds_aoi.inun.where(ds_aoi.inun > 0, 0)
                flooded = flooded.where(flooded==0, 1)
                flooded.rio.set_crs('epsg:4326');

                # convert to src and overlay to feature stats grid
                filename = join('tempfiles', f'src_temp_{self.region}_{subregion}.tiff')
                flooded.rio.to_raster(filename)
                src = rasterio.open(filename)
                remove(filename)

                feature_gen = rasterio.features.dataset_features(src, geographic=True, as_mask=True)
                feature_list = [feature for feature in feature_gen]
                geom = [shape(i['geometry']) for i in feature_list]
                # values = [i['properties']['val'] for i in feature_list]
                flood_gdf = gpd.GeoDataFrame({'geometry':geom}).set_crs(4326)
                feature_gdf = function_wrapper(flood_gdf, feature_gdf, get_grid_intersects, **{'col': 'deltares'})

            except Exception as e:
                self.logger.error(f"Error for Deltares data for {self.storm}, {self.region}, {subregion}:"\
                                    f"\n{e}\n\nCreating empty fields.")
                feature_gdf["deltares"] = [np.nan] * len(feature_gdf)

            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_fabdem(self, subregion, recalculate=False):
        """
        Get FABDEM DTM from Google Earth Engine.

        From awesome-gee-community-datasets.

        """
        feature_gdf = self.get_feature_gdf(subregion)

        if "fabdem" not in feature_gdf or recalculate:
            self.logger.info("Calculating FABDEM DTM...")
            try:
                self.start_gee(subregion)
                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]

                elev = ee.Image(ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
                    .filterBounds(aoi_ee)
                    .mosaic()
                    .setDefaultProjection('EPSG:4326', scale=30)
                    .clip(aoi_ee)
                    .unmask(-999))
                land = elev.gte(0)

                # Add reducer output to the Features in the collection.
                mean_elev = elev.mask(land).unmask(0).reduceRegions(collection=grid_ee,
                                                                    reducer=ee.Reducer.mean(),
                                                                    scale=self.gridsize)

                elev_list = mean_elev.getInfo()
                elev_list = [feat['properties'].get('mean', np.nan) for feat in elev_list['features']]
                feature_gdf["fabdem"] = elev_list

            except Exception as e:
                self.logger.warning(f"Error for fabdem for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty fields.")
                feature_gdf["fabdem"] = [np.nan] * len(feature_gdf)

            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_elevation(self, subregion, recalculate=False):
        """Calculate approx. elevation from GEBCO and FABDEM."""
        feature_gdf = self.get_feature_gdf(subregion)
        if "elevation" not in feature_gdf or recalculate:
            self.get_gebco(subregion)
            self.get_fabdem(subregion)
            feature_gdf = self.get_feature_gdf(subregion)
            feature_gdf['elevation'] = feature_gdf["gebco"] + feature_gdf["fabdem"]
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_permwater(self, subregion, recalculate):
        """Get JRC Permanent water from Google Earth Engine.

        JRC permanent water dataset: 30 arcseconds (approx. 1km).
        """
        feature_gdf = self.get_feature_gdf(subregion)
        if "jrc_permwa" not in feature_gdf or recalculate:
            self.logger.info("Recalculating permanent water...")
            try:
                self.start_gee(subregion)
                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]

                # Add reducer output to the Features in the collection.
                jrc_permwater = (ee.Image("JRC/GSW1_3/GlobalSurfaceWater")
                                .clip(aoi_ee)
                                .select("occurrence")
                                .unmask(0))
                mean_jrc_permwater = jrc_permwater.reduceRegions(collection=grid_ee,
                                                                 reducer=ee.Reducer.mean(),
                                                                 scale=self.gridsize)
                jrc_permwater_list = mean_jrc_permwater.getInfo()
                jrc_permwater_list = [feat['properties'].get('mean', np.nan) for feat in jrc_permwater_list['features']]
                feature_gdf["jrc_permwa"] = jrc_permwater_list

                # fill in any missing data in ocean (making all ocean have 100% occurence)
                ocean = self.get_oceanmask()
                assert feature_gdf.crs == ocean.crs, "gdfs have different coordinated reference systems."
                ocean_bool = inline_function_wrapper(feature_gdf, ocean.unary_union, 'intersects')
                ocean_ix = feature_gdf[ocean_bool].index
                feature_gdf.loc[ocean_ix, 'jrc_permwa'] = 100

            except Exception as e:
                self.logger.warning(f"Error for jrc_permwa for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty fields.")
                feature_gdf["jrc_permwa"] = [np.nan] * len(feature_gdf)

            # save to gdf
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_coast_dists(self, subregion, recalculate=False):
        """Calculate cell distances and slopes to nearest coastal water."""
        feature_gdf = self.get_feature_gdf(subregion)
        if "dist_coast" not in feature_gdf or recalculate:
            self.logger.info(f"Recalculating distances to coastline with using OSM coastline.")
            feature_gdf = self._feature_gdf[subregion]
            aoi_pm = self.aoi_pm[subregion]
            try:
                if "elevation" not in feature_gdf:
                    self.get_elevation(subregion)

                coastline = self.get_coastline().to_crs(3857)
                feature_gdf['dist_coast'] = inline_function_wrapper(feature_gdf.to_crs(3857), coastline.unary_union, 'distance')
                feature_gdf['slope_coast'] = (feature_gdf['elevation'] / feature_gdf['dist_coast']).replace(-np.inf, 0.0).replace(np.inf, 0.0)

            except Exception as e:
                self.logger.warning(f"Error for dist_coast and slope_coast for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty fields.")
                feature_gdf["dist_coast"] = [np.nan] * len(feature_gdf)
                feature_gdf["slope_coast"] = [np.nan] * len(feature_gdf)

            # save to gdf
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)

    
    def get_pw_dists(self, subregion, recalculate=False, thresh=pwater_thresh):
        """NOTE: Old, using dist_coast instead now."""

        feature_gdf = self.get_feature_gdf(subregion)

        if "dist_pw" not in feature_gdf or recalculate:
            self.logger.info(f"Recalculating distances to water with occurrence threshold={thresh}...")
            try:
                if "elevation" not in feature_gdf:
                    self.get_elevation(subregion)

                self.start_gee(subregion)
                aoi_ee = self.aoi_ee[subregion]

                def dist_to_water(gdf, water, geom_col='geometry'):
                    gdf = gdf.to_crs('EPSG:3857')
                    water = water.set_crs('EPSG:4326').to_crs('EPSG:3857')
                    water_union = water.unary_union
                    dist_list =[]
                    for _, row in gdf.iterrows():
                        nearest = nearest_points(row[geom_col], water_union)[1]
                        dist = row[geom_col].distance(nearest)
                        dist_list.append(dist)
                    return dist_list

                jrc_permwater = (ee.Image("JRC/GSW1_3/GlobalSurfaceWater")
                                 .clip(aoi_ee)
                                 .select("occurrence")
                                 .unmask(0))
                water = jrc_permwater.gt(thresh)
                water = water.reduceToVectors(reducer=ee.Reducer.countEvery())

                water_gdf = geemap.ee_to_geopandas(water).set_crs(4326)
                water_gdf = water_gdf[water_gdf.label == 1]
                ocean = self.get_oceanmask()
                water_gdf = function_wrapper(water_gdf, ocean, gpd.overlay, **{'how': 'union'})

                dist_list = dist_to_water(feature_gdf, water_gdf)
                feature_gdf['dist_pw'] = dist_list
            except Exception as e:
                self.logger.warning(f"Error for dist_pw a for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty fields.")
                feature_gdf["dist_pw"] = [np.nan] * len(feature_gdf)

            # save to gdf
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_precipitation(self, subregion, recalculate=False):
        """CHIRPS Daily Precipitation: 0.05 degrees daily"""

        feature_gdf = self.get_feature_gdf(subregion)
        if "precip" not in feature_gdf or recalculate:
            features = ['precipitation']  # could get other data from this dataset
            self.start_gee(subregion)
            aoi_ee = self.aoi_ee[subregion]
            grid_ee = self.grid_ee[subregion]

            for feature in features:
                self.logger.info(f"Calculating CHIRPS daily {feature} averages...")
                try:
                    # time range to average over
                    start = ee.Date(self.startdate)
                    end = ee.Date.parse('YYYY-MM-dd HH:mm', self.acquisition_time)

                    feat = ee.Image(ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                                      .select(feature)
                                      .filterBounds(aoi_ee)
                                      .filterDate(start, end)
                                      .mean()
                                      .clip(aoi_ee))

                    # unmask using the spatial average
                    spatial_mean = feat.reduceRegions(aoi_ee, ee.Reducer.mean(), scale=self.gridsize)
                    spatial_mean = spatial_mean.getInfo()['features'][0]['properties']['mean']
                    feat = feat.unmask(spatial_mean)

                    # Add reducer output to the Features in the collection.
                    mean_feat = feat.reduceRegions(collection=grid_ee,
                                                   reducer=ee.Reducer.mean(),
                                                   scale=self.gridsize)
                    feat_list = mean_feat.getInfo()
                    feat_list = [feat['properties'].get('mean', np.nan) for feat in feat_list['features']]

                    feature_gdf[feature[:6]] = feat_list

                except Exception as e:
                    self.logger.warning(f"Error for {feature} for {self.storm}, {self.region}, {subregion}:"\
                                        f"{e}\nCreating empty field.")
                    feature_gdf[feature[:6]] = [np.nan] * len(feature_gdf)

            # save output
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)



    def get_lulc(self, subregion, recalculate=False):
        """Land use and land cover."""

        feature_gdf = self.get_feature_gdf(subregion)
        if "lulc" not in feature_gdf or recalculate:
            self.start_gee(subregion)
            aoi_ee = self.aoi_ee[subregion]
            grid_ee = self.grid_ee[subregion]

            self.logger.info(f"Calculating dominant LULC type per grid cell...")
            try:
                feat = ee.Image(ee.ImageCollection("ESA/WorldCover/v100")
                                  .filterBounds(aoi_ee)
                                  .mode()
                                  .clip(aoi_ee)
                                  .unmask(0))

                # Add reducer output to the Features in the collection.
                mode_feat = feat.reduceRegions(collection=grid_ee,
                                               reducer=ee.Reducer.mode(),
                                               scale=self.gridsize)
                # feat_list = mode_feat.aggregate_array('mode').getInfo()
                feat_list = mode_feat.getInfo()
                feat_list = [feat['properties'].get('mode', np.nan) for feat in feat_list['features']]
                feature_gdf['lulc'] = feat_list
            except Exception as e:
                self.logger.warning(f"Error for lulc for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty field.")
                feature_gdf["lulc"] = [np.nan] * len(feature_gdf)
            
            feature_gdf['lulc'] = feature_gdf['lulc'].astype(str)  # make sure string for one-hot-encoding
            # save output
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_era5(self, subregion, recalculate=False):
        """ERA5 Daily MSLP, surface pressure and (x, y) MSW10 wind components.

        Dataset available: 1979-01-02T00:00:00Z – 2020-07-09T00:00:00
        """
        # which features from ERA5 Daily Aggregates to use
        feature_names = ['mslp', 'sp', 'u10_u', 'u10_v']
        features = ['mean_sea_level_pressure', 'surface_pressure', 'u_component_of_wind_10m', 'v_component_of_wind_10m']
        feature_gdf = self.get_feature_gdf(subregion)

        if pd.to_datetime('1979-01-02T00:00:00') <= pd.to_datetime(self.acquisition_time) <= pd.to_datetime('2020-07-09T00:00:00'):
            
            if "mslp" not in feature_gdf or recalculate:
                self.start_gee(subregion)
                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]

                for feature, feature_name in zip(features, feature_names):
                    self.logger.info(f"Calculating ERA5 {feature_name}...")
                    try:
                        feat = ee.Image(ee.ImageCollection("ECMWF/ERA5/DAILY")
                                        .select(feature)
                                        .filterBounds(aoi_ee)
                                        .filterDate(self.startdate, self.enddate)
                                        .mean()
                                        .clip(aoi_ee))

                        # unmask using the spatial average
                        spatial_mean = feat.reduceRegions(aoi_ee,
                                                        ee.Reducer.mean(),
                                                        crs="EPSG:4326",
                                                        scale=self.gridsize)
                        spatial_mean = spatial_mean.getInfo()['features'][0]['properties']['mean']
                        feat = feat.unmask(spatial_mean)

                        # Add reducer output to the Features in the collection.
                        mean_feat = feat.reduceRegions(collection=grid_ee,
                                                    reducer=ee.Reducer.mean(),
                                                    scale=self.gridsize,
                                                    crs="EPSG:4326")

                        # feat_list = mean_feat.aggregate_array('mean').getInfo()
                        feat_list = mean_feat.getInfo()
                        feat_list = [feat['properties'].get('mean', np.nan) for feat in feat_list['features']]
                        feature_gdf[feature_name] = feat_list

                    except Exception as e:
                        self.logger.warning(f"Error for {feature_name} for {self.storm}, {self.region}, {subregion}:"\
                                            f"{e}\nCreating empty field.")
                        feature_gdf[feature_name] = [np.nan] * len(feature_gdf)
        else:
            self.logger.info(f"{self.storm}, {self.region} outside of ERA5 time range:"\
                            f"\nCreating empty field.")
            for feature_name in feature_names:
                feature_gdf[feature_name] = [np.nan] * len(feature_gdf)
        
        # save output
        self._feature_gdf[subregion] = feature_gdf
        self.save_gdf(subregion)


    def get_soiltemp(self, subregion, recalculate=False, advance=1):
        """ERA5 Land Monthly Averaged ECMWF Climate Reanalysis.

        Dataset coverage: 1981-01-01T00:00:00Z – 2022-08-01T00:00:00

        Parameters:
        -----------
        advance : int (default=1)
            Number of months before storm start date to begin collecting data.
        """
        # which features from ERA5 Daily Aggregates to use
        feature_names = ['soiltemp1', 'soiltemp2']
        features = ['soil_temperature_level_1', 'soil_temperature_level_2']
        feature_gdf =  self.get_feature_gdf(subregion)
        if pd.to_datetime('1981-01-01T00:00:00') <= pd.to_datetime(self.acquisition_time) <= pd.to_datetime('2022-08-01T00:00:00'):
            if "soil_temperature_level_1" not in feature_gdf or recalculate:
                self.start_gee(subregion)
                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]

                for feature, feature_name in zip(features, feature_names):
                    self.logger.info(f"Calculating ERA5 {feature_name}...")
                    try:
                        collection = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY").select(feature)
                        start = ee.Date(self.startdate).advance(-advance, 'month')
                        end = ee.Date.parse('YYYY-MM-dd HH:mm', self.acquisition_time)
                        baseyear = 1981
                        feat, anomaly = get_anomaly(collection, start, end, baseyear, aoi_ee, self.gridsize, spatial_mean)

                        # Add reducer output to the Features in the collection.
                        mean_feat = feat.reduceRegions(collection=grid_ee,
                                                    reducer=ee.Reducer.mean(),
                                                    scale=self.gridsize,
                                                    crs="EPSG:4326")
                        feat_list = mean_feat.getInfo()
                        feat_list = [feat['properties'].get('mean', np.nan) for feat in feat_list['features']]
                        feature_gdf[feature_name] = feat_list

                        # Add anomaly reducer output to Features
                        mean_anomaly = anomaly.reduceRegions(collection=grid_ee,
                                                    reducer=ee.Reducer.mean(),
                                                    scale=self.gridsize,
                                                    crs="EPSG:4326")
                        anomaly_list = mean_anomaly.getInfo()
                        anomaly_list = [feat['properties'].get('mean', np.nan) for feat in anomaly_list['features']]
                        feature_gdf[f'{feature_name}_anom'] = anomaly_list


                    except Exception as e:
                        self.logger.warning(f"Error for {feature_name} for {self.storm}, {self.region}, {subregion}:"\
                                            f"{e}\nCreating empty field.")
                        feature_gdf[feature_name] = [np.nan] * len(feature_gdf)
                        feature_gdf[f'{feature_name}_anom'] = [np.nan] * len(feature_gdf)
        else:
            self.logger.info(f"{self.storm}, {self.region} outside of ERA5 soiltemp time range:"\
                            f"\nCreating empty field.")
            for feature_name in feature_names:
                feature_gdf[feature_name] = [np.nan] * len(feature_gdf)
            feature_gdf[feature_name] = [np.nan] * len(feature_gdf)
            feature_gdf[f'{feature_name}_anom'] = [np.nan] * len(feature_gdf)

        # save output
        self._feature_gdf[subregion] = feature_gdf
        self.save_gdf(subregion)


    def get_soilcarbon(self, subregion, recalculate=False):
        """Get soil organic carbon from Google Earth Engine."""

        feature_gdf = self.get_feature_gdf(subregion)
        if "soilcarbon" not in feature_gdf or recalculate:
            self.logger.info("Calculating soil carbon...")
            try:
                self.start_gee(subregion)
                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]

                soilcarbon = (ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02")
                              .select('b0')
                              .clip(aoi_ee)
                              .unmask(0))

                # Add reducer output to the Features in the collection
                soilcarbon.projection().getInfo()
                mean_soilcarbon = soilcarbon.reduceRegions(collection=grid_ee,
                                                         reducer=ee.Reducer.mean(), scale=self.gridsize)

                sc_dict = mean_soilcarbon.getInfo()
                soilcarbon_list = [feature['properties'].get('mean', np.nan) for feature in sc_dict['features']]
                feature_gdf["soilcarbon"] = soilcarbon_list
            except Exception as e:
                self.logger.warning(f"Error for soilcarbon for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty field.")
                feature_gdf["soilcarbon"] = [np.nan] * len(feature_gdf)

            # save output file
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_mangroves(self, subregion, recalculate=False):
        """Mangrove forests from year 2000 (Giri, 2011).
        
        Dataset has 30m resolution and binary presence absence values. This function calculates
        the fraction of each cell covered by mangrove."""
        feature_gdf = self.get_feature_gdf(subregion)

        if "mangrove" not in feature_gdf or recalculate:
            self.logger.info("Calculating mangrove cover...")
            try:
                # set up Google Earth Engine for subregion
                self.start_gee(subregion)
                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]

                mangrove = ee.Image(ee.ImageCollection("LANDSAT/MANGROVE_FORESTS")
                                           .filterBounds(aoi_ee)
                                           .first()
                                           .clip(aoi_ee)
                                           .unmask(0))

                # Add reducer output to the Features in the collection
                mean_mangrove = mangrove.reduceRegions(collection=grid_ee,
                                                         reducer=ee.Reducer.mean(), scale=self.gridsize)
                mangrove_list = mean_mangrove.aggregate_array('mean').getInfo()
                feature_gdf["mangrove"] = mangrove_list

            except Exception as e:
                self.logger.warning(f"Error for mangrove for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty field.")
                feature_gdf["mangrove"] = [np.nan] * len(feature_gdf)

            # save output
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_evi(self, subregion, recalculate=False, advance=2):
        """EVI (reprojected and masked from mangroves)

        Dataset coverage: 2000-02-18T00:00:00Z–2022-12-19T00:00:00

        Parameters:
        ----------
        advance : int (default=2)
            Number of months before storm start date to average evi from.
        """

        feature_gdf = self.get_feature_gdf(subregion)

        if "evi" not in feature_gdf or recalculate:
            self.logger.info("Calculating EVI...")

            try:
                # set up Google Earth Engine for subregion
                self.start_gee(subregion)
                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]

                # timeframe and collection
                collection = ee.ImageCollection("MODIS/006/MOD13Q1").select('EVI')
                start = ee.Date(self.startdate).advance(-advance, 'month')
                end = ee.Date.parse('YYYY-MM-dd HH:mm', self.acquisition_time)
                baseyear = 2000
                evi, anomaly = get_anomaly(collection, start, end, baseyear, aoi_ee, self.gridsize, 0)

                # calculate mean over feature collection
                mean_evi = evi.reduceRegions(collection=grid_ee,
                                               reducer=ee.Reducer.mean(), scale=self.gridsize)

                # evi_list = mean_evi.aggregate_array('mean').getInfo()
                evi_list = mean_evi.getInfo()
                evi_list = [feat['properties'].get('mean', np.nan) for feat in evi_list['features']]
                feature_gdf["evi"] = evi_list

                # Add anomaly reducer output to Features
                mean_anomaly = anomaly.reduceRegions(collection=grid_ee,
                                               reducer=ee.Reducer.mean(),
                                               scale=self.gridsize,
                                               crs="EPSG:4326")
                # anomaly_list = mean_anomaly.aggregate_array('mean').getInfo()
                anomaly_list = mean_anomaly.getInfo()
                anomaly_list = [feat['properties'].get('mean', np.nan) for feat in anomaly_list['features']]
                feature_gdf[f'evi_anom'] = anomaly_list

            except Exception as e:
                self.logger.warning(f"Error for evi for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty field.")
                feature_gdf["evi"] = [np.nan] * len(feature_gdf)
                feature_gdf["evi_anom"] = [np.nan] * len(feature_gdf)

            # save output
            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_wind_fields(self, subregion, recalculate=False):
        """Get wind fields from IBTrACs data using Holland (1980) method.

        Adds wind field columns where wind is nonzero to feature GeoDataFrame and
        average over all these wind field columns (i.e., over storm duration)
        """
        feature_gdf = self.get_feature_gdf(subregion)

        if not any("wnd" in col for col in feature_gdf.columns) or recalculate:
            self.logger.info("Calculating wind speeds...")

            try:
                # load and process IBTrACs data
                if self.year > 2017:
                    ibtracs_gdf = gpd.read_file(join(self.bd, 'ibtracs', "ibtracs_last3years.csv"))[1:].replace(" ", np.nan)
                    ibtracs_gdf = ibtracs_gdf[ibtracs_gdf.NAME == self.storm.upper()]
                    units_df = pd.read_csv(join(self.bd, 'ibtracs', "ibtracs_last3years.csv"), dtype=str, header=0)[0:1]
                else:
                    ibtracs_gdf = gpd.read_file(join(self.bd, 'ibtracs', "ibtracs_since1980.csv"))[1:].replace(" ", np.nan)
                    ibtracs_gdf = ibtracs_gdf[ibtracs_gdf.NAME == self.storm.upper()]
                    units_df = pd.read_csv(join(self.bd, 'ibtracs', "ibtracs_since1980.csv"), dtype=str, header=0)[0:1]

                # process IBTrACS data
                ibtracs_gdf, wind_col, pressure_col, rmw_col = process_ibtracs(ibtracs_gdf, self.storm)
                feature_gdf = get_wind_field(ibtracs_gdf, feature_gdf, units_df, wind_col, pressure_col, rmw_col, self.acquisition_time)

                # save average wind field
                timemask = ["wnd" in col for col in feature_gdf.columns]
                timestamps = feature_gdf.columns[timemask]
                feature_gdf["wind_avg"] = feature_gdf[timestamps].mean(axis=1)
                feature_gdf['wind_max'] = feature_gdf[timestamps].max(axis=1)

                timemask = ["pressure" in col for col in feature_gdf.columns]
                timestamps = feature_gdf.columns[timemask]
                feature_gdf["pressure_avg"] = feature_gdf[timestamps].mean(axis=1)
                feature_gdf["pressure_min"] = feature_gdf[timestamps].min(axis=1)

            except Exception as e:
                self.logger.warning(f"Error for wind fields for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty field.")
                feature_gdf["wind_avg"] = [np.nan] * len(feature_gdf)
                feature_gdf["wind_max"] = [np.nan] * len(feature_gdf)
                feature_gdf["pressure_avg"] = [np.nan] * len(feature_gdf)
                feature_gdf["pressure_min"] = [np.nan] * len(feature_gdf)

            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_exclusion_mask(self, subregion, recalculate=False):
        """Add exclusion mask if exclusion_mask.gpkg present in event directory.

        Exclusion mask from Copernicus GFM.
        """

        feature_gdf = self.get_feature_gdf(subregion)
        if "exclusion_mask" not in feature_gdf or recalculate:
            self.logger.info("Calculating exclusion mask...")

            try:
                feature_gdf['exclusion_mask'] = [np.nan] * len(feature_gdf)
                filepath = join(self.wd, "exclusion_mask.gpkg")
                if exists(filepath):
                    feature_gdf = self._feature_gdf[subregion]
                    exclusion_mask = gpd.read_file(filepath)
                    assert feature_gdf.crs == exclusion_mask.crs

                    feature_gdf = function_wrapper(exclusion_mask, feature_gdf, get_grid_intersects, **{'col': 'exclusion_mask'})
                    feature_gdf['exclusion_mask'] = feature_gdf['exclusion_mask'].apply(lambda x: 1 if x > 0 else 0)
                else:
                    self.logger.warning(f"No exclusion mask file with name {filepath}.")
                    feature_gdf["exclusion_mask"] = [0] * len(feature_gdf)
            except Exception as e:
                self.logger.warning(f"Error for exclusion mask for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty fields.")
                feature_gdf["exclusion_mask"] = [0] * len(feature_gdf)

            self._feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def process_all_subregions(self, recalculate_all=False, recalculate_features=False, feature_list=None):
        """Process all subregions and save to feature_stats directory."""

        for subregion in range(self.nsubregions):
            self.logger.info(f'\nProcessing subregion {subregion}\n')
            self.get_all_features(subregion, recalculate_all=recalculate_all,
                                  recalculate_features=recalculate_features,
                                  feature_list=feature_list)
            self.logger.info(f"Finished processing Storm {self.storm.capitalize()} in "\
                         f"{self.region.capitalize()}, subregion {subregion}.")


    def get_all_features(self, subregion, recalculate_all=False, recalculate_features=False, feature_list=None):
        """Get all features (or a subset given by feature_list) for the subregion."""
        if feature_list is None:
            feature_list = default_features

        # get a GeoDataFrame
        self.get_gdf(subregion, recalculate=recalculate_all)
        for feature in feature_list:
            print(f"Getting {feature}")
            getattr(self, f"get_{feature}")(subregion, recalculate_features)



# Helper functions
def function_wrapper(gdf1, gdf2, function, warning=RuntimeWarning, *args, **kwargs):
    """Wrapper to suppress RuntimeWarning for any function."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=warning)
        output = function(gdf1, gdf2, *args, **kwargs)
    return output


def inline_function_wrapper(gdf1, gdf2, function:str, warning=RuntimeWarning, *args, **kwargs):
    """Wrapper to suppress RuntimeWarning for any class method."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=warning)
        output = getattr(gdf1, function)(gdf2, *args, **kwargs)
    return output



def spatial_mean(feat, aoi_ee, gridsize):
    """Function to use with unmasking."""
    spatial_mean = feat.reduceRegions(aoi_ee,
                                      ee.Reducer.mean(),
                                      crs="EPSG:4326",
                                      scale=gridsize)
    spatial_mean = spatial_mean.getInfo()['features'][0]['properties']['mean']
    feat = feat.unmask(spatial_mean)

    return feat, spatial_mean


def get_feat(collection, start, end, aoi_ee, gridsize, unmask=0):
    """Get the feature from GEE and unmask using selected method.
    
    Parameters:
    -----------
    unmask : number or callable (default=0)
        If a number is supplied NaN values are replaced with this number. If a callable
        this function is used to interpolate the misisng values. See spatial_mean function
        for an example.
    """
    feat = ee.Image(collection
                    .filterBounds(aoi_ee)
                    .filterDate(start, end)
                    .sort('system:time_start', False)
                    .mean()
                    .clip(aoi_ee))

    if isinstance(unmask, Number):
        feat = feat.unmask(unmask)
    elif callable(unmask):
        feat, val = unmask(feat, aoi_ee, gridsize)
    return feat


def get_anomaly(collection, start, end, baseyear, aoi_ee, gridsize, unmask):
    """Get anomaly for any collection.

    Parameters:
    -----------
    collection : ee.ImageCollection
        Must have single band selected.
    start : ee.Date
    end : ee.Date
    baseyear : int


    >> feat, anomaly = get_anomaly(collection, start, end, baseyear, aoi_ee, gridsize)
    """
    years = ee.List.sequence(ee.Number(baseyear), start.get('year'))


    def window_average(year):
        yeardiff = ee.Number.parse(start.get('year').format()).getInfo() - baseyear
        window_start = start.advance(-yeardiff, 'years')
        window_end = end.advance(-yeardiff, 'years')

        mean = ee.Image(collection
                          .filterBounds(aoi_ee)
                          .filterDate(window_start, window_end)
                          .mean()
                          .clip(aoi_ee))
        return mean.set('year', year)

    # get temporal reference value
    temporal_means = ee.ImageCollection.fromImages(years.map(window_average))
    temporal_mean = ee.Image(temporal_means.mean())

    spatiotemporal_mean = temporal_mean.reduceRegions(aoi_ee,
                                                  ee.Reducer.mean(),
                                                  crs="EPSG:4326",
                                                  scale=gridsize)
    spatiotemporal_mean =  spatiotemporal_mean.getInfo()['features'][0]['properties']['mean']
    temporal_mean = temporal_mean.unmask(spatiotemporal_mean)


    # get image for present time window
    feat = get_feat(collection, start, end, aoi_ee, gridsize, unmask=unmask)
    anomaly = feat.subtract(temporal_mean)

    return feat, anomaly
