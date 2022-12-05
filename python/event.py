from os.path import join, exists
from os import remove
from ast import literal_eval
import logging

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, shape
from shapely.ops import nearest_points

# for deltares
import dask.distributed
import xarray as xr
import pystac_client  # issue on old Macbook
import rasterio
import rioxarray

import ee
import geemap  # need later

from data_utils import *
from model_utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

default_features = ["flood", "elevation", "permwater", "pw_dists", "precipitation",
                    "era5", "soilcarbon","soiltemp", "mangroves", "ndvi", "wind_fields", "aqueduct",
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
        self.wd = wd  # working dir
        self.bd = bd  # base dir
        self.indir = join(self.wd, f"{storm}_{region}")
        self.startdate, self.enddate = [*pd.read_csv(join(self.wd, "csvs", "event_dates.csv"),
                                   index_col="storm").loc[storm]]
        self.acquisition_time = pd.read_csv(join(self.wd, 'csvs', 'current_datasets.csv'),
                                            index_col=['event', 'region']).loc[(self.storm, self.region)]['acquisition_time']
        self.year = int(self.enddate[:4])

        self.aoi_pm = [None] * nsubregions
        self.aoi_lonlat = [None] * nsubregions
        self.grid_pm = [None] * nsubregions
        self.grid_lonlat = [None] * nsubregions
        self.feature_gdf = [None] * nsubregions
        self.aoi_ee = [None] * nsubregions
        self.location = [None] * nsubregions
        self.grid_ee = [None] * nsubregions
        self.flood = None
        self.connected_to_gee = -1


    def __repr__(self):
        string = f"\nStorm {self.storm.capitalize()} in {self.region.capitalize()}:\n"\
                "-------------------------------\n"

        for subregion, feature_gdf in enumerate(self.feature_gdf):
            if feature_gdf is not None:
                field_info = f"subregion {subregion} fields: {[*feature_gdf.columns]}\n"
                string += field_info
            else:
                string += f"subregion {subregion} not yet processed.\n"

        return string


    def process_all_subregions(self, recalculate_all=False, recalculate_features=False, feature_list=None):
        """Process all subregions and save to feature_stats directory."""

        for subregion in range(self.nsubregions):
            self.logger.info(f'\nProcessing subregion {subregion}\n')
            self.get_all_features(subregion, recalculate_all=recalculate_all,
                                  recalculate_features=recalculate_features,
                                  feature_list=feature_list)

            # save to output, feature_stats directory
            self.feature_gdf[subregion].to_file(join(self.wd, "feature_stats", f"{self.storm}_{self.region}_{subregion}.gpkg"),
                                                layer=f"{self.storm}_{self.region}_{subregion}", driver="GPKG")
            self.logger.info(f"Finished processing Storm {self.storm.capitalize()} in "\
                         f"{self.region.capitalize()}, subregion {subregion}.")



    def get_all_features(self, subregion, recalculate_all=False, recalculate_features=False, feature_list=None):
        """Get all features for the subregion."""
        if feature_list is None:
            feature_list = default_features

        # get a GeoDataFrame
        self.get_gdf(subregion, recalculate=recalculate_all)
        for feature in feature_list:
            getattr(self, f"get_{feature}")(subregion, recalculate_features)


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
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)
        else:
            file = join(self.indir, f'feature_stats_{subregion}.gpkg')
            self.logger.info(f"Looking for {file}")
            try:
                self.feature_gdf[subregion] = gpd.read_file(file)
                self.logger.info(f"Loaded existing shapefile {file}...")
            except:
                self.logger.info("No shapefile exists, creating new one...")
                feature_gdf = gpd.GeoDataFrame(grid_lonlat)
                feature_gdf["storm"] = [self.storm] * len(feature_gdf)
                feature_gdf["region"] = [self.region] * len(feature_gdf)
                feature_gdf["subregion"] = [subregion] * len(feature_gdf)

                feature_gdf = feature_gdf.set_crs("EPSG:4326")
                self.feature_gdf[subregion] = feature_gdf
                self.save_gdf(subregion)


    def save_gdf(self, subregion):
        """Save/update the GeoDataFrame as a shapefile."""
        feature_gdf = self.feature_gdf[subregion]

        # make sure only saves if correct size
        assert len(feature_gdf) == 64 * 64, f"Grid not of size 64x64"

        file = join(self.indir, f'feature_stats_{subregion}.gpkg')
        feature_gdf.to_file(file, driver="GPKG")
        self.logger.info(f"Saved as {file}...\n")


    def make_grids(self):
        """Calculate all the grids."""
        for n in self.subregions:
            self.get_grid(n)


    def get_grid(self, subregion):
        """Make grid from geoJSON file."""
#         geoJSON = literal_eval(pd.read_csv(join(self.wd, "csvs", 'event_geojsons.csv'),
#                                            index_col="region").loc[self.region][0])
        geoJSON = literal_eval(pd.read_csv(join(self.wd, "csvs", "current_datasets.csv"),  # 'event_geojsons.csv'
                                           index_col="region").loc[self.region]['subregion_geojsons'])  # [0]

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

        grid_pm = make_grid(*aoi_pm.total_bounds, length=self.gridsize, wide=self.gridsize)
        grid_lonlat = grid_pm.to_crs("EPSG:4326")

        grid_lonlat.to_file(join(self.indir, f'grid_lonlat_{subregion}.gpkg'))

        self.aoi_pm[subregion] = aoi_pm
        self.aoi_lonlat[subregion] = aoi_lonlat
        self.grid_pm[subregion] = grid_pm
        self.grid_lonlat[subregion] = grid_lonlat


    def get_flood(self, subregion, recalculate=False, cols=['det_method', 'obj_desc']):
        """Calculate flood fractions per gridcell."""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)
        if self.flood is None: self.get_floodfile()
        feature_gdf = self.feature_gdf[subregion]

        if "floodfrac" not in feature_gdf or recalculate:
            # calculate flood fraction over the grid cells in Pseudo-Mercator
            self.logger.info("Calculating flood fractions...")
            flood = self.flood
            aoi_lonlat  = self.aoi_lonlat[subregion]
            grid_pm = self.grid_pm[subregion]

            assert aoi_lonlat.crs == flood.crs,\
                    f"Flood file and subregion geometries have different crs {subregion} for {self.region.upper()}."

            flood = gpd.overlay(flood, aoi_lonlat, how="intersection")
            assert len(flood) > 0,\
                    f"Flood file does not intersect subregion {subregion} for {self.region.upper()}."

            flood_pm = flood.to_crs("EPSG:3857")
            floodfrac_gdf = get_grid_intersects(flood_pm, grid_pm)
            feature_gdf["floodfrac"] = floodfrac_gdf["floodfrac"]

            try:
                # get extra data on flood detection
                grid_flooded = gpd.overlay(feature_gdf.reset_index(drop=False), flood, how='intersection', keep_geom_type=False)
                grid_flooded = grid_flooded[['index'] + cols].groupby('index').agg(pd.Series.mode)
                feature_gdf  = pd.merge(feature_gdf, grid_flooded, left_index=True, right_index=True, how='left').fillna("")
            except Exception as e:
                self.logger.warning(f"{self.storm}_{self.region}_{subregion}: error adding extra flood info:\n{e}\nCreating empty fields")
                for col in cols:
                    feature_gdf[col] =  [""] * len(feature_gdf)

            # save the feature_stats shapefile with flood fraction
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_floodfile(self):
        """Load Copernicus EMS flood polygon."""

        indir = self.indir
        flood = gpd.read_file(join(self.indir, "flood.gpkg")).to_crs('EPSG:4326')

        if exists(join(indir, "hydrographyA.gpkg")):
            pwater = gpd.read_file(join(self.indir, "hydrographyA.gpkg"))
            trueflood = gpd.overlay(flood, pwater, how="difference")
        else:
            trueflood = flood

        self.flood = trueflood


    def start_gee(self, subregion):
        """Initialize Google Earth Engine with service account."""

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

        self.connected_to_gee = subregion

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

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)
        if "gebco" not in self.feature_gdf[subregion] or recalculate:
            self.logger.info("Calculating GEBCO bathymetry...")
            feature_gdf = self.feature_gdf[subregion]

            try:
                if self.connected_to_gee != subregion:
                    self.start_gee(subregion)

                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]

                # get GEE image collection
                gebco = ee.Image(ee.ImageCollection("projects/sat-io/open-datasets/gebco/gebco_grid")
                                  .filterBounds(aoi_ee)
                                  .select("b1")
                                  .median()
                                  .clip(aoi_ee)
                                  .unmask(999))

                ocean = gebco.lte(0)

                # Add reducer output to features
                mean_gebco = gebco.mask(ocean).unmask(0).reduceRegions(collection=grid_ee,
                                                         reducer=ee.Reducer.mean(), scale=self.gridsize)
                gebco_list = mean_gebco.aggregate_array('mean').getInfo()
                feature_gdf["gebco"] = gebco_list

            except Exception as e:
                self.logger.warning(f"Error for gebco for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty fields.")
                feature_gdf["gebco"] = [""] * len(feature_gdf)

            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_aqueduct(self, subregion, recalculate=False):
        """Download Aqueduct flood data.

        Aqueduct coastal flood hazard data.
        """
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        # get data from GEE
        if f"aqueduct" not in self.feature_gdf[subregion] or recalculate:
            feature_gdf = self.feature_gdf[subregion]

            try:
                if self.connected_to_gee != subregion:
                    self.start_gee(subregion)

                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]

                # cycle through all rps
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

                        # process floodfracs here and get correlations
                        flood_present = flood_depths.gt(0)

                        # Add reducer output to features
                        # NOTE: mean() works here because frac is sum(pixels) / len(pixels), same as mean()
                        floodfrac = flood_present.reduceRegions(collection=grid_ee, reducer=ee.Reducer.mean(), scale=self.gridsize)
                        aqueduct_list = floodfrac.aggregate_array('mean').getInfo()
                        feature_gdf[f"aqueduct_{rp}"] = aqueduct_list
                        feature_gdf[f"aqueduct_{rp}"] = feature_gdf[f"aqueduct_{rp}"].replace('', 0.0)
                        corr = feature_gdf['floodfrac'].corr(feature_gdf[f"aqueduct_{rp}"])
                        assert not np.isnan(corr), f"Got a NaN correlation for RP {rp}."
                        corrs[rp] = corr

                    except Exception as e:
                        self.logger.error(f"{self.storm}, {self.region}, {subregion}:\n{e}")
                        # feature_gdf[f"aqueduct_{rp}"] = [""] * len(feature_gdf)

                # choose rp most correlated to floodfrac
                best_rp = max(corrs, key=corrs.get)
                feature_gdf['aqueduct'] = feature_gdf[f"aqueduct_{best_rp}"]
                self.logger.info(f"Chose return period {best_rp} for {self.storm}, {self.region}, {subregion}\n"
                                 f"with correlation {max(corrs.values()):.4f}")

            except Exception as e:
                self.logger.error(f"Error for Aqueduct data for {self.storm}, {self.region}, {subregion}"\
                                    f"\n{e}\nCreating empty fields.")
                feature_gdf["aqueduct"] = [""] * len(feature_gdf)

            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_deltares(self, subregion, recalculate=False):
        """Download Deltares flood data.

        Deltares coastal flood hazard data.
        """
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        # get data from Deltares
        if "deltares" not in self.feature_gdf[subregion] or recalculate:
            feature_gdf = self.feature_gdf[subregion]

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
                values = [i['properties']['val'] for i in feature_list]
                flood_gdf = gpd.GeoDataFrame({'geometry':geom}).set_crs(4326)

                feature_gdf = get_grid_intersects(flood_gdf, feature_gdf, col='deltares')

            except Exception as e:
                self.logger.error(f"Error for Deltares data for {self.storm}, {self.region}, {subregion}:"\
                                    f"\n{e}\n\nCreating empty fields.")
                feature_gdf["deltares"] = [""] * len(feature_gdf)

            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_fabdem(self, subregion, recalculate=False):
        """
        Get FABDEM DTM from Google Earth Engine.

        From awesome-gee-community-datasets.

        """
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "fabdem" not in self.feature_gdf[subregion] or recalculate:
            self.logger.info("Calculating FABDEM DTM...")
            feature_gdf = self.feature_gdf[subregion]

            try:
                if self.connected_to_gee != subregion:
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
                                                         reducer=ee.Reducer.mean(), scale=self.gridsize)

                elev_list = mean_elev.aggregate_array('mean').getInfo()
                feature_gdf["fabdem"] = elev_list

            except Exception as e:
                self.logger.warning(f"Error for fabdem for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty fields.")
                feature_gdf["fabdem"] = [""] * len(feature_gdf)

            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_elevation(self, subregion, recalculate=False):
        """Calculate approx. elevation from GEBCO and FABDEM."""
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "elevation" not in self.feature_gdf[subregion] or recalculate:
            self.get_gebco(subregion)
            self.get_fabdem(subregion)
            self.feature_gdf[subregion]['elevation'] = self.feature_gdf[subregion]["gebco"]\
            + self.feature_gdf[subregion]["fabdem"]
            self.save_gdf(subregion)


    def get_permwater(self, subregion, recalculate):
        """Get JRC Permanent water from Google Earth Engine.

        JRC permanent water dataset: 30 arcseconds (approx. 1km). Needs a better way to impute missing ocean values.
        """
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "jrc_permwa" not in self.feature_gdf[subregion] or recalculate:
            self.logger.info("Recalculating permanent water...")
            feature_gdf = self.feature_gdf[subregion]

            try:
                if self.connected_to_gee != subregion:
                    self.start_gee(subregion)

                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]

                jrc_permwater = (ee.Image("JRC/GSW1_3/GlobalSurfaceWater")
                                 .clip(aoi_ee)
                                 .select("occurrence")
                                 .unmask(0))

                # Add reducer output to the Features in the collection.
                mean_jrc_permwater = jrc_permwater.reduceRegions(collection=grid_ee,
                                                         reducer=ee.Reducer.mean(), scale=self.gridsize)
                jrc_permwater_list = mean_jrc_permwater.aggregate_array('mean').getInfo()


                jrc_permwater_list2 = []
                feature_gdf["jrc_permwa"] = jrc_permwater_list

                # slightly hack-y way of filling-in ocean
                max_pw = feature_gdf.jrc_permwa.describe()["max"]
                pw_ix = feature_gdf[(feature_gdf['jrc_permwa']<=90) & (feature_gdf["gebco"]<-90)].index
                feature_gdf.loc[pw_ix, "jrc_permwa"] = 100

            except Exception as e:
                self.logger.warning(f"Error for jrc_permwa for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty fields.")
                feature_gdf["jrc_permwa"] = [""] * len(feature_gdf)

            # save to gdf
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_pw_dists(self, subregion, recalculate=False, thresh=pwater_thresh):
        """Calculate cell distances and slopes to nearest permanent water."""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "dist_pw" not in self.feature_gdf[subregion] or recalculate:
            self.logger.info(f"Recalculating distances to water with occurrence threshold={thresh}...")
            feature_gdf = self.feature_gdf[subregion]

            try:
                if "elevation" not in self.feature_gdf[subregion]:
                    self.get_elevation(subregion)

                if self.connected_to_gee != subregion:
                    self.start_gee(subregion)

                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]

                def dist_to_water(gdf, water, geom_col='geometry'):
                    gdf = gdf.to_crs('EPSG:3857')
                    water = water.set_crs('EPSG:4326').to_crs('EPSG:3857')
                    water_union = water.unary_union
                    dist_list =[]
                    for index, row in gdf.iterrows():
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

                water_gdf = geemap.ee_to_geopandas(water)
                water_gdf = water_gdf[water_gdf.label == 1]

                dist_list = dist_to_water(feature_gdf, water_gdf)
                feature_gdf['dist_pw'] = dist_list
                feature_gdf['slope_pw'] = (feature_gdf['elevation'] / feature_gdf['dist_pw']).replace(-np.inf, 0.0)

            except Exception as e:
                self.logger.warning(f"Error for dist_pw and slope_pw for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty fields.")
                feature_gdf["dist_pw"] = [""] * len(feature_gdf)
                feature_gdf["slope_pw"] = [""] * len(feature_gdf)

            # save to gdf
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_precipitation(self, subregion, recalculate=False):
        """CHIRPS Daily Precipitation: 0.05 degrees daily"""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "precip" not in self.feature_gdf[subregion] or recalculate:
            feature_gdf = self.feature_gdf[subregion]
            features = ['precipitation']

            # set up Google Earth Engine for subregion
            if self.connected_to_gee != subregion: self.start_gee(subregion)
            aoi_ee = self.aoi_ee[subregion]
            grid_ee = self.grid_ee[subregion]

            for feature in features:
                self.logger.info(f"Calculating CHIRPS daily {feature} averages...")
                try:
                    feat = ee.Image(ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                                      .select(feature)
                                      .filterBounds(aoi_ee)
                                      .filterDate(self.startdate, self.enddate)
                                      .mean()
                                      .clip(aoi_ee))

                    # unmask using the spatial average
                    spatial_mean = feat.reduceRegions(aoi_ee, ee.Reducer.mean(), scale=self.gridsize)
                    spatial_mean = spatial_mean.getInfo()['features'][0]['properties']['mean']
                    feat = feat.unmask(spatial_mean)

                    # Add reducer output to the Features in the collection.
                    mean_feat = feat.reduceRegions(collection=grid_ee,
                                                             reducer=ee.Reducer.mean(), scale=self.gridsize)
                    feat_list = mean_feat.aggregate_array('mean').getInfo()
                    feature_gdf[feature[:6]] = feat_list

                except Exception as e:
                    self.logger.warning(f"Error for {feature} for {self.storm}, {self.region}, {subregion}:"\
                                        f"{e}\nCreating empty field.")
                    feature_gdf[feature] = [""] * len(feature_gdf)

            # save output
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)



    def get_lulc(self, subregion, recalculate=False):
        """Land use and land cover."""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "lulc" not in self.feature_gdf[subregion] or recalculate:
            feature_gdf = self.feature_gdf[subregion]

            # set up Google Earth Engine for subregion
            if self.connected_to_gee != subregion: self.start_gee(subregion)
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
                feat_list = mode_feat.aggregate_array('mode').getInfo()
                feature_gdf['lulc'] = feat_list
            except Exception as e:
                self.logger.warning(f"Error for lulc for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty field.")
                feature_gdf["lulc"] = [""] * len(feature_gdf)

            # save output
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)



    def get_era5(self, subregion, recalculate=False):
        """ERA5 Daily MSLP, surface pressure and (x, y) U10 wind components.

        Dataset available: 1979-01-02T00:00:00Z – 2020-07-09T00:00:00
        """

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "mslp" not in self.feature_gdf[subregion] or recalculate:
            feature_gdf = self.feature_gdf[subregion]

            # set up Google Earth Engine for subregion
            if self.connected_to_gee != subregion: self.start_gee(subregion)
            aoi_ee = self.aoi_ee[subregion]
            grid_ee = self.grid_ee[subregion]

            # which features from ERA5 Daily Aggregates to use
            feature_names = ['mslp', 'sp', 'u10_u', 'u10_v']
            features = ['mean_sea_level_pressure', 'surface_pressure', 'u_component_of_wind_10m', 'v_component_of_wind_10m']

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

                    feat_list = mean_feat.aggregate_array('mean').getInfo()
                    feature_gdf[feature_name] = feat_list

                except Exception as e:
                    self.logger.warning(f"Error for {feature_name} for {self.storm}, {self.region}, {subregion}:"\
                                        f"{e}\nCreating empty field.")
                    feature_gdf[feature_name] = [""] * len(feature_gdf)


            # save output
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_soiltemp(self, subregion, recalculate=False):
        """ERA5 Land Monthly Averaged ECMWF Climate Reanalysis.

        Dataset availability: 1981-01-01T00:00:00Z – 2022-08-01T00:00:00
        """

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "soil_temperature_level_1" not in self.feature_gdf[subregion] or recalculate:
            feature_gdf = self.feature_gdf[subregion]

            # set up Google Earth Engine for subregion
            if self.connected_to_gee != subregion: self.start_gee(subregion)
            aoi_ee = self.aoi_ee[subregion]
            grid_ee = self.grid_ee[subregion]

            # which features from ERA5 Daily Aggregates to use
            feature_names = ['soiltemp1', 'soiltemp2']
            features = ['soil_temperature_level_1', 'soil_temperature_level_2']

            for feature, feature_name in zip(features, feature_names):
                self.logger.info(f"Calculating ERA5 {feature_name}...")
                try:
                    # include one month in advance to make sure covered
                    start = ee.Date(self.startdate).advance(-1, 'month')
                    end = ee.Date(self.enddate)
                    feat = ee.Image(ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY")
                                      .select(feature)
                                      .filterBounds(aoi_ee)
                                      .filterDate(start, end)
                                      .sort('system:time_start', False).first()
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
                    feat_list = mean_feat.aggregate_array('mean').getInfo()
                    feature_gdf[feature_name] = feat_list

                except Exception as e:
                    import pdb; pdb.set_trace()
                    self.logger.warning(f"Error for {feature_name} for {self.storm}, {self.region}, {subregion}:"\
                                        f"{e}\nCreating empty field.")
                    feature_gdf[feature_name] = [""] * len(feature_gdf)


            # save output
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_soilcarbon(self, subregion, recalculate=False):
        """Get soil organic carbon from Google Earth Engine."""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "soilcarbon" not in self.feature_gdf[subregion] or recalculate:
            self.logger.info("Calculating soil carbon...")
            feature_gdf = self.feature_gdf[subregion]
            try:
                # set up Google Earth Engine for subregion
                if self.connected_to_gee != subregion: self.start_gee(subregion)
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

                soilcarbon_list = mean_soilcarbon.aggregate_array('mean').getInfo()
                feature_gdf["soilcarbon"] = soilcarbon_list
            except Exception as e:
                self.logger.warning(f"Error for soilcarbon for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty field.")
                feature_gdf["soilcarbon"] = [""] * len(feature_gdf)

            # save output file
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_mangroves(self, subregion, recalculate=False):
        """Mangrove forests from year 2000 (Giri, 2011)"""
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "mangrove" not in self.feature_gdf[subregion] or recalculate:
            self.logger.info("Calculating mangrove cover...")
            feature_gdf = self.feature_gdf[subregion]

            try:
                # set up Google Earth Engine for subregion
                if self.connected_to_gee != subregion: self.start_gee(subregion)
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
                feature_gdf["mangrove"] = [""] * len(feature_gdf)

            # save output
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_ndvi(self, subregion, recalculate=False):
        """NDVI (reprojected and masked from mangroves)"""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "ndvi" not in self.feature_gdf[subregion] or recalculate:
            self.logger.info("Calculating NDVI...")
            feature_gdf = self.feature_gdf[subregion]

            try:
                # set up Google Earth Engine for subregion
                if self.connected_to_gee != subregion: self.start_gee(subregion)
                aoi_ee = self.aoi_ee[subregion]
                grid_ee = self.grid_ee[subregion]

                # NDVI
                ndvi = ee.Image(ee.ImageCollection("MODIS/006/MOD13Q1")
                                .filterBounds(aoi_ee)
                                .filterDate(ee.Date(self.startdate).advance(-2, 'month'), ee.Date(self.enddate))
                                .mean()
                                .clip(aoi_ee))

                ndvi = ndvi.select('NDVI')

                # # mask out mangroves -- TODO
                # # reload mangroves for masking
                # mangrove = ee.Image(ee.ImageCollection("LANDSAT/MANGROVE_FORESTS")
                #                                .filterBounds(aoi_ee)
                #                                .first()
                #                                .clip(aoi_ee))
                # self.logger.info("Masking out mangrove presence...")
                # mangrove = mangrove.unmask(0)
                # mangrove_mask = mangrove.eq(0)
                # ndvi_masked = ndvi.updateMask(mangrove_mask)
                # ndvi = ndvi_masked.unmask(0)
                ndvi = ndvi.unmask(0)  # remove this line if using mangroves

                # calculate mean over feature collection
                mean_ndvi = ndvi.reduceRegions(collection=grid_ee,
                                               reducer=ee.Reducer.mean(), scale=self.gridsize)

                ndvi_list = mean_ndvi.aggregate_array('mean').getInfo()
                feature_gdf["ndvi"] = ndvi_list

            except Exception as e:
                self.logger.warning(f"Error for ndvi for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty field.")
                feature_gdf["ndvi"] = [""] * len(feature_gdf)

            # save output
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_wind_fields(self, subregion, recalculate=False):
        """Get wind fields from IBTrACs data using Holland (1980) method.

        Adds wind field columns where wind is nonzero to feature GeoDataFrame and
        average over all these wind field columns (i.e., over storm duration)
        """
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if not any("wnd" in col for col in self.feature_gdf[subregion].columns) or recalculate:
            self.logger.info("Calculating wind speeds...")
            feature_gdf = self.feature_gdf[subregion]

            try:
                # load and process IBTrACs data
                if self.year > 2017:
                    ibtracs_gdf = gpd.read_file(join(self.wd, "ibtracs_last3years.csv"))[1:].replace(" ", np.nan)
                    ibtracs_gdf = ibtracs_gdf[ibtracs_gdf.NAME == self.storm.upper()]
                    units_df = pd.read_csv(join(self.wd, "ibtracs_last3years.csv"), dtype=str, header=0)[0:1]
                else:
                    ibtracs_gdf = gpd.read_file(join(self.wd, "ibtracs_since1980.csv"))[1:].replace(" ", np.nan)
                    ibtracs_gdf = ibtracs_gdf[ibtracs_gdf.NAME == self.storm.upper()]
                    units_df = pd.read_csv(join(self.wd, "ibtracs_since1980.csv"), dtype=str, header=0)[0:1]

                # process IBTrACS data
                ibtracs_gdf, wind_col, pressure_col, rmw_col = process_ibtracs(ibtracs_gdf, self.storm)
                feature_gdf = get_wind_field(ibtracs_gdf, feature_gdf, units_df, wind_col, pressure_col, rmw_col, self.acquisition_time)

                # save average wind field
                timemask = ["wnd" in col for col in feature_gdf.columns]
                timestamps = feature_gdf.columns[timemask]
                feature_gdf["wind_avg"] = feature_gdf[timestamps].mean(axis=1)

            except Exception as e:
                self.logger.warning(f"Error for wind fields for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty field.")
                feature_gdf["wind_avg"] = [""] * len(feature_gdf)

            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_exclusion_mask(self, subregion, recalculate=False):
        """Add exclusion mask if exclusion_mask.gpkg present in event directory.

        Exclusion mask from Copernicus GFM.
        """

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)
        if "exclusion_mask" not in self.feature_gdf[subregion] or recalculate:
            self.logger.info("Calculating exclusion mask...")
            feature_gdf = self.feature_gdf[subregion]

            try:
                feature_gdf['exclusion_mask'] = [""] * len(feature_gdf)
                filepath = join(self.wd, f"{self.storm}_{self.region}", "exclusion_mask.gpkg")
                if exists(filepath):
                    feature_gdf = self.feature_gdf[subregion]
                    exclusion_mask = gpd.read_file(filepath)
                    assert feature_gdf.crs == exclusion_mask.crs

                    feature_gdf = data_utils.get_grid_intersects(exclusion_mask, feature_gdf, col='exclusion_mask')
                    feature_gdf['exclusion_mask'] = feature_gdf['exclusion_mask'].apply(lambda x: 1 if x > 0 else 0)
                else:
                    self.logger.warning(f"No exclusion mask file with name {filepath}.")
                    feature_gdf["exclusion_mask"] = [""] * len(feature_gdf)
            except Exception as e:
                self.logger.warning(f"Error for exclusion mask for {self.storm}, {self.region}, {subregion}:"\
                                    f"{e}\nCreating empty fields.")
                feature_gdf["exclusion_mask"] = [""] * len(feature_gdf)

            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    # def add_intermediates_to_event(self, subregion, features_to_process: dict, thresh=pwater_thresh):
    #     """Helper function to for add_intermediate_features between cell and nearest water cell."""
    #
    #     if self.feature_gdf[subregion] is None: self.get_gdf(subregion)
    #     feature_gdf = self.feature_gdf[subregion]
    #     feature_gdf = add_intermediate_features(feature_gdf, features_to_process, thresh=thresh)
    #     self.feature_gdf[subregion] = feature_gdf
