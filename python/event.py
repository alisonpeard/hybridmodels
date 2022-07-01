"""
Class for creating storm events and all their features.
"""


from os.path import join, exists
from ast import literal_eval
import logging

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, shape
from shapely.ops import nearest_points

import ee
import geemap

from data_utils import *
from model_utils import *

# ignore pd.Index future warnings for now
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Event:

    def __init__(self, storm, region, nsubregions, wd, bd, gridsize=500, stagger=0):
        """Set up instance of the storm Event."""

        # set up logging
        logger = logging.getLogger(f"data_collection.{storm}")
        # log_file_path = join(bd, 'logfiles')
        # fh = logging.FileHandler(join(log_file_path, f"data_collection.{storm}.log"), 'w')
        self.logger = logger
        self.logger.info(f"Setting up Storm {storm.capitalize()} Event instance for "\
                f"{region.capitalize()} with {nsubregions} subregions.")

        # set up attributes
        self.storm = storm
        self.region = region
        self.gridsize = gridsize
        self.stagger = stagger
        self.nsubregions = nsubregions
        self.subregions = [x for x in range(nsubregions)]
        self.wd = wd  # working dir
        self.bd = bd  # base dir
        self.indir = join(self.wd, f"{storm}_{region}")
        self.startdate, self.enddate = [*pd.read_csv(join(self.wd, "event_dates.csv"),
                                   index_col="storm").loc[storm]]
        self.year = int(self.enddate[:4])

        self.aoi_pm = [None] * nsubregions
        self.aoi_lonlat = [None] * nsubregions
        self.grid_pm = [None] * nsubregions
        self.grid_lonlat = [None] * nsubregions
        self.feature_gdf = [None] * nsubregions
        self.aoi_ee = [None] * nsubregions
        self.location = [None] * nsubregions
        self.grid_ee = [None] * nsubregions
        self.connected_to_gee = -1

        # calculate all the grids
        grids = []
        for n in self.subregions:
            grids.append(self.get_grid(n))


    def process_all_subregions(self, feature_list=None):
        """Process all subregions and save to feature_stats directory."""

        for subregion in range(self.nsubregions):
            # check if subregion has been calculated
            storm_file = open(join(self.bd, "logfiles", 'storm_file.txt'), 'r')
            if f"{self.storm}_{self.region}_{subregion}" not in storm_file:
                self.logger.info(f'\nProcessing subregion {subregion}\n')
                self.get_all_features(subregion, recalculate=True, feature_list=feature_list)

                # save to output, feature_stats directory
                self.feature_gdf[subregion].to_file(join(self.wd, "feature_stats", f"{self.storm}_{self.region}_{subregion}_{self.stagger}.shp"))
                self.logger.info(f"Finished processing Storm {self.storm.capitalize()} in "\
                             f"{self.region.capitalize()}, subregion {subregion}.")
                storm_file.close()

                # append event and subregion to logfile
                storm_file = open(join(self.bd, "logfiles", 'storm_file.txt'), 'a')
                storm_file.write(f"{self.storm}_{self.region}_{subregion}\n")
                storm_file.close()


    def get_all_features(self, subregion, recalculate=False, feature_list=None):
        """Get all features for the subregion."""
        if feature_list is None:
            feature_list = ["flood", "elevation", "permwater", "pw_dists", "precipitation", "era5", "soilcarbon", "mangroves", "ndvi", "wind_fields"]

        # get a GeoDataFrame
        self.get_gdf(subregion, recalculate=recalculate)
        for feature in feature_list:
            getattr(self, f"get_{feature}")(subregion)

    def get_gdf(self, subregion, recalculate=False):
        """Load pre-existing gdf or create one if recalculating or doesn't exist."""

        grid_lonlat = self.grid_lonlat[subregion]

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
            file = join(self.indir, f'feature_stats_{subregion}_{self.stagger}.shp')
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

        file = join(self.indir, f'feature_stats_{subregion}_{self.stagger}.shp')
        feature_gdf.to_file(file)
        self.logger.info(f"Saved as {file}...\n")


    def get_grid(self, subregion):
        """Make grid from geoJSON file."""
        geoJSON = literal_eval(pd.read_csv(join(self.wd, "event_geojsons.csv"),
                                           index_col="region").loc[self.region][0])

        poly = [shape(x['geometry']) for x in geoJSON['features']]
        poly = gpd.GeoDataFrame({"geometry": poly[subregion]}, index=[0])
        poly = poly.set_crs("EPSG:4326").to_crs("EPSG:3857").geometry[0]

        # construct aoi polygon of standard 32x32 km size
        (x, y) = (poly.centroid.x + self.stagger, poly.centroid.y + self.stagger)
        dx = 16000
        dy = 16000
        poly = box(x-dx, y-dy, x+dx, y+dy)
        aoi_pm = gpd.GeoDataFrame({"geometry": poly}, index=[0], crs="EPSG:3857")
        aoi_lonlat = aoi_pm.to_crs("EPSG:4326")

        grid_pm = make_grid(*aoi_pm.total_bounds, length=self.gridsize, wide=self.gridsize)
        grid_lonlat = grid_pm.to_crs("EPSG:4326")

        self.aoi_pm[subregion] = aoi_pm
        self.aoi_lonlat[subregion] = aoi_lonlat
        self.grid_pm[subregion] = grid_pm
        self.grid_lonlat[subregion] = grid_lonlat


    def get_flood(self, subregion):
        """Calculate flood fractions per gridcell."""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)
        feature_gdf = self.feature_gdf[subregion]

        if "floodfrac" not in feature_gdf:
            # calculate flood fraction over the grid cells in Pseudo-Mercator
            self.logger.info("Calculating flood fractions...")
            self.get_floodfile(subregion)
            flood = self.flood
            aoi_lonlat  = self.aoi_lonlat[subregion]
            grid_pm = self.grid_pm[subregion]
            flood = gpd.clip(flood, aoi_lonlat)
            flood_pm = flood.to_crs("EPSG:3857")
            floodfrac_gdf = get_grid_intersects(flood_pm, grid_pm)

            # save the feature_stats shapefile with flood fraction
            feature_gdf["floodfrac"] = floodfrac_gdf["floodfrac"]
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_floodfile(self, subregion):
        """Load Copernicus EMS flood polygon."""

        indir = self.indir
        flood = gpd.read_file(join(self.indir, "flood.shp")).to_crs('EPSG:4326')

        if exists(join(indir, "hydrographyA.shp")):
            pwater = gpd.read_file(join(self.indir, "hydrographyA.shp"))
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


    def get_gebco(self, subregion):
        """Download GEBCO raster from GEE.

        GEBCO bathymetry data: 15 arcseconds (approx. 0.5km)
        """

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)
        if "gebco" not in self.feature_gdf[subregion]:
            self.logger.info("Calculating GEBCO bathymetry...")
            feature_gdf = self.feature_gdf[subregion]

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

            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_fabdem(self, subregion):
        """
        Get FABDEM DTM from Google Earth Engine.

        From awesome-gee-community-datasets.

        """
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "fabdem" not in self.feature_gdf[subregion]:
            self.logger.info("Calculating FABDEM DTM...")
            feature_gdf = self.feature_gdf[subregion]

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
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_elevation(self, subregion):
        """Calculate approx. elevation from GEBCO and FABDEM."""
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "elevation" not in self.feature_gdf[subregion]:
            self.get_gebco(subregion)
            self.get_fabdem(subregion)
            self.feature_gdf[subregion]['elevation'] = self.feature_gdf[subregion]["gebco"]\
            + self.feature_gdf[subregion]["fabdem"]
            self.save_gdf(subregion)


    def get_permwater(self, subregion):
        """Get JRC Permanent water from Google Earth Engine.

        JRC permanent water dataset: 30 arcseconds (approx. 1km). Needs a better way to impute missing ocean values.
        """
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "jrc_permwa" not in self.feature_gdf[subregion]:
            self.logger.info("Recalculating permanent water...")
            feature_gdf = self.feature_gdf[subregion]

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

            # save to gdf
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_pw_dists(self, subregion, thresh=60):
        """Calculate cell distances and slopes to nearest permanent water."""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "dist_pw" not in self.feature_gdf[subregion]:
            self.logger.info(f"Recalculating distances to water with occurrence threshold={thresh}...")
            feature_gdf = self.feature_gdf[subregion]

            if "dist_pw" not in self.feature_gdf[subregion]:
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
                # Return column
                return dist_list

            jrc_permwater = (ee.Image("JRC/GSW1_3/GlobalSurfaceWater")
                             .clip(aoi_ee)
                             .select("occurrence")
                             .unmask(0))

            water = jrc_permwater.gt(thresh)
            water = water.reduceToVectors(reducer=ee.Reducer.countEvery())

            water_gdf = geemap.ee_to_geopandas(water)
            water_gdf = water_gdf[water_gdf.label == 1]
            # water_gdf.plot(cmap="YlGnBu")

            dist_list = dist_to_water(feature_gdf, water_gdf)
            feature_gdf['dist_pw'] = dist_list
            feature_gdf['slope_pw'] = (feature_gdf['elevation'] / feature_gdf['dist_pw']).replace(-np.inf, 0.0)

            # save to gdf
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_precipitation(self, subregion):
        """CHIRPS Daily Precipitation: 0.05 degrees daily"""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "precip" not in self.feature_gdf[subregion]:
            self.logger.info("Calculating CHIRPS daily precipitation averages...")
            feature_gdf = self.feature_gdf[subregion]

            # set up Google Earth Engine for subregion
            if self.connected_to_gee != subregion: self.start_gee(subregion)
            aoi_ee = self.aoi_ee[subregion]
            grid_ee = self.grid_ee[subregion]

            # which features from CHIRPS Daily Aggregates to use
            features = ['precipitation']
            for feature in features:
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

            # save output
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)

    def get_era5(self, subregion):
        """ERA5 Daily MSLP and Surface pressure."""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "mslp" not in self.feature_gdf[subregion]:
            feature_gdf = self.feature_gdf[subregion]

            # set up Google Earth Engine for subregion
            if self.connected_to_gee != subregion: self.start_gee(subregion)
            aoi_ee = self.aoi_ee[subregion]
            grid_ee = self.grid_ee[subregion]

            # which features from ERA5 Daily Aggregates to use
            feature_names = ['mslp', 'sp']
            features = ['mean_sea_level_pressure', 'surface_pressure']
            for feature, feature_name in zip(features, feature_names):
                self.logger.info(f"Calculating ERA5 {feature_name}...")
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

            # save output
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_soilcarbon(self, subregion):
        """Get soil organic carbon from Google Earth Engine."""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "soilcarbon" not in self.feature_gdf[subregion]:
            self.logger.info("Calculating soil carbon...")
            feature_gdf = self.feature_gdf[subregion]

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

            # save output file
            soilcarbon_list = mean_soilcarbon.aggregate_array('mean').getInfo()
            feature_gdf["soilcarbon"] = soilcarbon_list
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_mangroves(self, subregion):
        """Mangrove forests from year 2000 (Giri, 2011)"""
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "mangrove" not in self.feature_gdf[subregion]:
            self.logger.info("Calculating mangrove cover...")
            feature_gdf = self.feature_gdf[subregion]

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
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_ndvi(self, subregion):
        """NDVI (reprojected and masked from mangroves)"""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if "ndvi" not in self.feature_gdf[subregion]:
            self.logger.info("Calculating NDVI...")
            feature_gdf = self.feature_gdf[subregion]

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
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)


    def get_wind_fields(self, subregion):
        """Get wind fields from IBTrACs data using Holland (1980) method."""
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion)

        if not any("wnd" in col for col in self.feature_gdf[subregion].columns):
            self.logger.info("Calculating wind speeds...")
            feature_gdf = self.feature_gdf[subregion]

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
            feature_gdf = get_wind_field(ibtracs_gdf, feature_gdf, units_df, wind_col, pressure_col, rmw_col)

            # save average wind field
            timemask = ["wnd" in col for col in feature_gdf.columns]
            timestamps = feature_gdf.columns[timemask]
            feature_gdf["wind_avg"] = feature_gdf[timestamps].mean(axis=1)
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)
