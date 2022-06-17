from os.path import join, exists
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, shape
from ast import literal_eval

from data_utils import *
from model_utils import *

import ee
import geemap
from shapely.ops import nearest_points

# ignore pd.Index future warnings for now
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Event:
    
    def __init__(self, storm, region, nsubregions, gridsize=500, stagger=0):
        print(f"Setting up storm {storm.capitalize()} Event instance for {region.capitalize()}.")
        self.storm = storm
        self.region = region
        self.gridsize = gridsize
        self.stagger = stagger
        self.nsubregions = nsubregions
        self.subregions = [x for x in range(nsubregions)]
        self.wd = join("..", "data")
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
        
        grids = []
        for n in self.subregions:
            grids.append(self.make_grid(n))
        
        
    def make_grid(self, subregion):
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
        print(f"subregion {subregion} area: {aoi_pm.area[0] / (1000 * 1000)} sqkm")
        grid_pm = make_grid(*aoi_pm.total_bounds, length=self.gridsize, wide=self.gridsize)
        grid_lonlat = grid_pm.to_crs("EPSG:4326")
        
        self.aoi_pm[subregion] = aoi_pm
        self.aoi_lonlat[subregion] = aoi_lonlat
        self.grid_pm[subregion] = grid_pm
        self.grid_lonlat[subregion] = grid_lonlat
        
    def get_floodfile(self, subregion, viz=True):
        indir = self.indir
        flood = gpd.read_file(join(self.indir, "flood.shp")).to_crs('EPSG:4326')

        if exists(join(indir, "hydrographyA.shp")):
            pwater = gpd.read_file(join(self.indir, "hydrographyA.shp"))
            trueflood = gpd.overlay(flood, pwater, how="difference")
        else:
            trueflood = flood
        self.flood = trueflood
        
        if viz:
        # plot flood data
            fig, ax = plt.subplots(1, figsize=(10, 5))
            trueflood.plot(ax=ax, color="#21c4c5")
            if exists(join(indir, "hydrographyA.shp")):
                pwater.plot(ax=ax, color="#01306e")
            self.grid_lonlat[subregion].boundary.plot(color="red", ax=ax, linewidth=0.02)
            self.aoi_lonlat[subregion].boundary.plot(color="red", ax=ax)
            ax.set_title(f"Flood in {self.region}_{subregion}");
            
    def get_gdf(self, subregion, recalculate=False):
        """Load pre-existing gdf or create one if recalculating or doesn't exist."""
        grid_lonlat = self.grid_lonlat[subregion]
        if not recalculate:
            file = join(self.indir, f'feature_stats_{subregion}_{self.stagger}.shp')
            print(f"Looking for {file}")
            try:
                self.feature_gdf[subregion] = gpd.read_file(file)
                print(f"Loaded existing shapefile {file}...\n")
            except:
                print("No shapefile exists, creating new one...\n")
                feature_gdf = gpd.GeoDataFrame(grid_lonlat)
                feature_gdf["storm"] = [self.storm] * len(feature_gdf)
                feature_gdf["region"] = [self.region] * len(feature_gdf)
                feature_gdf["subregion"] = [subregion] * len(feature_gdf)
                feature_gdf = feature_gdf.set_crs("EPSG:4326")
                self.feature_gdf[subregion] = feature_gdf
                self.save_gdf(subregion)
        else:
            print("Recalculating shapefile...\n")
            feature_gdf = gpd.GeoDataFrame(grid_lonlat)
            feature_gdf["storm"] = [self.storm] * len(feature_gdf)
            feature_gdf["region"] = [self.region] * len(feature_gdf)
            feature_gdf["subregion"] = [subregion] * len(feature_gdf)
            feature_gdf = feature_gdf.set_crs("EPSG:4326")
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)
            
    def save_gdf(self, subregion):
        """Save/update the GeoDataFrame as a shapefile."""
        file = join(self.indir, f'feature_stats_{subregion}_{self.stagger}.shp')
        self.feature_gdf[subregion].to_file(file)
        print(f"Saved as {file}...")
            
    def get_flood(self, subregion, recalculate=False, viz=True):
        self.get_gdf(subregion, recalculate=recalculate)
        self.get_floodfile(subregion)
        
        feature_gdf = self.feature_gdf[subregion]

        if "floodfrac" not in feature_gdf or recalculate:
            # calculate flood fraction over the grid cells in Pseudo-Mercator
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
        
        if viz:
            # plot output
            fig, ax = plt.subplots(1, 1)
            self.feature_gdf[subregion].plot(column="floodfrac", cmap="YlGnBu", ax=ax, legend=True);
            
    def start_gee(self, subregion):
        # workaround to solve conflict with collections
        print("Connecting to Google Earth Engine...\n")
        import collections
        collections.Callable = collections.abc.Callable

        try:
            ee.Initialize()
        except:
            ee.Authenticate()
            ee.Initialize()
            
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
        print(f"Grid size: {grid_ee.size().getInfo()}")
        
        self.aoi_ee[subregion] = aoi_ee
        self.location[subregion] = location
        self.grid_ee[subregion] = grid_ee
        
    def get_gebco(self, subregion, recalculate=False, viz=True):
        """Download GEBCO raster from GEE.
        
        GEBCO bathymetry data: 15 arcseconds (approx. 0.5km)
        """
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion, recalculate=recalculate)
        if "gebco" not in self.feature_gdf[subregion] or recalculate:
            print("Calculating GEBCO bathymetry...")
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
    
        if viz:
        # plot output
            fig, ax = plt.subplots(1, 1)
            feature_gdf.plot(column='gebco', cmap="YlGnBu_r", ax=ax, legend=True);
            
            
    def get_fabdem(self, subregion, recalculate=False, viz=True):
        """
        Get FABDEM DTM from Google Earth Engine.

        From awesome-gee-community-datasets.

        """
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion, recalculate=recalculate)
        
        if "fabdem" not in self.feature_gdf[subregion] or recalculate:
            print("Calculating FABDEM DTM...")
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

        if viz:
            # plot output
            fig, ax = plt.subplots(1, 1)
            feature_gdf.plot(column='fabdem', cmap="terrain", ax=ax, legend=True, vmin=-10)
    
    def get_elevation(self, subregion, recalculate=False, viz=True):
        """Calculate approx. elevation from GEBCO and FABDEM."""
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion, recalculate=recalculate)
            
        if "elevation" not in self.feature_gdf[subregion] or recalculate:
            self.get_gebco(subregion, viz=False)
            self.get_fabdem(subregion, viz=False)
            self.feature_gdf[subregion]['elevation'] = self.feature_gdf[subregion]["gebco"]\
            + self.feature_gdf[subregion]["fabdem"]
            self.save_gdf(subregion)
            
        if viz:
            fig, ax = plt.subplots(1, 1)
            self.feature_gdf[subregion].plot(column='elevation', cmap="terrain", ax=ax, legend=True);
            
    def get_permwater(self, subregion, recalculate=False, viz=True):
        """Get JRC Permanent water from Google Earth Engine.
        
        JRC permanent water dataset: 30 arcseconds (approx. 1km). Needs a better way to impute missing ocean values.
        """
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion, recalculate=recalculate)
            
        print("Recalculating permanent water...")
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
        if viz:
            # plot results
            fig, ax = plt.subplots(1, 1)
            self.feature_gdf[subregion].plot(column='jrc_permwa', ax=ax, cmap="YlGnBu", legend=True);
            
    def get_pw_dists(self, subregion, thresh=60, recalculate=False, viz=True):
        """Calculate cell distances and slopes to nearest permanent water."""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion, recalculate=recalculate)

        if "dist_pw" not in self.feature_gdf[subregion] or recalculate:
            print("Recalculating distances to water...")
            feature_gdf = self.feature_gdf[subregion]

            if "dist_pw" not in self.feature_gdf[subregion]:
                self.get_elevation(subregion, viz=False)

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

        if viz:
            # plot results
            fig, ax = plt.subplots(1, 1)
            self.feature_gdf[subregion].plot('dist_pw', cmap="YlGnBu_r", legend=True, ax=ax)
            ax.set_title(f"Distance to permanent water (m)\n(>{thresh} occurence)");

    def get_precipitation(self, subregion, recalculate=False, viz=True):
        """CHIRPS Daily Precipitation: 0.05 degrees daily"""

        if self.feature_gdf[subregion] is None: 


            _gdf(subregion)

        if "precip" not in self.feature_gdf[subregion] or recalculate:
            print("Calculating precipitation...")

            feature_gdf = self.feature_gdf[subregion]

            if self.connected_to_gee != subregion:
                self.start_gee(subregion)

            aoi_ee = self.aoi_ee[subregion]
            grid_ee = self.grid_ee[subregion]

            precip = ee.Image(ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                              .select('precipitation')
                              .filterBounds(aoi_ee)
                              .filterDate(self.startdate, self.enddate)
                              .mean()
                              .clip(aoi_ee))

            # unmask using the spatial average
            spatial_mean = precip.reduceRegions(aoi_ee, ee.Reducer.mean(), scale=self.gridsize)  # self.gridsize defined by output scale, always in (m)
            spatial_mean = spatial_mean.getInfo()['features'][0]['properties']['mean']
            precip = precip.unmask(spatial_mean)

            # Add reducer output to the Features in the collection.
            mean_precip = precip.reduceRegions(collection=grid_ee,
                                                     reducer=ee.Reducer.mean(), scale=self.gridsize)
            precip_list = mean_precip.aggregate_array('mean').getInfo()

            # save output
            feature_gdf["precip"] = precip_list
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)

        if viz:
            # plot results
            fig, ax = plt.subplots(1, 1)
            self.feature_gdf[subregion].plot(column='precip', ax=ax, cmap="Blues", legend=True)
            ax.set_title(f"Mean precipitation {self.startdate}-{self.enddate}");


    def get_soilcarbon(self, subregion, recalculate=False, viz=True):
        """Get soil organic carbon from Google Earth Engine."""

        if self.feature_gdf[subregion] is None: self.get_gdf(subregion, recalculate=recalculate)

        if "soilcarbon" not in self.feature_gdf[subregion] or recalculate:
            print("Calculating soil carbon...")

            feature_gdf = self.feature_gdf[subregion]

            if self.connected_to_gee != subregion:
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

            # save output file
            soilcarbon_list = mean_soilcarbon.aggregate_array('mean').getInfo()
            feature_gdf["soilcarbon"] = soilcarbon_list
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)
        if viz:
            # plot output
            fig, ax = plt.subplots(1, 1)
            self.feature_gdf[subregion].plot(column='soilcarbon', ax=ax, cmap="YlOrBr", legend=True)
            ax.set_title("Soil organic carbon");


    def get_mangroves(self, subregion, recalculate=False, viz=True):     
        """Mangrove forests from year 2000 (Giri, 2011)"""
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion, recalculate=recalculate)

        if "mangrove" not in self.feature_gdf[subregion] or recalculate:
            print("Calculating mangrove cover...")

            feature_gdf = self.feature_gdf[subregion]

            if self.connected_to_gee != subregion:
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
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)

        if viz:
            # plot output
            fig, ax = plt.subplots(1, 1)
            self.feature_gdf[subregion].plot(column='mangrove', ax=ax, cmap="YlGn", legend=True)
            ax.set_title("Mangroves 2000 baseline (Giri, 2011)")

    def get_ndvi(self, subregion, recalculate=False, viz=True):
        """NDVI (reprojected and masked from mangroves)"""
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion, recalculate=recalculate)

        if "ndvi" not in self.feature_gdf[subregion] or recalculate:
            print("Calculating NDVI...")

            feature_gdf = self.feature_gdf[subregion]

            if self.connected_to_gee != subregion:
                self.start_gee(subregion)

            aoi_ee = self.aoi_ee[subregion]
            grid_ee = self.grid_ee[subregion]

            # reload mangroves
            mangrove = ee.Image(ee.ImageCollection("LANDSAT/MANGROVE_FORESTS")
                                           .filterBounds(aoi_ee)
                                           .first()
                                           .clip(aoi_ee))

            # NDVI
            ndvi = ee.Image(ee.ImageCollection("MODIS/006/MOD13Q1")
                            .filterBounds(aoi_ee)
                            .filterDate(ee.Date(self.startdate).advance(-2, 'month'), ee.Date(self.enddate))
                            .mean()
                            .clip(aoi_ee))

            ndvi = ndvi.select('NDVI')

            # mask out mangroves
            print("Masking out mangrove presence...")
            mangrove = mangrove.unmask(0)
            mangrove_mask = mangrove.eq(0)
            ndvi_masked = ndvi.updateMask(mangrove_mask)
            ndvi = ndvi_masked.unmask(0)

            # calculate mean over feature collection
            mean_ndvi = ndvi.reduceRegions(collection=grid_ee,
                                           reducer=ee.Reducer.mean(), scale=self.gridsize)

            ndvi_list = mean_ndvi.aggregate_array('mean').getInfo()
            feature_gdf["ndvi"] = ndvi_list
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)

        if viz:
            fig, ax = plt.subplots(1, 1)
            self.feature_gdf[subregion].plot(column='ndvi', ax=ax, cmap="YlGn", legend=True)
            ax.set_title("NDVI (mangroves masked out)")


    # takes ages...
    def get_wind_fields(self, subregion, recalculate=False, viz=False):
        """Get wind fields from IBTrACs data using Holland (1980) method."""
        if self.feature_gdf[subregion] is None: self.get_gdf(subregion, recalculate=recalculate)

        if not any("wnd" in col for col in self.feature_gdf[subregion].columns) or recalculate:
            print("Calculating wind speeds...")
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

            # set geometry for gdf
            for col in ["LAT", "LON"]:
                ibtracs_gdf[col] = pd.to_numeric(ibtracs_gdf[col])
            ibtracs_gdf["geometry"] = gpd.points_from_xy(ibtracs_gdf.LON, ibtracs_gdf.LAT)
            ibtracs_gdf = ibtracs_gdf.set_crs("EPSG:4326")
            assert not ibtracs_gdf.BASIN.isna().any()

            # rescale wind speeds to MSW10
            for agency in IBTRACS_AGENCIES:
                scale, shift = IBTRACS_AGENCY_10MIN_WIND_FACTOR[agency]
                ibtracs_gdf[f'{agency}_wind'.upper()] = pd.to_numeric(ibtracs_gdf[f'{agency}_wind'.upper()])
                ibtracs_gdf[f'{agency}_wind'.upper()] += shift
                ibtracs_gdf[f'{agency}_wind'.upper()] *= scale

            # rescale WMO_WIND column to MSW10
            def rescale_wmo(wind, agency):
                wind = float(wind)
                if wind > 0:
                    scale, shift = IBTRACS_AGENCY_10MIN_WIND_FACTOR[agency]
                    return wind * scale + shift
                else:
                    return np.nan
            ibtracs_gdf['WMO_WIND'] = ibtracs_gdf[['WMO_WIND', 'WMO_AGENCY']]\
                .apply(lambda x: rescale_wmo(x[0], x[1]), axis=1)

            # grab most-recorded wind speed if WMO not available
            if not ibtracs_gdf.WMO_WIND.isna().all():
                wind_col = "WMO_WIND"
                pressure_col = "WMO_PRES"
                agency = ibtracs_gdf["WMO_AGENCY"].mode()
                rmw_col = f"USA_RMW"  # double-check this, f"{agency}_RMW"
            else:
                wind_col = ibtracs_gdf[WIND_COLS].notna().sum().idxmax()
                agency = wind_col.split("_")[0]
                pressure_col = f"{agency}_PRES"
                rmw_col = f"{agency}_RMW"
            ibtracs_gdf = ibtracs_gdf.dropna(subset=wind_col).reset_index()

            # fix timestamps formatting
            newtimes = []
            for time in ibtracs_gdf["ISO_TIME"]:
                if len(time) > 8:
                    date = time[:10]
                    newtimes.append(time)
                else:
                    newtime = f"{date} {time}"
                    newtimes.append(newtime)

            # start calculating wind field
            centroids = feature_gdf.to_crs("EPSG:3857").centroid.to_crs("EPSG:4326")
            wind_tracks = ibtracs_gdf.geometry
            lats = [*wind_tracks.y]
            lons = [*wind_tracks.x]

            # haversine distances
            h_distances = []
            for centroid in centroids:
                h_distances.append(haversine(centroid.x, centroid.y, lons, lats))
            h_distances = np.array(h_distances)
            assert len(ibtracs_gdf) == h_distances.shape[1]

            # calculate wind field for each time stamp
            timestamps = []
            for time in range(len(ibtracs_gdf)):

                # inputs for holland function
                h_dists = [x[time] for x in h_distances]
                basin = ibtracs_gdf["BASIN"][time]
                pressure_env = BASIN_ENV_PRESSURE[basin]
                pressure = float(ibtracs_gdf[pressure_col][time])
                lat = float(ibtracs_gdf["LAT"][time])

                # radius of maximum winds
                if units_df[rmw_col][0] == "nmile":
                    r = nmile_to_km(float(ibtracs_gdf[rmw_col][time]))
                else:
                    r = float(ibtracs_gdf[rmw_col][time])

                # maximum wind speed
                if units_df[wind_col][0] == "kts":
                    wind = knots_to_mps(float(ibtracs_gdf[wind_col][time]))
                else:
                    wind = ibtracs_gdf[wind_col][time]

                # calculate wind field
                wind_field = []
                for distance in h_dists:
                    wind_speed = holland_wind_field(r, wind, pressure, pressure_env, distance, lat)
                    wind_field.append(wind_speed)

                # reformat time string
                iso_time = ibtracs_gdf['ISO_TIME'][time]
                date, time = iso_time.split(" ")
                date = date[5:].replace("-", "")
                time = time[:2]

                # if non-neglible wind append to dataframe
                if sum(wind_field) > 0:
                    feature_gdf[f"wnd{date}_{time}"] = wind_field       

            timemask = ["wnd" in col for col in feature_gdf.columns]
            timestamps = feature_gdf.columns[timemask]
            feature_gdf["wind_avg"] = feature_gdf[timestamps].mean(axis=1)
            self.feature_gdf[subregion] = feature_gdf
            self.save_gdf(subregion)

        if viz:
            fig, ax = plt.subplots(1, 1)
            self.feature_gdf[subregion].plot(column="wind_avg", ax=ax, cmap="Spectral", legend=True);
            ax.set_title(f"Average wind {self.startdate}-{self.enddate}")

    def get_all_data(self, subregion, recalculate=False, viz=True):
        feature_list = ["flood", "elevation", "permwater", "pw_dists", "precipitation", "soilcarbon", "mangroves", "ndvi", "wind_fields"]
        for feature in feature_list:
            getattr(self, f"get_{feature}")(subregion, recalculate=recalculate, viz=viz)

    def process_all_subregions(self, recalculate=False, viz=True):
        """Process all subregions and save to feature_stats directory."""
        for subregion in range(self.nsubregions):
            print(f'\nProcessing subregion {subregion}\n')
            self.get_all_data(subregion, recalculate=recalculate, viz=viz)
            self.feature_gdf[subregion].to_file(join(self.wd, "feature_stats", f"{self.storm}_{self.region}_{subregion}_{self.stagger}.shp"))
