#######
# Process new Sentinel data to produce data frames of sample points,
# spectral bands, and their classifications. (Needs user input)
##########
rm(list=ls())
par(mfrow=c(1,1))
library(raster)
library(rgdal)  # readOGR()
library(maps)
library(sp)
library(sf)
library(terra)  # rgdal being retired

wd <- "/Users/alisonpeard/Documents/Oxford/DPhil/hybridmodels/miniproject/4_FLOOD HEIGHT/Regular Climate"
lonlat <- '+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0'
r1e1 <- readOGR(dsn = paste0(wd), layer = "REGION_1_E1")
crs(r1e1) <- lonlat
# 116.9271, 127.1762, 3.157889, 19.56799 

