# run python -m pytest -x from python folder

try:
    import pytest
    import os
    import sys
    import geopandas as gpd
    from model_utils import *
    from data_utils import default_features, intermediate_features
    WD = os.path.join("/Users", "alison", "Documents", "DPhil", "hybridmodels", "data")
except:
    pass

def test_import():
    import model_utils
    import data_utils


# ASPATIAL DATA CHECKS
@pytest.mark.parametrize("wd", [os.path.join(WD, "storm_events")])
def test_load_aspatial_files(wd):
    gdfs = load_aspatial_files(wd)
    assert isinstance(gdfs, list)
    for gdf in gdfs:
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert 'geometry' in gdf.columns
        assert 'storm' in gdf.columns
        assert 'region' in gdf.columns


@pytest.mark.parametrize("wd", [os.path.join(WD, "storm_events")])
def test_load_aspatial_data(wd):
    gdf, features, columns = load_aspatial_data(wd, default_features)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert isinstance(features, list)
    assert isinstance(columns, list)


@pytest.mark.parametrize("wd, feature", [
    (os.path.join(WD, "storm_events"), 'slope_coast'),
    (os.path.join(WD, "storm_events"), 'soilcarbon')
    ])
def test_feature_forna(wd, feature):
    gdf, *_ = load_aspatial_data(wd, [feature])
    assert gdf[feature].dtype == float, f"Field {feature} has type {gdf[feature].dtype}"
    assert not gdf[feature].isnull().any(), f"NaNs found in {feature} for {gdf.loc[gdf[feature].isnull(), 'event'].unique()}"


# SPATIAL DATA CHECKS
@pytest.mark.parametrize("wd", [os.path.join(WD, "feature_stats", "dataframes")])
def test_load_spatial_files(wd):
    gdfs = load_spatial_files(wd)
    assert isinstance(gdfs, list)
    for gdf in gdfs:
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert 'geometry' in gdf.columns
        assert 'event' in gdf.columns
        assert 'mangrove_spatial' in gdf.columns
        assert 'mangrove_to_pw' in gdf.columns


@pytest.mark.parametrize("wd", [os.path.join(WD, "feature_stats", "dataframes")])
def test_intermediate_features(wd):
    gdfs = load_spatial_files(wd)
    assert isinstance(gdfs, list)
    for gdf in gdfs:
        for feature in intermediate_features.keys():
            if gdf[feature].sum() > 0:
                assert gdf[f"{feature}_to_pw"].sum() > 0, f"{feature}_to_pw should not be all zero for {gdf.event[0]}."