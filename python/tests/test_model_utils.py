# run python -m pytest -x from python folder
import pytest
import os
import sys
import geopandas as gpd

WD = os.path.join("~", "Documents", "DPhil", "hybridmodels", "data")

def test_import():
    import model_utils

@pytest.mark.parametrize("wd", [os.path.join(WD, "storm_events")])
def test_load_aspatial_data(wd):
    from model_utils import load_aspatial_data
    gdfs = load_aspatial_data(wd)
    assert isinstance(gdfs, list)
    for gdf in gdfs:
        assert isinstance(gdfs, gpd.GeoDataFrame)
        assert 'geometry' in gdf.columns
        assert 'storm' in gdf.columns
        assert 'region' in gdf.columns