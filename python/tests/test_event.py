# run python -m pytest -x from python folder
from _pytest.fixtures import fixture
import pytest

try:
    import os
    import sys
    import numpy as np
    import geopandas as gpd
    from event import Event
    WD = os.path.join("/Users", "alison", "Documents", "DPhil", "hybridmodels", "data")
except:
    pass

def test_import():
    from event import Event

@pytest.fixture
def sample_events():
    events = [
        ['roanu', 'chittagong', 4, WD]
    ]
    sample_events = []
    for event in events:
        storm = event[0]
        region = event[1]
        nsubregions = event[2]
        wd = os.path.join(WD, 'storm_events', f"{storm}_{region}")
        event = Event(storm, region, nsubregions, wd, event[3])
        sample_events.append(event)
    return sample_events


@pytest.mark.parametrize("idx", [0])
def test_event_init(sample_events, idx):
    event = sample_events[idx]
    for subregion in event.subregions:
        assert event._aoi_event.crs == event._flood.crs, "Flood and AoI have different crs for {storm}_{region}_{subregion}"
        assert event._aoi_event.unary_union.intersects(event._flood.unary_union), "Flood polygon not in AoI for {storm}_{region}_{subregion}"


# does recalculate work properly
# are the grids aligned correctly with the provided GeoJSONS


@pytest.mark.parametrize("idx", [0])
def test_make_slope(sample_events, idx):
    event = sample_events[idx]
    for subregion in event.subregions:
        event.get_coast_dists(subregion, recalculate=True, save=False)
        assert event._feature_gdf[subregion]['slope_coast'].dtype == float, f"Slope not float for {event.storm}, {event.region} {subregion}."
        assert event._feature_gdf[subregion]['slope_coast'].notnull().all(), f"Null values created in slope for {event.storm}, {event.region} {subregion}."
        assert not event._feature_gdf[subregion]['slope_coast'].isin([np.inf, -np.inf]).any(), f"Infinite values found in slope for {event.storm}, {event.region} {subregion} "