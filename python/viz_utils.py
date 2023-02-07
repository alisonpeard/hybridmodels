import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

labels = {"floodfrac": "flooded fraction",
          "flood": "flood presence",
          "elevation": "elevation",
          "elevation_spatial": "neighbours elevation",
          "elevation_to_pw": "max elev. to p.water",
          "jrc_permwa": "permanent water",
          "dist_pw": "dist. to p. water",
          "slope_pw": "slope. to p.water",
          "precip": "precipitation",
          "soilcarbon": "soil organic carbon",
          "soilcarbon_to_pw": "avg. soil organic carbon\nto p.water",
          "evi": "EVI",
          "evi_anom": "EVI anomaly",
          "evi_to_pw": "avg. EVI to p.water",
          "evi_spatial": "neighbours EVI",
          "wind_avg": "avg. wind speed",
          "wind_max": "max. wind speed",
          "pressure_avg": "avg. pressure",
          "pressure_min": "min. pressure",
          "Tm6": "wind speed T-6h",
          "Tm3": "wind speed T-3h",
          "T": "wind speed T",
          "aqueduct": "Aqueduct flood map",
          "aqueduct_25": "Aqueduct flood map RP25",
          "aqueduct_50": "Aqueduct flood map RP50",
          "aqueduct_100":"Aqueduct flood map RP100",
          "deltares": "Deltares flood map",
          "lulc__10": "trees",
          "lulc__20": "shrubland",
          "lulc__30": "grassland",
          "lulc__40": "cropland",
          "lulc__50": "built-up areas",
          "lulc__60": "barren land",
          "lulc__70": "snow/ice",
          "lulc__80": "open water",
          "lulc__90": "herbaceous wetland",
          "lulc__95": "mangroves",
          "lulc__100": "moss and lichen",
          "vegetated": "vegetated (LULC)",
          "built_up": "built-up areas (LULC)",
          "bare_soil": "bare soil (LULC)",
          "water": "water (LULC)",
          "mangrove": "mangrove presence (Giri 2011)",
          "exclusion_mask": "exclusion mask",
          "soiltemp1": "soil temperature (0-7cm)",
          "soiltemp2": "soil temperature (7-28cm)",
          "soiltemp1_anom": "soil temperature anomaly (0-7cm)",
          "soiltemp2_anom": "soil temperature anomaly (7-28cm)",
          'sp': 'surface pressure',
          'mslp': 'mean sea-level pressure',
          'u10_u': 'u-component of wind (U10)',
          'u10_v': 'v-component of wind (U10)'
}


# tuple of cmap, set_under, set_over
cmap_key = {"floodfrac": ("YlGnBu", "lightgrey", "black"),
            "elevation": ("terrain", "black", "white"),
            "elevation_spatial": ("terrain", "black", "white"),
            "elevation_to_pw": ("terrain", "black", "white"),
            "jrc_permwa": ("YlGnBu", "lightgrey", "black"),
            "dist_pw": ("plasma", "black", "white"),
            "slope_pw": ("plasma", "black", "white"),
            "precip": ("YlGnBu", "lightgrey", "black"),
            "soilcarbon": ("YlOrBr", "lightgrey", "black"),
            "soilcarbon_spatial": ("YlOrBr", "lightgrey", "black"),
            "soilcarbon_to_pw": ("YlOrBr", "lightgrey", "black"),
            "soiltemp1": ("plasma", "lightgrey", "black"),
            "soiltemp2": ("plasma", "lightgrey", "black"),
            "evi": ("YlGn", "lightgrey", "black"),
            "evi_to_pw": ("YlGn", "lightgrey", "black"),
            "evi_spatial": ("YlGn", "lightgrey", "black"),
            "wind_avg": ("Spectral_r", "blue", "red"),
            "pressure_avg": ("Spectral_r", "blue", "red"),
            'u10_u': ("Spectral_r", "blue", "red"),
            'u10_v': ("Spectral_r", "blue", "red"),
            'sp': ("Spectral_r", "blue", "red"),
            'mslp': ("Spectral_r", "blue", "red"),
            "Tm6": ("Spectral_r", "blue", "red"),
            "Tm3": ("Spectral_r", "blue", "red"),
            "T": ("Spectral_r", "blue", "red"),
            "aqueduct":("YlGnBu", "lightgrey", "black"),
            "deltares":("YlGnBu", "lightgrey", "black"),
            "aqueduct_25":("YlGnBu", "lightgrey", "black"),
            "aqueduct_50":("YlGnBu", "lightgrey", "black"),
            "aqueduct_100":("YlGnBu", "lightgrey", "black"),
            "lulc":("Dark2", "white", "white"),
            "mangrove": ("YlGn", "lightgrey", "black")
}

cmap_range = {"floodfrac": (0, 1),
            "elevation": (-1000, 1000),
            "jrc_permwa": (0, 100),
            "dist_pw": (0, 2500),
            "slope_pw": (-100, 50),
            "precip": (0, 50),
            "soilcarbon": (0, 40),
            "evi": (-1500, 9000),
            "wind_avg": (8, 30),
            "Tm6": (8, 30),
            "Tm3": (8, 30),
            "T": (8, 30),
            "aqueduct": (0, 5),
            "aqueduct_25": (0, 5),
            "aqueduct_50": (0, 5),
            "aqueduct_100": (0, 5),
            "lulc": (10, 100)
}

soge_colours = {'orange': '#F68D2E',
               'yellowish green': '#C4D600',
               'dark turquoise': '#003E51',
               'medium turquoise': '#007D8A',
               'light turquoise': '#6DCDB8',
               'dark pink': '#F65275',
               'beige': '#EFEEEA',
               'oxford blue': '#002147',
               'eci blue': '#C8D1DF'
              }


def make_cmap(levels=np.arange(20, 65, 5), cmap='YlOrRd', under='lightgrey', over='black'):
    """
    Returns:
    -------
    cmap :
        Colourmap with defined levels
    src_norm:
        Norm to use when plotting with rasterio
    ax_norm:
        Norm to use when plotting with matplotlib
    """
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_under(under, 1.0)
    cmap.set_over(over, 1.0)
    src_norm = BoundaryNorm(boundaries=levels, ncolors=len(levels))  # cmap.N  # len(levels)
    ax_norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N)
    return cmap, src_norm, ax_norm
