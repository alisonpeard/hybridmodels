# tuple of cmap, set_under, set_over
cmap_key = {"floodfrac": ("YlGnBu", "lightgrey", "black"),
            "elevation": ("terrain", "black", "white"),
            "jrc_permwa": ("YlGnBu", "lightgrey", "black"),
            "dist_pw": ("plasma", "black", "white"),
            "slope_pw": ("plasma", "black", "white"),
            "precip": ("YlGnBu", "lightgrey", "black"),
            "soilcarbon": ("YlOrBr", "lightgrey", "black"),
            "ndvi": ("YlGn", "lightgrey", "black"),
            "wind_avg": ("Spectral_r", "blue", "red"),
            "Tm6": ("Spectral_r", "blue", "red"),
            "Tm3": ("Spectral_r", "blue", "red"),
            "T": ("Spectral_r", "blue", "red"),
            "aqueduct":("YlGnBu", "lightgrey", "black")
}

cmap_range = {"floodfrac": (0, 1),
            "elevation": (-1000, 1000),
            "jrc_permwa": (0, 100),
            "dist_pw": (0, 2500),
            "slope_pw": (-100, 50),
            "precip": (0, 50),
            "soilcarbon": (0, 40),
            "ndvi": (-1500, 9000),
            "wind_avg": (8, 30),
            "Tm6": (8, 30),
            "Tm3": (8, 30),
            "T": (8, 30),
            "aqueduct": (0, 5)
}