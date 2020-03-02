import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import SWOTdenoise as swotd

# Libraries needed in SWOT module
import netCDF4
import scipy

filedir  = 'example_data/'
filename = filedir + 'MED_fastPhase_1km_swotFAST_c01_p009.nc'

def test_read_data():
    assert type(swotd.read_data(filename, 'SSH_obs', 'lon', 'lat', 'x_ac', 'time')) is tuple
