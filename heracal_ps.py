from heracasa.data import uvconv
from heracasa import data
import aipy as a
import numpy as np
import astropy
import sys
import os
import shutil
import datetime
from pyuvdata import UVData
uvd = UVData()

Dir = '/rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_heracal/'
uvdata = 'zen.grp1.of2.xx.LST.0.52409.uvOCRSDL'
namein = Dir + uvdata


# Converting Miriad file to Measurement Set
fitsi = os.path.splitext(os.path.basename(namein))[0]+".uvfitsOCRSD"
uvd.read_miriad(namein)
uvd.write_uvfits(Dir+fitsi, spoof_nonessential=True, force_phase=True)
msout = fitsi[:-len('uvfitsuvOCRSD')] + 'msOCRSD'
importuvfits(fitsi, msout)


# Getting integration time of data
tb.open(msout)
time_obs = tb.getcol("TIME")
tb.close()
exposure = time_obs[-1] - time_obs[0]

# listobs(vis=zen)
plotms(vis=msout, xaxis='chan', yaxis='amp', ydatacolumn='corrected',
       dpi=600, highres=True, coloraxis='baseline', avgtime=str(exposure))

# now run baseline psepc analysis on grouped LST binned visibilities
