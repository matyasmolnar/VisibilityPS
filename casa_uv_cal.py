"""Calibration of single visibility dataset in miriad file format

Calibration steps for HERA IDR2 visibilities:
    1. Miriad visibilities are converted to measurement set (CASA) file format
    2.
"""

import os
import shutil
import sys

import numpy as np
from pyuvdata import UVData

from heracasa import data
from heracasa.data import uvconv

from calibration_function import cv, gcflagdata


namein = '/rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_heracal/zen.grp1.of2.xx.LST.2.96700.uvOCRSDL'


FornaxA_RA = '3h22m41.8s'
# FornaxA_RA = 3.378 in hour fraction
# +- 20min from FornaxA transit: lst_start = 3.05 ; lst_end = 3.71

# evaluate total time of observation
tb.open(zen)
time_obs = tb.getcol("TIME")
tb.close()
exposure = time_obs[-1] - time_obs[0]


def add_hera_obs_pos():
    """Adding HERA observatory position (at some point as PAPER_SA)"""
    obstablename = os.getenv('CASAPATH').split()[0] + '/data/geodetic/Observatories/'
    tb.open(obstablename, nomodify=False)
    paperi = (tb.getcol('Name') == 'PAPER_SA').nonzero()[0]
    tb.copyrows(obstablename, startrowin=paperi, startrowout=-1, nrow=1)
    tb.putcell('Name', tb.nrows()-1, 'HERA')
    tb.close()


def plot_visibilities():
    plotms(vis=zen, xaxis='chan', yaxis='amp', ydatacolumn='corrected',
           dpi=600, highres=True, coloraxis='baseline', avgtime=str(exposure))
    plotcal(caltable=cal, xaxis='chan', yaxis='amp')
    plotcal(caltable=cal, xaxis='chan', yaxis='amp', antenna='13&14')


def main():

    cv(namein)
    zen = namein[:-len('uvOCRSDL')] + 'ms'
    # Can copy RFI flags from uvOR files
    gcflagdata(zen)
    fringerot(zen)
    mkinitmodel()
    calinitial(zen)
    apply_calibration(zen, [kc, gc])

    dosplit(zen, "ical")
    zen_cal = zen + str('ical.ms')

    cleaninit(zen_cal)
    dobandpass(zen_cal)
    # calsf = cleanfinal(zen_cal)
    # viewer('GC.combined.img.image/', outfile='cleaned_GC.png', outdpi=600)

    dosplit(zen_cal, 'c2', spw='0:100~880')
    zen_cal_band_clean = zen_cal + str('c2.ms')

    cleaninit(zen_cal_band_clean)
    dobandpass(zen_cal_band_clean)
    cleanfinal(zen_cal_band_clean)

    dosplit(zen_cal_band_clean, 'final')

    zen_cal_band2_clean2 = zen_cal_band_clean + str('final.ms')
    plotms(vis=zen_cal_band2_clean2, xaxis='chan', yaxis='amp', ydatacolumn='corrected',
           dpi=600, highres=True, coloraxis='baseline', avgtime=str(exposure))
