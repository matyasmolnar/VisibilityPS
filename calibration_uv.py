"""Calibration of single visibility dataset in miriad file format

Calibration steps for HERA IDR2 visibilities:
    1. Miriad visibilities are converted to measurement set (CASA) file format
    2.
"""

import os

from calibration_function import cv, gcflagdata


DataDir = '/rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_heracal'
dataset = 'zen.grp1.of2.xx.LST.2.96700.uvOCRSDL'


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
