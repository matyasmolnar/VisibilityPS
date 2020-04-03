"""Batch traditional calibration of visibilities"""


import os
import shutil
import numpy as np
from pyuvdata import UVData as UV
from heracasa import data
from heracasa.data import uvconv
from recipe import casatasks as c


GCCleanMask='ellipse[[17h45m00.0s,-29d00m00.00s ], [ 11deg, 4deg ] , 30deg]'
FornaxACleanMask = 'ellipse[[3h22m41.8s,-37d12m29.52s], [ 1deg, 1deg ] , 10deg]'

c.repo.REPODIR = "/rds/project/bn204/rds-bn204-asterics/cache"

# testing locally. have created same directory hierarchy on hard drive
DataDir = "/Volumes/TOSHIBA_EXT/HERA_Data"
procdir = "/Volumes/TOSHIBA_EXT/HERA_Data/calibrated"

Pol = "xx"

InDays = [2458098]
InTimes = [31193]


def makemsfile(uvin, **kwargs):
    print(uvin)
    # hash function call
    hh = c.hf(makemsfile, uvin)
    mm = repo.get(hh)
    print(mm)
    print('MS file already hashed and cached')
    if not mm:
        print('Converting MS file and storing it in cache')
        UV = pyuvdata.UVData()
        UV.read_miriad(uvin)
        UV.phase_to_time(Time(UV.time_array[0], format='jd', scale='utc'))
        tempf = repo.mktemp()
        os.remove(tempf)
        UV.write_uvfits(tempf, spoof_nonessential=True)
        if not os.path.exists(tempf):
            raise RuntimeError("No output produced by mkuvfits!")
        foms = c.importuvfits(tempf)
        os.remove(tempf)
        #flms = c.flagdata(foms,autocorr=True)
        mm = repo.put(foms, hh)
    # to return MS file, would have to return foms
    return(mm)


# copy RFI flags from uvOR files - this is already done for data in RDS

# main script for this batch calibration
def main():
    # cleanspace()
    os.chdir(procdir)
    ms = [cv(x) for x in InData]
    # ms = ['zen.2458098.12552.xx.HH.ms']
    print('ms datasets to calibrate: '+str(ms))
    flagged = []
    for x in ms:
        # IDRDay_of_data = int(os.path.basename(os.path.dirname(os.path.dirname(x))))
        IDRDay_of_data = int(x.split('.')[1])
        time_of_data = int(x.split('.')[2])
        print('Flagging IDR dataset '+str(IDRDay_of_data)+'.'+str(time_of_data))
        gcflagdata(x, IDRDay_of_data)
        flagged.append(x)
    print('ms data succesfully flagged: '+str(flagged))
    ms = [fringerot(x) for x in flagged]
    mkinitmodel()
    cals1 = [calinitial(x) for x in ms]
    print('Gain and delay calibration tables produced: '+str(cals1))
    cals2 = [dobandpass(x) for x in ms]
    print('Calibration complete')


# checking quality of calibrated data with imaging and gain / delay calibration plots
# clean(vis=msin, niter=0, imagename='test.img', weighting='briggs', robust=0, imsize=[512,512], cell=['250arcsec'], mode='mfs')


if __name__ == "__main__":
    main()
