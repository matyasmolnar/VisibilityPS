import casa
import casac

import os
import sys
import glob
import shutil
import datetime
import aipy as a
import numpy as np
import pyuvdata
from astropy.time import Time
import multiprocessing

import heracasa.closure as hc
from heracasa import data
from heracasa.data import uvconv

from recipe import casatasks as c
from recipe import repo


DataDir = "/rds/project/bn204/rds-bn204-asterics/HERA/data"
procdir = "/rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_calib_v1"

# c.repo.REPODIR="/rds/project/bn204/rds-bn204-asterics/cache"

Pol = "xx"

# Multiprocessing specification
NUMBER_OF_CORES = len(IDRDays)


def npz_conversion(uvin):
    ms = cv(uvin)
    print('ms dataset to calibrate: '+str(ms))
    IDRDay_of_data = int(ms.split('.')[1])
    time_of_data = int(ms.split('.')[2])
    print('Flagging IDR dataset '+str(IDRDay_of_data)+'.'+str(time_of_data))
    gcflagdata(ms, IDRDay_of_data)
    print('ms data succesfully flagged: '+str(ms))
    fringerot(ms)
    # os.system('rm -rf FornaxA.cl')
    cals1 = calinitial(ms)
    print('Gain and delay calibration tables produced: '+str(cals1))
    cals2 = dobandpass(ms)
    print('Calibration complete')
    print('Saving calibrated data to npz array')
    npz = genvisibility(ms, baseline=inBaselines, alist=inAntenna)
    # shutil.rmtree(ms)
    print('IDR dataset '+str(IDRDay_of_data)+'.' +
          str(time_of_data)+' saved to npz array')
    return npz


# checking quality of calibrated data with imaging and gain / delay calibration plots
# clean(vis=msin, niter=0, imagename='test.img', weighting='briggs', robust=0, imsize=[512,512], cell=['250arcsec'], mode='mfs')


def multiprocess_wrapper(files):
    for f in files:
        npz_conversion(f)


def main():
    # cleanspace()
    os.chdir(procdir)
    # mkinitmodel()

    files_per_core = len(InData) / NUMBER_OF_CORES
    # print(files_per_core)
    split_files = [InData[i:i+files_per_core]
                   for i in range(0, len(InData), files_per_core)]
    # print(split_files)
    remainder_files = InData[files_per_core * NUMBER_OF_CORES:len(InData)]
    # print(remainder_files)

    print(str(len(InData))+' uv dataset(s) to reduce')
    # print('split files are '+str(split_files))
    # print('remainder files are '+str(remainder_files))

    jobs = []
    # for i, list_slice in enumerate(split_files):
    #     print('list slice is '+str(list_slice))
    #     j = multiprocessing.Process(target=multiprocess_wrapper, args=(list_slice,))
    #     jobs.append(j)

    for list_slice in split_files:
        # print('list slice is '+str(list_slice))
        j = multiprocessing.Process(
            target=multiprocess_wrapper, args=(list_slice,))
        jobs.append(j)

    for job in jobs:
        job.start()
        job.join()  # what does this do?

    for file in remainder_files:
        npz_conversion(file)


if ProcessData:
    main()


###################################################

# def mjd_to_jd(zen):
#     casa.tb.open(zen)
#     time_obs = tb.getcol("TIME")
#     # exposure = tb.getcol("EXPOSURE")
#     casa.tb.close()
#
#     time_stamp = time_obs[0] # units of second, follows MJD
#     mjd_day = time_stamp / 60 / 60 / 24
#     jd_day = au.mjdToJD(mjd_day)
#     return(jd_day)


# # main script for batch calibration - no multiprocess
# def conversion(fin):
#     cleanspace()
#     os.chdir(procdir)
#     ms=[cv(x) for x in fin]
#     print('ms datasets to calibrate: '+str(ms))
#     flagged = []
#     for x in ms:
#         # IDRDay_of_data = int(os.path.basename(os.path.dirname(os.path.dirname(x))))
#         # use this if working with files of the form 'zen.2458098.12552.xx.HH.ms':
#         IDRDay_of_data = int(x.split('.')[1])
#         time_of_data = int(x.split('.')[2])
#         # # use this if working with files stored in the cache:
#         # jd = mjd_to_jd(x)
#         # IDRDay_of_data = int(jd)
#         # time_of_data = jd - IDRDay_of_data
#
#         print('Flagging IDR dataset '+str(IDRDay_of_data)+'.'+str(time_of_data))
#         gcflagdata(x, IDRDay_of_data)
#         flagged.append(x)
#     print('ms data succesfully flagged: '+str(flagged))
#     ms=[fringerot(x) for x in flagged]
#     mkinitmodel()
#     cals1=[calinitial(x) for x in ms]
#     print('Gain and delay calibration tables produced: '+str(cals1))
#     cals2=[dobandpass(x) for x in ms]
#     print('Calibration complete')
#     print('Saving calibrated data to npz array')
#     for x in ms:
#         genvisibility(x, baseline=inBaselines, alist=inAntenna)
#         shutil.rmtree(x)
