"""Converts raw data visibilities


"""

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

from heracasa import closure as hc
from heracasa import data
from heracasa.data import uvconv

from recipe import casatasks as c
from recipe import repo


DataDir = "/rds/project/bn204/rds-bn204-asterics/HERA/data"
procdir = "/rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_raw"

Pol = "xx"

# IDR2 DATASET
IDR2 = [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, 2458106,
        2458107, 2458108, 2458109, 2458110, 2458111, 2458112, 2458113, 2458114, 2458115,
        2458116, 2458140]
IDRDays = [2458116, 2458140]

InData = []
for d in IDRDays:
    [InData.append(g) for g in glob.glob(
        os.path.join(DataDir, str(d), Pol, "*.uv"))]

# # Can further select times by specifying an InTimes arrays
# InTimes = []
# for d in IDRDays:
#     for t in InTimes:
#         [InData.append(g) for g in glob.glob(os.path.join(DataDir,str(d),Pol,"*."+str(t)+"*.uv"))]

InData = sorted(InData)

# Processing specification
ProcessData = True

# Multiprocessing specification
NUMBER_OF_CORES = 8

RefAnt = "53"
# Antennas removed: 86, 88, 137, 139
inAntenna = [0,   1,   2,  11,  12,  13,  14,  23,  24,  25,  26,  27,  36,
             37,  38,  39,  40,  41,  50,  51,  52,  53,  54,  55,  65,  66,
             67,  68,  69,  70,  71,  82,  83,  84,  85,  87, 120, 121, 122,
             123, 124, 140, 141, 142, 143]
# Baselines removed: [85,86], [86, 87], [87, 88], [136, 137], [137, 138], , [138, 139], [139, 140]
inBaselines = [[0, 1], [1, 2], [11, 12], [12, 13], [13, 14], [23, 24], [24, 25],
               [25, 26], [26, 27], [36, 37], [37, 38], [
                   38, 39], [39, 40], [40, 41], [50, 51],
               [51, 52], [52, 53], [53, 54], [54, 55], [
                   65, 66], [66, 67], [67, 68], [68, 69],
               [69, 70], [70, 71], [82, 83], [83, 84], [
                   84, 85], [120, 121], [121, 122],
               [122, 123], [123, 124], [140, 141], [141, 142], [142, 143]]


# miriad to MS conversion, without cache
def cv(uvin):
    """Convert visibilities from miriad to measurement set file format"""
    fitsi = os.path.splitext(os.path.basename(uvin))[0]+".uvfits"
    uvconv.cvuvfits(uvin, fitsi)
    uvconv.renumb(fitsi, fitsi)
    msout = fitsi[:-len('uvfits')] + 'ms'
    casa.importuvfits(fitsi, msout)  # why must casa be specified?
    os.remove(fitsi)
    return msout


# Bad antennas found here: http://hera.pbworks.com/w/page/123874272/H1C_IDR2
# Map IDRDay to bad antennas
bad_ants = np.empty((2, 19), list)
bad_ants[0, :] = IDR2
bad_ants_list = [
    [0, 136, 50, 2],
    [0, 50],
    [0, 50],
    [0, 50, 98],
    [0, 136, 50, 98],
    [50, 2],
    [0, 136, 50, 98],
    [0, 136, 50],
    [0, 136, 50, 98],
    [0, 136, 50],
    [137, 50, 2],
    [0, 136, 50],
    [0, 136, 50],
    [0, 50],
    [0, 136, 50, 98],
    [0, 136, 50, 11],
    [0, 136, 50],
    [0, 50, 98],
    [104, 50, 68, 117]
]
bad_ants[1, :] = np.array([np.array(bad_ants_day)
                           for bad_ants_day in bad_ants_list])


# Converting antenna numbers from HERA to CASA numbering (adding 1)
bad_ants_casa = np.copy(bad_ants)
bad_ants_casa[1, :] += 1


# Flagging measurement sets
def gcflagdata(msin):
    """Flag bad antennas for visibility dataset

    :param msin: Visibility in measurement set file format
    """
    IDRDay = int(msin.split('.')[1])  # Getting JD from filename
    bad_ants_index = np.where(bad_ants_casa[0, :] == IDRDay)[0][0]
    print('Flagged antennas for IDRDay ' + str(IDRDay) + ' are ' +
          str(bad_ants[1, bad_ants_index]))
    casa.flagdata(msin, flagbackup=True, mode='manual',
                  antenna=str(bad_ants_casa[1, bad_ants_index]).replace("[", "").replace("]", ""))
    # cutting visibilities at extremes of bandwidth
    casa.flagdata(msin, flagbackup=True, mode='manual', spw='0:0~65')
    casa.flagdata(msin, flagbackup=True, mode='manual', spw='0:930~1024')
    casa.flagdata(msin, autocorr=True)
    return msin


def fringerot(din, phasecenter='J2000 03h22m41.789s -37d12m29.52s'):
    """Fringe rotate visibilities

    Default is to rotate to Fornax A coordinates

    :param phasecenter: J2000 coordinates
    """
    casa.fixvis(din, din, phasecenter=phasecenter)
    return din


def cleanspace(dir):
    """Deletes all files in specified directory"""
    shutil.rmtree(dir, ignore_errors=True)
    os.mkdir(dir)


def genvisibility(fin, **kwargs):
    """Save the calibrated data arrays to npz file format"""
    fout = os.path.split(fin)[-1] + ".npz"
    r = hc.vis(fin, baseline=inBaselines, alist=inAntenna)
    np.savez(fout, **r)
    if not os.path.exists(fout):
        raise RuntimeError('No output produced by hc.vis!')
    return(fout)


def npz_conversion(uvin):
    """End to end conversion of raw visibility dataset to npz file format"""
    ms = cv(uvin)
    print('ms dataset to calibrate: '+str(ms))
    IDRDay_of_data = int(ms.split('.')[1])
    time_of_data = int(ms.split('.')[2])
    print('Flagging IDR dataset '+str(IDRDay_of_data)+'.'+str(time_of_data))
    gcflagdata(ms, IDRDay_of_data)
    print('ms data succesfully flagged: '+str(ms))
    fringerot(ms)
    print('Saving calibrated data to npz array')
    npz = genvisibility(ms, baseline=inBaselines, alist=inAntenna)
    # shutil.rmtree(ms)
    print('IDR dataset '+str(IDRDay_of_data)+'.' +
          str(time_of_data)+' saved to npz array')
    return npz


def multiprocess_wrapper(files):
    for f in files:
        npz_conversion(f)


def main():
    # cleanspace(procdir)
    os.chdir(procdir)
    # mkinitmodel()

    files_per_core = len(InData) / NUMBER_OF_CORES
    split_files = [InData[i:i+files_per_core]
                   for i in range(0, len(InData), files_per_core)]
    remainder_files = InData[files_per_core * NUMBER_OF_CORES:len(InData)]

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
        job.join()

    for file in remainder_files:
        npz_conversion(file)


if ProcessData:
    main()
