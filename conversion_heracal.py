# copying over to CSD3:
# scp /Users/matyasmolnar/HERA_Data/VisibilityPS/{conversion_raw.py,slurm_submit_raw.peta4-skylake} mdm49@login-cpu.hpc.cam.ac.uk:/rds/project/bn204/rds-bn204-asterics/mdm49

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


DataDir = "/rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_heracal"
procdir = "/rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_heracal"

InData = [DataDir + '/' + data for data in os.listdir(DataDir)]

InData = sorted(InData)

# processing specification
ProcessData = True

# Multiprocessing specification
NUMBER_OF_CORES = 8  # len(IDRDays) # reduce number of cores


# miriad to MS conversion, without cache
def cv(uvin):
    fitsi = os.path.splitext(os.path.basename(uvin))[0]+".uvfits"
    uvconv.cvuvfits(uvin, fitsi)
    uvconv.renumb(fitsi, fitsi)
    msout = fitsi[:-len('uvfits')] + 'ms'
    casa.importuvfits(fitsi, msout)  # why must casa be specified?
    os.remove(fitsi)
    return msout


def genvisibility(fin, **kwargs):
    fout = os.path.split(fin)[-1]+".npz"
    r = hc.vis(fin, baseline=inBaselines, alist=inAntenna)
    np.savez(fout, **r)
    if not os.path.exists(fout):
        raise RuntimeError("No output produced by hc.vis !")
    return(fout)


def npz_conversion(uvin):
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
