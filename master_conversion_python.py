# copying over to CSD3:
# scp /Users/matyasmolnar/HERA_Data/VisibilityPS/{master_conversion_python.py,slurm_submit_python.peta4-skylake} mdm49@login-cpu.hpc.cam.ac.uk:/rds/project/bn204/rds-bn204-asterics/mdm49


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

#GCCleanMask='ellipse[[17h45m00.0s,-29d00m00.00s ], [ 11deg, 4deg ] , 30deg]'
FornaxACleanMask = 'ellipse[[3h22m41.8s,-37d12m29.52s], [ 1deg, 1deg ] , 10deg]'

DataDir = "/rds/project/bn204/rds-bn204-asterics/HERA/data"
procdir = "/rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_calib_v1"

# c.repo.REPODIR="/rds/project/bn204/rds-bn204-asterics/cache"

# testing locally. have created same directory hierarchy on hard drive
# DataDir="/Volumes/TOSHIBA_EXT/HERA_Data"
# procdir="/Volumes/TOSHIBA_EXT/HERA_Data/calibrated"

Pol = "xx"

# where is this from?
# IDRDays=[2458098,2458099,2458143,2458144,2458145,2458146,2458147,2458148,2458149,2458150,2458151,2458152,2458153,2458154,2458155,2458156,2458157,2458158,2458159,2458160]

# IDR2 DATASET
IDR2 = [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, 2458106, 2458107,
        2458108, 2458109, 2458110, 2458111, 2458112, 2458113, 2458114, 2458115, 2458116, 2458140]
# IDRDays=[2458106, 2458107, 2458108, 2458109, 2458110, 2458111, 2458112, 2458113, 2458114, 2458115, 2458116, 2458140]
# need to do 2458114 - not exported..

# InTimes=[12552]

InData = []

for d in IDR2:
    [InData.append(g) for g in glob.glob(
        os.path.join(DataDir, str(d), Pol, "*.uv"))]

# # Selecting times
# for d in IDRDays:
#     for t in InTimes:
#         [InData.append(g) for g in glob.glob(os.path.join(DataDir,str(d),Pol,"*."+str(t)+"*.uv"))]

InData = sorted(InData)

# removing datasets that have already been converted
InData_filtered = list(InData)  # copying list of all sessions
for IDR2_session in InData:
    if os.path.exists(os.path.join(procdir, os.path.split(IDR2_session)[-1][:-len('uv')]+'ms.npz')):
        InData_filtered.remove(IDR2_session)

InData = InData_filtered

# processing specification
ProcessData = True

# Multiprocessing specification
NUMBER_OF_CORES = len(IDRDays)  # reduce number of cores

RefAnt = "53"  # as a string
# Deleted antennas are 86, 88, 137, 139
inAntenna = [0,   1,   2,  11,  12,  13,  14,  23,  24,  25,  26,  27,  36,
             37,  38,  39,  40,  41,  50,  51,  52,  53,  54,  55,  65,  66,
             67,  68,  69,  70,  71,  82,  83,  84,  85,  87, 120, 121, 122,
             123, 124, 140, 141, 142, 143]
inBaselines = [[0, 1], [1, 2], [11, 12], [12, 13], [13, 14], [23, 24], [24, 25], [25, 26], [26, 27], [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [50, 51], [51, 52], [52, 53], [53, 54], [
    54, 55], [65, 66], [66, 67], [67, 68], [68, 69], [69, 70], [70, 71], [82, 83], [83, 84], [84, 85], [120, 121], [121, 122], [122, 123], [123, 124], [140, 141], [141, 142], [142, 143]]
# Deleted baselines: [85,86], [86, 87], [87, 88], [136, 137], [137, 138], , [138, 139], [139, 140]
# See ms_data.py to find all EW baselines and output as list


# miriad to MS conversion, without cache
def cv(uvin):
    fitsi = os.path.splitext(os.path.basename(uvin))[0]+".uvfits"
    uvconv.cvuvfits(uvin, fitsi)
    uvconv.renumb(fitsi, fitsi)
    msout = fitsi[:-len('uvfits')] + 'ms'
    casa.importuvfits(fitsi, msout)  # why must casa be specified?
    os.remove(fitsi)
    return msout


# copy RFI flags from uvOR files - this is already done for data in RDS
# find bad antennas here: http://hera.pbworks.com/w/page/123874272/H1C_IDR2
# 2: IDRDay and bad antenna lists, 19: Number of IDRDays
bad_ants = np.zeros((2, 19), list)
# bad antennas also found in /users/mmolnar/mmolnar/hera_opm/pipelines/h1c/idr2/v1/bad_ants on the NRAO severs
bad_ants[0, :] = IDR2  # [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, 2458106, 2458107, 2458108, 2458109, 2458110, 2458111, 2458112, 2458113, 2458114, 2458115, 2458116, 2458140]
bad_ants[1, :] = [
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


bad_ants_casa = bad_ants
# Converting ant number from HERA to CASA (adding 1)
for i in range(len(bad_ants[1, :])):
    for j in range(len(bad_ants[1, i])):
        bad_ants_casa[1, i][j] = bad_ants[1, i][j] + 1


# add +1 to HERA antenna number
def gcflagdata(msin, IDRDay):
    bad_ants_index = np.where(bad_ants_casa[0, :] == IDRDay)[0][0]
    print('Flagged antennas for IDRDay '+str(IDRDay) +
          ' are '+str(bad_ants[1, bad_ants_index]))
    casa.flagdata(msin, flagbackup=True, mode='manual', antenna=str(
        bad_ants_casa[1, bad_ants_index]).replace("[", "").replace("]", ""))
    # flagdata(msin, flagbackup=True, mode='manual', antenna='1, 137, 51, 3')
    # cutting visibilities at extremes of bandwidth
    casa.flagdata(msin, flagbackup=True, mode='manual', spw='0:0~65')
    casa.flagdata(msin, flagbackup=True, mode='manual', spw='0:930~1024')
    casa.flagdata(msin, autocorr=True)
    return msin


# fringe-rotation to Fornax A:
def fringerot(din):
    """Fringe Rotate to Fornax A:"""
    casa.fixvis(din, din, phasecenter='J2000 03h22m41.789s -37d12m29.52s')
    return din


# model for Fornax A
def mkinitmodel():
    """ Initial model: just a point source in Fornax A direction"""
    os.system('rm -rf FornaxA.cl')
    casa.componentlist.done()
    casa.componentlist.addcomponent(flux=1.0,
                                    fluxunit='Jy',
                                    shape='point',
                                    dir='J2000 03h22m41.789s -37d12m29.52s')
    casa.componentlist.rename('FornaxA.cl')
    casa.componentlist.close()


# initial calibration
def calname(m, c):
    # build calibrator name based on filename
    return os.path.basename(m)+c+'.cal'


def calinitial(msin):
    "Initial calibration of the data set"

    # Fill the model column
    casa.ft(msin, complist='FornaxA.cl', usescratch=True)

    kc = calname(msin, 'K')  # delays
    gc = calname(msin, 'G')  # gain

    casa.gaincal(vis=msin, caltable=kc, gaintype='K', solint='inf',
                 refant='11', minsnr=1)  # , spw='0:100~130,0:400~600')
    # spw not necessary
    # ensure reference antenna exists and isn't faulty
    casa.gaincal(vis=msin, caltable=gc, gaintype='G', solint='inf',
                 refant='11', minsnr=1, calmode='ap', gaintable=kc)
    casa.applycal(msin, gaintable=[kc, gc])
    return [kc, gc]


def dobandpass(msin):
    bc = calname(msin, 'B')
    casa.bandpass(vis=msin, minsnr=1, solnorm=False, bandtype='B', caltable=bc)
    casa.applycal(msin, gaintable=[bc])
    return bc


def cleanspace():
    # deletes all files in procdir
    shutil.rmtree(procdir, ignore_errors=True)
    os.mkdir(procdir)


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
