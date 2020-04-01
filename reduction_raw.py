"""Data reduction of raw visibilities to npz file format (no calibration)

Data reduction steps for HERA IDR2 visibilities in miriad file format:
    1. Miriad visibilities are converted to measurement set (CASA) file format
    2. The visibilities are flagged in CASA:
        - Bad antennas removed
        - Band edges flagged
        - Autocorrelations flagged
    3. Visibilities are fringe rotated to the Fornax A coordinates
    4. The visibilities are saved as an ndarray to an npz file

heracasa package written by Bojan Nikolic, can be found at:
http://www.mrao.cam.ac.uk/~bn204/g/
"""

import glob
import logging
import multiprocessing
import os
import shutil
import sys

import casa
import numpy as np
from heracasa import closure as hc
from heracasa import data
from heracasa.data import uvconv


# IDR2 dataset
IDR2 = [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, 2458106,
        2458107, 2458108, 2458109, 2458110, 2458111, 2458112, 2458113, 2458114,
        2458115, 2458116, 2458140]


##############################################################
###### Modify the inputs in this section as appropriate ######
##############################################################

# Directory of visibilities in miriad file format to reduce
DataDir = "/rds/project/bn204/rds-bn204-asterics/HERA/data"
# Output directory
procdir = "/rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_raw"

# Polarization of visibilities to reduce
Pol = "xx"

# Select days to process
InDays = IDR2[-2:]
# OPTIONAL: can further select times by specifying an InTimes arrays
InTimes = [] # e.g. [12552]

# Multiprocessing specification
NUMBER_OF_CORES = 8

# Remove all files in procdir
clean_dir = True

# Check status of data reduction steps for each visibility dataset
conversion_verbose = True

##############################################################


# Adding paths of visibilities in miriad file format to be reduced
# Filtering done by specified InDays and InTimes
InData = []
for d in InDays:
    if InData:
        for t in InTimes:
            [InData.append(g) for g in glob.glob(
                os.path.join(DataDir, str(d), Pol, "*.{}*.uv".format(t)))]
    else:
        [InData.append(g) for g in glob.glob(
            os.path.join(DataDir, str(d), Pol, "*.uv"))]

InData = sorted(InData)

# Antennas removed as data from these shown to be bad: 86, 88, 137, 139
inAntenna = [0,   1,   2,  11,  12,  13,  14,  23,  24,  25,  26,  27,  36,
             37,  38,  39,  40,  41,  50,  51,  52,  53,  54,  55,  65,  66,
             67,  68,  69,  70,  71,  82,  83,  84,  85,  87, 120, 121, 122,
             123, 124, 140, 141, 142, 143]
# Baselines removed: [85,86], [86, 87], [87, 88], [136, 137], [137, 138],
#                    [138, 139], [139, 140]
inBaselines = [[0, 1], [1, 2], [11, 12], [12, 13], [13, 14], [23, 24], [24, 25],
               [25, 26], [26, 27], [36, 37], [37, 38], [38, 39], [39, 40],
               [40, 41], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55],
               [65, 66], [66, 67], [67, 68], [68, 69], [69, 70], [70, 71],
               [82, 83], [83, 84], [84, 85], [120, 121], [121, 122],
               [122, 123], [123, 124], [140, 141], [141, 142], [142, 143]]


def cv(uvin):
    """Convert visibilities from miriad to measurement set file format

    No caching done here
    """
    fitsi = os.path.splitext(os.path.basename(uvin))[0] + ".uvfits"
    uvconv.cvuvfits(uvin, fitsi)
    uvconv.renumb(fitsi, fitsi)
    msout = fitsi[:-len('uvfits')] + 'ms'
    casa.importuvfits(fitsi, msout)
    os.remove(fitsi)
    return msout


# Map JD to known bad antennas
# Bad antennas found here: http://hera.pbworks.com/w/page/123874272/H1C_IDR2
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
    print('Flagged antennas for JD {} are {}'.format(IDRDay,
          bad_ants[1, bad_ants_index]))
    casa.flagdata(msin, flagbackup=True, mode='manual',
                  antenna=str(bad_ants_casa[1, bad_ants_index]).replace("[", "").replace("]", ""))
    # Cutting visibilities at extremes of bandwidth
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
    """Removes all files in specified directory"""
    shutil.rmtree(dir, ignore_errors=True)
    os.mkdir(dir)


def genvisibility(fin, **kwargs):
    """Save the calibrated data arrays to npz file format"""
    fout = os.path.split(fin)[-1] + ".npz"
    r = hc.vis(fin, baseline=inBaselines, alist=inAntenna)
    np.savez(fout, **r)
    if not os.path.exists(fout):
        raise RuntimeError('No output produced by heracasa.closure.vis')
    return(fout)


def npz_conversion(uvin, rm_ms=True, verbose=conversion_verbose):
    """End to end conversion of raw visibility dataset to npz file format

    :param rm_ms: Remove MS format visibilities from working_directory
    :type rm_ms: bool
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

    logging.info('Reducing Miriad dataset {}'.format(uvin))
    ms = cv(uvin)
    logging.info('Miriad converted to MS file format')

    IDR_JD, IDR_time = ms.split('.')[1:3]
    gcflagdata(ms, IDR_JD)
    logging.info('{} flagged'.format(ms))

    fringerot(ms)
    logging.info('fringe rotated'.format(ms))

    npz = genvisibility(ms, baseline=inBaselines, alist=inAntenna)
    logging.info('Saved to {}\n'.format(npz))
    if rm_ms:
        shutil.rmtree(ms)
    return npz


def multiprocess_wrapper(files):
    for f in files:
        npz_conversion(f)


def main():
    if clean_dir:
        cleanspace(procdir)
    os.chdir(procdir)

    files_per_core = len(InData) / NUMBER_OF_CORES
    split_files = [InData[i:i+files_per_core]
                   for i in range(0, len(InData), files_per_core)]
    remainder_files = InData[files_per_core * NUMBER_OF_CORES:len(InData)]
    print('{} uv dataset(s) to reduce'.format(len(InData)))

    jobs = []
    for list_slice in split_files:
        j = multiprocessing.Process(
            target=multiprocess_wrapper, args=(list_slice,))
        jobs.append(j)

    for job in jobs:
        job.start()
        job.join()

    for file in remainder_files:
        npz_conversion(file)


if __name__ == "__main__":
    main()
