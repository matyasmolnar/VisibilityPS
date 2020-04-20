"""Data reduction of visibilities to npz file format (with calibration)

Data reduction steps for HERA IDR2 visibilities in miriad file format:
    1. Miriad visibilities are converted to measurement set (CASA) file format
    2. The visibilities are flagged in CASA:
        - Bad antennas removed
        - Band edges flagged
        - Autocorrelations flagged
    3. OPTIONAL: Visibilities are fringe rotated to the calibration point source.
       This step is for imaging purposes.
    4. The gain, delay and bandpass calibration solutions are calculated from a
       point source model and applied. No CLEANing is performed.
    5. The visibilities are saved as an ndarray to an npz file
"""


import logging
import multiprocessing
import os
import shutil

from calibration_functions import cv, gcflagdata, fringerot, mkinitmodel, kc_cal, \
bandpass_cal, dosplit, cleaninit, cleanfinal
from vis_utils import get_data_paths, cleanspace


#############################################################
####### Modify the inputs in this section as required #######
#############################################################

# Directory of visibilities in miriad file format to reduce
DataDir = "/rds/project/bn204/rds-bn204-asterics/HERA/data"
# Output directory
procdir = "/rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_raw"

Pol = 'xx' # Polarization of visibilities to process
InDays = 2458098 # Select days to process
InTimes = [] # OPTIONAL: select times e.g. [12552]

# Remove all files in procdir
clean_dir = True

# Check status of data reduction steps for each visibility dataset
conversion_verbose = True

# Multiprocessing specification
NUMBER_OF_CORES = len(IDRDays)

#############################################################


def npz_conversion(uvin, rm_ms=True, verbose=conversion_verbose):
    """Calibration of visibility dataset with export to npz file format

    Only doing one round of calibration (gain and delay, followed by bandpass)

    :param rm_ms: Remove ms format visibilities from procdir
    :type rm_ms: bool
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

    logging.info('Reducing dataset {}'.format(uvin))
    if os.path.basename(dataset).endswith('.uv'):
        ms = cv(uvin)
        logging.info('Miriad  dataset converted to ms file format')
    else:
        ms = uvin

    IDR_JD, IDR_time = ms.split('.')[1:3]
    gcflagdata(ms, 'IDR2', cut_edges=True, bad_chans=None)
    logging.info('{} flagged'.format(ms))

    # TODO: only if source in FoV
    ms = fringerot(ms)
    logging.info('{} fringe rotated'.format(ms))

    model_cl = mkinitmodel(cal_source)
    ms = kc_cal(ms, model_cl)
    ms = dobandpass(ms)
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

    InData = get_data_paths(DataDir, Pol, InDays, InTimes)
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