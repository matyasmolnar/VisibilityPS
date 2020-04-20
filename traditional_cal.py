"""Batch traditional calibration of visibilities

Traditional calibration in CASA of visibility datasets, with no multiprocessing.

Calibration steps for HERA IDR2 visibilities in miriad file format:
    1. Miriad visibilities are converted to measurement set (CASA) file format
    2. The visibilities are flagged in CASA:
        - Bad antennas removed
        - Band edges flagged
        - Autocorrelations flagged
    3. OPTIONAL: Visibilities are fringe rotated to the calibration point source.
       This step is for imaging purposes.
    4. A point source model for the calibrator is created
    5. Gain and delay calibration solutions are calculated with the model and applied
    6. Bandpass calibration solutions are calculated and applied
    7. A first round of CLEANing is done
    8. Bandpass calibration is done for a second time
    9. A second round of CLEANing is done, with images also produced at this stage
"""


import logging
import os

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

#############################################################


# Main script for batch traditional calibration
def main():

    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

    if clean_dir:
        cleanspace(procdir)
    os.chdir(procdir)

    InData = get_data_paths(DataDir, Pol, InDays, InTimes)
    data_files = [os.path.basename(dataset) for dataset in InData]
    logging.info('Datasets to calibrate: {}'.format(uv_files))

    # Only convert miriad datasets
    ms = [cv(dataset) if os.path.basename(dataset).endswith('.uv') else dataset
          for dataset in InData]
    if any('.uv' in data_file for data_file in data_files):
        logging.info('Converted miriad datasets to ms file format')

    for ms_file in ms:
        IDR_JD, IDR_time = ms_file.split('.')[1:3]
        gcflagdata(ms_file, 'IDR2', cut_edges=True, bad_chans=None)
        logging.info('{} flagged'.format(ms_file))

    # TODO: only if source in FoV
    ms = [fringerot(ms_file) for ms_file in ms]

    model_cl = mkinitmodel(cal_source)
    ms = [kc_cal(ms_file, model_cl) for ms_file in ms]
    ms = [bandpass_cal(ms_file) for ms_file in ms]
    calsi = [dosplit(ms_file, 'ical') for ms_file in ms]
    _ = [cleaninit(calsi_file, cal_source) for calsi_file in calsi]
    cals1 = [bandpass_cal(calsi_file) for calsi_file in calsi]
    _ = [cleanfinal(cals1_file, cal_source) for cals1_file in cals1]


if __name__ == "__main__":
    main()