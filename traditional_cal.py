"""Batch traditional calibration of visibilities

Traditional calibration in CASA of visibility datasets, with no multiprocessing.

example run:
$ python traditional_cal.py /Users/matyasmolnar/Downloads/HERA_Data/test_data \
--pol 'xx' --model 'FornaxA' --verbose
"""


import argparse
import logging
import os
import textwrap

from calibration_functions import cv, gcflagdata, fringerot, mkinitmodel, kc_cal, \
bandpass_cal, dosplit, cleaninit, cleanfinal
from vis_utils import get_data_paths, cleanspace


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.\
    RawDescriptionHelpFormatter, description=textwrap.dedent("""
    Traditional calibration in CASA of visibility datasets.

    Calibration steps for HERA IDR2 visibilities in Miriad file format:
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
    """))
    parser.add_argument('data_dir', help='Directory of visibilities in miriad \
                        file format to reduce', type='str', metavar='IN')
    parser.add_argument('-o', '--out_dir', required=False, default=None, \
                        metavar='O', type=str, help='Output directory')
    parser.add_argument('-p', '--pol', required=True, metavar='pol', type=str, \
                        help='Polarization to calibrate {"xx", "xy", "yy", "yx"}')
    parser.add_argument('-m', '--model', required=True, metavar='M', type=str, \
                        help='J2000 coordinates of point source model or name of \
                        well-known radio object (e.g. "GC, FornaxA")')
    parser.add_argument('-d', '--days', required=False, default=None, metavar='D', \
                        type=list, help='Selected JD days')
    parser.add_argument('-t', '--times', required=False, default=None, metavar='T', \
                        type=list, help='Selected fractional times')
    parser.add_argument('-c', '--cleandir', required=False, action='store_true', \
                        help='Remove all files in out_dir, only if specified')
    parser.add_argument('-v', '--verbose', required=False, action='store_true', \
                        help='Check status of data reduction steps for each \
                        visibility dataset')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

    if args.cleandir:
        cleanspace(procdir)
    os.chdir(procdir)

    InData = get_data_paths(args.data_dir, args.pol, days=args.days, times=args.times)
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
    # ms = [fringerot(ms_file) for ms_file in ms]

    cal_source = args.model

    model_cl = mkinitmodel(cal_source)
    ms = [kc_cal(ms_file, model_cl) for ms_file in ms]
    ms = [bandpass_cal(ms_file) for ms_file in ms]
    calsi = [dosplit(ms_file, 'ical') for ms_file in ms]
    _ = [cleaninit(calsi_file, cal_source) for calsi_file in calsi]
    cals1 = [bandpass_cal(calsi_file) for calsi_file in calsi]
    _ = [cleanfinal(cals1_file, cal_source) for cals1_file in cals1]


if __name__ == "__main__":
    main()
