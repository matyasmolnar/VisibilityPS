"""Data reduction of visibilities to npz file format (with calibration)
"""


import argparse
import logging
import multiprocessing
import os
import shutil
import textwrap

from calibration_functions import cv, gcflagdata, genvisibility, fringerot,
mkinitmodel, kc_cal, bandpass_cal, dosplit, cleaninit, cleanfinal
from idr2_info import idr2_ants, idr2_bls
from vis_utils import get_data_paths, cleanspace


def npz_conversion(uvin, model, raw, verbose, keep_ms):
    """Calibration of visibility dataset with export to npz file format

    :param uvin: Visibility dataset in miriad file format path
    :type uvin: str
    :param model: J2000 coordinates of point source model or name of well-known
    radio object (e.g. "GC, FornaxA")
    :type model: str
    :param raw: Whether to *not* perform traditional calibration, hence reducing
    the raw visibilities instead. Calibration involves doing one round of gain and
    delay calibration, followed by bandpass calibration.
    :type cal: bool
    :param verbose: Verbose
    :type verbose: bool
    :param keep_ms: Keep MS format visibilities
    :type keep_ms: bool

    :return: Visibility data saved to NpzFile file
    :rtype: str
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

    if not raw:
        # TODO: only if source in FoV
        ms = fringerot(ms)
        logging.info('{} fringe rotated'.format(ms))

        model_cl = mkinitmodel(model)
        ms = kc_cal(ms, model)
        ms = dobandpass(ms)
        logging.info('{} gain, delay and bandpass calibrated rotated'.format(ms))

    npz = genvisibility(ms)
    logging.info('Saved to {}\n'.format(npz))
    if rm_ms:
        shutil.rmtree(ms)
    return npz


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.\
    RawDescriptionHelpFormatter, description=textwrap.dedent("""
    Data reduction steps for HERA IDR2 visibilities in miriad file format:

    1. Miriad visibilities are converted to measurement set (CASA) file format
    2. The visibilities are flagged in CASA:
        - Bad antennas removed
        - Band edges flagged
        - Autocorrelations flagged
    3. OPTIONAL: Visibilities are fringe rotated to the calibration point source.
       This step is for imaging purposes.
    4. CALIBRATION: If specified, the visibilities undergo traditional CASA
    calibratio: the gain, delay and bandpass calibration solutions are calculated
    from a point source model and applied. No CLEANing is performed.
    5. The visibilities are saved as an ndarray to an npz file
    """))
    parser.add_argument('data_dir', help='Directory of visibilities in miriad \
                        file format to reduce', type='str', metavar='IN')
    parser.add_argument('-o', '--out_dir', required=False, default=None, \
                        metavar='O', type=str, help='Output directory')
    parser.add_argument('-p', '--pol', required=True, metavar='pol', type=str, \
                        help='Polarization to calibrate {"xx", "xy", "yy", "yx"}')
    parser.add_argument('-d', '--days', required=False, default=None, metavar='D', \
                        type=list, help='Selected JD days')
    parser.add_argument('-t', '--times', required=False, default=None, metavar='T', \
                        type=list, help='Selected fractional times')
    parser.add_argument('-m', '--model', required=True, metavar='M', type=str, \
                        help='J2000 coordinates of point source model or name of \
                        well-known radio object (e.g. "GC, FornaxA")')
    parser.add_argument('-r', '--raw', required=False, action='store_true', \
                        help='Default is to calibrate visibilities. Specify \
                        this argument to keep reduce the raw visibilities instead')
    parser.add_argument('-k', '--keep_ms', required=False, action='store_true', \
                        help='Keep intermediate MS format visibilities in out_dir')
    parser.add_argument('-c', '--cleandir', required=False, action='store_true', \
                        help='Remove all files in out_dir, only if specified')
    parser.add_argument('-v', '--verbose', required=False, action='store_true', \
                        help='Check status of data reduction steps for each \
                        visibility dataset')
    args = parser.parse_args()

    if args.cleandir:
        cleanspace(args.out_dir)
    os.chdir(args.out_dir)

    InData = get_data_paths(args.data_dir, args.pol, days=args.days, times=args.times)
    print('{} uv dataset(s) to reduce'.format(len(InData)))

    processes = []
    for data in InData:
        p = multiprocessing.Process(target=npz_conversion, args=(data, args.model, \
                                    args.raw, args.verbose, args.keep_ms))
        processes.append(p)

    for p in processes:
        p.start()
        p.join()


if __name__ == "__main__":
    main()
