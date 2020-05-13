"""Main power spectrum analysis script

Example run:
$ python ps_analysis.py /Users/matyasmolnar/Downloads/HERA_Data/test_data/aligned_smaller.npz \
--last 3.31 --chan_bounds 150~250 --clip_bls --clip_days --clip_rule amp \
--ps_type complex --stat_dom vis

TODO:
    - Add baseline functionality for EW, NS, 14m, 28m, individual baselines etc
      (although this done before export to npz..?)
    - Functionality to deal with statistics on either amplitudes, or complex quantities
      (also real and imag separately)
    - Load npz file of single visibility dataset
    - Flag visibilities where more tha 50% of data in a given dimension is flagged
    - Interpolate data where channel completely flagged
    - Averaging both across baselines, or across visibilities
    - Cross power spectrum - do all permutations and average
"""


import argparse
import os
import textwrap

import numpy as np

from idr2_info import hera_resolution, idr2_jds
from ps_functions import cps, clipping, dim_statistic, ps
from ps_utils import IndexVis


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.\
    RawDescriptionHelpFormatter, description=textwrap.dedent("""
    Power spectrum computation of aligned visibilities, after sigma clipping and
    statistics in the day, baseline and LAST axes.
    """))
    parser.add_argument('aligned_vis', help='Path to aligned visibilities in npz \
                        file format', type=str, metavar='IN')
    parser.add_argument('-o', '--out_dir', required=False, default=None, \
                        metavar='O', type=str, help='Output directory')
    parser.add_argument('-l', '--last', required=True, metavar='L', \
                        type=float, help='LAST to analyze')
    parser.add_argument('-t', '--tints', required=False, default=1, \
                        metavar='T', type=int, help='Number of time integration \
                        bins, each of ~10s cadence, to run statistics on {1, 6}')
    parser.add_argument('-d', '--days', required=False, default='IDR2', \
                        metavar='D', type=str, help='Selected JD days')
    parser.add_argument('-c', '--chan_bounds', required=False, default=None, \
                        metavar='C', type=str, help='Bounds of frequency channels \
                        to calibrate e.g. 100~250 {0, 1023}')
    parser.add_argument('--last_stat', required=False, default='median', \
                        metavar='LS', type=str, help='Statistic to perform on \
                        LAST axis {"median", "mean"}')
    parser.add_argument('--day_stat', required=False, default='median', \
                        metavar='DS', type=str, help='Statistic to perform on \
                        day axis {"median", "mean"}')
    parser.add_argument('-cb', '--clip_bls', required=False, action='store_true', \
                        help='Perform sigma clipping on baselines axis')
    parser.add_argument('-cd', '--clip_days', required=False, action='store_true', \
                        help='Perform sigma clipping on days axis')
    parser.add_argument('--clip_rule', required=False, default='amp', \
                        metavar='CR', type=str, help='Method to apply sigma \
                        clipping: either according to amplitudes, geometric mean \
                         or real and imaginary parts of the visibilities \
                         separately {"amp", "gmean", "complex"}')
    parser.add_argument('--clip_stds', required=False, default=5.0, \
                        metavar='SS', type=float, help='Number of standard \
                        deviations to use for both the lower and upper clipping \
                        limits')
    parser.add_argument('--ps_type', required=False, default='complex', \
                        metavar='PT', type=str, help='Specify if the power \
                        spectrum computation should be on the visibility \
                        amplitudes, or the complex visibilities themselves \
                        {"amp", "complex"}')
    parser.add_argument('--stat_dom', required=False, default='vis', \
                        metavar='SD', type=str, help='Specify if statistics are \
                        to be done on either the visibilities or power spectra \
                        domain, corresponding to either coherent of decoherent \
                        averaging {"vis", "ps"}')
    parser.add_argument('-s', '--save_figs', required=False, action='store_true', \
                        help='Save figures')
    parser.add_argument('--cleandir', required=False, action='store_true', \
                        help='Remove all files in out_dir, only if specified')
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(args.aligned_vis)

    InJDs  = args.days
    if InJDs == 'IDR2':
        InJDs = idr2_jds

    chan_bounds = tuple(map(int, args.chan_bounds.split('~')))

    # Loading npz file of aligned visibilities
    vis_data = np.load(args.aligned_vis)
    index_vis = IndexVis(vis_data, InJDs, args.last, args.tints, chan_bounds)
    visibilities, baselines, last, days, flags, chans_range = index_vis.do_indexing()
    print('Zero values in visibilities array: {}'.format(0+0j in visibilities))

    # Continuing analysis with either complex visibilities or visibility amplitudes
    if args.ps_type == 'complex':
        return_onesided_ps = False
    elif args.ps_type == 'amp':
        return_onesided_ps = True
        visibilities = np.abs(visibilities)
    else:
        raise ValueError('ps_type must be either "complex" or "amp"')

    visibilities = np.ma.masked_array(visibilities, mask=flags)

    visibilities = clipping(visibilities, sig_clip_days=args.clip_days, \
        sig_clip_bls=args.clip_bls, sig_stds=args.clip_stds, cenfunc='median', \
        clip_rule=args.clip_rule)

    # Statistic on LAST axis
    visibilities = getattr(np.ma, args.last_stat)(visibilities, axis=0)

    # Splitting visibilities into 2 halves (for CPS between mean/median of two
    # halves) about the day axis
    day_halves = np.array_split(days, 2)
    vis_halves = np.array_split(visibilities, 2, axis=0)

    # Statistic on day axis
    vis_half1 = dim_statistic(vis_halves[0], args.day_stat, 0)
    vis_half2 = dim_statistic(vis_halves[1], args.day_stat, 0)
    visibilities = dim_statistic(visibilities, args.day_stat, 0)

    # Find baselines where all data points are flagged
    masked_bls = [bl_row for bl_row in range(visibilities.shape[0]) if \
                  visibilities[bl_row, :].mask.all() == True]
    masked_bls_indexing = np.delete(np.arange(len(baselines)), masked_bls)
    print('Baseline(s) {} removed from analysis as all their visibilities are \
    flagged in the channel range {} - {}'.format([list(row) for row in \
    baselines[masked_bls]], *chan_bounds))

    vis_half1 = vis_half1[masked_bls_indexing, :]
    vis_half2 = vis_half2[masked_bls_indexing, :]
    visibilities = visibilities[masked_bls_indexing, :]
    baselines = baselines[masked_bls_indexing]

    vis_cps = cps(vis_half1, vis_half2, hera_resolution, window='blackmanharris', \
        length=None, scaling='spectrum', detrend=False, \
        return_onesided=return_onesided_ps)

    np.savez(os.path.join(out_dir, 'cps.npz'), vis_cps)


if __name__ == "__main__":
    main()
