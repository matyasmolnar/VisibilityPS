"""Main power spectrum analysis script

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


import numpy as np

from ps_functions import cps, clipping, ps
from ps_utils import hera_resolution, IndexVis


#############################################################
####### Modify the inputs in this section as required #######
#############################################################

# Statistics on either the visibilities or power spectra domain, corresponding
# to either coherent of decoherent averaging
stat_domain = 'ps' # {'vis', 'ps'}

# If the power spectrum computation should be on the visibility amplitudes, or
# the complex visibilities themselves
ps_intype = 'amp' # {'amp', 'complex'}

# Selected LAST
LAST = 3.31

# Number of time integration bins, each of ~10s cadence, to run statistics on
last_tints = 1 # {1:6}
last_stat = 'median'  # {median, mean}
day_stat = 'median' # {median, mean}

InJDs = [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, 2458106,
         2458107, 2458108, 2458110, 2458111, 2458112, 2458113, 2458116]

clip_rule = 'amp'  # sigma clip according to amplitudes, gmean or Re and Im parts
# of the complex visibilities separately?

sig_clip_days = True
sig_clip_bls = True
sig_stds = 5.0

chan_bounds =  (100, 250)

fig_path = '../test_data/'
savefigs = False

#############################################################


if InJDs == 'IDR2':
    InJDs = idr2_jds


def main():

    DataDir = '/Users/matyasmolnar/Downloads/HERA_Data/test_data'
    aligned_vis = 'aligned_smaller.npz'

    # Loading npz file of aligned visibilities
    vis_data = np.load(os.path.join(DataDir, aligned_vis))
    index_vis = IndexVis(vis_data, InJDs, LAST, last_tints, chan_bounds)
    visibilities, baselines, last, days, flags, chans_range = index_vis.do_indexing()
    print('Zero values in visibilities array: {}'.format(0+0j in visibilities))


    # continuing analysis in either complex visibilities or visibility amplitudes
    if ps_complex_analysis:
        return_onesided_ps = False
    else:
        return_onesided_ps = True
        visibilities = np.absolute(visibilities)

    visibilities = np.ma.masked_array(visibilities, mask=flags)


    visibilities = clipping(visibilities, sig_clip_days=True, \
        sig_clip_bls=True, sig_stds=5.0, cenfunc='median', clip_rule='amp')

    # Statistic on LAST axis
    visibilities = getattr(np.ma, last_stat)(visibilities, axis=0)

    # Splitting visibilities into 2 halves (for CPS between mean/median of two
    # halves) about the day axis
    day_halves = np.array_split(days, 2)
    vis_halves = np.array_split(visibilities, 2, axis=0)

    # Statistic on day axis
    vis_half1 = dim_statistic(vis_halves[0], day_stat, 0)
    vis_half2 = dim_statistic(vis_halves[1], day_stat, 0)
    visibilities = dim_statistic(visibilities, day_stat, 0)

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

    vic_cps = cps(vis_half1, vis_half2, resolution, window='blackmanharris', \
        length=None, scaling='spectrum', detrend=False, return_onesided=return_onesided_ps)


if __name__ == "__main__":
    main()
