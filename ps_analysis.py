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

from ps_functions import cps, clipping, dim_statistic, ps
from ps_plotting import baseline_vis_analysis, baseline_ps_analysis, \
plot_single_bl_vis, plot_stat_vis,


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

chan_start = 100
chan_end = 250

fig_path = '../test_data/'
savefigs = False

#############################################################


if InJDs == 'IDR2':
    InJDs = idr2_jds


if LAST < np.min(last) or LAST > np.max(last):
    raise ValueError('Specify LAST value in between {} and {}'.format(
        np.ceil(np.min(vis_data['last'])*100)/100, np.floor(np.max(vis_data['last'])*100)/100))

if chan_start < chans[0] or chan_end > chans[-1] or chan_start > chan_end:
    raise ValueError('Specify channels in between {} and {} with chan_start > \
                     chan_end'.format(chans[0], chans[-1]))


def main():

    DataDir = '/Users/matyasmolnar/Downloads/HERA_Data/test_data'
    aligned_vis = 'aligned_smaller.npz'

    # Loading npz file of aligned visibilities
    # Has items ['visibilities', 'baselines', 'last', 'days', 'flags']
    vis_data = np.load(os.path.join(DataDir, aligned_vis))
    visibilities = vis_data['visibilities'] # shape: (last bins, days, bls, chans)
    baselines = vis_data['baselines'] # shape: (bl, [ant1, ant2])
    last = vis_data['last'] # shape: (last bins, days)
    days = vis_data['days'].astype(int) # JD days
    flags = vis_data['flags'].astype(bool) # shape same dimensions as visibilities


    # all in MHz, for HERA H1C_IDR2
    bandwidth_start = 1.0e8
    bandwidth_end = 2.0e8
    resolution = 97656.25

    chans = np.arange(1024, dtype=int)
    chans_range = np.arange(chan_start-1, chan_end)
    freqs = np.arange(bandwidth_start, bandwidth_end, resolution)
    freqs_range = freqs[chans_range]


    # Removing days that do not appear in InJDs or that are misaligned
    misaligned_days = find_mis_days(last, days)
    flt_days = [day for day in list(set(InJDs) & set(days)) if day not in misaligned_days]
    flt_days_indexing = [np.where(days == flt_day)[0][0] for flt_day in flt_days]


    # Removing bad baselines
    bad_bls = [[50, 51]] # [66, 67], [67, 68], [68, 69],[82, 83], [83, 84], [122, 123]]
    bad_bls_idxs = [np.where(vis_data['baselines'] == bad_bl) for bad_bl in \
                    bad_bls if bad_bl in vis_data['baselines']]
    bad_bls_idxs = [bad_bls_idx[0][0] for bad_bls_idx in bad_bls_idxs if \
                    bad_bls_idx[0].size==2 and len(set(bad_bls_idxs[0][0]))==1]
    bl_indexing = [bl_idx for bl_idx in range(vis_data['baselines'].shape[0]) if \
                   bl_idx not in bad_bls_idxs]


    # Selecting desired LASTs
    last_idx = find_nearest(np.median(last, axis=1), LAST)[1]
    last_indexing = sorted(nsmallest(last_tints, np.arange(last.shape[0]), \
                                     key=lambda x: np.abs(x - last_idx)))


    # Indexing out bad baselines and only selecting specified channels
    vis_indexing = np.ix_(last_indexing, flt_days_indexing, bl_indexing, chans_range)
    visibilities = visibilities[vis_indexing]
    flags = flags[vis_indexing]
    baselines = baselines[bl_indexing, :]
    last = last[last_indexing, flt_days_indexing]
    days = flt_days
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
    baselines[masked_bls]], chan_start, chan_end))

    vis_half1 = vis_half1[masked_bls_indexing, :]
    vis_half2 = vis_half2[masked_bls_indexing, :]
    visibilities = visibilities[masked_bls_indexing, :]
    baselines = baselines[masked_bls_indexing]

    vic_cps = cps(vis_half1, vis_half2, window='blackmanharris', length=None, \
        scaling='spectrum', detrend=False, return_onesided=return_onesided_ps)


if __name__ == "__main__":
    main()
