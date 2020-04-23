"""Main power spectrum analysis script

TODO:
    - Add baseline functionality for EW, NS, 14m, 28m, individual baselines etc
      (although this done before export to npz..?)
    - Functionality to deal with statistics on either amplitudes, or complex quantities
      (also real and imag separately)
    - Load npz file of single visibility dataset
"""


import numpy as np


from ps_functions import plot_stat_vis, plot_single_bl_vis, baseline_vis_analysis, \
baseline_ps_analysis


#############################################################
####### Modify the inputs in this section as required #######
#############################################################

# run statistics on either the visibilities or power spectra
# either coherent of decoherent averaging
stat_vis_or_ps = 'ps' # {vis, ps}

# if visibility statistics: True - deal with the complex quantity
#                         : False - or the amplitudes for the statistics
ps_complex_analysis = True

# LAST to analyze
LAST = 3.31

# statistic in time between consecutive sessions - if False, closest time bin
# to LAST will be chosen
last_tints = 1
last_statistic_method = 'median'  # {median, mean}

# run statistics over all IDR2 Days? if False, fill in InJDs
statistic_all_IDR2 = False

InJDs = [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, 2458106,
         2458107, 2458108, 2458110, 2458111, 2458112, 2458113, 2458116]

days_statistic = 'median'  # {median, mean}

clip_rule = 'amp'  # sigma clip according to amplitudes, gmean or Re and Im parts
# of the complex visibilities separately?

sig_clip_days = True
sig_clip_bls = True
sig_stds = 5.0

chan_start = 100
chan_end = 250

fig_path = '/rds/project/bn204/rds-bn204-asterics/mdm49/'
savefigs = False

#############################################################


if statistic_all_IDR2:
    InJDs = idr2_jds

if ps_complex_analysis:
    fig_suffix = 'complex'
else:
    ps_complex_analysis = 'amplitude'

if stat_vis_or_ps == 'vis':
    stat_on_vis = True
    stat_on_ps = False
elif stat_vis_or_ps == 'ps':
    stat_on_vis = False
    stat_on_ps = True
else:
    raise ValueError('Choose to perform the statistics on either the \
                      visibibilities or power spectra')


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

    chans = np.arange(0, 1024, dtype=int)
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


    vis_amps_clipped = clipping(
        data=vis_analysis, cenfunc='median', clip_rule=clip_rule)

    # after clipping
    plot_stat_vis(np.ma.masked_array(np.absolute(vis_amps_clipped.data),
                  vis_amps_clipped.mask), 'mean', 'median', clipped=True, last_avg=False)

    # statistics on visibilities before PS computation
    day_halves = np.array_split(days_dayflg, 2)
    if stat_on_vis:
        if last_statistic:
            vis_amps_clipped_lastavg = getattr(np.ma, last_statistic_method)\
                                       (vis_amps_clipped, axis=0)
            # after LAST averaging
            plot_stat_vis(np.absolute(vis_amps_clipped_lastavg),
                          last_statistic_method, clipped=True, last_avg=True)

            # splitting array of days into 2 (for cross power spectrum between
            # mean/median of two halves) split visibilities array into two sub-arrays
            # of (near) equal days
            vis_halves = np.array_split(vis_amps_clipped_lastavg, 2, axis=0)

            vis_half1_preflag = day_statistic(vis_halves[0])
            vis_half2_preflag = day_statistic(vis_halves[1])
            vis_whole_preflag = day_statistic(vis_amps_clipped_lastavg)

            # find out where data is flagged from clipping - if statistic applied on
            # days, still get gaps in the visibilities
            # list(set(np.where(day_statistic(vis_halves[1]).mask == True)[0]))
            # finds flagged baselines
            baselines_clip_faulty = list(
                set(np.where(day_statistic(vis_amps_clipped_lastavg).mask == True)[0]))
            baselines_clip_flag = np.delete(
                np.arange(len(baselines_dayflg_blflg)), baselines_clip_faulty)
            print('Baseline(s) {} removed from analysis due to flagging from sigma \
            clipping - there are gaps in visibilities for these baselines for the \
            channel range {} - {}'.format(baselines_dayflg_blflg[baselines_clip_faulty], \
            chan_start, chan_end))

            vis_half1 = vis_half1_preflag[baselines_clip_flag, :]
            vis_half2 = vis_half2_preflag[baselines_clip_flag, :]
            vis_amps_final = vis_whole_preflag[baselines_clip_flag, :]
            baselines_dayflg_blflg_clipflg = baselines_dayflg_blflg[baselines_clip_flag]

            # window functions: boxcar (equivalent to no window at all), triang,
            # blackman, hamming, hann, bartlett, flattop, parzen, bohman,
            # blackmanharris, nuttall, barthann, kaiser
            vis_ps_final = power_spectrum(vis_half1, vis_half2, window='boxcar', \
            length=vis_half1.shape[1], scaling='spectrum', detrend=False, \
            return_onesided=return_onesided_ps)
            # vis_psd_final = power_spectrum(vis_half1, vis_half2, window='boxcar', \
            # length=vis_half1.shape[1], scaling='density', detrend=False,
            # return_onesided=False)


    else:
        vis_amps_clipped_lastavg = getattr(np.ma, last_statistic_method)\
                                   (vis_amps_clipped, axis=0)
        vis_whole_preflag = day_statistic(vis_amps_clipped_lastavg)
        baselines_clip_faulty = list(
            set(np.where(day_statistic(vis_amps_clipped_lastavg).mask == True)[0]))
        baselines_clip_flag = np.delete(
            np.arange(len(baselines_dayflg_blflg)), baselines_clip_faulty)
        baselines_dayflg_blflg_clipflg = baselines_dayflg_blflg[baselines_clip_flag]
        print('Baseline(s) {} removed from analysis due to flagging from sigma \
        clipping - there are gaps in visibilities for these baselines for the \
        channel range {} - {}'.format(baselines_dayflg_blflg[baselines_clip_faulty], \
        chan_start, chan_end))
        vis_amps_final = vis_whole_preflag[baselines_clip_flag, :]

        vis_amps_clipped_blflag = vis_amps_clipped[:, :, baselines_clip_flag, :]
        for time in range(vis_amps_clipped_blflag.shape[0]):
            for day in range(vis_amps_clipped_blflag.shape[1]):
                for bl in range(vis_amps_clipped_blflag.shape[2]):
                    if all(vis_amps_clipped_blflag[time, day, bl, :].mask == True):
                        print(str(time) + ' ' + str(day) + ' ' + str(bl))
                        print('-------')
        # use these to mask the power spectra before averaging
        # also create a mask for visibilities where more tha 50% of data is flagged - how to?
        # then for others need to interpolate where mask == True

        # list(set(np.where(vis_amps_clipped_blflag[].mask == True)[0]))

        ###############
        # here must verify that there are no gaps in visibilities
        ###############
        # checking where the masks are in the visibility hypercube
        # list(set(np.where(vis_amps_clipped_blflag.mask == True)[0]))
        # list(set(np.where(vis_amps_clipped_blflag.mask == True)[1]))
        # list(set(np.where(vis_amps_clipped_blflag.mask == True)[2]))
        # list(set(np.where(vis_amps_clipped_blflag.mask == True)[3]))

        # split visibilities array into two sub-arrays of (near) equal days
        vis_halves = np.array_split(vis_amps_clipped_blflag, 2, axis=1)
        # not running day statistics on visibilities
        # not removing any baselines at this point - unless some have gaps in the
        # visibilities across the chan_range would then have to apply Mark's psd_estimation
        vis_ps_raw = power_spectrum(vis_halves[0], vis_halves[1], window='boxcar', \
            length=vis_halves[0].shape[-1], scaling='spectrum', detrend=False, \
            return_onesided=return_onesided_ps)  # spectrum
        # now run statistics - run this on complex ps or on magnitude?
        if ps_complex_analysis:
            vis_ps_raw_analysis = vis_ps_raw
        else:
            vis_ps_raw_analysis = np.absolute(vis_ps_raw)
        # statistic on last bins
        vis_ps_raw_lastavg = getattr(np.ma, last_statistic_method)\
                             (vis_ps_raw_analysis, axis=0)
        vis_ps_final = day_statistic(vis_ps_raw_lastavg)
        # need to check data is smooth - no gaps


    # Choosing single day
    # if len(InJDs) == 1:
    #     vis_amps_single_day = vis_amps_clipped_lastavg[np.where(days_dayflg == IDRDay)[0][0] ,:,:]


    # plot_stat_vis(np.absolute(vis_analysis), 'mean', 'median', clipped=False,
    #               last_avg=False, figname=os.path.join(fig_path, \
    #               'mean_and_median_vis_{}.pdf'.format(fig_suffix)))
    #
    # plot_single_bl_vis(data=np.absolute(vis_chans_range), time=LAST, JD_day=2458101, \
    #                    bl=[82, 83])
    #
    # baseline_vis_analysis(np.ma.masked_array(np.absolute(vis_amps_final.data), \
    # mask=vis_amps_final.mask, dtype=float))
    #
    # baseline_ps_analysis(vis_ps_final)


if __name__ == "__main__":
    main()
