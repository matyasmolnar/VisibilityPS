"""HERA Visibility PS Computation Script

This script takes aligned HERA visibilities in LAST (as outputted by
align_lst.py) and computes various PS estimates using simple statistics over LASTs,
days and baselines.

To use on CSD3 must have the astroenv conda environment activated:
$ module load miniconda3-4.5.4-gcc-5.4.0-hivczbz
$ source activate astroenv

TODO:
    - Add baseline functionality for EW, NS, 14m, 28m, individual baselines etc
      (although this done before export to npz..?)
    - Functionality to deal with statistics on either amplitudes, or complex quantities
      (also real and imag separately)
    - Load npz file of single visibility dataset
"""


import functools
import os
from heapq import nsmallest

import astropy.stats
import math
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.stats import median_absolute_deviation as mad
from scipy.stats.mstats import gmean

from idr2_info import idr2_jds
from vis_utils import find_nearest


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

amp_clip = True  # sigma clip according to amplitudes? if vis_complex_analysis,
# will still mask according to behaviour of amplitudes

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

if vis_complex_analysis:
    fig_suffix = 'complex'
else:
    fig_suffix = 'amplitude'

if stat_vis_or_ps == 'vis':
    stat_on_vis = True
    stat_on_ps = False
elif stat_vis_or_ps == 'ps':
    stat_on_vis = False
    stat_on_ps = True
else:
    raise ValueError('Choose to perform the statistics on either the \
                      visibibilities or power spectra')


# Loading npz file of aligned visibilities
# Has items ['visibilities', 'baselines', 'last', 'days', 'flags']
vis_data = np.load('/rds/project/bn204/rds-bn204-asterics/mdm49/aligned/aligned_visibilities.npz')
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


if LAST < np.min(last) or LAST > np.trunc(np.max(last)*100)/100:
    raise ValueError('Specify LAST value in between {} and {}'.format(round(np.min(last), 2), \
                     np.trunc(np.max(last)*100)/100)))

if chan_start < chans[0] or chan_end > chans[-1] or chan_start > chan_end:
    raise ValueError('Specify channels in between {} and {} with chan_start > \
                     chan_end'.format(chans[0], chans[-1]))


# Checking alignment of LASTs. Issue with missing consecutive sessions. Need to
# check calibration and/or alignment script
def mod_zscore(arr):
    """Modified z-score, as defined by Iglewicz and Hoaglin"""
    return 0.6745*(arr - np.median(arr))/mad(arr)


def find_mis_days(last, mod_z_tresh=3.5):
    """Find misaligned days by computing the modified z-scores of the first and
    last LAST timestamps, and discarding any days that have a modified z-score
    that exceeds the treshold"""
    start_last, end_last = zip(*[(last[0, i], last[-1, i]) for i in \
                           range(last.shape[1])])
    start_zscores = np.where(mod_zscore(start_last) > mod_z_tresh)[0]
    end_zscores = np.where(mod_zscore(start_last) > mod_z_tresh)[0]
    mis_days_idx = list(set(start_zscores & end_zscores))
    mis_days = days[mis_days_idx]
    if mis_days.size:
        print('Misaligned days: {} - check alignment'.format(mis_days))
    return mis_days


# Removing days that do not appear in InJDs or that are misaligned
misaligned_days = find_mis_days(last)
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


return_onesided_ps = False
# continuing analysis in either complex visibilities or visibility amplitudes
if ps_complex_analysis:
    visibilities = np.absolute(visibilities)
    return_onesided_ps = True


def sig_clip(ma_vis, clip_dim='bls', cenfunc='median', sigma=sig_stds, clip_rule='amp'):
    """Sigma clipping of visibilities over given dimension, with clipping
    done about the mean or median value for the data

    Can perform sigma clipping on the visibilities according to either:
        - Their absolute values
        - Their geometric means
        - Their Re and Im values separately
    """
    clip_dim_dict = {'days': [2, 1], 'bls': [1, 2]}
    clip_rule_dict = {'amp': np.absolute, 'gmean': gmean}
    clip_dim = clip_dim_dict[clip_dim]
    old_axes = [0, 1, 2, 3]
    new_axes = [0, 3, *clip_dict[clip_dim]]
    ma_vis = np.moveaxis(ma_vis, old_axes, new_ax)
    days_mask = np.zeros_like(ma_vis.data)
    # iterate over last, freqs and {days, bls}
    for iter_dims in np.ndindex(ma_vis.shape[:3]):
        iter_data = ma_vis.data[iter_dims]
        # mask data array according to the clipping rule
        if clip_rule:
            iter_data = clip_dim_dict[clip_rule](iter_data)
        clip = astropy.stats.sigma_clip(iter_data, sigma=sigma, \
                maxiters=None, cenfunc=cenfunc, stdfunc='std', \
                masked=True, return_bounds=False)
        days_mask[iter_dims] = clip.mask
    print('Day clips: {}'.format(np.where(days_mask == True)[0].size))
    clipped_vis = np.ma.masked_where(days_mask, ma_vis)
    clipped_vis = np.moveaxis(ma_vis, new_axes, old_axes)
    return clipped_vis


def clipping(ma_vis, cenfunc='median', amp_clip=amp_clip):
    """Apply clipping"""
    no_day_clip = 0
    if sig_clip_days:
        visibilities = sig_clip(ma_vis, clip_dim='days', cenfunc=cenfunc, \
                                sigma=sig_stds, amp_clip=amp_clip)
        no_day_clip = np.sum(ma_vis.mask)
        print('Day clipping: {} visibilities masked'.format(no_day_clip))

    if sig_clip_bls:
        visibilities = sig_clip(ma_vis, clip_dim='bls', cenfunc=cenfunc, \
                                sigma=sig_stds, amp_clip=amp_clip)
        no_bl_clip = np.sum(ma_vis.mask) - no_day_clip
        print('Baseline clipping has been applied: {} visibilities masked'.\
              format(no_bl_clip))

    if sig_clip_days or sig_clip_bls:
        print('{} visibilities clipped out of {}'.format(np.sum(ma_vis.mask), \
              ma_vis.size))
    else:
        print('No clipping applied')
    return data


def dim_statistic(ma_vis, statistic, stat_dim):
    """Statistic over dimension

    :param ma_vis: Masked array of visibilities
    :type ma_vis: numpy MaskedArray
    :param stat_dim: Dimension on which to apply statistic. Can either be int,
    which represents axis, or str which represents the meaning of the dimension
    :type stat_dim: int, str

    :return: Statistic of visibilities over the specified dimension
    :rtype: numpy MaskedArray
    """
    if not isinstance(stat_dim, int):
        stat_dim = dim_dict[stat_dim]
    vis_stat = getattr(np.ma, statistic)(ma_vis, axis=stat_dim)
    return vis_stat


def power_spectrum(data1, data2=None, window='hann', length=None, \
                   scaling='spectrum', detrend=False, return_onesided=False):
    # CPS
    if data2 is not None:
        if stat_on_vis:
            # Finding dimension of returned delays
            delay_test, Pxy_spec_test = signal.csd(data1[0, :], data2[0, :], \
                fs=1./resolution, window=window, scaling=scaling, nperseg=length, \
                detrend=detrend, return_onesided=return_onesided)
            # how many data points the signal.periodogram calculates
            delayshape = delay_test.shape[0]
            # dimensions are [baselines, (delay, Pxx_spec), delayshape]
            vis_ps = np.zeros((data1.shape[0], 2, delayshape), dtype=complex)
            for i in range(0, data1.shape[0]):  # Iterating over all baselines
                delay, Pxy_spec = signal.csd(data1[i, :], data2[i, :], \
                    fs=1./resolution, window=window, scaling=scaling, \
                    nperseg=length, detrend=detrend, return_onesided=return_onesided)
                # Pxy_spec = np.absolute(Pxy_spec)
                vis_ps[i, :, :] = np.array(
                    [delay[np.argsort(delay)], Pxy_spec[np.argsort(delay)]])
        else:
            # statistics will be done later in the power spectrum domain
            delay_test, Pxy_spec_test = signal.csd(data1[0, 0, 0, :], \
                data2[0, 0, 0, :], fs=1./resolution, window=window, \
                scaling=scaling, nperseg=length, detrend=detrend, \
                return_onesided=return_onesided)
            # how many data points the signal.periodogram calculates
            delayshape = delay_test.shape[0]
            # dimensions are [baselines, (delay, Pxx_spec), delayshape]
            vis_ps = np.zeros((data1.shape[0], np.minimum(
                data1.shape[1], data2.shape[1]), data1.shape[2], 2, delayshape), \
                    dtype=complex)
            for time in range(vis_ps.shape[0]):  # Iterating over all last bins
                for day in range(vis_ps.shape[1]):
                    for bl in range(vis_ps.shape[2]):
                        delay, Pxy_spec = signal.csd(data1[time, day, bl, :], \
                            data2[time, day, bl, :], fs=1./resolution, window=window, \
                            scaling=scaling, nperseg=length, detrend=detrend, \
                            return_onesided=return_onesided)
                        # Pxy_spec = np.absolute(Pxy_spec)
                        vis_ps[time, day, bl, :, :] = np.array(
                            [delay[np.argsort(delay)], Pxy_spec[np.argsort(delay)]])
    # PS
    else:
        if stat_on_vis:
            # Finding dimension of returned delays
            delay_test, Pxx_spec_test = signal.periodogram(
                data1[1, :], fs=1./resolution, window=window, scaling=scaling, \
                    nfft=length, detrend=detrend)
            # how many data points the signal.periodogram calculates
            delayshape = delay_test.shape[0]
            # dimensions are [baselines, (delay, Pxx_spec), delayshape]
            vis_ps = np.zeros((data1.shape[0], 2, delayshape), dtype=complex)
            for i in range(data1.shape[0]):  # Iterating over all baselines
                delay, Pxx_spec = signal.periodogram(
                    data1[i, :], fs=1./resolution, window=window, scaling=scaling, \
                        nfft=length, detrend=detrend)
                vis_ps[i, :, :] = [delay, Pxx_spec]
                # equivalent to:
                # vis_ps[i,0,:] = delay
                # vis_ps[i,1,:] = Pxx_spec
    return vis_ps


vis_amps_clipped = clipping(
    data=vis_analysis, cenfunc='median', amp_clip=amp_clip)

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

    list(set(np.where(vis_amps_clipped_blflag[].mask == True)[0]))

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
