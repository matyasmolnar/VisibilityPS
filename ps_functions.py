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

import astropy.stats
import math
import matplotlib
import numpy as np
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import signal
from scipy import fftpack as sf
from scipy.stats import median_absolute_deviation as mad

from idr2_info import idr2_jds
from vis_utils import find_nearest

sns.set()
sns.set_style('whitegrid')


#############################################################
####### Modify the inputs in this section as required #######
#############################################################

# Inputs:

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
last_statistic = True
last_statistic_period = 40  # seconds; choose between {20,60}
last_statistic_method = 'median'  # {median, mean}

# run statistics over all IDR2 Days? if False, fill in InJDs
statistic_all_IDR2 = False

InJDs = [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, 2458106,
         2458107, 2458108, 2458110, 2458111, 2458112, 2458113, 2458116]

days_statistic = 'median'  # {median, mean}

amp_clip = True  # sigma clip according to amplitudes? if vis_complex_analysis,
# will still mask according to behaviour of amplitudes

sigma_clip_days = True
sc_days_stds = 5.0

sigma_clip_bls = True
sc_bls_stds = 5.0

channel_start = 100
channel_end = 250

fig_path = '/rds/project/bn204/rds-bn204-asterics/mdm49/'
savefigs = False

#############################################################


if statistic_all_IDR2:
    InJDs = idr2_jds

if last_statistic_period < 20 or last_statistic_period > 60:
    raise ValueError('Choose LAST averaging period to be between 20 and 60 seconds')


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
chans_range = np.arange(channel_start-1, channel_end)
freqs = np.arange(bandwidth_start, bandwidth_end, resolution)
freqs_range = freqs[chans_range]


if LAST < np.min(last) or LAST > np.trunc(np.max(last)*100)/100:
    raise ValueError('Specify LAST value in between {} and {}'.format(round(np.min(last), 2), \
                     np.trunc(np.max(last)*100)/100)))

if channel_start < chans[0] or channel_end > chans[-1] or channel_start > channel_end:
    raise ValueError('Specify channels in between {} and {} with channel_start > \
                     channel_end'.format(chans[0], chans[-1]))


# Checking alignment of LASTs. Issue with missing consecutive sessions. Need to
# check calibration...
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


# Removing days that do not appear in InJDs and that are not misaligned
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


# Indexing out bad baselines and only selecting specified channels
vis_indexing = np.ix_(np.arange(visibilities.shape[0]), flt_days_idx, \
                      bl_flag, chans_range)
visibilities = visibilities[vis_indexing]
flags = flags[vis_indexing]
baselines = baselines[bl_indexing, :]
last = last[:, flt_days_indexing]
days = flt_days


# parameter for getting correct grouping of LASTs from last_statistic_periods
sp = [21, 32, 43, 53, 64]  # possible periods for grouping sessions
# will average to 21 seconds
if last_statistic_period >= 20 and last_statistic_period < np.int(np.mean((sp[0], \
sp[1]))):
    last_statistic_period = 21
    gamma = 1.3
# will average to 32 seconds
elif last_statistic_period >= np.int(np.mean((sp[0], sp[1]))) \
and last_statistic_period < np.int(np.mean((sp[1], sp[2]))):
    last_statistic_period = 30
    gamma = 1.4
# will average to 43 seconds
elif last_statistic_period >= np.int(np.mean((sp[1], sp[2]))) \
and last_statistic_period < np.int(np.mean((sp[2], sp[3]))):
    last_statistic_period = 42
    gamma = 1.6
# will average to 43 seconds
elif last_statistic_period >= np.int(np.mean((sp[3], sp[4]))) \
and last_statistic_period <= 60:
    last_statistic_period = 42
    gamma = 1.6


# indexing to obtain LASTs within last_statistic_period
if last_statistic:
    a = 1  # for index change later - array will reduced in dimension if indexed
    # for 1 specific LAST selecting nearest LAST sessions to average over nearest
    # {20,60} second period (try for half either side) here gamma obtained from
    # trial and error to match desired time averaging with actual averaging
    last_bound_h = last_statistic_period / 60. / 60. / gamma
    # last_mask_array = np.empty_like(vis_chans_range, dtype=bool)
    # dims = np.zeros_like(vis_chans_range)
    # find number of LAST bins within last_statistic_period, and index array
    # accordingly
    test_idx = np.squeeze(np.where(
        (last[:, 0] < LAST + last_bound_h) & (last[:, 0] > LAST - last_bound_h)))
    vis_last = np.zeros_like(visibilities)[test_idx, :, :, :]
    fags_last = np.zeros_like(visibilities)[test_idx, :, :, :]
    actual_avg_all = []
    # must do this way as not every day will have the same last indices for the
    # times within last_statistic_period
    for day in range(last.shape[1]):
        # getting indices of LAST bins that are within last_statistic_period
        chosen_last_idx = np.squeeze(np.where(
            (last[:, day] < LAST + last_bound_h) & (last[:, day] > LAST - last_bound_h)))
        # finding difference between first and last LAST bins used for averaging
        actual_avg = (last[:, day][chosen_last_idx[-1]] -
                      last[:, day][chosen_last_idx[0]]) * 60**2
        # adding all periods to an array for averaging later to check
        actual_avg_all.append(actual_avg)
        # print('Actual average in LAST for chosen day(s) '+str(days[i])+' is '\
        # +str(actual_avg)+' seconds.')
        vis_last[:, day, :, :] = visibilities[chosen_last_idx, day, :, :]
        fags_last[:, day, :, :] = flags[chosen_last_idx, day, :, :]
    print('Actual average in LAST for chosen days is ~{} seconds'.format(
          np.int(np.mean(actual_avg_all))))
    if (np.max(actual_avg_all) - np.min(actual_avg_all)) > 0.1:
        raise ValueError(
            'Combined exposure by joining consecutive sessions inconsistent day \
            to day. Check actual_avg_all values.')
# Choosing specific LAST
elif not last_statistic:
    a = 0
    visibilities = visibilities[find_nearest(np.median(last, axis=1), LAST)[1], :, :, :]


# len(np.where(0+0j == vis_last_chans_range)[0])
# 0. + 0.j in vis_last_chans_range


# continuing analysis in either complex visibilities or visibility amplitudes
if vis_complex_analysis:
    vis_analysis = np.ma.masked_array(
        vis_last_chans_range, mask=flags_last_chans_range)
    return_onesided_ps = False
else:
    vis_analysis = np.ma.masked_array(np.absolute(
        vis_last_chans_range), mask=flags_last_chans_range)
    return_onesided_ps = True


def day_sigma_clip(data, cenfunc='median', sigma=sc_days_stds, amp_clip=amp_clip):
    """Sigma clipping of visibilities over days/baselines, with sigma clipping
    done about the mean/median value for the data"""
    return


def day_sigma_clip(data, cenfunc='median', sigma=sc_days_stds, amp_clip=amp_clip):
    # sigma clipping over days
    day_sc_mask = np.zeros_like(data.data)
    for bl in range(0, data.shape[1+a]):  # iterate over baselines
        for frq in range(0, data.shape[2+a]):  # iterate over frequencies
            # dimensions of vis_amps depends on last_statistic (+1 dimensions if True)
            if last_statistic:
                for lst in range(0, data.shape[0]):  # iterate over last
                    # mask data array according to amplitudes rather than complex quantities (Re and Im clip)
                    if amp_clip:
                        clip = astropy.stats.sigma_clip(np.absolute(
                            data[lst, :, bl, frq]), sigma=sigma, maxiters=None, \
                            cenfunc=cenfunc, stdfunc='std', masked=True, \
                            return_bounds=False)
                    else:
                        clip = astropy.stats.sigma_clip(
                            data[lst, :, bl, frq], sigma=sigma, maxiters=None, \
                            cenfunc=cenfunc, stdfunc='std', masked=True, \
                            return_bounds=False)
                    # day_sc_mask is the clipped array with -999 for clipped values
                    day_sc_mask[lst, :, bl, frq] = clip.mask
            elif not last_statistic:
                clip = astropy.stats.sigma_clip(
                    data[:, bl, frq], sigma=sc_days_stds, maxiters=None, \
                    cenfunc=cenfunc, stdfunc='std', masked=True, return_bounds=False)
                day_sc_mask[:, bl, frq] = clip.mask
    # print('Day clips = ' + str(len(list(np.where(day_sc_mask == True)[0]))))
    vis_amps_dayclip = np.ma.masked_where(day_sc_mask, data)
    return vis_amps_dayclip


def bl_sigma_clip(data, cenfunc='median', sigma=sc_bls_stds, amp_clip=amp_clip):
    # sigma clipping over baselines
    bl_sc_mask = np.zeros_like(data.data, dtype=bool)
    for day in range(0, data.shape[0+a]): # iterate over days
        for frq in range(0, data.shape[2+a]): # iterate over frequencies
            if last_statistic:
                for lst in range(0, data.shape[0]): # iterate over last
                    if amp_clip:
                        clip = astropy.stats.sigma_clip(np.absolute(
                            data[lst, day, :, frq]), sigma=sc_days_stds, \
                            maxiters=None, cenfunc=cenfunc, stdfunc='std', \
                            masked=True, return_bounds=False)
                    else:
                        clip = astropy.stats.sigma_clip(
                            data[lst, day, :, frq], sigma=sc_days_stds, \
                            maxiters=None, cenfunc=cenfunc, stdfunc='std', \
                            masked=True, return_bounds=False)
                    bl_sc_mask[lst, day, :, frq] = clip.mask
            elif not last_statistic:
                clip = astropy.stats.sigma_clip(
                    data[day, :, frq], sigma=sc_days_stds, maxiters=None, \
                    cenfunc=cenfunc, stdfunc='std', masked=True, return_bounds=False)
                bl_sc_mask[day, :, frq] = clip.mask
    # print('Baseline clips = ' + str(len(list(np.where(bl_sc_mask == True)[0]))))
    vis_amps_blclip = np.ma.masked_where(bl_sc_mask, data)
    return vis_amps_blclip


def clipping(data, cenfunc='median', amp_clip=amp_clip):
    if sigma_clip_days: # sigma clip over days - leaving LAST bins, baselines and frequencies intact
        data = day_sigma_clip(data=data, cenfunc=cenfunc,
                              sigma=sc_days_stds, amp_clip=amp_clip)
        if True in data.mask:
            days_clipped = True
            count_vis_day_clipped = np.sum(data.mask)
            print('Day clipping has been applied: {} visibilities masked'.format(count_vis_day_clipped))
        else:
            days_clipped = False
            print('No day clipping applied; all data within {} sigma for each day'.format(sc_days_stds))
    elif not sigma_clip_days:
        days_clipped = False

    if sigma_clip_bls:  # sigma clip over baselines - leaving LAST bins, days and frequencies intact
        data = bl_sigma_clip(data=data, cenfunc=cenfunc,
                             sigma=sc_bls_stds, amp_clip=amp_clip)
        if True in data.mask:
            print('Baseline clipping has been applied: {} visibilities masked'.\
                  format(np.sum(data.mask)))
        else:
            print('No baseline clipping applied; all data within {} sigma for \
                  each baseline'.format(sc_days_stds))
    # elif not sigma_clip_bls:
    #     bl_sc_data = np.ma.masked_all_like(data)
    if sigma_clip_days or sigma_clip_bls:
        print('{} visibilities clipped out of {}'.format(np.sum(data.mask), \
              data.size))
    if not sigma_clip_days and not sigma_clip_bls:
        print('No clipping applied')
    return data


# mean statistic over days
def day_statistic(vis_array, days_statistic=days_statistic):
    vis_avg = getattr(np.ma, days_statistic)(vis_array, axis=0)
    return vis_avg


def power_spectrum(data1, data2=None, window='hann', length=None, scaling='spectrum', detrend=False, return_onesided=False):
    # CPS
    if data2 is not None:
        if stat_on_vis:
            # Finding dimension of returned delays
            delay_test, Pxy_spec_test = signal.csd(data1[0, :], data2[0, :], fs=1./resolution, window=window,
                                                   scaling=scaling, nperseg=length, detrend=detrend, return_onesided=return_onesided)
            # how many data points the signal.periodogram calculates
            delayshape = delay_test.shape[0]
            # dimensions are [baselines, (delay, Pxx_spec), delayshape]
            vis_ps = np.zeros((data1.shape[0], 2, delayshape), dtype=complex)
            for i in range(0, data1.shape[0]):  # Iterating over all baselines
                delay, Pxy_spec = signal.csd(data1[i, :], data2[i, :], fs=1./resolution, window=window,
                                             scaling=scaling, nperseg=length, detrend=detrend, return_onesided=return_onesided)
                # Pxy_spec = np.absolute(Pxy_spec)
                vis_ps[i, :, :] = np.array(
                    [delay[np.argsort(delay)], Pxy_spec[np.argsort(delay)]])
        else:
            # statistics will be done later in the power spectrum domain
            delay_test, Pxy_spec_test = signal.csd(data1[0, 0, 0, :], data2[0, 0, 0, :], fs=1./resolution,
                                                   window=window, scaling=scaling, nperseg=length, detrend=detrend, return_onesided=return_onesided)
            # how many data points the signal.periodogram calculates
            delayshape = delay_test.shape[0]
            # dimensions are [baselines, (delay, Pxx_spec), delayshape]
            vis_ps = np.zeros((data1.shape[0], np.minimum(
                data1.shape[1], data2.shape[1]), data1.shape[2], 2, delayshape), dtype=complex)
            for time in range(vis_ps.shape[0]):  # Iterating over all last bins
                for day in range(vis_ps.shape[1]):
                    for bl in range(vis_ps.shape[2]):
                        delay, Pxy_spec = signal.csd(data1[time, day, bl, :], data2[time, day, bl, :], fs=1./resolution,
                                                     window=window, scaling=scaling, nperseg=length, detrend=detrend, return_onesided=return_onesided)
                        # Pxy_spec = np.absolute(Pxy_spec)
                        vis_ps[time, day, bl, :, :] = np.array(
                            [delay[np.argsort(delay)], Pxy_spec[np.argsort(delay)]])
    # PS
    else:
        if stat_on_vis:
            # Finding dimension of returned delays
            delay_test, Pxx_spec_test = signal.periodogram(
                data1[1, :], fs=1./resolution, window=window, scaling=scaling, nfft=length, detrend=detrend)
            # how many data points the signal.periodogram calculates
            delayshape = delay_test.shape[0]
            # dimensions are [baselines, (delay, Pxx_spec), delayshape]
            vis_ps = np.zeros((data1.shape[0], 2, delayshape), dtype=complex)
            for i in range(data1.shape[0]):  # Iterating over all baselines
                delay, Pxx_spec = signal.periodogram(
                    data1[i, :], fs=1./resolution, window=window, scaling=scaling, nfft=length, detrend=detrend)
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

        # splitting array of days into 2 (for cross power spectrum between mean/median of two halves)
        # split visibilities array into two sub-arrays of (near) equal days
        vis_halves = np.array_split(vis_amps_clipped_lastavg, 2, axis=0)

        vis_half1_preflag = day_statistic(vis_halves[0])
        vis_half2_preflag = day_statistic(vis_halves[1])
        vis_whole_preflag = day_statistic(vis_amps_clipped_lastavg)

        # find out where data is flagged from clipping - if statistic applied on days, still get gaps in the visibilities
        # list(set(np.where(day_statistic(vis_halves[1]).mask == True)[0])) # finds flagged baselines
        baselines_clip_faulty = list(
            set(np.where(day_statistic(vis_amps_clipped_lastavg).mask == True)[0]))
        baselines_clip_flag = np.delete(
            np.arange(len(baselines_dayflg_blflg)), baselines_clip_faulty)
        print('Baseline(s) '+str(baselines_dayflg_blflg[baselines_clip_faulty]) +
              ' removed from analysis due to flagging from sigma clipping - there are gaps in visibilities for these baselines for the channel range ' + str(channel_start) + '-' + str(channel_end) + '.')

        vis_half1 = vis_half1_preflag[baselines_clip_flag, :]
        vis_half2 = vis_half2_preflag[baselines_clip_flag, :]
        vis_amps_final = vis_whole_preflag[baselines_clip_flag, :]
        baselines_dayflg_blflg_clipflg = baselines_dayflg_blflg[baselines_clip_flag]

        # window functions: boxcar (equivalent to no window at all), triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann, kaiser
        vis_ps_final = power_spectrum(vis_half1, vis_half2, window='boxcar',
                                      length=vis_half1.shape[1], scaling='spectrum', detrend=False, return_onesided=return_onesided_ps)  # spectrum
        # vis_psd_final = power_spectrum(vis_half1, vis_half2, window='boxcar', length=vis_half1.shape[1], scaling='density', detrend=False, return_onesided=False)


else:
    vis_amps_clipped_lastavg = getattr(np.ma, last_statistic_method)\
                               (vis_amps_clipped, axis=0)
    vis_whole_preflag = day_statistic(vis_amps_clipped_lastavg)
    baselines_clip_faulty = list(
        set(np.where(day_statistic(vis_amps_clipped_lastavg).mask == True)[0]))
    baselines_clip_flag = np.delete(
        np.arange(len(baselines_dayflg_blflg)), baselines_clip_faulty)
    baselines_dayflg_blflg_clipflg = baselines_dayflg_blflg[baselines_clip_flag]
    print('Baseline(s) '+str(baselines_dayflg_blflg[baselines_clip_faulty]) +
          ' removed from analysis due to flagging from sigma clipping - there are gaps in visibilities for these baselines for the channel range ' + str(channel_start) + '-' + str(channel_end) + '.')
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
    # not removing any baselines at this point - unless some have gaps in the visibilities across the chans_range
    # would then have to apply Mark's psd_estimation
    vis_ps_raw = power_spectrum(vis_halves[0], vis_halves[1], window='boxcar', length=vis_halves[0].shape[-1],
                                scaling='spectrum', detrend=False, return_onesided=return_onesided_ps)  # spectrum
    # could clip again?
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
