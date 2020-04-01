# module load miniconda3-4.5.4-gcc-5.4.0-hivczbz
# source activate astroenv
# scp /Users/matyasmolnar/HERA_Data/VisibilityPS/* mdm49@login-cpu.hpc.cam.ac.uk:/rds/project/bn204/rds-bn204-asterics/mdm49
# scp mdm49@login-cpu.hpc.cam.ac.uk:/rds/project/bn204/rds-bn204-asterics/mdm49/*.pdf /Users/matyasmolnar/HERA_Data/Figures

# HERA Visibility PS Computation Script
#
# Matyas Molnar, University of Cambridge
# mdm49@cam.ac.uk
#
# This script takes aligned the aligned HERA visibilities in LAST (as outputted by
# align_lst.py) and computes various PS estimates using various statistics over LASTs,
# days and baselines, and provides a per baseline analysis of PS


from psd_estimation import *
import sys
import os
import astropy.stats
import math
import numpy as np
import scipy
from scipy import signal
from scipy import fftpack as sf
import matplotlib.pyplot as plt
import matplotlib
import functools
import seaborn as sns
sns.set()
sns.set_style("whitegrid")

# import warnings
# warnings.simplefilter("ignore", UserWarning)


#####################################################################################################

# Inputs:

# {vis, ps} run statistics on either the visibilities or power spectra
stat_vis_or_ps = 'ps'

# if visibility statistics: True - deal with the complex visibibilities
#                         : False - or the visibility amplitudes for the statistics (before PS computation)
vis_complex_analysis = True

# if power spectrum statistics: True - deal with the complex power spectra
#                             : False - or the power spectra amplitudes for the statistics (after PS computation)
ps_complex_analysis = True

# LAST to analyze
LAST = 3.31

# statistic in time between consecutive sessions - if False, closest time bin to LAST will be chosen
last_statistic = True
last_statistic_period = 40  # seconds; chose between {20,60}
last_statistic_method = 'median'  # {median, mean}

# run statistics over all IDR2 Days? if False, fill in chosen_days
statistic_all_IDR2 = False
IDR2 = [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, 2458106, 2458107,
        2458108, 2458109, 2458110, 2458111, 2458112, 2458113, 2458114, 2458115, 2458116, 2458140]

chosen_days = [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, 2458106, 2458107, 2458108,
               2458110, 2458111, 2458112, 2458113, 2458116]

days_statistic = 'median'  # {median, mean}

amp_clip = True  # sigma clip according to amplitudes? if vis_complex_analysis, will still mask according to behaviour of amplitudes

sigma_clip_days = True
sc_days_stds = 3.0

sigma_clip_bls = True
sc_bls_stds = 3.0

channel_start = 100
channel_end = 250

fig_path = '/rds/project/bn204/rds-bn204-asterics/mdm49/'
savefigs = False

#####################################################################################################

if statistic_all_IDR2:
    chosen_days = IDR2

if last_statistic_period < 20 or last_statistic_period > 60:
    raise ValueError(
        'Chose LAST averaging period to be between 20 and 60 seconds.')


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
    raise ValueError(
        'Chose to perform the statistics on either the visibibilities or power spectra.')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]  # index of nearest value, nearest value

# # Loading npz file of single visibility dataset:
# vis_data = np.load('/rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_arrays_python/zen.2458099.46852.xx.HH.ms.npz')
# vis_data.files #['bl', 'vis', 'flags', 'LAST']
# vis_data['bl'] #gives baselines (of which there are 35) - same ones as specified in inBaselines in master_conversion_python.py
# vis_data['vis'].shape #(35, 1, 1024, 60) #(baselines, ??, channels, LAST times)
# vis_data['flags'] #same dimensions as vis
# vis_data['LAST'].shape #60 - number of integrations in the observation session


# Loading npz file of all aligned visibilities
vis_data = np.load(
    '/rds/project/bn204/rds-bn204-asterics/mdm49/aligned/aligned_visibilities.npz')
vis_data.files  # ['visibilities', 'baselines', 'last', 'days', 'flags']
# shape: (220, 18, 35, 1024) = (last bins, day, baselines, channels)
visibilities = vis_data['visibilities']
# gives baselines (of which there are 35) - same ones as specified in inBaselines in master_conversion_python.py
baselines = vis_data['baselines']
last = vis_data['last']  # shape: (220,18) = (aligned data columns, IDRDays)??
days = np.array(vis_data['days'], dtype=int)  # JD days of HERA data
flags = np.array(vis_data['flags'], dtype=bool)  # shape same dimensions as vis


# all in MHz
bandwidth_start = 1.0e8
bandwidth_end = 2.0e8
resolution = 97656.25

channels = np.arange(0, 1024, dtype=int)
# index for channel & freqs arrays. Includes extremities.
chan_range = np.arange(channel_start-1, channel_end)
freqs = np.arange(bandwidth_start, bandwidth_end, resolution)
freq_range = freqs[chan_range]


if LAST < np.min(last) or LAST > np.trunc(np.max(last)*100)/100:
    raise ValueError('Chose LAST value in between ' + str(round(np.min(last), 2)
                                                          ) + ' and ' + str(np.trunc(np.max(last)*100)/100))

if channel_start < channels[0] or channel_end > channels[-1] or channel_start > channel_end:
    raise ValueError('Chose channels in between ' + str(channels[0]) + ' and ' + str(
        channels[-1]) + ' with channel_start > channel_end')


# checking alignment of LASTs. Issue with missing consecutive sessions. Need to check calibration.
def check_alignment():
    for i in range(0, last.shape[1]):
        print(i)
        print(last[0, i])
        print(last[-1, i])
        print('-----------------')
# check_alignment()


# removing days that do not appear in chosen_days
day_flag_idxs = []
for chosen_day in chosen_days:
    day_flag_idxs.append(np.ndarray.tolist(days).index(chosen_day))

# removing misaligned data - fault in align_lst.py script
day_mis_idxs = []
# misaligned_days = [10,15,17] corresponds to faulty days: 2458109, 2458115, 2458140
misaligned_days = [2458109, 2458115, 2458140]
for mis_day in misaligned_days:
    day_mis_idxs.append(np.ndarray.tolist(days).index(mis_day))
day_mis_flg = list(np.delete(np.arange(len(days)), day_mis_idxs))
print('Day(s) ' + str(misaligned_days) +
      ' removed from analysis due to misalignment in LAST bins.')

dayflg = np.array(list(set(day_flag_idxs + day_mis_flg)))

# indexing out faulty days due to misalignment
visibilities_dayflg = visibilities[:, dayflg, :, :]
flags_dayflg = flags[:, dayflg, :, :]
last_dayflg = last[:, dayflg]
days_dayflg = days[dayflg]

# removing faulty baselines
# , [66, 67], [67, 68], [68, 69],[82, 83], [83, 84], [122, 123]]
faulty_bls = [[50, 51]]
faulty_bl_idxs = []
for bl in faulty_bls:
    faulty_bl_idxs.append(np.ndarray.tolist(baselines).index(bl))
flags_bls = np.delete(np.arange(len(baselines)), faulty_bl_idxs)

# indexing out faulty baselines
visibilities_dayflg_blflg = visibilities_dayflg[:, :, flags_bls, :]
flags_dayflg_blflg = flags_dayflg[:, :, flags_bls, :]
baselines_dayflg_blflg = baselines[flags_bls, :]

# only considering specified channel range
vis_chan_range = visibilities_dayflg_blflg[:, :, :, chan_range]
flags_chan_range = flags_dayflg_blflg[:, :, :, chan_range]


# parameter for getting correct grouping of LASTs from last_statistic_periods
sp = [21, 32, 43, 53, 64]  # possible periods for grouping sessions
# will average to 21 seconds
if last_statistic_period >= 20 and last_statistic_period < np.int(np.mean((sp[0], sp[1]))):
    last_statistic_period = 21
    gamma = 1.3
# will average to 32 seconds
elif last_statistic_period >= np.int(np.mean((sp[0], sp[1]))) and last_statistic_period < np.int(np.mean((sp[1], sp[2]))):
    last_statistic_period = 30
    gamma = 1.4
# will average to 43 seconds
elif last_statistic_period >= np.int(np.mean((sp[1], sp[2]))) and last_statistic_period < np.int(np.mean((sp[2], sp[3]))):
    last_statistic_period = 42
    gamma = 1.6
# will average to 43 seconds
elif last_statistic_period >= np.int(np.mean((sp[3], sp[4]))) and last_statistic_period <= 60:
    last_statistic_period = 42
    gamma = 1.6


# indexing to obtain LASTs within last_statistic_period
if last_statistic:
    a = 1  # for index change later - array will reduced in dimension if indexed for 1 specific LAST
    # selecting nearest LAST sessions to average over nearest {20,60} second period (try for half either side)
    # here gamma obtained from trial and error to match desired time averaging with actual averaging
    last_bound_h = last_statistic_period / 60. / 60. / gamma
    # last_mask_array = np.empty_like(vis_chan_range, dtype=bool)
    # dims = np.zeros_like(vis_chan_range)
    # find number of LAST bins within last_statistic_period, and index array accordingly
    test_idx = np.squeeze(np.where(
        (last_dayflg[:, 0] < LAST + last_bound_h) & (last_dayflg[:, 0] > LAST - last_bound_h)))
    vis_last_chan_range = np.zeros_like(vis_chan_range)[test_idx, :, :, :]
    flags_last_chan_range = np.zeros_like(vis_chan_range)[test_idx, :, :, :]
    actual_avg_all = []
    # must do this way as not every day will have the same last indices for the times within last_statistic_period
    for day in range(last_dayflg.shape[1]):
        # getting indices of LAST bins that are within last_statistic_period
        chosen_last_idx = np.squeeze(np.where(
            (last_dayflg[:, day] < LAST + last_bound_h) & (last_dayflg[:, day] > LAST - last_bound_h)))
        # print(chosen_last_idx)
        # finding difference between first and last LAST bins used for averaging
        actual_avg = (last_dayflg[:, day][chosen_last_idx[-1]] -
                      last_dayflg[:, day][chosen_last_idx[0]]) * 60**2
        # adding all periods to an array for averaging later to check
        actual_avg_all.append(actual_avg)
        # print('Actual average in LAST for chosen day(s) '+str(days[i])+' is '+str(actual_avg)+' seconds.')
        vis_last_chan_range[:, day, :,
                            :] = vis_chan_range[chosen_last_idx, day, :, :]
        flags_last_chan_range[:, day, :,
                              :] = flags_chan_range[chosen_last_idx, day, :, :]
    print('Actual average in LAST for chosen days is ~' +
          str(np.int(np.mean(actual_avg_all)))+' seconds.')
    if (np.max(actual_avg_all) - np.min(actual_avg_all)) > 0.1:
        raise ValueError(
            'Combined exposure by joining consecutive sessions inconsistent day to day. Check actual_avg_all values.')
# chosing specific LAST
elif not last_statistic:
    a = 0
    vis_last_chan_range = vis_chan_range[find_nearest(
        last_dayflg[:, 0], LAST)[0], :, :, :]


# len(np.where(0+0j == vis_last_chan_range)[0])
# 0. + 0.j in vis_last_chan_range


# continuing analysis in either complex visibilities or visibility amplitudes
if vis_complex_analysis:
    vis_analysis = np.ma.masked_array(
        vis_last_chan_range, mask=flags_last_chan_range)
    return_onesided_ps = False
else:
    vis_analysis = np.ma.masked_array(np.absolute(
        vis_last_chan_range), mask=flags_last_chan_range)
    return_onesided_ps = True


# compare plots before after clipping and last averaging.
def plot_stat_vis(data, *statistic, clipped=False, last_avg=False, figname=None):
    plt.figure(figsize=[10, 7])
    stat_fn = 'np.ma.'
    for stat in statistic:
        if not last_avg:
            plt.plot(chan_range+1, eval(stat_fn+stat)(eval(stat_fn+stat)
                                                      (eval(stat_fn+stat)(np.squeeze(data), axis=0), axis=0), axis=0), label=stat)
        elif last_avg:
            plt.plot(chan_range+1, eval(stat_fn+stat)(eval(stat_fn+stat)
                                                      (np.squeeze(data), axis=0), axis=0), label=stat)
    plt.xlabel('Channel')
    plt.ylabel('Visibility Amplitude')
    plt.legend(loc='upper right')
    if clipped:
        clip_title = ' after clipping'
    else:
        clip_title = ''
    if last_avg:
        last_title = ' and LAST averaging'
    else:
        last_title = ''
    if len(statistic) == 1:
        plt.title(
            'Statistic visibility amplitudes over baselines and days' + clip_title + last_title)
    else:
        plt.title(
            'Statistic of visibility amplitudes over baselines and days' + clip_title + last_title)
    if savefigs:
        plt.savefig(figname, format='pdf', dpi=300)
    plt.ion()
    plt.show()


plot_stat_vis(np.absolute(vis_analysis), 'mean', 'median', clipped=False,
              last_avg=False, figname=fig_path+'mean_and_median_vis_'+fig_suffix+'.pdf')


def plot_single_bl_vis(data, time, day, bl):
    plt.figure(figsize=[10, 7])
    plt.plot(chan_range+1, data[find_nearest(last_dayflg[:, 0], time)[0], np.ndarray.tolist(
        days_dayflg).index(day), np.ndarray.tolist(baselines_dayflg_blflg).index(bl), :])
    plt.xlabel('Channel')
    plt.ylabel('Visibility Amplitude')
    plt.title('Amplitudes from antennas ' + str(bl) +
              ' at JD ' + str(day) + ' at LAST ' + str(time))
    plt.ion()
    plt.show()
# plot_single_bl_vis(data=np.absolute(vis_chan_range), time=LAST, day=2458101, bl=[82, 83])


# sigma clipping of visibilities over days and baselines, with sigma clipping done about the {median, mean} value for the data
# sigma clipping over days
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
                            data[lst, :, bl, frq]), sigma=sigma, maxiters=None, cenfunc=cenfunc, stdfunc='std', masked=True, return_bounds=False)
                    else:
                        clip = astropy.stats.sigma_clip(
                            data[lst, :, bl, frq], sigma=sigma, maxiters=None, cenfunc=cenfunc, stdfunc='std', masked=True, return_bounds=False)
                    # day_sc_mask is the clipped array with -999 for clipped values
                    day_sc_mask[lst, :, bl, frq] = clip.mask
            elif not last_statistic:
                clip = astropy.stats.sigma_clip(
                    data[:, bl, frq], sigma=sc_days_stds, maxiters=None, cenfunc=cenfunc, stdfunc='std', masked=True, return_bounds=False)
                day_sc_mask[:, bl, frq] = clip.mask
    # print('Day clips = ' + str(len(list(np.where(day_sc_mask == True)[0]))))
    vis_amps_dayclip = np.ma.masked_where(day_sc_mask, data)
    return vis_amps_dayclip


# sigma clipping over baselines
def bl_sigma_clip(data, cenfunc='median', sigma=sc_bls_stds, amp_clip=amp_clip):
    bl_sc_mask = np.zeros_like(data.data, dtype=bool)
    for day in range(0, data.shape[0+a]):  # iterate over days
        for frq in range(0, data.shape[2+a]):  # iterate over frequencies
            if last_statistic:
                for lst in range(0, data.shape[0]):  # iterate over last
                    if amp_clip:
                        clip = astropy.stats.sigma_clip(np.absolute(
                            data[lst, day, :, frq]), sigma=sc_days_stds, maxiters=None, cenfunc=cenfunc, stdfunc='std', masked=True, return_bounds=False)
                    else:
                        clip = astropy.stats.sigma_clip(
                            data[lst, day, :, frq], sigma=sc_days_stds, maxiters=None, cenfunc=cenfunc, stdfunc='std', masked=True, return_bounds=False)
                    bl_sc_mask[lst, day, :, frq] = clip.mask
            elif not last_statistic:
                clip = astropy.stats.sigma_clip(
                    data[day, :, frq], sigma=sc_days_stds, maxiters=None, cenfunc=cenfunc, stdfunc='std', masked=True, return_bounds=False)
                bl_sc_mask[day, :, frq] = clip.mask
    # print('Baseline clips = ' + str(len(list(np.where(bl_sc_mask == True)[0]))))
    vis_amps_blclip = np.ma.masked_where(bl_sc_mask, data)
    return vis_amps_blclip


def clipping(data, cenfunc='median', amp_clip=amp_clip):
    if sigma_clip_days:  # sigma clip over days - leaving LAST bins, baselines and frequencies intact
        data = day_sigma_clip(data=data, cenfunc=cenfunc,
                              sigma=sc_days_stds, amp_clip=amp_clip)
        if True in data.mask:
            days_clipped = True
            count_vis_day_clipped = np.sum(data.mask)
            print('Day clipping has been applied: ' +
                  str(count_vis_day_clipped) + ' visibilities masked.')
        else:
            days_clipped = False
            print('No day clipping applied; all data within ' +
                  str(sc_days_stds)+' sigma for each day.')
    elif not sigma_clip_days:
        days_clipped = False

    if sigma_clip_bls:  # sigma clip over baselines - leaving LAST bins, days and frequencies intact
        data = bl_sigma_clip(data=data, cenfunc=cenfunc,
                             sigma=sc_bls_stds, amp_clip=amp_clip)
        if True in data.mask:
            print('Baseline clipping has been applied: ' +
                  str(np.sum(data.mask)) + ' visibilities masked.')
        else:
            print('No baseline clipping applied; all data within ' +
                  str(sc_days_stds)+' sigma for each baseline.')
    # elif not sigma_clip_bls:
    #     bl_sc_data = np.ma.masked_all_like(data)
    if sigma_clip_days or sigma_clip_bls:
        print(str(np.sum(data.mask)) +
              ' visibilities clipped out of ' + str(data.size))
    if not sigma_clip_days and not sigma_clip_bls:
        print('No clipping applied.')
    return data


# mean statistic over days
def day_statistic(vis_array, days_statistic=days_statistic):
    vis_avg = eval('np.ma.' + days_statistic)(vis_array, axis=0)
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
        vis_amps_clipped_lastavg = eval(
            'np.ma.'+last_statistic_method)(vis_amps_clipped, axis=0)
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
    vis_amps_clipped_lastavg = eval(
        'np.ma.'+last_statistic_method)(vis_amps_clipped, axis=0)
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
    # not removing any baselines at this point - unless some have gaps in the visibilities across the chan_range
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
    vis_ps_raw_lastavg = eval(
        'np.ma.'+last_statistic_method)(vis_ps_raw_analysis, axis=0)
    vis_ps_final = day_statistic(vis_ps_raw_lastavg)
    # need to check data is smooth - no gaps


# chosing single day
# if len(chosen_days) == 1:
#     vis_amps_single_day = vis_amps_clipped_lastavg[np.where(days_dayflg == IDRDay)[0][0] ,:,:]


# convert units in dimensionless PS? check hera pspec for guidance
def plot_stat_ps(data, *statistics, scaling='spectrum', rms=False, figname):
    plt.figure()
    for statistic in statistics:
        stat = eval('np.' + statistic)(data, axis=0)
        # plt.figure()
        if scaling == 'density':
            plt.semilogy(np.real(stat[0]*1e6),
                         np.absolute(stat[1]), label=statistic)
            plt.ylabel('Cross power spectral density [Amp**2/s]')
        elif scaling == 'spectrum':
            if rms:
                plt.semilogy(
                    np.real(stat[0]*1e6), np.sqrt(np.absolute(stat[1])), label=statistic)
                plt.ylabel('Linear spectrum [Amp RMS]')
            else:
                plt.semilogy(np.real(stat[0]*1e6),
                             np.absolute(stat[1]), label=statistic)
                plt.ylabel('Cross power spectrum [Amp**2]')
                plt.legend(loc='upper right')
        else:
            raise ValueError('Chose either spectrum of density for scaling.')
        plt.xlabel('Geometric delay [$\mu$s]')
        if len(statistics) > 1:
            plt.title('Cross power spectrum over E-W baselines')
        else:
            plt.title('M' + statistic[1:] +
                      ' cross power spectrum over E-W baselines')
    if savefigs:
        plt.savefig(figname, format='pdf', dpi=300)
    plt.ion()
    plt.show()


plot_stat_ps(vis_ps_final, 'mean', 'median', scaling='spectrum', rms=False,
             figname=fig_path+'mean_and_median_ps_'+fig_suffix+'.pdf')  # comparing the statistics
# plot_stat_ps(vis_ps_final, 'median')


def factors(n):
    return np.asarray(sorted(functools.reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))


def plot_size(x):
    no_rows = int(find_nearest(factors(x), x/2.)[1])
    # if no_rows == 1:
    #   statement to break up into multiple rows etc.
    no_cols = int(x / no_rows)
    return no_rows, no_cols


plot_dim = plot_size(len(baselines_dayflg_blflg_clipflg))


def baseline_vis_analysis(data):
    if os.path.isfile(fig_path):
        os.remove(fig_path)
    no_rows = 7  # plot_dim[0]
    no_cols = 5  # plot_dim[1]
    fig, axs = plt.subplots(nrows=no_rows, ncols=no_cols,
                            sharex=True, sharey=True, squeeze=False)
    # xmin=(round(chan_range[0]/100)-1)*100
    # xmax=(round(chan_range[-1]/100)+1)*100
    # plt.axis(xmin=xmin, xmax=xmax)
    for row in range(no_rows):
        for col in range(no_cols):
            if (row*no_cols)+col <= len(baselines_dayflg_blflg_clipflg)-1:
                axs[row, col].plot(
                    chan_range+1, data[(row*no_cols)+col, :], linewidth=1)
                axs[row, col].legend([str(baselines_dayflg_blflg_clipflg[(
                    row*no_cols)+col])], loc='upper right', prop={'size': 6}, frameon=False)
                # axs[row,col].set_xticks(np.arange(xmin,xmax,200))
                # axs[row,col].set_xticks(np.arange(0,6,0.2), minor=True)
                # axs[row,col].set_yticks(np.power(np.ones(5)*10,-np.arange(1,10,2)))
                axs[row, col].xaxis.set_tick_params(width=1)
                axs[row, col].yaxis.set_tick_params(width=1)
    plt.suptitle('Visibility amplitudes for all E-W baselines', y=0.95)
    fig.text(0.5, 0.04, 'Channel', ha='center')
    fig.text(0.04, 0.5, 'Visibility amplitude',
             va='center', rotation='vertical')
    # fig.tight_layout()
    fig.set_size_inches(w=11, h=7.5)
    if savefigs:
        plt.savefig(fig_path+'visibility_analysis_' +
                    fig_suffix+'.pdf', format='pdf', dpi=300)
    plt.ion()
    plt.show()


baseline_vis_analysis(np.ma.masked_array(np.absolute(
    vis_amps_final.data), mask=vis_amps_final.mask, dtype=float))


def baseline_ps_analysis(ps):
    if os.path.isfile(fig_path):
        os.remove(fig_path)
    no_rows = 7  # plot_dim[0]
    no_cols = 5  # plot_dim[1]
    fig, axs = plt.subplots(nrows=no_rows, ncols=no_cols,
                            sharex=True, sharey=True, squeeze=False)
    plt.axis(xmin=-0.1, xmax=5.2, ymin=1e-10, ymax=1e-0)
    for row in range(no_rows):
        for col in range(no_cols):
            if (row*no_cols)+col <= len(baselines_dayflg_blflg_clipflg)-1:
                axs[row, col].semilogy(np.real(
                    ps[(row*no_cols)+col, 0, :])*1e6, np.absolute(ps[(row*no_cols)+col, 1, :]), linewidth=1)
                axs[row, col].legend([str(baselines_dayflg_blflg_clipflg[(
                    row*no_cols)+col])], loc='upper right', prop={'size': 6}, frameon=False)
            if return_onesided_ps:
                axs[row, col].set_xticks(np.arange(0, 7, 2))
                axs[row, col].set_xticks(np.arange(0, 6, 0.2), minor=True)
            else:
                axs[row, col].set_xticks(np.arange(-6, 7, 2))
                axs[row, col].set_xticks(np.arange(-6, 6, 0.5), minor=True)
            axs[row, col].set_yticks(
                np.power(np.ones(5)*10, -np.arange(1, 10, 2)))
            # minors = []
            # for p in range(6):
            #     for q in range(2,10):
            #         minors.append(q * (10**(-1*p)))
            # minors = np.array(minors)
            # # axs[row,col].set_yticks(minors, minor=True)
            # axs[row,col].xaxis.set_tick_params(width=1)
            # axs[row,col].yaxis.set_tick_params(width=1)
    plt.suptitle('(Cross) power spectrum for all E-W baselines', y=0.95)
    fig.text(0.5, 0.04, 'Geometric delay [$\mu$s]', ha='center')
    fig.text(
        0.04, 0.5, 'Cross power spectrum [Amp**2]', va='center', rotation='vertical')
    # fig.tight_layout()
    fig.set_size_inches(w=11, h=7.5)
    if savefigs:
        plt.savefig(fig_path+'baseline_analysis_' +
                    fig_suffix+'.pdf', format='pdf', dpi=300)
    plt.ion()
    plt.show()


baseline_ps_analysis(vis_ps_final)


# rm ~/Desktop/baseline_analysis.pdf
# scp mdm49@login-cpu.hpc.cam.ac.uk:/rds/project/bn204/rds-bn204-asterics/mdm49/baseline_analysis.pdf ~/Desktop/


# Add baseline functionality for EW, NS, 14m, 28m, individual baselines etc. - although this done before export to npz?
