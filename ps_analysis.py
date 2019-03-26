# module load miniconda3-4.5.4-gcc-5.4.0-hivczbz
# source activate astroenv
# scp /Users/matyasmolnar/HERA_Data/VisibilityPS/ps_analysis.py mdm49@login-cpu.hpc.cam.ac.uk:/rds/project/bn204/rds-bn204-asterics/mdm49
# scp /Users/matyasmolnar/HERA_Data/VisibilityPS/* mdm49@login-cpu.hpc.cam.ac.uk:/rds/project/bn204/rds-bn204-asterics/mdm49


# HERA Visibility PS Computation Script
#
# Matyas Molnar, University of Cambridge
# mdm49@cam.ac.uk
#
# This script takes aligned the aligned HERA visibilities in LAST (as outputted by
# align_lst.py) and computes various PS estimates using various statistics over LASTs,
# days and baselines, and provides a per baseline analysis of PS


import sys, os
import astropy.stats
import math
import numpy as np
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import functools
# import pickle
# from __future__ import print_function

#####################################################################################################

# to do
# currently method is:
# average over LAST for {20,60} seconds
# sigma clip over days
# sigma clip over baseline

# Input parameters:

# LAST to analyze
LAST = 3.31 # float

last_average = True
last_average_period = 40 #seconds; chose between {20,60}
last_statistic = 'median' # {median, mean}

statistic_all_IDR2 = False # run statistics over all IDR2 Days?
IDR2=[2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, 2458106, 2458107,
        2458108, 2458109, 2458110, 2458111, 2458112, 2458113, 2458114, 2458115, 2458116, 2458140]

chosen_days = [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, 2458106, 2458107, 2458108,
            2458110, 2458111, 2458112, 2458113, 2458116]

days_statistic = 'median' # {median, mean}

sigma_clip_days = True
sc_days_stds = 3.0

sigma_clip_bl = True
sc_bls_stds = 3.0

channel_start = 100
channel_end = 400

#####################################################################################################


if last_statistic == 'median':
    mean_last = False
    median_last = True
elif last_statistic == 'mean':
    mean_last = True
    median_last = False

if days_statistic == 'median':
    mean_days = False
    median_days = True
elif days_statistic == 'mean':
    mean_days = True
    median_days = False


if mean_last == median_last:
    raise ValueError('Chose either mean or median for the last statistic!')

if mean_days == median_days:
    raise ValueError('Chose either mean or median for the day statistic!')

if last_average_period < 20 or last_average_period > 60:
     raise ValueError('Chose LAST averaging period to be between 20 and 60 seconds.')


chan_range = np.arange(channel_start-1,channel_end) # index for channel & freqs arrays. Includes extremities.

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx] # index of nearest value, nearest value

# # Loading npz file of single visibility dataset:
# vis_data = np.load('/rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_arrays_python/zen.2458099.46852.xx.HH.ms.npz')
# vis_data.files #['bl', 'vis', 'flags', 'LAST']
# vis_data['bl'] #gives baselines (of which there are 35) - same ones as specified in inBaselines in master_conversion_python.py
# vis_data['vis'].shape #(35, 1, 1024, 60) #(baselines, ??, channels, LAST times)
# vis_data['flags'] #same dimensions as vis
# vis_data['LAST'].shape #60 - number of integrations in the observation session

# Loading npz file of all aligned visibilities
vis_data = np.load('/rds/project/bn204/rds-bn204-asterics/mdm49/aligned/aligned_visibilities.npz')
vis_data.files #['visibilities', 'baselines', 'last', 'days', 'flags']
visibilities = vis_data['visibilities'] # shape: (220, 18, 35, 1024) = (last bins, day, baselines, channels)
baselines = vis_data['baselines'] #gives baselines (of which there are 35) - same ones as specified in inBaselines in master_conversion_python.py
last = vis_data['last'] # shape: (220,18) = (aligned data columns, IDRDays)??
days = np.array(vis_data['days'], dtype=int) #JD days of HERA data
flags = np.array(vis_data['flags'], dtype=bool) # shape same dimensions as vis

# all in MHz
bandwidth_start = 1.0e8
bandwidth_end = 2.0e8
resolution = 97656.25

freqs = np.arange(bandwidth_start, bandwidth_end, resolution)
freq_range = freqs[chan_range]

channels = np.arange(0,1024, dtype=int)
frc=np.zeros((2,1024))
frc[0,:] = channels
frc[1,:] = freqs

# # Checking aligment of LASTs. Issue with missing consecutive sessions. Need to check calibration.
# for i in range(0,last.shape[1]):
#     print(i)
#     print(last[0,i])
#     print(last[-1,i])
#     print('-----------------')

# removing misalignmed data - fault in align_lst.py script
misaligned_days = [10,15,17] # corresponds to faulty days: 2458109, 2458115, 2458140
flags_mis = np.delete(np.arange(0,18), misaligned_days)

# indexing out faulty days due to misalignment
visibilities = visibilities[:,flags_mis,:,:]
last = last[:,flags_mis]
days = days[flags_mis]
flags = flags[:,flags_mis,:,:]

# only considering specified channel range
vis_chan_range = visibilities[:,:,:,chan_range]
channels = channels[chan_range]

# parameter for getting correct grouping of LASTs from last_average_periods
sp = [21,32,43,53,64] # possible periods for grouping sessions
if last_average_period >= 20 and last_average_period < np.int(np.mean((sp[0],sp[1]))): # will average to 21 seconds
    last_average_period = 21
    gamma = 1.3
elif last_average_period >= np.int(np.mean((sp[0],sp[1]))) and last_average_period < np.int(np.mean((sp[1],sp[2]))): # will average to 32 seconds
    last_average_period = 30
    gamma = 1.4
elif last_average_period >= np.int(np.mean((sp[1],sp[2]))) and last_average_period < np.int(np.mean((sp[2],sp[3]))): # will average to 43 seconds
    last_average_period = 42
    gamma = 1.6
elif last_average_period >= np.int(np.mean((sp[3],sp[4]))) and last_average_period <= 60: # will average to 43 seconds
    last_average_period = 42
    gamma = 1.6


# indexing to obtain LASTs within last_average_period
if last_average:
    a = 1 # for index change later - array will reduced in dimension if indexed for 1 specific LAST
    # selecting nearest LAST sessions to average over nearest {20,60} second period (try for half either side)
    last_bound_h = last_average_period / 60. / 60. / gamma # here gamma obtained from trial and error to match desired time averaging with actual averaging
    # last_mask_array = np.empty_like(vis_chan_range, dtype=bool)
    dims = np.zeros_like(vis_chan_range)
    # find number of LAST bins within last_average_period, and index array accordingly
    test_idx = np.squeeze(np.where((last[:,0] < LAST + last_bound_h) & (last[:,0] > LAST - last_bound_h)))
    vis_last_chan_range = dims[test_idx,:,:,:]
    actual_avg_all =[]
    for day in range(last.shape[1]):
        # getting indices of LAST bins that are within last_average_period
        chosen_last_idx = np.squeeze(np.where((last[:,day] < LAST + last_bound_h) & (last[:,day] > LAST - last_bound_h)))
        # finding difference between first and last LAST bins used for averaging
        actual_avg = (last[:,day][chosen_last_idx[-1]] - last[:,day][chosen_last_idx[0]]) * 60**2
        # adding all periods to an array for averaging later to check
        actual_avg_all.append(actual_avg)
        # print('Actual average in LAST for chosen day(s) '+str(days[i])+' is '+str(actual_avg)+' seconds.')
        vis_last_chan_range[:,day,:,:] = vis_chan_range[chosen_last_idx,day,:,:]
    print('Actual average in LAST for chosen days is ~'+str(np.int(np.mean(actual_avg_all)))+' seconds.')
    if (np.max(actual_avg_all) - np.min(actual_avg_all)) > 0.1:
        raise ValueError('Combined exposure by joining consecutive sessions inconsistent day to day. Check actual_avg_all values.')
# chosing specific LAST
elif not last_average:
    a = 0
    vis_last_chan_range = vis_chan_range[find_nearest(last[:,0], LAST)[0], :, :, :]

# 0. + 0.j in vis_last_chan_range


# vis_flagged = vis_range[flags] # flagging visibilities - flagging currently not working...
vis_amps = np.absolute(vis_last_chan_range) # computing visibility amplitude
dims = np.zeros_like(vis_amps)

# test by plotting visibility amplitude vs channel

def plot_mean_vis():
    plt.figure()
    # mean over LAST, days and baselines
    plt.plot(channels, np.mean(np.mean(np.mean(np.squeeze(vis_amps), axis=0), axis=0), axis=0))
    plt.xlabel('Channel')
    plt.ylabel('Visibility Amplitude')
        # plt.savefig('test.pdf', format='pdf')
    plt.ion()
    plt.show()

def plot_single_bl_vis(time_idx, day_idx, bl_idx):
    plt.figure()
    plt.plot(channels, vis_amps[time_idx, day_idx, bl_idx, :])
    plt.xlabel('Channel')
    plt.ylabel('Visibility Amplitude')
    plt.ion()
    plt.show()

# plot_mean_vis()
# plot_single_bl_vis(3, 0, 10)


# sigma clipping of visibilities over days and baselines, with sigma clipping done about the median value for the data
# sigma clipping over days
if sigma_clip_days: # sigma clip over days - leaving LAST bins, baselines and frequencies intact
    # what does axis here mean? does not seem to clip as required
    # vis_amps = astropy.stats.sigma_clip(vis_amps, sigma=sc_days_stds, maxiters=None, cenfunc='median', stdfunc='std', axis=1, masked=True, return_bounds=False)
    dummy = dims
    for bl in range(0, vis_amps.shape[1+a]): # iterate over baselines
        for frq in range(0, vis_amps.shape[2+a]): # iterate over frequencies
            # or use:
            # vis_amps[i,:,j,k] = scipy.stats.sigmaclip(vis_amps[i,:,j,k], low=sc_days_stds, high=sc_days_stds)
            # although this does not mask array, does not retain dimensionality
            if last_average: # dimensions of vis_amps depends on last_average (+1 dimensions if True)
                for lst in range(0, vis_amps.shape[0]): # iterate over last
                    l = astropy.stats.sigma_clip(vis_amps[lst,:,bl,frq], sigma=sc_days_stds, maxiters=None, cenfunc='median', stdfunc='std', masked=True, return_bounds=False)
                    np.ma.set_fill_value(l,-999)
                    dummy[lst,:,bl,frq] = l.filled()
            elif not last_average:
                l = astropy.stats.sigma_clip(vis_amps[:,bl,frq], sigma=sc_days_stds, maxiters=None, cenfunc='median', stdfunc='std', masked=True, return_bounds=False)
                np.ma.set_fill_value(l,-999)
                dummy[:,bl,frq] = l.filled()
    vis_amps = np.ma.masked_where(dummy==-999, dummy)

if -999 in vis_amps.data:
    days_clipped = True
    count_days_clipped = len(np.where(vis_amps.data == -999)[0])
    print('Day clipping has been applied.')
else:
    days_clipped = False
    print('No day clipping applied; all data within '+str(sc_days_stds)+' sigma for each day.')


# sigma clipping over baselines
if sigma_clip_bl: # sigma clip over days - leaving LAST bins, baselines and frequencies intact
    # vis_amps = astropy.stats.sigma_clip(vis_amps, sigma=sc_bls_stds, maxiters=None, cenfunc='median', stdfunc='std', axis=2, masked=False, return_bounds=False)
    dummy_2 = dims
    for day in range(0, vis_amps.shape[0]): # iterate over days
        for frq in range(0, vis_amps.shape[2]): # iterate over frequencies
            if last_average:
                for lst in range(0, vis_amps.shape[0]): # iterate over last
                    l = astropy.stats.sigma_clip(vis_amps[lst,day,:,frq], sigma=sc_bls_stds, maxiters=None, cenfunc='median', stdfunc='std', masked=True, return_bounds=False)
                    np.ma.set_fill_value(l,-999)
                    dummy_2[lst,day,:,frq] = l.filled()
            elif not last_average:
                l = astropy.stats.sigma_clip(vis_amps[day,:,frq], sigma=sc_bls_stds, maxiters=None, cenfunc='median', stdfunc='std', masked=True, return_bounds=False)
                np.ma.set_fill_value(l,-999)
                dummy_2[day,:,frq] = l.filled()
    vis_amps = np.ma.masked_where(dummy_2==-999, dummy_2)

if -999 in vis_amps.data:
    if days_clipped:
        if len(np.where(vis_amps.data == -999)[0]) > count_days_clipped:
            print('Baseline clipping has been applied.')
        else:
            print('No baseline clipping applied; all data within '+str(sc_days_stds)+' sigma for each day.')
    elif not days_clipped:
        print('No baseline clipping applied; all data within '+str(sc_days_stds)+' sigma for each baseline.')


# statistic over LAST
if last_average:
    if mean_last:
        vis_amps = np.ma.mean(vis_amps, axis=0)
    elif median_last:
        vis_amps = np.ma.median(vis_amps, axis=0)

# chosing single day
if len(chosen_days) == 1:
    vis_amps_single_day = vis_amps[np.where(days == IDRDay)[0][0] ,:,:]


# splitting array of days into 2 (for cross power spectrum between mean/median of two halves)
vis_amps_halves = np.array_split(vis_amps,2, axis=0) # split visibilities array into two sub-arrays of (near) equal days
day_halves = np.array_split(days,2)


# mean statistic over days
def day_statistic(vis_array, selected_days, statistic_method = days_statistic):
    if statistic_all_IDR2:
        vis_avg = eval('np.ma.' + statistic_method)(vis_array, axis=0)
    else:
        day_flags = [] # removing days that do not appear in IDR2
        for i in range(0,len(selected_days)):
            day_flags.append(selected_days[i] in chosen_days)
        day_flags = np.array(day_flags, dtype=bool)
        vis_avg = eval('np.ma.' + statistic_method)(vis_array[day_flags, :, :], axis=0)
    return vis_avg


vis_half1 = day_statistic_new(vis_amps_halves[0], day_halves[0]).data
vis_half2 = day_statistic(vis_amps_halves[1], day_halves[1]).data

vis_amps_final =  day_statistic(vis_amps, days).data


def power_spectrum(data1, data2 = None, window = 'hann', length = None, scaling = 'spectrum', detrend = False):

    # CPS
    if data2 is not None:
        # Finding dimension of returned delays
        delay_test, Pxy_spec_test = signal.csd(data1[1,:], data2[1,:], fs=1./resolution, window=window, scaling=scaling, nperseg=length, detrend=detrend)
        delayshape = delay_test.shape[0] #how many data points the signal.periodogram calculates

        vis_ps = np.zeros((data1.shape[0], 2, delayshape)) # dimensions are [baselines, (delay, Pxx_spec), delayshape]
        for i in range(0, data1.shape[0]): # Iterating over all baselines
            delay, Pxy_spec = signal.csd(data1[i,:], data2[i,:], fs=1./resolution, window=window, scaling=scaling, nperseg=length, detrend=detrend)
            Pxy_spec = np.absolute(Pxy_spec)
            vis_ps[i,:,:] = [delay, Pxy_spec]
        return vis_ps

    # PS
    else:
        # Finding dimension of returned delays
        delay_test, Pxx_spec_test = signal.periodogram(data1[1,:], fs=1./resolution, window=window, scaling=scaling, nfft=length, detrend=detrend)
        delayshape = delay_test.shape[0] #how many data points the signal.periodogram calculates
        # print(delayshape)
        vis_ps = np.zeros((data1.shape[0], 2, delayshape)) # dimensions are [baselines, (delay, Pxx_spec), delayshape]
        for i in range(0, data1.shape[0]): # Iterating over all baselines
            delay, Pxx_spec = signal.periodogram(data1[i,:], fs=1./resolution, window=window, scaling=scaling, nfft=length, detrend=detrend)
            vis_ps[i,:,:] = [delay, Pxx_spec]
            # vis_ps[i,0,:] = delay
            # vis_ps[i,1,:] = Pxx_spec
        return vis_ps


# def compute_cps():
#     # Length of the FFT used
#     infft = 2**7
#
#     # Finding dimension of returned delays
#     delay_test, Pxy_spec_test = signal.csd(vis_half1[1,:], vis_half2[1,:], fs=1./resolution, window='flattop', scaling='spectrum', nperseg=infft, detrend='linear')
#     delayshape = delay_test.shape[0] #how many data points the signal.periodogram calculates
#     # print(delayshape)
#     vis_cps = np.zeros((vis_half1.shape[0], 2, delayshape)) # dimensions are [baselines, (delay, Pxx_spec), delayshape]
#     for i in range(0, vis_half1.shape[0]): # Iterating over all baselines
#         delay, Pxy_spec = signal.csd(vis_half1[i,:], vis_half2[i,:], fs=1./resolution, window='flattop', scaling='spectrum', nperseg=infft, detrend='linear')
#         Pxy_spec = np.absolute(Pxy_spec)
#         vis_cps[i,:,:] = [delay, Pxy_spec]
#     return vis_cps
# # cps returns complex Pxy, since Im parts won't cancel out
#
# def compute_ps():
#     # Length of the FFT used
#     infft = 2**7
#
#     # Finding dimension of returned delays
#     delay_test, Pxx_spec_test = signal.periodogram(vis_amps_final[1,:], fs=1./resolution, window='flattop', scaling='spectrum', nfft=infft, detrend='linear')
#     delayshape = delay_test.shape[0] #how many data points the signal.periodogram calculates
#     # print(delayshape)
#     vis_ps = np.zeros((vis_amps_final.shape[0], 2, delayshape)) # dimensions are [baselines, (delay, Pxx_spec), delayshape]
#     for i in range(0, vis_amps_final.shape[0]): # Iterating over all baselines
#         delay, Pxx_spec = signal.periodogram(vis_amps_final[i,:], fs=1./resolution, window='flattop', scaling='spectrum', nfft=infft, detrend='linear')
#         vis_ps[i,:,:] = [delay, Pxx_spec]
#         # vis_ps[i,0,:] = delay
#         # vis_ps[i,1,:] = Pxx_spec
#     return vis_ps

# window functions: boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann, kaiser

cps = power_spectrum(vis_half1, vis_half2, window = 'boxcar', length = None, scaling = 'spectrum', detrend = False) # spectrum
cps_stat_ew = np.median(cps, axis=0)

# bang units in dimensionless PS? check hera pspec for guidance
# plot multiple graphs on same plot
def plot_stat_ps(stat, rms=False):
    plt.figure()
    # for data in args:
    if str(stat) =='cpsd_mean_ew':
        plt.semilogy(stat[0]*1e6, stat[1])
        plt.ylabel('Cross power spectrum [Amp**2/s]')
    else:
        if rms:
            plt.semilogy(stat[0]*1e6, np.sqrt(stat[1]))
            plt.ylabel('Cross power spectrum [Amp RMS]')
        else:
            plt.semilogy(stat[0]*1e6, stat[1])
            plt.ylabel('Cross power spectrum [Amp**2]')
    plt.xlabel('Geometric delay [$\mu$s]')
    if str(stat) == 'cps_mean_ew':
        plt.title('Mean cross power spectrum over baselines')
    elif str(stat) == 'cps_median_ew':
        plt.title('Median cross power spectrum over baselines')
    # plt.savefig('test.pdf', format='pdf')
    # plt.ion()
    plt.show()

plot_stat_ps(cps_stat_ew, rms=False)

# def plot_mean_cps():
#     plt.figure()
#     plt.semilogy(cps_mean_ew[0]*1e6, np.sqrt(cps_mean_ew[1]))
#     plt.xlabel('Geometric delay [$\mu$s]')
#     plt.ylabel('Cross power spectrum [Amp RMS]')
#     plt.title('Mean cross power spectrum over baselines')
#     # plt.savefig('test.pdf', format='pdf')
#     plt.ion()
#     plt.show()
# def plot_median_cps():
#     plt.figure()
#     plt.semilogy(cps_median_ew[0]*1e6, np.sqrt(cps_median_ew[1]))
#     plt.xlabel('Geometric delay [$\mu$s]')
#     plt.ylabel('Cross power spectrum [Amp RMS]')
#     # plt.savefig('test.pdf', format='pdf')
#     plt.title('Median cross power spectrum over baselines')
#     plt.ion()
#     plt.show()


def factors(n):
    return np.asarray(sorted(functools.reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))


def plot_size(x):
    no_rows = int(find_nearest(factors(x), x/2.)[1])
    # if no_rows == 1:
    #   statement to break up into multiple rows etc.
    no_cols = int(x / no_rows)
    return no_rows, no_cols
plot_dim = plot_size(len(baselines))


def baseline_analysis(ps):
    fig_path = '/rds/project/bn204/rds-bn204-asterics/mdm49/baseline_analysis.pdf'
    if os.path.isfile(fig_path):
        os.remove(fig_path)
    no_rows = plot_dim[0]
    no_cols = plot_dim[1]
    fig, axs = plt.subplots(nrows=7, ncols=5, sharex=True, sharey=True, squeeze=False)
    for row in range(no_rows):
        for col in range(no_cols):
            axs[row,col].semilogy(ps[(row*no_cols)+col,0,:]*1e6, ps[(row*no_cols)+col,1,:], linewidth=1)
            axs[row,col].legend([str(baselines[(row*no_cols)+col])],loc="upper right", prop={'size': 6}, frameon=False)
            axs[row,col].set_xticks(np.arange(0,6))
            axs[row,col].set_xticks(np.arange(0,6,0.2), minor=True)
            axs[row,col].set_yticks(np.power(np.ones(5)*10,-np.arange(1,10,2)))
            minors = []
            for p in range(6):
                for q in range(2,10):
                    minors.append(q * (10**(-1*p)))
            minors = np.array(minors)
            # axs[row,col].set_yticks(minors, minor=True)
            axs[row,col].xaxis.set_tick_params(width=1)
            axs[row,col].yaxis.set_tick_params(width=1)
    plt.suptitle('(Cross) power spectrum for all E-W baselines', y=0.95)
    fig.text(0.5, 0.04, 'Geometric delay [$\mu$s]', ha='center')
    fig.text(0.04, 0.5, 'Cross power spectrum [Amp**2]', va='center', rotation='vertical')
    # fig.tight_layout()
    fig.set_size_inches(w=11,h=7.5)
    plt.savefig(fig_path, format='pdf', dpi=300)
    plt.ion()
    plt.show()
baseline_analysis(cps)

# def baseline_analysis_old(ps):
#     os.remove('/rds/project/bn204/rds-bn204-asterics/mdm49/baseline_analysis.pdf')
#     plt.figure()
#     label_size = 6
#     matplotlib.rcParams['xtick.labelsize'] = label_size
#     # plt.tight_layout
#     for i in range(0,len(baselines)):
#         plt.subplot(7,5,i+1)# sharex=True, sharey=True)
#         plt.semilogy(ps[i,0,:]*1e6, np.sqrt(ps[i,1,:]))
#         plt.legend([str(baselines[i])],loc="upper right", prop={'size': 6})#, fontsize='x-small')
#     plt.subplots_adjust(hspace=0.6, wspace=0.6)
#     # plt.xlabel('Geometric delay [$\mu$s]')
#     # plt.ylabel('Power spectrum [Amp RMS]')
#     plt.suptitle('(Cross) power spectrum for all E-W baselines')
#     plt.savefig('baseline_analysis.pdf', format='pdf')
#     plt.ion()
#     plt.show()

# rm ~/Desktop/baseline_analysis.pdf
# scp mdm49@login-cpu.hpc.cam.ac.uk:/rds/project/bn204/rds-bn204-asterics/mdm49/baseline_analysis.pdf ~/Desktop/


# Add baseline functionality for EW, NS, 14m, 28m, individual baselines etc. - although this done before export to npz?
# Flagging of visibilities
# averaging over LAST range? (40 second range max)
