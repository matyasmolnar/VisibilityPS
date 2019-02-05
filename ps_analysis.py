import sys, os, casac
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pickle
from __future__ import print_function

# Input parameters:
LAST = 3.31 # float
IDRDay = 2458098 # int
Average_IDR2 = True

# # Loading npz file of single visibility dataset:
# npz_file = np.load('/rds/project/bn204/rds-bn204-asterics/mdm49/IDRDays/zen.2458099.46852.xx.HH.ms.npz')
# npz_file.files #['bl', 'vis', 'flags', 'LAST']
# vis_data['bl'] #gives baselines (of which there are 35) - same ones as specified in inBaselines in master_conversion_python.py
# vis_data['vis'].shape #(35, 1, 1024, 60) #(baselines, ??, channels, LAST times)
# vis_data['flags'] #same dimensions as vis
# vis_data['LAST'].shape #60 - number of integrations in the observation session

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]



# Loading npz file of single visibility dataset:
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
channels = np.arange(0,1024, dtype=int)
frc=np.zeros((2,1024))
frc[0,:] = channels
frc[1,:] = freqs

if Average_IDR2 = True:
    vis_avg = np.mean(visibilities, axis=1, dtype=np.complex_)
    vis_medfilt = signal.medfilt(visibilities)

# To Do
# Modify PS function to work with above arrays
# Add averaging functionality and chosing IDRDay, LST etc
# Add baseline functionality for EW, NS, 14m, 28m, individual baselines etc. - although this done before export to npz?

def compute_ps():
    # Length of the FFT used
    infft = 2**7

    # Finding dimension of returned delays
    delay_test, Pxx_spec_test = signal.periodogram(vis_amps[1,:], 1./resolution, 'flattop', scaling='spectrum', nfft=infft)
    delayshape = delay_test.shape[0] #how many data points the signal.periodogram calculates
    # print(delayshape)
    vis_ps = np.zeros((vis_amps.shape[0], 2, delayshape)) # dimensions are [baselines, (delay, Pxx_spec), delayshape]
    for i in range(0, vis_amps.shape[0]): # Iterating over all baselines
        delay, Pxx_spec = signal.periodogram(vis_amps[i,:], 1./vis_res, 'flattop', scaling='spectrum', nfft=infft, detrend='linear')
        vis_ps[i,:,:] = [delay, Pxx_spec]
        # vis_ps[i,0,:] = delay
        # vis_ps[i,1,:] = Pxx_spec
    return vis_ps

power_spectra = compute_ps()

print "power_spectra numpy ndarray has dimensions", power_spectra.shape
