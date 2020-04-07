"""Power spectrum computation of visibility amplitudes

Power spectrum computation of the amplitudes of a visibility dataset in
measurement set file format.

TODO:
    - Better way of selecting shortest EW baselines
    - Work with corrected data and compare visibilities before and after calibration
    and calibration + cleaning
"""


import os
import pickle
import sys

import casac
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


DataDir = '/Volumes/TOSHIBA_EXT/HERA_Data/imaging/hera67_2458098.35667_FornaxA/'
InMS = 'zen.2458098.35667.xx.HH.ms'
procdir = DataDir


class Visibility:
    """
    Creates visibility class object.

    Attributes:
        VV          Final data (Re + Im)
        uu          v-coordinates of visibilities (units of lambda)
        vv          v-coordinates of visibilities (units of lambda)
        data_wgts   Data Weights
        freqs       Frequencies
        time        Time
        res         Resolution
        ant1        Antenna 1 (for given baseline)
        ant2        Antenna 2 (for given baseline)
        flags       Flags
    """

    def __init__(self, VV, uu, vv, wgts, freqs, time, res, ant1, ant2, flags):
        self.VV = np.array(VV) # [Jy]
        self.uu = np.array(uu) # [m]
        self.vv = np.array(vv) # [m]
        self.wgts = np.array(wgts) # [Jy^-2]
        self.freqs = np.array(freqs) # [Hz]
        self.time = np.array(time)
        self.res = np.array(res)
        self.ant1 = np.array(ant1)
        self.ant2 = np.array(ant2)
        self.flags = np.array(flags)


def import_data_ms(filename):
    """Imports data from a casa measurement set and returns visibility object"""

    tb = casac.casac.table()
    ms = casac.casac.ms()

    # Antenna information
    tb.open(filename)
    data = tb.getcol("DATA")
    uvw = tb.getcol("UVW")
    weight = tb.getcol("WEIGHT")
    ant1 = tb.getcol("ANTENNA1")
    ant2 = tb.getcol("ANTENNA2")
    flags = tb.getcol("FLAG")
    time = tb.getcol("TIME")
    tb.close()

    # Spectral window information
    ms.open(filename)
    spw_info = ms.getspectralwindowinfo()
    nchan = spw_info["0"]["NumChan"]
    npol = spw_info["0"]["NumCorr"]
    ms.close()

    # Frequency information
    tb.open(filename+"/SPECTRAL_WINDOW")
    freqs = tb.getcol("CHAN_FREQ")
    rfreq = tb.getcol("REF_FREQUENCY")
    resolution = tb.getcol("CHAN_WIDTH")
    tb.close()

    uu = uvw[0, :]
    vv = uvw[1, :]

    # Check if pols are already averaged
    data = np.squeeze(data)
    weight = np.squeeze(weight)
    flags = np.squeeze(flags)

    if npol == 1:
        Re = data.real
        Im = data.imag
        wgts = weight

    else:
        # Polarization averaging
        Re_xx = data[0, :].real
        Re_yy = data[1, :].real
        Im_xx = data[0, :].imag
        Im_yy = data[1, :].imag
        weight_xx = weight[0, :]
        weight_yy = weight[1, :]
        flags = flags[0, :]*flags[1, :]

        # Weighted averages
        with np.errstate(divide='ignore', invalid='ignore'):
            Re = np.where((weight_xx + weight_yy) != 0, (Re_xx * weight_xx + \
                 Re_yy*weight_yy) / (weight_xx + weight_yy), 0.)
            Im = np.where((weight_xx + weight_yy) != 0, (Im_xx * weight_xx + \
                 Im_yy*weight_yy) / (weight_xx + weight_yy), 0.)
        wgts = (weight_xx + weight_yy)

    # Toss out the autocorrelations
    xc = np.where(ant1 != ant2)[0]

    # Check if there's only a single channel
    if nchan == 1:
        data_real = Re[np.newaxis, xc]
        data_imag = Im[np.newaxis, xc]
        flags = flags[xc]
    else:
        data_real = Re[:, xc]
        data_imag = Im[:, xc]
        flags = flags[:, xc]

        # If the majority of points in any channel are flagged, it probably
        # means an entire channel is flagged - spit warning
        if np.mean(flags.all(axis=0)) > 0.5:
            print('WARNING: Over half of the (u,v) points in at least one \
            channel are marked as flagged. If you did not expect this, it is \
            likely due to having an entire channel flagged in the ms. Please \
            double check this and be careful if model fitting or using diff mode.')

        # Collapse flags to single channel, because weights are not currently channelized
        flags = flags.any(axis=0)

    data_wgts = wgts[xc]
    data_uu = uu[xc]
    data_vv = vv[xc]

    ant1 = ant1[xc]
    ant2 = ant2[xc]

    data_VV = data_real + data_imag*1.0j

    # Warning that flagged data was imported
    if np.any(flags):
        print('WARNING: Flagged data was imported. Visibility interpolation can \
        proceed normally, but be careful with chi^2 calculations.')

    return Visibility(data_VV.T, data_uu, data_vv, data_wgts, freqs, time, \
                      resolution, ant1, ant2, flags)


def plot_ps_bl(amps, bl_no):
    """Plot the power spectrum for a single baseline"""
    vis_bl = vis_amps[bl_no,:]
    delay, Pxx_spec = signal.periodogram(vis_bl, 1./vis_res, 'flattop', \
                                         scaling='spectrum', nfft=128)
    plt.figure()
    plt.semilogy(delay, np.sqrt(Pxx_spec))
    plt.title('Visibility amplitude PS for baseline {}'.format(bl_no))
    plt.xlabel('Geometric delay [s]')
    plt.ylabel('Power spectrum [Amp RMS]')
    plt.show()



def compute_ps(amps, res, infft=2**7):
    """Compute power spectra for all baselines

    :param amps: Visibility amplitudes
    :param resolution: Resolution of visibility measurements
    :param infft: Length of the FFT used
    """
    infft = 2**7 # Length of the FFT used
    # Finding dimension of returned delays
    delay_test, Pxx_spec_test = signal.periodogram(
        amps[1, :], 1./res, 'flattop', scaling='spectrum', nfft=infft)
    # How many data points the signal.periodogram calculates
    delayshape = delay_test.shape[0] # dimensions are [baselines, (delay, Pxx_spec), delayshape]
    vis_ps = np.zeros((amps.shape[0], 2, delayshape))
    for i in range(0, amps.shape[0]):  # Iterating over all baselines
        delay, Pxx_spec = signal.periodogram(
            amps[i, :], 1./res, 'flattop', scaling='spectrum', nfft=infft, \
            detrend='linear')
        vis_ps[i, :, :] = [delay, Pxx_spec]
    return vis_ps


def shortest_ew_mask():
    """Creates masks to only include shortest EW baselines"""
    vis_u = vis_data.uu
    vis_v = vis_data.vv
    uv_dist = np.sqrt(np.square(vis_u) + np.square(vis_v))
    shortest_baselines = np.ma.masked_where(
        uv_dist < (np.amin(uv_dist) + 1), vis_u)

    # create mask to only include EW baselines -- constraint on v distance. Not sufficient as v distance changes with time, thus other longer baselines will be included in this.
    # plt.hist(vis_v, bins = 1000)
    # check that vis_v[0] corresponds to shortest baseline between antenna 1 and antenna 2 (do plotants(zen) in CASA)
    ew_baselines = np.ma.masked_where(vis_v < (vis_v[0]*1.5), vis_u)

    # make sure antenna number consecutive
    consec_baselines = np.ma.masked_where(
        np.absolute(vis_data.ant1 - vis_data.ant2) < 1.5, vis_u)

    # total mask selecting only the shortest EW consecutive baselines
    total_mask = shortest_baselines.mask & ew_baselines.mask & consec_baselines.mask & flags
    return total_mask


def main():
    os.chdir(procdir)
    InData = os.path.join(DataDir, InMS)
    vis_class = os.path.splitext(InMS)[0] + 'vis_class.npz'
    if os.path.exists(vis_class):
        with open(vis_class, 'rb') as file:
            vis_data = pickle.load(file)
    else:
        vis_data = import_data_ms(msfile)
        with open(vis_class, 'wb') as file:
            pickle.dump(vis_data, file, pickle.HIGHEST_PROTOCOL)

    flags = vis_data.flags
    vis_freqs = np.squeeze(vis_data.freqs)
    vis_res = vis_data.res.item(0)
    exposure = vis_data.time[-1] - vis_data.time[0]
    ant1 = vis_data.ant1
    ant2 = vis_data.ant2

    # Dimensions (baselines, freq chans) and has complex entries
    vis_amps = np.squeeze(np.absolute(vis_data.VV))[flags, :]

    power_spectra = compute_ps(vis_amps, vis_res)
    print('Power_spectra ndarray has dimensions {}'.format(power_spectra.shape))

    vis_amps_short_ew = vis_amps[total_mask, :]
    power_spectra_short_ew = power_spectra[total_mask, :, :]
    x = np.mean(power_spectra_short_ew, axis=0)

    plt.figure()
    plt.semilogy(x[0]*1e6, np.sqrt(x[1]))
    plt.xlabel('Geometric delay [$\mu$s]')
    plt.ylabel('Power spectrum [Amp RMS]')
    plt.savefig('test.pdf', format='pdf')
    plt.show()

    # Finding which baselines to be used when extracting more visibility data for IDR2
    ant1_ew = ant1[total_mask]
    ant2_ew = ant2[total_mask]
    antenna_pairs = np.zeros((2, ant1_ew.shape[0]))
    antenna_pairs[0, :] = ant1_ew
    antenna_pairs[1, :] = ant2_ew

    ant_pairs_list = np.zeros(ant1_ew.shape[0], dtype=list)
    for i in range(0, ant1_ew.shape[0]):
        ant_pairs_list[i] = [ant1_ew[i], ant2_ew[i]]

    unique_ant_pairs_t = list(sorted(set(tuple(row) for row in ant_pairs_list)))
    unique_ant_pairs = list(list(row) for row in unique_ant_pairs_t)
    print(unique_ant_pairs, end='')
