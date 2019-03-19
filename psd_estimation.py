import numpy as np
import scipy.fftpack as sf

def psd_est(tsamp, data, dwei, ncorr=None, use_fft=True):
    '''Make estimate of power spectral density.'''

    nsamp = len(data)
    if ncorr is None:
        ncorr = nsamp

    # Get raw correlation function of data and weights.

    if use_fft:
        corr, cwei = corrfn_fft(data, dwei, ncorr=ncorr)
    else:
        corr, cwei = corrfn_rsp(data, dwei, ncorr=ncorr)

    # Get unbiassed correlation function by dividing by weights.

    if np.any(cwei == 0.0):
        # Some weights are zero, so avoid these points in the division
        # and interpolate the correlation function afterwards.
        good = (cwei != 0.0)
        corr[good] /= cwei[good]
        ind = np.arange(ncorr)
        corr[~good] = np.interp(ind[~good], ind[good], corr[good])
    else:
        corr /= cwei

    # Apply window function.

    i = np.arange(ncorr)
    #w = 1.0 - i/ncorr # Triangular
    w = 0.5 * (1.0 + np.cos(np.pi*i/ncorr)) # Hann
    corr *= w

    # Get PSD from correlation function.

    nfft = sf.next_fast_len(2*ncorr-1)
    tmp = np.zeros(nfft)
    tmp[:ncorr] = corr
    tmp[-ncorr+1:] = corr[-1:0:-1]
    psd = np.fft.rfft(tmp, n=nfft).real * tsamp

    return psd

def corrfn_fft(data, dwei, ncorr=None):
    '''Compute raw correlation function using FFT.'''
    nsamp = len(data)
    if ncorr is None:
        ncorr = nsamp
    nfft = sf.next_fast_len(2*nsamp-1)
    data_ps = np.abs(np.fft.rfft(data, n=nfft))**2
    dwei_ps = np.abs(np.fft.rfft(dwei, n=nfft))**2
    corr = np.fft.irfft(data_ps, n=nfft)[0:ncorr]
    cwei = np.fft.irfft(dwei_ps, n=nfft)[0:ncorr]
    return corr, cwei

def corrfn_rsp(data, dwei, ncorr=None):
    '''Compute raw correlation function in real space.'''
    nsamp = len(data)
    if ncorr is None:
        ncorr = nsamp
    corr = np.zeros(ncorr)
    cwei = np.zeros(ncorr)
    for i in range(ncorr):
        corr[i] = np.sum(data[i:]*data[:nsamp-i])
        cwei[i] = np.sum(dwei[i:]*dwei[:nsamp-i])
    return corr, cwei
