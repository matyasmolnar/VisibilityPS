"""HERA Visibility PS Computation Functions

Collection of functions that take aligned HERA visibilities in LAST (as outputted by
align_lst) and compute various PS estimates using simple statistics over LASTs,
days and baselines.
"""


import os
from heapq import nsmallest

import astropy.stats
import numpy as np
from scipy import signal
from scipy.stats import median_absolute_deviation as mad
from scipy.stats.mstats import gmean

from vis_utils import find_nearest


def mod_zscore(arr):
    """Modified z-score, as defined by Iglewicz and Hoaglin

    :param arr: Array
    :type arr: array-like

    :return: Modified z-scores of the elements of the input array
    :type: ndaray
    """
    return 0.6745*(np.asarray(arr) - np.median(arr))/mad(arr)


def find_mis_days(last, jd_days, mod_z_tresh=3.5):
    """Find misaligned days by computing the modified z-scores of the first and
    last LAST timestamps, and discarding any days that have a modified z-score
    that exceeds the treshold.

    :param last: LAST
    :type last: float
    :param mod_z_tresh: Threshold of modified z-score to discard a day of data
    :type mod_z_thresh: float

    :return: Misaligned days
    :rtype: ndarray
    """
    start_last, end_last = zip(*[(last[0, i], last[-1, i]) for i in \
                                 range(last.shape[1])])
    start_zscores = np.where(mod_zscore(start_last) > mod_z_tresh)[0]
    end_zscores = np.where(mod_zscore(start_last) > mod_z_tresh)[0]
    mis_days_idx = list(set(start_zscores & end_zscores))
    mis_days = np.asarray(jd_days)[mis_days_idx]
    if mis_days:
        print('Misaligned days: {} - check alignment'.format(mis_days))
    return mis_days


def sig_clip(ma_vis, clip_dim, cenfunc='median', sigma=5.0, clip_rule='amp'):
    """Sigma clipping of visibilities over given dimension, with clipping
    done about the mean or median value for the data

    Can perform sigma clipping on the visibilities according to either:
        - Their absolute values
        - Their geometric means
        - Their Re and Im values separately

    :param ma_vis: Masked visibility dataset
    :type ma_vis: MaskedArray
    :param clip_dim: Dimension to sigma clip {'bls', 'days'}
    :type clip_dim: str
    :param cenfunc: Statistic used to compute the center value for the clipping
    {'mean', 'median'}
    :type cenfunc: str
    :param sigma: Number of standard deviations to use for both the lower and
    upper clipping limit
    :type sigma: float
    :param clip_rule: Rule for sigma clipping: on what aspect of the complex data
    should the sigma clipping be applied {'amp', 'gmean', None}
    :type clip_rule: str, None

    :return: Sigma clipped Masked visibility dataset
    :rtype: MaskedArray
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


def clipping(ma_vis, sig_clip_days=True, sig_clip_bls=True, sig_stds=5.0, \
             cenfunc='median', clip_rule='amp'):
    """Apply clipping

    :param ma_vis: Masked visibility dataset
    :type ma_vis: MaskedArrays
    :param sig_clip_days: Whether to perform sigma clipping over days
    :type sig_clip_days: bool
    :param sig_clip_bls: Whether to perform sigma clipping over baselines
    :type sig_clip_bls: bool
    :param sigma: Number of standard deviations to use for both the lower and
    upper clipping limit
    :type sigma: float
    :param cenfunc: Statistic used to compute the center value for the clipping
    {'mean', 'median'}
    :type cenfunc: str
    :param clip_rule: Rule for sigma clipping: on what aspect of the complex data
    should the sigma clipping be applied {'amp', 'gmean', None}
    :type clip_rule: str, None

    :return: Sigma clipped Masked visibility dataset
    :rtype: MaskedArray
    """
    no_day_clip = 0
    if sig_clip_days:
        visibilities = sig_clip(ma_vis, clip_dim='days', cenfunc=cenfunc, \
                                sigma=sig_stds, clip_rule=clip_rule)
        no_day_clip = np.sum(ma_vis.mask)
        print('Day clipping: {} visibilities masked'.format(no_day_clip))

    if sig_clip_bls:
        visibilities = sig_clip(ma_vis, clip_dim='bls', cenfunc=cenfunc, \
                                sigma=sig_stds, clip_rule=clip_rule)
        no_bl_clip = np.sum(ma_vis.mask) - no_day_clip
        print('Baseline clipping has been applied: {} visibilities masked'.\
              format(no_bl_clip))

    if sig_clip_days or sig_clip_bls:
        print('{} visibilities clipped out of {}'.format(np.sum(ma_vis.mask), \
              ma_vis.size))
    else:
        print('No clipping applied')
    return ma_vis


def dim_statistic(ma_vis, statistic, stat_dim):
    """Statistic over dimension

    :param ma_vis: Masked visibility dataset
    :type ma_vis: MaskedArray
    :param statistic: Statistic used on the dataset {'mean', 'median'}
    :type statistic: str
    :param stat_dim: Dimension on which to apply statistic. Can either be int,
    which represents axis, or str which represents the meaning of the dimension
    :type stat_dim: int, str

    :return: Statistic of masked visibibilities over the specified dimension
    :rtype: MaskedArray
    """
    if not isinstance(stat_dim, int):
        stat_dim = dim_dict[stat_dim]
    vis_stat = getattr(np.ma, statistic)(ma_vis, axis=stat_dim)
    return vis_stat


def get_ps_dim(vis, return_onesided):
    """Returns an empty array with the required dimensions to fit the results
    of the power spectra, when iterated over baselines

    :param vis: Visibility dataset
    :type vis: ndarray
    :param return_onesided: Whether to return a one-sided spectrum. For complex
    data, a two-sided spectrum is always returned.
    :type return_onesided: bool

    :return: Empty array with the required shape to fill in the results from the
    power spectra computations
    :rtype: ndarray
    """
    if return_onesided and not np.iscomplexobj(vis):
        pspec_dim = int(np.ceil(vis.shape[1]/2))
    else:
        pspec_dim = vis.shape[1]
    vis_ps = np.empty((2, vis.shape[0], pspec_dim))
    return vis_ps


def cps(vis1, vis2, resolution, window='hann', length=None, scaling='spectrum', \
        detrend=False, return_onesided=False):
    """Cross power spectrum computation of visibilities

    Returns ndarray of delay and visibility cross power spectrum/power spectral
    density, for each baseline. The dimensions of the resulting array are [2, bls, frqs],
    where vis_ps[0, :, :] = delays and vis_ps[1, :, :] = Pxy_specs.

    :param vis1: Visibility dataset 1
    :type vis1: ndarray
    :param vis2: Visibility dataset 2
    :type vis2: ndarray
    :param resolution: resolution of visibilities (frequency gap between
    successive channels). This is used to calculate the sampling frequency.
    :type resolution: float
    :param window: FFT window
    :type window: str
    :param scaling: Selects between computing the power spectral density (units
    Amp**2/Hz) and the power spectrum (units of Amp**2) {'density', 'spectrum')
    :type scaling: str
    :param length: Length of the FFT used. If None the length of the vis freq
    axis will be used.
    :type length: int, None
    :param detrend: Specifies how to detrend each segment
    :type detrend: str, False
    :param return_onesided: Whether to return a one-sided spectrum. For complex
    data, a two-sided spectrum is always returned.
    :type return_onesided: bool

    :return: Delay and visibility power spectra
    :rtype: ndarray
    """
    assert vis1.size == vis2.size
    vis_cps = get_ps_dim(vis1, return_onesided).astype(complex)
    for bl in range(vis1.shape[0]): # Iterating over baselines
        delay, Pxy_spec = signal.csd(vis1[bl, :], vis2[bl, :], fs=1./resolution, \
            window=window, scaling=scaling, nperseg=length, detrend=detrend, \
            return_onesided=return_onesided)
        delay_sort = np.argsort(delay)
        vis_cps[:, bl, :] = [delay[delay_sort], Pxy_spec[delay_sort]]
    return vis_cps


def ps(vis, resolution, window='hann', scaling='spectrum', length=None, \
       detrend=False, return_onesided=False):
    """Power spectrum computation of visibilities

    Returns ndarray of delay and visibility power spectrum/power spectral density,
    for each baseline. The dimensions of the resulting array are [2, bls, frqs],
    where vis_ps[0, :, :] = delays and vis_ps[1, :, :] = Pxx_specs.

    :param vis: Visibility dataset
    :type vis: ndarray
    :param resolution: resolution of visibilities (frequency gap between
    successive channels). This is used to calculate the sampling frequency.
    :type resolution: float
    :param window: FFT window
    :type window: str
    :param scaling: Selects between computing the power spectral density (units
    Amp**2/Hz) and the power spectrum (units of Amp**2) {'density', 'spectrum')
    :type scaling: str
    :param length: Length of the FFT used. If None the length of the vis freq
    axis will be used.
    :type length: int, None
    :param detrend: Specifies how to detrend each segment
    :type detrend: str, False
    :param return_onesided: Whether to return a one-sided spectrum. For complex
    data, a two-sided spectrum is always returned.
    :type return_onesided: bool

    :return: Delay and visibility power spectra
    :rtype: ndarray
    """
    vis_ps = get_ps_dim(vis, return_onesided)
    for bl in range(vis.shape[0]): # Iterating over baselines
        delay, Pxx_spec = signal.periodogram(vis[bl, :], fs=1./resolution, \
            window=window, scaling=scaling, nfft=length, detrend=detrend, \
            return_onesided=return_onesided)
        delay_sort = np.argsort(delay)
        vis_ps[:, bl, :] = [delay[delay_sort], Pxx_spec[delay_sort]]
    return vis_ps
