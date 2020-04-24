"""Plotting functions for power spectra"""


import functools

import numpy as np
from matplotlib import pyplot as plt

from vis_utils import find_nearest


vis_type_dict = {'amp':'abs', 'phase':'angle'}


def figname_format(figname, format='pdf'):
    """Format figure name to have correct file extension

    :param figname: Figure name
    :type figname: str
    :param format: Format to save the figure
    :type format: str

    :return: Formatted figure name
    :rtype: str
    """
    ext = '.{}'.format(format)
    if not fig_name.endswith(ext):
        fig_name = fig_name + ext
    return fig_name


def plot_stat_vis(ma_vis, chans, statistics, vis_type='amp', savefig=False, \
                  fig_name='vis_amp_stats.pdf'):
    """Plot averaged visibilities

    Visibilities are averaged (by mean or median statistics) over redundant
    baselines, days, and LASTs, and then plotted.

    :param ma_vis: Masked visibility dataset
    :type ma_vis: MaskedArray
    :param chans: Frequency channels
    :type chans: ndarray
    :param statistics: Statistic to peform over bls, days and lsts {'mean', 'median'}
    :type statistics: str, list
    :param vis_type: Plot amplitude or phase {'amp', 'phase'}
    :type vis_type: str
    :param savefig: Whether to save the figure
    :type savefig: bool
    :param fig_name: Figure name
    :type fig_name: str
    """
    plt.figure(figsize=(10, 7))
    ma_vis =  getattr(np.ma, vis_type_dict[vis_type])(ma_vis)
    if isinstance(statistics, str):
        statistics = [statistics]
    for statistic in statistics:
        stat = getattr(np.ma, statistic)
        # can't do stat(ma_vis, axis=(0, 1, 2)) because median of median != whole median
        stat_vis = ma_vis.copy()
        for stat_dim in range(ma_vis.ndim - 1):
            stat_vis = stat(stat_vis, axis=0)
        plt.plot(chans+1, stat_vis, label=statistic)
    plt.xlabel('Channel')
    plt.ylabel('Visibility {}'.format(vis_type))
    plt.legend(loc='upper right')
    plt.title('Statistic of visibility amplitudes over baselines, days and LASTs')
    if savefig:
        plt.savefig(figname_format(fig_name), format='pdf', dpi=300)
    plt.ion()
    plt.show()


def plot_sample_vis(ma_vis, chans, vis_type='amp', tint_idx=0, day_idx=0, \
    bl_idx=0, savefig=False, fig_name='sample_vis.pdf'):
    """Plot sample visibility

    :param ma_vis: Masked visibility dataset
    :type ma_vis: MaskedArray
    :param chans: Frequency channels
    :type chans: ndarray
    :param vis_type: Plot amplitude or phase {'amp', 'phase'}
    :type vis_type: str
    :param tint_idx: Index of time integration of visibility dataset to sample
    :type tint_idx: int
    :param day_idx: Index of Julian day of visibility dataset to sample
    :type day_idx: int
    :param bl_idx:Index of baseline of visibility dataset to sample
    :type bl_idx: int
    :param savefig: Whether to save the figure
    :type savefig: bool
    :param fig_name: Figure name
    :type fig_name: str
    """
    sample_vis = ma_vis[tint_idx, day_idx, bl_idx, :]
    if sample_vis.mask.all():
        print('All visibilities for the specified indices are flagged')
    else:
        ma_vis = getattr(np.ma, vis_type_dict[vis_type])(ma_vis[tint_idx, day_idx, \
                                                                 bl_idx, :])
        plt.figure(figsize=(10, 7))
        plt.plot(chans+1, ma_vis)
        plt.xlabel('Channel')
        plt.ylabel('Visibility {}'.format(vis_type))
        if savefig:
            plt.savefig(figname_format(fig_name), format='pdf', dpi=300)
        plt.ion()
        plt.show()


def plot_stat_ps(ps_data, statistics, scaling='spectrum', savefig=False, \
                 fig_name='sample_vis.pdf'):
    """Plotting the mean and/or median power spectra

    :param ps_data: Power spectrum results
    :type ps_data: ndarray
    :param statistics: Statistic to peform over bls, days and lsts {'mean', 'median'}
    :type statistics: str, list
    :param scaling: Scaling chosen during PS computation {'spectrum', 'density'}
    :type scaling: str
    :param savefig: Whether to save the figure
    :type savefig: bool
    :param fig_name: Figure name
    :type fig_name: str
    """
    plt.figure(figsize=(10, 7))
    if isinstance(statistics, str):
        statistics = [statistics]
    for statistic in statistics:
        stat_ps = getattr(np, statistic)(ps_data, axis=1)
        plt.semilogy(stat_ps[0].real*1e6, np.abs(stat_ps[1]), label=statistic)
    if scaling == 'spectrum':
        plt.ylabel('Cross power spectrum [Amp**2]')
    if scaling == 'density':
        plt.ylabel('Cross power spectral density [Amp**2/s]')
    plt.xlabel('Geometric delay [$\mu$s]')
    plt.legend(loc='upper right')
    plt.title('Cross power spectrum over E-W baselines')
    if savefig:
        plt.savefig(figname_format(fig_name), format='pdf', dpi=300)
    plt.ion()
    plt.show()


def round_to(x, base):
    """Round number to nearest base

    :param x: Number to round
    :type x: float, int
    :param base: Base
    :type bas: int

    :return: Rounded number
    :rtype: int
    """
    return base * round(x/base)


def per_bl_vis_plot(ma_vis, chans, bls, vis_type='amp', no_cols=5, figsize=(14, 10), \
                    savefig=False, fig_name='vis_per_bl.pdf'):
    """Per baseline visibility analysis

    :param ma_vis: Masked visibility dataset
    :type ma_vis: MaskedArray
    :param chans: Frequency channels
    :type chans: ndarray
    :param bls: Baselines
    :type bls: ndarray
    :param vis_type: Plot amplitude or phase {'amp', 'phase'}
    :type vis_type: str
    :param no_cols: Number of columns to plot
    :type no_cols: int
    :param figsize: Size of figure in inches (w, h)
    :type figsize: tuple
    :param savefig: Whether to save the figure
    :type savefig: bool
    :param fig_name: Figure name
    :type fig_name: str
    """
    no_bls = bls.shape[0]
    no_rows = int(round_to(no_bls, no_cols) / no_cols)
    fig, axs = plt.subplots(nrows=no_rows, ncols=no_cols, sharex=True, \
                            sharey=True, squeeze=False)
    fig.set_size_inches(w=figsize[0], h=figsize[1])
    ma_vis =  getattr(np.ma, vis_type_dict[vis_type])(ma_vis)
    for row in range(no_rows):
        for col in range(no_cols):
            if (row*no_cols)+col <= no_bls-1:
                axs[row, col].plot(chans+1, ma_vis[(row*no_cols)+col, :], \
                                   linewidth=1)
                axs[row, col].legend([str(bls[(row*no_cols)+col])], \
                    loc='upper right', prop={'size': 6}, frameon=False)
                axs[row, col].xaxis.set_tick_params(width=1)
                axs[row, col].yaxis.set_tick_params(width=1)
    plt.suptitle('Visibility {}s for all E-W baselines'.format(vis_type), y=0.95)
    fig.text(0.5, 0.04, 'Channel', ha='center')
    fig.text(0.04, 0.5, 'Visibility {}'.format(vis_type), va='center', \
             rotation='vertical')
    if savefig:
        plt.savefig(figname_format(fig_name), format='pdf', dpi=300)
    plt.ion()
    plt.show()


def per_bl_ps_plot(ps_data, bls, no_cols=5, figsize=(14, 10), savefig=False, \
                   fig_name='ps_per_bl.pdf'):
    """Per baseline power spectrum analysis

    :param ps_data: Power spectrum results
    :type ps_data: ndarray
    :param bls: Baselines
    :type bls: ndarray
    :param no_cols: Number of columns to plot
    :type no_cols: int
    :param figsize: Size of figure in inches (w, h)
    :type figsize: tuple
    :param savefig: Whether to save the figure
    :type savefig: bool
    :param fig_name: Figure name
    :type fig_name: str
    """
    no_bls = bls.shape[0]
    no_rows = int(round_to(no_bls, no_cols) / no_cols)
    fig, axs = plt.subplots(nrows=no_rows, ncols=no_cols, sharex=True, \
                            sharey=True, squeeze=False)
    fig.set_size_inches(w=figsize[0], h=figsize[1])
    delays = ps_data[0, 0, :].real
    onesided = not any(dly < 0 for dly in delays)
    scaling = 1e6
    x_rng= np.ceil(np.max(delays) * 1e6)
    for row in range(no_rows):
        for col in range(no_cols):
            if (row*no_cols)+col <= no_bls-1:
                axs[row, col].semilogy(ps_data[0, (row*no_cols)+col, :].real*scaling, \
                                       np.abs(ps_data[1, (row*no_cols)+col, :]), linewidth=1)
                axs[row, col].legend([str(bls[(row*no_cols)+col])], \
                    loc='upper right', prop={'size': 6}, frameon=False)
                if onesided:
                    axs[row, col].set_xticks(np.arange(0, x_rng+1, 2))
                    axs[row, col].set_xticks(np.arange(0, x_rng, 0.2), minor=True)
                else:
                    axs[row, col].set_xticks(np.arange(-x_rng, x_rng+1, 2))
                    axs[row, col].set_xticks(np.arange(-x_rng, x_rng, 0.5), minor=True)
                axs[row, col].set_yticks(np.power(np.ones(5)*10, -np.arange(1, 10, 2)))
    plt.suptitle('Power spectrum for all E-W baselines', y=0.95)
    fig.text(0.5, 0.04, 'Geometric delay [$\mu$s]', ha='center')
    fig.text(0.04, 0.5, 'Power spectrum [Amp**2]', va='center', \
             rotation='vertical')
    if savefig:
        plt.savefig(figname_format(fig_name), format='pdf', dpi=300)
    plt.ion()
    plt.show()
