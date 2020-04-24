"""Plotting functions for power spectra"""


import functools

import numpy as np
from matplotlib import pyplot as plt


vis_type_dict = {'amp':'abs', 'phase':'angle'}


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
        # stat_vis = stat(stat(stat(ma_vis, axis=0), axis=0), axis=0)
        plt.plot(chans+1, stat_vis, label=statistic)
    plt.xlabel('Channel')
    plt.ylabel('Visibility {}'.format(vis_type))
    plt.legend(loc='upper right')
    plt.title('Statistic of visibility amplitudes over baselines, days and LASTs')
    if savefig:
        plt.savefig(fig_name, format='pdf', dpi=300)
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
            plt.savefig(fig_name, format='pdf', dpi=300)
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
    plt.figure(figsize=(12, 8))
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
        plt.savefig(figname, format='pdf', dpi=300)
    plt.ion()
    plt.show()


def factors(int_num):
    """Finds the factors of a given integer

    :param n: Number to factorize
    :type n: int

    :return: Array of all possible factors
    :rtype: ndarray
    """
    facs = np.asarray(sorted(functools.reduce(list.__add__, ([i, int_num//i] \
        for i in range(1, int(np.sqrt(int_num)) + 1) if int_num % i == 0))))
    return facs


def plot_size(no_plots):
    """Returns subplot dimensions for a given number of subplots

    :param no_plots: Number of subplots required
    :type no_plots: int

    :return: Subplot nrows and ncols dimensions
    :rtype: tuple
    """
    no_rows = int(find_nearest(factors(no_plots), no_plots/2.)[0])
    no_cols = int(no_plots / no_rows)
    return no_rows, no_cols


def baseline_vis_analysis(data, fig_path, fig_name='vis_bl_analysis', \
                          save_fig=False):
    """Per baseline visibility analysis"""
    # should be plot_size(data[***]) - need to find out which dimension
    no_rows = plot_size()[0]
    no_cols = plot_size()[1]
    fig, axs = plt.subplots(nrows=no_rows, ncols=no_cols, sharex=True, \
                            sharey=True, squeeze=False)
    for row in range(no_rows):
        for col in range(no_cols):
            if (row*no_cols)+col <= len(baselines_dayflg_blflg_clipflg)-1:
                axs[row, col].plot(chans_range+1, data[(row*no_cols)+col, :], \
                                   linewidth=1)
                axs[row, col].legend([str(baselines_dayflg_blflg_clipflg[(
                    row*no_cols)+col])], loc='upper right', prop={'size': 6}, \
                    frameon=False)
                axs[row, col].xaxis.set_tick_params(width=1)
                axs[row, col].yaxis.set_tick_params(width=1)
    plt.suptitle('Visibility amplitudes for all E-W baselines', y=0.95)
    fig.text(0.5, 0.04, 'Channel', ha='center')
    fig.text(0.04, 0.5, 'Visibility amplitude', va='center', rotation='vertical')
    fig.set_size_inches(w=11, h=7.5)
    if save_fig:
        if not fig_name.endswith('.pdf'):
            fig_name = fig_name+'.pdf'
        plt.savefig(os.path.join(fig_path, fig_name, format='pdf', dpi=300))
    plt.ion()
    plt.show()


def baseline_ps_analysis(ps, fig_path, fig_name='ps_bl_analysis', save_fig=False):
    """Per baseline power spectrum analysis"""
    # should be plot_size(data[***]) - need to find out which dimension
    no_rows = plot_size()[0]
    no_cols = plot_size()[1]
    fig, axs = plt.subplots(nrows=no_rows, ncols=no_cols, sharex=True, \
                            sharey=True, squeeze=False)
    fig.set_size_inches(w=11, h=7.5)
    # TODO - automatically get xlims and ylims
    plt.axis(xmin=-0.1, xmax=5.2, ymin=1e-10, ymax=1e-0)
    for row in range(no_rows):
        for col in range(no_cols):
            if (row*no_cols)+col <= len(baselines_dayflg_blflg_clipflg)-1:
                axs[row, col].semilogy(np.real(
                    ps[(row*no_cols)+col, 0, :])*1e6, np.absolute(ps[(row*no_cols)+col, \
                    1, :]), linewidth=1)
                axs[row, col].legend([str(baselines_dayflg_blflg_clipflg[(
                    row*no_cols)+col])], loc='upper right', prop={'size': 6}, \
                    frameon=False)
            if return_onesided_ps:
                axs[row, col].set_xticks(np.arange(0, 7, 2))
                axs[row, col].set_xticks(np.arange(0, 6, 0.2), minor=True)
            else:
                axs[row, col].set_xticks(np.arange(-6, 7, 2))
                axs[row, col].set_xticks(np.arange(-6, 6, 0.5), minor=True)
            axs[row, col].set_yticks(
                np.power(np.ones(5)*10, -np.arange(1, 10, 2)))
    plt.suptitle('(Cross) power spectrum for all E-W baselines', y=0.95)
    fig.text(0.5, 0.04, 'Geometric delay [$\mu$s]', ha='center')
    fig.text(0.04, 0.5, 'Cross power spectrum [Amp**2]', va='center', \
             rotation='vertical')
    if save_fig:
        if not fig_name.endswith('.pdf'):
            fig_name = fig_name+'.pdf'
        plt.savefig(os.path.join(fig_path, fig_name, format='pdf', dpi=300))
    plt.ion()
    plt.show()
