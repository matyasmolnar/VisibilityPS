"""Plotting functions for power spectra"""


import numpy as np
from matplotlib import pyplot as plt


# Compare plots before after clipping and last averaging
def plot_stat_vis(data, statistics, last_avg=False, fig_name=None):
    plt.figure(figsize=(10, 7))
    stat_fn = np.ma
    if isinstance(statistics, str):
        statistics = [statistics]
    for statistic in statistics:
        if not last_avg:
            plt.plot(chans_range+1, getattr(stat_fn, statistic)(getattr(stat_fn, statistic)
            (getattr(stat_fn, statistic)(np.squeeze(data), axis=0), axis=0), axis=0), \
            label=statistic)
        elif last_avg:
            plt.plot(chans_range+1, getattr(stat_fn, statistic)(getattr(stat_fn, statistic)
            (np.squeeze(data), axis=0), axis=0), label=statistic)
    plt.xlabel('Channel')
    plt.ylabel('Visibility Amplitude')
    plt.legend(loc='upper right')
    if last_avg:
        last_title = ' and LAST averaging'
    else:
        last_title = ''
    plt.title('Statistic of visibility amplitudes over baselines and days'.\
              format(last_title))
    if savefigs:
        plt.savefig(fig_name, format='pdf', dpi=300)
    plt.ion()
    plt.show()


def plot_single_bl_vis(data, time, JD, bl):
    """Plot visibility amplitudes for single baseline"""
    plt.figure(figsize=(10, 7))
    plt.plot(chans_range+1, data[find_nearest(last_dayflg[:, 0], time)[1],
             np.ndarray.tolist(days_dayflg).index(day), \
             np.ndarray.tolist(baselines_dayflg_blflg).index(bl), :])
    plt.xlabel('Channel')
    plt.ylabel('Visibility amplitude')
    plt.title('Amplitudes from baselines {} at JD {} at LAST {}'.format(bl, JD_day, time))
    plt.ion()
    plt.show()


def plot_stat_ps(data, figname, statistics, scaling='spectrum'):
    """Plotting the mean and/or median power spectra"""
    plt.figure()
    if isinstance(statistics, str):
        statistics = [statistics]
    for statistic in statistics:
        stat = getattr(np, statistic)(data, axis=0)
        plt.semilogy(np.real(stat[0]*1e6), np.abs(stat[1]), label=statistic)
        if scaling == 'density':
            plt.ylabel('Cross power spectral density [Amp**2/s]')
        elif scaling == 'spectrum':
            plt.ylabel('Cross power spectrum [Amp**2]')
        else:
            raise ValueError('Choose either spectrum of density for scaling.')
        plt.legend(loc='upper right')
        plt.xlabel('Geometric delay [$\mu$s]')
        plt.title('Cross power spectrum over E-W baselines')
    if savefigs:
        plt.savefig(figname, format='pdf', dpi=300)
    plt.ion()
    plt.show()


def factors(n):
    facs = np.asarray(sorted(functools.reduce(list.__add__, ([i, n//i] for i in \
                   range(1, int(np.sqrt(n)) + 1) if n % i == 0))))
    return facs


def plot_size(x):
    no_rows = int(find_nearest(factors(x), x/2.)[0])
    no_cols = int(x / no_rows)
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


def baseline_ps_analysis(ps, fig_path, fig_name, fig_name='ps_bl_analysis', \
                         save_fig=False):
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
