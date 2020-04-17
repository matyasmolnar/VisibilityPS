"""Main power spectrum analysis script"""


from ps_functions import plot_stat_vis, plot_single_bl_vis, baseline_vis_analysis, \
baseline_ps_analysis


def main():

    plot_stat_vis(np.absolute(vis_analysis), 'mean', 'median', clipped=False,
                  last_avg=False, figname=os.path.join(fig_path, \
                  'mean_and_median_vis_{}.pdf'.format(fig_suffix)))

    plot_single_bl_vis(data=np.absolute(vis_chans_range), time=LAST, JD_day=2458101, \
                       bl=[82, 83])

    baseline_vis_analysis(np.ma.masked_array(np.absolute(vis_amps_final.data), \
    mask=vis_amps_final.mask, dtype=float))

    baseline_ps_analysis(vis_ps_final)
