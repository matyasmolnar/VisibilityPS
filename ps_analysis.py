"""Main power spectrum analysis script"""


from ps_functions import baseline_vis_analysis


def main():
    baseline_vis_analysis(np.ma.masked_array(np.absolute(vis_amps_final.data), \
    mask=vis_amps_final.mask, dtype=float))
    baseline_ps_analysis(vis_ps_final)
