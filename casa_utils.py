"""Set of commonly used casa functions

To be run in a casapython shell.
"""

import os

import casa


def add_hera_obs_pos():
    """Adding HERA observatory position (at some point as PAPER_SA)"""
    obstablename = os.getenv('CASAPATH').split()[0] + '/data/geodetic/Observatories/'
    if not os.path.exists(obstablename + 'HERA'):
        casa.tb.open(obstablename, nomodify=False)
        paperi = (tb.getcol('Name') == 'PAPER_SA').nonzero()[0]
        casa.tb.copyrows(obstablename, startrowin=paperi, startrowout=-1, nrow=1)
        casa.tb.putcell('Name', tb.nrows()-1, 'HERA')
        casa.tb.close()


def plot_visibilities(ms):
    """Plot the average visibilities over the observation period of the dataset"""
    # Evaluate total time of observation
    casa.tb.open(ms)
    time_obs = tb.getcol("TIME")
    casa.tb.close()
    exposure = time_obs[-1] - time_obs[0]

    casa.plotms(vis=ms, xaxis='chan', yaxis='amp', ydatacolumn='corrected',
                dpi=600, highres=True, coloraxis='baseline', avgtime=str(exposure))
    # plotcal(caltable=cal, xaxis='chan', yaxis='amp')
    # plotcal(caltable=cal, xaxis='chan', yaxis='amp', antenna='13&14')
