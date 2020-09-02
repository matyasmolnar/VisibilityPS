"""Set of commonly used casa functions

To be run in a casapython shell.
"""

import os

from casatools import table


def add_hera_obs_pos():
    """Adding HERA observatory position (at some point as PAPER_SA)"""
    obstablename = os.getenv('CASAPATH').split()[0] + '/data/geodetic/Observatories/'
    if not os.path.exists(obstablename + 'HERA'):
        table.open(obstablename, nomodify=False)
        paperi = (table.getcol('Name') == 'PAPER_SA').nonzero()[0]
        table.copyrows(obstablename, startrowin=paperi, startrowout=-1, nrow=1)
        table.putcell('Name', table.nrows()-1, 'HERA')
        table.close()


def plot_visibilities(msin):
    """Plot the average visibilities over the observation period of the dataset

    :param msin: Visibility dataset in measurement set format path
    :type msin: str
    """
    # Evaluate total time of observation
    table.open(msin)
    time_obs = table.getcol("TIME")
    table.close()
    exposure = time_obs[-1] - time_obs[0]

    casa.plotms(vis=msin, xaxis='chan', yaxis='amp', ydatacolumn='corrected',
                dpi=600, highres=True, coloraxis='baseline', avgtime=str(exposure))
    # plotcal(caltable=cal, xaxis='chan', yaxis='amp')
    # plotcal(caltable=cal, xaxis='chan', yaxis='amp', antenna='13&14')
