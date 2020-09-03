"""Set of commonly used casa functions

To be run in a casapython shell.
"""

import os

import casadata
from casatools import table


def add_hera_obs_pos():
    """Adding HERA observatory position (at some point as PAPER_SA)

    Only needed for the older versions of CASA
    """
    obstablename = os.path.dirname(casadata.__file__) + \
                   '/__data__/geodetic/Observatories/'
    tbl = table()
    tbl.open(obstablename, nomodify=False)
    if not (tbl.getcol('Name') == 'HERA').any():
        paperi = (tbl.getcol('Name') == 'PAPER_SA').nonzero()[0]
        tbl.copyrows(obstablename, startrowin=paperi, startrowout=-1, nrow=1)
        tbl.putcell('Name', tbl.nrows()-1, 'HERA')
        tbl.close()


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
