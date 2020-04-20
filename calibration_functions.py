"""Set of commonly used calibration functions

heracasa package written by Bojan Nikolic, can be found at:
http://www.mrao.cam.ac.uk/~bn204/g/
"""


import os

import casa
import numpy as np

from heracasa import closure as hc
from heracasa.data import uvconv

from idr2_info import idr2_bad_ants_casa


GC_coords = 'J2000 17h45m40.04s -29d00m28.12s'
GCCleanMask = 'ellipse[[17h45m40.04s, -29d00m28.12s], [11deg, 4deg], 30deg]'

FornaxA_coords = 'J2000 03h22m41.79s -37d12m29.52s'
FornaxACleanMask = 'ellipse[[3h22m41.79s, -37d12m29.52s], [1deg, 1deg], 10deg]'

cal_source_dct = {'GC': {'coords': GC_coords,
                         'mask': GCCleanMask},
                  'FornaxA': {'coords': FornaxA_coords,
                              'mask': FornaxACleanMask}}


def cv(uvin):
    """Convert visibilities from miriad to measurement set file format

    Conversion from miriad to ms done with the intermediate step of converting
    to intermediate uvfits file format

    :param uvin: Visibility dataset in miriad file format
    :type uvin: miriad

    :returns: Visibility dataset in ms file format
    :rtype: Measurement set
    """
    fitsi = os.path.splitext(os.path.basename(uvin))[0] + ".uvfits"
    uvconv.cvuvfits(uvin, fitsi)
    uvconv.renumb(fitsi, fitsi)
    msout = fitsi[:-len('uvfits')] + 'ms'
    casa.importuvfits(fitsi, msout)
    os.remove(fitsi)
    return msout


def get_bad_ants(msin, bad_ants_arr, verbose=True):
    """Get the bad antennas for HERA for a given JD

    :param msin: Visibility dataset
    :type msin: Measurement set
    :param bad_ants_arr: Mapping of JDs to bad antennas. For IDR2, bad_ants_arr
                         = idr2_bad_ants_casa, and can be found in idr2_info.py
    :type bad_ants_arr: ndarray of shape shape (2, no_ants)
    """
    JD = int(msin.split('.')[1]) # Get JD from filename
    bad_ants_index = np.where(bad_ants_arr[0, :] == JD)[0][0]
    bad_ants = bad_ants_arr[1, bad_ants_index]
    if verbose:
        print('Flagged antennas for JD {} are {}'.format(JD, bad_ants))
    return bad_ants


def gcflagdata(msin, bad_ants, cut_edges=True, bad_chans=None):
    """Flag bad antennas for visibility dataset

    :param msin: Visibility dataset
    :type msin: Measurement set
    :param bad_ants: Bad antennas to flag - can be specified by a list of
                     antennas, or can be given by a string specifying the
                     data release
    :type bad_ants: list or str
    :param cut_edges: Specify if the band edges should be flagged
    :type cut_edges: bool
    :param bad_chans: Specify channels to flag
    :type bad_chans: list of strings (e.g. ['207', '377~378'])

    :param msin: Flagged visibility dataset
    :type msin: Measurement set
    """

    if bad_ants == 'IDR2':
        JD = int(msin.split('.')[1]) # Get JD from filename
        if JD in idr2_bad_ants_casa[0, :]:
            bad_ants = get_bad_ants(msin, idr2_bad_ants_casa, verbose=True)
        else:
            bad_ants = None
            print('Visibility dataset {} not in IDR2 - no antennas flagged'.format(msin))

    # Flagging bad antennas known for that JD
    if bad_ants:
        casa.flagdata(msin, flagbackup=True, mode='manual',
                      antenna=str(bad_ants).replace("[", "").replace("]", ""))

    # Cutting visibilities at extremes of bandwidth
    if cut_edges:
        casa.flagdata(msin, flagbackup=True, mode='manual', spw='0:0~65')
        casa.flagdata(msin, flagbackup=True, mode='manual', spw='0:930~1024')

    # Flagging known bad channels
    if bad_chans:
        for bad_chan in bad_chans:
            casa.flagdata(msin, flagbackup=True, mode='manual', spw='0:'+str(bad_chan))

    # Flag autocorrelations
    casa.flagdata(msin, autocorr=True)
    return msin


def fringerot(msin, phasecenter):
    """Fringe rotate visibilities

    :param phasecenter: J2000 coordinates of point source model or name of well-known
    radio object
    :type phasecenter: str
    """
    if phasecenter == 'FornaxA':
        phasecenter = FornaxA_coords
    if coords == 'GC':
        phasecenter = GC_coords

    casa.fixvis(msin, msin, phasecenter=phasecenter)
    return msin


def mkinitmodel(coords, **kwargs):
    """Make an initial point source model

    :param coords: J2000 coordinates of point source model or name of well-known
    radio object
    :type coords: str
    """

    # Check if point source model has already been created
    if os.path.exists(coords+'.cl'):
        print('Model for {} already created'.format(coords))
    else:
        if coords in cal_source_dct.keys():
            dir = cal_source_dct[coords]['coords']
        else:
            dir = coords
            coords = 'model'

        casa.componentlist.done()
        casa.componentlist.addcomponent(flux=1.0,
                                        fluxunit='Jy',
                                        shape='point',
                                        dir=dir)
        casa.componentlist.rename(coords+'.cl')
        casa.componentlist.close()
    return coords+'.cl'


def dosplit(msin, inf, datacolumn='corrected', spw=''):
    """Split the initial calibrated data"""
    newms = os.path.basename(msin) + inf + '.ms'
    casa.split(msin, newms, datacolumn=datacolumn, spw=spw)
    return newms


def calname(msin, cal_type):
    """Build calibrator name based on filename"""
    return os.path.basename(m) + cal_type + '.cal'


def kc_cal(msin, model_cl):
    """Get gain and delay calibration solutions"""
    # Fill the model column
    casa.ft(msin, complist=model_cl, usescratch=True)

    kc = calname(msin, 'K') # Delays
    gc = calname(msin, 'G') # Gains

    casa.gaincal(vis=msin, caltable=kc, gaintype='K', solint='inf',
            refant='11', minsnr=1)
    # Ensure reference antenna exists and isn't faulty
    casa.gaincal(vis=msin, caltable=gc, gaintype='G', solint='inf',
            refant='11', minsnr=1, calmode='ap', gaintable=kc)
    casa.applycal(msin, gaintable=[kc, gc])
    return [kc, gc]


def bandpass_cal(msin):
    """Bandpass calbration"""
    bc = calname(msin, 'B')
    casa.bandpass(vis=msin, minsnr=1, solnorm=False, bandtype='B', caltable=bc)
    casa.applycal(msin, gaintable=[bc])
    return bc


def cleaninit(msin, cal_source):
    """First CLEANing round"""
    imgname = os.path.basename(msin) + '.init.img'
    if cal_source in cal_source_dct.keys():
        clean_mask = cal_source_dct[cal_source]['mask']
    else:
        clean_mask = None
    casa.clean(vis=msin,
               imagename=imgname,
               niter=500,
               weighting='briggs',
               robust=0,
               imsize=[512, 512],
               cell=['250arcsec'],
               mode='mfs',
               nterms=1,
               spw='0:150~900',
               mask=clean_mask)


def cleanfinal(msin, cal_source):
    """Second CLEANing round and imaging"""
    imgname = os.path.basename(msin) + '.fin.img'
    if cal_source in cal_source_dct.keys():
        clean_mask = cal_source_dct[cal_source]['mask']
    else:
        cal_source = '' # TODO create mask for arbitrary point source given
        clean_mask = None
    casa.clean(vis=msin,
               imagename=imgname,
               spw='0:60~745',
               niter=3000,
               weighting='briggs',
               robust=-1,
               imsize=[512, 512],
               cell=['250arcsec'],
               mode='mfs',
               nterms=1,
               mask=clean_mask)
    imggal = cal_source + 'combined.galcord'
    casa.imregrid(imagename=imgname+'.image', output=imggal, template='GALACTIC')


def genvisibility(fin, **kwargs):
    """Save the calibrated data arrays to npz file format"""
    fout = os.path.split(fin)[-1] + '.npz'
    r = hc.vis(fin, baseline=idr2_ants, alist=idr2_bls)
    np.savez(fout, **r)
    if not os.path.exists(fout):
        raise RuntimeError('No output produced by heracasa.closure.vis')
    return(fout)


def plot_ms(msin):
    """Plotting of visibilities in CASA"""
    # Evaluate total time of observation
    casa.tb.open(msin)
    time_obs = tb.getcol('TIME')
    casa.tb.close()
    exposure = time_obs[-1] - time_obs[0]

    casa.plotms(vis=msin, xaxis='chan', yaxis='amp', ydatacolumn='corrected',
                dpi=600, highres=True, coloraxis='baseline', avgtime=str(exposure))