"""Set of commonly used calibration functions

heracasa package written by Bojan Nikolic, can be found at:
http://www.mrao.cam.ac.uk/~bn204/soft/py/

Modular installation of CASA 6 used. Documentation for this can be found at:
https://casa.nrao.edu/casadocs/latest/usingcasa/obtaining-and-installing
"""


import os

import numpy as np
from astropy.coordinates import Angle
from hera_cal.utils import LST2JD
from pyuvdata import UVData

from casatasks import applycal, bandpass, fixvis, flagdata, ft, gaincal, \
importuvfits, imregrid, split, tclean
from casatools import componentlist, table

from heracasa import closure as hc

from idr2_info import idr2_ants, idr2_bad_ants_casa, idr2_bls


FornaxA_RA_hours = '03h22m41.789s'
GC_RA_hours =  '17h45m40.04s'

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

    :param uvin: Visibility dataset in miriad file format path
    :type uvin: str

    :returns: Visibility dataset in measurement set format path
    :rtype: str
    """
    fitsi = os.path.splitext(os.path.basename(uvin))[0] + '.uvfits'
    uvd = UVData()
    uvd.read_miriad(uvin)
    uvd.write_uvfits(fitsi, spoof_nonessential=True, force_phase=True)
    msout = fitsi[:-len('uvfits')] + 'ms'
    importuvfits(fitsi, msout)
    os.remove(fitsi)
    return msout


def get_bad_ants(msin, bad_ants_arr, verbose=True):
    """Get the bad antennas for HERA for a given JD

    :param msin: Visibility dataset in measurement set format path
    :type msin: str
    :param bad_ants_arr: Mapping of JDs to bad antennas. For IDR2, bad_ants_arr
    = idr2_bad_ants_casa, and can be found in idr2_info
    :type bad_ants_arr: ndarray of shape shape (2, no_ants)
    :param verbose: Verbose
    :type verbose: bool

    :returns: Bad antennas
    :rtype: ndarray
    """
    JD = int(msin.split('.')[1]) # Get JD from filename
    bad_ants_index = np.where(bad_ants_arr[0, :] == JD)[0][0]
    bad_ants = bad_ants_arr[1, bad_ants_index]
    if verbose:
        print('Flagged antennas for JD {} are {}'.format(JD, bad_ants))
    return bad_ants


def gcflagdata(msin, bad_ants, cut_edges=True, bad_chans=None):
    """Flag bad antennas for visibility dataset

    :param msin: Visibility dataset in measurement set format path
    :type msin: str
    :param bad_ants: Bad antennas to flag - can be specified by a list of
    antennas, or can be given by a string specifying the data release
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
    if bad_ants is not None:
        flagdata(msin, flagbackup=True, mode='manual', \
                 antenna=str(bad_ants).replace("[", "").replace("]", ""))

    # Cutting visibilities at extremes of bandwidth
    if cut_edges:
        flagdata(msin, flagbackup=True, mode='manual', spw='0:0~65')
        flagdata(msin, flagbackup=True, mode='manual', spw='0:930~1024')

    # Flagging known bad channels
    if bad_chans is not None:
        for bad_chan in bad_chans:
            flagdata(msin, flagbackup=True, mode='manual', spw='0:'+str(bad_chan))

    # Flag autocorrelations
    flagdata(msin, autocorr=True)
    return msin


def fringerot(msin, phasecenter):
    """Fringe rotate visibilities inplace

    :param msin: Visibility dataset in measurement set format path
    :type msin: str
    :param phasecenter: J2000 coordinates of point source model or name of well-known
    radio object
    :type phasecenter: str
    """
    if phasecenter == 'FornaxA':
        phasecenter = FornaxA_coords
    if phasecenter == 'GC':
        phasecenter = GC_coords

    fixvis(msin, msin, phasecenter=phasecenter)


def mkinitmodel(coords):
    """Make an initial point source model

    :param coords: J2000 coordinates of point source model or name of well-known
    radio object
    :type coords: str

    :return: Point source model file path
    :rtype: str
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

        cl = componentlist()
        cl.done()
        cl.addcomponent(flux=1.0,
                                   fluxunit='Jy',
                                   shape='point',
                                   dir=dir)
        cl.rename(coords+'.cl')
        cl.close()
    return coords + '.cl'


def dosplit(msin, inf, datacolumn='corrected', spw=''):
    """Split the initial calibrated data into a visibility subset

    :param msin: Visibility dataset in measurement set format path
    :type msin: str
    :param inf: Split extension
    :type inf: str
    :param datacolumn: Data column to split
    :type datacolumn: str
    :param spw: Select spectral window
    :type spw: str

    :return: Visibility subset in measurement set format path
    :rtype: str
    """
    newms = os.path.basename(msin) + inf + '.ms'
    split(msin, newms, datacolumn=datacolumn, spw=spw)
    return newms


def calname(msin, cal_type):
    """Build calibrator name based on filename

    :param msin: Visibility dataset in measurement set format path
    :type msin: str
    :param cal_type: Type of calibrator (e.g. gain, bandpass)
    :type cal_type: str

    :return: Calibrator path
    :rtype: str
    """
    return os.path.basename(msin) + cal_type + '.cal'


def kc_cal(msin, model_cl):
    """Get gain and delay calibration solutions

    :param msin: Visibility dataset in measurement set format path
    :type msin: str
    :param model_cl: Point source model path
    :type model_cl: str

    :return: Calibration solutions tables
    :rtype: list
    """
    # Fill the model column
    ft(msin, complist=model_cl, usescratch=True)

    kc = calname(msin, 'K') # Delays
    gc = calname(msin, 'G') # Gains

    gaincal(vis=msin, caltable=kc, gaintype='K', solint='inf',
            refant='11', minsnr=1)
    # Ensure reference antenna exists and isn't faulty
    gaincal(vis=msin, caltable=gc, gaintype='G', solint='inf',
            refant='11', minsnr=1, calmode='ap', gaintable=kc)
    applycal(msin, gaintable=[kc, gc])
    return [kc, gc]


def bandpass_cal(msin):
    """Bandpass calbration

    :param msin: Visibility dataset in measurement set format path
    :type msin: str

    :return: Calibration solutions table
    :type: str
    """
    bc = calname(msin, 'B')
    bandpass(vis=msin, minsnr=1, solnorm=False, bandtype='B', caltable=bc)
    applycal(msin, gaintable=[bc])
    return bc


def cleaninit(msin, cal_source):
    """First CLEANing round

    :param msin: Visibility dataset in measurement set format path
    :type msin: str
    :param cal_source: Calibration source for masking
    :type cal_source: str

    TODO: Argument to specify mask
    """
    imgname = os.path.basename(msin) + '.init.img'
    if cal_source in cal_source_dct.keys():
        clean_mask = cal_source_dct[cal_source]['mask']
    else:
        clean_mask = None
    tclean(vis=msin,
           imagename=imgname,
           niter=500,
           weighting='briggs',
           robust=0,
           imsize=[512, 512],
           cell=['250arcsec'],
           specmode='mfs',
           nterms=1,
           spw='0:150~900',
           mask=clean_mask)


def cleanfinal(msin, cal_source):
    """Second CLEANing round and imaging

    :param msin: Visibility dataset in measurement set format path
    :type msin: str
    :param cal_source: Calibration source for masking
    :type cal_source: str
    """
    imgname = os.path.basename(msin) + '.fin.img'
    if cal_source in cal_source_dct.keys():
        clean_mask = cal_source_dct[cal_source]['mask']
    else:
        cal_source = '' # TODO create mask for arbitrary point source given
        clean_mask = None
    tclean(vis=msin,
           imagename=imgname,
           spw='0:60~745',
           niter=3000,
           weighting='briggs',
           robust=-1,
           imsize=[512, 512],
           cell=['250arcsec'],
           specmode='mfs',
           nterms=1,
           mask=clean_mask)
    imggal = cal_source + 'combined.galcord'
    imregrid(imagename=imgname+'.image', output=imggal, template='GALACTIC')


def genvisibility(msin, **kwargs):
    """Save the calibrated data arrays to npz file format

    :param msin: Visibility dataset in measurement set format path
    :type msin: str

    :return: Visibility data saved to NpzFile file
    :rtype: str
    """
    fout = os.path.split(msin)[-1] + '.npz'
    r = hc.vis(msin, baseline=idr2_bls, alist=idr2_ants)
    np.savez(fout, **r)
    if not os.path.exists(fout):
        raise RuntimeError('No output produced by heracasa.closure.vis')
    return(fout)


def plot_ms(msin):
    """Plotting of visibilities in CASA

    :param msin: Visibility dataset in measurement set format path
    :type msin: str
    """
    # Evaluate total time of observation
    table.open(msin)
    time_obs = table.getcol('TIME')
    table.close()
    exposure = time_obs[-1] - time_obs[0]

    plotms(vis=msin, xaxis='chan', yaxis='amp', ydatacolumn='corrected',
           dpi=600, highres=True, coloraxis='baseline', avgtime=str(exposure))


def calibrator_in_fov(msin, calibrator_RA, fov='0h30m'):
    """Determine if calibrator in FOV of dataset

    :param msin: Visibility dataset in measurement set format path
    :type msin: str
    :param calibrator_RA: Right ascension of calibrator
    :type calibrator_RA: str
    :param fov: (Half) Field of view of intererometer in hours
    :type fov: str

    :return: If there is a calibrator in the field of view for the given dataset
    :rtype: bool
    """

    if calibrator_RA == 'FornaxA':
        calibrator_RA = FornaxA_RA_hours
    if calibrator_RA == 'GC':
        calibrator_RA = GC_RA_hours

    uvd = UVData()
    uvd.read_ms(msin)

    calibrator_angle = Angle(calibrator_RA)
    fov = Angle(fov)

    ra_min = (calibrator_angle - fov).radian
    ra_max = (calibrator_angle + fov).radian

    return np.logical_and(uvd.lst_array > ra_min, uvd.lst_array < ra_max).any()
