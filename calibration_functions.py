"""Set of commonly used calibration functions

heracasa package written by Bojan Nikolic, can be found at:
http://www.mrao.cam.ac.uk/~bn204/g/
"""


import os

import casa
import numpy as np
from heracasa import closure as hc
from heracasa import data
from heracasa.data import uvconv


FornaxA_coords = 'J2000 03h22m41.79s -37d12m29.52s'
GC_coords = 'J2000 17h45m40.04s -29d00m28.12s'

FornaxACleanMask = 'ellipse[[3h22m41.79s, -37d12m29.52s], [ 1deg, 1deg ], 10deg]'
GCCleanMask = 'ellipse[[17h45m40.04s, -29d00m28.12s ], [ 11deg, 4deg ], 30deg]'


def cv(uvin):
    """Convert visibilities from miriad to measurement set file format

    No caching done here

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


def gcflagdata(msin, bad_ants_arr, cut_edges=True, bad_chans=None):
    """Flag bad antennas for visibility dataset

    :param msin: Visibility dataset
    :type msin: Measurement set
    :param bad_ants_arr: Bad antennas array of shape (2, no_ants), that maps JDs
                         to bad antennas
    :type bad_ants_arr: ndarray
    :param cut_edges: Specify if the band edges should be flagged
    :type cut_edges: bool
    :param bad_chans: Specify channels to flag
    :type bad_chans: list of strings (e.g. ['207', '377~378'])

    :param msin: Flagged visibility dataset
    :type msin: Measurement set
    """

    JD = int(msin.split('.')[1]) # Get JD from filename
    bad_ants_index = np.where(bad_ants_arr[0, :] == JD)[0][0]
    print('Flagged antennas for JD {} are {}'.format(JD,
          bad_ants_arr[1, bad_ants_index]))
    # Flagging bad antennas known for that JD
    casa.flagdata(msin, flagbackup=True, mode='manual',
                  antenna=str(bad_ants_arr[1, bad_ants_index]).replace("[", "").replace("]", ""))

    # Cutting visibilities at extremes of bandwidth
    if cut_edges:
        casa.flagdata(msin, flagbackup=True, mode='manual', spw='0:0~65')
        casa.flagdata(msin, flagbackup=True, mode='manual', spw='0:930~1024')

    # Flagging known bad channels
    if bad_chans:
        for bad_chan in bad_chans:
            flagdata(msin, flagbackup=True, mode='manual', spw='0:'+str(bad_chan))

    # Flag autocorrelations (where ant i = ant j)
    casa.flagdata(msin, autocorr=True)
    return msin


def fringerot(din, phasecenter=FornaxA_coords):
    """Fringe rotate visibilities

    Default is to rotate to Fornax A coordinates

    :param phasecenter: J2000 coordinates
    """
    casa.fixvis(din, din, phasecenter=phasecenter)
    return din


def mkinitmodel(**kwargs):
    """Make an initial point source model

    Default is to create one for Fornax A
    """
    casa.componentlist.done()
    casa.componentlist.addcomponent(flux=1.0,
                                    fluxunit='Jy',
                                    shape='point',
                                    dir='J2000 03h22m41.789s -37d12m29.52s')
    casa.componentlist.rename('FornaxA.cl')
    casa.componentlist.close()


def dosplit(msin, inf, datacolumn="corrected", spw=""):
    """Split the initial calibrated data"""
    newms = os.path.basename(msin)+inf+".ms"
    split(msin, newms, datacolumn=datacolumn, spw=spw)
    return newms


def calname(m, c):
    """Build calibrator name based on filename"""
    return os.path.basename(m) + c + '.cal'


def kc_cal(msin):
    """Get gain and delay calibration solutions"""
    # Fill the model column
    casa.ft(msin, complist='FornaxA.cl', usescratch=True)

    kc = calname(msin, 'K') # Delays
    gc = calname(msin, 'G') # Gains

    casa.gaincal(vis=msin, caltable=kc, gaintype='K', solint='inf',
            refant='11', minsnr=1)
    # Ensure reference antenna exists and isn't faulty
    casa.gaincal(vis=msin, caltable=gc, gaintype='G', solint='inf',
            refant='11', minsnr=1, calmode='ap', gaintable=kc)
    return [kc, gc]


def bandpass_cal(msin):
    """Bandpass calbration"""
    bc = calname(msin, 'B')
    casa.bandpass(vis=msin, minsnr=1, solnorm=False, bandtype='B', caltable=bc)
    return bc


def cleaninit(msin):
    imgname = os.path.basename(msin)+".init.img"
    clean(vis=msin,
          imagename=imgname,
          niter=500,  # 500
          weighting='briggs',
          robust=0,
          imsize=[512, 512],
          cell=['250arcsec'],
          mode='mfs',
          nterms=1,
          spw='0:150~900',
          mask=FornaxACleanMask)


def cleanfinal(msl):
    """CLEANing and imaging"""
    imgname = "FornaxA.combined.yy.img"
    clean(vis=msl,
          imagename=imgname,
          spw='0:60~745',
          niter=3000,  # 5000
          weighting='briggs',
          robust=-1,
          imsize=[512, 512],
          cell=['250arcsec'],
          mode='mfs',
          nterms=1,
          mask=FornaxACleanMask)

    imggal = "FornaxA.combinedyy.galcord"
    imregrid(imagename=imgname+".image", output=imggal, template="GALACTIC")


def genvisibility(fin, **kwargs):
    """Save the calibrated data arrays to npz file format"""
    fout = os.path.split(fin)[-1] + ".npz"
    r = hc.vis(fin, baseline=idr2_ants, alist=idr2_bls)
    np.savez(fout, **r)
    if not os.path.exists(fout):
        raise RuntimeError('No output produced by heracasa.closure.vis')
    return(fout)
