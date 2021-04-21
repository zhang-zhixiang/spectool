import os
import numpy as np
from astropy.io import fits
from PyAstronomy.pyasl import read1dFitsSpec
from PyAstronomy.pyasl import hmsToDeg, dmsToDeg
from . import spec_func


def read_iraf_spec(fn, band=0):
    hdul = fits.open(fn)
    data = hdul[0].data
    head = hdul[0].header
    size = head['NAXIS1']
    step = head['CD1_1']
    start = head['CRVAL1']
    wave = np.arange(size) * step + start
    flux = data[0, band, :].astype('float64')
    err = data[3, band, :].astype('float64')
    return wave, flux, err


def read_iraf_echell(fn):
    """read the spectra file created by iraf package echell,
    Caution: this is a temporary implementation, waiting for the
    support from specutils

    Args:
        fn (fitname): fits name 

    Returns:
        [[wave, flux], [wave, flux],...]: A spectra list
    """
    from pyraf import iraf
    import tempfile
    tmpdir = tempfile.gettempdir()
    hdul = fits.open(fn)
    dim = hdul[0].data.shape[0]
    iraf.onedspec()
    outlst = []
    for ind in range(1, dim+1):
        inname = fn + '[*,%d]' % ind
        basename, ext = os.path.splitext(fn)
        name = ''.join([basename, str(ind), ext])
        outname = os.sep.join([tmpdir, name])
        outtxtname = os.path.splitext(outname)[0] + '.txt'
        if not os.path.isfile(outname) or not os.path.isfile(outtxtname):
            if os.path.isfile(outname):
                os.remove(outname)
            if os.path.isfile(outtxtname):
                os.remove(outtxtname)
            iraf.scopy(inname, outname)
            iraf.wspectext(outname, outtxtname, header='No')
        data = np.loadtxt(outtxtname)
        wave = data[:, 0]
        flux = data[:, 1]
        arg = np.argsort(wave)
        wave = wave[arg]
        flux = flux[arg]
        outlst.append([wave, flux])
    return outlst


def read_lte_spec(fn):
    hdul = fits.open(fn)
    data = hdul[1].data
    wave = data['WAVELENGTH'].astype('float64')
    flux = data['FLUX'].astype('float64')
    bflux = data['BBFLUX'].astype('float64')
    return wave, flux, bflux


def read_lamost_low(fn):
    # fit = fits.open(fn)
    # first = fit[0].header['CRVAL1']
    # step = fit[0].header['CD1_1']
    # length = fit[0].header['NAXIS1']
    # logwave = np.arange(length)*step + first
    # wave = 10**logwave
    # flux = fit[0].data[0, :]
    # data = fit[0].data
    # invar = data[1, :].astype('float64')
    # arg = np.where(invar == 0)
    # invar[arg] = 1
    # err = 1 / invar.astype('float64')
    # err[arg] = np.inf
    # return wave, flux, err


    hdul = fits.open(fn)
    data = hdul[0].data
    wave = data[2, :].astype('float64')
    flux = data[0, :].astype('float64')
    invar = data[1, :].astype('float64')
    arg = np.where(invar == 0)
    invar[arg] = 1
    err = 1 / invar.astype('float64')
    err[arg] = np.inf
    arg = np.argsort(wave)
    return wave[arg], flux[arg], err[arg]


def read_lamost_med(fn, hduid):
    data = fits.getdata(fn, ext=hduid)
    wave = 10**data['loglam'].astype('float64')
    flux = data['flux'].astype('float64')
    arg = np.argsort(wave)
    wave = wave[arg]
    flux = flux[arg]
    return wave, flux, None


def read_txt_spec(fn):
    data = np.loadtxt(fn)
    wave = data[:, 0]
    flux = data[:, 1]
    if data.shape[1] >= 3:
        err = data[:, 2]
    else:
        err = None
    return wave, flux, err


def read_sdss_spec(fn):
    hdul = fits.open(fn)
    data = hdul[1].data
    wave = data['loglam'].astype('float64')
    flux = data['flux'].astype('float64')
    ivar = data['ivar'].astype('float64')
    return wave, flux, ivar


def read_spec(fn):
    data = 123


def get_ra_dec(fn):
    strra = fits.getval(fn, 'RA')
    strdec = fits.getval(fn, 'DEC')
    if isinstance(strra, float):
        return strra, strdec
    if ':' in strra:
        hh, mm, ss = [float(i) for i in strra.split(':')]
        ra = hmsToDeg(hh, mm, ss)
        dd, mm, ss = [float(i) for i in strdec.split(':')]
        dec = dmsToDeg(dd, mm, ss)
        return ra, dec
    else:
        return float(strra), float(strdec)


class Spectrum(object):
    def __init__(self):
        self.wave = None
        self.flux = None
        self.error = None