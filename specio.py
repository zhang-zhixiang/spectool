import numpy as np
from astropy.io import fits
from PyAstronomy.pyasl import read1dFitsSpec


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


def read_lte_spec(fn):
    hdul = fits.open(fn)
    data = hdul[1].data
    wave = data['WAVELENGTH'].astype('float64')
    flux = data['FLUX'].astype('float64')
    bflux = data['BBFLUX'].astype('float64')
    return wave, flux, bflux


def read_txt_spec(fn):
    data = np.loadtxt(fn)
    wave = data[:, 0]
    flux = data[:, 1]
    if data.shape[1] >= 3:
        err = data[:, 2]
    else:
        err = None
    return wave, flux, err


def read_spec(fn):
    data = 123


class Spectrum(object):
    def __init__(self):
        self.wave = None
        self.flux = None
        self.error = None