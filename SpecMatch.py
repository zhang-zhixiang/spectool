import numpy as np
from . import spec_func
from . import spec_filter
from . import convol
from lmfit import Parameters
from lmfit.models import GaussianModel


def get_scale_model(degree=15):
    parscale = ['scale%d' % ind for ind in range(degree)]

    def get_scale(wave, pars):
        arrpar = np.array([pars[key].value for key in parscale])
        normwave = spec_func.normalize_wave(wave)
        return spec_func.legendre_polynomial(normwave, arrpar)

    lmparas = Parameters()
    for parname in parscale:
        lmparas.add(parname, 0.0)
    lmparas[parscale[0]].value = 1.0

    return get_scale, lmparas


def get_fwhm_model(degree=2):
    parfwhm = ['fwhm%d' % ind for ind in range(degree)]

    def get_fwhm(wave, pars):
        arrpar = np.array([pars[key].value for key in parfwhm])
        normwave = spec_func.normalize_wave(wave)
        return np.abs(spec_func.legendre_polynomial(normwave, arrpar))

    lmpars = Parameters()
    for parname in parfwhm:
        lmpars.add(parname, 0.0)
    lmpars[parfwhm[0]] = 1.0

    return get_fwhm, lmpars


def get_shift_model(degree=1, type='velocity'):
    parshift = ['wshift%d' % ind for ind in range(degree)]

    def get_shifted_wave(wave, pars):
        arrpar = np.array([pars[key].value for key in parshift])
        if arrpar.size == 1 and type == 'velocity':
            return spec_func.shift_wave(wave, arrpar[0])
        else:
            return spec_func.shift_wave_mutable(wave, arrpar, type)

    lmpars = Parameters()
    for parname in parshift:
        lmpars.add(parname, 0.0)

    return get_shifted_wave, lmpars


class SpecMatch:
    def __init__(self, wave, flux, wave_temp, flux_temp, degree_scale=15,
                 degree_fwhm=1, degree_shift=1, shifttype='velocity'):
        self.set_spectrum(wave, flux)
        self.set_template(wave_temp, flux_temp)
        self.func_scale, parscale = get_scale_model(degree_scale)
        self.func_fwhm, parfwhm = get_fwhm_model(degree_fwhm)
        self.func_shift, parshift = get_shift_model(degree_shift, shifttype)
        self._parameters = parscale + parfwhm + parshift

    def get_parameters(self):
        return self._parameters

    def res(self, pars, mask=None):
        _res = self._flux - self.get_flux(pars)
        if mask is not None:
            arg = spec_func.mask_wave(self._wave, mask)
            _res = _res[arg]
        return _res

    def get_flux(self, pars):
        scale = self.get_scale(pars)
        fwhmlst = self.get_fwhm(pars)
        nrefwave = self.get_ref_newave(pars)
        fluxref_rebin = np.array(spec_func.rebin.rebin(nrefwave, self._fluxref, self._wave))
        fluxref_fwhm = spec_filter.gauss_filter_mutable(self._wave, fluxref_rebin, fwhmlst)
        return fluxref_fwhm * scale

    def get_scale(self, pars):
        return self.func_scale(self._wave, pars) / self._unitr * self._unitf

    def get_fwhm(self, pars):
        return self.func_fwhm(self._wave, pars)

    def get_ref_newave(self, pars):
        return self.func_shift(self._waveref, pars)

    def set_spectrum(self, wave, flux, error=None):
        self._wave = wave.copy()
        self._flux = flux.copy()
        self._unitf = spec_func.get_unit(self._flux)
        self._fluxunit = self._flux / self._unitf

    def set_template(self, wave, flux):
        self._waveref = wave
        self._fluxref = flux
        self._unitr = spec_func.get_unit(self._fluxref)
        self._fluxrefunit = self._fluxref / self._unitr
