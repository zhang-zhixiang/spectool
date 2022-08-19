import numpy as np
import matplotlib.pyplot as plt
from . import spec_func
from . import spec_filter
from . import convol
from . import ccf
from lmfit import Parameters, Minimizer, report_fit, minimize
from scipy.optimize import curve_fit


import spectool


class SpecMatch:
    def __init__(self, wave=None, flux=None, ivar=None, wave_temp=None, flux_temp=None, masks=None, degree_fwhm=2, degree_scale=9):
        self.set_spec(wave, flux, ivar)
        self.set_temp(wave_temp, flux_temp)
        self.set_masks(masks)
        self.set_degree_fwhm(degree_fwhm)
        self.set_degree_scale(degree_scale)

    def set_spec(self, wave, flux, ivar):
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        if wave is not None and flux is not None and ivar is None:
            self.ivar = np.ones(flux.shape)
        if wave is not None:
            self.norm_wave = spec_func.normalize_wave(wave)
        else:
            self.norm_wave = None

    def set_temp(self, wave_temp, flux_temp):
        self.wave_ref = wave_temp
        self.flux_ref = flux_temp

    def set_masks(self, masks):
        self.masks = masks

    def set_degree_fwhm(self, degree_fwhm):
        self.degree_fwhm = degree_fwhm
        self.parName_fwhms = []
        for ind in range(degree_fwhm):
            self.parName_fwhms.append('fwhm%d' % ind)

    def set_degree_scale(self, degree_scale):
        self.degree_scale = degree_scale
        self.parName_scales = []
        for ind in range(degree_scale+1):
            self.parName_scales.append('scale%d' % ind)

    def measure_velocity(self, degree=5, plot=False, broad_ref=500):
        if broad_ref is not None:
            # print('wave_ref =', self.wave_ref)
            # print('flux_ref =', self.flux_ref)
            # print('broad_ref =', broad_ref)
            flux_tmp = spec_filter.gaussian_filter(self.wave_ref, self.flux_ref, broad_ref)
        else:
            flux_tmp = self.flux_ref
        vel = ccf.find_radial_velocity2(self.wave, 
                                                 self.flux, 
                                                 self.wave_ref, 
                                                 flux_tmp,
                                                 maskwindow=self.masks,
                                                 degree=degree,
                                                 plot=plot
                                                )
        self.RV = vel
        self.wave_ref_shift = spec_func.shift_wave(self.wave_ref, vel)
        self.flux_ref_rebin = np.array(spec_func.rebin.rebin_padvalue(self.wave_ref_shift, self.flux_ref, self.wave))
        if plot is True:
            plt.show()
        return vel

    def get_velocity(self):
        return self.RV

    def get_fwhmlst(self, pars):
        parfwhm = [pars[val].value for val in self.parName_fwhms]
        ret = np.zeros(self.norm_wave.shape)
        for ind in range(self.degree_fwhm):
            ret += parfwhm[ind] * np.power(self.norm_wave, ind)
        ret[ret < 0] = 1.0
        return ret

    def get_scale_arr(self, pars=None):
        if pars is None:
            pars = self.pars_fitresult
        parscale = [pars[val].value for val in self.parName_scales]
        scale = np.array(spec_func.legendre_polynomial(self.norm_wave, parscale))
        return scale

    def get_modified_temp(self, pars):
        fwhm = self.get_fwhmlst(pars)
        scale = self.get_scale_arr(pars)
        if hasattr(self, 'flux_ref_rebin'):
            flux_tmp = self.flux_ref_rebin
        else:
            flux_tmp = spec_func.rebin.rebin_padvalue(self.wave_ref, self.flux_ref, self.wave)
        nflux_tmp = spec_filter.gauss_filter_mutable(self.wave, flux_tmp, fwhm)
        nflux_tmp = nflux_tmp * scale
        return nflux_tmp

    def get_residual(self, pars):
        y_model = self.get_modified_temp(pars)
        out = (self.flux - y_model) * self.ivar
        if self.masks is not None:
            arg = spec_func.mask_wave(self.wave, self.masks)
            out = out[arg]
        return out

    def match(self):
        pars = Parameters()
        for ind, val in enumerate(self.parName_fwhms):
            if ind == 0:
                pars.add(val, value=800, min=-3000, max=3000)
            else:
                pars.add(val, value=10, min=-3000, max=3000)
        # flux_ref_rebin = spectool.rebin.rebin_padvalue(self.wave_ref, self.flux_ref, self.wave)
        # arg = flux_ref_rebin == 0
        # print(flux_ref_rebin[arg])
        # arg = self.flux == 0
        # print(self.flux[arg])
        # print('+'*30)
        # flux_ratio = self.flux / flux_ref_rebin
        # ini_scale_pars = spectool.spec_func.fit_profile_par(self.wave, flux_ratio, degree=self.degree_scale)
        for ind, val in enumerate(self.parName_scales):
            pars.add(val, value=(-0.5)**ind)
            # pars.add(val, value=ini_scale_pars[ind])
        self.measure_velocity()
        minner = Minimizer(self.get_residual, pars)
        result = minner.minimize()
        self.fit_result = result
        self.pars_fitresult = result.params
        self.chisq = result.chisqr

    def report_fit_result(self):
        report_fit(self.fit_result)

    def get_chisq(self):
        if hasattr(self, 'chisq'):
            return self.chisq

    def get_fit_flux(self):
        return self.get_modified_temp(self.pars_fitresult)

    def plot_fitted_spec(self, outname=None, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(12, 8))
        else:
            fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(self.wave, self.flux, 'k', label='data')
        ax.plot(self.wave, self.get_fit_flux(), 'r', label='template', linewidth=1)
        if self.masks is not None:
            for win in self.masks:
                ax.axvspan(win[0], win[1], color='gray', alpha=0.3)
        plt.legend()
        if outname is not None:
            plt.savefig(outname)
        # plt.show()

    def plot_fwhm(self, outname=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.wave, self.get_fwhmlst(self.pars_fitresult))
        if outname is not None:
            plt.savefig(outname)
        # plt.show()

    def plot_scale(self, outname=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.wave, self.get_scale_arr(self.pars_fitresult))
        if outname is not None:
            plt.savefig(outname)
        # plt.show()


def fit_scale_pars(wave1, flux1, wave2, flux2, degree=9):
    """fit the scale parameters of two spectra

    Args:
        wave1 (_type_): _description_
        flux1 (_type_): _description_
        wave2 (_type_): _description_
        flux2 (_type_): _description_
        degree (int, optional): _description_. Defaults to 9.
    """

    def func(x, *par):
        return np.array(convol.legendre_poly(x, par))

    nflux2 = spec_func.rebin_padvalue(wave2, flux2, wave1)
    norm_wave = spec_func.normalize_wave(wave1)
    profile = nflux2 / flux1
    pars = [1.0]
    ideg = 1
    while ideg < degree + 1:
        tmppopt, _ = curve_fit(func, norm_wave, profile, p0=pars)
        ideg = ideg + 1
        pars = np.zeros(ideg)
        pars[:-1] = tmppopt
    return pars[:-1]


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
    lmpars[parfwhm[0]].value = 1.0

    return get_fwhm, lmpars


def get_shift_model(degree=1):
    parshift = ['wshift%d' % ind for ind in range(degree)]

    def get_shift(wave, pars):
        arrpar = np.array([pars[key].value for key in parshift])
        nwave = wave / 10000.0
        shift = np.zeros(nwave.shape)
        for ind, pari in enumerate(arrpar):
            shift = shift + pari * nwave**ind
        return shift

    lmpars = Parameters()
    for parname in parshift:
        lmpars.add(parname, 0.0)

    return get_shift, lmpars


class SpecTransform:
    def __init__(self, wave, flux, degree_scale=15, degree_fwhm=1,
                 degree_shift=1, shift_type='velocity'):
        """transform a spectrum to a new style with given parameters.
        The transform include wavelength shift, velocity broading, flux
        scale. The class is designed to fit and match two different spectra.

        the broading unit is km/s
        the shift can work in two unit -- velocity: km/s, lambda: lambda(AA)
        Here we assume the wavelength is in the unit of AA

        An advice: if the flux is normalized before using, like flux = flux / np.median(flux),
        the scale par will be fited more easily. Or you can choose fit in log scale.

        ------------------- a demo ------------------------
        from lmfit import minimize
        from spectool.SpecMatch import SpecTransform
        import matplotlib.pyplot as plt

        wave, flux, err = read_spec()
        wave_t, flux_t = read_template()
        unit1 = np.median(flux)
        unit2 = np.median(flux_t)
        flux = flux / unit1
        err = err / unit1
        flux_t = flux_t / unit2
        spec_tf = SpecTransform(wave_t, flux_t)
        pars = spec_tf.get_parameters()

        def residual(params, x, data, eps_data):
            model = spec_tf.get_flux(params, x)
            return (data-model) / eps_data

        out = minimize(residual, pars, args=(wave, flux, err))
        flux_fit = spec_tf.get_flux(out.params, wave)
        plt.plot(wave, flux, label='spec')
        plt.plot(wave, flux_fit='template')
        plt.show()
        -----------------------------------------------------


        Args:
            wave (numpy.ndarray(float64)): spec wave (will be transformed)
            flux (numpy.ndarray(float64)): spec flux (will be transformed)
            degree_scale (int or None, optional): the degree of scale poly, if set to None, the scale function will be closed. Defaults to 15.
            degree_fwhm (int or None, optional): the degree of broading poly, if set to None, the broading function will be closed. Defaults to 1.
            degree_shift (int or None, optional): the degree of wavelength shift, if set to None, the shift function will be closed. Defaults to 1.
            shift_type (str, optional): the shift mode of the wavelength, 'velocity' or 'lambda'. Defaults to 'velocity'.
        """
        self._wave = wave.copy()
        self._flux = flux.copy()
        parall = None
        if degree_scale is not None:
            self.func_scale, parscale = get_scale_model(degree_scale)
        else:
            self.func_scale = None
            parscale = Parameters()

        if degree_fwhm is not None:
            self.func_fwhm, parfwhm = get_fwhm_model(degree_fwhm)
        else:
            self.func_fwhm = None
            parfwhm = Parameters()
        
        if degree_shift is not None:
            self.func_shift, parshift = get_shift_model(degree_shift)
        else:
            self.func_shift = None
            parshift = Parameters()
        parall = parscale + parfwhm + parshift
        self._parameters = parall
        self._shift_type  = shift_type

    def get_parameters(self):
        """return the lmfit.Parameters object, which can be used to transform this object.

        Returns:
            lmfit.Parameters: the pars is a pair of this object
        """
        return self._parameters

    def get_flux(self, pars, wave):
        """for given pars and wave, the function will return flux array after a transform

        Args:
            pars (lmfit.Parameters): pars used to transform this object, paramery pars should get from get_parameters function
            wave (numpy.ndarray(float64)): wavelength array

        Returns:
            numpy.ndarray: the transformed flux
        """
        if self.func_shift is not None:
            shiftlst = self.func_shift(wave, pars)
            nwave_temp = spec_func.shift_wave_mutable(self._wave, shiftlst, shifttype=self._shift_type)
        else:
            nwave_temp = self._wave
        nflux_temp = spec_func.rebin.rebin_padvalue(nwave_temp, self._flux, wave)

        if self.func_fwhm is not None:
            arrvelocity = self.func_fwhm(wave, pars)
            nflux_temp = spec_filter.gauss_filter_mutable(wave, nflux_temp, arrvelocity)

        if self.func_scale is not None:
            scale = self.func_scale(wave, pars)
            nflux_temp = nflux_temp * scale
        
        return nflux_temp

    def get_fwhm(self, pars, wave):
        if self.func_fwhm is not None:
            return self.func_fwhm(wave, pars)
        else:
            return None

    def get_scale(self, pars, wave):
        if self.func_scale is not None:
            return self.func_scale(wave, pars)
        else:
            return None

    def get_shift(self, pars, wave):
        if self.func_shift is not None:
            return self.func_shift(wave, pars)
        else:
            return None
