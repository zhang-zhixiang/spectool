import numpy as np
from . import spec_func
from . import spec_filter
from . import convol
from lmfit import Parameters


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