import numpy as np
from scipy.interpolate import interp1d


def rebin(wave: np.ndarray, flux: np.ndarray, new_wave: np.ndarray, interp_kind='linear'):
    """
    Rebins the provided flux data to a new wavelength grid using interpolation.

    Parameters:
        wave (np.ndarray): The original wavelength array.
        flux (np.ndarray): The original flux array corresponding to the `wave` array.
        new_wave (np.ndarray): The new wavelength array to which the flux will be rebinned.
        interp_kind (str, optional): The interpolation method to use. Options are 'linear', 'cubic', or 'quadratic'. Default is 'linear'.

    Returns:
        np.ndarray: The rebinned flux array corresponding to `new_wave`.

    Raises:
        ValueError: If `interp_kind` is not one of 'linear', 'cubic', or 'quadratic'.

    Notes:
        - If interp_kind 'cubic' is chosen, the resulting spectrum will be smoother, 
          but this will be computationally more expensive.
        - This function calculates the edges of the wavelength bins and interpolates the cumulative flux 
          to the new wavelength grid, then computes the new flux values by dividing the difference in the 
          cumulative flux by the new bin widths.
    """
    if interp_kind not in ['linear', 'cubic', 'quadratic']:
        raise ValueError('interp_kind must be "linear", "cubic" or "quadratic"')

    def get_edge(wave):
        dif = np.diff(wave) / 2.0
        d0, de = dif[0], dif[-1]
        dif = np.hstack((dif, [de]))
        edge = wave + dif
        e0 = wave[0] - d0
        edge = np.hstack(([e0], edge))
        return edge

    def get_new_flux(newedge, oldedge, flux):
        oldwidth = np.diff(oldedge)
        newwidth = np.diff(newedge)
        intflux = np.cumsum(flux*oldwidth)
        intflux = np.hstack(([0], intflux))
        if interp_kind == 'linear':
            newintflux = np.interp(newedge, oldedge, intflux, left=intflux[0], right=intflux[-1])
        else:
            f = interp1d(oldedge, intflux, kind=interp_kind, fill_value='extrapolate')
            newintflux = f(newedge)
        newflux = np.diff(newintflux) / newwidth
        return newflux

    oldedge = get_edge(wave)
    newedge = get_edge(new_wave)
    # unit = 1.0 / np.median(flux)
    # inverseunit = 1.0 / unit
    # tempflux = flux * unit
    newflux = get_new_flux(newedge, oldedge, flux)
    # newflux = newflux * inverseunit
    return newflux


def rebin_err(wave: np.ndarray, error: np.ndarray, new_wave: np.ndarray):
    """
    Rebins the error values to match the new wavelength grid.

    Parameters:
        wave (np.ndarray): The original wavelength array.
        error (np.ndarray): The original error array corresponding to the wavelengths.
        new_wave (np.ndarray): The new wavelength array to rebin the error values to.

    Returns:
        np.ndarray: The rebinned error array, with errors set to infinity for wavelengths outside the original range.
    """
    err2 = error * error
    newerr2 = rebin(wave, err2, new_wave)
    newerr = np.sqrt(newerr2)
    wl, wr = wave[0], wave[-1]
    arg = (new_wave < wl) | (wr < new_wave)
    newerr[arg] = np.inf
    return newerr


def rebin_padvalue(wave: np.ndarray, flux: np.ndarray, new_wave: np.ndarray, interp_kind='linear'):
    """
    Rebin a flux array to a new wavelength grid with padding for out-of-bounds values.

    This function interpolates the provided `flux` onto a new wavelength grid (`new_wave`),
    using the `wave` and `flux` arrays. For wavelengths outside the original `wave` range,
    the function pads the corresponding flux values with the first and last values of the original
    flux array. The interpolation method can be chosen through the `interp_kind` parameter.

    Parameters:
        wave (np.ndarray): 1D array of the original wavelength grid.
        flux (np.ndarray): 1D array of flux values corresponding to the original wavelength grid.
        new_wave (np.ndarray): 1D array of the new wavelength grid onto which the flux will be interpolated.
        interp_kind (str, optional): Interpolation method. Default is 'linear'. 
                                     Other options may include 'nearest', 'cubic', etc.

    Returns:
        np.ndarray: The rebinned flux values corresponding to `new_wave`, with padding for out-of-bounds wavelengths.
    """
    dwl_old = wave[1] - wave[0]
    dwr_old = wave[-1] - wave[-2]
    newflux = rebin(wave, flux, new_wave, interp_kind=interp_kind)
    wl, wr = wave[0], wave[-1]
    argl = new_wave < wl + dwl_old
    argr = new_wave > wr - dwr_old
    newflux[argl] = flux[0]
    newflux[argr] = flux[-1]
    return newflux