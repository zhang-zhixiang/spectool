import numpy as np


def rebin(wave: np.ndarray, flux: np.ndarray, new_wave: np.ndarray):
    """
    if err is None; return flux
    else return newflux, newerr
    """
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
        newintflux = np.interp(newedge, oldedge, intflux, left=intflux[0], right=intflux[-1])
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
    err2 = error * error
    newerr2 = rebin(wave, err2, new_wave)
    newerr = np.sqrt(newerr2)
    wl, wr = wave[0], wave[-1]
    arg = (new_wave < wl) | (wr < new_wave)
    newerr[arg] = np.inf
    return newerr


def rebin_padvalue(wave: np.ndarray, flux: np.ndarray, new_wave: np.ndarray):
    dwl = new_wave[1] - new_wave[0]
    dwr = new_wave[-1] - new_wave[-2]
    wave_tmp = np.concatenate([])
    newflux = rebin(wave, flux, new_wave)
    wl, wr = wave[0], wave[-1]
    argl = new_wave < wl
    argr = wr < new_wave
    newflux[argl] = flux[0]
    newflux[argr] = flux[-1]
    return newflux