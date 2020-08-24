from astropy.io import fits
from astropy.time import Time
from PyAstronomy.pyasl import hmsToDeg, dmsToDeg


def get_ra_dec(fn, ext=None):
    if ext is None:
        strra = fits.getval(fn, 'RA')
        strdec = fits.getval(fn, 'DEC')
    else:
        strra = fits.getval(fn, 'RA', ext=ext)
        strdec = fits.getval(fn, 'DEC', ext=ext)
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


def get_obs_location(fitname, ext=None):
    """return the obs telescope location info: Longtitude, Latitude, altitude

    Args:
        fitname (str): fits file name
        ext (int ot str, optional): extension name. Defaults to None.

    Returns:
        float, float, float: Longtitude, Latitude, altitude
    """
    try:
        val = fits.getval(fitname, 'TELESCOP')
        if val.lower().strip() == 'lamost':
            return 117.58, 40.39, 950
    except KeyError as err:
        pass
    try:
        val = fits.getval(fitname, 'TELID')
        if val.strip() == '200':
            return 360 - 116.865, 33.356389, 1713
    except KeyError as err:
        return None


def get_spec_type(fitname, ext=None):
    """return the spectrum group info, P200 or Lamost

    Args:
        fitname (str): fits name
        ext (int, optional): extension name. Defaults to None.
    """
    try:
        val = fits.getval(fitname, 'TELESCOP')
        if val.lower().strip() == 'lamost':
            return 'Lamost'
    except KeyError as err:
        pass
    try:
        val = fits.getval(fitname, 'TELID')
        if val.strip() == '200':
            return 'P200'
    except KeyError as err:
        return None


def get_obs_jd(fitname, ext=None):
    spec_type = get_spec_type(fitname)
    if spec_type == 'Lamost':
        if ext is not None:
            utc = fits.getval(fitname, 'DATE-OBS', ext=ext)
        else:
            utc = fits.getval(fitname, 'DATE-OBS')
        time_obs_begin = Time(utc, format='isot', scale='utc')
        jd = time_obs_begin.jd
        return jd
    if spec_type == 'P200':
        if ext is not None:
            utc = fits.getval(fitname, 'UTSHUT', ext=ext)
        else:
            utc = fits.getval(fitname, 'UTSHUT')
        time_obs_begin = Time(utc, format='isot', scale='utc')
        jd = time_obs_begin.jd
        return jd