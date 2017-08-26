import numpy as np
import astropy.units as unit

from astropy.coordinates import Angle
from astropy.time import Time


def cart2pol(dx, dy, dx_err=0, dy_err=0, radec=True):
    '''
    Convert cartesian to polar coordinates, with error propagation

    Parameters
    ----------

    dx : float
        Position delta in x

    dy : float
        Position delta in y

    dx_err : float, optional
        Error on position delta in x. Default is 0

    dy_err : float, optional
        Error on position delta in y. Default is 0

    radec : bool, optional
        Are coordinates expressed in RA/DEC on sky. Default is True
    
    Returns
    -------

    sep : float
        Separation

    pa : float
        Position angle, in degrees

    sep_err : float
        Error on separation

    pa_err : float
        Error on position angle, in degrees
    '''
    
    sep = np.sqrt(dx**2 + dy**2)
    if radec:
        pa  = np.mod(np.rad2deg(np.arctan2(dy, -dx)) + 270, 360)
    else:
        pa  = np.mod(np.rad2deg(np.arctan2(dy, dx)) + 360, 360)

    sep_err = np.sqrt(dx**2 * dx_err**2 + dy**2 * dy_err**2) / sep
    pa_err  = np.rad2deg(np.sqrt(dy**2 * dx_err**2 + dx**2 * dy_err**2) / sep**2)


    return sep, pa, sep_err, pa_err


def pol2cart(sep, pa, sep_err=0, pa_err=0, radec=True):
    '''
    Convert cartesian to polar coordinates, with error propagation

    Parameters
    ----------

    sep : float
        Separation

    pa : float
        Position angle, in degrees

    sep_err : float, optional
        Error on separation. Default is 0

    pa_err : float, optional
        Error on position angle, in degrees. Default is 0

    radec : bool, optional
        Are coordinates expressed in RA/DEC on sky. Default is True
    
    Returns
    -------

    dx : float
        Position delta in x

    dy : float
        Position delta in y

    dx_err : float
        Error on position delta in x

    dy_err : float
        Error on position delta in y
    '''

    pa = np.deg2rad(pa)
    pa_err = np.deg2rad(pa_err)
    
    if radec:
        dx = -sep*np.cos(pa+np.pi/2)
        dy = sep*np.sin(pa+np.pi/2)
    else:
        dx = sep*np.cos(pa)
        dy = sep*np.sin(pa)

    dx_err = np.sqrt(np.cos(pa)**2 * sep_err**2 + sep**2 * np.sin(pa)**2 * pa_err**2)
    dy_err = np.sqrt(np.sin(pa)**2 * sep_err**2 + sep**2 * np.cos(pa)**2 * pa_err**2)
    
    return dx, dy, dx_err, dy_err
        
    

def ten(hms):
    '''
    Convert from sexagesimal to decimal degrees

    Parameters
    ----------
    
    dms : str, list, tuple
        Value in sexagesimal notation (HMS)

    Returns
    -------
    
    deg : float
        Value in decimal degrees

    '''
    
    a = Angle(hms, unit=unit.deg)
    
    return a.degree


def sixty(deg):
    '''
    Convert from decimal degrees to sexagesimal

    Parameters
    ----------
    
    deg : float
        Value in decimal degrees

    Returns
    -------
    
    dms : float
        Value in sexagesimal degrees (DMS)
    
    '''
    
    a = Angle(deg, unit='deg')
    
    return a.dms.d, a.dms.m, a.dms.s


def date(dates, format='jd'):
    '''
    Convert dates with format YYYY-MM-DD

    Parameters
    ----------
    
    dates : str
        List of dates in format YYYY-MM-DD

    format : str
        Format for the new dates; default is 'jd'. Possibilities are: jd, mjd, iso, yr

    Returns
    -------
    
    '''

    format = format.lower()
    
    ndates = []
    for date in dates:
        time = Time(date)
        
        if format == 'jd':
            ndates.append(time.jd)
        elif format == 'mjd':
            ndates.append(time.mjd)
        elif format == 'iso':
            ndates.append(time.isot)
        elif format == 'yr':
            ndates.append(time.jyear)

    return np.array(ndates)
