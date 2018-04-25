import numpy as np
import astropy.units as unit
import collections

import astropy.units as unt
import astropy.constants as cst

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
        
    

def ten(dms):
    '''
    Convert from sexagesimal to decimal degrees

    Accepted formats are the same as astropy.Angle, e.g.:
      - Angle('1:2:30.43 degrees')
      - Angle('1 2 0 hours')
      - Angle('1°2′3″')
      - Angle('1d2m3.4s')
      - Angle('-1h2m3s')
      - Angle('-1h2.5m')
      - Angle('-1:2.5', unit=u.deg)
      - Angle((10, 11, 12), unit='hourangle')  # (h, m, s)
      - Angle((-1, 2, 3), unit=u.deg)  # (d, m, s)
  
    Parameters
    ----------    
    dms : str, list
        Value in sexagesimal notation (DMS).

    Returns
    -------    
    deg : float
        Value in decimal degrees

    '''

    a = Angle(dms, unit=unit.deg)
        
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


def date(date, format='jd'):
    '''
    Convert dates with format YYYY-MM-DD

    Parameters
    ----------    
    date : str, array of str
       Date in format YYYY-MM-DD

    format : str
        Format for the new date; default is 'jd'. Possibilities are: jd, mjd, iso, yr

    Returns
    -------
    ndates :
    '''

    format = format.lower()

    time = Time(date)
    
    if format == 'jd':
        return time.jd
    elif format == 'mjd':
        return time.mjd
    elif format == 'iso':
        return time.isot
    elif format == 'yr':
        return time.jyear


def stellar_parameters(radius=None, mass=None, logg=None):
    '''
    Determine one of the missing radius/mass/logg parameters from the
    two others

    Parameters
    ----------
    radius : float
        Stellar radius, in Rsun

    mass : float
        Stellar mass, in Msun

    logg : float
        log of surface gravity, in dex cgs
    '''

    # init
    r = radius
    m = mass
    l = logg

    # radius determination
    if (radius is None) and (mass is not None) and (logg is not None):
        m = m * cst.M_sun
        g = 10**(l) * unt.cm / unt.s**2
        r = np.sqrt(cst.G * m / g).to(unt.m)

    # mass determination
    if (radius is not None) and (mass is None) and (logg is not None):
        r = r * cst.R_sun
        g = 10**(l) * unt.cm / unt.s**2
        m = (g * r**2 / cst.G).to(unt.kg)

    # logg determination
    if (radius is not None) and (mass is not None) and (logg is None):
        r = r * cst.R_sun
        m = m * cst.M_sun
        g = (cst.G * m / r**2).to(unt.cm / unt.s**2)
        l = np.log10(g.value)

    if (r is not None) and (m is not None) and (l is not None):
        print('Radius = {:.2f} Rsun'.format((r / cst.R_sun).value))
        print('Mass   = {:.2f} Msun'.format((m / cst.M_sun).value))
        print('log(g) = {:.2f} dex cgs'.format(l))
    else:
        print('Warning: you need to provide at least 2 parameters of radius/mass/logg')

        
def magnitude(mag=None, absmag=None, distance=None, parallax=None):
    '''
    Magnitude/absolute magnitude conversion

    Parameters
    ----------
    mag : float
        Magnitude of object

    absmag : float
        Absolute magnitude of object

    distance : float
        Distance in pc

    parallax : float
        Parallax in mas
    '''

    # distance
    if (distance is not None) and (parallax is None):
        d = distance
        p = 1000 / d
    elif (distance is None) and (parallax is not None):
        p = parallax
        d = 1000 / p
    else:
        print('Warning: you must provide either distance or parallax')
        return
    
    # magnitude
    if (mag is not None) and (absmag is None):
        m = mag
        M = m - 5*(np.log10(d) - 1)
    elif (mag is None) and (absmag is not None):
        M = absmag
        m = M + 5*(np.log10(d) - 1)
    else:
        print('Warning: you must provide either mag or absmag')
        return

    print('distance = {:.2f} pc'.format(d))
    print('parallax = {:.2f} mas'.format(d))
    print('mag      = {:.2f}'.format(m))
    print('abs_mag  = {:.2f}'.format(m))

