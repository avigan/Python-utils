import numpy as np
import astropy.constants as cst
import astropy.units as u


def blackbody(wave, T):
    '''
    Blacbody function

    Parameters
    ----------
    wave : float
        Wavelength(s) in micron

    T : float
        Temperature in Kelvin

    Results
    -------
    bb_spectrum : float
        Black body spectrum in W/m2/micron/arcsec2
    '''

    if not hasattr(wave, 'unit'):
        wave = wave * u.micron

    if not hasattr(T, 'unit'):
        T = T * u.K

    exp_part = np.exp(cst.h*cst.c/(wave*cst.k_B*T))
    bb_spectrum = (2*cst.h*cst.c**2/wave**5*1e10)*(exp_part - 1)**(-1) / u.sr
    bb_spectrum = bb_spectrum.to('W/m2/micron/arcsec2')/1e10
    # *1e10 is a trick to avoid rounding errors...

    return bb_spectrum
