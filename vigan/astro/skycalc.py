import io

from skycalc_cli import SkyCalc
from astropy.io import fits


def compute_sky_model(
        # observatory
        observatory=    '2640',

        # airmass
        airmass=           1.0,

        # season and period of night
        pwv_mode=        'pwv',
        season=              0,
        time=                0,
        
        # precipitable water vapor
        pwv=               2.5,
        
        # monthly averaged solar flux
        msolflux=        130.0,
        
        # scattered moon light
        incl_moon=         'Y',
        moon_sun_sep=     90.0,
        moon_target_sep=  45.0,
        moon_alt=         45.0,
        moon_earth_dist=   1.0,
        
        # star light
        incl_starlight=    'Y',
        
        # zodiacal light
        incl_zodiacal=     'Y',
        ecl_lon=         135.0,
        ecl_lat=          90.0,
        
        # molecular emission of lower atmosphere
        incl_loweratm=     'Y',
        
        # molecular emission of upper atmosphere
        incl_upperatm=     'Y',
        
        # airglow continuum
        incl_airglow=      'Y',
        
        # instrument thermal emission
        incl_therm=        'N',
        therm_t1=          0.0,
        therm_e1=          0.0,
        therm_t2=          0.0,
        therm_e2=          0.0,
        therm_t3=          0.0,
        therm_e3=          0.0,
        
        # wavelength grid
        vacair=          'vac',
        wmin=            900.0,
        wmax=           3000.0,
        wgrid_mode= 'fixed_wavelength_step',
        wdelta=            0.1,
        wres=            20000,
        
        # line spread function
        lsf_type=       'none',
        lsf_gauss_fwhm=    5.0,
        lsf_boxcar_fwhm=   5.0):

    '''Compute a sky model using the ESO SkyCalc online tool

    The documentation for each parameter is available online:

    https://www.eso.org/observing/etc/doc/skycalc/helpskycalccli.html

    Below is a very short documentation for the most usefull and
    common parameters

    Parameters
    ----------
    observatory : str
        Possible values are '2400' for La Silla, '2640' for Paranal
        and '3060' for Armazones. Default is '2640'

    airmass : float
        Airmass value. Default is 1.0

    pwv : float
        Precipitable water vapor, in mm. Default is 2.5 (Paranal median)

    wmin : float
        Minimal wavelength, in nanometer. Default is 900 nm

    wmax : float
        Maximal wavelength, in nanometer. Default is 3000 nm

    wgrid_mode : str
        Wavelength grid mode. Possible values are
        'fixed_wavelength_step' or
        'fixed_spectral_resolution'. Default is
        'fixed_wavelength_step'

    wdelta : float
        Wavelength sampling dlambda, in nm. Default is 0.1 nm

    wres : float
        Spectral resolution. Default is 20000

    Returns
    -------
    sky_data : table
        Sky model table. The main columns of interest are:
            * lam: wavelength
            * trans: sky transmission
            * dtrans1: sky transmission -1 sigma uncertainty
            * dtrans2: sky transmission +1 sigma uncertainty
            * flux: sky emission
            * dflux1: sky emission -1 sigma uncertainty
            * dflux2: sky emission +1 sigma uncertainty    
    '''

    sky_model = SkyCalc.SkyModel()
    sky_model.callwith(skydict)

    b = io.BytesIO(sky_model.getdata())
    data = fits.getdata(b)

    return data

