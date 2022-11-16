import numpy as np
import astropy.units as u
import scipy.signal as signal
import scipy.interpolate as interpolate


def wavelength_grid(wave_min, wave_max, wave_step=None, resolution=None):
    '''
    Create a wavelength grid, either with constant wavelength step or constant resolution

    All values must be provided in the same unit!
    
    Parameters
    ----------
    wave_min : float
        Minimum wavelength

    wave_max : float
        Maximum wavelength

    wave_step : float
        Spectral sampling
    
    resolution : float
        Spectral resolution

    Returns
    -------
    wave : array
        Wavelength grid

    dwave : array
        Spectral bins grid
    '''

    assert wave_step or resolution, 'Provide either wave_step or resolution'
    assert not (wave_step and resolution), 'Specify either wave_step or resolution'

    if wave_step:
        print(f'Compute wavelength grid with fixed wavelength step ({wave_step})')

        wave  = np.arange(wave_min, wave_max + wave_step/10, wave_step)
        dwave = np.full(len(wave), wave_step)
    else:
        print(f'Compute wavelength grid with fixed resolution ({resolution:.0f})')
        
        wave  = [wave_min]
        dwave = []
        done  = False
        while not done:
            cw = wave[-1]
            dw = cw / resolution
            nw = cw + dw

            dwave.append(dw)
            wave.append(nw)

            if nw > wave_max:
                done = True

        wave  = np.array(wave)[:-1]
        dwave = np.array(dwave)

    return wave, dwave


def downgrade_resolution(wave, flux, wave_step=None, resolution=None, reinterpolate=False):
    '''
    Downgrades the spectral resolution of a spectrum

    Parameters
    ----------
    wave : array
        Wavelength

    flux : array
        Spectrum

    wave_step : float
        New spectral sampling. The unit must be the same as `wave`
    
    resolution : float
        New spectral resolution

    reinterpolate : bool
        Force the reinterpolation of the input data on a regular grid.
        The default is False, just to make sure that the user is aware
        that a reinterpolation happened

    Returns
    -------
    flux : array
        The downgraded spectrum, on the same wavelength grid as the input
    '''

    # basic checks
    assert wave_step or resolution, 'Provide either the new wave_step or the new resolution'
    assert not (wave_step and resolution), 'Specify either the new wave_step or the new resolution'

    # size of wavelength bins
    current_wave_step = np.unique(np.diff(wave))
    if (len(current_wave_step) != 1) and not reinterpolate:
        raise ValueError('The model is not on a regular wavelength grid. Use keyword `reinterpolate=True` to force reinterpolation')

    # reinterpolate on regular grid if needed
    if reinterpolate:
        current_wave_step = current_wave_step.min()
        wave_new, _ = wavelength_grid(wave.min(), wave.max(), wave_step=current_wave_step)

        if np.any(flux < 0):
            interp_f = interpolate.interp1d(wave, flux)
            flux = interp_f(wave_new)
        else:
            interp_f = interpolate.interp1d(wave, np.log10(flux))
            flux = 10**interp_f(wave_new)
    else:
        wave_new = wave.copy()

    # compute new wavelength step if user specified a resolution
    if resolution:
        wave_step = wave.mean() / resolution

    # downgrade the resolution using convolution
    ndw = int(np.ceil(wave_step / current_wave_step))
    kernel = np.zeros(2*ndw)
    kernel[ndw-ndw//2:ndw+ndw//2] = 1
    kernel = kernel / np.sum(kernel)

    flux = signal.convolve(flux, kernel, mode='same')

    # reinterpolate on original wavelength grid if needed
    if reinterpolate:
        if np.any(flux < 0):
            interp_f = interpolate.interp1d(wave_new, flux, bounds_error=False, fill_value=np.nan)
            flux = interp_f(wave)
        else:
            interp_f = interpolate.interp1d(wave_new, np.log10(flux), bounds_error=False, fill_value=np.nan)
            flux = 10**interp_f(wave)

    return flux
