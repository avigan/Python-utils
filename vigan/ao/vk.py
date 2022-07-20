import numpy as np
import scipy.fftpack as fft
import logging
import multiprocessing
import ctypes
import os

from scipy.special import gamma

_log = logging.getLogger(__name__)


def vk_fit_psd(f, r0, L0, fc, turb=False):
    _log.debug('Fitting error')
    
    # constants
    cst = (gamma(11/6)**2 / (2*np.pi**(11/3))) * (24*gamma(6/5)/5)**(5/6)

    # compute PSD
    out = np.zeros(f.shape)
    if turb:
        out = cst*r0**(-5/3) * (f**2 + (1/L0)**2)**(-11/6)
    else:
        f_ind = np.where(f >= fc)
        out[f_ind] = cst*r0**(-5/3) * (f[f_ind]**2 + (1/L0)**2)**(-11/6)

    # total variance
    var_fit = np.sum(out)*f.ravel()[1]**2
    _log.debug(' * variance = {:.6f} rad^2'.format(var_fit))
    
    return out, var_fit


def vk_servo_psd(f, arg_f, Cn2, z, dz, v, arg_v, r0, L0, Td, Ti, fc, gain):
    _log.debug('Servo error')
    
    # constants
    cst = (gamma(11/6)**2 / (2*np.pi**(11/3))) * (24*gamma(6/5)/5)**(5/6)
    h2  = 1

    # PROBLEM IN IDL CODE!!!! ==> to be checked with JFS
    Td = Ti*(np.pi/(2*gain)-1)/2

    # compute PSD
    out   = np.zeros(f.shape)
    f_ind = np.where(f < fc)
    lsum  = 0
    n_layers = len(Cn2)
    for l in range(n_layers):
        lsum = lsum + dz[l]*Cn2[l] *  \
            (1 - 2*np.cos(2*np.pi*Td*v[l]*f[f_ind]*np.cos(arg_f[f_ind]-arg_v[l])) * \
            np.sinc(Ti*v[l]*f[f_ind]*np.cos(arg_f[f_ind]-arg_v[l])) + \
            np.sinc(Ti*v[l]*f[f_ind]*np.cos(arg_f[f_ind]-arg_v[l]))**2)
    
    out[f_ind] = cst*h2*r0**(-5/3)*(f[f_ind]**2+(1/L0)**2)**(-11/6)*lsum

    # total variance
    var_servo = np.sum(out)*f.ravel()[1]**2
    _log.debug(' * variance = {:.6f} rad^2'.format(var_servo))
    
    return out, var_servo
    

def vk_alias_psd(f, arg_f, Cn2, z, dz, v, arg_v, r0, L0,
                 Td, Ti, fc, pyr=False, inf_fun=False):
    _log.debug('Aliasing error')
    
    # constants
    cst = (gamma(11/6)**2 / (2*np.pi**(11/3))) * (24*gamma(6/5)/5)**(5/6)
    h2  = 1

    # useful variables
    out   = np.zeros(f.shape)
    f_ind = np.where(f < fc)
    lsum  = 0
    nmax  = 5
    n_layers = len(Cn2)

    # frequencies
    f_x = f[f_ind]*np.cos(arg_f[f_ind])
    f_y = f[f_ind]*np.sin(arg_f[f_ind])

    if pyr:
        #
        # Pyramid WFS
        #
        raise ValueError('Pyramid not implemented yet')
    else:
        #
        # Shack-Hartmann WFS
        #

        # compute PSD
        for l in range(n_layers):
            tsum = 0
            for n in range(1, nmax+1):
                for m in range(1, nmax+1):
                    f_nx = f_x - 2*n*fc
                    f_my = f_y - 2*m*fc
                    f_nm_sq = f_nx*f_nx + f_my*f_my

                    if inf_fun:
                        raise ValueError('Not implemented')
                    else:
                        spectrum = (f_nm_sq+(1/L0)**2)**(-11/6)
                    tmp  = np.sinc(Ti*v[l]*np.sqrt(f_nm_sq)*np.cos(np.arctan2(f_my, f_nx)-arg_v[l]))
                    tsum = tsum + spectrum*tmp*tmp*(np.sin(2*arg_f[f_ind])/2)**2*(f_y/f_nx + f_x/f_my)**2

                    f_nx = f_x + 2*n*fc
                    f_my = f_y + 2*m*fc
                    f_nm_sq = f_nx*f_nx + f_my*f_my
                     
                    if inf_fun:
                        raise ValueError('Not implemented')
                    else:
                        spectrum = (f_nm_sq+(1/L0)**2)**(-11/6)
                    tmp = np.sinc(Ti*v[l]*np.sqrt(f_nm_sq)*np.cos(np.arctan2(f_my, f_nx)-arg_v[l]))
                    tsum = tsum + spectrum*tmp*tmp*(np.sin(2*arg_f[f_ind])/2)**2*(f_y/f_nx + f_x/f_my)**2

            m = 0
            for n in range(1, nmax+1):
                f_nx = f_x - 2.*n*fc
                f_my = f_y
                f_nm_sq = f_nx*f_nx + f_my*f_my
              
                spectrum = (f_nm_sq+(1/L0)**2)**(-11/6)
                tmp  = np.sinc(Ti*v[l]*np.sqrt(f_nm_sq)*np.cos(np.arctan2(f_my, f_nx)-arg_v[l]))
                tsum = tsum + spectrum*tmp*tmp
              
                f_nx = f_x + 2.*n*fc
                f_nm_sq = f_nx*f_nx + f_my*f_my
              
                spectrum = (f_nm_sq+(1/L0)**2)**(-11/6)
                tmp = np.sinc(Ti*v[l]*np.sqrt(f_nm_sq)*np.cos(np.arctan2(f_my, f_nx)-arg_v[l]))
                tsum = tsum + spectrum*tmp*tmp
                
            n = 0
            for m in range(1, nmax+1):
                f_nx = f_x
                f_my = f_y - 2.*m*fc
                f_nm_sq = f_nx*f_nx + f_my*f_my
              
                spectrum = (f_nm_sq+(1/L0)**2)**(-11/6)
                tmp = np.sinc(Ti*v[l]*np.sqrt(f_nm_sq)*np.cos(np.arctan2(f_my, f_nx)-arg_v[l]))
                tsum = tsum + spectrum*tmp*tmp
              
                f_my = f_y + 2.*m*fc
                f_nm_sq = f_nx*f_nx + f_my*f_my
              
                spectrum = (f_nm_sq+(1/L0)**2)**(-11./6.)
                tmp = np.sinc(Ti*v[l]*np.sqrt(f_nm_sq)*np.cos(np.arctan2(f_my, f_nx)-arg_v[l]))
                tsum = tsum + spectrum*tmp*tmp

            lsum = lsum + dz[l]*Cn2[l]*tsum

        out[f_ind] = cst*h2*r0**(-5/3)*lsum

    # total variance
    var_alias = np.sum(out)*f.ravel()[1]**2
    _log.debug(' * variance = {:.6f} rad^2'.format(var_alias))
    
    return out, var_alias


def vk_aniso_psd(f, arg_f, Cn2, z, dz, r0, L0, fc, theta, azimuth):
    _log.debug('Anisoplanetism error')
    
    # constants
    cst = (gamma(11/6)**2 / (2*np.pi**(11/3))) * (24*gamma(6/5)/5)**(5/6)
    tht = theta / (180/np.pi*3600)
    h2  = 1

    # compute PSD
    out   = np.zeros(f.shape)
    lsum  = 0
    f_ind = np.where(f < fc)
    n_layers = len(Cn2)
    for l in range(n_layers):
        lsum = lsum + dz[l]*Cn2[l]*(1-np.cos(2*np.pi*z[l]*f[f_ind]*np.tan(tht)*np.cos(arg_f[f_ind]-azimuth)))

    out[f_ind] = cst*h2*r0**(-5/3)*(f[f_ind]**2+(1/L0)**2)**(-11/6)*lsum
    
    # total variance
    var_aniso = np.sum(out)*f.ravel()[1]**2
    _log.debug(' * variance = {:.6f} rad^2'.format(var_aniso))

    return out, var_aniso


def n_lambda(wave):
    c1 =    64.328
    c2 = 29498.1
    c3 =   146.0
    c4 =   255.4
    c5 =    41.0

    wave_mic = wave*1e6   # wavelength in micron

    return (c1 + c2/(c3-1/wave_mic**2) + c4/(c5-1/wave_mic**2))*1e-6


def dn_lambda(wave):
    c1 =    64.328
    c2 = 29498.1
    c3 =   146.0
    c4 =   255.4
    c5 =    41.0

    wave_mic = wave*1e6   # wavelength in micron
    
    return -2e-6 * (c2/(c3-1/wave_mic**2)**2 + c4/(c5-1/wave_mic**2)**2) / wave_mic**3


def aniso_refrac(zenith0, wave0):
    return n_lambda(wave0)*np.tan(zenith0)


def aniso_refrac_diff(zenith0, wave0, wave):
    return (n_lambda(wave0)-n_lambda(wave))*np.tan(zenith0)


def vk_diff_refr_psd(f, arg_f, Cn2, z, dz, r0, L0, fc, zenith, wfs_wave, img_wave, azimuth=0):
    _log.debug('Differential refraction error')
    # constants
    cst = (gamma(11/6)**2 / (2*np.pi**(11/3))) * (24*gamma(6/5)/5)**(5/6)
    h2  = 1

    # differential atmospheric refraction (in as)
    theta = aniso_refrac_diff(zenith, wfs_wave, img_wave)

    # compute PSD
    out   = np.zeros(f.shape)
    lsum  = 0
    f_ind = np.where(f < fc)
    n_layers = len(Cn2)
    for l in range(n_layers):
        lsum = lsum + dz[l]*Cn2[l]*(1-np.cos(2*np.pi*z[l]*f[f_ind]*np.tan(theta)*np.cos(arg_f[f_ind]-azimuth)))

    out[f_ind] = cst*h2*r0**(-5/3)*(f[f_ind]**2+(1/L0)**2)**(-11/6)*lsum
    
    # total variance
    var_diff_refr = np.sum(out)*f.ravel()[1]**2
    _log.debug(' * variance = {:.6f} rad^2'.format(var_diff_refr))

    return out, var_diff_refr
    

def vk_diffr_psd(f, fc, alpha):
    _log.debug('Diffraction error')
    f_ind = np.where((f != 0) & (f < fc))

    # compute PSD
    out = np.zeros(f.shape)
    out[f_ind] = alpha*f[f_ind]**(-8/3)/(np.sum(f[f_ind]**(-8/3))*f.ravel()[1]**2)

    # total variance
    var_diffr = np.sum(out)*f.ravel()[1]**2
    _log.debug(' * variance = {:.6f} rad^2'.format(var_diffr))

    return out, var_diffr


def reconstructor_psd(f, arg_f, Dtel, a, r0, L0):
    out = np.zeros_like(f)

    f_x = f*np.cos(arg_f)
    f_y = f*np.sin(arg_f)
    
    red = ((f_x+f_y) / (2*np.pi*f*f*np.sinc(a*f_x)*np.sinc(a*f_y)))**2
    
    fc  = 1/(2*a)
    f_ind = np.where((f != 0) & (f <= fc))
    out[f_ind] = red[f_ind]

    fm  = 1/(2*Dtel)
    f_ind = np.where(f <= fm)
    out[f_ind] = 1/((2*np.pi*fm*np.sinc(a*fm))**2)

    return out
    

def vk_noise_psd(f, arg_f, Dtel, fc, var_wfs, r0, L0, gain=0, pyr=False):
    _log.debug('Reconstruction noise error')
    
    if pyr:
        #
        # Pyramid WFS
        #
        raise ValueError('Pyramid not implemented yet')
    else:
        #
        # Shack-Hartmann WFS
        #
        a = 1/(2*fc)

        # compute PSD
        out = reconstructor_psd(f, arg_f, Dtel, a, r0, L0)

        if gain:
            out = 4*(np.sqrt(gain/0.5)/12.5)*var_wfs*out/np.pi
        else:
            out = 4*var_wfs*out/np.pi

    # total variance
    var_noise = np.sum(out)*f.ravel()[1]**2
    _log.debug(' * variance = {:.6f} rad^2'.format(var_noise))

    return out, var_noise


def vk_ao_psd(f, arg_f, Cn2, z, dz, v, arg_v, r0, L0,
              Td, Ti, Dtel, fc, theta, var_wfs, alpha,
              pyr=False,
              turb=False,
              fit=True,
              servo=True,
              gain=0.5,
              alias=True,
              coeff_alias=0.5,
              diff_refr=True,
              zenith=0.524,
              azimuth=0,
              wfs_wave=0.7e-6,
              img_wave=1.6e-6,
              aniso=True,              
              chrom=True,
              diffr=True,
              noise=True):
    '''
    Compute the PSD of an adaptive optics system

    Parameters
    ----------
    Parameters
    ----------
    f : array
        Module of the spatial frequencies, in m^-1

    agr_f : array
        Argument of the spatial frequencies, in rad
    
    Cn2 : array
        Cn2 profile divided by the integral of the profile

    z : array 
        Altitude of the turbulent layers, in m

    dz : array 
        Thickness of the turbulent layers, in m

    v : array
        Wind speed profile, in m/s

    arg_v : array
        Wind directions, in rad

    r0 : float
        Fried parameter, in m

    L0 : float
        Outer scale, in m

    Td : float
        Time delay in the AO loop, in s

    Ti : float
        Integration time, in s

    Dtel : float
        Telescope dimater, in m

    fc : float
        Cutoff frequency, in m^-1

    theta : float
        ? 

    var_wfs : float
        WFS error variance, in rad^2

    alpha : float
        ?

    pyr : bool
        Use pyramid WFS instead of a Shack-Hartmann. Default is False

    turb : bool
        Only turbulent phase. Default is False

    fit : bool
        Include fitting error term. Default is True

    servo : bool
        Include servo-lag error term. Default is True

    gain : float 
        AO loop gain. Default is 0.5

    alias : bool
        Include aliasing error term. Default is True

    coeff_alias : float
        Aliasing coefficient. A value of 1 means full alias, while a 
        value of 0 means no aliasing. Default is 1

    diff_refr : bool
        Include differential refraction error term. Default is True

    zenith : float
        Guide star zenith angle, in rad. Default is 30° = 0.524 rad

    azimuth : float
        Guide star azimuth angle, in rad. Default is 0

    wfs_wave : float
        WFS wavelength, in m. Default is 0.7e-6

    img_wave : float
        Science imaging wavelength, in m. Default is 1.6e-6

    aniso : bool
        Include anisoplanetism error term. Default is True

    chrom : bool
        Include chromatism error term. Default is True
    
    diffr : bool
        Include diffraction error term. Default is True

    noise : bool
        Include recontruction noise error term. Default is True

    Returns
    -------
    '''

    _log.debug('Computing AO system PSD')
    
    psd_out = np.zeros(f.shape)
    var_ao  = {}
    
    if fit:
        psd, var_fit = vk_fit_psd(f, r0, L0, fc, turb=turb)
        psd_out += psd
        var_ao['fit'] = var_fit

    if not turb:
        if servo:
            psd, var_servo = vk_servo_psd(f, arg_f, Cn2, z, dz, v, arg_v, r0, L0, Td, Ti, fc, gain)
            psd_out += psd
            var_ao['servo'] = var_servo

        if alias:
            psd, var_alias = vk_alias_psd(f, arg_f, Cn2, z, dz, v, arg_v, r0, L0, Td, Ti, fc, pyr=pyr)
            psd_out += coeff_alias * psd
            var_ao['alias'] = var_alias

        if diff_refr:
            psd, var_diff_refr = vk_diff_refr_psd(f, arg_f, Cn2, z, dz, r0, L0, fc, zenith, wfs_wave, img_wave, azimuth=azimuth)
            psd_out += psd
            var_ao['diff_refr'] = var_diff_refr
            
        if aniso:
            psd, var_aniso = vk_aniso_psd(f, arg_f, Cn2, z, dz, r0, L0, fc, theta, azimuth)
            psd_out += psd
            var_ao['aniso'] = var_aniso
            
        if chrom:
            n1 =  23.7 + 6839.4 / (130-(wfs_wave/1.e-6)**(-2))+45.47/(38.9-(wfs_wave/1.e-6)**(-2))
            n2 =  23.7 + 6839.4 / (130-(img_wave/1.e-6)**(-2))+45.47/(38.9-(img_wave/1.e-6)**(-2))

            psd, var_chr = vk_fit_psd(f, r0, L0, fc, turb=True)
            psd_out += ((n2-n1)/n2)**2 * psd
            var_chr *= ((n2-n1)/n2)**2
            var_ao['chrom'] = var_chr

        if diffr:
            psd, var_diffr = vk_diffr_psd(f, fc, alpha)
            psd_out += psd
            var_ao['diffr'] = var_diffr
            
        if noise:
            psd, var_noise = vk_noise_psd(f, arg_f, Dtel, fc, var_wfs, r0, L0, gain=gain, pyr=pyr)
            psd_out += psd
            var_ao['noise'] = var_noise

    return psd_out, var_ao


def array_to_numpy(shared_array, shape, dtype):
    if shared_array is None:
        return None

    numpy_array = np.frombuffer(shared_array, dtype=dtype)
    if shape is not None:
        numpy_array.shape = shape

    return numpy_array


def tpool_init(phs_data_i, phs_shape_i):
    global phs_data, phs_shape

    phs_data  = phs_data_i
    phs_shape = phs_shape_i

    
def compute_phase_screen(idx, local_dim, local_L, psd, img_wave):
    global phs_data, phs_shape

    if (idx % 100) == 0:
        _log.info(f' ==> phase screen {idx}')
    
    phs_np = array_to_numpy(phs_data, phs_shape, dtype=np.float32)

    # random draw of Gaussian noise
    tab = np.random.normal(loc=0, scale=1, size=(local_dim, local_dim))

    # switch to Fourier space
    tab = fft.ifft2(tab)

    # normalize
    tab *= local_dim*local_L

    # multiply with PSD
    tab = tab * np.sqrt(psd)

    # switch back to direct space
    tab = fft.fft2(tab).real

    # normalize
    tab *= img_wave / (2*np.pi) / local_L**2

    # save
    phs_np[idx] = tab


def residual_screen(dim, L, scale, Cn2, z, dz, v, arg_v, r0, L0,
                    Td, Ti, Dtel, fc, theta, var_wfs, alpha,
                    layers=False,
                    full=False,
                    pyr=False,
                    psd_only=False,
                    n_screen=1,
                    chunk_size=500,
                    parallel=False,
                    seed=None,
                    turb=False,
                    fit=True,
                    servo=True,                    
                    gain=0.5,
                    alias=True,
                    coeff_alias=1,
                    diff_refr=True,
                    zenith=0.524,
                    azimuth=0,
                    wfs_wave=0.7e-6,
                    img_wave=1.6e-6,
                    aniso=True,
                    chrom=True,
                    diffr=True,
                    noise=True):
    '''
    Computes a residual phase screen after correction by an 
    adaptive optics system

    Parameters
    ----------
    dim : int
        Numerical size of the screen, in pix
    
    L : float
        Physical size of the screen, in m

    scale : float
        scaling factor of the computed phase screen. Its size will 
        be (scale*L)^2 and its dimension (scale*dim)^2

    Cn2 : array
        Cn2 profile divided by the integral of the profile

    z : array 
        Altitude of the turbulent layers, in m

    dz : array 
        Thickness of the turbulent layers, in m

    v : array
        Wind speed profile, in m/s

    arg_v : array
        Wind directions, in rad

    r0 : float
        Fried parameter, in m

    L0 : float
        Outer scale, in m

    Td : float
        Time delay in the AO loop, in s

    Ti : float
        Integration time, in s

    Dtel : float
        Telescope dimater, in m

    fc : float
        Cutoff frequency, in m^-1

    theta : float
        ? 

    var_wfs : float
        WFS error variance, in rad^2

    alpha : float
        ?

    layers : bool
        Compute and returns all layers independently. Default is False

    full : bool
        Phase screens have a size equal to (dim*scale)*(dim*scale), 
        otherwise their size is dim*dim. Default is False

    pyr : bool
        Use pyramid WFS instead of a Shack-Hartmann. Default is False

    psd_only : bool
        Simply return the PSD, not a residual phase screen

    n_screen : bool
        Number of phase screens to generate. Default is 1. This parameter
        is ignored if psd_only is True

    chunk_size : int
        Number of phase screens generated in parallel. Useful when a very 
        large number of phase screens are requested. Default is 500

    parallel : bool
        Compute screens in parallel rather than in chunks. Default is False
    
    seed : int
        Seed for andom number generator. Default is None

    turb : bool
        Only turbulent phase. Default is False

    fit : bool
        Include fitting error term. Default is True

    servo : bool
        Include servo-lag error term. Default is True

    gain : float 
        AO loop gain. Default is 0.5

    alias : bool
        Include aliasing error term. Default is True

    coeff_alias : float
        Aliasing coefficient. A value of 1 means full alias, while a 
        value of 0 means no aliasing. Default is 1

    diff_refr : bool
        Include differential refraction error term. Default is True

    zenith : float
        Guide star zenith angle, in rad. Default is 30° = 0.524 rad

    azimuth : float
        Guide star azimuth angle, in rad. Default is 0

    wfs_wave : float
        WFS wavelength, in m. Default is 0.7e-6

    img_wave : float
        Science imaging wavelength, in m. Default is 1.6e-6

    aniso : bool
        Include anisoplanetism error term. Default is True

    chrom : bool
        Include chromatism error term. Default is True
    
    diffr : bool
        Include diffraction error term. Default is True

    noise : bool
        Include recontruction noise error term. Default is True

    Returns
    -------
    phs : array
        Phase screen computed from the PSD, in meters. If layers is True, 
        phs is a 3D cube containing the inpependent PSD of each layer.

    OR

    psd : array
        If psd_only is True, the function returns the PSD of the residual 
        phase. If layers is True, the PSD is a 3D cube containing the 
        inpependent PSD of each layer.

    var_ao : dict
        Variance of the different error terms in the PSD.
    '''

    local_dim = dim
    local_L   = L
    fin_dim   = local_dim

    local_dim = local_dim*scale
    local_L   = local_L*scale

    step = local_L / local_dim

    # make sure we work with numpy arrays
    Cn2   = np.array(Cn2)
    z     = np.array(z)
    dz    = np.array(dz)
    v     = np.array(v)
    arg_v = np.array(arg_v)

    if step > (1/(2*fc)):
        raise ValueError('Warning: undersampling')

    # random seed
    if seed:
        np.random.seed(seed)
    
    if layers:
        #
        # compute all layers independently
        #
        n_layers = len(z)
        _log.debug('Computing {} layers'.format(n_layers))

        # make sure L0 has the right number 
        if isinstance(L0, (list, tuple, np.ndarray)):
            nL0 = len(L0)
        else:
            nL0 = 1
        
        if nL0 != n_layers:
            L0 = np.full(n_layers, L0)

        var_ao = []
        if full:
            tab = np.empty((n_layers, n_screen, local_dim, local_dim))
        else:
            tab = np.empty((n_layers, n_screen, dim, dim))

        for layer in range(n_layers):
            _log.debug(' * phase screen {}/{}'.format(layer+1, n_layers))
            local_r0      = Cn2[layer]**(-3/5)*r0
            local_L0      = L0[layer]
            local_var_wfs = var_wfs/n_layers

            idx = (np.arange(n_layers) == layer)
            ctab, cvar_ao = residual_screen(dim, L, scale, Cn2[idx], z[idx], dz[idx], v[idx], arg_v[idx],
                                            local_r0, local_L0, Td, Ti, Dtel, fc, theta, local_var_wfs,
                                            alpha, layers=False, full=full, pyr=pyr, psd_only=psd_only,
                                            turb=turb, fit=fit, servo=servo, gain=gain, alias=alias,
                                            coeff_alias=coeff_alias, diff_refr=diff_refr, zenith=zenith,
                                            azimuth=azimuth, wfs_wave=wfs_wave, img_wave=img_wave,
                                            aniso=aniso, chrom=chrom, noise=noise, diffr=diffr)
            tab[layer] = ctab
            var_ao.append(cvar_ao)

        return tab, var_ao
    else:
        #
        # compute integrated phase screen
        #
        
        # array of spatial frequencies
        if local_dim % 2:
            fx, fy = np.meshgrid((np.arange(local_dim) - (local_dim-1)//2)/local_L,
                                 (np.arange(local_dim) - (local_dim-1)//2)/local_L)
            fx = np.roll(fft.fftshift(fx), 1, axis=1)
            fy = np.roll(fft.fftshift(fy), 1, axis=0)
        else:
            fx, fy = np.meshgrid((np.arange(local_dim) - local_dim//2)/local_L,
                                 (np.arange(local_dim) - local_dim//2)/local_L)
            fx = fft.fftshift(fx)
            fy = fft.fftshift(fy)

        freq = np.sqrt(fx**2 + fy**2)
        arg_freq = np.arctan2(fy, fx)

        # filter noise with the PSD of the residual phase
        psd, var_ao = vk_ao_psd(freq, arg_freq, Cn2, z, dz, v, arg_v, r0, L0, Td, Ti, Dtel, fc,
                                theta, var_wfs, alpha, pyr=pyr, turb=turb, fit=fit, servo=servo,
                                gain=gain, alias=alias, coeff_alias=coeff_alias, diff_refr=diff_refr,
                                zenith=zenith, azimuth=azimuth, wfs_wave=wfs_wave, img_wave=img_wave,
                                aniso=aniso, chrom=chrom, diffr=diffr, noise=noise)

        if psd_only:
            #
            # return just the PSD
            #
            if local_dim % 2:
                psd = np.roll(np.roll(fft.fftshift(psd), -1, axis=1), -1, axis=0)
            else:
                psd = fft.fftshift(psd)
                
            return psd, var_ao
        else:
            #
            # compute phase screens
            #
            _log.info('Generating {} phase screen(s)'.format(n_screen))

            if parallel:
                phs_shape = (n_screen, local_dim, local_dim)
                phs_data  = multiprocessing.RawArray(ctypes.c_float, int(np.prod(phs_shape)))

                ncpu = int(os.environ.get('SLURM_JOB_CPUS_PER_NODE', multiprocessing.cpu_count()))
                pool = multiprocessing.Pool(processes=ncpu, initializer=tpool_init, initargs=(phs_data, phs_shape))

                tasks = []
                for idx in range(n_screen):
                    tasks.append(pool.apply_async(compute_phase_screen, args=(idx, local_dim, local_L, psd, img_wave)))
                    # compute_correlation(i)
    
                pool.close()
                pool.join()

                phs = array_to_numpy(phs_data, phs_shape, np.float32)
            else:
                # split in several chunks
                nchunk = n_screen // chunk_size
                rem    = n_screen % chunk_size

                chunks = np.array([], dtype=np.int64)
                if nchunk > 0:
                    chunks = np.append(chunks, np.full(nchunk, chunk_size))
                if rem > 0:
                    chunks = np.append(chunks, rem)
                    nchunk += 1

                phs = np.empty((n_screen, local_dim, local_dim), dtype=np.float32)
                for c in range(nchunk):
                    if nchunk > 1:
                        _log.info(' * chunk {} / {}'.format(c+1, nchunk))

                    # size of current chunk
                    chunk = chunks[c]
                    idx   = np.sum(chunks[:c])

                    # random draw of Gaussian noise
                    tab = np.random.normal(loc=0, scale=1, size=(chunk, local_dim, local_dim))

                    # switch to Fourier space
                    tab = fft.ifft2(tab)

                    # normalize
                    tab *= local_dim*local_L

                    # multiply with PSD
                    tab = tab * np.sqrt(psd)

                    # switch back to direct space
                    tab = fft.fft2(tab).real

                    # normalize
                    tab *= img_wave / (2*np.pi) / local_L**2

                    # save
                    phs[idx:idx+chunk, ...] = tab

            if not full:
                phs = phs[0:fin_dim, 0:fin_dim]

            return phs, var_ao

    
def residual_screen_sphere(seeing, L0, z, Cn2, v, arg_v, magnitude, zenith, azimuth,
                           spat_filter=0.7, img_wave=1.593e-6, dim_pup=240, n_screen=1, chunk_size=500, parallel=False,
                           turb=False, fit=True, servo=True, alias=True, noise=True, diff_refr=True,
                           psd_only=False, seed=None):
    '''
    Produce residual phase screens based on the VLT/SPHERE system

    Parameters
    ----------
    seeing : as
        Seeing @ 0.5 micron, in arcsec

    L0 : float
        Outer scale, in m
    
    z : array 
        Altitude of the turbulent layers, in m

    Cn2 : array
        Cn2 profile divided by the integral of the profile

    v : array
        Wind speed profile, in m/s

    arg_v : array
        Wind directions, in rad

    magnitude : float
        R-band magnitude of guide star

    zenith : float
        Zenith angle of guide star, in deg

    azimuth : float
        Azimuth angle of guide star, in deg

    spat_filter : float
        Efficiency of the spatial filter. A value of 1 means no aliasing, while a 
        value of 0 means complete aliasing. Default value is 0.7

    img_wave : float
        Science imaging wavelength, in m. Default is 1.6e-6

    dim_pup : int
        Size of the simulated pupil, in pixel.

    n_screen : bool
        Number of phase screens to generate. Default is 1. This parameter
        is ignored if psd_only is True

    chunk_size : int
        Number of phase screens generated in parallel. Useful when a very 
        large number of phase screens are requested. Default is 500

    parallel : bool
        Compute screens in parallel rather than in chunks. Default is False
    
    turb : bool
        Only turbulent phase. Default is False

    fit : bool
        Include fitting error term. Default is True

    servo : bool
        Include servo-lag error term. Default is True

    alias : bool
        Include aliasing error term. Default is True

    noise : bool
        Include recontruction noise error term. Default is True

    diff_refr : bool
        Include differential refraction error term. Default is True

    psd_only : bool
        Simply return the PSD, not a residual phase screen

    seed : int
        Seed for andom number generator. Default is None

    Returns
    -------
    phs : array
        Phase screen computed from the PSD, in meters.

    OR

    psd : array
        If psd_only is True, the function returns the PSD of the residual 
        phase.
    '''
    
    #
    # system parameters
    #

    # telescope
    Dtel          = 8            # telescope diameter [m]
    ri            = 0.14         # central obscuration ratio
    truezeropoint = 2.1e10       # [e-/m2/s]

    # AO user parameters
    nsspup   = 40           # linear number of sub-apertures
    Ti       = 0.725e-3     # integration time [s]

    # AO internal choices
    chrom    = False
    diffr    = False
    aniso    = False

    Td       = 1.56e-3        # detector readout + computation
    wfs_wave = 0.7e-6         # WFS central wavelength [m]
    ron      = 0.1            # detector readout noise [e-]
    emccd    = True           # use of EMCCD
    wcog     = True           # weighted center of gravity
    Nd       = 2
    f00      = 1 / (4*(Ti + 2*Td))
    gain     = 2*(np.pi*f00*Ti)**2 / (np.sin(np.pi*f00*Ti))

    #
    # simulation parameters
    #
    tabscale = 2  # size of the simulated corrected phase screen in Dtel unit

    #=================================================================
    #=================================================================

    #
    # preliminary computations
    #
    nl     = len(z)                       # number of layers
    z      = np.array(z)*1000             # km ==> m
    dz     = np.ones(nl)                  # thickness [m]
    Cn2    = np.array(Cn2)/100            # % ==> fraction
    arg_v  = np.deg2rad(np.array(arg_v))  # deg ==> rad
    zenith = np.deg2rad(zenith)

    r01       = 0.976*0.5/seeing/4.85*(np.cos(zenith))**(3/5)
    r0        = r01*(img_wave/(0.5e-6))**(6/5)
    r0wfs     = r01*(wfs_wave/(0.5e-6))**(6/5)
    # zeropoint = truezeropoint * (np.pi*Dtel**2/4) / 2

    pitch  =  Dtel/nsspup                 # pitch: inter-actuator distance [m]
    fc     =  1/(2*pitch)                 # pitch frequency (1/2a) [m^-1]
    # N      =  np.pi*(Dtel*fc)**2          # number of actuator

    Nphoton =  truezeropoint*10**(-0.4*magnitude)  # [ph/s/m^2/micron]
    Nphoton =  Nphoton*pitch**2*Ti                 # [photon]

    coeff_alias =  1-spat_filter

    #
    # computation of noise on WFS measurements
    #

    # computation of Nt
    if pitch/r0wfs < 2:
        Nt = Nd
    else:
        # formule de la PSF corrige du tilt
        Nt = Nd*pitch/r0wfs*(np.sqrt(1-(r0wfs/pitch)**(1/3)))

    if wcog:
        # computation of Nw
        if Nphoton/ron <= 2:
            Nw = Nt
        if Nphoton/ron > 5:
            Nw = 1.5 * Nt
        if Nphoton/ron > 10:
            Nw = 2.0/ron * Nt
        if Nphoton/ron > 20:
            Nw = 3.0 * Nt

        alphar      = Nw**2 / (Nt**2 + Nw**2)
        var_pho_wfs = np.pi**2/(2*np.log(2))*1/Nphoton*(Nt/Nd)**2*((Nt**2+Nw**2)/(2*Nt**2+Nw**2))**2/alphar**2
        var_ron_wfs = np.pi**3/(32*(np.log(2))**2)*(ron**2/Nphoton**2)*((Nt**2+Nw**2)/Nd)**2/alphar**2
        gain1       = gain*alphar
    else:
        # Computation of Ns
        Ns          = 1.5 * Nt
        var_pho_wfs = np.pi**2/(2*np.log(2))*1/Nphoton*(Nt/Nd)**2
        var_ron_wfs = np.pi**2/3*(ron/Nphoton)**2*(Ns**2/Nd)**2
        gain1       = gain

    if emccd:
        var_pho_wfs = 2*var_pho_wfs

    var_wfs = var_pho_wfs + var_ron_wfs
    
    L       = tabscale*Dtel           # size of the simulated phase screen [m]
    dim     = tabscale*dim_pup        # size of the simulated phase screen [pixels]

    alpha   = 0    # ?
    theta   = 0    # ?

    #=================================================================
    #=================================================================

    # additional fixed parameters
    full    = True
    layers  = False
        
    # compute result
    res, var_ao = residual_screen(dim, L, 1, Cn2, z, dz, v, arg_v, r0, L0,
                                  Td, Ti, Dtel, fc, theta, var_wfs, alpha,
                                  layers=layers, turb=turb, seed=seed, diff_refr=diff_refr,
                                  wfs_wave=wfs_wave, img_wave=img_wave, chrom=chrom,
                                  zenith=zenith, azimuth=azimuth, aniso=aniso, 
                                  fit=fit, servo=servo, alias=alias, noise=noise,
                                  gain=gain1, coeff_alias=coeff_alias, diffr=diffr,
                                  full=full, psd_only=psd_only, n_screen=n_screen,
                                  chunk_size=chunk_size, parallel=parallel)

    return res



if __name__ == '__main__':
    #
    # atmopsheric parameters
    #

    # realistic values
    seeing = 0.64                                      # @ 0.5 micron ["]
    L0     = 25                                        # outer scale [m]
    z      = [0.03, 0.09, 0.15,   5.5,  10.5,  13.5]   # altitude [km]
    Cn2    = [  48,   24,   19,     2,     3,     4]   # weight [%]
    v      = [ 5.5,  5.5,  5.1,  11.5,  32.0,  14.5]   # wind speed [m/s]
    arg_v  = [69.4, 69.4, 70.1, 284.3, 276.9, 275.9]   # wind direction [deg]

    #
    # guide star parameters
    #
    magnitude = 4                # GS magnitude (WFS band)
    zenith    = 30               # zenith angle
    azimuth   = 0                # azimuth angle

    #
    # simulation parameters
    #
    seed      = None
    dim_pup   = 240
    
    n_screen  = 5

    # imports for example
    import vigan.optics.aperture as aperture
    import vigan.optics.mft as mft
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    
    #
    # creation of corrected phase screen
    #
    phs = residual_screen_sphere(seeing, L0, z, Cn2, v, arg_v, magnitude, zenith, azimuth,
                                 spat_filter=0.5, img_wave=1.593e-6, dim_pup=dim_pup, n_screen=n_screen,
                                 fit=True, servo=True, alias=True, noise=True, diff_refr=True,
                                 psd_only=False, seed=seed)
    
    # reformating    
    dim = phs.shape[-1]
    pupil = aperture.disc_obstructed(dim_pup, dim_pup, 0.14, diameter=True, cpix=False)
    phs = phs[..., dim_pup:2*dim_pup, dim_pup:2*dim_pup]
    phs[..., :, :] *= pupil

    # plot result
    cphs = phs[0]*1e9
    plt.figure(0)
    plt.clf()
    cim = plt.imshow(cphs, interpolation='nearest', vmin=-200, vmax=200)
    plt.text(0.05, 0.93, '{:.1f} nm rms'.format(cphs[pupil != 0].std()), color='w',
             weight='bold', fontsize='x-large', transform=plt.gca().transAxes)
    plt.colorbar(cim, label='Phase error [microns]')
    plt.tight_layout()
