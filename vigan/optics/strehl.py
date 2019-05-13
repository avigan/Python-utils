import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.fftpack as fft
import scipy.interpolate as interp
import vigan

from ..utils import imutils
from . import aperture

from pathlib import Path
from astropy.io import fits


def strehl(img, sampling, center=True, rebin=2,
           background_fit=True, background_fit_order=2, pixel_tf=True,
           pupil_shape='circular', central_obscuration=0, apodizer='none',
           disp=False, ymin=1e-4):
    '''
    Compute Strehl ratio estimation from a PSF image

    Parameters
    ----------
    img : array
        PSF image

    sampling : float
        Number of pixels sampling one resolution element (lambda/D)

    center : bool
        Recenter the PSF. Default value is True

    rebin : int
        Rebin factor for accurate recentering. Must be even. Default is 2

    background_fit : bool
        Fit and subtract the background from the OTF. Default is True

    background_fit_order : bool
        Order of the polynomial to fit the background in the OTF. Default is 2

    pixel_tf : bool
        Taken into account the pixel transfer function. Default is True

    pupil_shape : str
        Type of pupil. Possible values are circular or vlt. Default is circular

    central_obscuration : float
        Value of the central obscuration. Default is 0. Does not apply if
        pupil_shape='vlt'

    apodizer : str
        Apodizer name. Possible values are none, sphere-apo1, sphere-apo2, 
        and sphere-sllc. Default is none

    disp : bool
        Display a summary plot. Default is False

    ymin : float
        Minimal y value in the summary plot. Default is 1e-4

    Returns
    -------
    strehl : float
        Strehl ratio measured on the PSF
    '''

    dim = img.shape[-1]

    if center:
        # image oversampling for subpixel accuracy on the image center
        # determination
        if rebin:
            if ((rebin % 2) != 0) and (rebin != 1):
                raise ValueError('rebin must be even')
        else:
            rebin = 1

        # oversampling
        tmp = fft.fftshift(fft.ifft2(fft.fftshift(img)))
        dim1 = dim * rebin
        tmp = np.pad(tmp, (dim1-dim)//2, mode='constant')
        img_big = fft.fftshift(fft.fft2(fft.fftshift(tmp))).real

        # find maximum
        imax = np.argmax(img_big)
        cy, cx = np.unravel_index(imax, img_big.shape)
        sx, sy = dim1 // 2 - cx, dim1 // 2 - cy

        # recenter
        img_big = imutils.shift(img_big, (sx, sy))

        # OTF
        otf = fft.fftshift(fft.ifft2(fft.fftshift(img_big)).real)
        otf = otf[(dim1-dim)//2:(dim1+dim)//2, (dim1-dim)//2:(dim1+dim)//2]
    else:
        # PSF already centered
        otf = fft.fftshift(np.abs(fft.ifft2(fft.fftshift(img))))

    if background_fit:
        # background subtraction using a linear fit on the first OTF
        # points
        otf_1d, r = imutils.profile(otf, ptype='mean', step=1, exact=False)

        dimfit = background_fit_order + 2
        u = np.arange(1, dimfit+1, dtype=np.float)
        v = otf_1d[1:dimfit+1]

        coeffs = np.polyfit(u, v, background_fit_order)
        poly   = np.poly1d(coeffs)
        fit    = poly(np.arange(dimfit+1))

        if fit[0] >= otf_1d[0]:
            print('Background lower than 0. No correction')
            otf_corr = otf / otf.max()
        else:
            otf_corr = otf.copy()
            otf_corr[dim//2, dim//2] = fit[0]
            otf_corr = otf_corr / fit[0]
    else:
        otf_corr = otf / otf.max()

    # noise subtraction:
    #  NOT IMPLEMENTED

    # pixel transfer function
    if pixel_tf:
        u, v = np.meshgrid(np.arange(dim) - dim // 2, np.arange(dim) - dim // 2)
        pix_tf = np.sinc(1 / dim * u) * np.sinc(1 / dim * v)
    else:
        pix_tf = 1

    # frequencies larger than D/lambda are set to zero
    rt = (dim - 1) / sampling * 2
    mask = aperture.disc(dim, int(rt//2), diameter=False, strict=False, cpix=True)
        
    # divide by pixel transfer function to account for spatial frequencies
    # over one pixel
    otf_corr = otf_corr / pix_tf * mask

    # resolved object:
    #  NOT IMPLEMENTED

    # ideal pupil
    rr = np.ceil(dim / (sampling))
    if pupil_shape.lower() == 'circular':
        pupil = aperture.disc_obstructed(dim, int(rr), central_obscuration, diameter=True, strict=False, cpix=True)
    elif pupil_shape.lower() == 'vlt':
        pupil = aperture.vlt_pupil(dim, int(rr), dead_actuators=None, strict=False, cpix=True)
    else:
        raise ValueError('Unknown pupil shape {}'.format(pupil_shape))

    # apodizer
    if apodizer.lower() == 'none':
        apod = np.ones_like(pupil)
    else:
        if apodizer.lower() == 'sphere-apo1':
            apod_big = fits.getdata(Path(vigan.__file__).parent / 'data' / 'sphere' / 'sphere_APO1_field_transmission_map.fits')
        elif apodizer.lower() == 'sphere-apo2':
            apod_big = fits.getdata(Path(vigan.__file__).parent / 'data' / 'sphere' / 'sphere_APO2_field_transmission_map.fits')
        elif apodizer.lower() == 'sphere-sllc':
            apod_big = fits.getdata(Path(vigan.__file__).parent / 'data' / 'sphere' / 'sphere_SLLC_field_transmission_map.fits')
        else:
            raise ValueError('Unknown apodizer {}'.format(apodizer))
        
        d = apod_big.shape[-1]
        p = apod_big[d//2-1, d//2-1:]        
        r = np.arange(p.size) / (p.size-1)
        
        rho, theta = aperture.coordinates(dim, int(rr), diameter=True, strict=False, cpix=True, polar=True)
        
        interp_fun = interp.interp1d(r, p)
        apod = interp_fun(rho)
        apod = np.nan_to_num(apod)
    
    pupil = pupil*apod
        
    # pupil autocorrelation
    otf_pupil = fft.fftshift(fft.fft2(np.abs(fft.ifft2(pupil))**2).real)
    otf_pupil = otf_pupil / otf_pupil.max()

    # strehl ratio
    strehl = np.sum(otf_corr) / np.sum(otf_pupil)

    # display result
    if disp:
        otf = otf / otf.max()
        otf_1d, r_otf = imutils.profile(otf, ptype='mean', step=1, rmax=dim//2-1)
        otf_corr_1d, r = imutils.profile(otf_corr, ptype='mean', step=1, rmax=dim//2-1)
        otf_pupil_1d, r = imutils.profile(otf_pupil, ptype='mean', step=1, rmax=dim//2-1)

        r_otf = r_otf / (dim//2 - 1) * sampling / 2

        plt.figure('Strehl estimation', figsize=(12, 9))
        plt.clf()
        plt.semilogy(r_otf, otf_1d, lw=2, label='OTF')
        plt.semilogy(r_otf, otf_corr_1d, lw=2, linestyle='--', label='OTF (corrected)')
        plt.semilogy(r_otf, otf_pupil_1d, lw=2, linestyle='-.', label='TF pupil')

        if pixel_tf:
            otf_pixel_1d, r = imutils.profile(pix_tf, ptype='mean', step=1, rmax=dim//2-1)
            plt.semilogy(r_otf, otf_pixel_1d, lw=2, linestyle=':', label='TF pixel')

        plt.text(0.5, 0.90, 'Sr = {:.1f}%'.format(strehl*100), transform=plt.gca().transAxes,
                 fontsize='xx-large', fontweight='bold', ha='center')

        plt.xlim(0, sampling / 2)
        plt.xlabel('Cutoff frequency')

        plt.ylim(ymin, 1)
        plt.ylabel('OTF')

        plt.legend(loc='upper right')
        plt.tight_layout()

    return strehl
