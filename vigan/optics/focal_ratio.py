import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.fftpack as fft

from ..utils import imutils


def focal_ratio(img, xthreshold=None, ythreshold=0.001, wave=None, pixel=None, center=True, rebin=2,
                background_fit=True, background_fit_order=2, disp=False, ymin=1e-4):
    '''
    Compute the focal ratio from a PSF image using MTF = |OTF|

    Parameters
    ----------
    img : array
        PSF image

    threshold : float
        Threshold to determine the cutoff of the OTF. Default is 0.001

    wave : float
        Wavelength, in meters. Default is None

    pixel : float
        Pixel size, in meters. Default is None

    center : bool
        Recenter the PSF. Default value is True

    rebin : int
        Rebin factor for accurate recentering. Must be even. Default is 2

    background_fit : bool
        Fit and subtract the background from the OTF. Default is True

    background_fit_order : bool
        Order of the polynomial to fit the background in the OTF. Default is 2

    disp : bool
        Display a summary plot. Default is False

    ymin : float
        Minimal y value in the summary plot. Default is 1e-4

    Returns
    -------
    sampling : float
        Sampling of the PSF, in pixels

    fratio : float
        Focal ratio of the system. Computed only if wave and pixel are both provided.
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
        
    # first value of OTF below threshold
    otf_corr_1d, r = imutils.profile(otf_corr, ptype='mean', step=1, rmax=dim//2-1)
    
    if not xthreshold:
        xthreshold = r.max()
    
    rmax = r[(r <= xthreshold) & (otf_corr_1d >= ythreshold)].max()

    # sampling
    sampling = dim/rmax

    # compute uncertainty related to image sampling
    sampling_min = dim / (rmax + 1)
    sampling_max = dim / (rmax - 1)
    print(f'Sampling: {sampling:.2f} ({sampling_min:.2f}-{sampling_max:.2f})')
    
    # focal ratio
    fratio = None
    if wave is not None and pixel is not None:
        fratio = sampling * pixel / wave

        # compute uncertainty related to image sampling
        fratio_min = sampling_min * pixel / wave
        fratio_max = sampling_max * pixel / wave

        print(f'F ratio: {fratio:.2f} ({fratio_min:.2f}-{fratio_max:.2f})')

    # display result
    if disp:
        otf = otf / otf.max()
        otf_1d, r_otf = imutils.profile(otf, ptype='mean', step=1, rmax=dim//2-1)
        otf_corr_1d, r = imutils.profile(otf_corr, ptype='mean', step=1, rmax=dim//2-1)

        # r_otf = r_otf / (dim//2 - 1) * sampling / 2

        plt.figure('F ratio estimation', figsize=(12, 9))
        plt.clf()
        
        plt.semilogy(r_otf, otf_1d, lw=2, label='MTF')
        plt.semilogy(r_otf, otf_corr_1d, lw=2, linestyle='--', label='MTF (corrected)')

        plt.axhline(ythreshold, linestyle='--', color='r', lw=1)
        plt.axvline(rmax, linestyle='--', color='r', lw=1)

        plt.text(0.2, 0.93, 'sampling = {:.2f} pix / ($\lambda/D$)'.format(sampling),
                 transform=plt.gca().transAxes,
                 fontsize='xx-large', fontweight='bold', ha='left')

        if fratio is not None:
            plt.text(0.2, 0.85, 'F ratio = {:.2f}'.format(fratio),
                     transform=plt.gca().transAxes,
                     fontsize='xx-large', fontweight='bold', ha='left')        
        
        plt.xlim(0, xthreshold)
        plt.xlabel('Cutoff pixel')

        plt.ylim(ymin, 1)
        plt.ylabel('MTF')        
        
        plt.legend(loc='center right')
        plt.tight_layout()
                

    return sampling, fratio
