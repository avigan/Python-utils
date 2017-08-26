# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 17:35:52 2016

@author: avigan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft

import vigan.utils.aperture as aperture


def single_power(dim, Dpup, error, power=-2., fmin=None, fmax=None):
    '''
    Produces a phase screen for a power law PSD
    
    Parameters
    ----------
    dim : int
        Dimension of the array, in pixel
     
    Dpup : int
        Diameter of the pupil, in pixel
      
    error : float
        Expected standard deviation of the phase scree, in nm
     
    power : float, optional
        Power of the law. Default is -2, typically corresponding to good quality
        polished optics
     
    fmin : float, optional
        Minimum frequency range, in cycle/pupil. The PSD values below that minimum
        frequency will be set to 0. Default value is 1.

    fmax : float, optional
        Maximum frequency range, in cycle/pupil. The PSD values below that maximum
        frequency will be set to 0. Default value is None
     
    Returns
    -------
    wf : array
        Phase screen expressed in the same unit as error, and with a standard
        deviation equal to error
    '''
    x = np.arange(dim, dtype=np.float64) - dim//2
    y = np.arange(dim, dtype=np.float64) - dim//2
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    
    # mask out null frequency
    rr[dim//2, dim//2] = rr[dim//2, dim//2+1]
    
    # frequencies and PSD
    freq = rr / dim * Dpup
    psd  = freq**power

    # filter out some frequencies
    if (fmin is None):
        fmin = freq.min()
        
    if (fmax is None) or (fmax > Dpup/2):
        fmax = Dpup/2

    psd[(freq < fmin) | (freq > fmax)] = 1e-20
    filt = np.sqrt(psd)
    
    np.random.seed(12345)
    rand = np.empty((dim, dim), dtype=np.complex)
    rand.real = np.random.normal(size=(dim, dim)) * filt
    rand.imag = np.random.normal(size=(dim, dim)) * filt
    
    rand   = fft.fftshift(rand)
    screen = fft.ifft2(rand)
    screen = fft.fftshift(screen)
    
    wf = screen.real
    wf *= error / wf.std()
    wf -= wf.mean()
    
    return wf


if __name__ == '__main__':
    Dpup   = 700
    dim    = 2048
    error  = 10.
    
    wf = single_power(dim, Dpup, error, fmin=1, fmax=300)
    
    print(wf.std())
    plt.imshow(wf)
