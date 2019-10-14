import numpy as np

from scipy import optimize
from astropy.modeling import models, fitting


def _distance_from_center(c, x, y):
    '''
    Distance of each 2D points from the center (cx, cy)

    Parameters
    ----------
    c : array_like
        Coordinates of the center

    x,y : array_like
        Arrays with the x,y coordinates
    '''
    cx = c[0]
    cy = c[1]
    
    Ri = np.sqrt((x-cx)**2 + (y-cy)**2)
    
    return Ri - Ri.mean()


def circle(x, y):
    '''
    Least-square circle fit

    Parameters
    ----------
    x,y : array_like
        x,y coordinates of the points belonging to the circle    

    Returns
    -------
    cx, cy : float
        Center of the circle
    
    radius : float
        Radius of the circle
    '''
    
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    
    center, ier = optimize.leastsq(_distance_from_center, center_estimate, args=(x, y))

    # results
    cx, cy = center
    Ri     = np.sqrt((x-cx)**2 + (y-cy)**2)
    radius = Ri.mean()    
    # residuals = np.sum((Ri - R)**2)
    
    return cx, cy, radius


def peak_center(img, window=0, edges=0, mask=None):
    '''Determine the center position of a peak 
    
    Parameters
    ----------
    img : array
        Image where the peak must be found
     
    window : int
        Half-size of sub-window in which to do the fit. If 0, then the fit
        is performed over the full image. Default is 0

    edges : int
        Number of pixels to hide on the edges. Default is 0

    mask : array
        Mask to apply to the data. Default is None

    Returns
    -------
    cx, cy : tuple
        Center of the peak

    '''

    img = img.copy()
    
    # hide edges
    if edges > 0:
        img[:edges, :] = 0
        img[:, :edges] = 0
        img[edges:, :] = 0
        img[:, edges:] = 0

    # apply mask
    if mask is not None:
        img *= mask

    # estimate peak position    
    imax = np.unravel_index(np.argmax(img), img.shape)
    cx_int = imax[1]
    cy_int = imax[0]
    
    # sub-window
    win = window // 2
    if window > 0:        
        if (window % 2):
            raise ValueError('window parameter must be even')
        
        sub = img[cy_int-win:cy_int+win, cx_int-win:cx_int+win]

        cx_init = win
        cy_init = win
    else:
        sub = img
 
        cx_init = cx_int
        cy_init = cy_int
        
    # fit
    x, y = np.meshgrid(np.arange(sub.shape[1]), np.arange(sub.shape[0]))
    g_init = models.Gaussian2D(amplitude=sub.max(), x_mean=cx_init, y_mean=cy_init) + \
                                          models.Const2D(amplitude=0)
    
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, x, y, sub)
    
    cx = g[0].x_mean.value - cx_init + cx_int
    cy = g[0].y_mean.value - cy_init + cy_int
    
    return cx, cy
