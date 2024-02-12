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

def gaussian1d(x, y, window=0, edges=0, mask=None):
    '''Gaussian 1d fit
    
    Parameters
    ----------
    x : array
        Vector of x value

    y : array
        Vector of y value
     
    window : int
        Half-size of sub-window in which to do the fit. If 0, then the fit
        is performed over the full vector. Default is 0

    edges : int
        Number of pixels to hide on the edges. Default is 0

    mask : array
        Mask to apply to the data. Default is None

    Returns
    -------
    params : ?
        Gaussian fit parameters
    '''

    x = x.copy()
    y = y.copy()
    if mask is not None:
        mask = mask.copy()
    
    # hide edges
    if edges > 0:
        x = x[edges:-edges]
        y = y[edges:-edges]
        mask = mask[edges:-edges]

    # apply mask
    if mask is not None:
        x = x[mask]
        y = y[mask]

    # estimate peak position    
    c_int = np.argmax(y)

    # sub-window
    win = window // 2
    if window > 0:        
        if (window % 2):
            raise ValueError('window parameter must be even')
        
        x_sub = x[c_int-win:c_int+win]
        y_sub = y[c_int-win:c_int+win]
        c_init = win
    else:
        x_sub = x
        y_sub = y
        c_init = c_int

    # fit
    g_init = models.Gaussian1D(amplitude=y_sub.max(), mean=x[c_init]) + models.Const1D(amplitude=0)
    
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, x_sub, y_sub)

    return g

def gaussian2d(img, window=0, edges=0, mask=None):
    '''Gaussian 2d fit
    
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
    params : ?
        Gaussian fit parameters
    '''

    img = img.copy()
    
    # hide edges
    if edges > 0:
        img[0:edges, :] = 0
        img[-edges:, :] = 0
        img[:, 0:edges] = 0
        img[:, -edges:] = 0

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
    
    return g


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
        img[0:edges, :] = 0
        img[-edges:, :] = 0
        img[:, 0:edges] = 0
        img[:, -edges:] = 0

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

def polywarp(xi, yi, xo, yo, degree=1):
    """
    Fit a function of the form
    xi[k] = sum over i and j from 0 to degree of: kx[i,j] * xo[k]^i * yo[k]^j
    yi[k] = sum over i and j from 0 to degree of: ky[i,j] * xo[k]^i * yo[k]^j
    Return kx, ky
    len(xo) must be greater than or equal to (degree+1)^2
    """
    if len(xo) != len(yo) or len(xo) != len(xi) or len(xo) != len(yi):
        raise ValueError("Error: length of xo, yo, xi, and yi must be the same")
        return
    if len(xo) < (degree+1.)**2.:
        raise ValueError("Error: length of arrays must be greater than (degree+1)^2")
        return
    
    # ensure numpy arrays
    xo = np.array(xo)
    yo = np.array(yo)
    xi = np.array(xi)
    yi = np.array(yi)

    # set up some useful variables
    degree2 = (degree+1)**2
    x = np.array([xi, yi])
    u = np.array([xo, yo])
    ut = np.zeros([degree2,len(xo)])
    u2i = np.zeros(degree+1)
    for i in range(len(xo)):
        u2i[0] = 1.
        zz = u[1,i]
        for j in range(1,degree+1):
            u2i[j] = u2i[j-1]*zz
        ut[0:degree+1, i] = u2i
        for j in range(1,degree+1):
            ut[j*(degree+1):j*(degree+1)+degree+1,i] = u2i*u[0,i]**j

    uu = ut.T
    kk = np.dot(np.linalg.inv(np.dot(ut,uu).T).T, ut)
    kx = np.dot(kk, x[0,:].T).reshape(degree+1, degree+1)
    ky = np.dot(kk, x[1,:].T).reshape(degree+1, degree+1)

    return kx, ky