import numpy as np

from scipy import optimize


def distance_from_center(c, x, y):
    '''
    Distance of each 2D points from the center (xc, yc)

    Parameters
    ----------
    c : array_like
        Coordinates of the center

    x,y : array_like
        Arrays with the x,y coordinates
    '''
    xc = c[0]
    yc = c[1]
    
    Ri = np.sqrt((x-xc)**2 + (y-yc)**2)
    
    return Ri - Ri.mean()


def least_square_circle(x, y):
    '''
    Least-square determination of the center of a circle

    Parameters
    ----------
    x,y : array_like
        Arrays with the x,y coordinates of the points on/inside the circle    
    '''
    
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    
    center, ier = optimize.leastsq(distance_from_center, center_estimate, args=(x, y))

    # results
    xc, yc = center
    Ri     = np.sqrt((x-xc)**2 + (y-yc)**2)
    R      = Ri.mean()    
    residu = np.sum((Ri - R)**2)
    
    return xc, yc, R, residu
