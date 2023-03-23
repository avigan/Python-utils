import numpy as np

from collections import deque


def monotonic_sublists(lst):
    '''
    Extract monotonic sublists from a list of values
    
    Given a list of values that is not sorted (such that for some valid
    indices i,j, i<j, sometimes lst[i] > lst[j]), produce a new
    list-of-lists, such that in the new list, each sublist *is*
    sorted: for all sublist \elem returnval: assert_is_sorted(sublist)
    and furthermore this is the minimal set of sublists required to
    achieve the condition.

    Thus, if the input list lst is actually sorted, this returns
    [list(lst)].

    Parameters
    ----------
    lst : list or array
        List of values

    Returns
    -------
    ret_i : list
        List of indices of monotonic sublists
    
    ret_v : list
        List of values of monotonic sublists
    '''

    # Make a copy of lst before modifying it; use a deque so that
    # we can pull entries off it cheaply.
    idx = deque(range(len(lst)))
    deq = deque(lst)
    ret_i = []
    ret_v = []
    while deq:
        sub_i = [idx.popleft()]
        sub_v = [deq.popleft()]

        if len(deq) > 1:
            if deq[0] <= sub_v[-1]:
                while deq and deq[0] <= sub_v[-1]:
                    sub_i.append(idx.popleft())
                    sub_v.append(deq.popleft())
            else:
                while deq and deq[0] >= sub_v[-1]:
                    sub_i.append(idx.popleft())
                    sub_v.append(deq.popleft())
                    
        ret_i.append(sub_i)
        ret_v.append(sub_v)
        
    return ret_i, ret_v

def confidence_interval_statistics(data, CI=[0.99, 0.95, 0.68]):
    '''
    Confidence interval statistics on a set of 2D data

    Parameters
    ----------
    data : array
        An array of data for which the statistics must be computed

    CI : list
        List of confidence intervals (values between 0 and 1)

    Returns
    -------
    data_med : array
        Median of the data points

    data_lim : array
        Limits of the data for the requested confidence intervals
    '''
    
    # confidence intervals
    CI = np.array(CI)

    # lengths
    nci  = len(CI)
    nval = data.shape[0]
    ny   = data.shape[1]
    
    data_med = np.zeros(ny)
    data_lim = np.zeros((nci, 2, ny))
    for y in range(ny):
        values = np.sort(data[:, y])

        cmed = np.median(values)
        data_med[y] = cmed

        for c in range(len(CI)):
            imean = np.argmin(np.abs(values - cmed))
            cmin  = values[int(imean - nval*CI[c]/2)]
            cmax  = values[int(imean + nval*CI[c]/2)]

            data_lim[c, 0, y] = cmin
            data_lim[c, 1, y] = cmax

    return data_med, data_lim

def lines_intersect(a1, a2, b1, b2):
    '''
    Determines the intersection point of two lines passing by points
    (a1,a2) and (b1,b2).

    See https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

    Parameters
    ----------

    a, b : 2D tuples
        Coordinates of points on line 1

    c, d : 2D tuples
        Coordinates of points on line 2

    Returns
    -------
    val
        Returns None is lines are parallel, (cx,cy) otherwise.
    '''

    # make sure we have arrays
    a1 = np.array(a1)
    a2 = np.array(a2)
    b1 = np.array(b1)
    b2 = np.array(b2)

    # test lines
    da = a2 - a1                # vector from A1 to A2
    db = b2 - b1                # vector from B1 to B2
    dp = a1 - b1
    pda = [-da[1], da[0]]       # perpendicular to A1-A2 vector

    # parallel lines
    if (pda*db).sum() == 0:
        return None

    # find intersection
    denom = pda @ db
    num   = pda @ dp

    return (num / denom)*db + b1
