import numpy as np
import os.path as path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import astropy.units as unit

from astropy.time import Time
from astropy.coordinates import Angle


def convert_dates(dates):
    '''
    Convert dates with format YYYY-MM-DD

    Parameters
    ----------
    
    dates : str, array_like
        List of dates in format YYYY-MM-DD

    Returns
    -------
    ndates : array_like
        Array of dates in the new formats: jd, j2000 and yr
    '''

    ref = Time('2000-01-01T12:00:00')
    
    ndates_jd = []
    ndates_j2 = []
    ndates_yr = []
    for date in dates:
        time = Time(date)
        ndates_jd.append(time.jd)
        ndates_yr.append(time.jyear)

        day = float(time.yday.split(':')[1])
        j2  = (np.floor(time.jyear) - ref.jyear)*365.25 + day
        ndates_j2.append(j2)

    return np.array(ndates_jd), np.array(ndates_j2), np.array(ndates_yr)


def earth_coord(day):
    '''
    Normalized Earth coordinates on its orbit

    Parameters
    ----------
    
    day : float, array_like
        Array of Julian dates

    Returns
    -------
    
    x, y, z : float, array_like
        Normalized cartesian coordinates
    '''
    n = 2190.5 + 4*365.24 + day
    L = 280.460 + 0.9856474*n
    g = 357.528 + 0.9856003*n
    nL = L//360.
    ng = L//360.
    Lc = L - nL*360.
    gc = g - ng*360.
    
    lmbda = Lc + 1.915 * np.sin(np.deg2rad(gc)) + 0.020 * np.sin(np.deg2rad(2*gc))
    beta  = 0.
    eps   = 23.439 - 0.0000004*n
    R     = 1.00014-0.01671*np.cos(np.deg2rad(gc)) - 0.00014*np.cos(np.deg2rad(2*gc))
    
    x = -R * np.cos(np.deg2rad(lmbda))
    y = -R * np.sin(np.deg2rad(lmbda)) * np.cos(np.deg2rad(eps))
    z = -R * np.sin(np.deg2rad(eps)) * np.sin(np.deg2rad(lmbda))

    return x, y, z


def cart2pol(dx, dy, dx_err=0, dy_err=0, radec=True):
    '''
    Convert cartesian to polar coordinates, with error propagation

    Parameters
    ----------

    dx : float
        Position delta in x

    dy : float
        Position delta in y

    dx_err : float, optional
        Error on position delta in x. Default is 0

    dy_err : float, optional
        Error on position delta in y. Default is 0

    radec : bool, optional
        Are coordinates expressed in RA/DEC on sky. Default is True
    
    Returns
    -------

    sep : float
        Separation

    pa : float
        Position angle, in degrees

    sep_err : float
        Error on separation

    pa_err : float
        Error on position angle, in degrees
    '''
    
    sep = np.sqrt(dx**2 + dy**2)
    if radec:
        pa  = np.mod(np.rad2deg(np.arctan2(dy, -dx)) + 270, 360)
    else:
        pa  = np.mod(np.rad2deg(np.arctan2(dy, dx)) + 360, 360)

    sep_err = np.sqrt(dx**2 * dx_err**2 + dy**2 * dy_err**2) / sep
    pa_err  = np.rad2deg(np.sqrt(dy**2 * dx_err**2 + dx**2 * dy_err**2) / sep**2)


    return sep, pa, sep_err, pa_err


def pol2cart(sep, pa, sep_err=0, pa_err=0, radec=True):
    '''
    Convert cartesian to polar coordinates, with error propagation

    Parameters
    ----------

    sep : float
        Separation

    pa : float
        Position angle, in degrees

    sep_err : float, optional
        Error on separation. Default is 0

    pa_err : float, optional
        Error on position angle, in degrees. Default is 0

    radec : bool, optional
        Are coordinates expressed in RA/DEC on sky. Default is True
    
    Returns
    -------

    dx : float
        Position delta in x

    dy : float
        Position delta in y

    dx_err : float
        Error on position delta in x

    dy_err : float
        Error on position delta in y
    '''

    pa = np.deg2rad(pa)
    pa_err = np.deg2rad(pa_err)
    
    if radec:
        dx = -sep*np.cos(pa+np.pi/2)
        dy = sep*np.sin(pa+np.pi/2)
    else:
        dx = sep*np.cos(pa)
        dy = sep*np.sin(pa)

    dx_err = np.sqrt(np.cos(pa)**2 * sep_err**2 + sep**2 * np.sin(pa)**2 * pa_err**2)
    dy_err = np.sqrt(np.sin(pa)**2 * sep_err**2 + sep**2 * np.cos(pa)**2 * pa_err**2)
    
    return dx, dy, dx_err, dy_err


def track(dates, target_info):
    '''
    Proper motion tracks for a given target and dates

    The target_info parameter is a dictionary with the target
    properties. Mandatory fields are:
      - ra: right ascension as a string in the form 'HH MM SS.sss'
            or as a float in *degrees*
      - dec: declination as a string in the form 'DD MM SS.sss'
             or as a float in *degrees*
      - dist or plx: distance in parsec or parallax in mas
      - pm: proper motions for RA/DEC in mas

    In addition, the following fields can be specified:
      - dist_err or plx_err: distance or parallax error
      - pm_err: proper motions errors
    
    Parameters
    ----------

    dates : str, array_like
        List of dates for the astrometry

    target_info : dict
        Dictionary with essential target properties.

    Returns
    -------

    time : float, array_like
        Time covering the time span of the observations, in years

    dra_track, ddec_track : float, array_like
        Astrometry of a background stationary object, in mas. The astrometry
        is relative to the star.

    '''

    #######################################
    # target info
    #
    
    # coordinates
    ra = target_info['ra']
    if isinstance(ra, str):
        radeg = Angle(ra, unit=unit.hour).degree
    elif isinstance(ra, (int, float)):
        radeg = ra        
    else:
        raise ValueError('Right ascension has not the right type')

    dec = target_info['dec']
    if isinstance(dec, str):
        decdeg = Angle(dec, unit=unit.deg).degree
    elif isinstance(dec, (int, float)):
        decdeg = dec        
    else:
        raise ValueError('Declination has not the right type')

    # proper motion
    pm = target_info['pm']

    # distance
    use_plx  = True
    if ('plx' in target_info):
        plx      = target_info['plx']
    
        if np.logical_not(np.isfinite(plx)):
            use_plx = False
    
    use_dist = False
    if (use_plx is False) and ('dist' in target_info):
        dist     = target_info['dist']
        plx      = 1000/dist
        use_dist = True
    
    if (use_dist is False) and (use_plx is False):
        raise ValueError('Either the distance or the parallax (and associated errors) need to be provided')

    #######################################
    # calculations
    #
    
    # date conversion
    jd, j2, yr = convert_dates(dates)

    # sort
    ii = np.argsort(jd)
    jd = jd[ii]
    j2 = j2[ii]

    # epochs
    delay_days = j2-j2[0]
    days  = np.arange(j2[0], j2[0] + delay_days.max(), 1)
    ndays = days.size
    
    # Earth motion
    x_earth, y_earth, z_earth = earth_coord(days)
    time = np.arange(ndays)/365.25
    
    # parallactic motion constants
    ra  = np.deg2rad(radeg)
    dec = np.deg2rad(decdeg)
    
    eps = np.deg2rad(23.4)
    c = np.cos(ra)
    d = np.sin(ra)
    cp = np.tan(eps) * np.cos(dec) - np.sin(ra) * np.sin(dec)
    dp = np.cos(ra) * np.sin(dec)
    
    # motion of the primary
    pm_ra  = pm[0]
    pm_dec = pm[1]
    
    dra_track  = 0 + pm_ra*time  + plx * ( d*x_earth  - c*y_earth  - d*x_earth[0]  + c*y_earth[0])
    ddec_track = 0 + pm_dec*time + plx * ( dp*x_earth - cp*y_earth - dp*x_earth[0] + cp*y_earth[0])

    return time, dra_track, ddec_track


def plots(target, dates, dra, dra_err, ddec, ddec_err, target_info, link=False, legend_loc=None,
          axes=None, filename=None):
    '''
    Proper motion plot for a given target and candidate astrometry

    The target_info parameter is a dictionary with the target
    properties. Mandatory fields are:
      - ra: right ascension as a string in the form 'HH MM SS.sss'
            or as a float in *degrees*
      - dec: declination as a string in the form 'DD MM SS.sss'
             or as a float in *degrees*
      - dist or plx: distance in parsec or parallax in mas
      - pm: proper motions for RA/DEC in mas

    In addition, the following fields can be specified:
      - dist_err or plx_err: distance or parallax error
      - pm_err: proper motions errors
    
    Optionally, the user can specify the expected distribution of
    proper motion for background objects based on galactic population
    models (e.g. Galaxia):

      - pm_bkg: mean proper motion for background stars
      - pm_bkg_err: error on the mean proper motion for background stars

    When these parameters are provided, the proper motion curves
    taking into account the motion of the background stars are
    overplotted in grey on top of the usual curves.

    Parameters
    ----------
    target : str
        Target or candidate name

    dates : str, array_like
        List of dates for the astrometry

    dra, dra_err, ddec, ddec_err : float, array_like
        Astrometry of the candidate at each date, in mas. The astrometry
        is relative to the central star.

    target_info : dict
        Dictionary with essential target properties: radeg, decdeg, dist,
        dist_err, plx, plx_err, pm, pm_err.

    link : bool
        Link data points with their expected position if background; default is True.

    legend_loc : str or int
        Location of the legend in the RA/DEC plot. Default is None
    
    axes : matplotlib axes
        User provided set of axes. Useful for embedding pm plots in 
        a GUI. Default is None

    filename : str
        Path and file name where to save the plot. The plot is saved only
        if a file name is provided. Default is None
    '''

    #######################################
    # target info
    #
    
    # coordinates
    ra = target_info['ra']
    if isinstance(ra, str):
        radeg = Angle(ra, unit=unit.hour).degree
    elif isinstance(ra, (int, float)):
        radeg = ra        
    else:
        raise ValueError('Right ascension has not the right type')

    dec = target_info['dec']
    if isinstance(dec, str):
        decdeg = Angle(dec, unit=unit.deg).degree
    elif isinstance(dec, (int, float)):
        decdeg = dec        
    else:
        raise ValueError('Declination has not the right type')

    # proper motion
    pm     = target_info['pm']
    pm_err = target_info['pm_err']

    miss_pm = False
    if np.logical_not(np.isfinite(pm_err[0])):
        pm_err[0] = 0
        miss_pm = True

    if np.logical_not(np.isfinite(pm_err[1])):
        pm_err[1] = 0
        miss_pm = True

    # proper motion for background objects
    pm_bkg = target_info.get('pm_bkg', None)
    pm_bkg_err = target_info.get('pm_bkg_err', None)
    if (pm_bkg is not None) and (pm_bkg_err is None):
        pm_bkg_err = [0, 0]

    # distance
    miss_dist = False
    
    use_plx = False
    if ('plx' in target_info) and ('plx_err' in target_info):
        plx      = target_info['plx']
        plx_err  = target_info['plx_err']
        use_plx  = True
        
        if np.logical_not(np.isfinite(plx)):
            use_plx = False
        else:
            if np.logical_not(np.isfinite(plx_err)):            
                print('Missing parallax error: using null error.')
                miss_dist = True
                plx_err = 0
    
    use_dist = False
    if (use_plx is False) and ('dist' in target_info) and ('dist_err' in target_info):
        dist     = target_info['dist']
        dist_err = target_info['dist_err']
        plx      = 1000/dist
        use_dist = True
        
        if np.logical_not(np.isfinite(dist_err)):
            print('Missing distance error: using null error.')
            miss_dist = True
            dist_err = 0
            
    if (use_dist is False) and (use_plx is False):
        raise ValueError('Either the distance or the parallax (and associated errors) need to be provided')

    #######################################
    # calculations
    #
    
    # date conversion
    jd, j2, yr = convert_dates(dates)
    
    # sort
    ii = np.argsort(jd)
    jd       = jd[ii]
    j2       = j2[ii]
    yr       = yr[ii]
    dra      = np.array(dra)[ii]
    dra_err  = np.array(dra_err)[ii]
    ddec     = np.array(ddec)[ii]
    ddec_err = np.array(ddec_err)[ii]
    
    # epochs
    nepoch = jd.size
    delay_days = j2-j2[0]

    # Earth motion
    ext  = 2
    days = np.concatenate((np.flip(np.arange(j2[0] - 1, j2[0] - ext*delay_days.max(), -1)),
                           np.arange(j2[0], j2[0] + (1+ext)*delay_days.max(), 1)))
    ndays = days.size

    x_earth, y_earth, z_earth = earth_coord(days)
    day_min = np.min(np.where(days >= j2[0])[0])
    day_max = np.max(np.where(days <= j2[-1])[0])
    time = np.arange(ndays)/365.25
    
    # parallactic motion constants
    ra  = np.deg2rad(radeg)
    dec = np.deg2rad(decdeg)
    
    eps = np.deg2rad(23.4)
    c = np.cos(ra)
    d = np.sin(ra)
    cp = np.tan(eps) * np.cos(dec) - np.sin(ra) * np.sin(dec)
    dp = np.cos(ra) * np.sin(dec)
    
    # motion of the primary
    pm_ra  = pm[0]
    pm_dec = pm[1]
    pm_ra_err  = pm_err[0]
    pm_dec_err = pm_err[1]
    
    dra_track  = 0 + pm_ra*time  + plx * ( d*x_earth  - c*y_earth  - d*x_earth[day_min]  + c*y_earth[day_min])
    ddec_track = 0 + pm_dec*time + plx * ( dp*x_earth - cp*y_earth - dp*x_earth[day_min] + cp*y_earth[day_min])

    dra_track  -= dra_track[day_min]
    ddec_track -= ddec_track[day_min]

    # motion of the primary, taking into account motion of background objects
    if pm_bkg is not None:
        pm_bkg_ra  = pm[0] - pm_bkg[0]
        pm_bkg_dec = pm[1] - pm_bkg[1]
        pm_bkg_ra_err  = np.sqrt(pm_ra_err**2 + pm_bkg_err[0]**2)
        pm_bkg_dec_err = np.sqrt(pm_dec_err**2 + pm_bkg_err[1]**2)

        dra_bkg_track  = 0 + pm_bkg_ra*time  + plx * ( d*x_earth  - c*y_earth  - d*x_earth[day_min]  + c*y_earth[day_min])
        ddec_bkg_track = 0 + pm_bkg_dec*time + plx * ( dp*x_earth - cp*y_earth - dp*x_earth[day_min] + cp*y_earth[day_min])

        dra_bkg_track  -= dra_bkg_track[day_min]
        ddec_bkg_track -= ddec_bkg_track[day_min]
    
    # error propagation
    if (use_plx is True):
        delta = time-time[day_min]        
        
        dra_track_err = np.sqrt(dra_err[0]**2 + (pm_ra_err*delta)**2 +
                                plx_err**2*((d*x_earth - c*y_earth) - (d*x_earth[day_min] - c*y_earth[day_min]))**2)
        ddec_track_err = np.sqrt(ddec_err[0]**2 + (pm_dec_err*delta)**2 +
                                 plx_err**2*((dp*x_earth - cp*y_earth) - (dp*x_earth[day_min] - cp*y_earth[day_min]))**2)
        
        # error on the motion of the primary, taking into account motion of background objects
        if pm_bkg is not None:
            dra_bkg_track_err = np.sqrt(dra_err[0]**2 + (pm_bkg_ra_err*delta)**2 +
                                        plx_err**2*((d*x_earth - c*y_earth) - (d*x_earth[day_min] - c*y_earth[day_min]))**2)
            ddec_bkg_track_err = np.sqrt(ddec_err[0]**2 + (pm_bkg_dec_err*delta)**2 +
                                         plx_err**2*((dp*x_earth - cp*y_earth) - (dp*x_earth[day_min] - cp*y_earth[day_min]))**2)
            
    else:
        delta = time-time[day_min]

        p1 = 1000/(dist+dist_err)
        p2 = 1000/(dist-dist_err)
    
        dra_track_p1 = 0 + pm_ra*delta + p1 * ( d*x_earth - c*y_earth - d*x_earth[day_min] + c*y_earth[day_min])
        ddec_track_p1 = 0 + pm_dec*delta + p1 * ( dp*x_earth - cp*y_earth - dp*x_earth[day_min] + cp*y_earth[day_min])
    
        dra_track_p2 = 0 + pm_ra*delta + p2 * ( d*x_earth - c*y_earth - d*x_earth[day_min] + c*y_earth[day_min])
        ddec_track_p2 = 0 + pm_dec*delta + p2 * ( dp*x_earth - cp*y_earth - dp*x_earth[day_min] + cp*y_earth[day_min])
    
        dra_dist_err = np.maximum(np.abs(dra_track-dra_track_p1), np.abs(dra_track-dra_track_p2))
        ddec_dist_err = np.maximum(np.abs(ddec_track-ddec_track_p1), np.abs(ddec_track-ddec_track_p2))
    
        dra_track_err = np.sqrt(dra_err[0]**2 + (pm_ra_err*delta)**2 + dra_dist_err**2)
        ddec_track_err = np.sqrt(ddec_err[0]**2 + (pm_dec_err*delta)**2 + ddec_dist_err**2)

        # error on the motion of the primary, taking into account motion of background objects
        if pm_bkg is not None:
            dra_track_p1 = 0 + pm_bkg_ra*delta + p1 * ( d*x_earth - c*y_earth - d*x_earth[day_min] + c*y_earth[day_min])
            ddec_track_p1 = 0 + pm_bkg_dec*delta + p1 * ( dp*x_earth - cp*y_earth - dp*x_earth[day_min] + cp*y_earth[day_min])

            dra_track_p2 = 0 + pm_bkg_ra*delta + p2 * ( d*x_earth - c*y_earth - d*x_earth[day_min] + c*y_earth[day_min])
            ddec_track_p2 = 0 + pm_bkg_dec*delta + p2 * ( dp*x_earth - cp*y_earth - dp*x_earth[day_min] + cp*y_earth[day_min])

            dra_dist_err = np.maximum(np.abs(dra_track-dra_track_p1), np.abs(dra_track-dra_track_p2))
            ddec_dist_err = np.maximum(np.abs(ddec_track-ddec_track_p1), np.abs(ddec_track-ddec_track_p2))

            dra_bkg_track_err = np.sqrt(dra_err[0]**2 + (pm_bkg_ra_err*delta)**2 + dra_dist_err**2)
            ddec_bkg_track_err = np.sqrt(ddec_err[0]**2 + (pm_bkg_dec_err*delta)**2 + ddec_dist_err**2)
    
    # convert data to polar coordinates
    sep, pa, sep_err, pa_err = cart2pol(dra, ddec, dra_err, ddec_err, radec=True)
    
    sep_track, pa_track, sep_track_err, pa_track_err = cart2pol(dra[0]-dra_track, ddec[0]-ddec_track,
                                                                dra_track_err, ddec_track_err, radec=True)

    if pm_bkg is not None:
        sep_bkg_track, pa_bkg_track, \
            sep_bkg_track_err, pa_bkg_track_err = cart2pol(dra[0]-dra_bkg_track, ddec[0]-ddec_bkg_track,
                                                           dra_bkg_track_err, ddec_bkg_track_err, radec=True)
    
    ###########################################
    # plots
    #
    zoom = 1.8
    color_nom = 'k'
    color_bkg = (0.5, 0.5, 0.5)
    color_dat = ['black', 'red', 'teal', 'orange', 'forestgreen', 'purple', 'coral', 'navy', 'gold',
                 'black', 'red', 'teal', 'orange', 'forestgreen', 'purple', 'coral', 'navy', 'gold']
    width = 2
    ms    = 8

    # get fontsize
    fs_init = matplotlib.rcParams['font.size']

    # bigger font for this plot
    matplotlib.rcParams.update({'font.size': 17})

    # create figure if not provided
    if not axes:
        fig = plt.figure(0, figsize=(17, 8))
        fig.clf()
        gs = gridspec.GridSpec(2, 2)
    else:
        ax_main, ax_sep, ax_pa = axes
    
    # RA/DEC plot
    xmin = np.min([np.min(dra[0] - dra_track[day_min:day_max]), dra.min()])
    xmax = np.max([np.max(dra[0] - dra_track[day_min:day_max]), dra.max()])
    ymin = np.min([np.min(ddec[0] - ddec_track[day_min:day_max]), ddec.min()])
    ymax = np.max([np.max(ddec[0] - ddec_track[day_min:day_max]), ddec.max()])
    
    dx = xmax-xmin
    dy = ymax-ymin
    ext = np.max([dx, dy])
    
    dra_min  = (xmin+xmax)/2 - ext/2*zoom
    dra_max  = (xmin+xmax)/2 + ext/2*zoom
    ddec_min = (ymin+ymax)/2 - ext/2*zoom
    ddec_max = (ymin+ymax)/2 + ext/2*zoom

    if not axes:
        ax_main = fig.add_subplot(gs[:, 0])
    ax_main.clear()
    
    ax_main.plot(dra[0] - dra_track, ddec[0] - ddec_track, linestyle='dotted', color=color_nom, zorder=0)
    ax_main.plot(dra[0] - dra_track[day_min:day_max], ddec[0] - ddec_track[day_min:day_max], linestyle='-', color=color_nom, zorder=0)
    
    for e in range(0, nepoch):
        col = colors.to_rgba(color_dat[e])

        ax_main.errorbar(dra[e], ddec[e], xerr=dra_err[e], yerr=ddec_err[e], linestyle='none', marker='o',
                         mew=width, ms=ms, mec=col, color=col, ecolor=col, elinewidth=width, capsize=0,
                         label=dates[e])
    
        if e > 0:
            idx = delay_days.astype(int)-1
            ax_main.errorbar(dra[0] - dra_track[day_min+idx[e]], ddec[0] - ddec_track[day_min+idx[e]],
                             xerr=dra_err[0], yerr=ddec_err[0], linestyle='none', marker='o',
                             mew=0, ms=ms, mec=col, color='w', ecolor=col, elinewidth=width, capsize=0, zorder=-1)
            ax_main.errorbar(dra[0] - dra_track[day_min+idx[e]], ddec[0] - ddec_track[day_min+idx[e]], linestyle='none',
                        marker='o', mew=width, ms=ms, mec=col, color='none', ecolor=col, elinewidth=width, capsize=0,
                        zorder=+1, label=dates[e]+' (if background)')

            if link:
                ax_main.plot((dra[e], dra[0] - dra_track[day_min+idx[e]]), (ddec[e], ddec[0] - ddec_track[day_min+idx[e]]),
                             linestyle='-', color=col)

    if pm_bkg is not None:
        ax_main.plot(dra[0] - dra_bkg_track, ddec[0] - ddec_bkg_track, linestyle='dotted', color=color_bkg, zorder=0)
        ax_main.plot(dra[0] - dra_bkg_track[day_min:day_max], ddec[0] - ddec_bkg_track[day_min:day_max],
                     linestyle='-', color=color_bkg, zorder=0)

        for e in range(1, nepoch):
            col0 = colors.to_rgba(color_dat[e])
            col1 = (col0[0], col0[1], col0[2], 0.5)
            
            ax_main.errorbar(dra[0] - dra_bkg_track[day_min+idx[e]], ddec[0] - ddec_bkg_track[day_min+idx[e]],
                             xerr=dra_err[0], yerr=ddec_err[0], linestyle='none', marker='o', alpha=0.5,
                             mew=0, ms=ms, mec=col, color='w', ecolor=col0, elinewidth=width, capsize=0, zorder=-1)
            ax_main.errorbar(dra[0] - dra_bkg_track[day_min+idx[e]], ddec[0] - ddec_bkg_track[day_min+idx[e]], linestyle='none',
                             marker='o', mew=width, ms=ms, mec=col1, color='none',
                             ecolor=col, elinewidth=width, capsize=0, zorder=+1)
                
    if legend_loc is not None:
        ax_main.legend(loc=legend_loc)
    
    # warnings
    off = 0
    if miss_dist:
        ax_main.text(dra_max-0.03*ext, ddec_max-0.05*ext, u'\u26A0 Missing distance error', fontsize=12)
        off += 0.05

    if miss_pm:
        ax_main.text(dra_max-0.03*ext, ddec_max-(0.05+off)*ext, u'\u26A0 Missing stellar proper motion errors', fontsize=12)
            
    ax_main.set_xlim(dra_max, dra_min)
    ax_main.set_ylim(ddec_min, ddec_max)
    
    ax_main.set_xlabel(r'$\Delta\alpha$ [mas]')
    ax_main.set_ylabel(r'$\Delta\delta$ [mas]')
    
    ax_main.set_title(target)
    
    ax_main.set_aspect('equal', adjustable='box')
    
    # separation plot
    zoom = 1.4
    
    xmin = yr[0]+time[day_min].min()-time[day_min]
    xmax = yr[0]+time[day_max].max()-time[day_min]

    ext = xmax-xmin
    t_min = xmin-ext*0.1
    t_max = xmax+ext*0.1
    
    ymin = np.min([sep_track[day_min:day_max].min(), np.min(sep_track[day_min:day_max]-sep_track_err[day_min:day_max]),
                   np.min(sep_track[day_min:day_max]+sep_track_err[day_min:day_max]), sep.min()])
    ymax = np.max([sep_track[day_min:day_max].max(), np.max(sep_track[day_min:day_max]-sep_track_err[day_min:day_max]),
                   np.max(sep_track[day_min:day_max]+sep_track_err[day_min:day_max]), sep.max()])
    
    ext = ymax-ymin
    sep_min = (ymin+ymax)/2 - ext/2*zoom
    sep_max = (ymin+ymax)/2 + ext/2*zoom

    if not axes:
        ax_sep = fig.add_subplot(gs[0, 1:])
    ax_sep.clear()
    
    ax_sep.axhline(y=sep[0], linestyle='dashed', color='k')

    # user-provided axes ==> usually for plots embeded in Qt app.
    # In that case, fill_between() is extremely slow
    if not axes:
        ax_sep.fill_between(yr[0]+time-time[day_min], sep_track-sep_track_err, sep_track+sep_track_err,
                            linestyle='-', color='b', alpha=0.25)
    else:
        ax_sep.fill_between(yr[0]+time-time[day_min], sep_track-sep_track_err, sep_track+sep_track_err,
                            linestyle='-', color='none', ec='0.5')
    
    ax_sep.plot(yr[0]+time-time[day_min], sep_track, linestyle='-', color=color_nom)

    if pm_bkg is not None:
        ax_sep.fill_between(yr[0]+time-time[day_min], sep_bkg_track-sep_bkg_track_err, sep_bkg_track+sep_bkg_track_err,
                            linestyle='-', color=color_bkg, alpha=0.25)
    
        ax_sep.plot(yr[0]+time-time[day_min], sep_bkg_track, linestyle='-', color=color_bkg)
    
    for e in range(0, nepoch):
        col = colors.to_rgba(color_dat[e])
    
        ax_sep.errorbar(yr[e], sep[e], yerr=sep_err[e], linestyle='none', marker='o',
                        mew=width, ms=ms, mec=col, color=col, ecolor=col, elinewidth=width, capsize=0)
        
    ax_sep.set_ylabel(r'Separation [mas]')
    
    ax_sep.set_xlim(t_min, t_max)
    ax_sep.set_ylim(sep_min, sep_max)
    
    ax_sep.tick_params(labelbottom='off')
        
    # position angle plot
    ymin = np.min([pa_track[day_min:day_max].min(), np.min(pa_track[day_min:day_max]-pa_track_err[day_min:day_max]),
                   np.min(pa_track[day_min:day_max]+pa_track_err[day_min:day_max]), pa.min()])
    ymax = np.max([pa_track[day_min:day_max].max(), np.max(pa_track[day_min:day_max]-pa_track_err[day_min:day_max]),
                   np.max(pa_track[day_min:day_max]+pa_track_err[day_min:day_max]), pa.max()])
    
    ext = ymax-ymin
    pa_min = (ymin+ymax)/2 - ext/2*zoom
    pa_max = (ymin+ymax)/2 + ext/2*zoom

    if not axes:
        ax_pa = fig.add_subplot(gs[1, 1:], sharex=ax_sep)
    ax_pa.clear()
    
    ax_pa.axhline(y=pa[0], linestyle='dashed', color='k')

    # user-provided axes ==> usually for plots embeded in Qt app.
    # In that case, fill_between() is extremely slow
    if not axes:
        ax_pa.fill_between(yr[0]+time-time[day_min], pa_track-pa_track_err, pa_track+pa_track_err,
                           linestyle='-', color='b', alpha=0.25)
    else:
        ax_pa.fill_between(yr[0]+time-time[day_min], pa_track-pa_track_err, pa_track+pa_track_err,
                           linestyle='-', color='none', ec='0.5')
    
    ax_pa.plot(yr[0]+time-time[day_min], pa_track, linestyle='-', color=color_nom)

    if pm_bkg is not None:
        ax_pa.fill_between(yr[0]+time-time[day_min], pa_bkg_track-pa_bkg_track_err, pa_bkg_track+pa_bkg_track_err,
                           linestyle='-', color=color_bkg, alpha=0.25)
    
        ax_pa.plot(yr[0]+time-time[day_min], pa_bkg_track, linestyle='-', color=color_bkg)
    
    for e in range(0, nepoch):
        col = colors.to_rgba(color_dat[e])
    
        ax_pa.errorbar(yr[e], pa[e], yerr=pa_err[e], linestyle='none', marker='o',
                       mew=width, ms=ms, mec=col, color=col, ecolor=col, elinewidth=width, capsize=0)
    
    
    ax_pa.set_ylabel(r'Position angle [deg]')
    ax_pa.set_xlabel(r'Epoch [yr]')
    
    ax_pa.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax_pa.ticklabel_format(axis='y', style='plain', useOffset=False)
    
    ax_pa.set_xlim(t_min, t_max)
    ax_pa.set_ylim(pa_min, pa_max)

    if not axes:
        plt.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.95)

    # save
    if filename:
        plt.savefig(path.expanduser(filename))

    # restore fontsize
    matplotlib.rcParams.update({'font.size': fs_init})


if __name__ == '__main__':
    target   = 'HD_xxxxxx'
    dates    = ['2015-06-07', '2015-07-05', '2016-04-21']
    dra      = [635, 644, 633]
    dra_err  = [1, 6, 4]
    ddec     = [-2368, -2363, -2335]
    ddec_err = [3, 11, 10]
    
    prop = {
        'ra': '18 24 29.7805',
        'dec': '-29 46 49.325',
        'dist': 145.0,
        'dist_err': 25.799999,
        'plx': 1000/145,
        'plx_err': 5.2,
        'pm': [-2.10, -40.2],
        'pm_err': [1.5, 1.5],
        'pm_bkg': [-20, -30],
        'pm_bkg_err': [6.5, 5.8]
    }
    
    plots(target, dates, dra, dra_err, ddec, ddec_err, prop)
