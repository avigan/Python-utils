import numpy as np

from astropy.time import Time
from astropy.table import Table


def ema(ma, ecc):
    '''
    eccentric mean anomaly
    '''

    # find the eccentric anomaly from the mean anomaly
    # using Newton's method
    #
    # Smart Eq. 71 p. 114 
    eps = np.double(1e-6)

    # max number of iterations
    niter = 15
    diff = 1e0
    err  = 1e0

    # bound orbit 
    if ecc < 1e0:
        # first guess  ecc anomaly from G. R. Smith
        # 1979 Celestial Mech 19 163
        #
        # M(-E) = -M(E)  hence for E < 0 compute -M(|E|) 
        # true for both elliptical and hyperbolic orbits 
        sgnma = np.sign(ma)      # get the sign of MA
        ma    = sgnma * ma       # make MA positive

        ea = ma + ecc*np.sin(ma) / (1e0 - np.sin(ma+ecc) + np.sin(ma))

        # Other simple first guesses
        #   ea = ma + ecc*np.sin(ma)  #  M < E < M+e
        #   ea = ma +  ecc            #  M < E < M+e
  
        for i in range(niter):
            if (err > eps):
                diff = (ma + ecc * np.sin(ea) - ea) / (ecc * np.cos(ea) - 1.0)
                ea = ea - diff
                err = np.abs( ma-(ea-ecc*np.sin(ea))) 
                # print(i,ea,ma,err)
            else:
                break

        # Try a different start point 
        if err > eps:
            ea = ma + ecc*np.sin(ma) 
            for i in range(niter):
                if (err > eps):
                    diff = (ma + ecc * np.sin(ea) - ea) / (ecc * np.cos(ea) - 1.0)
                    ea = ea - diff
                    err = np.abs( ma-(ea-ecc*np.sin(ea)))
                else:
                    break

        # Try a different start point 
        if err > eps:
            ea = ma + ecc*ma
            for i in range(niter):
                if (err > eps):
                    diff = (ma + ecc * np.sin(ea) - ea) / (ecc * np.cos(ea) - 1.0)
                    ea = ea - diff
                    err = np.abs( ma-(ea-ecc*np.sin(ea)))
                else:
                    break

        # Try a different start point 
        if err > eps:
            ea = ma + ecc
            for i in range(niter):
                if (err > eps):
                    diff = (ma + ecc * np.sin(ea) - ea) / (ecc * np.cos(ea) - 1.0)
                    ea = ea - diff
                    err = np.abs( ma-(ea-ecc*np.sin(ea)))
                else:
                    break

        if err > eps:
            print('EMA failed to converge.')

    else:
        print('Eccentricity > 1')

    return sgnma*ea


def koe(epochs, a, tau, argp, lan, inc, ecc, parallax, Mstar):
    '''
    keplerian orbital elements
    '''
    
    # Epochs in MJD
    # 
    # date twice but x & y are computed 
    # The data are returned so that the values in the array
    # alternate x and y pairs.
    #

    # Gauss's constant for orbital
    # motion ... (kp)^2 = a^3
    #  = sqrt(G*Msun)
    kgauss   =  np.double(0.017202098950)  

    #
    # Keplerian Elements
    #
    # epochs --- dates [JD]
    # a      --- semimajor axis [au]
    # tau    --- epoch of peri in units of the orbital period
    # argp   --- argument of peri [radians]
    # lan    --- longitude of ascending node [radians]
    # inc    --- inclination [radians]
    # ecc    --- eccentricity 
    #
    # Derived quantities
    # manom   --- mean anomaly
    # eccanom --- eccentric anomaly
    # truan   --- true anomaly
    # theta   --- longitude
    # radius  --- star-planet separation

    n = kgauss*np.sqrt(Mstar)*(a)**(-1.5)  # compute mean motion in rad/day

    # ---------------------------------------
    # Compute the anomalies (all in radians)
    #
    # manom = n * (epochs - tau) # mean anomaly 

    manom = n*epochs - 2*np.pi*tau  # mean anomaly w/ tau in units of period
    eccanom = np.array([])
    for man in manom:
        eccanom = np.append(eccanom, ema(man, ecc))

    # ---------------------------------------
    # compute the true anomaly and the radius
    #
    # Elliptical orbit only

    truan = 2.*np.arctan(np.sqrt( (1.0 + ecc)/(1.0 - ecc))*np.tan(0.5*eccanom))
    radius = a * (1.0 - ecc * np.cos(eccanom))

    # ---------------------------------------
    # Compute the vector components in 
    # ecliptic cartesian coordinates (normal convention Green Eq. 7.7)
    #
    #xp = radius *(np.cos(theta)*np.cos(lan) - np.sin(theta)*np.sin(lan)*np.cos(inc))
    #yp = radius *(np.cos(theta)*np.sin(lan) + np.sin(theta)*np.cos(lan)*np.cos(inc))
    #
    # write in terms of \Omega +/- omega --- see Mathematica notebook trig-id.nb
        
    c2i2 = np.cos(0.5*inc)**2
    s2i2 = np.sin(0.5*inc)**2

    arg1 = truan + argp + (np.pi/2-lan)
    arg2 = truan + argp - (np.pi/2-lan)

    c1 = np.cos(arg1)
    c2 = np.cos(arg2)
    s1 = np.sin(arg1)
    s2 = np.sin(arg2)

    xp = radius*(c2i2*c1 + s2i2*c2)
    yp = radius*(c2i2*s1 - s2i2*s2)
    
    # Interleave x & y
    # put x data in odd elements and y data in even elements

    ndate = 2*len(epochs)
    data = np.zeros(ndate)
    data[0::2]  = xp
    data[1::2]  = yp

    return data*parallax  # results in seconds of arc

