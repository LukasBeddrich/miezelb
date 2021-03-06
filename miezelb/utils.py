#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

utilities

"""

###############################################################################
####################        IMPORTS        ####################################
###############################################################################

from numpy import abs, exp, max, min, pi, random, sin, sqrt, sum, zeros#
from lmfit import Model

###############################################################################
###############################################################################

"""
        FOR CONTRAST FITTING AND MIEZE ANALYSIS
"""

def sinus(x, amp, omega, phase, offset, randomize = True):
    """
    generalized sine function.
    --------------------------------------------------
    
    Arguments
    ----------
    x : np.ndarray : values where sine function is evaluated
    amp : float : amplitude of the sine funtion
        --> range of function values 2*amp
    omega : float : angular frequency
    offset : float : offset of the function
    randomize : bool, optional : if True --> gaussian error with
        std deviation given by sqrt(y = sinus(...)) is added to result
    
    Returns
    ----------
    y : ndarray with float : result of amp * sin(x * omega + phase) + offset
    """
    y = amp * sin(x * omega + phase) + offset
    if randomize:
        y_rndn = sqrt(abs(y))
        return y + y_rndn * random.randn(len(y))
    else:
        return y

#------------------------------------------------------------------------------

def resid_sin(params, x, data, eps_data):
    """
    Calculates residuals for fitting a sine to the MIEZE date with lmfit package
    --------------------------------------------------
    
    Arguments
    ----------
    params : :class:`lmfit.Parameters` parameters for sinus function including
        amp, omega, phase, offset
    x : ndarray : x - values of the MIEZE data (time-bins)
    data : ndarray : y - values of the MIEZE data (neutron counts)
    eps_data : ndarray : yerr - values to weight the residuals
    
    Return
    ----------
    residuals : ndarray : calculated according to
        residuals = (data - sinus(x,amp,omega,phase,offset))/eps_data
    """
    
    y_th = sinus(x, params['amp'], params['omega'], params['phase'], params['offset'], False)
    
    return (data - y_th)/eps_data


###############################################################################
###############################################################################

"""
            FOR THE alternative constructor of DataFrame_NICOS objects
"""

def gen_filename(experiment_number, file_number):
    """
    generates NICOS data file names for TAS .dat and MIEZE .tof files
    --------------------------------------------------
    
    Arguments:
    ----------
    experiment_number   : int   : usually a 5 digit number of an neutron scattering experiment (same as proposal)
                                  if 0 is given, the file is deterimend to be a .tof file by CASCADE
    file_number         : int   : up to 8 digit number describing a certain file of an experiment
    
    Return:
    ----------
    fname               : str   : filename constructed from input
    """
    
    if experiment_number != 0:
        return '{:05d}_{:08d}.dat'.format(experiment_number, file_number)
    else:
        return '{:08d}.tof'.format(experiment_number, file_number)

###############################################################################
###############################################################################

"""
            FOR DETERMINATION OF BEAMSPOTS ON THE 2D CASCADE detector
"""

def ggaussian(x, amp, x0, sigma, offset):
    """
    generalized gaussian function.
    --------------------------------------------------
    
    Arguments
    ----------
    x           : ndarray   : values where gaussian function is evaluated
    amp         : float     : scaling factor of the gaussian funtion
    x0          : float     : center of the gaussian
    sigma       : float     : std deviation
    offset      : float     : offset of the function or constant background
    
    Returns
    ----------
    y           : ndarray   : result of amp / sqrt(2 * pi * sigma**2) * exp(-0.5 * ((x - x0)/sigma)**2) + offset
    """
    
    return amp / sqrt(2 * pi * sigma**2) * exp(-0.5 * ((x - x0)/sigma)**2) + offset

#------------------------------------------------------------------------------

def centcalc_by_weight(data):
    """
    Determines the center (of grtavity) of a neutron beam on a 2D detector by weigthing each pixel with its count
    --------------------------------------------------
    
    Argments:
    ----------
    data        : ndarray   : l x m x n array with 'pixel' - data to weight over m and n
    
    Return:
    ----------
    centers     : ndarray   : l x 2 array with all the centers (cx, cy)
    
    INFO:
    ----------
    1. Method implemented by C. Herb
    2. CHECK the order of cx, cy if it fits to all other interpretations of 2d dimnensions
    """
    
    centerdata = zeros((data.shape[0], 2))
    for row in centerdata:
        x_int = sum(data,axis = 0)
        y_int = sum(data,axis = 1)
        row[0] = sum([i* xval for i,xval  in enumerate(x_int) ])/sum(x_int)
        row[1] = sum([j* yval for j,yval in enumerate(y_int) ])/sum(y_int)
    return centerdata

#------------------------------------------------------------------------------

def centcalc_by_gaussian(data):
    """
    Determines the center (of grtavity) of a neutron beam on a 2D detector by fitting a gaussian to the integrated
    beam profile along one direction (marginal distribution of neutrons)
    --------------------------------------------------
    
    Argments:
    ----------
    data        : ndarray   : l x m x n array with 'pixel' - data to weight over m and n
    
    Return:
    ----------
    centers     : ndarray   : l x 2 array with all the centers (cx, cy)
    
    INFO:
    ----------
    1. Method implemented by C. Herb
    2. CHECK the order of cx, cy if it fits to all other interpretations of 2d dimnensions
    """
    
    centerdata = zeros((data.shape[0], 2))
    gmodel = Model(ggaussian)
    
    for row in centerdata:
        x_int = sum(data,axis = 0)
        y_int = sum(data,axis = 1)
        x_fit = [i for i,_ in enumerate(x_int)]    
        y_fit = [i for i,_ in enumerate(x_int)]
    
        xresult = gmodel.fit(data = x_int, amp = max(x_int), x0 = sum([i * xval for i,xval  in enumerate(x_int)])/sum(x_int), sigma = 15.0, offset = min(x_int), x = x_fit)
        yresult = gmodel.fit(data = y_int, amp = max(y_int), x0 = sum([i * yval for i,yval  in enumerate(y_int)])/sum(y_int), sigma = 15.0, offset = min(x_int), x = y_fit)
        
        row[0] = xresult.params['x0'].value
        row[1] = yresult.params['x0'].value
        
    return centerdata

#------------------------------------------------------------------------------



