#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

utilities

"""

###############################################################################
####################        IMPORTS        ####################################
###############################################################################

from numpy import abs, random, sin, sqrt

###############################################################################
###############################################################################

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

#------------------------------------------------------------------------------














