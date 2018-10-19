#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Experiment class

"""

###############################################################################
####################        IMPORTS        ####################################
###############################################################################

from __future__ import print_function

###############################################################################
###############################################################################

class Instrument():
    """
    Instrument class to store instrument related values (might need overhaul in the future)
    """

    RESEDA = {}
    MIRA = {}
    
    RESEDA['distance_SD'] = (2.25, 'm')
    RESEDA['distance_SD_err'] = (0.001,'m')     # 'm' meaning absolute error in the respective unit
    MIRA['distance_SD'] = (1.5, 'm')
    MIRA['distance_SD_err'] = (0.001,'m')
    
    RESEDA['wavelength'] = (6.0, 'A-1')
    RESEDA['wavelength_err'] = (10.0, 'rel')    # 'rel' meaning relative error in percent
    MIRA['wavelength'] = (4.33, 'A-1')
    MIRA['wavelength_err'] = (1.0, 'rel')

#------------------------------------------------------------------------------

    @classmethod
    def get_Parameters(cls, mainkey, *subkeys):
        """
        Retrievs parameters of a specified instrument (At some point, catch sub- and mainkeys individually)
        
        Arguments:
        ----------
        mainkey     : str   : one of the Instruments 'MIRA', 'RESEDA'
        *subkeys    : list  : list of parameters to extract, 'wavelength' etc.
        
        Return:
        ----------
        retdict     : dict  : dictionary containing the parameters specified in the arguments
        """

        retdict = {}
        try:
            for subkey in subkeys:
                print(subkey)
                if mainkey == 'MIRA':
                    print('MIRA checkpoint 1')
                    par = cls.MIRA[subkey]
                    err = cls.MIRA['{}_err'.format(subkey)]
                    if err[1] != 'rel':
                        print('MIRA checkpoint 2')
                        retdict[subkey] = (par[0], err[0], par[1])
                    else:
                        print('MIRA checkpoint 3')
                        retdict[subkey] = (par[0], par[0] * err[0]/100.0, par[1])
                elif mainkey == 'RESEDA':
                    par = cls.RESEDA[subkey]
                    err = cls.RESEDA['{}_err'.format(subkey)]
                    if err[1] != 'rel':
                        retdict[subkey] = (par[0], err[0], par[1])
                    else:
                        retdict[subkey] = (par[0], par[0] * err[0]/100.0, par[1])
        except KeyError:
            print('A key was not recognized. Check for typing error! Individual catches will be implemented later...')
        
        finally:
            return retdict

#------------------------------------------------------------------------------