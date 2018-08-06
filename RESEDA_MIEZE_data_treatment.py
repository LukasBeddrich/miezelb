#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:14:14 2017

@author: lbeddric
"""

###############################################################################
##################                      #######################################
################## RESEDA data analysis #######################################
##################                      #######################################
###############################################################################

### Basic Imports #############################################################

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

from re import split
from matplotlib.cm import plasma, Greys
from matplotlib.colors import LogNorm
#from scipy.optimize import curve_fit
from lmfit import minimize, Parameters
from scipy import constants as const

###############################################################################

### Pathes ####################################################################

global drp
drp = os.path.dirname(os.path.abspath(__file__))

### Basic functions ###########################################################

def dpath(fname, drootpath):
    return os.path.join(drootpath, fname)

#------------------------------------------------------------------------------

def sinus(x, amp, omega, phase, offset, random = True):
    y = amp * np.sin(x * omega + phase) + offset
    if random:
        y_rndn = np.sqrt(np.abs(y))
        return y + y_rndn * np.random.randn(len(y))
    else:
        return y

#------------------------------------------------------------------------------
    
def resid_sin(params, x, data, eps_data):
    """
    lmfit residuals for sin fit
    """
    
    y_th = sinus(x, params['amp'], params['omega'], params['phase'], params['offset'], False)
    
    return (data - y_th)/eps_data
    

###############################################################################

class DataFrame_Base(object):
    """
    Base class providing meta and experimental data of the IGOR hdf5 files
    of the RESEDA experiment
    """

    def __init__(self, dfile, drootpath = ''):
        self.fpath = dpath(dfile, drootpath)
        self.rawhdfdata = h5py.File(self.fpath, 'r')
#       No reason not to load jobs immediately
#        self.jobs = []
        self.__load_jobs()
        
        self.data_dict = {}
        self.mieze_taus = {}
        self.monitor = {}
        
        # IMPORTANT: just for this experiment!
        self.wavelength = 6.0 #Angstroem
        self.d_SD = 2.25 #sample-detector-distance in meters

#------------------------------------------------------------------------------
        
    def __repr__(self):
        """
        official string description
        """
        return 'DataFrame_Base class instance:\
        hdf_filepath = {}'.format(str(self.fpath))
    
#------------------------------------------------------------------------------
    
    def __load_monitor_counts(self):
        """
        loads monitor counts for all jobs in self.jobs
        """
        for key in self.jobs:
            self.monitor[key] = self.rawhdfdata[key+'/ew_Counts_0'].attrs['ev_Monitor']

#------------------------------------------------------------------------------

    def __load_jobs(self):
        """
        loads list of differen MIEZEecho jobs
        """
        self.jobs = sorted([k for k in self.rawhdfdata.keys() if "Echo" in k], \
                            key = lambda z: int(split('(\d+)',z)[1]))

#------------------------------------------------------------------------------
    
    def _calc_miezetau(self, **kwargs):
        """
        calculate mieze time from experimental setup
        
        WARNING: The current output is not necessarily correct. check with someone who knows RESEDA and MIEZE!!
        """
        if 'job' in kwargs:
            freqs = self.rawhdfdata[kwargs['job'] + '/ew_Counts_0'].attrs['ChangedHF']
        elif 'jobind' in kwargs:
            freqs = self.rawhdfdata[self.jobs[kwargs['jobind']] + '/ew_Counts_0'].attrs['ChangedHF']
        else:
            raise KeyError("No job in hdf file is specified properly.")
            
        deltaFreq = freqs[1][1]-freqs[1][2] # looks closest to what makes sense...
            
        tau = 2 * deltaFreq * self.d_SD * 2*const.m_n**2 * (self.wavelength * 1.0e-10)**3 / const.h**2 * 1.0e9
        # errorcalculation
        tau_err = tau * (0.117 + 0.0005)
        
#        tau_err = 0.117 * (self.wavelength * 1e-10) * (2 * const.m_n**2 / const.h ** 2 * 3 * (self.wavelength * 1e-10) * (2*deltaFreq) * self.d_SD * 1e9)** 2 + 0.0005 * (2 * const.m_n ** 2 / const.h ** 2 * 3 * (self.wavelength * 1e-10) ** 2 * (2*deltaFreq) * 1e9)                    
#        tau_err = 0.117 * (wavelength * 1e-10) * (2 * const.m_n**2 / const.h ** 2 * 3 * (wavelength * 1e-10) * deltaFreq * distance * 1e9)** 2 + 0.0005 * (2 * const.m_n ** 2 / const.h ** 2 * 3 * (wavelength * 1e-10) ** 2 * deltaFreq * 1e9)
                    
        return tau, tau_err
            
    
#------------------------------------------------------------------------------    
    
    def getjobs(self):
        try:
            print self.jobs
        except AttributeError:
            self.__load_jobs()
            print self.jobs
            
#------------------------------------------------------------------------------
            
    def load_specificjobdata(self, **kwargs):
        """
        loads the detector foil data, monitor counts, mieze time of one specific job
        
        """
        if 'job' in kwargs:
            key = kwargs['job']
        elif 'jobind' in kwargs:
            key = self.jobs[kwargs['jobind']]
            
# =============================================================================
#         try:
#             self.monitor[key] = self.rawhdfdata[key+'/ew_Counts_0'].attrs['ev_Monitor']
#             self.mieze_taus[key] = self._calc_miezetau(job = key)
#             self.data_dict[key] = self.rawhdfdata[key]['ew_MDMiezeCounts_0'].value.reshape(8,16,128,128)
#             if 'norm_mon' in kwargs:
#                 if kwargs['norm_mon']: self.data_dict[key] /= self.monitor[key]
#                 
#         except AttributeError:
#             print 'jobs might not been loaded!'
# =============================================================================
        # add errors so that it becomes a 5dim array
        self.data_dict[key] = np.zeros((2,8,16,128,128))
        try:
            self.monitor[key] = self.rawhdfdata[key+'/ew_Counts_0'].attrs['ev_Monitor']
            self.mieze_taus[key] = self._calc_miezetau(job = key)
            self.data_dict[key][0] = self.rawhdfdata[key]['ew_MDMiezeCounts_0'].value.reshape(8,16,128,128)
            self.data_dict[key][1] = np.sqrt(self.rawhdfdata[key]['ew_MDMiezeCounts_0'].value.reshape(8,16,128,128))
            if 'norm_mon' in kwargs:
                if kwargs['norm_mon']:
                    self.data_dict[key][1] = np.sqrt(1./self.data_dict[key][0] + 1./self.monitor[key])
                    self.data_dict[key][0] /= self.monitor[key]
                    # dR = R * sqrt( |dI/I|**2 + |dmon/mon|**2) with dI = sqrt(I) and dmon = sqrt(mon) 
                    self.data_dict[key][1] *= self.data_dict[key][0]
                    
# =============================================================================
#                     self.data_dict[key][0] /= self.monitor[key]
#                     # dR = R * sqrt( |dI/I|**2 + |dmon/mon|**2) with dI = sqrt(I) and dmon = sqrt(mon) 
#                     self.data_dict[key][1] = self.data_dict[key][0] * np.sqrt(1./self.data_dict[key][0] + 1./self.monitor[key])
# =============================================================================
                
        except AttributeError:
            print 'jobs might not been loaded!'

        
#------------------------------------------------------------------------------
        
    def load_alljobsdata(self):
        """
        
        """
        for job in self.jobs:
            if job not in self.data_dict:
                self.load_specificjobdata(job = job, norm_mon = True)
#            if job not in self.data_dict.keys():
#                self.data_dict[job] = self.rawhdfdata[job]['ew_MDMiezeCounts_0'].value.reshape(8,16,128,128)
                
#------------------------------------------------------------------------------
    
    def load_metadata(self):
        """
        
        """
        pass
    
#------------------------------------------------------------------------------
    
    @staticmethod
    def show_image(Arr, cmap = plasma, norm = LogNorm(), origin = 'lower', **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if kwargs['log']:
            del kwargs['log']
            ax.imshow(Arr, cmap = cmap, norm = norm, origin = origin, **kwargs)
        else:
            del kwargs['log']
            ax.imshow(Arr, cmap = cmap, origin = origin, **kwargs)
        ax.set_xlabel('horizontal detector range [pixel]')
        ax.set_ylabel('vertical detector range [pixel]')
    
#------------------------------------------------------------------------------
    
    def show_job2D(self, jobind, foil = 7, tc = 0, log = True):
        """
        
        """
# =============================================================================
#         if len(self.jobs) == 0:
#             self.load_jobs()
#             
#         if len(self.data_dict) == 0:
#             self.load_specificjobdata(jobind)
#             DataFrame_Base.show_image(self.data_dict[self.jobs[jobind]][foil, tc, :, :], log = log)
#             return None
# =============================================================================
        
        if self.jobs[jobind] in self.data_dict.keys():
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if log:
#                ax.imshow(self.data_dict[self.jobs[jobind]][foil, tc, :, :], cmap = plasma, norm = LogNorm(), origin = 'lower')
                ax.imshow(self.data_dict[self.jobs[jobind]][0, foil, tc, :, :], cmap = plasma, norm = LogNorm(), origin = 'lower')
            else:
#                ax.imshow(self.data_dict[self.jobs[jobind]][foil, tc, :, :], cmap = plasma, origin = 'lower')
                ax.imshow(self.data_dict[self.jobs[jobind]][0, foil, tc, :, :], cmap = plasma, origin = 'lower')
            ax.set_xlabel('Pixel')
            ax.set_ylabel('Pixel')
            return None

###############################################################################
###############################################################################
###############################################################################

class Mask_Base(object):
    """
    Mask Base class
    """
    def __init__(self, nn = 128):
        """
        Mask Base init function
        arguments:
                    nn(int):        mask array with dimension nnxnn
        """
        self.nn = nn
        # Create a basic value class eventually!
        self.d_SD = 2.25 # sample-detector-distance in meters
        self.pixelsize = 0.0015625 # dimension of a quadratic pixel of the CASCADE detector
        self.mask = np.zeros((self.nn, self.nn), dtype = np.float)
        self.masktype = 'Mask_Base'
        
    def __repr__(self):
        """
        official string description
        """
        return '{}x{} {}'.format(str(self.nn), str(self.nn), self.masktype)
    
    def getMask(self):
        """
        returns mask
        """
        return self.mask
    
    def shape(self):
        """
        returns mask shape
        """
        return self.mask.shape
    
    @staticmethod
    def combine_masks(pres, posts, non_bool = True):
        """
        mainly for visualization purpose
        pres and posts are Pre_mask or Post_sector_mask instances
        combines [pre1, pre2, ..., pren] and [post1, post2, ..., postm] to
        [[pre1 * post1, pre1 * post2 , ..., pre1 * postm], [..., pre2 * postm],...[..., pren * postm]]
        """
        comb_masks = []
        for pre in pres:
            line = []
            for post in posts:
                if non_bool:
                    line.append(pre.getMask() * post.getMask())
                else:
                    line.append(pre.getboolMask() * post.getMask())
            comb_masks.append(line)
        return comb_masks
    
    @staticmethod
    def show_mask(m_array, title = None):
        """
        
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(m_array, cmap = Greys, origin = 'lower')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Pixel')
        if title is not None: ax.set_title('{}'.format(title))
        return None


###############################################################################


class Pre_mask(Mask_Base):
    """
    Pre_masks Class for pre-grouping Cascade Detector pixel with quadratic tiles
    """
    def __init__(self, nn, tile_size):
        """
        constructor for pre_mask.
            arguments:
                        nn (int):                   see Mask_Base
                        tile_size (int):            dimension of quadratic tiles
        """
        super(Pre_mask, self).__init__(nn)
        self.masktype = 'Pregrouping mask'
        if nn % tile_size == 0:
            self.tile_size = tile_size
        else:
            print 'tile_size is not a divisor of nn! tile_size set to 1.'
            self.tile_size = 1
        self.create_pre_mask()
        
    def changetile_size(self, tile_size):
        """
        
        """
        if self.nn % tile_size == 0:
            self.tile_size = tile_size
        else:
            print 'tile_size is not a divisor of nn! tile_size set to 1.'
            self.tile_size = 1
            
    def create_pre_mask(self):
        """
        creates tiled pregrouping mask array
        """
        ratio = self.nn/self.tile_size
        for i in xrange(ratio):
            for j in xrange(ratio):
                self.mask[i*self.tile_size:(i + 1)*self.tile_size, j*self.tile_size:(j + 1)*self.tile_size] = i*ratio + j

    def shape(self, mod_ts = True):
        """
        show pre_mask dimensions mod tile_size
        """
        if mod_ts:
            return (self.nn/self.tile_size,)*2
        else:
            return super(self.__class__, self).shape()

    def show_pre_mask(self):
        """
        
        """
        temparr = np.where(self.mask %2 == 1, 1, -1)
        if (self.nn / self.tile_size) % 2 == 0:
            temparr = np.abs(temparr + temparr.T) - 1
        Mask_Base.show_mask(temparr, self.masktype)
        return None
        
    def getboolMask(self):
        """
        
        """
        temparr = np.where(self.mask %2 == 1, 1, -1)
        if (self.nn / self.tile_size) % 2 == 0:
            temparr = np.abs(temparr + temparr.T) - 1
        return temparr

###############################################################################

class Post_sector_mask(Mask_Base):
    """
    Post mask with circular or sector shape
    """
    
    def __init__(self, nn, centre, inner_radius, outer_radius, angle_range):
        """
        
        arguments:
                    angle_range (tuple):                (start_angle, stop_angle) in deg from [0,360)
        """
        super(Post_sector_mask, self).__init__(nn)
        self.masktype = 'Sector mask'
        self.centre = centre
        self.r_i = inner_radius
        self.r_o = outer_radius
        self.tmin, self.tmax = np.deg2rad(angle_range)
        self.create_post_mask()
        self.qxyz = np.zeros((self.nn, self.nn, 3))
        
#------------------------------------------------------------------------------
        
    def create_post_mask(self):
        x,y = np.ogrid[:self.nn,:self.nn]
        cx,cy = self.centre
        
        #ensure stop angle > start angle
        if self.tmax<self.tmin:
            self.tmax += 2*np.pi
        #convert cartesian --> polar coordinates
        r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
        theta = np.arctan2(x-cx,y-cy) - self.tmin
        #wrap angles between 0 and 2*pi
        theta %= (2*np.pi)
        #circular mask
        circmask = r2 <= self.r_o*self.r_o
        circmask2 = r2 >= self.r_i*self.r_i
        # angular mask
        anglemask = theta <= (self.tmax-self.tmin)

        self.mask = circmask*circmask2*anglemask

#------------------------------------------------------------------------------
        
    def every_q(self):
        """
        Calculates the qx, qy, qz value of a neutron arriving at a certain detector pixel,
        considering the center of the mask to be the direct beam spot at on the detector.
        
        """

        cx, cy = self.centre
        qq = (2*np.pi/6.0)

        for x in xrange(cx - (self.r_o + 1), cx + (self.r_o + 2)):
            for y in xrange(cy - (self.r_o + 1), cy + (self.r_o + 2)):
                n_path_length = np.sqrt(self.d_SD**2 + self.pixelsize**2*(x-cx)**2 + self.pixelsize**2*(y-cy)**2)
                try:
                    self.qxyz[y,x,0] = self.pixelsize*(x-cx)/n_path_length * qq
                    self.qxyz[y,x,1] = self.pixelsize*(y-cy)/n_path_length * qq
                    self.qxyz[y,x,2] = (self.d_SD/n_path_length - 1) * qq
                    
                except IndexError:
                    pass
#        for x in xrange(self.nn):
#            for y in xrange(self.nn):
#                n_path_length = np.sqrt(self.d_SD**2 + self.pixelsize**2*(x-cx)**2 + self.pixelsize**2*(y-cy)**2)
#                self.qxyz[y,x,0] = self.pixelsize*(x-cx)/n_path_length * qq
#                self.qxyz[y,x,1] = self.pixelsize*(y-cy)/n_path_length * qq
#                self.qxyz[y,x,2] = (self.d_SD/n_path_length - 1) * qq

#------------------------------------------------------------------------------

    def q(self, counter = 0):
        """
        Calculates the average |q| value of a sector mask.
        """
        
        while counter < 2:
#            q_abs = np.sqrt(np.sum(self.qxyz**2, axis = 2)) * self.mask / self.mask.sum()
            q_abs = np.sum(np.sqrt(np.sum(self.qxyz**2, axis = 2)) * self.mask) / self.mask.sum()
            q_abs_err = np.sqrt(1.0/(self.mask.sum() - 1) * np.sum(((np.sqrt(np.sum(self.qxyz**2, axis = 2)) - q_abs) * self.mask)**2))
            if q_abs.any() != 0:
                return q_abs, q_abs_err
            else:
                self.every_q()
                self.q(counter + 1)
        
#------------------------------------------------------------------------------
        
    def show_post_mask(self):
        """
        
        """
        Mask_Base.show_mask(np.where(self.mask == True, 1, 0), self.masktype)
        return None

###############################################################################

class Post_square_mask(Mask_Base):
    """
    Post mask with rectangular shape(s)
    """

    def __init__(self, nn, llbh, *args):
        """
        
        arguments:
                    llbh (tuple):               (left, length, bottom, height) in pixels
                    args (tuple):               for more squares in one map args = (left2, length2, bottom2, height2, left3, ...)
        """
        super(Post_square_mask, self).__init__(nn)
        self.masktype = 'Square mask'
        self.lefts, self.lengths, self.bottoms, self.heights = [[val] for val in llbh]
        if len(args) % 4 == 0 and len(args) != 0:
            for i, el in enumerate(args):
                if i % 4 == 2:
                    self.lefts.append(el)
                elif i % 4 == 3:
                    self.lengths.append(el)
                elif i % 4 == 0:
                    self.bottoms.append(el)
                elif i % 4 == 1:
                    self.heights.append(el)
                
# =============================================================================
#                 # consistency check
#             if len(self.lefts) == len(self.lengths) and len(self.bottoms) == len(self.heights) and len(self.lefts) == len(self.bottoms):
#                 pass
#             else:
#                 raise AttributeError
# =============================================================================
        
        self.mask = self.mask.astype(np.bool)
        for llbhval in xrange(len(self.lefts)):
            self.mask[self.lefts[llbhval]:self.lefts[llbhval] + self.lengths[llbhval], self.bottoms[llbhval]:self.bottoms[llbhval] + self.heights[llbhval]] = True
        
#------------------------------------------------------------------------------

    

###############################################################################
###############################################################################
###############################################################################
        
#class ContrastFit(DataFrame_Base, Post_sector_mask, Pre_mask):
class ContrastFit(DataFrame_Base):
    """
    Sinus fits to grouped Data sets
    """
    
    def __init__(self, dfile, drootpath = ''):
        super(ContrastFit, self).__init__(dfile, drootpath)
#        self.load_jobs()
        self.load_alljobsdata()
        # obsolete since 'load_alljobsdata' gets the monitor counts
#        self.__load_monitor_counts()
        self.maskdict = {'pre_masks' : {}, 'post_masks' : {}}
#        self.masktype = 'Ambiguous'
        self.local_memory = {}      # memory for intermediate results

#------------------------------------------------------------------------------

    def dump_to_memory(self, key, item):
        """
        stores item in 'local_memory'
        """
        self.local_memory.update({key : item})
        return None

#------------------------------------------------------------------------------
    
    def get_from_memory(self, key):
        """
        returns value from self.local_memory[key]
        """
        return self.local_memory[key]
    
#------------------------------------------------------------------------------
        
    def remove_from_memory(self, key):
        """
        removes item with key from memory
        """
        del self.local_memory[key]
        return None
        
#------------------------------------------------------------------------------

    def update_maskdict(self, mask, key):
        """
        if key == 'pre_masks':
            tempdict = dict((('nn', self.nn), ('tile_size', self.tile_size),\
                             ('mask', self.mask)))
            self.maskdict['pre_masks'].update({str(len(self.maskdict['pre_masks'])) : tempdict})
        elif key == 'post_masks':
            tempdict = dict((('nn', self.nn), ('centre', self.centre),\
                             ('r_i', self.r_i), ('r_o', self.r_o),\
                             ('angles', (self.tmin, self.tmax)),\
                             ('mask', self.mask)))
            self.maskdict['post_masks'].update({str(len(self.maskdict['post_masks'])) : tempdict})
        """
        if key == 'pre_masks':
            self.maskdict['pre_masks'].update({len(self.maskdict['pre_masks']) : mask})
            
        elif key == 'post_masks':
            self.maskdict['post_masks'].update({len(self.maskdict['post_masks']) : mask})
            
#------------------------------------------------------------------------------
    
    def initialize_pre_mask(self, nn, tile_size):
        """
        adds a pre-grouping mask instance to maskdict
        """
        
        self.update_maskdict(Pre_mask(nn, tile_size), 'pre_masks')
            
#------------------------------------------------------------------------------
        
    def initialize_post_sector_mask(self, nn, centre, inner_radius, outer_radius, angle_range):
        """
        add a post-grouping mask instance to maskdict
        """
        
        self.update_maskdict(Post_sector_mask(nn, centre, inner_radius, outer_radius, angle_range), 'post_masks')
            
#------------------------------------------------------------------------------
        
    def initialize_post_square_mask(self, nn, llbh, *args):
        """
        add a post-grouping mask instance to maskdict
        """
        
        self.update_maskdict(Post_square_mask(nn, llbh, *args), 'post_masks')
            
#------------------------------------------------------------------------------
    
    @staticmethod
    def _contract_data(pre_mask, data_set):
        """
        assuming that input is pre_mask instance
        """
        tiles_per_row = pre_mask.nn/pre_mask.tile_size
        temp = np.zeros(tiles_per_row*tiles_per_row)
        for i in xrange(tiles_per_row*tiles_per_row):
            mask_tile = np.where(pre_mask.mask == i, 1., 0.)
            temp[i] = np.nansum(mask_tile*data_set)
            
        return temp.reshape((tiles_per_row, tiles_per_row))
    
#------------------------------------------------------------------------------

    def contract_data(self, mask_key, jobind, foil = (7,), tc = (0,), dump = False):
        """
        more easy for CF-object
        """
        shape = (len(foil), len(tc), self.maskdict['pre_masks'][mask_key].nn/self.maskdict['pre_masks'][mask_key].tile_size, self.maskdict['pre_masks'][mask_key].nn/self.maskdict['pre_masks'][mask_key].tile_size)
        temp_contr = np.zeros(shape)
        temp_contr_err = np.zeros(shape)
        for find, f in enumerate(foil):
            for tind, t in enumerate(tc):
                temp_contr[find, tind] = ContrastFit._contract_data(self.maskdict['pre_masks'][mask_key],\
                                              self.data_dict[self.jobs[jobind]][0, f, t, :, :])
                temp_contr_err[find, tind] = ContrastFit._contract_data(self.maskdict['pre_masks'][mask_key],\
                                              self.data_dict[self.jobs[jobind]][1, f, t, :, :])
        # line above from real calcumlation dR = sum(dr_i) where dr_i are all the error summed up equivalent to R = sum(r)
#        temp_contr_err = np.sqrt(temp_contr)
            
        if dump:            
            try:
                if dump != True: self.dump_to_memory(dump, np.array([temp_contr, temp_contr_err]))
            except KeyError:
                print "No valid key was passed in 'dump'!"
            finally:
                return np.array([temp_contr, temp_contr_err])
        else:
            return np.array([temp_contr, temp_contr_err])
        
# =============================================================================
#         if norm_mon:
#             return ContrastFit._contract_data(self.maskdict['pre_masks'][mask_key],\
#                                               self.data_dict[self.jobs[jobind]][foil, tc, :, :]\
#                                               /self.monitor[self.jobs[jobind]])
#         else:
#             return ContrastFit._contract_data(self.maskdict['pre_masks'][mask_key],\
#                                               self.data_dict[self.jobs[jobind]][foil, tc, :, :])
# =============================================================================
            
#------------------------------------------------------------------------------

    @staticmethod
    def _expand_data(pre_mask, data_set):
        """
        assuming that input is pre_mask instance which was used for prior contraction
        """
        
        tile_size = pre_mask.tile_size
        temp = np.zeros((pre_mask.nn,)*2)
        for i, row in enumerate(data_set):
            for j, el in enumerate(row):
                temp[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = el
            
        return temp
        
#------------------------------------------------------------------------------

    def expand_data(self, mask_key, memory_keys = (), dump_again = True):
        """
        assuming that input is something stored in the local memory which was processed
        """
        
        if len(memory_keys) != 0:
            expanded_data = []
            for mkey in memory_keys:
                temp = np.zeros(self.get_from_memory(mkey).shape)
                for find, f in enumerate(temp[0,:]):
                    for tind, t in f:
                        # make easier by using t-variable!
                        temp[0, find, tind] = self._expand_data(self.maskdict['pre_masks'][mask_key][0, find, tind], self.get_from_memory(mkey))
                        temp[1, find, tind] = self._expand_data(self.maskdict['pre_masks'][mask_key][1, find, tind], self.get_from_memory(mkey))
                expanded_data.append(temp)
                if dump_again:
                    dump_key = 'exp_{}'.format(mkey)
                    self.dump_to_memory(dump_key, temp)
            return expanded_data
        else:
            return None
        
#------------------------------------------------------------------------------     
    
    def def_ROI(self):
        """
        
        """
        pass

#------------------------------------------------------------------------------
    
    def apply_pre_mask(self, pre_key, jobind, tc, foil = 7, contract = True):
        """
        applies for one time bin
        """
        
        mask = self.maskdict['pre_masks'][pre_key]
        raw_data = self.data_dict[self.jobs[jobind]][:, foil, tc]
        if contract:
            return ContrastFit._contract_data(mask, raw_data)
# =============================================================================
#         mask = self.maskdict['pre_masks'][pre_key]
#         raw_data = self.data_dict[self.jobs[jobind]][foil, tc]
#         if contract:
#             return ContrastFit.contract_data(mask, raw_data)
#         else:
#             return self.expand_data(mask, self.contract_data(mask, raw_data))
# =============================================================================
            
#------------------------------------------------------------------------------

    def apply_post_mask(self, pre_key, post_key, jobind, tc, foil = 7, contracted = True):
        """
        
        """
        if contracted:
            return self.maskdict['post_masks'][post_key].mask

#------------------------------------------------------------------------------
        
    @staticmethod
    def single_sinus_fit(tc_data, eps_tc_data, plot = False):
        """
        Filter out "nan" values more sophisticated than np.ma.masked_equal() ...
        """
        offset_est = np.mean(tc_data)
#        omega_est = np.pi/2. / np.abs(np.argmax(tc_data) - np.argmin(np.abs(tc_data-offset_est))) #max to zero
#        omega_est = np.pi / np.abs(np.argmin(tc_data) - np.argmax(tc_data))
        dphi_1 = tc_data[1]-tc_data[0]
        dphi_2 = tc_data[2]-tc_data[0]
        
        params = Parameters()
        params.add('offset', value=offset_est, min = 0.)
#        params.add('omega', value=omega_est, min=0, max=np.pi/4.)      # is a fixed parameter!!!
        params.add('omega', value=np.pi/8.,vary = False)
        
        params.add('pol_bound', value = 0.5, min = 0., max = 1., vary = True)
        params.add('amp', value=(max(tc_data)-min(tc_data))/2., min = 0., expr = 'pol_bound*offset')
#        params.add('amp', value=(max(tc_data)-min(tc_data))/2., min = 0.)

#        params.add('phase', value=0, min = 0, max = 2.*np.pi)
        
        if tc_data[0] > params['offset'] and dphi_1 > 0. and dphi_2 > 0.:
            params.add('phase', value = np.pi/4., min = -np.pi/4., max = 3.*np.pi/4.)
            
        elif tc_data[0] > params['offset'] and dphi_1 < 0. and dphi_2 < 0.:
            params.add('phase', value = 3*np.pi/4., min = np.pi/4., max = 5.*np.pi/4.)
            
        elif tc_data[0] < params['offset'] and dphi_1 < 0. and dphi_2 < 0.:
            params.add('phase', value = 5*np.pi/4., min = 3./4.*np.pi, max = 7./4.*np.pi)
            
        elif tc_data[0] < params['offset'] and dphi_1 > 0. and dphi_2 > 0.:
            params.add('phase', value = 7*np.pi/4., min = 5./4.*np.pi, max = 9.*np.pi/4.)
            
        elif tc_data[0] > params['offset'] and dphi_2 > 0.:
            params.add('phase', value = np.pi/4., min = -np.pi/4., max = 3*np.pi/4.)
            
        elif tc_data[0] > params['offset'] and dphi_2 < 0.:
            params.add('phase', value = 3*np.pi/4., min = np.pi/4., max = 5*np.pi/4.)
            
        elif tc_data[0] < params['offset'] and dphi_2 < 0.:
            params.add('phase', value = 5*np.pi/4., min = 3*np.pi/4., max = 7*np.pi/4.)
            
        elif tc_data[0] < params['offset'] and dphi_2 > 0.:
            params.add('phase', value = 7*np.pi/4., min = 5*np.pi/4., max = 9*np.pi/4.)
            
        else:
            params.add('phase', value = -np.pi/4., min = -np.pi, max = np.pi)
#            print 'Still Possible?!'

#        out = params
#        out = minimize(resid_sin, params, args = (np.arange(16.), tc_data, eps_tc_data))
        out = minimize(resid_sin, params, args = (np.arange(16.), np.ma.masked_equal(tc_data, 0.0), np.ma.masked_equal(eps_tc_data, 0.0)))
        # add chi^2 calculation
        
        if plot:
            fig = plt.figure()
            plt.errorbar(range(16), tc_data, eps_tc_data, ls='None', marker = 'o', mfc = 'steelblue', mec = 'steelblue', ecolor = 'steelblue', label = 'data')
            plt.plot(np.linspace(0.,15.), sinus(np.linspace(0.,15.), out.params['amp'], out.params['omega'], out.params['phase'], out.params['offset'], False), color = 'maroon')
            params.pretty_print()
            out.params.pretty_print()
            print out.chisqr
        
        return out
        
#------------------------------------------------------------------------------
    
    def polarization_analysis(self, jobind, foil = (7,), pre_mask_key = False, to_memory = False, mon_norm = True):
        """
        fits pol
        arguments:
                    jobind[int]:                index of job in self.jobs
                    foil[tuple]:                index/indecies of foils used for analysis
                    pre_mask[str, bool]:        False, or pre_mask key in maskdict
                    to_memory[bool]:            should be dumped to self.local_memory
                    mon_norm[bool]:             should be normed to mon counts
        """
        if pre_mask_key:
            try:
                contracted_data = np.zeros(np.concatenate(((len(foil), 16), self.maskdict['pre_masks'][pre_mask_key].shape())))
                fitted_data = np.zeros(np.concatenate(((len(foil),), self.maskdict['pre_masks'][pre_mask_key].shape())))
            
            except TypeError:
                contracted_data = np.zeros(np.concatenate(((1, 16), self.maskdict['pre_masks'][pre_mask_key].shape())))
                fitted_data = np.zeros(np.concatenate(((len(foil),), self.maskdict['pre_masks'][pre_mask_key].shape())))

            except AttributeError:
                print 'contracted_data array could not be initialized. Return None!'
                return None
            
            for ind, foil_ind in enumerate(foil):
                for tc in xrange(16):
                    contracted_data[ind, tc] = self.apply_pre_mask(pre_mask_key, jobind, tc, foil_ind)

        else:
            try:
                contracted_data = np.array([self.data_dict[self.jobs[jobind]][list(foil)]])
                fitted_data = np.zeros((len(foil), 128, 128))

            except KeyError:
                print 'No data contraction. Could not initialize usable data from data_dict. Return None!'
                return None
            
            except:
                print 'Sth went wrong with data_contraction in polatization_analysis! Return None!'
                return None
       
        # mask contracted data
        contracted_data = np.ma.masked_less_equal(contracted_data, 0.)
#        return contracted_data
        # norm contracted data
        # proper error determination
        
        for i in xrange(len(fitted_data)):
            for j in xrange(len(fitted_data[i])):
                for k in xrange(len(fitted_data[i,j])):
                    out = self.single_sinus_fit(contracted_data[i,:,j,k], np.sqrt(contracted_data[i,:,j,k]))
                    fitted_data[i,j,k] = out.params['amp'].value / out.params['offset'].value
                    if fitted_data[i,j,k] < 0. or fitted_data[i,j,k] > 1.:
                        print i,j,k
                        out.params.pretty_print()
        
#        for i, tc_panel in enumerate(contracted_data):
#            for j in xrange(len(fitted_data[i])):
#                for k in xrange(len(fitted_data[i,j])):
#                    out = self.single_sinus_fit(contracted_data[i,:,j,k], np.sqrt(contracted_data[i,:,j,k]))
#                    fitted_data[i,j,k] = out.params['amp'].value / out.params['offset'].value
                    
        # add dump to self.local_memory
        
        return np.ma.masked_greater_equal(fitted_data, 1.)
         
#------------------------------------------------------------------------------
        
    @staticmethod
    def _analysis(sine_data_arr):
        """
        fits a sine to an array of (#dat_err, #foils, #tc, #pixel, #pixel)
        """
        
        temp = np.zeros((2, sine_data_arr.shape[1], sine_data_arr.shape[-2], sine_data_arr.shape[-1], 5))
        
        for find, f in enumerate(sine_data_arr[0]):
            for iind, line in enumerate(f[0]):
                for jind in xrange(len(line)):
                    out = ContrastFit.single_sinus_fit(sine_data_arr[0, find, :, iind, jind], sine_data_arr[0, find, :, iind, jind])
                    temp[0, find, iind, jind] = np.array([val.value for val in out.params.values()])
                    temp[1, find, iind, jind] = np.array([val.stderr for val in out.params.values()])
        return temp
                

        
        
#------------------------------------------------------------------------------
    
    def analysis(self, jobind, foil = (7,), pre_mask_key = False, output = 'pol_bound', to_memory = False, mon_norm = True):
        """
        fits pol
        arguments:
                    jobind[int]:                index of job in self.jobs
                    foil[tuple]:                index/indecies of foils used for analysis
                    pre_mask[str, bool]:        False, or pre_mask key in maskdict
                    to_memory[bool]:            should be dumped to self.local_memory
                    mon_norm[bool]:             should be normed to mon counts
        """
        if pre_mask_key:
            try:
                contracted_data = np.zeros(np.concatenate(((len(foil), 16), self.maskdict['pre_masks'][pre_mask_key].shape())))
                fitted_data = np.zeros(np.concatenate(((len(foil),), self.maskdict['pre_masks'][pre_mask_key].shape())))
            
            except TypeError:
                contracted_data = np.zeros(np.concatenate(((1, 16), self.maskdict['pre_masks'][pre_mask_key].shape())))
                fitted_data = np.zeros(np.concatenate(((len(foil),), self.maskdict['pre_masks'][pre_mask_key].shape())))

            except AttributeError:
                print 'contracted_data array could not be initialized. Return None!'
                return None
            
            for ind, foil_ind in enumerate(foil):
                for tc in xrange(16):
                    contracted_data[ind, tc] = self.apply_pre_mask(pre_mask_key, jobind, tc, foil_ind)

        else:
            try:
                contracted_data = np.array([self.data_dict[self.jobs[jobind]][list(foil)]])
                fitted_data = np.zeros((len(foil), 128, 128))

            except KeyError:
                print 'No data contraction. Could not initialize usable data from data_dict. Return None!'
                return None
            
            except:
                print 'Sth went wrong with data_contraction in polatization_analysis! Return None!'
                return None
       
        # mask contracted data
        contracted_data = np.ma.masked_less_equal(contracted_data, 0.)
        
        for i in xrange(len(fitted_data)):
            for j in xrange(len(fitted_data[i])):
                for k in xrange(len(fitted_data[i,j])):
                    # Here occures fitting of the sine
                    out = self.single_sinus_fit(contracted_data[i,:,j,k], np.sqrt(contracted_data[i,:,j,k]))
                    if output == 'pol_bound':
                        fitted_data[i,j,k] = out.params['pol_bound'].value
                    elif output == 'amp':
                        fitted_data[i,j,k] = out.params['amp'].value
                    elif output == 'offset':
                        fitted_data[i,j,k] = out.params['offset'].value
                    elif output == 'phase':
                        fitted_data[i,j,k] = out.params['phase'].value
        
        if output == 'pol_bound':
            return np.ma.masked_greater_equal(fitted_data, 1.)
        elif output == 'phase':
            return fitted_data % (2*np.pi) / np.pi
        else:
            return fitted_data
#------------------------------------------------------------------------------
        
    @staticmethod
    def normalization_count_error_propagation(quantity, floatargs, arrayargs):
        """
        !! WRONG !!
        counting error as args_error = sqrt(c)
        """
        return quantity * np.sqrt(np.sum([1./fa for fa in floatargs]) + np.sum([1./aa for aa in arrayargs], axis  = 0))

# =============================================================================
# #------------------------------------------------------------------------------
#     
#     @staticmethod
#     def basic_norm_arr(a1, a2):
#         return 
# =============================================================================

