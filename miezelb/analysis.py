#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Data Frame class

"""

###############################################################################
####################        IMPORTS        ####################################
###############################################################################

import matplotlib.pyplot as plt
from matplotlib.cm import plasma
from matplotlib.colors import LogNorm

from masks import Pre_mask, Post_sector_mask, Post_square_mask
import utils

from h5py import File
from os import path
from scipy.constants import h, m_n
from numpy import arange, array, concatenate, fromfile, int32, linspace, ma, mean, nansum, where, sqrt, zeros
from lmfit import Parameters, minimize
from math import pi



###############################################################################
###############################################################################

class DataFrame_NICOS():
    """
    Base class providing meta and experimental data of the NICOS .tof files
    of the MIRA and RESEDA instruments.
    """
    
    def __init__(self, fname, drootpath = './data/cascade'):
        """
        Constructor for the MIRA and RESEDA DataFrame based on the NICOS .tof output
        --------------------------------------------------
        
        Arguments:
        ----------
        fname       : str               : name of the data file (*.tof)
        drootpath   : str               : path to the data directory, default './data/CASCADE'
        
        Return:
        ----------
        obj         : DataFrame_NICOS   : DataFrame_NICOS object associated with the given file.
                                          Makes metadata and raw CASCADE output accessible with python
        """
        
# =============================================================================
#         # declare intance variables
#         self._metadata = {}
#         self._cascadedata = {}
# =============================================================================
        
        # instantiate variables
        self.fpath = path.join(drootpath, fname)
        self._metadata = self.__extract_metadata()
        self._cascadedata = self.__extract_cascadedata()

#------------------------------------------------------------------------------
        
    @classmethod
    def init_from_nums(cls, fnum, expnum = 0, drootpath = "./data/CASCADE"):
        """
        Alternative constructor from experiment- and filenumber
        --------------------------------------------------
        
        Arguments:
        ----------
        expnum      : int               : number of the experiment/proposal
        fnum        : int               : number of the file
        drootpath   : str               : path to the data directory, default './data/CASCADE'
        
        Return:
        ----------
        obj         : DataFrame_NICOS   : DataFrame_NICOS object associated with the given file.
                                          Makes metadata and raw CASCADE output accessible with python
        """

        return cls(utils.gen_filename(expnum, fnum), drootpath)

#------------------------------------------------------------------------------
        
    def __extract_metadata(self):
        """
        extracts the metadata from NICOS .tof - file footers and stores it in self._metadata
        --------------------------------------------------
        
        Arguments:
        ----------
        None
        
        Return:
        ----------
        metadata    : dict  : contains the most important metadata as returned by NICOS
        """
        
        metadata = {'info' : {},
                    'TAS' : {},
                    'sample_table' : {},
                    'MIEZE_setup' : {},
                    'coils' : {},
                    'fast_results' : {}}
    
        keys = {'info' : ['Exp_Remark', 'timer', 'mon2'],
                'TAS' : ['Ef_value', 'Ei_value', 'ana_value', 'ath_value', 'att_value', 'ki_value', 'kf_value', 'lam_value', 'sth_value', 'stt_value'],
                'sample_table' : ['stx_value', 'sty_value', 'stz_value'],
                'MIEZE_setup' : ['echotime_value', 'cbox1_fg_freq_value', 'cbox2_fg_freq_value', 'psd_chop_freq_value', 'psd_timebin_freq_value'],
                'coils' : ['hrf1_value', 'hrf2_value', 'hsf1_value', 'hsf2_value', 'sf1_value', 'sf2_value'],
                'fast_results' : ['psd_channel.roi', 'psd_channel.total', 'fit.contrast', 'fit.contrastErr', 'roi.contrast', 'roi.contrastErr']}
    
        aliases = {'info' : ['Remark', 'time', 'monitor'],
                   'TAS' : ['Ef', 'Ei', 'analysator_wavevector', 'ath', 'att', 'ki', 'kf', 'lambda', 'sth', 'stt'],
                   'sample_table' : ['stx', 'sty', 'stz'],
                   'MIEZE_setup' : ['tau_mieze', 'freq_A', 'freq_B', 'chop_freq', 'timebin_freq'],
                   'coils' : ['B0_A', 'B0_B', 'BHelmholtz_A', 'BHemlholtz_B', 'Bflip_A', 'Bflip_B'],
                   'fast_results' : ['roi_counts', 'total_counts', 'fit_contrast', 'fit_contrastErr', 'roi_contrast', 'roi_contrastErr']}
    
        with open(self.fpath, 'r') as f:
            fi = f.readlines()
            fi.reverse()
            for i, line in enumerate(fi):
                splitted_line = line.split()
                for ky, kys in keys.items():
                    try:
                        if splitted_line[0] in kys and splitted_line[2] != splitted_line[-1]:
                            temp = (float(splitted_line[2]), splitted_line[-1])
                            metadata[ky][aliases[ky][keys[ky].index(splitted_line[0])]] = temp
                        else:
                            metadata[ky][aliases[ky][keys[ky].index(splitted_line[0])]] = splitted_line[2]
                    except:
                        pass
                if i > len(fi)/4: # shorter runtime^^
                    break
        return metadata

#------------------------------------------------------------------------------
        
    def __extract_cascadedata(self):
        """
        extracts the raw CASCADE-detector data from NICOS .tof - files and stores it in self._cascadedata
        --------------------------------------------------
        
        Arguments:
        ----------
        None
        
        Return:
        ----------
        cascadedata     : ndarray   : returns a numpy.ndarray object of .shape = (8, 16, 128, 128) and dtype = np.int32
                                      data format is (foil, timebins, ypixel, xpixel)
        
        INFO:
        -----
        Check whether it actually is ..., ypixel, xpixel) array
        """
        
        return fromfile(self.fpath, dtype = int32)[:128*128*16*8].reshape(8,16,128,128)

#------------------------------------------------------------------------------

    def get_metadata(self, *keys):
        """
        Returns the metadata specified by keys. Can be either 'mainkey' or 'subkey'.
        Returns the entire dictionary if no key is given.
        --------------------------------------------------
        
        Arguments:
        ----------
        *keys       : list  : list of keys for metadata retrieval
        
        Returns:
        ----------
        seldict     : dict  : dictionary containing metadata {'mainkey' : subdict, 'subkey' : item ,...}
        """

        seldict = {}
        if keys:
            for key in keys:
                assert isinstance(key, str)
                if key in self._metadata.keys():
                    seldict.update({key : self._metadata[key]})
                else:
                    for subdict in self._metadata.itervalues():
                        if key in subdict.keys():
                            seldict.update({key : subdict[key]})
            return seldict

        else:
            return self._metadata

#------------------------------------------------------------------------------

    def show_2D_panel(self, foil_index = 0, timebin_index = 0, **kwargs):
        """
        Fast visualization tool for 2D cascade data by specifing foil_index and timebin_index
        --------------------------------------------------
        
        Arguments:
        ----------
        foil_index      : int   : selects foil of the cascade data set
        timebin_index   : int   : selects timebin of the cascade data set
        
        Returns:
        ----------
        None
        """
        
        try:
            self.show_image(self._cascadedata[foil_index, timebin_index], **kwargs)
        except IndexError:
            print("Probably foil_index = {} and timebin_index = {} are out of range with data array shape being {}".format(foil_index, timebin_index, self._cascadedata.shape[:2]))

#------------------------------------------------------------------------------

    @staticmethod
    def show_image(Arr, cmap = plasma, norm = LogNorm(), origin = 'lower', **kwargs):
        """
        Fast visualization tool for 2D data if available as an array.
        --------------------------------------------------
        
        Arguments:
        ----------
        Arr         : ndarray   : 2D data to be visualized
        **kwargs    : dict      : **kwargs will be passed to plt.imshow function
        
        Return:
        ----------
        None
        
        INFO:
        ----------
        
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(Arr, cmap = cmap, norm = norm, origin = origin, **kwargs)
        ax.set_xlabel('horizontal detector range [pixel]')
        ax.set_ylabel('vertical detector range [pixel]')


###############################################################################
###############################################################################

class AnalysisFrame_Base(DataFrame_NICOS):
    """
    AnalysisFrame class to apply individual analysis tools to the "DataFrame_NICOS"' raw data
    Supports a 'local_memory' dictionary to store intermediate results
    """
    
    def __init__(self, fname, drootpath = "./data/cascade", instrument = "MIRA"):
        """
        Constructor of an 'AnalysisFrame_Base' instance
        --------------------------------------------------
        
        Arguments:
        ----------
        fname       : str                   : name of the data file (*.tof)
        drootpath   : str                   : path to the data directory, default './data/CASCADE'
        instrument  : str                   : name of the instrument "MIRA", "RESEDA", "RESEDAlegacy" to use correct loading routine
        
        Return:
        ----------
        obj         : AnalysisFrame_NICOS   : AnalysisFrame_NICOS object associated with the given file.
                                              Supports functionality for the MIEZE analysis and applies it to the loaded data
        
        INFO:
        ----------
        NEED to implement loader/DataFrame for "RESEDA" and "RESEDAlegacy" options
        """
        
        if instrument == "MIRA":
            DataFrame_NICOS.__init__(self, fname, drootpath)
        
        elif instrument == "RESEDA":
            print("Loading RESEDA-NICOS data is not supported, yet.")
        
        elif instrument == "RESEDAlegacy":
            print("Loading RESEDAlegacy-IGOR data is not supported, yet.")
        
        else:
            print("No valid instrument has been specified, no data has been loaded. Options are: 'MIRA', 'RESEDA', 'RESEDAlegacy'")
        
        self._local_memory = {}
        self._maskdict = {}

#------------------------------------------------------------------------------

    def dump_to_memory(self, key, value):
        """
        stores value in '_local_memory'
        supposed to reduce re-computing intermediate results for further processing
        --------------------------------------------------
        
        Arguments:
        ----------
        key     : str       : key to label value in _local_memory
        value   : anything  : value to be stored in '_local_memory'
        
        Returns:
        ----------
        None
        
        INFO:
        ----------
        A consistent naming convention should be developed to indicate the changes and analysis steps performed
        in comparison to prioir steps, or the raw data.
        """
        
        self._local_memory.update({key : value})
        return None

#------------------------------------------------------------------------------
    
    def get_from_memory(self, key):
        """
        returns value from 'self.local_memory[key]'
        --------------------------------------------------
        
        Arguments:
        ----------
        key     : str   : key to label value in _local_memory
        
        Returns:
        ----------
        self._local_memory[key] : anything  : value to be stored in '_local_memory'
        """
        
        return self._local_memory[key]
    
#------------------------------------------------------------------------------
        
    def remove_from_memory(self, key):
        """
        removes item with key from memory
        
        Arguments:
        ----------
        key     : str   : key to label value in _local_memory
        
        Returns:
        ----------
        None
        """
        
        del self._local_memory[key]
        return None

#------------------------------------------------------------------------------


###############################################################################
###############################################################################

class DataFrame_Base(object):
    """
    Base class providing meta and experimental data of the IGOR hdf5 files
    of the RESEDA experiment
    """

    def __init__(self, dfile, drootpath = ''):
        self.fpath = path.join(drootpath, dfile)
        self.rawhdfdata = File(self.fpath, 'r')
#       No reason not to load jobs immediately --> oh yeah there are dumbass!
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
                            key = lambda s: int(s.split('_')[-1]))

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
            
        tau = 2 * deltaFreq * self.d_SD * 2*m_n**2 * (self.wavelength * 1.0e-10)**3 / h**2 * 1.0e9
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
        self.data_dict[key] = zeros((2,8,16,128,128))
        try:
            self.monitor[key] = self.rawhdfdata[key+'/ew_Counts_0'].attrs['ev_Monitor']
            self.mieze_taus[key] = self._calc_miezetau(job = key)
            self.data_dict[key][0] = self.rawhdfdata[key]['ew_MDMiezeCounts_0'].value.reshape(8,16,128,128)
            self.data_dict[key][1] = sqrt(self.rawhdfdata[key]['ew_MDMiezeCounts_0'].value.reshape(8,16,128,128))
            if 'norm_mon' in kwargs:
                if kwargs['norm_mon']:
                    self.data_dict[key][1] = sqrt(1./self.data_dict[key][0] + 1./self.monitor[key])
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
        temp = zeros(tiles_per_row*tiles_per_row)
        for i in xrange(tiles_per_row*tiles_per_row):
            mask_tile = where(pre_mask.mask == i, 1., 0.)
            temp[i] = nansum(mask_tile*data_set)
            
        return temp.reshape((tiles_per_row, tiles_per_row))
    
#------------------------------------------------------------------------------

    def contract_data(self, mask_key, jobind, foil = (7,), tc = (0,), dump = False):
        """
        more easy for CF-object
        """
        shape = (len(foil), len(tc), self.maskdict['pre_masks'][mask_key].nn/self.maskdict['pre_masks'][mask_key].tile_size, self.maskdict['pre_masks'][mask_key].nn/self.maskdict['pre_masks'][mask_key].tile_size)
        temp_contr = zeros(shape)
        temp_contr_err = zeros(shape)
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
                if dump != True: self.dump_to_memory(dump, array([temp_contr, temp_contr_err]))
            except KeyError:
                print "No valid key was passed in 'dump'!"
            finally:
                return array([temp_contr, temp_contr_err])
        else:
            return array([temp_contr, temp_contr_err])
        
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
        temp = zeros((pre_mask.nn,)*2)
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
                temp = zeros(self.get_from_memory(mkey).shape)
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
        offset_est = mean(tc_data)
#        omega_est = np.pi/2. / np.abs(np.argmax(tc_data) - np.argmin(np.abs(tc_data-offset_est))) #max to zero
#        omega_est = np.pi / np.abs(np.argmin(tc_data) - np.argmax(tc_data))
        dphi_1 = tc_data[1]-tc_data[0]
        dphi_2 = tc_data[2]-tc_data[0]
        
        params = Parameters()
        params.add('offset', value=offset_est, min = 0.)
#        params.add('omega', value=omega_est, min=0, max=np.pi/4.)      # is a fixed parameter!!!
        params.add('omega', value=pi/8.,vary = False)
        
        params.add('pol_bound', value = 0.5, min = 0., max = 1., vary = True)
        params.add('amp', value=(max(tc_data)-min(tc_data))/2., min = 0., expr = 'pol_bound*offset')
#        params.add('amp', value=(max(tc_data)-min(tc_data))/2., min = 0.)

#        params.add('phase', value=0, min = 0, max = 2.*np.pi)
        
        if tc_data[0] > params['offset'] and dphi_1 > 0. and dphi_2 > 0.:
            params.add('phase', value = pi/4., min = -pi/4., max = 3.*pi/4.)
            
        elif tc_data[0] > params['offset'] and dphi_1 < 0. and dphi_2 < 0.:
            params.add('phase', value = 3*pi/4., min = pi/4., max = 5.*pi/4.)
            
        elif tc_data[0] < params['offset'] and dphi_1 < 0. and dphi_2 < 0.:
            params.add('phase', value = 5*pi/4., min = 3./4.*pi, max = 7./4.*pi)
            
        elif tc_data[0] < params['offset'] and dphi_1 > 0. and dphi_2 > 0.:
            params.add('phase', value = 7*pi/4., min = 5./4.*pi, max = 9.*pi/4.)
            
        elif tc_data[0] > params['offset'] and dphi_2 > 0.:
            params.add('phase', value = pi/4., min = -pi/4., max = 3*pi/4.)
            
        elif tc_data[0] > params['offset'] and dphi_2 < 0.:
            params.add('phase', value = 3*pi/4., min = pi/4., max = 5*pi/4.)
            
        elif tc_data[0] < params['offset'] and dphi_2 < 0.:
            params.add('phase', value = 5*pi/4., min = 3*pi/4., max = 7*pi/4.)
            
        elif tc_data[0] < params['offset'] and dphi_2 > 0.:
            params.add('phase', value = 7*pi/4., min = 5*pi/4., max = 9*pi/4.)
            
        else:
            params.add('phase', value = -pi/4., min = -pi, max = pi)
#            print 'Still Possible?!'

#        out = params
#        out = minimize(resid_sin, params, args = (np.arange(16.), tc_data, eps_tc_data))
        out = minimize(utils.resid_sin, params, args = (arange(16.), ma.masked_equal(tc_data, 0.0), ma.masked_equal(eps_tc_data, 0.0)))
        # add chi^2 calculation
        
        if plot:
            fig = plt.figure()
            plt.errorbar(range(16), tc_data, eps_tc_data, ls='None', marker = 'o', mfc = 'steelblue', mec = 'steelblue', ecolor = 'steelblue', label = 'data')
            plt.plot(linspace(0.,15.), utils.sinus(linspace(0.,15.), out.params['amp'], out.params['omega'], out.params['phase'], out.params['offset'], False), color = 'maroon')
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
                contracted_data = zeros(concatenate(((len(foil), 16), self.maskdict['pre_masks'][pre_mask_key].shape())))
                fitted_data = zeros(concatenate(((len(foil),), self.maskdict['pre_masks'][pre_mask_key].shape())))
            
            except TypeError:
                contracted_data = zeros(concatenate(((1, 16), self.maskdict['pre_masks'][pre_mask_key].shape())))
                fitted_data = zeros(concatenate(((len(foil),), self.maskdict['pre_masks'][pre_mask_key].shape())))

            except AttributeError:
                print 'contracted_data array could not be initialized. Return None!'
                return None
            
            for ind, foil_ind in enumerate(foil):
                for tc in xrange(16):
                    contracted_data[ind, tc] = self.apply_pre_mask(pre_mask_key, jobind, tc, foil_ind)

        else:
            try:
                contracted_data = array([self.data_dict[self.jobs[jobind]][list(foil)]])
                fitted_data = zeros((len(foil), 128, 128))

            except KeyError:
                print 'No data contraction. Could not initialize usable data from data_dict. Return None!'
                return None
            
            except:
                print 'Sth went wrong with data_contraction in polatization_analysis! Return None!'
                return None
       
        # mask contracted data
        contracted_data = ma.masked_less_equal(contracted_data, 0.)
#        return contracted_data
        # norm contracted data
        # proper error determination
        
        for i in xrange(len(fitted_data)):
            for j in xrange(len(fitted_data[i])):
                for k in xrange(len(fitted_data[i,j])):
                    out = self.single_sinus_fit(contracted_data[i,:,j,k], sqrt(contracted_data[i,:,j,k]))
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
        
        return ma.masked_greater_equal(fitted_data, 1.)
         
#------------------------------------------------------------------------------
        
    @staticmethod
    def _analysis(sine_data_arr):
        """
        fits a sine to an array of (#dat_err, #foils, #tc, #pixel, #pixel)
        """
        
        temp = zeros((2, sine_data_arr.shape[1], sine_data_arr.shape[-2], sine_data_arr.shape[-1], 5))
        
        for find, f in enumerate(sine_data_arr[0]):
            for iind, line in enumerate(f[0]):
                for jind in xrange(len(line)):
                    out = ContrastFit.single_sinus_fit(sine_data_arr[0, find, :, iind, jind], sine_data_arr[0, find, :, iind, jind])
                    temp[0, find, iind, jind] = array([val.value for val in out.params.values()])
                    temp[1, find, iind, jind] = array([val.stderr for val in out.params.values()])
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
                contracted_data = zeros(concatenate(((len(foil), 16), self.maskdict['pre_masks'][pre_mask_key].shape())))
                fitted_data = zeros(concatenate(((len(foil),), self.maskdict['pre_masks'][pre_mask_key].shape())))
            
            except TypeError:
                contracted_data = zeros(concatenate(((1, 16), self.maskdict['pre_masks'][pre_mask_key].shape())))
                fitted_data = zeros(concatenate(((len(foil),), self.maskdict['pre_masks'][pre_mask_key].shape())))

            except AttributeError:
                print 'contracted_data array could not be initialized. Return None!'
                return None
            
            for ind, foil_ind in enumerate(foil):
                for tc in xrange(16):
                    contracted_data[ind, tc] = self.apply_pre_mask(pre_mask_key, jobind, tc, foil_ind)

        else:
            try:
                contracted_data = array([self.data_dict[self.jobs[jobind]][list(foil)]])
                fitted_data = zeros((len(foil), 128, 128))

            except KeyError:
                print 'No data contraction. Could not initialize usable data from data_dict. Return None!'
                return None
            
            except:
                print 'Sth went wrong with data_contraction in polatization_analysis! Return None!'
                return None
       
        # mask contracted data
        contracted_data = ma.masked_less_equal(contracted_data, 0.)
        
        for i in xrange(len(fitted_data)):
            for j in xrange(len(fitted_data[i])):
                for k in xrange(len(fitted_data[i,j])):
                    # Here occures fitting of the sine
                    out = self.single_sinus_fit(contracted_data[i,:,j,k], sqrt(contracted_data[i,:,j,k]))
                    if output == 'pol_bound':
                        fitted_data[i,j,k] = out.params['pol_bound'].value
                    elif output == 'amp':
                        fitted_data[i,j,k] = out.params['amp'].value
                    elif output == 'offset':
                        fitted_data[i,j,k] = out.params['offset'].value
                    elif output == 'phase':
                        fitted_data[i,j,k] = out.params['phase'].value
        
        if output == 'pol_bound':
            return ma.masked_greater_equal(fitted_data, 1.)
        elif output == 'phase':
            return fitted_data % (2*pi) / pi
        else:
            return fitted_data
#------------------------------------------------------------------------------
        
    @staticmethod
    def normalization_count_error_propagation(quantity, floatargs, arrayargs):
        """
        !! WRONG !!
        counting error as args_error = sqrt(c)
        """
        return quantity * sqrt(nansum([1./fa for fa in floatargs]) + nansum([1./aa for aa in arrayargs], axis  = 0))

# =============================================================================
# #------------------------------------------------------------------------------
#     
#     @staticmethod
#     def basic_norm_arr(a1, a2):
#         return 
# =============================================================================







