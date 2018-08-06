#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

mask classes

"""

###############################################################################
####################        IMPORTS        ####################################
###############################################################################

import matplotlib.pyplot as plt
from matplotlib.cm import Greys
from numpy import abs, arctan2, bool, deg2rad, float, sqrt, sum, ogrid, where, zeros
from math import pi

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
        self.mask = zeros((self.nn, self.nn), dtype = float)
        self.masktype = 'Mask_Base'
        
#------------------------------------------------------------------------------
        
    def __repr__(self):
        """
        official string description
        """
        return '{}x{} {}'.format(str(self.nn), str(self.nn), self.masktype)

#------------------------------------------------------------------------------

    def getMask(self):
        """
        returns mask
        """
        return self.mask

#------------------------------------------------------------------------------

    def shape(self):
        """
        returns mask shape
        """
        return self.mask.shape

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------

    def changetile_size(self, tile_size):
        """
        
        """
        if self.nn % tile_size == 0:
            self.tile_size = tile_size
        else:
            print 'tile_size is not a divisor of nn! tile_size set to 1.'
            self.tile_size = 1

#------------------------------------------------------------------------------

    def create_pre_mask(self):
        """
        creates tiled pregrouping mask array
        """
        ratio = self.nn/self.tile_size
        for i in xrange(ratio):
            for j in xrange(ratio):
                self.mask[i*self.tile_size:(i + 1)*self.tile_size, j*self.tile_size:(j + 1)*self.tile_size] = i*ratio + j

#------------------------------------------------------------------------------

    def shape(self, mod_ts = True):
        """
        show pre_mask dimensions mod tile_size
        """
        if mod_ts:
            return (self.nn/self.tile_size,)*2
        else:
            return super(self.__class__, self).shape()

#------------------------------------------------------------------------------

    def show_pre_mask(self):
        """
        
        """
        temparr = where(self.mask %2 == 1, 1, -1)
        if (self.nn / self.tile_size) % 2 == 0:
            temparr = abs(temparr + temparr.T) - 1
        Mask_Base.show_mask(temparr, self.masktype)
        return None

#------------------------------------------------------------------------------

    def getboolMask(self):
        """
        
        """
        temparr = where(self.mask %2 == 1, 1, -1)
        if (self.nn / self.tile_size) % 2 == 0:
            temparr = abs(temparr + temparr.T) - 1
        return temparr

###############################################################################
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
        self.tmin, self.tmax = deg2rad(angle_range)
        self.create_post_mask()
        self.qxyz = zeros((self.nn, self.nn, 3))

#------------------------------------------------------------------------------

    def create_post_mask(self):
        x,y = ogrid[:self.nn,:self.nn]
        cx,cy = self.centre
        
        #ensure stop angle > start angle
        if self.tmax<self.tmin:
            self.tmax += 2*pi
        #convert cartesian --> polar coordinates
        r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
        theta = arctan2(x-cx,y-cy) - self.tmin
        #wrap angles between 0 and 2*pi
        theta %= (2*pi)
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
        qq = (2*pi/6.0)

        for x in xrange(cx - (self.r_o + 1), cx + (self.r_o + 2)):
            for y in xrange(cy - (self.r_o + 1), cy + (self.r_o + 2)):
                n_path_length = sqrt(self.d_SD**2 + self.pixelsize**2*(x-cx)**2 + self.pixelsize**2*(y-cy)**2)
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
            q_abs = sum(sqrt(sum(self.qxyz**2, axis = 2)) * self.mask) / self.mask.sum()
            q_abs_err = sqrt(1.0/(self.mask.sum() - 1) * sum(((sqrt(sum(self.qxyz**2, axis = 2)) - q_abs) * self.mask)**2))
            if q_abs.any() != 0:
                return q_abs, q_abs_err
            else:
                self.every_q()
                self.q(counter + 1)

#------------------------------------------------------------------------------

    def show_post_mask(self):
        """
        
        """
        Mask_Base.show_mask(where(self.mask == True, 1, 0), self.masktype)
        return None

###############################################################################
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
        
        self.mask = self.mask.astype(bool)
        for llbhval in xrange(len(self.lefts)):
            self.mask[self.lefts[llbhval]:self.lefts[llbhval] + self.lengths[llbhval], self.bottoms[llbhval]:self.bottoms[llbhval] + self.heights[llbhval]] = True

###############################################################################
###############################################################################





