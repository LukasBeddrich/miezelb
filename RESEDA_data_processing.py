#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 09:03:05 2017

@author: lbeddric
"""

#%%
import RESEDA_MIEZE_data_treatment as RMdt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#%%

"""
R_780 =  RMdt.ContrastFit('038780')
R_780.initialize_pre_mask(128, 4)

R_780.show_job2D(3)
R_780.maskdict['pre_masks']['0'].show_pre_mask()
for i in xrange(16):
    R_780.dump_to_memory('contracted{}'.format(i), R_780.contract_data(R_780.maskdict['pre_masks']['0'],
                                                        R_780.data_dict[R_780.jobs[3]][7,i]))
    
R_780.polarization_analysis(3, pre_mask_key='0')
#for i in xrange(16):
#     contracted_array[:,:,i] = R_780.contract_data(R_780.maskdict['pre_masks']['0'], R_780.data_dict[R_780.jobs[0]][7,i])
"""

###############################################################################
########### Construct reasonable post masks ###################################

"""R_780 =  RMdt.ContrastFit('038780')"""
#R_780.initialize_pre_mask(128, 4)
#R_780.initialize_post_mask(128, (105, 100), 10, 14, (120,240)) # '0'
#R_780.initialize_post_mask(128, (105, 100), 14, 18, (130,230)) # '1'
#R_780.initialize_post_mask(128, (105, 100), 18, 22, (140,220)) # '2'
#R_780.initialize_post_mask(128, (105, 100), 22, 26, (140,220)) # '3'
#R_780.initialize_post_mask(128, (105, 100), 26, 30, (140,220)) # '4'
#R_780.initialize_post_mask(128, (105, 100), 30, 34, (145,215)) # '5'
#R_780.initialize_post_mask(128, (70, 105), 0, 4, (0,360)) # '6'


#for i in xrange(len(R_780.maskdict['post_masks'])):
#    R_780.maskdict['post_masks']['{}'.format(i)].show_post_mask()

""" masks defined """

###############################################################################

###############################################################################
########## check phases for Bckg and MiezeEcho_33 in direct beam ##############

"""R_784 = RMdt.ContrastFit('038784')"""
#R_780.single_sinus_fit(R_780.data_dict['MiezeEcho_33'][7,:,71,107],R_780.data_dict['MiezeEcho_33'][7,:,71,107], True)
#R_784.single_sinus_fit(R_784.data_dict['MiezeEcho_33'][7,:,71,107],R_784.data_dict['MiezeEcho_33'][7,:,71,107], True)
"""
for k in np.array([[(i, j) for j in xrange(104,108)] for i in xrange(69,73)]).reshape((-1,2)):
    print k
    R_780.single_sinus_fit(R_780.data_dict['MiezeEcho_33'][7,:,k[0],k[1]],R_780.data_dict['MiezeEcho_33'][7,:,k[0],k[1]], True)
    print '\n'
    R_784.single_sinus_fit(R_784.data_dict['MiezeEcho_33'][7,:,k[0],k[1]],R_784.data_dict['MiezeEcho_33'][7,:,k[0],k[1]], True)
    print '\n \n \n'
"""
"""
for i in R_780.jobs:
    R_780.single_sinus_fit(R_780.data_dict[i][7,:,71,105],R_780.data_dict[i][7,:,71,105], True)
"""

#R_779 = RMdt.ContrastFit('038779')

#R_780.initialize_pre_mask(128, 4)
#R_780.initialize_post_mask(128, (105, 100), 10, 14, (120,240)) # '0'
#R_780.initialize_post_mask(128, (105, 100), 14, 18, (130,230)) # '1'
#R_780.initialize_post_mask(128, (105, 100), 18, 22, (140,220)) # '2'
#R_780.initialize_post_mask(128, (105, 100), 22, 26, (140,220)) # '3'
#R_780.initialize_post_mask(128, (105, 100), 26, 30, (140,220)) # '4'
#R_780.initialize_post_mask(128, (105, 100), 30, 34, (145,215)) # '5'
#R_780.initialize_post_mask(128, (70, 105), 0, 4, (0,360)) # '6'



#%%

fwcol = '/home/lbeddric/Dokumente/Data/RESEDAdata/038779'
fwocol = '/home/lbeddric/Dokumente/Data/RESEDAdata/038780'
fbgwcol = '/home/lbeddric/Dokumente/Data/RESEDAdata/038782'
fbgwocol = '/home/lbeddric/Dokumente/Data/RESEDAdata/038784'

# MIEZE data set
CFw = RMdt.ContrastFit(fwcol)
CFw.initialize_pre_mask(128,4)
CFw.initialize_post_square_mask(32, (12,2,24,3))
# Background MIT NORMALISIERUNG
CFbw = RMdt.ContrastFit(fbgwcol)
CFbw.initialize_pre_mask(128,4)
CFbw.initialize_post_square_mask(32, (12,2,25,2))
# Background OHNE NORMALISIERUNG
#CFb2 = RMdt.ContrastFit(fbgwcol)
#CFb2.initialize_pre_mask(128,4)
#CFb2.load_specificjobdata(norm_mon = False, jobind = 0)

#%%
# contraction and data analysis with sinus fit
print '\nStart contraction calculation\n'
for job in CFw.jobs:
    jobind = CFw.jobs.index(job)
    dump_key = 'c_ME_{}'.format(job.split('_')[-1])
    CFw.contract_data( 0, jobind, foil = (7,), tc = range(16), dump = dump_key)
    
CFbw.contract_data( 0, 0, foil = (7,), tc = range(16), dump = 'c_ME_33')

#%%
print '\nStart fitting and analysis of both data sets\n'

for key in CFw.local_memory.keys():
    dump_key = 'ana_{}'.format(key)    
    CFw.dump_to_memory(dump_key, CFw._analysis(CFw.get_from_memory(key)))
    
CFbw.dump_to_memory('ana_c_ME_33', CFbw._analysis(CFbw.get_from_memory('c_ME_33')))

#%%
# =============================================================================
# # Some plotting to see the polarization data
# print "plot of some exemplary polarization data"
# 
# CFw.show_image(np.ma.masked_less(CFw.get_from_memory('ana_c_ME_33')[0,0,:,:,2], 1.0e-3), log = True, norm = LogNorm())
# plt.title('Polarization in helical phase')
# CFbw.show_image(np.ma.masked_less(CFbw.get_from_memory('ana_c_ME_33')[0,0,:,:,2], 1.0e-3), log = True, norm = LogNorm())
# plt.title('Polarization of background')
# =============================================================================

#%%
print '\nCorrect for intrument resolution: \nChoose ROI mask for normalization and caluclate\n'

for key in CFw.local_memory.keys():
    if 'ana_' in key:
        CFw_res = np.sum(CFw.maskdict['post_masks'][0].mask * CFw.get_from_memory(key)[0,0,:,:,2])/np.sum(CFw.maskdict['post_masks'][0].mask)
        CFw_res_err = np.nansum((CFw.maskdict['post_masks'][0].mask * CFw.get_from_memory(key)[1,0,:,:,2])**2)

        CFw.dump_to_memory('norm-pol_ana_c_ME_{}'.format(key.split('_')[-1]), CFw.get_from_memory(key)[0,0,:,:,2] / CFw_res)
        CFw.dump_to_memory('norm-pol_ana_c_ME_{}'.format(key.split('_')[-1]), np.array((CFw.get_from_memory('norm-pol_ana_c_ME_{}'.format(key.split('_')[-1])),
                                                    CFw.get_from_memory('norm-pol_ana_c_ME_{}'.format(key.split('_')[-1])) *\
                                                    np.sqrt((CFw_res_err/CFw_res)**2 + (CFw.get_from_memory('ana_c_ME_{}'.format(key.split('_')[-1]))[1,0,:,:,2]/CFw.get_from_memory('ana_c_ME_{}'.format(key.split('_')[-1]))[0,0,:,:,2])**2))))


CFbw_res = np.sum(CFbw.maskdict['post_masks'][0].mask * CFbw.get_from_memory('ana_c_ME_33')[0,0,:,:,2])/np.sum(CFbw.maskdict['post_masks'][0].mask)
CFbw_res_err = np.nansum((CFbw.maskdict['post_masks'][0].mask * CFbw.get_from_memory('ana_c_ME_33')[1,0,:,:,2])**2)

CFbw.dump_to_memory('norm-pol_ana_c_ME_33', CFbw.get_from_memory('ana_c_ME_33')[0,0,:,:,2] / CFbw_res)
CFbw.dump_to_memory('norm-pol_ana_c_ME_33', np.array((CFbw.get_from_memory('norm-pol_ana_c_ME_33'),
                                                    CFbw.get_from_memory('norm-pol_ana_c_ME_33') *\
                                                    np.sqrt((CFbw_res_err/CFbw_res)**2 + (CFbw.get_from_memory('ana_c_ME_33')[1,0,:,:,2]/CFbw.get_from_memory('ana_c_ME_33')[0,0,:,:,2])**2))))

#%%
print "\nExpand the polarization data again\n"

CFbw.dump_to_memory('exp_norm-pol_ana_c_ME_33', np.array([CFbw._expand_data(CFbw.maskdict['pre_masks'][0], CFbw.local_memory['ana_c_ME_33'][0,0,:,:,2]), CFbw._expand_data(CFbw.maskdict['pre_masks'][0], CFbw.local_memory['ana_c_ME_33'][1,0,:,:,2])]))

for key in CFw.local_memory.keys():
    if "norm-pol" in key:
        CFw.dump_to_memory('exp_{}'.format(key), np.array([CFw._expand_data(CFw.maskdict['pre_masks'][0], CFw.local_memory['ana_c_ME_{}'.format(key.split('_')[-1])][0,0,:,:,2]), CFw._expand_data(CFw.maskdict['pre_masks'][0], CFw.local_memory['ana_c_ME_{}'.format(key.split('_')[-1])][1,0,:,:,2])]))



#%%
print "\nBuild some sector masks\n" 
# =============================================================================
# post_sec_params = [[(101, 101)]*6,
#                  [0, 5, 10, 15, 20, 25],
#                  [4, 9, 14, 19, 24, 29],
#                  [(0, 360), (120, 230), (120, 230), (120, 230), (120, 230), (120, 230)]]
# =============================================================================

post_sec_params = [[(101, 101)]*5,
                 [0, 5, 10, 15, 20],
                 [4, 9, 14, 19, 24],
                 [(0, 360), (120, 230), (120, 230), (120, 230), (120, 230)]]

for ind, param in enumerate(post_sec_params[0]):
    CFw.initialize_post_sector_mask(128, param, post_sec_params[1][ind], post_sec_params[2][ind], post_sec_params[3][ind])

#%%
# =============================================================================
# # visualize sector masks
# for mask in CFw.maskdict['post_masks'].values():
#     if mask.masktype == 'Sector mask':
#         mask.show_post_mask()
# =============================================================================

#%%
print "Average contrast and its error in the sector mask"

tauind = 0
C_q_tau = np.zeros((2, 5, len(CFw.mieze_taus)))
# (#val/err, #tau/q, len(qarray), len(tauarray))
q_tau = np.zeros((2, 2, 5, len(CFw.mieze_taus)))

for key in CFw.local_memory.keys():
    qind = 0
    if 'exp_norm-pol' in key:
        temptau, temptauerr = CFw.mieze_taus['MiezeEcho_{}'.format(key.split('_')[-1])]
        for mask in CFw.maskdict['post_masks'].values():
            if mask.masktype == 'Sector mask':
                print mask.r_o
                tempq, tempqerr = mask.q()
                C_q_tau[0, qind, tauind] = np.nansum(CFw.local_memory[key][0] * mask.mask) / mask.mask.sum()
                C_q_tau[1, qind, tauind] = np.sqrt(np.nansum(((CFw.local_memory[key][0] - C_q_tau[0, qind, tauind])*mask.mask)**2)/mask.mask.sum())
                q_tau[: ,: , qind, tauind] = np.array([[tempq, temptau],[tempqerr, temptauerr]])
                qind += 1
        tauind += 1

#%%

print '\nplotting the contrast over tau and maybe q'

fig = plt.figure()
for qind, q in enumerate(q_tau[0,0,:,0]):
    plt.errorbar(x = q_tau[0,1,qind], xerr = q_tau[1,1,qind], y = C_q_tau[0,qind], yerr = C_q_tau[1,qind], label = '{}'.format(round(q, 5)) + r'$\pm$' + '{}'.format(q_tau[1,0,qind,0].round(5)), marker = 'o', ms = 6., ls = 'None')
plt.xscale('log')
plt.legend(loc = 'best', numpoints = 1)
plt.xlim(xmin = 1.0e-7, xmax = 5.0)
plt.ylim(ymin = -0.05, ymax = 1.05)
plt.show()














