#!/usr/bin/env python

"""compare two tractor catalogues that should have same objects
"""

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg') #display backend
import os
import sys
import logging
import argparse
import numpy as np
from scipy import stats as sp_stats
#import seaborn as sns

import matplotlib.pyplot as plt
from astropy.io import fits

#import thesis_code.targets as targets
from legacyanalysis import targets 
from legacyanalysis.pathnames import get_outdir
from legacyanalysis.combine_cats import Matched_DataSet
import common_plots as plots

class BassMzls_Decals_Info(object):
    '''object for storing all info to make BASS/MzLS v DECaLS comparisons'''
    def __init__(self, decals_list,bassmos_list):
        # Data
        self.data= match_two_cats(decals_list, bassmos_list)
        self.ds= DataSets(self.data)
        # Additional info
        self.ref_name= 'DECaLS'
        self.test_name= 'BASS/MzLS'
        self.outdir= get_outdir('bmd')
        # Kwargs for plots
        self.kwargs= plots.PlotKwargs()

#if __name__ == 'main':
parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='DECaLS simulations.')
parser.add_argument('--decals_list', type=str, help='process this brick (required input)',required=True)
parser.add_argument('--bassmos_list', type=str, help='object type (STAR, ELG, LRG, BGS)',required=True) 
args = parser.parse_args()

# Get all data could need
d= Matched_DataSet(list_of_ref_cats,list_of_test_cats)
# Validation plots
plots.nobs(d.ref_matched, name='DECaLS')
plots.nobs(d.test_matched, name='BASS_MzLS')
plots.radec(d.ref_matched, name='Matched')
plots.radec(d.ref_missed, name='Missed_Ref')
plots.radec(d.test_missed, name='Missed_Test')
plots.hist_types(d.ref_matched, name='Ref')
plots.hist_types(d.test_matched, name='Test')
plots.confusion_matrix(d.ref_matched,d.test_matched, name='Default',\
                       ref_name='DECaLS',test_name='BASS_MzLS'):

# Change mask to union of default masks
d.ref_matched.extend_mask(d.test_matched.masks['default'])
d.test_matched.extend_masks(d.ref_matched.masks['default'])
plots.radec(d.ref_matched, name='Matched_DefaultUnion')
plots.hist_types(d.ref_matched, name='Ref_DefaultUnion')
plots.hist_types(d.test_matched, name='Test_DefaultUnion')

# Change mask to default + PSF
d.ref_matched.apply_masks_by_name(['default','psf'])
d.test_matched.apply_masks_by_name(['default','psf'])
plots.sn_vs_mag(d.ref_matched, name="Ref_DefaultPsf")
plots.sn_vs_mag(d.test_matched, name="Test_DefaultPsf")


############ as called by durham vers of bass decals plotter
info= BassMzls_Decals_Info(args.decals_list, args.bassmos_list)
#update masks for matched objects to be the join of decam and bokmos masks
mask= np.any((b['m_decam'].mask, b['m_bokmos'].mask),axis=0)
b['m_decam'].update_masks_for_everything(mask=np.any((b['m_decam'].mask, b['m_bokmos'].mask),axis=0),\
                                    mask_wise=np.any((b['m_decam'].mask_wise, b['m_bokmos'].mask_wise),axis=0) )
b['m_bokmos'].update_masks_for_everything(mask=np.any((b['m_decam'].mask, b['m_bokmos'].mask),axis=0),\
                                    mask_wise=np.any((b['m_decam'].mask_wise, b['m_bokmos'].mask_wise),axis=0) )

#plots
#plot_radec(b)
#plot_matched_separation_hist(b['d12'])
# Depths are very different so develop a cut to make fair comparison
#plot_SN_vs_mag(b, found_by='matched',type='psf')
# mask=True where BASS SN g < 5 or BASS SN r < 5
sn_crit=5.
mask= np.any((b['m_bokmos'].data['gflux']*np.sqrt(b['m_bokmos'].data['gflux_ivar']) < sn_crit,\
              b['m_bokmos'].data['rflux']*np.sqrt(b['m_bokmos'].data['rflux_ivar']) < sn_crit),\
                axis=0)
b['m_decam'].update_masks_for_everything(mask=mask, mask_wise=mask)
b['m_bokmos'].update_masks_for_everything(mask=mask, mask_wise=mask)
# contintue with fairer comparison
#plot_radec(b,addname='snGe5')
#plot_HistTypes(b,m_types=['m_decam','m_bokmos'],addname='snGe5')
#plot_SN_vs_mag(b, found_by='matched',type='psf',addname='snGe5')
#plot_SN_vs_mag(b, found_by='matched',type='all')
#plot_SN_vs_mag(b, found_by='matched',type='lrg')
#plot_SN_vs_mag(b, found_by='unmatched',type='all')
#plot_SN_vs_mag(b, found_by='unmatched',type='psf')
#plot_SN_vs_mag(b, found_by='unmatched',type='lrg')
#cm,names= create_confusion_matrix(b)
#plot_confusion_matrix(cm,names,addname='snGe5')
plot_dflux_chisq(b,type='psf',addname='snGe5')
#plot_dflux_chisq(b,type='all',addname='snGe5')
# Number density cutting to requirement mags: grz<=24,23.4,22.5
print('square deg covered by decam=',b['deg2_decam'],'and by bokmos=',b['deg2_bokmos'])
#plot_N_per_deg2(b,type='psf',addname='snGe5')
#plot_N_per_deg2(b,type='lrg',addname='snGe5')
plot_magRatio_vs_mag(b,type='psf',addname='snGe5')
########################


# plots
print('square deg covered by %s=%.2f and %s=%.2f' % \
    (info.ref_name,info.data['deg2']['ref'],info.test_name,info.data['deg2']['test']))
plots.radec(info)
plots.matched_separation_hist(info)
# Depths are very different so develop a cut to make fair comparison
plots.SN_vs_mag(info,obj_type='PSF')
# Keep matched where BASS SN g < 5 or BASS SN r < 5
sn_crit=5.
keep= np.any((info.data['test_matched']['decam_flux'][:,1]*np.sqrt(\
                info.data['test_matched']['decam_flux_ivar'][:,1]) < sn_crit,\
              info.data['test_matched']['decam_flux'][:,2]*np.sqrt(\
                info.data['test_matched']['decam_flux_ivar'][:,2]) < sn_crit),\
                axis=0)
for key in ['ref_matched','test_matched']: 
    info.ds[key].update_keep_indices(keep)
# Continue plotting
plots.radec(info,addname='snGe5')
plots.HistTypes(info,addname='snGe5')
plots.SN_vs_mag(info,obj_type='PSF',addname='snGe5')
cm,names= plots.create_confusion_matrix(info)
plots.confusion_matrix(cm,names, info,addname='snGe5')
#plots.dflux_chisq(info,obj_type='PSF',addname='snGe5')
plots.N_per_deg2(info,obj_type='PSF',addname='snGe5')
#plots.N_per_deg2(info,obj_type='LRG',addname='snGe5')

