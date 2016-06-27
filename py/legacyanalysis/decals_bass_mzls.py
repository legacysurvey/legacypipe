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
from legacyanalysis.combine_cats import match_two_cats,DataSets
import legacyanalysis.common_plots as plots


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

info= BassMzls_Decals_Info(args.decals_list, args.bassmos_list)
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

print('finished comparison: BASS/MzLS v DECaLS')

