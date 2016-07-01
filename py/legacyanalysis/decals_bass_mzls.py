#!/usr/bin/env python

"""compare two tractor catalogues that should have same objects
"""

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import argparse

from legacyanalysis.pathnames import get_outdir
from legacyanalysis.combine_cats import Matched_DataSet
import common_plots as plots

import pickle

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='DECaLS simulations.')
parser.add_argument('--decals_list', type=str, help='list of tractor cats',required=True)
parser.add_argument('--bassmos_list', type=str, help='ditto',required=True) 
args = parser.parse_args()

# Get data
#d= Matched_DataSet(args.decals_list, args.bassmos_list, \
#                   comparison='test')
#fout=open('matched_dataset.pickle','w')
#pickle.dump(d,fout)
#fout.close()
#print('wrote %s' % fout)

fin=open('matched_dataset.pickle','r')
d= pickle.load(fin)
fin.close()


d.ref_matched.apply_mask_by_names(['all'])
d.test_matched.apply_mask_by_names(['all'])
plots.confusion_matrix(d.ref_matched,d.test_matched, name='ALL',\
                       ref_name='DECaLS',test_name='BASS_MzLS')
sys.exit()
# Plots
plots.nobs(d.ref_matched, name='DECaLS')
plots.nobs(d.test_matched, name='BASS_MzLS')
plots.radec(d.ref_matched, name='Matched')
plots.radec(d.ref_missed, name='Missed_Ref')
plots.radec(d.test_missed, name='Missed_Test')
plots.matched_dist(d.ref_matched,d.meta['d_matched'], name='')
plots.hist_types(d.ref_matched, name='Ref')
#print('exiting early')
#sys.exit()
plots.hist_types(d.test_matched, name='Test')
plots.n_per_deg2(d.ref_matched, deg2=d.meta['deg2']['ref'], name='Ref_MatchedDefault')
plots.n_per_deg2(d.test_matched, deg2=d.meta['deg2']['test'], name='Test_MatchedDefault')

# Change mask to union of default masks
d.ref_matched.add_to_current_mask(d.test_matched.masks['default'])
d.test_matched.add_to_current_mask(d.ref_matched.masks['default'])
plots.radec(d.ref_matched, name='Matched_DefaultUnion')
plots.hist_types(d.ref_matched, name='Ref_DefaultUnion')
plots.hist_types(d.test_matched, name='Test_DefaultUnion')
plots.confusion_matrix(d.ref_matched,d.test_matched, name='UnionDefault',\
                       ref_name='DECaLS',test_name='BASS_MzLS')

# union of default + PSF
d.ref_matched.apply_mask_by_names(['current','psf'])
d.test_matched.apply_mask_by_names(['current','psf'])
plots.delta_mag_vs_mag(d.ref_matched,d.test_matched, name='UnionDefaultPsf')
plots.chi_v_gaussian(d.ref_matched,d.test_matched, low=-8.,hi=8., name='UnionDefaultPsf')

# Change mask to default + PSF
d.ref_matched.apply_mask_by_names(['default','psf'])
d.test_matched.apply_mask_by_names(['default','psf'])
plots.sn_vs_mag(d.ref_matched, name="Ref_DefaultPsf")
plots.sn_vs_mag(d.test_matched, name="Test_DefaultPsf")


