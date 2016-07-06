#!/usr/bin/env python

"""combine list of tractor cats and match them along the way

Example of Matched Comparison:
from legacyanalysis.combine_cats import Matched_DataSet
import common_plots as plots
d= Matched_DataSet(list_of_ref_cats,list_of_test_cats)
plots.confusion_matrix(d.ref_matched.t,d.test_matched.t)
"""

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg') #display backend
import os
import sys
import logging
import numpy as np

from astropy.io import fits
from astropy.table import vstack, Table, Column
#from scipy.spatial import KDTree
from astrometry.libkd.spherematch import match_radec

from legacyanalysis.validation.pathnames import get_outdir

def read_lines(fn):
    fin=open(fn,'r')
    lines=fin.readlines()
    fin.close()
    return list(np.char.strip(lines))

def deg2_lower_limit(ra,dec):
    '''deg2 spanned by objects in each data set, lower limit'''
    ra_wid= ra.max()-ra.min()
    assert(ra_wid > 0.)
    dec_wid= abs(dec.max()-dec.min())
    return ra_wid*dec_wid

def kdtree_match(ref_ra,ref_dec, ra,dec, k=1, dsmax=1./3600):
    '''finds approx. distance between reference and test points and
    returns array of indices and distance where distance < dsmax (degrees)
    '''
    assert(len(ref_ra) == len(ref_dec))
    assert(len(ra) == len(dec))
    # Index the tree
    tree = KDTree(np.transpose([dec.copy(),ra.copy()])) 
    # Find the k nearest neighbors to each reference ra,dec
    ds, i_tree = tree.query(np.transpose([ref_dec.copy(),ref_ra.copy()]), k=k) 
    # Get indices where match
    i_ref,i_other={},{}
    ref_match= np.arange(len(ref_ra))[ds<=dsmax]
    other_match= i_tree[ds<=dsmax]
    assert(len(ref_ra[ref_match]) == len(ra[other_match]))
    return np.array(ref_match),np.array(other_match),np.array(ds[ds<=dsmax])

def combine_single_cats(cat_list, debug=False):
    '''return dict containing astropy Table of concatenated tractor cats'''
    # Set the debugging level
    lvl = logging.INFO
    logging.basicConfig(format='%(message)s', level=lvl, stream=sys.stdout)
    log = logging.getLogger('__name__')

    #get lists of tractor cats to compare
    fns= read_lines(ref_cats_file) 
    log.info('Combining tractor catalogues: ')
    for fn in fns: log.info("%s" % fn) 
    #object to store concatenated matched tractor cats
    bigtractor = []
    deg2= 0.
    # One catalogue for quick debugging
    if debug: fns= fns[:1]
    # Loop over cats
    for cnt,fn in zip(range(len(fns)),fns):
        log.info('Reading %s' % fn)
        tractor = Table(fits.getdata(fn, 1), masked=True)
        # Build combined catalogs
        if len(bigtractor) == 0:
            bigtractor = tractor
        else:
            bigtractor = vstack((bigtractor, tractor), metadata_conflicts='error')
        deg2+= deg2_lower_limit(tractor['ra'],tractor['dec'])
    
    return bigtractor, dict(deg2=deg2)


def match_two_cats(ref_cats_file,test_cats_file, debug=False):
    '''return dict containing astropy Table of concatenated tractor cats
    one Table for matched and missed reference and test objects, each'''
    # Set the debugging level
    lvl = logging.INFO
    logging.basicConfig(format='%(message)s', level=lvl, stream=sys.stdout)
    log = logging.getLogger('__name__')

    #get lists of tractor cats to compare
    fns_1= read_lines(ref_cats_file) 
    fns_2= read_lines(test_cats_file) 
    log.info('Comparing tractor catalogues: ')
    for one,two in zip(fns_1,fns_2): log.info("%s -- %s" % (one,two)) 
    #if fns_1.size == 1: fns_1,fns_2= [fns_1],[fns_2]
    #object to store concatenated matched tractor cats
    ref_matched = []
    ref_missed = []
    test_matched = []
    test_missed = []
    d_matched= 0.
    deg2= dict(ref=0.,test=0.,matched=0.)
    # One catalogue for quick debugging
    if debug: fns_1,fns_2= fns_1[:1],fns_2[:1]
    # Loop over cats
    for cnt,cat1,cat2 in zip(range(len(fns_1)),fns_1,fns_2):
        log.info('Reading %s -- %s' % (cat1,cat2))
        ref_tractor = Table(fits.getdata(cat1, 1), masked=True)
        test_tractor = Table(fits.getdata(cat2, 1), masked=True)
        m1, m2, d12 = match_radec(ref_tractor['ra'].data.copy(), ref_tractor['dec'].data.copy(),\
                                  test_tractor['ra'].data.copy(), test_tractor['dec'].data.copy(), \
                                  1.0/3600.0)
        #m1, m2, d12= kdtree_match(ref_tractor['ra'].copy(), ref_tractor['dec'].copy(),\
        #                          test_tractor['ra'].copy(), test_tractor['dec'].copy(),\
        #                          k=1, dsmax=1./3600)
        print("Matched: %d/%d objects" % (m1.size,len(ref_tractor['ra'])))
        miss1 = np.delete(np.arange(len(ref_tractor)), m1, axis=0)
        miss2 = np.delete(np.arange(len(test_tractor)), m2, axis=0)

        # Build combined catalogs
        if len(ref_matched) == 0:
            ref_matched = ref_tractor[m1]
            ref_missed = ref_tractor[miss1]
            test_matched = test_tractor[m2]
            test_missed = test_tractor[miss2]
            d_matched= d12
        else:
            ref_matched = vstack((ref_matched, ref_tractor[m1]), metadata_conflicts='error')
            ref_missed = vstack((ref_missed, ref_tractor[miss1]), metadata_conflicts='error')
            test_matched = vstack((test_matched, test_tractor[m2]), metadata_conflicts='error')
            test_missed = vstack((test_missed, test_tractor[miss2]), metadata_conflicts='error')
            d_matched= np.concatenate([d_matched, d12])
        deg2['ref']+= deg2_lower_limit(ref_tractor['ra'],ref_tractor['dec'])
        deg2['test']+= deg2_lower_limit(test_tractor['ra'],test_tractor['dec'])
        deg2['matched']+= deg2_lower_limit(ref_matched['ra'],ref_matched['dec'])
    
    return dict(ref_matched = ref_matched,
                ref_missed = ref_missed,
                test_matched = test_matched,
                test_missed = test_missed), \
           dict(d_matched= d_matched, deg2= deg2)

class Single_TractorCat(object):
    '''given a astropy table of a Tractor Catalogue, this object computes
    everything you could need to know for the Tractor Catalogue'''
    def __init__(self,astropy_t, comparison='test'):
        self.t= astropy_t
        # clean up fields, e.g. type = 'PSF' instead of 'PSF '
        self.t['type']= np.char.strip(self.t['type']) 
        # outdir for when making plots
        self.outdir= get_outdir(comparison)
        # AB Mags
        self.get_magAB()
        self.get_magAB_ivar()
        # Masks (True where mask OUT values)
        self.masks= dict(default= self.basic_cuts(),\
                         all= np.zeros(len(self.t['ra'])).astype(bool),\
                         psf= self.cut_on_type('PSF'),\
                         simp= self.cut_on_type('SIMP'),\
                         exp= self.cut_on_type('EXP'),\
                         dev= self.cut_on_type('DEV'),\
                         comp= self.cut_on_type('COMP'),\
                         extended= self.cut_on_type('EXTENDED'))
        # Initialize with default mask
        self.apply_mask_by_names(['default'])
        # Target selection
        #self.targets= self.get_TargetSelectin(data)

    def get_magAB(self):
        self.t['decam_mag']= self.t['decam_flux']/self.t['decam_mw_transmission']
        self.t['decam_mag']= 22.5 -2.5*np.log10(self.t['decam_mag'])

    def get_magAB_ivar(self):
        self.t['decam_mag_ivar']= np.power(np.log(10.)/2.5*self.t['decam_flux'], 2)* \
                                 self.t['decam_flux_ivar']
 
    def basic_cuts(self): 
        '''True where BAD, return boolean array'''
        return  np.any((self.t['decam_flux'].data[:,1] < 0,\
                        self.t['decam_flux'].data[:,2] < 0,\
                        self.t['decam_flux'].data[:,4] < 0, \
                        self.t['decam_anymask'].data[:,1] != 0,\
                        self.t['decam_anymask'].data[:,2] != 0,\
                        self.t['decam_anymask'].data[:,4] != 0,\
                        self.t['decam_fracflux'].data[:,1] > 0.05,\
                        self.t['decam_fracflux'].data[:,2] > 0.05,\
                        self.t['decam_fracflux'].data[:,4] > 0.05,\
                        self.t['brick_primary'].data == False),axis=0)

    def apply_boolean_mask(self, mask):
        '''set each Column's mask to specified mask'''
        # Tell masks dict what that this is the new current mask
        self.masks['current']= mask
        for key in self.t.keys(): self.t[key].mask= self.masks['current']

    def add_to_current_mask(self,newmask):
        '''mask additional pixels to current mask'''
        assert(self.masks is not None)
        mask= np.any((self.masks['current'],newmask), axis=0)
        self.apply_boolean_mask(mask)

    def combine_mask_by_names(self,list_of_names):
        '''return boolean union of all masks in "list_of_names"'''
        assert(self.masks is not None)
        for name in list_of_names: assert(name in self.masks.keys())
        # return union
        mask= self.masks[list_of_names[0]]
        if len(list_of_names) > 1:
            for i,name in enumerate(list_of_names[1:]): 
                mask= np.any((mask, self.masks[name]), axis=0)
        return mask

    def apply_mask_by_names(self,list_of_names):
        '''set mask to union of masks in "list of names"'''
        union= self.combine_mask_by_names(list_of_names)
        self.apply_boolean_mask(union)

    def number_not_masked(self,list_of_names):
        '''for mask being union of "list of names", 
        returns number object not masked out 
        example -- n_psf= self.number_masked(['psf'])'''
        union= self.combine_mask_by_names(list_of_names)
        # Where NOT masked out
        return np.where(union == False)[0].size
    
    def cut_on_type(self,name):
        '''True where BAD, return boolean array'''
        # Return boolean array
        if name in ['PSF','SIMP','EXP','DEV','COMP']:
            keep= self.t['type'].data == name
        elif name == 'EXTENDED':
            keep= self.t['type'].data != 'PSF'
        else: raise ValueError
        return keep == False

    #def get_TargetSelectin(self, data):
    ##### store as self.t.masks['elg'], self.t.masks['lrg'], etc
    #    d={}
    #    d['desi_target'], d['bgs_target'], d['mws_target']= \
    #                    cuts.apply_cuts( data )
    #    return d

    

class Single_DataSet(object):
    '''a Single_DataSet contains a concatenated tractor catalogue as Astropy table
    with additional columns for mags, masks, target selection, etc'''
    def __init__(self,cat_list, comparison='test',debug=False):
        # Store tractor catalogue as astropy Table
        astropy_t, self.meta= combine_single_cats(cat_list, debug=debug)
        self.single= Single_TractorCat(astropy_t, comparison)

class Matched_DataSet(object):
    '''a Matched_DataSet contains a dict of 4 concatenated, matched, tractor catalogue astropy Tables
    each table has additional columns for mags, target selection, masks, etc.'''
    def __init__(self,ref_cats_file,test_cats_file, comparison='test',debug=False):
        astropy_ts, self.meta= match_two_cats(ref_cats_file,test_cats_file, debug=debug)
        # have 4 tractor catalogues
        self.ref_matched= Single_TractorCat(astropy_ts['ref_matched'], comparison)
        self.test_matched= Single_TractorCat(astropy_ts['test_matched'], comparison)
        self.ref_missed= Single_TractorCat(astropy_ts['ref_missed'], comparison)
        self.test_missed= Single_TractorCat(astropy_ts['test_missed'], comparison) 


