#!/usr/bin/env python

"""combine list of tractor cats and match them along the way
"""

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg') #display backend
import os
import sys
import logging
import numpy as np

from astropy.io import fits
from astropy.table import vstack, Table

from astrometry.libkd.spherematch import match_radec

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

def match_two_cats(ref_cats_file,test_cats_file):
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
    #for cnt,cat1,cat2 in zip(range(len(fns_1)),fns_1,fns_2):
    for cnt,cat1,cat2 in zip([0],[fns_1[0]],[fns_2[0]]):
        log.info('Reading %s -- %s' % (cat1,cat2))
        ref_tractor = Table(fits.getdata(cat1, 1))
        test_tractor = Table(fits.getdata(cat2, 1))
        m1, m2, d12 = match_radec(ref_tractor['ra'].copy(), ref_tractor['dec'].copy(),\
                                  test_tractor['ra'].copy(), test_tractor['dec'].copy(), \
                                  1.0/3600.0)
        miss1 = np.delete(np.arange(len(ref_tractor)), m1, axis=0)
        miss2 = np.delete(np.arange(len(test_tractor)), m2, axis=0)
        log.info('matched %d/%d' % (len(m2),len(test_tractor['ra'])))

        # Build combined catalogs
        if len(ref_matched) == 0:
            ref_matched = ref_tractor[m1]
            ref_missed = ref_tractor[miss1]
            test_matched = test_tractor[m2]
            test_missed = test_tractor[miss2]
            d_matched= d12
        else:
            ref_matched = vstack((ref_matched, ref_tractor[m1]))
            ref_missed = vstack((ref_missed, ref_tractor[miss1]))
            test_matched = vstack((test_matched, test_tractor[m2]))
            test_missed = vstack((test_missed, test_tractor[miss2]))
            d_matched= np.concatenate([d_matched, d12])
        deg2['ref']+= deg2_lower_limit(ref_tractor['ra'],ref_tractor['dec'])
        deg2['test']+= deg2_lower_limit(test_tractor['ra'],test_tractor['dec'])
        deg2['matched']+= deg2_lower_limit(ref_matched['ra'],ref_matched['dec'])
    
    return dict(ref_matched = ref_matched,
                ref_missed = ref_missed,
                test_matched = test_matched,
                test_missed = test_missed,
                d_matched= d_matched,
                deg2= deg2)

class DataSet(object):
    '''a LegacySurvey data set contains bands, masks, mags, target selection
    data is a astropy Table with same columns as Tractor Catalogue'''
    def __init__(self,data):
        self.bands= ['g', 'r', 'z']
        self.ibands= [1, 2, 4]
        if 'wise_flux' in data.keys(): 
            self.bands+= ['w1','w2']
        # Create mask
        self.keep= self.keep_indices(data)
        # Mags
        self.magAB,self.magAB_ivar={},{}
        for band,iband in zip(['g', 'r', 'z'],[1,2,4]):
            self.magAB[band]= self.get_magAB(data,iband)
            self.magAB_ivar[band]= self.get_magAB_ivar(data,iband)
        # Get targets
        #self.ts= self.get_TargetSelectin(data)
        # index for where type = psf,simp,exp etc
        self.type= self.get_b_types(data)

    def keep_indices(self,data): 
        '''clean set of sources'''
        return  np.all((data['decam_flux'][:,1] > 0,\
                        data['decam_flux'][:,2] > 0,\
                        data['decam_flux'][:,4] > 0, \
                        data['decam_anymask'][:,1] == 0,\
                        data['decam_anymask'][:,2] == 0,\
                        data['decam_anymask'][:,4] == 0,\
                        data['decam_fracflux'][:,1] <= 0.05,\
                        data['decam_fracflux'][:,2] <= 0.05,\
                        data['decam_fracflux'][:,4] <= 0.05,\
                        data['brick_primary'] == True),axis=0)

    def update_keep_indices(self,newkeep):
        assert(self.keep is not None)
        self.keep= np.all((self.keep,newkeep), axis=0)

    #def get_TargetSelectin(self, data):
    #    d={}
    #    d['desi_target'], d['bgs_target'], d['mws_target']= \
    #                    cuts.apply_cuts( data )
    #    return d

    def get_magAB(self, data,iband):
        mag= data['decam_flux'][:,iband]/data['decam_mw_transmission'][:,iband]
        return 22.5 -2.5*np.log10(mag)

    def get_magAB_ivar(self, data,iband):
        return np.power(np.log(10.)/2.5*data['decam_flux'][:,iband], 2)* \
               data['decam_flux_ivar'][:,iband]  
    
    def get_b_types(self, data):
        '''return boolean mask for type = psf,simp,exp, etc'''
        b_index={}
        for typ in ['PSF','SIMP','EXP','DEV','COMP']:
            b_index[typ.strip()]= data['type'] == typ 
        return b_index

class FillIn(object):
    def __init__(self):
        pass

def DataSets(data, \
             required=['ref_matched','test_matched','ref_missed','test_missed']):
    '''return dict of DataSet objects, containing all info for validation
    data -- returned by match_two_cats()
    ref_name,test_name -- DECaLS, BASS/MzLS if comparing those two'''
    for key in required: 
        assert(key in data.keys())
    ds={}
    for key in required:
        ds[key]= DataSet(data[key])
    # Combine matched masks,type indices, etc
    ds['both']= {}
    ds['both']['keep']= np.all((ds['ref_matched'].keep,\
                                ds['test_matched'].keep), axis=0)
    for typ in ['PSF','SIMP','EXP','DEV','COMP']:
        ds['both'][typ]= np.all((ds['ref_matched'].type[typ],\
                                 ds['test_matched'].type[typ]), axis=0)
    return ds

if __name__ == 'main':
    log.info('do a import combine_and_match')
    ds={}
    for key in ['ref_matched','test_matched','ref_missed','test_missed']:
        ds[key]= DataSet(data[key])
    #combine masks
    mask_match= np.any((ds['ref_matched'],\
                        ds['test_matched']), axis=0)
    mask_missed= np.any((ds['ref_missed'],\
                         ds['test_missed']), axis=0)
    for key in ['ref_matched','test_matched']:
        ds[key].mask= mask_match
    for key in ['ref_missed','test_missed']:
        ds[key].mask= mask_missed



