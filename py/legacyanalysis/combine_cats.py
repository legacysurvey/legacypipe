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

def read_lines(fn):
    fin=open(fn,'r')
    lines=fin.readlines()
    fin.close()
    return list(np.char.strip(lines))

def deg2_lower_limit(ra,dec):
    '''deg2 spanned by objects in each data set, lower limit'''
    ra_wid= ra.max()-ra.min()
    assert(ra > 0.)
    dec_wid= abs(dec.max()-dec.min())
    return ra*dec

def combine_and_match(ref_cats_file,test_cats_file):
    # Set the debugging level
    if args.verbose:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO
    logging.basicConfig(format='%(message)s', level=lvl, stream=sys.stdout)
    log = logging.getLogger('__name__')

    #get lists of tractor cats to compare
    fns_1= read_lines(ref_cats_file) 
    fns_2= read_lines(test_cats_fiile) 
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
    for cnt,cat1,cat2 in zip(range(len(fns_1)),fns_1,fns_2):
        log.info('Reading %s -- %s' % (cat1,cat2))
        ref_tractor = Table(fits.getdata(cat1, 1))
        test_tractor = Table(fits.getdata(cat2, 1))
        m1, m2, d12 = matching.johan_tree(ref_tractor['ra'].copy(), ref_tractor['dec'].copy(),\
                                          test_tractor['ra'].copy(), test_tractor['dec'].copy(), \
                                          dsmax=1.0/3600.0)
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
    d_matched=[]
    deg2=dict(ref=0.,test=0.,ma

    result dict(ref_matched = ref_matched,
                ref_missed = ref_missed,
                test_matched = test_matched,
                test_missed = test_missed,
                d_matched= d_matched,
                deg2= deg2\
                )
    return result

    if __name__ == 'main':
        log.info('do a import combine_and_match')



