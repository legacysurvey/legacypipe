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
#import seaborn as sns

import matplotlib.pyplot as plt

from astropy.io import fits
from astrometry.libkd.spherematch import match_radec

#from thesis_code.fits import tractor_cat
import thesis_code.targets as targets

class Matched_Cats(self):
    def __init__(self):
        self.data={}
        for key in ['m_decam','m_bokmos','u_decam','u_bokmos']: self.data[key]= np.array([])
    def add_dict(self,camera,new_data):
        for key in self.data[camera].keys(): 
            self.data[camera][key]= np.concatenate([self.data[camera][key],new_data[camera][key]])

def match_it(cat1,cat2):
    '''cat1,2 are tractor catalogue to match objects between'''
    #match cats
    data_1= targets.read_from_tractor_cat(cat1)
    data_2= targets.read_from_tractor_cat(cat2)
    #all the 'all1' objects that have match in 'all2' 
    m1, m2, d12 = match_radec(data_1['ra'],data_1['dec'],data_2['ra'],data_2['dec'],\
                            1.0/3600.0,nearest=True)
    m1_unm = np.delete(np.arange(len(data_1['ra'])),m1,axis=0)
    m2_unm = np.delete(np.arange(len(data_2['ra'])),m2,axis=0)
    return data_1,data_2,m1,m2,m1_unm,m2_unm


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='DECaLS simulations.')
parser.add_argument('-fn1', type=str, help='process this brick (required input)')
parser.add_argument('-fn2', type=str, help='object type (STAR, ELG, LRG, BGS)') 

args = parser.parse_args()

#get lists of tractor cats to compare
fns_1=np.loadtxt(args.fn1,dtype=str)
fns_2=np.loadtxt(args.fn2,dtype=str)
#object to store concatenated matched tractor cats
a=Matched_Cats()
for cat1,cat2 in zip(fns_1,fns_2):
    data_1,data_2,m1,m2,m1_unm,m2_unm= match_it(cat1,cat2)
    a.add_dict('m_decam', data_extract(data_1,m1) )
    a.add_dict('m_bokmos', data_extract(data_2,m2))
    a.add_dict('u_decam', data_extract(data_1,m1_unm))
    a.add_dict('u_bokmos', data_extract(data_2,m2_unm))
#each key a.data[key] becomes DECaLS() object with grz mags,i_lrg, etc

#unm['decam']= targets.DECaLS(decam, w1=True)
#unm['bokmos']= targets.DECaLS(bokmos, w1=True)

