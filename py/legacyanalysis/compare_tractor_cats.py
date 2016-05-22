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

class Matched_Cats():
    def __init__(self,):
        self.data={}
    def initialize(self,data_1,data_2,m1,m2,m1_unm,m2_unm):
        #self.data['all_1']= targets.data_extract(data_1,range(len(data_1['ra']))) 
        self.data['m_decam']= targets.data_extract(data_1,m1) 
        self.data['m_bokmos']= targets.data_extract(data_2,m2)
        self.data['u_decam']= targets.data_extract(data_1,m1_unm)
        self.data['u_bokmos']= targets.data_extract(data_2,m2_unm)
    def add_dict(self,match_type,new_data):
        '''match_type -- m_decam,m_bokmos,u_decam, etc
        new data -- data returend from read_from..() to be concatenated with existing m_decam, etc'''
        for key in self.data[match_type].keys(): 
            self.data[match_type][key]= np.concatenate([self.data[match_type][key],new_data[key]])

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

def read_lines(fn):
    fin=open(fn,'r')
    lines=fin.readlines()
    fin.close()
    return list(np.char.strip(lines))

#plotting vars
laba=dict(fontweight='bold',fontsize='x-large')
kwargs_axtext=dict(fontweight='bold',fontsize='x-large',va='top',ha='left')
leg_args=dict(loc=1,frameon=True)


def plot_SN(obj,m_types=['m_decam','m_bokmos'], index='all'): 
    '''decam,bokmos -- DECaLS() objects with matched OR unmatched indices
    index -- "all, ptf, lrg" and just those indices will be plotted'''
    #matching or unmatched objects?
    if m_types[0].startswith('m_') and m_types[1].startswith('m_'): matched=True
    elif m_types[0].startswith('u_') and m_types[1].startswith('u_'): matched=False   
    else: raise ValueError
    #set seaborn panel styles
    #sns.set_style('ticks',{"axes.facecolor": ".97"})
    #sns.set_palette('colorblind')
    #setup plot
    fig,ax=plt.subplots(1,3,figsize=(9,3)) #,sharey=True)
    plt.subplots_adjust(wspace=0.5)
    #plot
    color=['b','g']
    for cnt,val in zip(range(3),['rmag','gmag','zmag']):
        for ith,camera in zip(range(2),m_types):
            SN= obj[camera].data[val]*np.sqrt(obj[camera].data[val+'_ivar'])
            if index == 'all': inds= np.arange(obj[camera].data[val].size)
            elif index == 'psf': inds= obj[camera].data['type'] == 'PSF'
            elif index == 'lrg': inds= obj[camera].data['i_lrg']
            else: raise ValueError
            ax[cnt].scatter(obj[camera].data[val][inds], SN[ inds ], \
                            edgecolor=color[ith],c='none',lw=2.,label=camera.split('_')[-1])
        xlab=ax[cnt].set_xlabel(val, **laba)
        ax[cnt].set_xlim(20,30)
        ax[cnt].set_ylim(1e0,1e3)
        ax[cnt].set_yscale('log')
    ylab=ax[0].set_ylabel(r'$m / \sigma_m$', **laba)
    ax[2].legend(**leg_args)
    if matched: sup= '%s, Matched' % index
    else: sup= '%s, Unmatched' % index
    sup=plt.suptitle(sup,**laba)
    #save
    #sns.despine()
    if matched: fout='sn_%s.png' % index
    else: fout='sn_%s_unmatched.png' % index
    plt.savefig(fout, bbox_extra_artists=[xlab,ylab,sup], bbox_inches='tight',dpi=150)
    plt.close()

def plot_HistTypes(obj,m_types=['m_decam','m_bokmos']):
    '''decam,bokmos -- DECaLS() objects with matched OR unmatched indices'''
    #matched or unmatched objects
    if m_types[0].startswith('m_') and m_types[1].startswith('m_'): matched=True
    elif m_types[0].startswith('u_') and m_types[1].startswith('u_'): matched=False   
    else: raise ValueError
    #sns.set_style("whitegrid")
    #sns.set_palette('colorblind')
    #c1=sns.color_palette()[2] 
    #c2=sns.color_palette()[0] #'b'
    c1= 'b' 
    c2= 'r'
    ###
    types= ['PSF','SIMP','EXP','DEV','COMP']
    ind = np.arange(len(types))  # the x locations for the groups
    width = 0.35       # the width of the bars
    ###
    ht_decam, ht_bokmos= np.zeros(5,dtype=int),np.zeros(5,dtype=int)
    for cnt,typ in enumerate(types):
        ht_decam[cnt]= np.where(obj[m_types[0]].data['type'] == typ)[0].shape[0]
        ht_bokmos[cnt]= np.where(obj[m_types[1]].data['type'] == typ)[0].shape[0]
    ###
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, ht_decam, width, color=c1)
    rects2 = ax.bar(ind + width, ht_bokmos, width, color=c2)
    ylab= ax.set_ylabel("N")
    if matched: ti= ax.set_title('Matched')
    else: ti= ax.set_title('Unmatched')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(types)
    ax.legend((rects1[0], rects2[0]), ('decam', 'bokmos'),**leg_args)
    #save
    if matched: name='hist_types_Matched.png'
    else: name='hist_types_Unmatched.png'
    plt.savefig(name, bbox_extra_artists=[ylab,ti], bbox_inches='tight',dpi=150)
    plt.close()

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='DECaLS simulations.')
parser.add_argument('-fn1', type=str, help='process this brick (required input)')
parser.add_argument('-fn2', type=str, help='object type (STAR, ELG, LRG, BGS)') 

args = parser.parse_args()

#get lists of tractor cats to compare
fns_1= read_lines(args.fn1) 
fns_2= read_lines(args.fn2) 
#if fns_1.size == 1: fns_1,fns_2= [fns_1],[fns_2]
#object to store concatenated matched tractor cats
a=Matched_Cats()
for cnt,cat1,cat2 in zip(range(len(fns_1)),fns_1,fns_2):
    data_1,data_2,m1,m2,m1_unm,m2_unm= match_it(cat1,cat2)
    if cnt == 0:
        a.initialize(data_1,data_2,m1,m2,m1_unm,m2_unm)
    else:  
        #a.add_dict('all_1', targets.data_extract(data_1,range(len(data_1['ra'])) ))
        a.add_dict('m_decam', targets.data_extract(data_1,m1) )
        a.add_dict('m_bokmos', targets.data_extract(data_2,m2))
        a.add_dict('u_decam', targets.data_extract(data_1,m1_unm))
        a.add_dict('u_bokmos', targets.data_extract(data_2,m2_unm))
#each key a.data[key] becomes DECaLS() object with grz mags,i_lrg, etc
b={}
for match_type in a.data.keys(): b[match_type]= targets.DECaLS(a.data[match_type], w1=True)
#plots
plot_SN(b,m_types=['m_decam','m_bokmos'], index='all')
plot_SN(b,m_types=['m_decam','m_bokmos'], index='psf')
plot_SN(b,m_types=['m_decam','m_bokmos'], index='lrg')
plot_SN(b,m_types=['u_decam','u_bokmos'], index='psf')
plot_SN(b,m_types=['u_decam','u_bokmos'], index='lrg')

plot_HistTypes(b,m_types=['m_decam','m_bokmos'])
plot_HistTypes(b,m_types=['u_decam','u_bokmos'])
