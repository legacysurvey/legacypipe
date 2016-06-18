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
from astrometry.libkd.spherematch import match_radec

#import thesis_code.targets as targets
from legacyanalysis import targets 
from legacyanalysis.pathnames import get_outdir


class Matched_Cats():
    def __init__(self):
        self.data={}
    def initialize(self,data_1,data_2,m1,m2,m1_unm,m2_unm,d12, deg2_decam,deg2_bokmos):
        self.d12= d12 #deg separations between matches objects
        self.deg2_decam= deg2_decam 
        self.deg2_bokmos= deg2_bokmos 
        self.data['m_decam']= targets.data_extract(data_1,m1) 
        self.data['m_bokmos']= targets.data_extract(data_2,m2)
        self.data['u_decam']= targets.data_extract(data_1,m1_unm)
        self.data['u_bokmos']= targets.data_extract(data_2,m2_unm)
    def add_d12(self,d12):
        '''concatenate new d12 with existing matched deg separation array'''
        self.d12= np.concatenate([self.d12, d12])
    def add_dict(self,match_type,new_data):
        '''match_type -- m_decam,m_bokmos,u_decam, etc
        new data -- data returend from read_from..() to be concatenated with existing m_decam, etc'''
        for key in self.data[match_type].keys(): 
            self.data[match_type][key]= np.concatenate([self.data[match_type][key],new_data[key]])

def deg2_lower_limit(data):
    '''deg2 spanned by objects in each data set, lower limit'''
    ra= data['ra'].max()-data['ra'].min()
    assert(ra > 0.)
    dec= abs(data['dec'].max()-data['dec'].min())
    return ra*dec

def match_it(cat1,cat2):
    '''cat1,2 are tractor catalogue to match objects between'''
    #match cats
    data_1= targets.read_from_tractor_cat(cat1)
    data_2= targets.read_from_tractor_cat(cat2)
    #deg2 spanned by objects in each data set
    deg2_decam= deg2_lower_limit(data_1)
    deg2_bokmos= deg2_lower_limit(data_2)
    #all the 'all1' objects that have match in 'all2' 
    m1, m2, d12 = match_radec(data_1['ra'],data_1['dec'],data_2['ra'],data_2['dec'],\
                            1.0/3600.0,nearest=True)
    m1_unm = np.delete(np.arange(len(data_1['ra'])),m1,axis=0)
    m2_unm = np.delete(np.arange(len(data_2['ra'])),m2,axis=0)
    return data_1,data_2,m1,m2,m1_unm,m2_unm,d12, deg2_decam,deg2_bokmos

def read_lines(fn):
    fin=open(fn,'r')
    lines=fin.readlines()
    fin.close()
    return list(np.char.strip(lines))

#plotting vars
laba=dict(fontweight='bold',fontsize='medium')
kwargs_axtext=dict(fontweight='bold',fontsize='large',va='top',ha='left')
leg_args=dict(frameon=True,fontsize='x-small')

def plot_nobs(b):
    '''make histograms of nobs so can compare depths of g,r,z between the two catalogues'''   
    hi=0 
    for cam in ['m_decam','m_bokmos']:
        for band in 'grz':
            hi= np.max((hi, b[cam].data[band+'_nobs'].max()))
    bins= hi
    for cam in ['m_decam','m_bokmos']:
        for band in 'grz':
            junk=plt.hist(b[cam].data[band+'_nobs'],bins=bins,normed=True,cumulative=True,align='mid')
            xlab=plt.xlabel('nobs %s' % band, **laba)
            ylab=plt.ylabel('CDF', **laba)
            plt.savefig(os.path.join(get_outdir('bmd'),'hist_nobs_%s_%s.png' % (band,cam[2:])), bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()

#def plot_nobs_2(b):
#    '''improved version of plot_nobs'''
#    for cam in ['m_decam','m_bokmos']:
#        for band in 'grz':
#            junk=plt.hist(b[cam].data[band+'_nobs'],bins=10,normed=True,cumulative=True,align='mid')
#            xlab=plt.xlabel('nobs %s' % band, **laba)
#            ylab=plt.xlabel('CDF', **laba)
#            plt.savefig(os.path.join(get_outdir('bmd'),'hist_nobs_%s_%s.png' % (band,cam[2:])), bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
#            plt.close()
#
#
#    c1= 'b' 
#    c2= 'r'
#    ###
#    decam_max= [b['m_decam'].data[b+'_nobs'].max() for b in 'grz']
#    bokmos_max= [b['m_bokmos'].data[b+'_nobs'].max() for b in 'grz']
#    types= np.arange(1, np.max((decam_max,bokmos_max)) +1)
#    ind = types.copy() # the x locations for the groups
#    width = 1       # the width of the bars
#    ###
#    ht_decam, ht_bokmos= np.zeros(5,dtype=int),np.zeros(5,dtype=int)
#    for cnt,typ in enumerate(types):
#        ht_decam[cnt]= np.where(obj[m_types[0]].data['type'] == typ)[0].shape[0] / float(obj['deg2_decam'])
#        ht_bokmos[cnt]= np.where(obj[m_types[1]].data['type'] == typ)[0].shape[0] / float(obj['deg2_bokmos'])
#    ###
#    fig, ax = plt.subplots()
#    rects1 = ax.bar(ind, ht_decam, width, color=c1)
#    rects2 = ax.bar(ind + width, ht_bokmos, width, color=c2)
#    ylab= ax.set_ylabel("counts/deg2")
#    if matched: ti= ax.set_title('Matched')
#    else: ti= ax.set_title('Unmatched')
#    ax.set_xticks(ind + width)
#    ax.set_xticklabels(types)
#    ax.legend((rects1[0], rects2[0]), ('decam', 'bokmos'),**leg_args)
#    #save
#    if matched: name='hist_types_Matched.png'
#    else: name='hist_types_Unmatched.png'
#    plt.savefig(os.path.join(get_outdir('bmd'),name), bbox_extra_artists=[ylab,ti], bbox_inches='tight',dpi=150)
#    plt.close()
#

def plot_radec(obj, addname=''): 
    '''obj[m_types] -- DECaLS() objects with matched OR unmatched indices'''
    #set seaborn panel styles
    #sns.set_style('ticks',{"axes.facecolor": ".97"})
    #sns.set_palette('colorblind')
    #setup plot
    fig,ax=plt.subplots(1,2,figsize=(9,6),sharey=True,sharex=True)
    plt.subplots_adjust(wspace=0.25)
    #plt.subplots_adjust(wspace=0.5)
    #plot
    ax[0].scatter(obj['m_decam'].data['ra'], obj['m_decam'].data['dec'], \
                edgecolor='b',c='none',lw=1.)
    ax[1].scatter(obj['u_decam'].data['ra'], obj['u_decam'].data['dec'], \
                edgecolor='b',c='none',lw=1.,label='DECaLS')
    ax[1].scatter(obj['u_bokmos'].data['ra'], obj['u_bokmos'].data['dec'], \
                edgecolor='g',c='none',lw=1.,label='BASS/MzLS')
    for cnt,ti in zip(range(2),['Matched','Unmatched']):
        ti=ax[cnt].set_title(ti,**laba)
        xlab=ax[cnt].set_xlabel('RA', **laba)
    ylab=ax[0].set_ylabel('DEC', **laba)
    ax[0].legend(loc='upper left',**leg_args)
    #save
    #sns.despine()
    plt.savefig(os.path.join(get_outdir('bmd'),'radec%s.png' % addname), bbox_extra_artists=[xlab,ylab,ti], bbox_inches='tight',dpi=150)
    plt.close()


def plot_HistTypes(obj,m_types=['m_decam','m_bokmos'], addname=''):
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
        ht_decam[cnt]= np.where(obj[m_types[0]].data['type'] == typ)[0].shape[0] / float(obj['deg2_decam'])
        ht_bokmos[cnt]= np.where(obj[m_types[1]].data['type'] == typ)[0].shape[0] / float(obj['deg2_bokmos'])
    ###
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, ht_decam, width, color=c1)
    rects2 = ax.bar(ind + width, ht_bokmos, width, color=c2)
    ylab= ax.set_ylabel("counts/deg2")
    if matched: ti= ax.set_title('Matched')
    else: ti= ax.set_title('Unmatched')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(types)
    ax.legend((rects1[0], rects2[0]), ('DECaLS', 'BASS/MzLS'),**leg_args)
    #save
    if matched: name='hist_types_Matched%s.png' % addname
    else: name='hist_types_Unmatched%s.png' % addname
    plt.savefig(os.path.join(get_outdir('bmd'),name), bbox_extra_artists=[ylab,ti], bbox_inches='tight',dpi=150)
    plt.close()


def bin_up(data_bin_by,data_for_percentile, bin_edges=np.arange(20.,26.,0.25)):
    '''finds indices for 0.25 bins, returns bin centers and q25,50,75 percentiles of data_percentile in each bin
    bin_edges: compute percentiles for each sample between bin_edges
    '''
    count= np.zeros(len(bin_edges)-1)+np.nan
    q25,q50,q75= count.copy(),count.copy(),count.copy()
    for i,low,hi in zip(range(len(count)), bin_edges[:-1],bin_edges[1:]):
        ind= np.all((low <= data_bin_by,data_bin_by < hi),axis=0)
        if np.where(ind)[0].size > 0:
            count[i]= np.where(ind)[0].size
            q25[i]= np.percentile(data_for_percentile[ind],q=25)
            q50[i]= np.percentile(data_for_percentile[ind],q=50)
            q75[i]= np.percentile(data_for_percentile[ind],q=75)
        else:
            pass #given qs nan, which they already have
    return (bin_edges[1:]+bin_edges[:-1])/2.,count,q25,q50,q75

def indices_for_type(obj,inst='m_decam',type='all'):
    '''return mask for selecting type == all,psf,lrg
    data -- obj['m_decam'].data
    lrg mask -- obje['m_decam'].lrg'''
    if type == 'all': 
        return np.ones(obj[inst].data['type'].size, dtype=bool) #1 = True
    elif type == 'psf': 
        return obj[inst].data['type'] == 'PSF'
    elif type == 'lrg': 
        return obj[inst].lrg
    else: raise ValueError


def plot_SN_vs_mag(obj, found_by='matched',type='all', addname=''):
    '''obj['m_decam'] is DECaLS() object
    found_by -- 'matched' or 'unmatched' 
    type -- all,psf,lrg'''
    #indices for type == all,psf, or lrg
    assert(found_by == 'matched' or found_by == 'unmatched')
    prefix= found_by[0]+'_' # m_ or u_
    index={}
    for key in ['decam','bokmos']:
        index[key]= indices_for_type(obj,inst=prefix+key,type=type)
    #bin up SN values
    min,max= 18.,25.
    bin_SN=dict(decam={},bokmos={})
    for key in bin_SN.keys():
        for band in ['g','r','z']:
            bin_SN[key][band]={}
            i= index[key]
            bin_edges= np.linspace(min,max,num=30)
            bin_SN[key][band]['binc'],count,bin_SN[key][band]['q25'],bin_SN[key][band]['q50'],bin_SN[key][band]['q75']=\
                    bin_up(obj[prefix+key].data[band+'mag'][i], \
                           obj[prefix+key].data[band+'flux'][i]*np.sqrt(obj[prefix+key].data[band+'flux_ivar'][i]),\
                                bin_edges=bin_edges)
    #setup plot
    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
    plt.subplots_adjust(wspace=0.25)
    #plot SN
    for cnt,band in zip(range(3),['g','r','z']):
        #horiz line at SN = 5
        ax[cnt].plot([1,40],[5,5],'k--',lw=2)
        #data
        for inst,color,lab in zip(['decam','bokmos'],['b','g'],['DECaLS','BASS/MzLS']):
            ax[cnt].plot(bin_SN[inst][band]['binc'], bin_SN[inst][band]['q50'],c=color,ls='-',lw=2,label=lab)
            ax[cnt].fill_between(bin_SN[inst][band]['binc'],bin_SN[inst][band]['q25'],bin_SN[inst][band]['q75'],color=color,alpha=0.25)
    #labels
    ax[2].legend(loc=1,**leg_args)
    for cnt,band in zip(range(3),['g','r','z']):
        ax[cnt].set_yscale('log')
        xlab=ax[cnt].set_xlabel('%s' % band, **laba)
        ax[cnt].set_ylim(1,100)
        ax[cnt].set_xlim(20.,26.)
    ylab=ax[0].set_ylabel('S/N', **laba)
    text_args= dict(verticalalignment='bottom',horizontalalignment='right',fontsize=10)
    ax[2].text(26,5,'S/N = 5  ',**text_args)
    plt.savefig(os.path.join(get_outdir('bmd'),'sn_%s_%s%s.png' % (found_by,type,addname)), bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()

def plot_matched_dmag_vs_psf_fwhm(obj, type='psf'):
    '''using matched sample, plot diff in mags vs. DECAM psf_fwhm in bins 
    obj['m_decam'] is DECaLS() object'''
    #indices
    index= np.all((indices_for_type(b,inst='m_decam',type=type),\
                    indices_for_type(b,inst='m_bokmos',type=type)), axis=0) #both bokmos and decam of same type
    #bin up by DECAM psf_fwhm
    bin_edges= np.linspace(0,3,num=6)
    vals={}
    for band in ['g','r','z']:
        vals[band]={}
        vals[band]['binc'],count,vals[band]['q25'],vals[band]['q50'],vals[band]['q75']=\
                bin_up(obj['m_decam'].data[band+'_psf_fwhm'][index], \
                       obj['m_bokmos'].data[band+'mag'][index]- obj['m_decam'].data[band+'mag'][index], \
                            bin_edges=bin_edges)
#setup plot
    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
    plt.subplots_adjust(wspace=0.25)
    text_args= dict(verticalalignment='center',horizontalalignment='left',fontsize=10)
    #plot
    for cnt,band in zip(range(3),['g','r','z']):
        ax[cnt].plot(vals[band]['binc'], vals[band]['q50'],c='b',ls='-',lw=2)
        ax[cnt].fill_between(vals[band]['binc'],vals[band]['q25'],vals[band]['q75'],color='b',alpha=0.25)
        ax[cnt].text(0.05,0.95,band,transform=ax[cnt].transAxes,**text_args)
    #finish
    xlab=ax[1].set_xlabel('decam PSF_FWHM', **laba)
    ylab=ax[0].set_ylabel(r'Median $\Delta \, m$ (decam - bokmos)', **laba)
    ti= plt.suptitle('%s Objects, Matched' % type.upper())
    plt.savefig(os.path.join(get_outdir('bmd'),'dmag_vs_psf_fwhm_%s.png' % type), bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()

def plot_matched_decam_vs_bokmos_psf_fwhm(obj, type='psf'):
    '''using matched sample, plot decam psf_fwhm vs. bokmos psf_fwhm 
    obj['m_decam'] is DECaLS() object'''
    #indices
    index= np.all((indices_for_type(b,inst='m_decam',type=type),\
                    indices_for_type(b,inst='m_bokmos',type=type)), axis=0) #both bokmos and decam of same type
    #setup plot
    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
    plt.subplots_adjust(wspace=0.25)
    text_args= dict(verticalalignment='center',horizontalalignment='left',fontsize=10)
    #plot
    for cnt,band in zip(range(3),['g','r','z']):
        ax[cnt].scatter(obj['m_bokmos'].data[band+'_psf_fwhm'][index], obj['m_decam'].data[band+'_psf_fwhm'][index],\
                        edgecolor='b',c='none',lw=1.)
        ax[cnt].text(0.05,0.95,band,transform=ax[cnt].transAxes,**text_args)
    #finish
    for cnt,band in zip(range(3),['g','r','z']):
        ax[cnt].set_xlim(0,3)
        ax[cnt].set_ylim(0,3)
    xlab=ax[1].set_xlabel('PSF_FWHM (bokmos)', **laba)
    ylab=ax[0].set_ylabel('PSF_FWHM (decam)', **laba)
    ti= plt.suptitle('%s Objects, Matched' % type.upper())
    plt.savefig(os.path.join(get_outdir('bmd'),'decam_vs_bokmos_psf_fwhm_%s.png' % type), bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()



def plot_confusion_matrix(cm,ticknames, addname=''):
    '''cm -- NxN array containing the Confusion Matrix values
    ticknames -- list of strings of length == N, column and row names for cm plot'''
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues,vmin=0,vmax=1)
    cbar=plt.colorbar()
    plt.xticks(range(len(ticknames)), ticknames)
    plt.yticks(range(len(ticknames)), ticknames)
    ylab=plt.ylabel('True (DECaLS)')
    xlab=plt.xlabel('Predicted (BASS/MzLS)')
    for row in range(len(ticknames)):
        for col in range(len(ticknames)):
            if np.isnan(cm[row,col]):
                plt.text(col,row,'n/a',va='center',ha='center')
            elif cm[row,col] > 0.5:
                plt.text(col,row,'%.2f' % cm[row,col],va='center',ha='center',color='yellow')
            else:
                plt.text(col,row,'%.2f' % cm[row,col],va='center',ha='center',color='black')
    plt.savefig(os.path.join(get_outdir('bmd'),'confusion_matrix%s.png' % addname), bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()

def create_confusion_matrix(obj):
    '''compares MATCHED decam (truth) to bokmos (prediction)
    return 5x5 confusion matrix and colum/row names
    obj[m_decam'] is DECaLS object'''
    cm=np.zeros((5,5))-1
    types=['PSF','SIMP','EXP','DEV','COMP']
    for i_dec,dec_type in enumerate(types):
        ind= np.where(obj['m_decam'].data['type'] == dec_type)[0]
        for i_bass,bass_type in enumerate(types):
            n_bass= np.where(obj['m_bokmos'].data['type'][ind] == bass_type)[0].size
            if ind.size > 0: cm[i_dec,i_bass]= float(n_bass)/ind.size #ind.size is constant for each loop over bass_types
            else: cm[i_dec,i_bass]= np.nan
    return cm,types

def plot_matched_separation_hist(d12):
    '''d12 is array of distances in degress between matched objects'''
    #pixscale to convert d12 into N pixels
    pixscale=dict(decam=0.25,bokmos=0.45)
    #sns.set_style('ticks',{"axes.facecolor": ".97"})
    #sns.set_palette('colorblind')
    #setup plot
    fig,ax=plt.subplots()
    #plot
    ax.hist(d12*3600,bins=50,color='b',align='mid')
    ax2 = ax.twiny()
    ax2.hist(d12*3600./pixscale['bokmos'],bins=50,color='g',align='mid',visible=False)
    xlab= ax.set_xlabel("arcsec")
    xlab= ax2.set_xlabel("pixels [BASS]")
    ylab= ax.set_ylabel("Matched")
    #save
    #sns.despine()
    plt.savefig(os.path.join(get_outdir('bmd'),"separation_hist.png"), bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()

def plot_psf_hists(decam,bokmos, zoom=False):
    '''decam,bokmos are DECaLS() objects matched to decam ra,dec'''
    #divide into samples of 0.25 mag bins, store q50 of each
    width=0.25 #in mag
    low_vals= np.arange(20.,26.,width)
    med={}
    for b in ['g','r','z']: med[b]=np.zeros(low_vals.size)-100
    for i,low in enumerate(low_vals):
        for band in ['g','r','z']:
            ind= np.all((low <= decam[band+'mag'],decam[band+'mag'] < low+width),axis=0)
            if np.where(ind)[0].size > 0:
                med[band][i]= np.percentile(bokmos[band+'mag'][ind] - decam[band+'mag'][ind],q=50)
            else: 
                med[band][i]= np.nan
    #make plot
    #set seaborn panel styles
    #sns.set_style('ticks',{"axes.facecolor": ".97"})
    #sns.set_palette('colorblind')
    #setup plot
    fig,ax=plt.subplots(1,3,figsize=(9,3)) #,sharey=True)
    plt.subplots_adjust(wspace=0.5)
    #plot
    for cnt,band in zip(range(3),['r','g','z']):
        ax[cnt].scatter(low_vals, med[band],\
                       edgecolor='b',c='none',lw=2.) #,label=m_type.split('_')[-1])
        xlab=ax[cnt].set_xlabel('bins of %s (decam)' % band, **laba)
        ylab=ax[cnt].set_ylabel('q50[%s bokmos - decam]' % band, **laba)
        if zoom: ax[cnt].set_ylim(-0.25,0.25)
    # sup=plt.suptitle('decam with matching bokmos',**laba)
    #save
    #sns.despine()
    if zoom: name="median_color_diff_zoom.png"
    else: name="median_color_diff.png"
    plt.savefig(os.path.join(get_outdir('bmd'),name), bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()

##########
#funcs for flux diff / sqrt(inv var + inv var)
def n_gt_3_sigma(sample, low=-8.,hi=8.):
    '''for a sample that should be distributed as N(mean=0,stddev=1), returns mask for the N that are greater 3 sigma
    low,hi -- minimum and maximum sample values that will be considered'''
    i_left= np.all((sample >= low,sample <= -3.),axis=0)
    i_right= np.all((sample <= hi,sample>=3),axis=0)
    #assert i_left and i_right are mutually exclusive
    false_arr= np.all((i_left,i_right),axis=0) #should be array of Falses
    assert( np.all(false_arr == False) ) #should be np.all([True,True,...]) which evaluates to True
    return np.any((i_left,i_right),axis=0)

def gauss_stats(n_samples=10000):
    '''returns mean,std,q25, frac outliers > 3 sigma for n_samples drawn from unit gaussian N(0,1)'''
    G= sp_stats.norm(0,1)
    mean=std=q25=perc_out=0.
    for i in range(10): #draw 10 times, take avg of the 10 measurements of each statistic
        draws= G.rvs(n_samples) 
        mean+= np.mean(draws)
        std+= np.std(draws)
        q25+= np.percentile(draws,q=25)
        perc_out+= 2*G.cdf(-3)*100 #HACH same number ea time
    mean/= 10.
    std/= 10.
    q25/= 10.
    perc_out/= 10.
    tol=1e-1
    assert(abs(mean) <= tol)
    assert(abs(std-1.) <= tol)
    return mean,std,q25,perc_out

def sample_gauss_stats(sample, low=-20,hi=20):
    '''return dictionary of stats about the data and stats for a sample that is unit gaussian distributed
    low,hi -- minimum and maximum sample values that will be considered'''
    a=dict(sample={},gauss={})
    #vals for unit gaussian distributed data
    a['gauss']['mean'],a['gauss']['std'],a['gauss']['q25'],a['gauss']['perc_out']= gauss_stats(n_samples=sample.size)
    #vals for actual sample
    a['sample']['mean'],a['sample']['std'],a['sample']['q25'],a['sample']['q75']= \
            np.mean(sample),np.std(sample),np.percentile(sample,q=25),np.percentile(sample,q=75) 
    i_outliers= n_gt_3_sigma(sample, low=low,hi=hi)
    a['sample']['perc_out']= sample[i_outliers].size/float(sample.size)*100.
    return a


text_args= dict(verticalalignment='center',fontsize=8)
def plot_dflux_chisq(b,type='psf', low=-8.,hi=8.,addname=''):
    #join indices b/c matched
    i_type= np.all((indices_for_type(b, inst='m_decam',type=type),\
                    indices_for_type(b, inst='m_bokmos',type=type)), axis=0) #both bokmos and decam of same type
    #get flux diff for each band
    hist= dict(g=0,r=0,z=0)
    binc= dict(g=0,r=0,z=0)
    stats=dict(g=0,r=0,z=0) 
    for band in ['g','r','z']:
        sample=(b['m_decam'].data[band+'flux'][i_type]-b['m_bokmos'].data[band+'flux'][i_type])/np.sqrt(\
                    np.power(b['m_decam'].data[band+'flux_ivar'][i_type],-1)+np.power(b['m_bokmos'].data[band+'flux_ivar'][i_type],-1))
        hist[band],bins,junk= plt.hist(sample,range=(low,hi),bins=50,normed=True)
        db= (bins[1:]-bins[:-1])/2
        binc[band]= bins[:-1]+db
        stats[band]= sample_gauss_stats(sample)
    plt.close() #b/c plt.hist above
    #for drawing unit gaussian N(0,1)
    G= sp_stats.norm(0,1)
    xvals= np.linspace(low,hi)
    #plot
    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
    plt.subplots_adjust(wspace=0.25)
    for cnt,band in zip(range(3),['g','r','z']):
        ax[cnt].step(binc[band],hist[band], where='mid',c='b',lw=2)
        ax[cnt].plot(xvals,G.pdf(xvals))
        #for yloc,key in zip([0.95,0.85,0.75,0.65,0.55],['mean','std','q25','q75','perc_out']):
        #    ax[cnt].text(0.1,yloc,"%s %.2f" % (key,stats[band]['sample'][key]),transform=ax[cnt].transAxes,horizontalalignment='left',**text_args)
    #ax[2].text(0.9,0.95,"N(0,1)",transform=ax[2].transAxes,horizontalalignment='right',**text_args)
    #for yloc,key in zip([0.85,0.75,0.65,0.55],['mean','std','q25','perc_out']):
    #    ax[2].text(0.9,yloc,"%s %.2f" % (key,stats['g']['gauss'][key]),transform=ax[2].transAxes,horizontalalignment='right',**text_args)
    #labels
    for cnt,band in zip(range(3),['g','r','z']):
        if band == 'r': xlab=ax[cnt].set_xlabel(r'%s  $(F_{d}-F_{bm})/\sqrt{\sigma^2_{d}+\sigma^2_{bm}}$' % band, **laba)
        else: xlab=ax[cnt].set_xlabel('%s' % band, **laba)
        #xlab=ax[cnt].set_xlabel('%s' % band, **laba)
        ax[cnt].set_ylim(0,0.6)
        ax[cnt].set_xlim(low,hi)
    ylab=ax[0].set_ylabel('PDF', **laba)
    ti=ax[1].set_title(type,**laba)
    #put stats in suptitle
    plt.savefig(os.path.join(get_outdir('bmd'),'dflux_chisq_%s%s.png' % (type,addname)), bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()
################

text_args= dict(verticalalignment='center',fontsize=8)
def plot_N_per_deg2(obj,type='all',req_mags=[24.,23.4,22.5],addname=''):
    '''image requirements grz<=24,23.4,22.5
    compute number density in each bin for each band mag [18,requirement]'''
    #indices for type for matched and unmatched samples
    index={}
    for inst in ['m_decam','u_decam','m_bokmos','u_bokmos']:
        index[inst]= indices_for_type(obj, inst=inst,type=type) 
    bin_nd=dict(decam={},bokmos={})
    for inst in ['decam','bokmos']:
        bin_nd[inst]={}
        for band,req in zip(['g','r','z'],req_mags):
            bin_nd[inst][band]={}
            bin_edges= np.linspace(18.,req,num=15)
            i_m,i_u= index['m_'+inst], index['u_'+inst] #need m+u
            #join m_decam,u_decam OR m_bokmos,u_bokmos and only with correct all,psf,lrg index
            sample= np.ma.concatenate((obj['m_'+inst].data[band+'mag'][i_m], obj['u_'+inst].data[band+'mag'][i_u]),axis=0)
            bin_nd[inst][band]['binc'],bin_nd[inst][band]['cnt'],q25,q50,q75=\
                    bin_up(sample,sample,bin_edges=bin_edges)
    #plot
    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
    plt.subplots_adjust(wspace=0.25)
    for cnt,band in zip(range(3),['g','r','z']):
        for inst,color,lab in zip(['decam','bokmos'],['b','g'],['DECaLS','BASS/MzLS']):
            ax[cnt].step(bin_nd[inst][band]['binc'],bin_nd[inst][band]['cnt']/obj['deg2_'+inst], where='mid',c=color,lw=2,label=lab)
    #labels
    for cnt,band in zip(range(3),['g','r','z']):
        xlab=ax[cnt].set_xlabel('%s' % band) #, **laba)
        #ax[cnt].set_ylim(0,0.6)
        #ax[cnt].set_xlim(maglow,maghi)
    ax[0].legend(loc='upper left', **leg_args)
    ylab=ax[0].set_ylabel('counts/deg2') #, **laba)
    ti=plt.suptitle("%ss" % type.upper(),**laba)
    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()
    #put stats in suptitle
    plt.savefig(os.path.join(get_outdir('bmd'),'n_per_deg2_%s%s.png' % (type,addname)), bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()


parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description='DECaLS simulations.')
parser.add_argument('-fn1', type=str, help='process this brick (required input)',required=True)
parser.add_argument('-fn2', type=str, help='object type (STAR, ELG, LRG, BGS)',required=True) 
args = parser.parse_args()

#get lists of tractor cats to compare
fns_1= read_lines(args.fn1) 
fns_2= read_lines(args.fn2) 
#if fns_1.size == 1: fns_1,fns_2= [fns_1],[fns_2]
#object to store concatenated matched tractor cats
a=Matched_Cats()
for cnt,cat1,cat2 in zip(range(len(fns_1)),fns_1,fns_2):
    data_1,data_2,m1,m2,m1_unm,m2_unm,d12, deg2_decam,deg2_bokmos= match_it(cat1,cat2)
    if cnt == 0:
        a.initialize(data_1,data_2,m1,m2,m1_unm,m2_unm,d12, deg2_decam,deg2_bokmos)
    else:  
        a.add_d12(d12)
        a.deg2_decam+= deg2_decam
        a.deg2_bokmos+= deg2_bokmos
        a.add_dict('m_decam', targets.data_extract(data_1,m1) )
        a.add_dict('m_bokmos', targets.data_extract(data_2,m2))
        a.add_dict('u_decam', targets.data_extract(data_1,m1_unm))
        a.add_dict('u_bokmos', targets.data_extract(data_2,m2_unm))
#each key a.data[key] becomes DECaLS() object with grz mags,i_lrg, etc
b={}
b['d12']= a.d12
b['deg2_decam']= a.deg2_decam
b['deg2_bokmos']= a.deg2_bokmos
for match_type in a.data.keys(): b[match_type]= targets.DECaLS(a.data[match_type], w1=True)
#store N matched objects not masked before join decam,bokmos masks
m_decam_not_masked,m_bokmos_not_masked= b['m_decam'].count_not_masked(),b['m_bokmos'].count_not_masked()
#update masks for matched objects to be the join of decam and bokmos masks
mask= np.any((b['m_decam'].mask, b['m_bokmos'].mask),axis=0)
b['m_decam'].update_masks_for_everything(mask=np.any((b['m_decam'].mask, b['m_bokmos'].mask),axis=0),\
                                    mask_wise=np.any((b['m_decam'].mask_wise, b['m_bokmos'].mask_wise),axis=0) )
b['m_bokmos'].update_masks_for_everything(mask=np.any((b['m_decam'].mask, b['m_bokmos'].mask),axis=0),\
                                    mask_wise=np.any((b['m_decam'].mask_wise, b['m_bokmos'].mask_wise),axis=0) )

#plots
plot_radec(b)
plot_matched_separation_hist(b['d12'])
# Depths are very different so develop a cut to make fair comparison
plot_SN_vs_mag(b, found_by='matched',type='psf')
# mask=True where BASS SN g < 5 or BASS SN r < 5
sn_crit=5.
mask= np.any((b['m_bokmos'].data['gflux']*np.sqrt(b['m_bokmos'].data['gflux_ivar']) < sn_crit,\
              b['m_bokmos'].data['rflux']*np.sqrt(b['m_bokmos'].data['rflux_ivar']) < sn_crit),\
                axis=0)
b['m_decam'].update_masks_for_everything(mask=mask, mask_wise=mask)
b['m_bokmos'].update_masks_for_everything(mask=mask, mask_wise=mask)
# contintue with fairer comparison
plot_radec(b,addname='snGe5')
plot_HistTypes(b,m_types=['m_decam','m_bokmos'],addname='snGe5')
plot_SN_vs_mag(b, found_by='matched',type='psf',addname='snGe5')
#plot_SN_vs_mag(b, found_by='matched',type='all')
#plot_SN_vs_mag(b, found_by='matched',type='lrg')
#plot_SN_vs_mag(b, found_by='unmatched',type='all')
#plot_SN_vs_mag(b, found_by='unmatched',type='psf')
#plot_SN_vs_mag(b, found_by='unmatched',type='lrg')
cm,names= create_confusion_matrix(b)
plot_confusion_matrix(cm,names,addname='snGe5')
plot_dflux_chisq(b,type='all',addname='snGe5')
plot_dflux_chisq(b,type='psf',addname='snGe5')
# Number density cutting to requirement mags: grz<=24,23.4,22.5
print('square deg covered by decam=',b['deg2_decam'],'and by bokmos=',b['deg2_bokmos'])
plot_N_per_deg2(b,type='psf',addname='snGe5')
plot_N_per_deg2(b,type='lrg',addname='snGe5')





print('exiting early')
sys.exit()

plot_matched_dmag_vs_psf_fwhm(b, type='psf')
plot_matched_decam_vs_bokmos_psf_fwhm(b, type='psf')

print('finished comparison: bass-mosaic-decals')
#sys.exit()
#
#
##REVISE THIS BELOW
##print stats of total objects, each group, # masked, etc
#print("---- DECAM ----")
#print("N not masked due to grz= %d, N total= %d" % \
#        (m_decam_not_masked+b['u_decam'].count_not_masked(), b['m_decam'].count_total()+b['u_decam'].count_total()))
#print("-- Matched --")
#print("N not masked before join bokmos mask= %d, N not masked after= %d" % \
#        (m_decam_not_masked, b['m_decam'].count_not_masked()))
#print("-- Unmatched -- ")
#print("N masked before join bokmos mask = N masked after = %d" % \
#        (b['u_decam'].count_total()- b['u_decam'].count_not_masked()))
####bokmos
#print("---- BOKMOS ----")
#print("N not masked due to grz= %d, N total= %d" % \
#        (m_bokmos_not_masked+b['u_bokmos'].count_not_masked(), b['m_bokmos'].count_total()+b['u_bokmos'].count_total()))
#print("-- Matched --")
#print("N not masked before join decam mask= %d, N not masked after= %d" % \
#        (m_bokmos_not_masked, b['m_bokmos'].count_not_masked()))
#print("-- Unmatched -- ")
#print("N masked before join decam mask = N masked after = %d" % \
#        (b['u_bokmos'].count_total()- b['u_bokmos'].count_not_masked()))
#print('done')


