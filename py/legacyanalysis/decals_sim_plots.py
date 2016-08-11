#!/usr/bin/env python

"""Analyze the output of decals_simulations.

EXAMPLE
=======
8 500 star chunks for brick 2523p355 are here 
/project/projectdirs/desi/image_sims/2523p355
you can analyze them like this:
export DECALS_SIM_DIR=/project/projectdirs/desi/image_sims 
python legacyanalysis/decals_sim_plots.py -b 2523p355 -o STAR -out your/relative/output/path
out is optional, default is brickname/objtype

Missing object and annotated coadd plots
========================================
python legacyanalysis/decals_sim_plots.py ... --extra_plots
default is to NOT make them because chunks > 50
"""
from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg') # display backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.image as mpimg

import os
import sys
import pdb
import logging
import argparse
import glob
import numpy as np

from astropy.io import fits
from astropy.table import vstack, Table
from astropy import units
from astropy.coordinates import SkyCoord

# import seaborn as sns
from PIL import Image, ImageDraw
import photutils

def bright_dmag_cut(matched_simcat,matched_tractor):
    '''return: 
    1) median dmag of bright sources
    2) indices of sources that are bright AND |dmag[band]| -|median_dmag[band]| > 0.005 in ANY band
    3) indices of sources that are bright AND |dmag[band]| -|median_dmag[band]| <= 0.005 in ALL bands'''
    bright= dict(G=20.,R=19.,Z=18.)
    b_good,junk= basic_cut(matched_tractor)
    # Store median dmag of bright sources
    med,b_bright={},{}
    for band,iband in zip(['G','R','Z'],[1,2,4]):
        # Cut to bright and good sources
        inputflux = matched_simcat[band+'FLUX']
        inputmag = 22.5-2.5*np.log10(inputflux)
        b_bright[band]= inputmag < bright[band]
        b_bright[band]= np.all((b_bright[band],b_good),axis=0)
        #i_bright= np.where(b_bright)[0]
        #print('type(i_bright)= ',type(i_bright),"i_bright=",i_bright)
        # Compute median for each band
        inputflux = matched_simcat[band+'FLUX'][b_bright[band]]
        tractorflux = matched_tractor['decam_flux'][:, iband][b_bright[band]]
        mag_diff= -2.5*np.log10(tractorflux/inputflux)
        med[band]= np.percentile(mag_diff,q=50)
    # Boolean mask for each band
    b={}
    for band,iband in zip(['G','R','Z'],[1,2,4]):
        inputflux = matched_simcat[band+'FLUX']
        tractorflux = matched_tractor['decam_flux'][:, iband]
        mag_diff= -2.5*np.log10(tractorflux/inputflux)
        b[band]= np.abs(mag_diff) - abs(med[band]) > 0.001
    # total boolean mask
    b_bright= np.any((b_bright['G'],b_bright['R'],b_bright['Z']),axis=0)
    b_large_dmag= np.all((b_bright,b_good,b['G'],b['R'],b['Z']),axis=0)
    b_small_dmag= np.all((b_bright,b_good,b['G']==False,b['R']==False,b['Z']==False),axis=0)
    return med,np.where(b_large_dmag)[0],np.where(b_small_dmag)[0]
 
def plot_cutouts_by_index(simcat,index, brickname,lobjtype,chunksuffix, \
                          indir=None, img_name='simscoadd',qafile='test.png'):
    hw = 30 # half-width [pixels]
    rad = 14
    ncols = 5
    nrows = 5
    nthumb = ncols*nrows
    dims = (ncols*hw*2,nrows*hw*2)
    mosaic = Image.new('RGB',dims)

    xpos, ypos = np.meshgrid(np.arange(0, dims[0], hw*2, dtype='int'),
                             np.arange(0, dims[1], hw*2, dtype='int'))
    im = Image.open( os.path.join(indir, 'qa-{}-{}-{}-{:02d}.jpg'.format(brickname, lobjtype, img_name, int(chunksuffix))) )
    sz = im.size
    iobj = 0
    for ic in range(ncols):
        if iobj >= len(index) or iobj >= ncols*nrows: break
        for ir in range(nrows):
            if iobj >= len(index) or iobj >= ncols*nrows: break
            xx = int(simcat['X'][index[iobj]])
            yy = int(sz[1]-simcat['Y'][index[iobj]])
            crop = (xx-hw, yy-hw, xx+hw, yy+hw)
            box = (xpos[ir, ic], ypos[ir, ic])
            thumb = im.crop(crop)
            mosaic.paste(thumb, box)
            iobj+= 1

    # Add a border and circle the missing source.
    draw = ImageDraw.Draw(mosaic)
    sz = mosaic.size
    for ic in range(ncols):
        for ir in range(nrows):
            draw.rectangle([(xpos[ir, ic], ypos[ir, ic]),
                            (xpos[ir, ic]+hw*2, ypos[ir, ic]+hw*2)])
            xx = xpos[ir, ic] + hw
            yy = ypos[ir, ic] + hw
            draw.ellipse((xx-rad, sz[1]-yy-rad, xx+rad, sz[1]-yy+rad), outline='yellow')
    mosaic.save(qafile)


def plot_annotated_coadds(simcat, brickname, lobjtype, chunksuffix,\
                          indir=None,img_name='simscoadd',qafile='test.png'):
    rad = 7/0.262
    #imfile = os.path.join(cdir, 'qa-{}-{}-{}-{}.jpg'.format(brickname, lobjtype, suffix, chunksuffix))
    # HARDCODED fix this!!!!!
    imfile = os.path.join(indir, 'qa-{}-{}-{}-{:02d}.jpg'.format(brickname, lobjtype, img_name, int(chunksuffix)))
    im = Image.open(imfile)
    sz = im.size
    draw = ImageDraw.Draw(im)
    [draw.ellipse((cat['X']-rad, sz[1]-cat['Y']-rad, cat['X']+rad,
                   sz[1]-cat['Y']+rad), outline='yellow') for cat in simcat]
    im.save(qafile)


def bin_up(data_bin_by,data_for_percentile, bin_minmax=(18.,26.),nbins=20):
    '''bins "data_for_percentile" into "nbins" using "data_bin_by" to decide how indices are assigned to bins
    returns bin center,N,q25,50,75 for each bin
    '''
    bin_edges= np.linspace(bin_minmax[0],bin_minmax[1],num= nbins+1)
    vals={}
    for key in ['q50','q25','q75','n']: vals[key]=np.zeros(nbins)+np.nan
    vals['binc']= (bin_edges[1:]+bin_edges[:-1])/2.
    for i,low,hi in zip(range(nbins), bin_edges[:-1],bin_edges[1:]):
        keep= np.all((low < data_bin_by,data_bin_by <= hi),axis=0)
        if np.where(keep)[0].size > 0:
            vals['n'][i]= np.where(keep)[0].size
            vals['q25'][i]= np.percentile(data_for_percentile[keep],q=25)
            vals['q50'][i]= np.percentile(data_for_percentile[keep],q=50)
            vals['q75'][i]= np.percentile(data_for_percentile[keep],q=75)
        else:
            vals['n'][i]=0 
    return vals

def plot_injected_mags(allsimcat, log,qafile='test.png'):
    gr_sim = -2.5*np.log10(allsimcat['GFLUX']/allsimcat['RFLUX'])
    rz_sim = -2.5*np.log10(allsimcat['RFLUX']/allsimcat['ZFLUX'])
    grrange = (-0.2, 2.0)
    rzrange = (-0.4, 2.5)
    fig, ax = plt.subplots(2,1,figsize=(6,8))
    ax[0].hist(22.5-2.5*np.log10(allsimcat['RFLUX']),bins=20,align='mid')
    ax[1].scatter(rz_sim,gr_sim,
                   s=10,edgecolor='b',c='none',lw=1.)
    for i,x_lab,y_lab in zip(range(2),['r AB','r-z'],['N','g-r']):
        xlab=ax[i].set_xlabel(x_lab)
        ylab=ax[i].set_ylabel(y_lab)
    ax[1].set_xlim(rzrange)
    ax[1].set_ylim(grrange)
    fig.subplots_adjust(wspace=0.25)
    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile,bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
    plt.close()
 
def plot_good_bad_ugly(allsimcat,bigsimcat,bigsimcat_missing, nmagbin,rminmax, b_good,b_bad, log,qafile='test.png'):
    #rmaghist, magbins = np.histogram(allsimcat['R'], bins=nmagbin, range=rminmax)
    found=dict(good={},bad={},missed={})
    for index,name in zip([b_good,b_bad],['good','bad']):
        # bin on true r mag of matched objects, count objects in each bin
        found[name]= bin_up(bigsimcat['R'][index],bigsimcat['R'][index], bin_minmax=rminmax,nbins=nmagbin) # bin_edges=magbins)
    name='missed'
    found[name]= bin_up(bigsimcat_missing['R'],bigsimcat_missing['R'], bin_minmax=rminmax,nbins=nmagbin) #bin_edges=magbins)
    fig, ax = plt.subplots(1, figsize=(8,6))
    for name,color in zip(['good','bad','missed'],['k','b','r']):
        ax.step(found[name]['binc'],found[name]['n'], c=color,lw=2,label=name)
    xlab=ax.set_xlabel('Input r AB')
    ylab=ax.set_ylabel('Number Recovered')
    leg=ax.legend(loc=(0.,1.01),ncol=3)
    #fig.subplots_adjust(bottom=0.15)
    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile, bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight')
    plt.close()

def plot_tractor_minus_answer(bigsimcat,bigtractor, b_good,rminmax, log,qafile='test.png'):
    fig, ax = plt.subplots(3, sharex=True, figsize=(6,8))

    col = ['b', 'k', 'c', 'm', 'y', 0.8]
    rmag = bigsimcat['R']
    for thisax, thiscolor, band, indx in zip(ax, col, ('G', 'R', 'Z'), (1, 2, 4)):
        inputflux = bigsimcat[band+'FLUX']
        tractorflux = bigtractor['decam_flux'][:, indx]
        tractorivar = bigtractor['decam_flux_ivar'][:, indx]
        #import pickle
        #fout=open('test.pickle','w')
        #pickle.dump((tractorflux,inputflux,b_good),fout)
        #fout.close()
        #print('exiting early')
        #sys.exit()
        inputmag = 22.5-2.5*np.log10(inputflux[b_good])
        mag_diff= -2.5*np.log10(tractorflux[b_good]/inputflux[b_good])
        thisax.scatter(inputmag, mag_diff,
                       s=10,edgecolor=thiscolor,c='none',lw=1.)
        thisax.set_ylim(-0.1,0.1)
        thisax.set_xlim(inputmag.min()-0.1, inputmag.max()+0.1)
        thisax.axhline(y=0.0,lw=2,ls='solid',color='gray')
        #arr= np.ma.masked_array(mag_diff, mask= np.isfinite(mag_diff) == False)
        #med= np.median(arr)
        #print('arr=',arr)
        med,junk1,junk2= bright_dmag_cut(bigsimcat,bigtractor)
        thisax.axhline(y=med[band],lw=2,ls='dashed',color='red',label='Median=%.3f' % med[band])
        thisax.legend(loc='upper left',fontsize='x-small')
        #thisax.text(0.05,0.05, band.lower(), horizontalalignment='left',
                    #verticalalignment='bottom',transform=thisax.transAxes,
                    #fontsize=16)
    ax[0].set_ylabel('$\Delta$g')
    ax[1].set_ylabel('$\Delta$r (Tractor - Input)')
    ylab=ax[2].set_ylabel('$\Delta$z')
    xlab=ax[2].set_xlabel('Input magnitude (AB mag)')
    fig.subplots_adjust(left=0.18,hspace=0.1)
    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile,bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
    plt.close()

def plot_chi(bigsimcat,bigtractor, b_good,rminmax, log,qafile='test.png'):
    col = ['b', 'k', 'c', 'm', 'y', 0.8]
    fig, ax = plt.subplots(3, sharex=True, figsize=(6,8))

    rmag = bigsimcat['R']
    for thisax, thiscolor, band, indx in zip(ax, col, ('G', 'R', 'Z'), (1, 2, 4)):
        simflux = bigsimcat[band+'FLUX']
        tractorflux = bigtractor['decam_flux'][:, indx]
        tractorivar = bigtractor['decam_flux_ivar'][:, indx]
        #thisax.scatter(rmag[bcut], -2.5*np.log10(tractorflux[bcut]/simflux[bcut]),
        #               s=10,edgecolor=newcol,c='none',lw=1.,label=label)
        thisax.scatter(rmag[b_good], (tractorflux[b_good] - simflux[b_good])*np.sqrt(tractorivar[b_good]),
                       s=10,edgecolor=thiscolor,c='none',lw=1.)
        #thisax.set_ylim(-0.7,0.7)
        thisax.set_ylim(-4,4)
        thisax.set_xlim(rminmax + [-0.1, 0.0])
        thisax.axhline(y=0.0,lw=2,ls='solid',color='gray')
        #thisax.text(0.05,0.05, band.lower(), horizontalalignment='left',
                    #verticalalignment='bottom',transform=thisax.transAxes,
                    #fontsize=16)
    for i,b in enumerate(['g','r','z']):   
        ylab=ax[i].set_ylabel(r'%s: $(F_{tractor} - F)/\sigma_{tractor}$' %  b) 
    #ax[0].set_ylabel('$\Delta$g')
    #ax[1].set_ylabel('$\Delta$r (Tractor minus Input)')
    #ax[2].set_ylabel('$\Delta$z')
    xlab=ax[2].set_xlabel('Input r magnitude (AB mag)')

    fig.subplots_adjust(left=0.18,hspace=0.1)
    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile,bbox_extra_artists=[xlab,ylab], bbox_inches='tight')
    plt.close()
 
def plot_color_tractor_minus_answer(bigtractor,bigsimcat, rminmax, brickname,lobjtype, log,qafile='test.png'):
    gr_tra = -2.5*np.log10(bigtractor['decam_flux'][:, 1]/bigtractor['decam_flux'][:, 2])
    rz_tra = -2.5*np.log10(bigtractor['decam_flux'][:, 2]/bigtractor['decam_flux'][:, 4])
    gr_sim = -2.5*np.log10(bigsimcat['GFLUX']/bigsimcat['RFLUX'])
    rz_sim = -2.5*np.log10(bigsimcat['RFLUX']/bigsimcat['ZFLUX'])
    rmag = bigsimcat['R']

    col = ['b', 'k', 'c', 'm', 'y', 0.8]
    fig, ax = plt.subplots(2,sharex=True,figsize=(6,8))
    
    ax[0].scatter(rmag, gr_tra-gr_sim, color=col[0], s=10)
    ax[1].scatter(rmag, rz_tra-rz_sim, color=col[1], s=10)
    
    [thisax.set_ylim(-0.7,0.7) for thisax in ax]
    [thisax.set_xlim(rminmax + [-0.1, 0.0]) for thisax in ax]
    [thisax.axhline(y=0.0, lw=2, ls='solid', color='gray') for thisax in ax]
    
    ax[0].set_ylabel('$\Delta$(g - r) (Tractor minus Input)')
    ax[1].set_ylabel('$\Delta$(r - z) (Tractor minus Input)')
    ax[1].set_xlabel('Input r magnitude (AB mag)')
    fig.subplots_adjust(left=0.18,hspace=0.1)

    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile)
    plt.close()

def plot_fraction_recovered(allsimcat,bigsimcat, nmagbin,rminmax, brickname, lobjtype, log,qafile='test.png'):
    rmaghist, magbins = np.histogram(allsimcat['R'], bins=nmagbin, range=rminmax)
    cmagbins = (magbins[:-1] + magbins[1:]) / 2.0
    ymatch, binsmatch = np.histogram(bigsimcat['R'], bins=nmagbin, range=rminmax)
    fig, ax = plt.subplots(1, figsize=(8,6))
    ax.step(cmagbins, 1.0*ymatch/rmaghist, c='k',lw=3,label='All objects')
    #ax.step(cmagbins, 1.0*ymatchgood/rmaghist, lw=3, ls='dashed', label='|$\Delta$m|<0.3')
    ax.axhline(y=1.0,lw=2,ls='dashed',color='k')
    ax.set_xlabel('Input r magnitude (AB mag)')
    ax.set_ylabel('Fraction Recovered'.format(lobjtype))
    ax.set_ylim([0.0, 1.1])
    ax.legend('lower left')
    fig.subplots_adjust(bottom=0.15)
    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile)
    plt.close()

def plot_sn_recovered(allsimcat,bigsimcat,bigtractor, brickname, lobjtype, log,qafile='test.png'):
    # min,max mag of all bands
    grrange = (-0.2, 2.0)
    rzrange = (-0.4, 2.5)
    rmin,rmax= allsimcat['R'].min(), allsimcat['R'].max()
    mag_min= np.min((rmin,rmin+grrange[0],rmin-rzrange[1]))
    mag_max= np.max((rmax,rmax+grrange[1],rmax-rzrange[0]))
    s2n=dict(g={},r={},z={})
    for band,ith in zip(['g','r','z'],[1,2,4]):
        mag= 22.5-2.5*np.log10(bigsimcat[band.upper()+'FLUX'])
        # HARDCODED mag range
        s2n[band]= bin_up(mag, bigtractor['decam_flux'][:,ith]*np.sqrt(bigtractor['decam_flux_ivar'][:,ith]), \
                          bin_minmax=(18,26),nbins=20) 
    fig, ax = plt.subplots(1, figsize=(8,6))
    xlab=ax.set_xlabel('Input magnitude (AB)')
    ylab=ax.set_ylabel(r'Median S/N = $F/\sigma$',fontweight='bold',fontsize='large')
    title= ax.set_title('S/N of Recovered Objects')
    for band,color in zip(['g','r','z'],['g','r','b']):
        ax.plot(s2n[band]['binc'], s2n[band]['q50'],c=color,ls='-',lw=2,label=band)
        #ax.fill_between(s2n[band]['cbin'],s2n[band]['q25'],s2n[band]['q75'],color=color,alpha=0.25)
    ax.axhline(y=5.,lw=2,ls='dashed',color='k',label='S/N = 5')
    ax.set_yscale('log')
    leg=ax.legend(loc=3)
    fig.subplots_adjust(bottom=0.15)
    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile,bbox_extra_artists=[leg,xlab,ylab,title], bbox_inches='tight',dpi=150)
    plt.close()

def plot_recovered_types(bigsimcat,bigtractor, nmagbin,rminmax, objtype,log,qafile='test.png'):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    rmaghist, magbins = np.histogram(bigsimcat['R'], bins=nmagbin, range=rminmax)
    cmagbins = (magbins[:-1] + magbins[1:]) / 2.0
    tractortype = np.char.strip(bigtractor['type'].data)
    for otype in ['PSF', 'SIMP', 'EXP', 'DEV', 'COMP']:
        these = np.where(tractortype == otype)[0]
        if len(these)>0:
            yobj, binsobj = np.histogram(bigsimcat['R'][these], bins=nmagbin, range=rminmax)
            #plt.step(cmagbins,1.0*yobj,lw=3,alpha=0.5,label=otype)
            plt.step(cmagbins,1.0*yobj/rmaghist,lw=3,alpha=0.5,label=otype)
    plt.axhline(y=1.0,lw=2,ls='dashed',color='gray')
    plt.xlabel('Input r magnitude (AB mag)')
    #plt.ylabel('Number of Objects')
    plt.ylabel('Fraction of {}s classified'.format(objtype))
    plt.ylim([0.0,1.1])
    plt.legend(loc='center left', bbox_to_anchor=(0.08,0.5))
    fig.subplots_adjust(bottom=0.15)

    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile)
    plt.close()


 
def create_confusion_matrix(answer_type,predict_type, types=['PSF','SIMP','EXP','DEV','COMP'],slim=True):
    '''compares classifications of matched objects, returns 2D array which is conf matrix and xylabels
    return 5x5 confusion matrix and colum/row names
   answer_type,predict_type -- arrays of same length with reference and prediction types'''
    for typ in set(answer_type): assert(typ in types)
    for typ in set(predict_type): assert(typ in types)
    # if a type was not in answer (training) list then don't put in cm
    if slim: ans_types= set(answer_type)
    # put in cm regardless
    else: ans_types= set(types)
    cm=np.zeros((len(ans_types),len(types)))-1
    for i_ans,ans_type in enumerate(ans_types):
        ind= np.where(answer_type == ans_type)[0]
        for i_pred,pred_type in enumerate(types):
            n_pred= np.where(predict_type[ind] == pred_type)[0].size
            if ind.size > 0: cm[i_ans,i_pred]= float(n_pred)/ind.size # ind.size is constant for loop over pred_types
            else: cm[i_ans,i_pred]= np.nan
    if slim: return cm,ans_types,types #size ans_types != types
    else: return cm,types

def plot_confusion_matrix(cm,answer_names,all_names, log, qafile='test.png'):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0,vmax=1)
    cbar=plt.colorbar()
    plt.xticks(range(len(all_names)), all_names)
    plt.yticks(range(len(answer_names)), answer_names)
    ylab=plt.ylabel('True')
    xlab=plt.xlabel('Predicted (tractor)')
    for row in range(len(answer_names)):
        for col in range(len(all_names)):
            if np.isnan(cm[row,col]): 
                plt.text(col,row,'n/a',va='center',ha='center')
            elif cm[row,col] > 0.5: 
                plt.text(col,row,'%.2f' % cm[row,col],va='center',ha='center',color='yellow')
            else: 
                plt.text(col,row,'%.2f' % cm[row,col],va='center',ha='center',color='black')
    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile, bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()

def plot_cm_stack(cm_stack,stack_names,all_names, log, qafile='test.png'):
    '''cm_stack -- list of single row confusion matrices
    stack_names -- list of same len as cm_stack, names for each row of cm_stack'''
    # combine list into single cm
    cm=np.zeros((len(cm_stack),len(all_names)))+np.nan
    for i in range(cm.shape[0]): cm[i,:]= cm_stack[i]
    # make usual cm, but labels repositioned
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar=plt.colorbar()
    plt.xticks(range(len(all_names)), all_names)
    plt.yticks(range(len(stack_names)), stack_names)
    ylab=plt.ylabel('True = PSF')
    xlab=plt.xlabel('Predicted (tractor)')
    for row in range(len(stack_names)):
        for col in range(len(all_names)):
            if np.isnan(cm[row,col]): 
                plt.text(col,row,'n/a',va='center',ha='center')
            elif cm[row,col] > 0.5: 
                plt.text(col,row,'%.2f' % cm[row,col],va='center',ha='center',color='yellow')
            else: 
                plt.text(col,row,'%.2f' % cm[row,col],va='center',ha='center',color='black')
            #if np.isnan(cm[row,col]): 
            #    plt.text(col,row,'n/a',va='center',ha='center')
            #else: plt.text(col,row,'%.2f' % cm[row,col],va='center',ha='center')
    log.info('Writing {}'.format(qafile))
    plt.savefig(qafile, bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()

def make_stacked_cm(bigsimcat,bigtractor, b_good, log,qafile='test.png'):
    types= ['PSF ', 'SIMP', 'EXP ', 'DEV ', 'COMP']
    cm_stack,stack_names=[],[]
    rbins= np.array([18.,20.,22.,23.,24.])
    for rmin,rmax in zip(rbins[:-1],rbins[1:]):
        # master cut
        br_cut= np.all((bigsimcat['R'] > rmin,bigsimcat['R'] <= rmax, b_good),axis=0)
        stack_names+= ["%d < r <= %d" % (int(rmin),int(rmax))]
        cm,ans_names,all_names= create_confusion_matrix(np.array(['PSF ']*bigtractor['ra'].data[br_cut].shape[0]),
                                                        bigtractor['type'].data[br_cut], \
                                                        types=types)
        cm_stack+= [cm]
    plot_cm_stack(cm_stack, stack_names,all_names, log, qafile=qafile)
 

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='DECaLS simulations.')
    parser.add_argument('-b', '--brick', type=str, default='2428p117', metavar='', 
                        help='process this brick (required input)')
    parser.add_argument('-o', '--objtype', type=str, choices=['STAR', 'ELG', 'LRG', 'BGS'], default='STAR', metavar='', 
                        help='object type (STAR, ELG, LRG, BGS)') 
    parser.add_argument('-out', '--output_dir', type=str, default=None, metavar='', 
                        help='relative path to output directory') 
    parser.add_argument('-extra', '--extra_plots', action='store_true', 
                        help='make missing, annotated, coadded plots, dont if many chunks')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='toggle on verbose output')

    args = parser.parse_args()
    if args.brick is None:
        parser.print_help()
        sys.exit(1)
    if args.extra_plots: extra_plots= True
    else: extra_plots= False
    print('extra_plots=',extra_plots)

    # Set the debugging level
    if args.verbose:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO
    logging.basicConfig(format='%(message)s', level=lvl, stream=sys.stdout)
    log = logging.getLogger('__name__')

    brickname = args.brick
    objtype = args.objtype.upper()
    lobjtype = objtype.lower()
    log.info('Analyzing objtype {} on brick {}'.format(objtype, brickname))

    if 'DECALS_SIM_DIR' in os.environ:
        decals_sim_dir = os.getenv('DECALS_SIM_DIR')
    else:
        decals_sim_dir = '.'
    input_dir= os.path.join(decals_sim_dir,brickname,lobjtype)
    if args.output_dir is None: output_dir= os.path.join(decals_sim_dir,brickname,'qaplots_'+lobjtype)
    else: output_dir= args.output_dir
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
    # Plotting preferences
    #sns.set(style='white',font_scale=1.6,palette='dark')#,font='fantasy')
    #col = sns.color_palette('dark')
    col = ['b', 'k', 'c', 'm', 'y', 0.8]
    
    # Read metadata catalog.
    metafile = os.path.join(input_dir, 'metacat-{}-{}.fits'.format(brickname, lobjtype))
    log.info('Reading {}'.format(metafile))
    meta = fits.getdata(metafile, 1)
    
    # We need this for our histograms below
    magbinsz = 0.2
    rminmax = np.squeeze(meta['RMAG_RANGE'])
    nmagbin = long((rminmax[1]-rminmax[0])/magbinsz)

    # Work in chunks.
    allsimcat = []
    bigsimcat = []
    bigsimcat_missed = []
    bigtractor = []
    chunk_dirs= glob.glob(os.path.join(input_dir,'0*'))
    nchunk= len(chunk_dirs)
    if nchunk == 0: raise ValueError
    # Loop through chunk dirs 000,001,...,999
    for ichunk,cdir in enumerate([chunk_dirs[0]]):
        chunksuffix = os.path.basename(cdir) #'{:02d}'.format(ichunk)
        log.info('Working on chunk {:02d}/{:02d}'.format(ichunk+1, nchunk))
        
        # Read the simulated object catalog
        simcatfile = os.path.join(cdir, 'simcat-{}-{}-{:02d}.fits'.format(brickname, lobjtype, int(chunksuffix)))
        log.info('Reading {}'.format(simcatfile))
        simcat = Table(fits.getdata(simcatfile, 1))

        # Read Tractor catalog
        tractorfile = os.path.join(cdir, 'tractor-{}-{}-{:02d}.fits'.format(brickname, lobjtype, int(chunksuffix)))
        log.info('Reading {}'.format(tractorfile))
        tractor = Table(fits.getdata(tractorfile, 1))
        # Match
        cat1 = SkyCoord(ra=tractor['ra']*units.degree, dec=tractor['dec']*units.degree)
        cat2 = SkyCoord(ra=simcat['RA']*units.degree, dec=simcat['DEC']*units.degree)
        m2, d2d, d3d = cat1.match_to_catalog_3d(cat2)
        b= np.array(d2d) <= 1./3600
        m2= np.array(m2)[b]
        m1= np.arange(len(tractor))[b]
        print('matched %d/%d' % (len(m2),len(simcat['RA'])))
        
        missing = np.delete(np.arange(len(simcat)), m2, axis=0)
        log.info('Missing {}/{} sources'.format(len(missing), len(simcat)))

        #good = np.where((np.abs(tractor['decam_flux'][m1,2]/simcat['rflux'][m2]-1)<0.3)*1)

        # Build matching catalogs for the plots below.
        if len(bigsimcat) == 0:
            bigsimcat = simcat[m2]
            bigtractor = tractor[m1]
            bigsimcat_missing = simcat[missing]
        else:
            bigsimcat = vstack((bigsimcat, simcat[m2]))
            bigtractor = vstack((bigtractor, tractor[m1]))
            bigsimcat_missing = vstack((bigsimcat_missing, simcat[missing]))
        if len(allsimcat) == 0:
            allsimcat = simcat
        else:
            allsimcat = vstack((allsimcat, simcat))

        # Get cutouts of the bright matched sources with small/large delta mag 
        if extra_plots:
            for img_name in ['image','resid','simscoadd']:
                # Indices of large and small dmag
                junk,i_large_dmag,i_small_dmag= bright_dmag_cut(simcat[m2],tractor[m1])
                # Large dmag cutouts
                qafile = os.path.join(output_dir, 'qa-{}-{}-{}-bright-large-dmag-{:02d}.png'.format(\
                                            brickname, lobjtype, img_name, int(chunksuffix)))
                plot_cutouts_by_index(simcat,i_large_dmag, brickname,lobjtype,chunksuffix, \
                                      indir=cdir,img_name=img_name,qafile=qafile)
                log.info('Wrote {}'.format(qafile))
                # Small dmag cutouts
                qafile = os.path.join(output_dir, 'qa-{}-{}-{}-bright-small-dmag-{:02d}.png'.format(\
                                            brickname, lobjtype, img_name, int(chunksuffix)))
                plot_cutouts_by_index(simcat,i_small_dmag, brickname,lobjtype,chunksuffix, \
                                      indir=cdir,img_name=img_name,qafile=qafile)
                log.info('Wrote {}'.format(qafile))
         
        # Get cutouts of the missing sources in each chunk (if any)
        if len(missing) > 0 and extra_plots:
            for img_name in ['image']: #,'simscoadd']:
                qafile = os.path.join(output_dir, 'qa-{}-{}-{}-missing-{:02d}.png'.format(\
                                            brickname, lobjtype, img_name, int(chunksuffix)))
                miss = missing[np.argsort(simcat['R'][missing])]
                plot_cutouts_by_index(simcat,miss, brickname,lobjtype,chunksuffix, \
                                      indir=cdir,img_name=img_name,qafile=qafile)
                log.info('Wrote {}'.format(qafile))
            
            

        # Annotate the coadd image and residual files so the simulated sources
        # are labeled.
        if extra_plots:
            for img_name in ('simscoadd','image', 'resid'):
                qafile = os.path.join(output_dir, 'qa-{}-{}-{}-{:02d}-annot.png'.format(\
                                      brickname, lobjtype,img_name, int(chunksuffix)))
                plot_annotated_coadds(simcat, brickname, lobjtype, chunksuffix, \
                                      indir=cdir,img_name=img_name,qafile=qafile)
                log.info('Wrote {}'.format(qafile))

    # now operate on concatenated catalogues from multiple chunks
    # Grab flags
    b_good,b_bad= basic_cut(bigtractor)

    # mags and colors of ALL injected sources
    plot_injected_mags(allsimcat, log, qafile=\
           os.path.join(output_dir, 'qa-{}-{}-injected-mags.png'.format(brickname, lobjtype)))
   
    # number of detected sources that are bad, good and number of undetected, binned by r mag
    plot_good_bad_ugly(allsimcat,bigsimcat,bigsimcat_missing, nmagbin,rminmax, b_good,b_bad, log, qafile=\
            os.path.join(output_dir, 'qa-{}-{}-N-good-bad-missed.png'.format(brickname, lobjtype)))

    # Flux residuals vs input magnitude
    plot_tractor_minus_answer(bigsimcat,bigtractor, b_good,rminmax, log, qafile=\
            os.path.join(output_dir, 'qa-{}-{}-good-flux.png'.format(brickname, lobjtype)))
 
    # chi plots: Flux residual / estimated Flux error
    plot_chi(bigsimcat,bigtractor, b_good,rminmax, log, qafile=\
             os.path.join(output_dir, 'qa-{}-{}-chi-good.png'.format(brickname, lobjtype)))
   
    # Color residuals
    plot_color_tractor_minus_answer(bigtractor,bigsimcat, rminmax, brickname,lobjtype, log, qafile =\
             os.path.join(output_dir, 'qa-{}-{}-color.png'.format(brickname, lobjtype)))

    # Fraction of recovered sources
    plot_fraction_recovered(allsimcat,bigsimcat, nmagbin,rminmax, brickname, lobjtype, log, qafile =\
             os.path.join(output_dir, 'qa-{}-{}-frac.png'.format(brickname, lobjtype)))

    # S/N of recovered sources (S/N band vs. AB mag band)
    plot_sn_recovered(allsimcat,bigsimcat,bigtractor, brickname, lobjtype, log, qafile =\
             os.path.join(output_dir, 'qa-{}-{}-SN.png'.format(brickname, lobjtype)))

    # Distribution of object types for matching sources.
    plot_recovered_types(bigsimcat,bigtractor, nmagbin,rminmax, objtype,log, qafile =\
             os.path.join(output_dir, 'qa-{}-{}-type.png'.format(brickname, lobjtype)))

    # Confusion matrix for distribution of object types
    # Basic cm, use slim=False
    types= ['PSF ', 'SIMP', 'EXP ', 'DEV ', 'COMP']
    cm,all_names= create_confusion_matrix(np.array(['PSF ']*bigtractor['ra'].data[b_good].shape[0]),
                                                            bigtractor['type'].data[b_good], \
                                                            types=types,slim=False)
    qafile = os.path.join(output_dir, 'qa-{}-{}-{}-confusion.png'.format(brickname, lobjtype,'good'))
    plot_confusion_matrix(cm,all_names,all_names, log,qafile)
    
    # Now a stacked confusion matrix
    # Compute a row for each r mag range and stack rows
    make_stacked_cm(bigsimcat,bigtractor, b_good, log,qafile =\
             os.path.join(output_dir, 'qa-{}-{}-good-confusion-stack.png'.format(brickname, lobjtype)))
   
    '''
    # Morphology plots
    if objtype=='ELG':
        fig = plt.figure(figsize=(8,4))
        plt.subplot(1,3,1)
        plt.plot(rmag,deltam,'s',markersize=3)
        plt.axhline(y=0.0,lw=2,ls='solid',color='gray')
        plt.xlim(rminmax)
        plt.xlabel('r (AB mag)')

        plt.subplot(1,3,2)
        plt.plot(bigsimcat['R50_1'],deltam,'s',markersize=3)
        plt.axhline(y=0.0,lw=2,ls='solid',color='gray')
        plt.xlabel('$r_{50}$ (arcsec)')

        plt.subplot(1,3,3)
        plt.plot(bigsimcat['BA_1'],deltam,'s',markersize=3)
        plt.axhline(y=0.0,lw=2,ls='solid',color='gray')
        plt.xlabel('b/a')
        plt.xlim([0.2,1.0])
        fig.subplots_adjust(bottom=0.18)
        qafile = os.path.join(output_dir,'qa-'+brickname+'-'+lobjtype+'-morph.png')
        log.info('Writing {}'.format(qafile))
        plt.savefig(qafile)
    '''
    
if __name__ == "__main__":    
    main()
