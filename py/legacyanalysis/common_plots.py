'''plotting functions for Validation
input -- one or more Single_TractorCat() objects
      -- ref is reference Single_TractorCat()
         test is test ...
'''
import matplotlib
matplotlib.use('Agg') #display backend
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from scipy import stats as sp_stats

# Globals
class PlotKwargs(object):
    def __init__(self):
        self.ax= dict(fontweight='bold',fontsize='medium')
        self.text=dict(fontweight='bold',fontsize='medium',va='top',ha='left')
        self.leg=dict(frameon=True,fontsize='x-small')
        self.save= dict(bbox_inches='tight',dpi=150)

kwargs= PlotKwargs()
##########

# Helpful functions for making plots
def bin_up(data_bin_by,data_for_percentile, bin_minmax=(18.,26.),nbins=20):
    '''bins "data_for_percentile" into "nbins" using "data_bin_by" to decide how indices are assigned to bins
    returns bin center,N,q25,50,75 for each bin
    '''
    bin_edges= np.linspace(bin_minmax[0],bin_minmax[1],num= nbins+1)
    N= np.zeros(nbins)+np.nan
    q25,q50,q75= N.copy(),N.copy(),N.copy()
    for i,low,hi in zip(range(nbins-1), bin_edges[:-1],bin_edges[1:]):
        keep= np.all((low <= data_bin_by,data_bin_by < hi),axis=0)
        if np.where(keep)[0].size > 0:
            N[i]= np.where(keep)[0].size
            q25[i]= np.percentile(data_for_percentile[keep],q=25)
            q50[i]= np.percentile(data_for_percentile[keep],q=50)
            q75[i]= np.percentile(data_for_percentile[keep],q=75)
        else:
            # already initialized to nan
            pass
    return (bin_edges[1:]+bin_edges[:-1])/2.,N,q25,q50,q75

# Main plotting functions 
def nobs(obj, name=''):
    '''make histograms of nobs so can compare depths of g,r,z between the two catalogues
    obj -- Single_TractorCat()'''   
    hi= np.max(obj.t['decam_nobs'][:,[1,2,4]])
    fig,ax= plt.subplots(3,1)
    for i, band,iband in zip(range(3),['g','r','z'],[1,2,4]):
        ax[i].hist(obj.t['decam_nobs'][:,iband],\
                   bins=hi+1,normed=True,cumulative=True,align='mid')
        xlab=ax[i].set_xlabel('nobs %s' % band, **kwargs.ax)
        ylab=ax[i].set_ylabel('CDF', **kwargs.ax)
    plt.savefig(os.path.join(obj.outdir,'nobs_%s.png' % name), \
                bbox_extra_artists=[xlab,ylab], **kwargs.save)
    plt.close()

def radec(obj,name=''): 
    '''ra,dec distribution of objects
    obj -- Single_TractorCat()'''
    plt.scatter(obj.t['ra'], obj.t['dec'], \
                edgecolor='b',c='none',lw=1.)
    xlab=plt.xlabel('RA', **kwargs.ax)
    ylab=plt.ylabel('DEC', **kwargs.ax)
    plt.savefig(os.path.join(obj.outdir,'radec_%s.png' % name), \
                bbox_extra_artists=[xlab,ylab], **kwargs.save)
    plt.close()


def hist_types(obj, name=''):
    '''number of psf,exp,dev,comp, etc
    obj -- Single_TractorCat()'''
    types= ['PSF','SIMP','EXP','DEV','COMP']
    # the x locations for the groups
    ind = np.arange(len(types))  
    # the width of the bars
    width = 0.35       
    ht= np.zeros(len(types),dtype=int)
    for cnt,typ in enumerate(types):
        # Mask to type desired
        ht[cnt]= obj.number_not_masked(['current',typ.lower()])
    # Plot
    fig, ax = plt.subplots()
    rects = ax.bar(ind, ht, width, color='b')
    ylab= ax.set_ylabel("counts")
    ax.set_xticks(ind + width)
    ax.set_xticklabels(types)
    plt.savefig(os.path.join(obj.outdir,'hist_types_%s.png' % name), \
                bbox_extra_artists=[ylab], **kwargs.save)
    plt.close()


def sn_vs_mag(obj, mag_minmax=(18.,26.),name=''):
    '''plots Signal to Noise vs. mag for each band
    obj -- Single_TractorCat()'''
    min,max= mag_minmax
    # Bin up SN values
    bin_SN={}
    for band,iband in zip(['g','r','z'],[1,2,4]):
        bin_SN[band]={}
        bin_SN[band]['binc'],N,bin_SN[band]['q25'],bin_SN[band]['q50'],bin_SN[band]['q75']=\
                bin_up(obj.t['decam_mag'][:,iband], \
                       obj.t['decam_flux'][:,iband]*np.sqrt(obj.t['decam_flux_ivar'][:,iband]),\
                       bin_minmax=mag_minmax)
    #setup plot
    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
    plt.subplots_adjust(wspace=0.25)
    #plot SN
    for cnt,band,color in zip(range(3),['g','r','z'],['g','r','m']):
        #horiz line at SN = 5
        ax[cnt].plot([mag_minmax[0],mag_minmax[1]],[5,5],'k--',lw=2)
        #data
        ax[cnt].plot(bin_SN[band]['binc'], bin_SN[band]['q50'],c=color,ls='-',lw=2)
        ax[cnt].fill_between(bin_SN[band]['binc'],bin_SN[band]['q25'],bin_SN[band]['q75'],color=color,alpha=0.25)
    #labels
    for cnt,band in zip(range(3),['g','r','z']):
        ax[cnt].set_yscale('log')
        xlab=ax[cnt].set_xlabel('%s' % band, **kwargs.ax)
        ax[cnt].set_ylim(1,100)
        ax[cnt].set_xlim(mag_minmax)
    ylab=ax[0].set_ylabel('S/N', **kwargs.ax)
    ax[2].text(26,5,'S/N = 5  ',**kwargs.text)
    plt.savefig(os.path.join(obj.outdir,'sn_%s.png' % name), \
                bbox_extra_artists=[xlab,ylab], **kwargs.save)
    plt.close()

def create_confusion_matrix(ref_obj,test_obj):
    '''compares MATCHED reference (truth) to test (prediction)
    ref_obj,test_obj -- reference,test Single_TractorCat()
    return 5x5 confusion matrix and colum/row names'''
    cm=np.zeros((5,5))-1
    types=['PSF','SIMP','EXP','DEV','COMP']
    for i_ref,ref_type in enumerate(types):
        n_ref= ref_obj.number_not_masked(['current',ref_type.lower()])
        for i_test,test_type in enumerate(types):
            n_test= test_obj.number_not_masked(['current',test_type.lower()])
            if n_ref > 0: cm[i_ref,i_test]= float(n_test)/n_ref
            else: cm[i_ref,i_test]= np.nan
    return cm,types


def confusion_matrix(ref_obj,test_obj, name='',\
                     ref_name='ref',test_name='test'):
    '''plot confusion matrix
    ref_obj,test_obj -- reference,test Single_TractorCat()'''
    cm,ticknames= create_confusion_matrix(ref_obj,test_obj)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues,vmin=0,vmax=1)
    cbar=plt.colorbar()
    plt.xticks(range(len(ticknames)), ticknames)
    plt.yticks(range(len(ticknames)), ticknames)
    ylab=plt.ylabel('True (%s)' % ref_name, **kwargs.ax)
    xlab=plt.xlabel('Predicted (%s)' % test_name, **kwargs.ax)
    for row in range(len(ticknames)):
        for col in range(len(ticknames)):
            if np.isnan(cm[row,col]):
                plt.text(col,row,'n/a',va='center',ha='center')
            elif cm[row,col] > 0.5:
                plt.text(col,row,'%.2f' % cm[row,col],va='center',ha='center',color='yellow')
            else:
                plt.text(col,row,'%.2f' % cm[row,col],va='center',ha='center',color='black')
    plt.savefig(os.path.join(ref_obj.outdir,'confusion_matrix_%s.png' % name), \
                bbox_extra_artists=[xlab,ylab], **kwargs.save)
    plt.close()



def matched_dist(obj,dist, name=''):
    '''dist -- array of distances in degress between matched objects'''
    pixscale=dict(decam=0.25,bass=0.45)
    # Plot
    fig,ax=plt.subplots()
    ax.hist(dist*3600,bins=50,color='b',align='mid')
    ax2 = ax.twiny()
    ax2.hist(dist*3600./pixscale['bass'],bins=50,color='g',align='mid',\
                visible=False)
    xlab= ax.set_xlabel("arcsec")
    xlab= ax2.set_xlabel("pixels [BASS]")
    ylab= ax.set_ylabel("Matched")
    plt.savefig(os.path.join(obj.outdir,"separation_hist_%s.png" % name), \
                bbox_extra_artists=[xlab,ylab], **kwargs.save)
    plt.close()

def chi_v_gaussian(ref_obj,test_obj, low=-8.,hi=8., name=''):
    # Compute Chi
    chi={} 
    for band,iband in zip(['g','r','z'],[1,2,4]):
        chi[band]= (ref_obj.t['decam_flux'][:,iband]-test_obj.t['decam_flux'][:,iband])/\
                   np.sqrt( np.power(ref_obj.t['decam_flux_ivar'][:,iband],-1)+\
                            np.power(test_obj.t['decam_flux_ivar'][:,iband],-1))
    for b_low,b_hi in zip([18,19,20,21,22,23],[19,20,21,22,23,24]):
        #loop over mag bins, one 3 panel for each mag bin
        hist= dict(g=0,r=0,z=0)
        binc= dict(g=0,r=0,z=0)
        stats=dict(g=0,r=0,z=0)
        # Counts per bin
        for band,iband in zip(['g','r','z'],[1,2,4]):
            imag= np.all((b_low <= ref_obj.t['decam_mag'][:,iband],\
                          ref_obj.t['decam_mag'][:,iband] < b_hi),axis=0)
            hist[band],bins= np.histogram(chi[band][imag],\
                                    range=(low,hi),bins=50,normed=True)
            db= (bins[1:]-bins[:-1])/2
            binc[band]= bins[:-1]+db
        # Unit gaussian N(0,1)
        G= sp_stats.norm(0,1)
        xvals= np.linspace(low,hi)
        # Plot for each mag range
        fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
        plt.subplots_adjust(wspace=0.25)
        for cnt,band in zip(range(3),['g','r','z']):
            ax[cnt].step(binc[band],hist[band], where='mid',c='b',lw=2)
            ax[cnt].plot(xvals,G.pdf(xvals))
        #labels
        for cnt,band in zip(range(3),['g','r','z']):
            if band == 'r': xlab=ax[cnt].set_xlabel(r'%s  $(F_{d}-F_{bm})/\sqrt{\sigma^2_{d}+\sigma^2_{bm}}$' % band, **kwargs.ax)
            else: xlab=ax[cnt].set_xlabel('%s' % band, **kwargs.ax)
            ax[cnt].set_ylim(0,0.6)
            ax[cnt].set_xlim(low,hi)
        ylab=ax[0].set_ylabel('PDF', **kwargs.ax)
        ti=ax[1].set_title("%s (%.1f <= %s < %.1f)" % (type,b_low,band,b_hi),**kwargs.ax)
        #put stats in suptitle
        plt.savefig(os.path.join(ref_obj.outdir,'chi_%.1f-%s-%.1f_%s.png' % (b_low,band,b_hi, name)), \
                    bbox_extra_artists=[ti,xlab,ylab], **kwargs.save)
        plt.close()

def delta_mag_vs_mag(ref_obj,test_obj, name='',\
                     ref_name='ref',test_name='test'):
    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
    plt.subplots_adjust(wspace=0.25)
    for cnt,band,iband in zip(range(3),['g','r','z'],[1,2,4]):
        delta= test_obj.t['decam_mag'][:,iband]- ref_obj.t['decam_mag'][:,iband]
        ax[cnt].scatter(ref_obj.t['decam_mag'][:,iband],\
                        delta,\
                        c='b',edgecolor='b',s=5) #,c='none',lw=2.)
    for cnt,band in zip(range(3),['g','r','z']):
        xlab=ax[cnt].set_xlabel('%s AB' % band, **kwargs.ax)
        ax[cnt].set_ylim(-0.1,0.1)
        ax[cnt].set_xlim(18,26)
    ylab=ax[0].set_ylabel('mag (%s) - mag(%s)' % (test_name,ref_name), **kwargs.ax)
    plt.savefig(os.path.join(ref_obj.outdir,'delta_mag_%s.png' % name), \
                bbox_extra_artists=[xlab,ylab], **kwargs.save)
    plt.close()

def n_per_deg2(obj,deg2=1., req_mags=[24.,23.4,22.5],name=''):
    '''compute number density in each bin for each band mag [18,requirement]
    deg2 -- square degrees spanned by sources in obj table
    req_mags -- image requirements grz<=24,23.4,22.5'''
    bin_nd={}
    for band,iband,req in zip(['g','r','z'],[1,2,4],req_mags):
        bin_nd[band]={}
        bins= np.linspace(18.,req,num=15)
        bin_nd[band]['cnt'],junk= np.histogram(obj.t['decam_mag'][:,iband], bins=bins)
        bin_nd[band]['binc']= (bins[1:]+bins[:-1])/2.
        # bins and junk should be identical arrays
        assert( np.all(np.all((bins,junk),axis=0)) )
    # Plot
    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
    plt.subplots_adjust(wspace=0.25)
    for cnt,band in zip(range(3),['g','r','z']):
        ax[cnt].step(bin_nd[band]['binc'],bin_nd[band]['cnt']/deg2, \
                     where='mid',c='b',lw=2)
    #labels
    for cnt,band in zip(range(3),['g','r','z']):
        xlab=ax[cnt].set_xlabel('%s' % band, **kwargs.ax) 
    ylab=ax[0].set_ylabel('counts/deg2', **kwargs.ax)
    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()
    plt.savefig(os.path.join(obj.outdir,'n_per_deg2_%s.png' % name), \
                bbox_extra_artists=[xlab,ylab], **kwargs.save)
    plt.close()



