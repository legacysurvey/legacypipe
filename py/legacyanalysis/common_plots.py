'''functions that make validation plots
input -- object like BassMzls_Decals_Info() containing data,ds,ref_name,test_name,outdir

Example:
use class BassMzls_Decals_Info() to get data and relevant info for Bass/Mzls v Decals comparison
info=BassMzls_Decals_Info()
import common_plots as plots
plots.nobs(info)
'''
import matplotlib
matplotlib.use('Agg') #display backend
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from scipy import stats as sp_stats

# Object containg axis,legend, etc kwargs
class PlotKwargs(object):
    def __init__(self):
        self.ax=dict(fontweight='bold',fontsize='medium')
        self.text=dict(fontweight='bold',fontsize='medium',va='top',ha='left')
        self.leg=dict(frameon=True,fontsize='x-small')

# Helpful funcs for making plots
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


def create_confusion_matrix(info):
    '''compares MATCHED decam (truth) to bokmos (prediction)
    return 5x5 confusion matrix and colum/row names
    obj[m_decam'] is DECaLS object'''
    cm=np.zeros((5,5))-1
    types=['PSF','SIMP','EXP','DEV','COMP']
    for i_dec,dec_type in enumerate(types):
        ref_cut= np.all((info.ds['ref_matched'].type[dec_type],\
                         info.ds['ref_matched'].keep),axis=0)
        ind= np.where( ref_cut )[0]
        for i_bass,bass_type in enumerate(types):
            test_cut= np.all((info.ds['test_matched'].type[bass_type],\
                              info.ds['test_matched'].keep),axis=0)
            n_bass= np.where( test_cut )[0].size
            if ind.size > 0: cm[i_dec,i_bass]= float(n_bass)/ind.size #ind.size is constant for each loop over bass_types
            else: cm[i_dec,i_bass]= np.nan
    return cm,types



# Main plotting routines 
def nobs(info):
    '''make histograms of nobs so can compare depths of g,r,z between the two catalogues'''   
    hi=0 
    for key in ['ref_matched','test_matched']:
        for band,iband in zip(['g','r','z'],[1,2,4]):
            hi= np.max((hi, info.data[key]['decam_nobs'][:,iband].max()))
    bins= hi
    for key in ['ref_matched','test_matched']:
        for band,iband in zip(['g','r','z'],[1,2,4]):
            junk=plt.hist(info.data[key]['decam_nobs'][:,iband],\
                        bins=bins,normed=True,cumulative=True,align='mid')
            xlab=plt.xlabel('nobs %s' % band, **info.kwargs.ax)
            ylab=plt.ylabel('CDF', **info.kwargs.ax)
            plt.savefig(os.path.join(info.outdir,'hist_nobs_%s_%s.png' % (band,cam[2:])), \
                    bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()

def radec(info,addname=''): 
    '''obj[m_types] -- DECaLS() objects with matched OR unmatched indices'''
    bcut={}
    for key  in ['ref_matched','ref_missed','test_missed']:
        bcut[key]= info.ds[key].keep
    #set seaborn panel styles
    #sns.set_style('ticks',{"axes.facecolor": ".97"})
    #sns.set_palette('colorblind')
    #setup plot
    fig,ax=plt.subplots(1,2,figsize=(9,6),sharey=True,sharex=True)
    plt.subplots_adjust(wspace=0.25)
    #plt.subplots_adjust(wspace=0.5)
    #plot
    ax[0].scatter(info.data['ref_matched']['ra'][ bcut['ref_matched'] ], \
                  info.data['ref_matched']['dec'][ bcut['ref_matched'] ], \
                edgecolor='b',c='none',lw=1.)
    ax[1].scatter(info.data['ref_missed']['ra'][ bcut['ref_missed'] ], \
                  info.data['ref_missed']['dec'][ bcut['ref_missed'] ], \
                edgecolor='b',c='none',lw=1.,label=info.ref_name)
    ax[1].scatter(info.data['test_missed']['ra'][ bcut['test_missed'] ], \
                  info.data['test_missed']['dec'][ bcut['test_missed'] ], \
                edgecolor='g',c='none',lw=1.,label=info.test_name)
    for cnt,ti in zip(range(2),['Matched','Unmatched']):
        ti=ax[cnt].set_title(ti,**info.kwargs.ax)
        xlab=ax[cnt].set_xlabel('RA', **info.kwargs.ax)
    ylab=ax[0].set_ylabel('DEC', **info.kwargs.ax)
    ax[0].legend(loc='upper left',**info.kwargs.leg)
    #save
    #sns.despine()
    plt.savefig(os.path.join(info.outdir,'radec%s.png' % addname), \
            bbox_extra_artists=[xlab,ylab,ti], bbox_inches='tight',dpi=150)
    plt.close()


def HistTypes(info,addname=''):
    '''decam,bokmos -- DECaLS() objects with matched OR unmatched indices'''
    #matched or unmatched objects
    #c2=sns.color_palette()[0] #'b'
    c1= 'b' 
    c2= 'r'
    ###
    types= ['PSF','SIMP','EXP','DEV','COMP']
    ind = np.arange(len(types))  # the x locations for the groups
    width = 0.35       # the width of the bars
    ###
    ht_ref, ht_test= np.zeros(5,dtype=int),np.zeros(5,dtype=int)
    for cnt,typ in enumerate(types):
        bcut= np.all((info.ds['ref_matched'].type[typ],\
                      info.ds['ref_matched'].keep),axis=0)
        ht_ref[cnt]= len(info.data['ref_matched']['ra'][ bcut ])/ \
                        float(info.data['deg2']['ref'])
        bcut= np.all((info.ds['test_matched'].type[typ],\
                      info.ds['test_matched'].keep),axis=0)
        ht_test[cnt]= len(info.data['test_matched']['ra'][ bcut ])/ \
                        float(info.data['deg2']['test'])
    ###
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, ht_ref, width, color=c1)
    rects2 = ax.bar(ind + width, ht_test, width, color=c2)
    ylab= ax.set_ylabel("counts/deg2")
    ti= ax.set_title('Matched')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(types)
    ax.legend((rects1[0], rects2[0]), (info.ref_name, info.test_name),**info.kwargs.leg)
    #save
    name='hist_types_Matched%s.png' % addname
    plt.savefig(os.path.join(info.outdir,name), \
                bbox_extra_artists=[ylab,ti], bbox_inches='tight',dpi=150)
    plt.close()


def SN_vs_mag(info,obj_type='PSF', addname=''):
    ''''''
    #bin up SN values
    min,max= 18.,25.
    bin_SN=dict(ref_matched={},test_matched={})
    for key in bin_SN.keys():
        for band,iband in zip(['g','r','z'],[1,2,4]):
            bin_SN[key][band]={}
            bcut= np.all((info.ds[key].type[obj_type],\
                          info.ds[key].keep),axis=0)
            bin_edges= np.linspace(min,max,num=30)
            bin_SN[key][band]['binc'],count,bin_SN[key][band]['q25'],bin_SN[key][band]['q50'],bin_SN[key][band]['q75']=\
                    bin_up(info.ds[key].magAB[band][bcut], \
                           info.data[key]['decam_flux'][:,iband][bcut]*np.sqrt(info.data[key]['decam_flux_ivar'][:,iband][bcut]),\
                                bin_edges=bin_edges)
    #setup plot
    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
    plt.subplots_adjust(wspace=0.25)
    #plot SN
    for cnt,band in zip(range(3),['g','r','z']):
        #horiz line at SN = 5
        ax[cnt].plot([1,40],[5,5],'k--',lw=2)
        #data
        for inst,color,lab in zip(bin_SN.keys(),['b','g'],[info.ref_name,info.test_name]):
            ax[cnt].plot(bin_SN[inst][band]['binc'], bin_SN[inst][band]['q50'],c=color,ls='-',lw=2,label=lab)
            ax[cnt].fill_between(bin_SN[inst][band]['binc'],bin_SN[inst][band]['q25'],bin_SN[inst][band]['q75'],color=color,alpha=0.25)
    #labels
    ax[2].legend(loc=1,**info.kwargs.leg)
    for cnt,band in zip(range(3),['g','r','z']):
        ax[cnt].set_yscale('log')
        xlab=ax[cnt].set_xlabel('%s' % band, **info.kwargs.ax)
        ax[cnt].set_ylim(1,100)
        ax[cnt].set_xlim(20.,26.)
    ylab=ax[0].set_ylabel('S/N', **info.kwargs.ax)
    ax[2].text(26,5,'S/N = 5  ',**info.kwargs.text)
    plt.savefig(os.path.join(info.outdir,'sn_%s%s.png' % (obj_type,addname)), bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()

def confusion_matrix(cm,ticknames, info,addname=''):
    '''cm -- NxN array containing the Confusion Matrix values
    ticknames -- list of strings of length == N, column and row names for cm plot'''
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues,vmin=0,vmax=1)
    cbar=plt.colorbar()
    plt.xticks(range(len(ticknames)), ticknames)
    plt.yticks(range(len(ticknames)), ticknames)
    ylab=plt.ylabel('True (%s)' % info.ref_name)
    xlab=plt.xlabel('Predicted (%s)' % info.test_name)
    for row in range(len(ticknames)):
        for col in range(len(ticknames)):
            if np.isnan(cm[row,col]):
                plt.text(col,row,'n/a',va='center',ha='center')
            elif cm[row,col] > 0.5:
                plt.text(col,row,'%.2f' % cm[row,col],va='center',ha='center',color='yellow')
            else:
                plt.text(col,row,'%.2f' % cm[row,col],va='center',ha='center',color='black')
    plt.savefig(os.path.join(info.outdir,'confusion_matrix%s.png' % addname), \
                bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()



def matched_separation_hist(info):
    '''d12 is array of distances in degress between matched objects'''
    #pixscale to convert d12 into N pixels
    pixscale=dict(decam=0.25,bokmos=0.45)
    #sns.set_style('ticks',{"axes.facecolor": ".97"})
    #sns.set_palette('colorblind')
    #setup plot
    fig,ax=plt.subplots()
    #plot
    ax.hist(info.data['d_matched']*3600,bins=50,color='b',align='mid')
    ax2 = ax.twiny()
    ax2.hist(info.data['d_matched']*3600./pixscale['bokmos'],bins=50,color='g',align='mid',\
                visible=False)
    xlab= ax.set_xlabel("arcsec")
    xlab= ax2.set_xlabel("pixels [BASS]")
    ylab= ax.set_ylabel("Matched")
    #save
    #sns.despine()
    plt.savefig(os.path.join(info.outdir,"separation_hist.png"), \
                    bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()

def psf_hists(info):
    '''decam,bokmos are DECaLS() objects matched to decam ra,dec'''
    #divide into samples of 0.25 mag bins, store q50 of each
    width=0.25 #in mag
    low_vals= np.arange(20.,26.,width)
    med={}
    for b in ['g','r','z']: med[b]=np.zeros(low_vals.size)-100
    for i,low in enumerate(low_vals):
        for band in ['g','r','z']:
            ind= np.all((low <= info.ds['ref_matched'].magAB[band],\
                         info.ds['ref_matched'].magAB[band] < low+width),axis=0)
            if np.where(ind)[0].size > 0:
                med[band][i]= np.percentile(info.ds['ref_test'].magAB[band][ind] - \
                                            info.ds['ref_matched'].magAB[band][ind],q=50)
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
        xlab=ax[cnt].set_xlabel('bins of %s (%s)' % (band,info.ref_name), **info.kwargs.ax)
        ylab=ax[cnt].set_ylabel('q50[%s %s - %s]' % (band,info.test_name,info.ref_name), **info.kwargs.ax)
        if zoom: ax[cnt].set_ylim(-0.25,0.25)
    # sup=plt.suptitle('decam with matching bokmos',**laba)
    #save
    #sns.despine()
    name="median_color_diff.png"
    plt.savefig(os.path.join(info.outdir,name), \
                bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()


def dflux_chisq(info,obj_type='PSF', low=-8.,hi=8.,addname=''):
    #get flux diff for each band
    hist= dict(g=0,r=0,z=0)
    binc= dict(g=0,r=0,z=0)
    stats=dict(g=0,r=0,z=0)
    # Cut to objects that are both obj_type and both kept in ref,test 
    print('dflux_chisq: np.where(info.ds[both][obj_type][0].size= ',np.where(info.ds['both'][obj_type])[0].size, 'same for both keep= ', np.where(info.ds['both']['keep'])[0].size)
    i= np.all((info.ds['both'][obj_type],\
               info.ds['both']['keep']),axis=0)
    print('np.where(i)[0].size=',np.where(i)[0].size)
    for band,iband in zip(['g','r','z'],[1,2,4]):
        sample=(info.data['ref_matched']['decam_flux'][:,iband][i]-\
                    info.data['test_matched']['decam_flux'][:,iband][i])/\
                np.sqrt(np.power(info.data['ref_matched']['decam_flux_ivar'][:,iband][i],-1)+\
                        np.power(info.data['test_matched']['decam_flux_ivar'][:,iband][i],-1))
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
        if band == 'r': xlab=ax[cnt].set_xlabel(r'%s  $(F_{d}-F_{bm})/\sqrt{\sigma^2_{d}+\sigma^2_{bm}}$' % band, **info.kwargs.ax)
        else: xlab=ax[cnt].set_xlabel('%s' % band, **info.kwargs.ax)
        #xlab=ax[cnt].set_xlabel('%s' % band, **laba)
        ax[cnt].set_ylim(0,0.6)
        ax[cnt].set_xlim(low,hi)
    ylab=ax[0].set_ylabel('PDF', **info.kwargs.ax)
    ti=ax[1].set_title(type,**info.kwargs.ax)
    #put stats in suptitle
    plt.savefig(os.path.join(info.outdir,'dflux_chisq_%s%s.png' % (type,addname)), \
                bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()
################

def N_per_deg2(info,obj_type='PSF',req_mags=[24.,23.4,22.5],addname=''):
    '''image requirements grz<=24,23.4,22.5
    compute number density in each bin for each band mag [18,requirement]'''
    bin_nd=dict(ref={},test={})
    for key in bin_nd.keys():
        bin_nd[key]={}
        for band,iband,req in zip(['g','r','z'],[1,2,4],req_mags):
            bin_nd[key][band]={}
            bin_edges= np.linspace(18.,req,num=15)
            i_m= np.all((info.ds[key+'_matched'].type[obj_type],\
                         info.ds[key+'_matched'].keep),axis=0)
            i_u= np.all((info.ds[key+'_missed'].type[obj_type], \
                         info.ds[key+'_missed'].keep),axis=0)
            # Join m_decam,u_decam OR m_bokmos,u_bokmos 
            sample= np.ma.concatenate((info.ds[key+'_matched'].magAB[band][i_m], \
                                       info.ds[key+'_missed'].magAB[band][i_u]),\
                                       axis=0)
            bin_nd[key][band]['binc'],bin_nd[key][band]['cnt'],q25,q50,q75=\
                    bin_up(sample,sample,bin_edges=bin_edges)
    #plot
    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
    plt.subplots_adjust(wspace=0.25)
    for cnt,band in zip(range(3),['g','r','z']):
        for key,color,lab in zip(bin_nd.keys(),['b','g'],[info.ref_name,info.test_name]):
            ax[cnt].step(bin_nd[key][band]['binc'],bin_nd[key][band]['cnt']/info.data['deg2'][key], \
            where='mid',c=color,lw=2,label=lab)
    #labels
    for cnt,band in zip(range(3),['g','r','z']):
        xlab=ax[cnt].set_xlabel('%s' % band) #, **laba)
        #ax[cnt].set_ylim(0,0.6)
        #ax[cnt].set_xlim(maglow,maghi)
    ax[0].legend(loc='upper left', **info.kwargs.leg)
    ylab=ax[0].set_ylabel('counts/deg2') #, **laba)
    ti=plt.suptitle("%ss" % obj_type,**info.kwargs.ax)
    # Make space for and rotate the x-axis tick labels
    fig.autofmt_xdate()
    #put stats in suptitle
    plt.savefig(os.path.join(info.outdir,'n_per_deg2_%s%s.png' % (type,addname)), bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()


