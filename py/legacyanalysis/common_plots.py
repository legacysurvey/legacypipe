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
kwargs= dict(ax= dict(fontweight='bold',fontsize='medium'),\
             text=dict(fontweight='bold',fontsize='medium',va='top',ha='left'),\
             leg=dict(frameon=True,fontsize='x-small'),\
             save= dict(bbox_inches='tight',dpi=150))
##########

# Helpful functions for making plots
def bin_up(data_bin_by,data_for_percentile, bin_minmax=(18.,26.),nbins=20):
    '''bins "data_for_percentile" into "nbins" using "data_bin_by" to decide how indices are assigned to bins
    returns bin center,N,q25,50,75 for each bin
    '''
    q25,q50,q75,N= [],[],[],[]
    bin_edges= np.linspace(bin_minmax[0],bin_minmax[1],num= nbins)
    for i,low,hi in zip(range(nbins-1), bin_edges[:-1],bin_edges[1:]):
        keep= np.all((low <= data_bin_by,data_bin_by < hi),axis=0)
        if np.where(keep)[0].size > 0:
            N+= [np.where(keep)[0].size]
            q25+= np.percentile(data_for_percentile[keep],q=25)
            q50+= np.percentile(data_for_percentile[keep],q=50)
            q75+= np.percentile(data_for_percentile[keep],q=75)
        else:
            N+= [np.nan]
            q25+= [np.nan]
            q50+= [np.nan] 
            q75+= [np.nan] 
    return (bin_edges[1:]+bin_edges[:-1])/2.,\
            np.array(N),np.array(q25),np.array(q50),np.array(q75)

# Main plotting functions 
def nobs(obj, name=''):
    '''make histograms of nobs so can compare depths of g,r,z between the two catalogues
    obj -- Single_TractorCat()'''   
    global kwargs
    hi= np.max(obj.t['decam_nobs'][:,[1,2,4]])
    fig,ax= plt.subplots(3,1)
    for i, band,iband in zip(range(3),['g','r','z'],[1,2,4]):
        ax[i].hist(obj.t['decam_nobs'][:,iband],\
                   bins=hi+1,normed=True,cumulative=True,align='mid')
        xlab=ax[i].set_xlabel('nobs %s' % band, **kwargs.ax)
        ylab=ax[i].set_ylabel('CDF', **kwargs.ax)
    plt.savefig(os.path.join(obj.outdir,'nobs_%s.png' % name)), \
                bbox_extra_artists=[xlab,ylab], **kwargs.save)
    plt.close()

def radec(obj,name=''): 
    '''ra,dec distribution of objects
    obj -- Single_TractorCat()'''
    global kwargs
    plot.scatter(obj.t['ra'], obj.t['dec'], \
                edgecolor='b',c='none',lw=1.)
    xlab=plt.xlabel('RA', **kwargs.ax)
    ylab=plt.ylabel('DEC', **kwargs.ax)
    plt.savefig(os.path.join(obj.outdir,'radec_%s.png' % name), \
                bbox_extra_artists=[xlab,ylab], **kwargs.save)
    plt.close()


def hist_types(obj, name=''):
    '''number of psf,exp,dev,comp, etc
    obj -- Single_TractorCat()'''
    global kwargs
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
    name='hist_types_Matched%s.png' % addname
    plt.savefig(os.path.join(obj.outdir,'hist_types_%s.png' % name), \
                bbox_extra_artists=[ylab], **kwargs.save)
    plt.close()


def sn_vs_mag(obj, mag_minmax=(18.,26.),name=''):
    '''plots Signal to Noise vs. mag for each band
    obj -- Single_TractorCat()'''
    global kwargs
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
        ax[cnt].plot(bin_SN[band]['binc'], bin_SN[band]['q50'],c=color,ls='-',lw=2,label=lab)
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
    global kwargs
    cm,names= create_confusion_matrix(ref_obj,test_obj)
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
    plt.savefig(os.path.join(info.outdir,'confusion_matrix_%s.png' % name), \
                bbox_extra_artists=[xlab,ylab], **kwargs.save)
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


text_args= dict(verticalalignment='center',fontsize=8)
def plot_dflux_chisq(b,type='psf', low=-8.,hi=8.,addname=''):
    #join indices b/c matched
    i_type= np.all((indices_for_type(b, inst='m_decam',type=type),\
                    indices_for_type(b, inst='m_bokmos',type=type)), axis=0) #both bokmos and decam of same type
    #get flux diff for each band
    hist= dict(g=0,r=0,z=0)
    binc= dict(g=0,r=0,z=0)
    stats=dict(g=0,r=0,z=0)
    #chi 
    sample,mag={},{}
    for band in ['g','r','z']:
        sample[band]= (b['m_decam'].data[band+'flux'][i_type]-b['m_bokmos'].data[band+'flux'][i_type])/np.sqrt(\
                    np.power(b['m_decam'].data[band+'flux_ivar'][i_type],-1)+np.power(b['m_bokmos'].data[band+'flux_ivar'][i_type],-1))
        mag[band]= 22.5-2.5*np.log10(b['m_decam'].data[band+'flux'][i_type])
    #loop over mag bins, one 3 panel for each mag bin
    for b_low,b_hi in zip([18,19,20,21,22,23],[19,20,21,22,23,24]):
        #plot each filter
        for band in ['g','r','z']:
            imag= np.all((b_low <= mag[band],mag[band] < b_hi),axis=0)
            #print("len(imag)=",len(imag),"len(sample)=",len(sample),"len(sample[imag])=",len(sample[imag]))
            hist[band],bins,junk= plt.hist(sample[band][imag],range=(low,hi),bins=50,normed=True)
            db= (bins[1:]-bins[:-1])/2
            binc[band]= bins[:-1]+db
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
        #labels
        for cnt,band in zip(range(3),['g','r','z']):
            if band == 'r': xlab=ax[cnt].set_xlabel(r'%s  $(F_{d}-F_{bm})/\sqrt{\sigma^2_{d}+\sigma^2_{bm}}$' % band, **laba)
            else: xlab=ax[cnt].set_xlabel('%s' % band, **laba)
            #xlab=ax[cnt].set_xlabel('%s' % band, **laba)
            ax[cnt].set_ylim(0,0.6)
            ax[cnt].set_xlim(low,hi)
        ylab=ax[0].set_ylabel('PDF', **laba)
        ti=ax[1].set_title("%s (%.1f <= %s < %.1f)" % (type,b_low,band,b_hi),**laba)
        #put stats in suptitle
        plt.savefig(os.path.join(get_outdir('bmd'),'dflux_chisq_%s_%.1f-%s-%.1f%s.png' % (type,b_low,band,b_hi,addname)), bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
        plt.close()


#def dflux_chisq(info,obj_type='PSF', low=-8.,hi=8.,addname=''):
#    #get flux diff for each band
#    hist= dict(g=0,r=0,z=0)
#    binc= dict(g=0,r=0,z=0)
#    stats=dict(g=0,r=0,z=0)
#    # Cut to objects that are both obj_type and both kept in ref,test 
#    print('dflux_chisq: np.where(info.ds[both][obj_type][0].size= ',np.where(info.ds['both'][obj_type])[0].size, 'same for both keep= ', np.where(info.ds['both']['keep'])[0].size)
#    i= np.all((info.ds['both'][obj_type],\
#               info.ds['both']['keep']),axis=0)
#    print('np.where(i)[0].size=',np.where(i)[0].size)
#    for band,iband in zip(['g','r','z'],[1,2,4]):
#        sample=(info.data['ref_matched']['decam_flux'][:,iband][i]-\
#                    info.data['test_matched']['decam_flux'][:,iband][i])/\
#                np.sqrt(np.power(info.data['ref_matched']['decam_flux_ivar'][:,iband][i],-1)+\
#                        np.power(info.data['test_matched']['decam_flux_ivar'][:,iband][i],-1))
#        hist[band],bins,junk= plt.hist(sample,range=(low,hi),bins=50,normed=True)
#        db= (bins[1:]-bins[:-1])/2
#        binc[band]= bins[:-1]+db
#        stats[band]= sample_gauss_stats(sample)
#    plt.close() #b/c plt.hist above
#    #for drawing unit gaussian N(0,1)
#    G= sp_stats.norm(0,1)
#    xvals= np.linspace(low,hi)
#    #plot
#    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
#    plt.subplots_adjust(wspace=0.25)
#    for cnt,band in zip(range(3),['g','r','z']):
#        ax[cnt].step(binc[band],hist[band], where='mid',c='b',lw=2)
#        ax[cnt].plot(xvals,G.pdf(xvals))
#        #for yloc,key in zip([0.95,0.85,0.75,0.65,0.55],['mean','std','q25','q75','perc_out']):
#        #    ax[cnt].text(0.1,yloc,"%s %.2f" % (key,stats[band]['sample'][key]),transform=ax[cnt].transAxes,horizontalalignment='left',**text_args)
#    #ax[2].text(0.9,0.95,"N(0,1)",transform=ax[2].transAxes,horizontalalignment='right',**text_args)
#    #for yloc,key in zip([0.85,0.75,0.65,0.55],['mean','std','q25','perc_out']):
#    #    ax[2].text(0.9,yloc,"%s %.2f" % (key,stats['g']['gauss'][key]),transform=ax[2].transAxes,horizontalalignment='right',**text_args)
#    #labels
#    for cnt,band in zip(range(3),['g','r','z']):
#        if band == 'r': xlab=ax[cnt].set_xlabel(r'%s  $(F_{d}-F_{bm})/\sqrt{\sigma^2_{d}+\sigma^2_{bm}}$' % band, **info.kwargs.ax)
#        else: xlab=ax[cnt].set_xlabel('%s' % band, **info.kwargs.ax)
#        #xlab=ax[cnt].set_xlabel('%s' % band, **laba)
#        ax[cnt].set_ylim(0,0.6)
#        ax[cnt].set_xlim(low,hi)
#    ylab=ax[0].set_ylabel('PDF', **info.kwargs.ax)
#    ti=ax[1].set_title(type,**info.kwargs.ax)
#    #put stats in suptitle
#    plt.savefig(os.path.join(info.outdir,'dflux_chisq_%s%s.png' % (type,addname)), \
#                bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
#    plt.close()

def plot_magRatio_vs_mag(b,type='psf',addname=''):
    #join indices b/c matched
    i_type= np.all((indices_for_type(b, inst='m_decam',type=type),\
                    indices_for_type(b, inst='m_bokmos',type=type)), axis=0) #both bokmos and decam of same type
    #plot
    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
    plt.subplots_adjust(wspace=0.25)
    for cnt,band in zip(range(3),['g','r','z']):
        magRatio= np.log10(b['m_bokmos'].data[band+'flux'][i_type])/np.log10(b['m_decam'].data[band+'flux'][i_type]) -1.
        mag= 22.5-2.5*np.log10(b['m_decam'].data[band+'flux'][i_type])
        ax[cnt].scatter(mag,magRatio, c='b',edgecolor='b',s=5) #,c='none',lw=2.)
    #labels
    for cnt,band in zip(range(3),['g','r','z']):
        xlab=ax[cnt].set_xlabel('%s AB' % band, **laba)
        ax[cnt].set_ylim(-0.5,0.5)
        ax[cnt].set_xlim(18,26)
    ylab=ax[0].set_ylabel(r'$m_{bm}/m_d - 1$', **laba)
    ti=ax[1].set_title("%s" % type,**laba)
    #put stats in suptitle
    plt.savefig(os.path.join(get_outdir('bmd'),'magRatio_vs_mag_%s%s.png' % (type,addname)), bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()

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


#
#
#def N_per_deg2(info,obj_type='PSF',req_mags=[24.,23.4,22.5],addname=''):
#    '''image requirements grz<=24,23.4,22.5
#    compute number density in each bin for each band mag [18,requirement]'''
#    bin_nd=dict(ref={},test={})
#    for key in bin_nd.keys():
#        bin_nd[key]={}
#        for band,iband,req in zip(['g','r','z'],[1,2,4],req_mags):
#            bin_nd[key][band]={}
#            bin_edges= np.linspace(18.,req,num=15)
#            i_m= np.all((info.ds[key+'_matched'].type[obj_type],\
#                         info.ds[key+'_matched'].keep),axis=0)
#            i_u= np.all((info.ds[key+'_missed'].type[obj_type], \
#                         info.ds[key+'_missed'].keep),axis=0)
#            # Join m_decam,u_decam OR m_bokmos,u_bokmos 
#            sample= np.ma.concatenate((info.ds[key+'_matched'].magAB[band][i_m], \
#                                       info.ds[key+'_missed'].magAB[band][i_u]),\
#                                       axis=0)
#            bin_nd[key][band]['binc'],bin_nd[key][band]['cnt'],q25,q50,q75=\
#                    bin_up(sample,sample,bin_edges=bin_edges)
#    #plot
#    fig,ax=plt.subplots(1,3,figsize=(9,3),sharey=True)
#    plt.subplots_adjust(wspace=0.25)
#    for cnt,band in zip(range(3),['g','r','z']):
#        for key,color,lab in zip(bin_nd.keys(),['b','g'],[info.ref_name,info.test_name]):
#            ax[cnt].step(bin_nd[key][band]['binc'],bin_nd[key][band]['cnt']/info.data['deg2'][key], \
#            where='mid',c=color,lw=2,label=lab)
#    #labels
#    for cnt,band in zip(range(3),['g','r','z']):
#        xlab=ax[cnt].set_xlabel('%s' % band) #, **laba)
#        #ax[cnt].set_ylim(0,0.6)
#        #ax[cnt].set_xlim(maglow,maghi)
#    ax[0].legend(loc='upper left', **info.kwargs.leg)
#    ylab=ax[0].set_ylabel('counts/deg2') #, **laba)
#    ti=plt.suptitle("%ss" % obj_type,**info.kwargs.ax)
#    # Make space for and rotate the x-axis tick labels
#    fig.autofmt_xdate()
#    #put stats in suptitle
#    plt.savefig(os.path.join(info.outdir,'n_per_deg2_%s%s.png' % (type,addname)), bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
#    plt.close()


