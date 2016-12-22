import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os
from astropy.io import fits
import sys
from sklearn.mixture import GMM 
from pandas import DataFrame

import legacyanalysis.decals_sim_priors as priors

# Globals
xyrange=dict(x_star=[-0.5,2.2],\
             y_star=[-0.3,2.],\
             x_elg=[-0.5,2.2],\
             y_elg=[-0.3,2.],\
             x_lrg= [0, 2.5],\
             y_lrg= [-2, 6])
####


def getbic(X, ncomp=[3]):
    '''Simple function to compute the Bayesian information criterion.'''
    bic = [GMM(n_components=nc, covariance_type="full").fit(X).bic(X) for nc in ncomp]
    #for ii in range(len(ncomp)):
    #    print(ncomp[ii], bic[ii])
    return bic

def qa_plot_MoG(Xall,ncomp=2, src='STAR',nsamp=10000,outdir='.',extra=False,append=''): 
    '''Build a color-color plot.  Show the data on the left-hand panel and random draws from 
    the MoGs on the right-hand panel.'''
    if src == 'STAR':
        mog_file = 'legacypipe/data/star_colors_mog.fits'
        xrange = xyrange['x_%s' % src.lower()]
        yrange = xyrange['y_%s' % src.lower()]
        xlab='r - z'
        ylab='g - r'
    elif src == 'ELG':
        mog_file = 'legacypipe/data/elg_colors_mog.fits'
        xrange = xyrange['x_%s' % src.lower()]
        yrange = xyrange['y_%s' % src.lower()]
        xlab='r - z'
        ylab='g - r'
    elif src == 'LRG':
        mog_file = 'legacypipe/data/lrg_colors_mog.fits'
        xrange = xyrange['x_%s' % src.lower()]
        yrange = xyrange['y_%s' % src.lower()]
        xlab='r - z'
        ylab='r - w1'
    else: raise ValueError('src=%s not supported' % src)
    # Build MoG
    if ncomp is None:
        mog = priors._GaussianMixtureModel.load(mog_file)
    else:
        from sklearn.mixture import GMM
        mog = GMM(n_components=ncomp, covariance_type="full").fit(Xall)
    samp = mog.sample(n_samples=nsamp)
    #if extra: 
    #    # Higher accuracy sampling, but more time consuming and negligible improvment
    #    samp= mog.sample_full_pdf(nsamp)
    fig, ax = plt.subplots(1, 3, sharey=True,figsize=(12, 4))
    ax[0].plot(Xall[:,0],Xall[:,1], 'o', c='b', markersize=3)
    priors.add_MoG_curves(ax[1], mog.means_, mog.covars_, mog.weights_)
    ax[2].plot(samp[:,0], samp[:,1], 'o', c='b', markersize=3)
    # Add ts box
    if src != 'STAR':
        ts= priors.TSBox(src=src)
        for i in [0,2]: 
            ts.add_ts_box(ax[i], xlim=xrange,ylim=yrange)
    for i,title in zip(range(3),['Data','Gaussian Mixture','%d Draws' % nsamp]):
        xlab1=ax[i].set_xlabel(xlab)
        ax[i].set_xlim(xrange)
        ax[i].set_ylim(yrange)
        ti=ax[i].set_title(title)
    ylab1=ax[0].set_ylabel(ylab)
    fig.subplots_adjust(wspace=0) #, hspace=0.1)
    for i in range(2):
        priors.rm_last_ticklabel(ax[i])
    name= os.path.join(outdir,'qa-mog-sample-%s%s.png' % (src,append))
    print('Writing {}'.format(name))
    plt.savefig(name, bbox_extra_artists=[xlab1,ylab1,ti], bbox_inches='tight',dpi=150)
    plt.close()

def qa_plot_BIC(Xall,src='STAR',append=''):
    '''Number componentes from Bayesian Information Criterion'''
    ncomp = np.arange(1, 6)
    bic = getbic(Xall, ncomp)
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    ax.plot(ncomp, bic, marker='s', ls='-')
    ax.set_xlim((0, 10))
    ax.set_xlabel('Number of Gaussian Components')
    ax.set_ylabel('Bayesian Information Criterion')
    if src == 'STAR':
        plt.legend(labels=['%s g-r, r-z colors' % src])
    elif src == 'ELG':
        plt.legend(labels=['%s g-r, r-z colors' % src])
    elif src == 'LRG':
        plt.legend(labels=['%s r-w1, r-z colors' % src])
    else: raise ValueError('src=%s not supportd' % src)
    plt.tight_layout()
    name='qa-mog-bic-%s%s.png' % (src,append)
    print('Writing {}'.format(name))
    plt.savefig(name)
    plt.close()

def create_joinplot(df,xkey,ykey,xlab,ylab,xlim,ylim,color,src='ELG'):
    import seaborn as sns
    g = sns.JointGrid(x=xkey, y=ykey, data=df, xlim=xlim, ylim=ylim)
    g = g.plot_joint(plt.scatter, color=color, edgecolor="white")
    g = g.plot_marginals(sns.distplot, kde=False, color=color)
    g = g.set_axis_labels(xlab,ylab)
    def f_cut(junk1,junk2):
        return 0
    g = g.annotate(f_cut, template="Cut to r50 > {val:d}",\
                   stat="", loc="upper right", fontsize=12)
    name='qa-priors-%s-%s-%s.png' % (xkey,ykey,src)
    print('Writing {}'.format(name))
    g = g.savefig(name)

def qa_plot_Priors(d=None,src='ELG'):
    '''d -- dictionary of morphology params'''
    import seaborn as sns
    assert(d is not None)
    # JoinGrid needs pandas DataFrame
    df= DataFrame(d)
    if src == 'ELG':
        grrange = (-0.2, 2.0)
        rzrange = (-0.4, 2.5)
    else: raise ValueError('src=%s not supported')
    col = sns.color_palette()
    # Put each in sep plot window
    color=col[0]
    for xkey,ykey,xlab,ylab,xlim,ylim in zip(\
            ['rz','r50','ba'],['gr','n','n'],\
            ['r-z','Half-light radius (arcsec)','Axis ratio (b/a)'], ['g-r','Sersic n','Sersic n'],\
            [rzrange,(0,1.5),(0,1)], [grrange,(0,4),(0,4)]):
        create_joinplot(df,xkey,ykey,xlab,ylab,xlim,ylim,color,src=src)


if __name__ == "__main__":
    # Stars
    Xall= priors.star_data()
    qa_plot_BIC(Xall,src='STAR')
    qa_plot_MoG(Xall,ncomp=6, src='STAR') 
    # Save Model
    mog = GMM(n_components=6, covariance_type="full").fit(Xall)
    star_mogfile= 'legacypipe/data/star_colors_mog.fits'
    if os.path.exists(star_mogfile):
        print('STAR MoG exists, not overwritting: %s' % star_mogfile)
    else:
        print('Writing {}'.format(star_mogfile))
        # with 6 comp, 6th is noise, 1-5 are physical
        priors._GaussianMixtureModel.save(mog, star_mogfile,index=np.arange(5))
    qa_plot_MoG(Xall,ncomp=None, src='STAR',append='saved')
    # ELGs
    # FDR data
    Xall,cuts= priors.elg_data_for_FDR()
    priors.plot_FDR(Xall,cuts,src='ELG')
    b= cuts['any_elg']
    qa_plot_BIC(Xall[b,:], src='ELG',append='_FDR')
    qa_plot_MoG(Xall[b,:],ncomp=6, src='ELG',append='_FDR') #,extra=True)
    # Fit template spectra data
    Xall,cuts, morph= priors.elg_data()
    qa_plot_BIC(Xall, src='ELG',append='_synth')
    qa_plot_MoG(Xall,ncomp=3, src='ELG',append='_synth') #,extra=True)
    b= cuts['has_morph']
    qa_plot_BIC(Xall[b,:], src='ELG',append='_synth+morph')
    qa_plot_MoG(Xall[b,:],ncomp=4, src='ELG',append='_synth+morph') #,extra=True)
    # only have priors for morph cut
    qa_plot_Priors(d=morph,src='ELG')
    # Save 3 component synth MoG
    mog = GMM(n_components=3, covariance_type="full").fit(Xall)
    elg_mogfile='legacypipe/data/elg_colors_mog.fits'
    if os.path.exists(elg_mogfile):
        print('ELG MoG exists, not overwritting: %s' % elg_mogfile)
    else:
        print('Writing {}'.format(elg_mogfile))
        priors._GaussianMixtureModel.save(mog, elg_mogfile)
    qa_plot_MoG(Xall,ncomp=None, src='ELG',append='saved')
    # LRGs
    Xall,cuts= lrg_data_for_FDR()
    priors.plot_FDR(Xall,cuts,src='LRG')
    b= cuts['lrg']
    qa_plot_BIC(Xall[b,:], src='LRG')
    qa_plot_MoG(Xall[b,:], ncomp=2,src='LRG') #,extra=True)
    # Save 2 comp model
    b= cuts['lrg']
    mog = GMM(n_components=2, covariance_type="full").fit(Xall[b,:])
    lrg_mogfile= 'legacypipe/data/lrg_colors_mog.fits'
    if os.path.exists(lrg_mogfile):
        print('LRG MoG exists, not overwritting: %s' % lrg_mogfile)
    else:
        print('Writing {}'.format(lrg_mogfile))
        priors._GaussianMixtureModel.save(mog, lrg_mogfile)
    qa_plot_MoG(Xall,ncomp=None, src='LRG',append='saved')
     
    print('done')
  

