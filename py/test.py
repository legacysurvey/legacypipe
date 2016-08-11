import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os
import seaborn as sns
from astropy.io import fits
import sys
from sklearn.mixture import GMM 

import legacyanalysis.decals_sim_priors as priors

def star_colors():
    '''Model the g-r, r-z color-color sequence for stars'''
    # Build a sample of stars with good photometry from a single sweep.
    rbright = 18
    rfaint = 19.5
    swp_dir='/global/project/projectdirs/cosmo/data/legacysurvey/dr2/sweep/2.0'
    sweep = fits.getdata(os.path.join(swp_dir,'sweep-340p000-350p005.fits'), 1)
    keep = np.where((sweep['TYPE'].strip() == 'PSF')*
                    (np.sum((sweep['DECAM_FLUX'][:, [1,2,4]] > 0)*1, axis=1)==3)*
                    (np.sum((sweep['DECAM_ANYMASK'][:, [1,2,4]] > 0)*1, axis=1)==0)*
                    (np.sum((sweep['DECAM_FRACFLUX'][:, [1,2,4]] < 0.05)*1, axis=1)==3)*
                    (sweep['DECAM_FLUX'][:,2]<(10**(0.4*(22.5-rbright))))*
                    (sweep['DECAM_FLUX'][:,2]>(10**(0.4*(22.5-rfaint)))))[0]
    stars = sweep[keep]
    print('Found {} stars with good photometry.'.format(len(stars)))
    gg = 22.5-2.5*np.log10(stars['DECAM_FLUX'][:, 1])
    rr = 22.5-2.5*np.log10(stars['DECAM_FLUX'][:, 2])
    zz = 22.5-2.5*np.log10(stars['DECAM_FLUX'][:, 4])
    gr = gg - rr
    rz = rr - zz
    return np.array([rz, gr]).T # Npts x 2

def getbic(X, ncomp=[3]):
    '''Simple function to compute the Bayesian information criterion.'''
    bic = [GMM(n_components=nc, covariance_type="full").fit(X).bic(X) for nc in ncomp]
    #for ii in range(len(ncomp)):
    #    print(ncomp[ii], bic[ii])
    return bic

def plot_mog_and_sample(samp,Xall, src='STAR',name='test.png'): 
    '''Build a color-color plot.  Show the data on the left-hand panel and random draws from 
    the MoGs on the right-hand panel.'''
    sns.set(style='white', font_scale=1.6, palette='deep')
    col = sns.color_palette()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 5), sharey=True)
    
    from sklearn.mixture import GMM
    # show ncomp+1 b/c +1 shows noise contribution
    if src == 'STAR':
        ncomp=6
        grrange = (-0.2, 2.0)
        rzrange = (-0.4, 2.5)
        xlab='r - z'
        ylab='g - r'
    elif src == 'ELG':
        ncomp=7
        grrange = (-0.2, 2.0)
        rzrange = (-0.4, 2.5)
        xlab='r - z'
        ylab='g - r'
    else: raise ValueError('src=%s not supported' % src)
    mog = GMM(n_components=ncomp, covariance_type="full").fit(Xall)
    priors.add_MoG_curves(ax1, mog.means_, mog.covars_, mog.weights_, label='ncomp+1')
    ax1.set_xlim(rzrange)
    ax1.set_ylim(grrange)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    #ax1.legend(loc='lower right', prop={'size': 14}, labelspacing=0.25, markerscale=2)
    ax2.plot(samp[:,0], samp[:,1], 'o', label='%d comp, %.2g Draws' % (ncomp-1,float(len(samp[:,0]))), c=col[0], markersize=3)
    ax2.set_xlim(rzrange)
    ax2.set_ylim(grrange)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel(xlab)
    ax2.set_ylabel(ylab)
    # ax2.legend(loc='lower right', prop={'size': 14}, labelspacing=0.25, markerscale=2)
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    print('Writing {}'.format(name))
    plt.savefig(name)
    plt.close()

def qa_plot_MoG(Xall, src='STAR',extra=False):
    '''Simple function to compute the Bayesian information criterion.'''
    nsamp= 10000
    name= 'qa-mog-sample-%s.png' % src
    if src=='STAR': 
        mog = priors._GaussianMixtureModel.load('legacypipe/data/star_colors_mog.fits')
        samp = mog.sample(nsamp)
        plot_mog_and_sample(samp,Xall, src=src,name=name)
        if extra: 
            # Higher accuracy sampling, but more time consuming and negligible improvment
            samp= mog.sample_full_pdf(nsamp)
            plot_mog_and_sample(samp,Xall, src=src,name=name.replace('-sample-','-sampleAllmogs-'))
    elif src=='ELG':
        mog = priors._GaussianMixtureModel.load('legacypipe/data/elg_colors_mog.fits')
        samp = mog.sample(nsamp)
        plot_mog_and_sample(samp,Xall, src=src,name=name)
        plot_priors(src=src)
        

def qa_plot_BIC(Xall,src='STAR'):
    '''Number componentes from Bayesian Information Criterion'''
    ncomp = np.arange(2, 10)
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
    else: raise ValueError('src=%s not supportd' % src)
    plt.tight_layout()
    name='qa-mog-bic-%s.png' % src
    print('Writing {}'.format(name))
    plt.savefig(name)
    plt.close()


def qa_plot_Priors(d=None,src='ELG'):
    '''d -- dictionary of morphology params'''
    assert(d is not None)
    fig,ax = plt.subplots(1, 3, figsize=(10., 5))
    col = sns.color_palette()
    if src == 'ELG':
        cut='r50 > 0'
        grrange = (-0.2, 2.0)
        rzrange = (-0.4, 2.5)
    else: raise ValueError('src=%s not supported')
    ax[1].set_title(cut)
    # plot
    ax[0].scatter(d['rz'], d['gr'], s=3, color=col[0])
    ax[0].set_xlim(rzrange)
    ax[0].set_ylim(grrange)
    ax[0].set_xlabel('r - z')
    ax[0].set_ylabel('g - r')
    ax[0].legend(loc='upper left', prop={'size': 14}, labelspacing=0.25, markerscale=2)

    ax[1].scatter(d['n'], d['r50'], s=3, color=col[0])
    ax[1].set_xlabel('sersic n')
    ax[1].set_ylabel('r50 (arcsec)')

    ax[2].scatter(d['n'], d['ba'], s=3, color=col[0])
    ax[2].set_xlabel('sersic n')
    ax[2].set_ylabel('ba (Minor/Major)')

    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.tight_layout()
    name='qa-priors-%s.png' % src
    print('Writing {}'.format(name))
    plt.savefig(name)
    plt.close()



############
# Stars
Xall= star_colors()
# Choose 5 compoenents
mog = GMM(n_components=5, covariance_type="full").fit(Xall)
star_mogfile= 'legacypipe/data/star_colors_mog.fits'
if os.path.exists(star_mogfile):
    print('STAR MoG exists, not overwritting: %s' % star_mogfile)
else:
    print('Writing {}'.format(star_mogfile))
    priors._GaussianMixtureModel.save(mog, star_mogfile)
# QA plots
qa_plot_BIC(Xall,src='STAR')
qa_plot_MoG(Xall, src='STAR') #,extra=True)
############
# ELGs
# Use DEEP2 ELGs whose SEDs have been modeled.
def elg_data():
    elgs = fits.getdata('/project/projectdirs/desi/spectro/templates/basis_templates/v2.2/elg_templates_v2.0.fits', 1)
    keep = elgs['radius_halflight'] > 0
    print('Grabbed {} ELGs, of which {} have HST morphologies.'.format(len(elgs), len(elgs[morph])))
    # Colors
    gg = elgs['DECAM_G']
    rr = elgs['DECAM_R']
    zz = elgs['DECAM_Z']
    gr = gg - rr
    rz = rr - zz
    Xall = np.array([rz, gr]).T
    # Morphology
    morph= {}
    morph['rz'] = rz[keep]
    morph['gr'] = gr[keep]
    morph['r50'] = elgs['RADIUS_HALFLIGHT'][keep] #arcsec
    morph['n'] = elgs['SERSICN'][keep]                            
    morph['ba'] = elgs['AXIS_RATIO'][keep] #minor/major
    return Xall,morph                            

Xall, morph= elg_data()
# Choose 6 comps
mog = GMM(n_components=6, covariance_type="full").fit(Xall)
elg_mogfile='legacypipe/data/elg_colors_mog.fits'
if os.path.exists(elg_mogfile):
    print('ELG MoG exists, not overwritting: %s' % elg_mogfile)
else:
    print('Writing {}'.format(elg_mogfile))
    priors._GaussianMixtureModel.save(mog, elg_mogfile)
# QA plots
qa_plot_BIC(Xall, src='ELG')
qa_plot_MoG(Xall, src='ELG') #,extra=True)
qa_plot_Priors(d=morph,src='ELG')
 
print('done')
  

