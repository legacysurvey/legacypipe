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

def plot_mog_and_sample(samp,Xall, name='test.png'): 
    '''Build a color-color plot.  Show the data on the left-hand panel and random draws from 
    the MoGs on the right-hand panel.'''
    sns.set(style='white', font_scale=1.6, palette='deep')
    col = sns.color_palette()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 5), sharey=True)
    grrange = (-0.2, 2.0)
    rzrange = (-0.4, 2.5)
    
    # show 6 components b/c 6th is noise
    from sklearn.mixture import GMM
    mog6 = GMM(n_components=6, covariance_type="full").fit(Xall)
    priors.add_MoG_curves(ax1, mog6.means_, mog6.covars_, mog6.weights_)
    ax1.set_xlim(rzrange)
    ax1.set_ylim(grrange)
    ax1.set_xlabel('r - z')
    ax1.set_ylabel('g - r')
    #ax1.legend(loc='lower right', prop={'size': 14}, labelspacing=0.25, markerscale=2)
    ax2.plot(samp[:,0], samp[:,1], 'o', label='5 comp, 10k Draws', c=col[0], markersize=3)
    ax2.set_xlim(rzrange)
    ax2.set_ylim(grrange)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel('r - z')
    ax2.set_ylabel('g - r')
    # ax2.legend(loc='lower right', prop={'size': 14}, labelspacing=0.25, markerscale=2)
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    print('Writing {}'.format(name))
    plt.savefig(name)
    plt.close()

def qa_plot_MoG(Xall, type='STAR',extra=False):
    '''Simple function to compute the Bayesian information criterion.'''
    nsamp= 1000
    name= 'qa-mog-sample-%s.png' % type
    if type=='STAR': 
        mog = priors._GaussianMixtureModel.load('legacypipe/data/star_colors_mog.fits')
        samp = mog.sample(nsamp)
        plot_mog_and_sample(samp,Xall, name=name)
        if extra: 
            # Higher accuracy sampling, but more time consuming and negligible improvment
            samp= mog.sample_full_pdf(nsamp)
            plot_mog_and_sample(samp,Xall, name=name.replace('-sample-','-sampleAllmogs-'))

def qa_plot_BIC(Xall,type='STAR'):
    '''Number componentes from Bayesian Information Criterion'''
    ncomp = np.arange(2, 10)
    bic = getbic(Xall, ncomp)
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    ax.plot(ncomp, bic, marker='s', ls='-')
    ax.set_xlim((0, 10))
    ax.set_xlabel('Number of Gaussian Components')
    ax.set_ylabel('Bayesian Information Criterion')
    plt.legend(labels=['Star g-r, r-z colors'])
    plt.tight_layout()
    name='qa-mog-bic-%s.png' % type
    print('Writing {}'.format(name))
    plt.savefig(name)
    plt.close()



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
qa_plot_BIC(Xall,type='STAR')
qa_plot_MoG(Xall, type='STAR',extra=True)
print('done')
  

