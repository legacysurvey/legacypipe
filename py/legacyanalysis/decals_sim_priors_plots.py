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
from scipy.optimize import newton

import legacyanalysis.decals_sim_priors as priors

# Globals
xyrange=dict(x_star=[-0.5,2.2],\
             y_star=[-0.3,2.],\
             x_elg=[-0.5,2.2],\
             y_elg=[-0.3,2.],\
             x_lrg= [0, 2.5],\
             y_lrg= [-2, 6])
####


class TSBox(object):
    '''functions to add Target Selection box to ELG, LRG, etc plot
    add_ts_box -- main functino to call'''
    def __init__(self,src='ELG'):
        self.src=src

    def add_ts_box(self, ax, xlim=None,ylim=None):
        '''draw color selection box'''
        assert(xlim is not None and ylim is not None)
        if self.src == 'ELG':
            #g-r vs. r-z
            xint= newton(self.ts_root,np.array([1.]),args=('y1-y2',))
            
            x=np.linspace(xlim[0],xlim[1],num=1000)
            y=np.linspace(ylim[0],ylim[1],num=1000)
            x1,y1= x,self.ts_box(x,'y1')
            x2,y2= x,self.ts_box(x,'y2')
            x3,y3= np.array([0.3]*len(x)),y
            x4,y4= np.array([0.6]*len(x)),y
            b= np.all((x >= 0.3,x <= xint),axis=0)
            x1,y1= x1[b],y1[b]
            b= np.all((x >= xint,x <= 1.6),axis=0)
            x2,y2= x2[b],y2[b]
            b= y3 <= np.min(y1)
            x3,y3= x3[b],y3[b]
            b= y4 <= np.min(y2)
            x4,y4= x4[b],y4[b]
            ax.plot(x1,y1,'k--',lw=2)
            ax.plot(x2,y2,'k--',lw=2)
            ax.plot(x3,y3,'k--',lw=2)
            ax.plot(x4,y4,'k--',lw=2) 
        elif self.src == 'LRG':
            #r-w1 vs. r-z
            x=np.linspace(xlim[0],xlim[1],num=1000)
            y=np.linspace(ylim[0],ylim[1],num=1000)
            x1,y1= x,self.ts_box(x,'y1')
            x2,y2= np.array([1.5]*len(x)),y
            b= x >= 1.5
            x1,y1= x1[b],y1[b]
            b= y2 >= np.min(y1)
            x2,y2= x2[b],y2[b]
            ax.plot(x1,y1,'k--',lw=2)
            ax.plot(x2,y2,'k--',lw=2)
        else: raise ValueError('src=%s not supported' % src)

    def ts_box(self, x,name):
        if self.src == 'ELG':
            if name == 'y1': return 1.15*x-0.15
            elif name == 'y2': return -1.2*x+1.6
            else: raise ValueError
        elif self.src == 'LRG':
            if name == 'y1': return 1.8*x-1.
            else: raise ValueError
        else: raise ValueError('src=%s not supported' % self.src)

    def ts_root(self,x,name):
        if self.src == 'ELG':
            if name == 'y1-y2': return self.ts_box(x,'y1')-self.ts_box(x,'y2')
            else: raise ValueError
        else: raise ValueError('non ELG not supported')

def rm_last_ticklabel(ax):
    '''for multiplot'''
    labels=ax.get_xticks().tolist()
    labels=np.array(labels).astype(float) #prevent from making float
    labels=list(labels)
    labels[-1]=''
    ax.set_xticklabels(labels)


def flux2mag(nanoflux):
    return 22.5-2.5*np.log10(nanoflux)

def elg_data_for_FDR():
    '''version 3.0 of data discussed in
    https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=912'''
    zcat = fits.getdata('/project/projectdirs/desi/target/analysis/deep2/v3.0/deep2-field1-oii.fits.gz', 1)
    # Cuts
    oiicut1 = 8E-17 # [erg/s/cm2]
    zmin = 0.6
    rfaint = 23.4
    loz = np.all((zcat['ZHELIO']<zmin,\
                  zcat['CFHTLS_R']<rfaint),axis=0)
    oiifaint = np.all((zcat['ZHELIO']>zmin,\
                       zcat['CFHTLS_R']<rfaint,\
                       zcat['OII_3727_ERR']!=-2.0,\
                       zcat['OII_3727']<oiicut1),axis=0)
    oiibright_loz = np.all((zcat['ZHELIO']>zmin,\
                            zcat['ZHELIO']<1.0,\
                            zcat['CFHTLS_R']<rfaint,\
                            zcat['OII_3727_ERR']!=-2.0,\
                            zcat['OII_3727']>oiicut1),axis=0)
    oiibright_hiz = np.all((zcat['ZHELIO']>1.0,\
                            zcat['CFHTLS_R']<rfaint,\
                            zcat['OII_3727_ERR']!=-2.0,\
                            zcat['OII_3727']>oiicut1),axis=0)
    any_elg= np.all((zcat['CFHTLS_R']<rfaint,\
                     zcat['OII_3727_ERR']!=-2.0,\
                     zcat['OII_3727']>oiicut1),axis=0)
    # color data
    rz= (zcat['CFHTLS_R'] - zcat['CFHTLS_Z'])
    gr= (zcat['CFHTLS_G'] - zcat['CFHTLS_R'])
    Xall = np.array([rz,gr]).T
    cuts=dict(loz=loz,\
              oiifaint=oiifaint,\
              oiibright_loz=oiibright_loz,\
              oiibright_hiz=oiibright_hiz,\
              any_elg=any_elg)
    return Xall,cuts            


def plot_FDR(Xall,cuts,src='ELG'):
    # Object to add target selection box
    ts= TSBox(src=src)
    import seaborn as sns
    sns.set(style='ticks', font_scale=1.6, palette='deep')
    col = sns.color_palette()
    fig, ax = plt.subplots()
    if src == 'ELG':
        # Set up figure
        xlab='r - z'
        ylab='g - r'
        xrange = xyrange['x_%s' % src.lower()]
        yrange = xyrange['y_%s' % src.lower()]
        # Plot
        # Add box
        ts.add_ts_box(ax, xlim=xrange,ylim=yrange)
        # Add points
        b= cuts['loz']
        ax.scatter(Xall[:,0][b],Xall[:,1][b], marker='^', color=col[2], label=r'$z<0.6$')

        b=cuts['oiifaint']
        ax.scatter(Xall[:,0][b],Xall[:,1][b], marker='s', color='tan',
                        label=r'$z>0.6, [OII]<8\times10^{-17}$')

        b= cuts['oiibright_loz']
        ax.scatter(Xall[:,0][b],Xall[:,1][b], marker='o', color='powderblue',
                        label=r'$z>0.6, [OII]>8\times10^{-17}$')

        b=cuts['oiibright_hiz']
        ax.scatter(Xall[:,0][b],Xall[:,1][b], marker='o', color='powderblue', edgecolor='black',
                        label=r'$z>1.0, [OII]>8\times10^{-17}$')
    elif src == 'LRG':
        # Set up figure
        xlab='r - z'
        ylab='r - w1'
        xrange = xyrange['x_%s' % src.lower()]
        yrange = xyrange['y_%s' % src.lower()]
        # Plot
        # Add box
        ts.add_ts_box(ax, xlim=xrange,ylim=yrange)
        # Add points
        b= cuts['lrg']
        ax.scatter(Xall[:,0][b],Xall[:,1][b], marker='^', color='b', label='LRG')
        #b= cuts['psf']
        #ax.scatter(Xall[:,0][b],Xall[:,1][b], marker='o', color='g', label='PSF')
    else: 
        raise ValueError('src=%s not supported' % src)
    # Finish labeling
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    xlab= ax.set_xlabel(xlab)
    ylab= ax.set_ylabel(ylab)
    ax.legend(loc='upper left', prop={'size': 14}, labelspacing=0.2,
              markerscale=1.5)
    name='%s_fdr.png' % src
    print('Writing {}'.format(name))
    kwargs= dict(bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.savefig(name, **kwargs)
    plt.close()


def star_data():
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

def elg_data():
    '''Use DEEP2 ELGs whose SEDs have been modeled.'''
    elgs = fits.getdata('/project/projectdirs/desi/spectro/templates/basis_templates/v2.2/elg_templates_v2.0.fits', 1)
    # Colors
    gg = elgs['DECAM_G']
    rr = elgs['DECAM_R']
    zz = elgs['DECAM_Z']
    gr = gg - rr
    rz = rr - zz
    Xall = np.array([rz, gr]).T
    # Cuts
    has_morph = elgs['radius_halflight'] > 0
    cuts= dict(has_morph=has_morph) 
    print('%d/%d of Fit Template Spectra ELGs have morphologies' % (len(elgs[has_morph]), len(elgs)))
    # Morphology
    morph= {}
    morph['rz'] = rz[has_morph]
    morph['gr'] = gr[has_morph]
    morph['r50'] = elgs['RADIUS_HALFLIGHT'][has_morph] #arcsec
    morph['n'] = elgs['SERSICN'][has_morph]                            
    morph['ba'] = elgs['AXIS_RATIO'][has_morph] #minor/major
    return Xall,cuts,morph                            


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
        ts= TSBox(src=src)
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
        rm_last_ticklabel(ax[i])
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

def lrg_data_for_FDR():
    '''various Truth catalogues have been used to make the FDR 
    and the LRG ts is being revised as I write, VIPERS covers more 
    deg2 than cosmos and should be plenty for our purposes'''
    dr2 = fits.getdata('/project/projectdirs/desi/target/analysis/truth/decals-dr2-vipers-w4.fits.gz', 1)
    vip = fits.getdata('/project/projectdirs/desi/target/analysis/truth/vipers-w4.fits.gz', 1)
    #zcat = fits.getdata('/project/projectdirs/desi/target/analysis/lrg/decals-dr2-lrg-sdss.fits.gz', 1)
    # Cuts (don't do isdss >  19.9 OR rAB < 19.5, consider full sample)
    flux={}
    for iband,band in zip([2,4],['r','z']):
        flux[band]= dr2['DECAM_FLUX'][:,iband] / dr2['DECAM_MW_TRANSMISSION'][:,iband]
    for iband,band in zip([0],['w1']):
        flux[band]= dr2['WISE_FLUX'][:,iband] / dr2['WISE_MW_TRANSMISSION'][:,iband]
    # LRG
    lrg= np.all((flux['z'] > 10**((22.5-20.46)/2.5),\
                 flux['z'] > flux['r']*10**(1.5/2.5),\
                 flux['w1']*np.power(flux['r'],1.8-1) > np.power(flux['z'],1.8)*10**(-1.0/2.5),\
                 flux['w1'] > 0),axis=0)
    psf= dr2['TYPE'] == 'PSF '
    # Quality
    # ANYMASK -> ALLMASK
    good= np.all((dr2['BRICK_PRIMARY'] == 'T',\
                  dr2['DECAM_ANYMASK'][:,2] == 0,\
                  dr2['DECAM_ANYMASK'][:,4] == 0),axis=0)  
    lrg= np.all((lrg,good),axis=0)
    psf= np.all((psf,good),axis=0)
    # Good z
    flag= vip['ZFLG'].astype(int)
    good_z= np.all((flag >= 2, flag <= 9),axis=0) 
    cuts=dict(lrg=lrg,\
              psf=psf,\
              good_z=good_z)
    # color data
    mag={}
    for key in flux.keys(): mag[key]= flux2mag(flux[key])
    rz= mag['r']-mag['z']
    rw1= mag['r']-mag['w1']
    Xall = np.array([rz,rw1]).T
    return Xall,cuts              


if __name__ == "__main__":
    # Stars
    Xall= star_data()
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
    Xall,cuts= elg_data_for_FDR()
    plot_FDR(Xall,cuts,src='ELG')
    b= cuts['any_elg']
    qa_plot_BIC(Xall[b,:], src='ELG',append='_FDR')
    qa_plot_MoG(Xall[b,:],ncomp=6, src='ELG',append='_FDR') #,extra=True)
    # Fit template spectra data
    Xall,cuts, morph= elg_data()
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
    plot_FDR(Xall,cuts,src='LRG')
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
  

