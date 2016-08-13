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
from pandas import DataFrame
from scipy.optimize import newton

import legacyanalysis.decals_sim_priors as priors

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
            
            x=np.linspace(xlim[0],xlim[1],num=100)
            y=np.linspace(ylim[0],ylim[1],num=100)
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
            ax.plot(x1,y1,'k-',lw=2)
            ax.plot(x2,y2,'k-',lw=2)
            ax.plot(x3,y3,'k-',lw=2)
            ax.plot(x4,y4,'k-',lw=2) 
        elif self.src == 'LRG':
            #r-w1 vs. r-z
            x=np.linspace(xlim[0],xlim[1],num=100)
            y=np.linspace(ylim[0],ylim[1],num=100)
            x1,y1= x,self.ts_box(x,'y1')
            x2,y2= np.array([1.5]*len(x)),y
            b= x >= 1.5
            x1,y1= x1[b],y1[b]
            b= y2 >= np.min(y1)
            x2,y2= x2[b],y2[b]
            ax.plot(x1,y1,'k-',lw=2)
            ax.plot(x2,y2,'k-',lw=2)
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
    cuts=dict(loz=loz,\
              oiifaint=oiifaint,\
              oiibright_loz=oiibright_loz,\
              oiibright_hiz=oiibright_hiz)
    return zcat,cuts              


def plot_FDR(zcat,cuts,src='ELG'):
    # Object to add target selection box
    ts= TSBox(src=src)
    if src == 'ELG':
        def getgrz(zcat, index=None,inbox=False):
            assert(index is not None)
            if inbox:
                y = zcat['CFHTLS_G'] - zcat['CFHTLS_R']
                x = zcat['CFHTLS_R'] - zcat['CFHTLS_Z']
                b=np.all((index,\
                          y < ts.ts_box(x,'y1'), y < ts.ts_box(x,'y2'), x > 0.3, x < 1.6),axis=0)
                gr= y[b]
                rz= x[b]
            else:
                gr = zcat['CFHTLS_G'][index] - zcat['CFHTLS_R'][index]
                rz = zcat['CFHTLS_R'][index] - zcat['CFHTLS_Z'][index]
            return gr, rz
        # Set up figure
        grrange = (-0.2, 2.0)
        rzrange = (-0.4, 2.5)
        sns.set(style='white', font_scale=1.6, palette='deep')
        col = sns.color_palette()
        fig, ax = plt.subplots()
        # Plot
        # Add box
        ts.add_ts_box(ax, xlim=rzrange,ylim=grrange)
        # Add points
        gr, rz = getgrz(zcat, index=cuts['loz'])
        ax.scatter(rz, gr, marker='^', color=col[2], label=r'$z<0.6$')

        gr, rz = getgrz(zcat, index=cuts['oiifaint'])
        ax.scatter(rz, gr, marker='s', color='tan',
                        label=r'$z>0.6, [OII]<8\times10^{-17}$')

        gr, rz = getgrz(zcat, index=cuts['oiibright_loz'])
        ax.scatter(rz, gr, marker='o', color='powderblue',
                        label=r'$z>0.6, [OII]>8\times10^{-17}$')

        gr, rz = getgrz(zcat, index=cuts['oiibright_hiz'])
        ax.scatter(rz, gr, marker='o', color='powderblue', edgecolor='black',
                        label=r'$z>1.0, [OII]>8\times10^{-17}$')
        ax.set_xlim(rzrange)
        ax.set_ylim(grrange)
        ax.legend(loc='upper left', prop={'size': 14}, labelspacing=0.2,
                  markerscale=1.5)
        # Label
        xlab=ax.set_xlabel('r - z')
        ylab=ax.set_ylabel('g - r')
    elif src == 'LRG':
        def getcolor(zcat, index=None):
            assert(index is not None)
            # get mags
            mag={}
            for iband,band in zip([2,4],['r','z']):
                mag[band]= flux2mag( zcat['DECAM_FLUX'][:,iband][index] / zcat['DECAM_MW_TRANSMISSION'][:,iband][index] )
            for iband,band in zip([0],['w1']):
                mag[band]= flux2mag( zcat['WISE_FLUX'][:,iband][index] / zcat['WISE_MW_TRANSMISSION'][:,iband][index] )
            # return color
            return mag['r']-mag['z'],mag['r']-mag['w1']
        # Set up figure
        xrange = (0, 2.5)
        yrange = (-2, 6)
        sns.set(style='white', font_scale=1.6, palette='deep')
        col = sns.color_palette()
        fig, ax = plt.subplots()
        # Plot
        # Add box
        ts.add_ts_box(ax, xlim=xrange,ylim=yrange)
        # Add points
        rz, rw1 = getcolor(zcat, index=cuts['lrg'])
        ax.scatter(rz, rw1, marker='^', color='b', label='LRG')

        #rz, rw1 = getcolor(zcat, index=cuts['psf'])
        #ax.scatter(rz, rw1, marker='s', color='g',label='PSF')
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.legend(loc='upper left', prop={'size': 14}, labelspacing=0.2,
                  markerscale=1.5)
        # Label
        xlab=ax.set_xlabel('r - z')
        ylab=ax.set_ylabel('r - w1')
    else: 
        raise ValueError('src=%s not supported' % src)
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
    keep = elgs['radius_halflight'] > 0
    print('Grabbed {} ELGs, of which {} have HST morphologies.'.format(len(elgs), len(elgs[keep])))
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
        xrange = (-0.2, 2.0)
        yrange = (-0.4, 2.5)
        xlab='r - z'
        ylab='g - r'
    elif src == 'ELG':
        ncomp=7
        xrange = (-0.2, 2.0)
        yrange = (-0.4, 2.5)
        xlab='r - z'
        ylab='g - r'
    elif src == 'LRG':
        ncomp=4
        xrange = (0, 2.5)
        yrange = (-2, 6)
        xlab='r - z'
        ylab='r - w1'
    else: raise ValueError('src=%s not supported' % src)
    mog = GMM(n_components=ncomp, covariance_type="full").fit(Xall)
    priors.add_MoG_curves(ax1, mog.means_, mog.covars_, mog.weights_)
    ax1.set_xlim(xrange)
    ax1.set_ylim(yrange)
    xlab=ax1.set_xlabel(xlab)
    ylab=ax1.set_ylabel(ylab)
    ti=ax1.set_title('N comp + 1 extra')
    
    ax2.plot(samp[:,0], samp[:,1], 'o', c=col[0], markersize=3)
    ax2.set_xlim(xrange)
    ax2.set_ylim(yrange)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    xlab=ax2.set_xlabel(xlab)
    ylab2=ax2.set_ylabel(ylab)
    ti=ax2.set_title('%.2g Draws, %d comps' % (float(len(samp[:,0])),ncomp-1))
    # ax2.legend(loc='lower right', prop={'size': 14}, labelspacing=0.25, markerscale=2)
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    print('Writing {}'.format(name))
    plt.savefig(name, bbox_extra_artists=[xlab,ylab,ylab2,ti], bbox_inches='tight',dpi=150)
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
    elif src == 'LRG':
        mog = priors._GaussianMixtureModel.load('legacypipe/data/lrg_colors_mog.fits')
        samp = mog.sample(nsamp)
        plot_mog_and_sample(samp,Xall, src=src,name=name)
    else:
        raise ValueError('src=%s not supported' % src)
        

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
    elif src == 'LRG':
        plt.legend(labels=['%s r-w1, r-z colors' % src])
    else: raise ValueError('src=%s not supportd' % src)
    plt.tight_layout()
    name='qa-mog-bic-%s.png' % src
    print('Writing {}'.format(name))
    plt.savefig(name)
    plt.close()

def create_joinplot(df,xkey,ykey,xlab,ylab,xlim,ylim,color,src='ELG'):
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
    mag={}
    for key in flux.keys(): mag[key]= flux2mag(flux[key])
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
    rz= (mag['r']-mag['z'])[lrg]
    rw1= (mag['r']-mag['w1'])[lrg]
    Xall = np.array([rz,rw1]).T
    return dr2,cuts,Xall              


if __name__ == "__main__":
    # LRGs
    zcat,cuts,Xall= lrg_data_for_FDR()
    plot_FDR(zcat,cuts,src='LRG')
    # Choose 2 compoenents
    mog = GMM(n_components=2, covariance_type="full").fit(Xall)
    lrg_mogfile= 'legacypipe/data/lrg_colors_mog.fits'
    if os.path.exists(lrg_mogfile):
        print('LRG MoG exists, not overwritting: %s' % lrg_mogfile)
    else:
        print('Writing {}'.format(lrg_mogfile))
        priors._GaussianMixtureModel.save(mog, lrg_mogfile)
    #
    qa_plot_BIC(Xall, src='LRG')
    qa_plot_MoG(Xall, src='LRG') #,extra=True)
    sys.exit('exit after LRG')
    #qa_plot_Priors(d=morph,src='LRG')
    # Stars
    Xall= star_data()
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
    # ELGs
    # Target selection plot 
    zcat,cuts= elg_data_for_FDR()
    plot_FDR(zcat,cuts,src='ELG')
    # Mog for Synthetic spectra -> colors
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
  

