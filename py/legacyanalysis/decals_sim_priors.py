if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.table import vstack, Table
import os
import sys
from scipy.optimize import newton

class _GaussianMixtureModel(object):
    """Read and sample from a pre-defined Gaussian mixture model.
    This assumes 'sklearn.mixture.GMM' has already been used to determine MoGs 
    """
    def __init__(self, weights_, means_, covars_, covtype):
        self.weights_ = weights_
        self.means_ = means_
        self.covars_ = covars_
        self.covtype = covtype
        self.n_components, self.n_dimensions = self.means_.shape
    
    @staticmethod
    def save(model, filename,index=None):
        '''index: optional nddex array for subset of compenents to save'''
        hdus = fits.HDUList()
        hdr = fits.Header()
        hdr['covtype'] = model.covariance_type
        if index is None:
            index=np.arange(len(model.weights_))
        hdus.append(fits.ImageHDU(model.weights_[index], name='weights_', header=hdr))
        hdus.append(fits.ImageHDU(model.means_[index,...], name='means_'))
        hdus.append(fits.ImageHDU(model.covars_[index,...], name='covars_'))
        hdus.writeto(filename, clobber=True)
        
    @staticmethod
    def load(filename):
        hdus = fits.open(filename, memmap=False)
        hdr = hdus[0].header
        covtype = hdr['covtype']
        model = _GaussianMixtureModel(
            hdus['weights_'].data, hdus['means_'].data, hdus['covars_'].data, covtype)
        hdus.close()
        return model
    
    def sample(self, n_samples=1, random_state=None):
        
        if self.covtype != 'full':
            return NotImplementedError(
                'covariance type "{0}" not implemented yet.'.format(self.covtype))
        
        # Code adapted from sklearn's GMM.sample()
        if random_state is None:
            random_state = np.random.RandomState()

        weight_cdf = np.cumsum(self.weights_)
        X = np.empty((n_samples, self.n_dimensions))
        rand = random_state.rand(n_samples)
        # decide which component to use for each sample
        comps = weight_cdf.searchsorted(rand)
        # for each component, generate all needed samples
        for comp in range(self.n_components):
            # occurrences of current component in X
            comp_in_X = (comp == comps)
            # number of those occurrences
            num_comp_in_X = comp_in_X.sum()
            if num_comp_in_X > 0:
                X[comp_in_X] = random_state.multivariate_normal(
                    self.means_[comp], self.covars_[comp], num_comp_in_X)
        return X
    
    def sample_full_pdf(self, n_samples=1, random_state=None):
        ''''sample() uses the component datum x is closest too, sample_full_pdf() uses sum of components at datum x
        this is more time consuming than sample() and difference is negligible'''
        if self.covtype != 'full':
            return NotImplementedError(
                'covariance type "{0}" not implemented yet.'.format(self.covtype))

        from scipy.stats import multivariate_normal
        def get_mv(means_,covars_):
            mv=[]
            for mu, C in zip(means_, covars_):
                mv+= [ multivariate_normal(mean=mu, cov=C) ]
            return mv
                
        def prob_map(means_,covars_,weights_,\
                     xrng=(0.,1.),yrng=(0.,1.),npts=2**10):
            '''returns 
            -pmap: 2d probability map, with requirement that integral(pdf d2x) within 1% of 1
            -xvec,yvec: vectors where x[ix] and y[ix] data points have probability pmap[ix,iy]'''
            assert(xrng[1] > xrng[0] and yrng[1] > yrng[0])
            xstep= (xrng[1]-xrng[0])/float(npts-1)
            ystep= (yrng[1]-yrng[0])/float(npts-1)
            x,y  = np.mgrid[xrng[0]:xrng[1]+xstep:xstep, yrng[0]:yrng[1]+ystep:ystep]
            pos = np.empty(x.shape + (2,)) #npts x npts x 2
            pos[:, :, 0] = x; pos[:, :, 1] = y
            maps= np.zeros(x.shape+ (len(weights_),)) # x n_components
            # Multi-Variate function
            mv= get_mv(means_,covars_)
            # probability map for each component
            for dist,W,cnt in zip(mv,weights_, range(len(weights_))):
                maps[:,:,cnt]= dist.pdf(pos) * W
                print "map %d, dist.pdf max=%.2f, wt=%.3f" % (cnt,dist.pdf(pos).max(),W)
            # summed prob map
            pmap= np.sum(maps, axis=2) #some over components*weights
            xvec= x[:,0]
            yvec= y[0,:]
            # intregal of pdf over 2d map = 1
        #     assert( abs(1.- pmap.sum()*xstep*ystep) <= 0.01 )
            assert( np.diff(xvec).min() > 0. and np.diff(yvec).min() > 0.)
            return pmap, xvec,yvec
            
        # 2D probability map
        grrange = (-0.2, 2.0)
        rzrange = (-0.4, 2.5)
        pmap,xvec,yvec= prob_map(self.means_,self.covars_,self.weights_,\
                                 xrng=rzrange,yrng=grrange)
        # Sample self.n_dimensions using map
        if random_state is None:
            r = np.random.RandomState()
        # Make max pdf = 1 so always pick that cell
        pmap/= pmap.max()
        # Store x,y values to use
        X = np.empty((n_samples, self.n_dimensions))+np.nan
        # Get samples
        cnt=0
        # pick a random cell
        ix,iy=(r.rand(2)*len(xvec)).astype(int)
        # Random [0,1)
        likely= r.rand(1)
        while cnt < n_samples:
            if likely <= pmap[ix,iy]: # Sample it!
                X[cnt,:]= xvec[ix],yvec[iy]
                cnt+=1
            # Pick new cell in either case
            ix,iy=(r.rand(2)*len(xvec)).astype(int) 
            likely= r.rand(1)
        assert( np.where(~np.isfinite(X))[0].size == 0)
        return X


def add_MoG_curves(ax, means_, covars_, weights_):
    '''plot 2-sigma ellipses for each multivariate component'''
    ax.scatter(means_[:, 0], means_[:, 1], c='w')
    scale=2.
    for cnt, mu, C, w in zip(range(means_.shape[0]),means_, covars_, weights_):
    #     draw_ellipse(mu, C, scales=[1.5], ax=ax, fc='none', ec='k')
        # Draw MoG outlines
        sigma_x2 = C[0, 0]
        sigma_y2 = C[1, 1]
        sigma_xy = C[0, 1]

        alpha = 0.5 * np.arctan2(2 * sigma_xy,
                             (sigma_x2 - sigma_y2))
        tmp1 = 0.5 * (sigma_x2 + sigma_y2)
        tmp2 = np.sqrt(0.25 * (sigma_x2 - sigma_y2) ** 2 + sigma_xy ** 2)

        sigma1 = np.sqrt(tmp1 + tmp2)
        sigma2 = np.sqrt(tmp1 - tmp2)
        print('comp=%d, sigma1=%f,sigma2=%f' % (cnt+1,sigma1,sigma2))

        ax.text(mu[0],mu[1],str(cnt+1),color='blue')
        ax.add_patch(Ellipse((mu[0], mu[1]),
                     2 * scale * sigma1, 2 * scale * sigma2,
                     alpha * 180. / np.pi,\
                     fc='none', ec='k'))

def get_rgb_cols():
    return [(255,0,255),(102,255,255),(0,153,153),\
            (255,0,0),(0,255,0),(0,0,255),\
            (0,0,0)]

def flux2mag(nanoflux):
    return 22.5-2.5*np.log10(nanoflux)

# Globals
xyrange=dict(x_star=[-0.5,2.2],\
             y_star=[-0.3,2.],\
             x_elg=[-0.5,2.2],\
             y_elg=[-0.3,2.],\
             x_lrg= [0, 2.5],\
             y_lrg= [-2, 6])
def rm_last_ticklabel(ax):
    '''for multiplot'''
    labels=ax.get_xticks().tolist()
    labels=np.array(labels).astype(float) #prevent from making float
    labels=list(labels)
    labels[-1]=''
    ax.set_xticklabels(labels)



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



def color_color_plot(Xall,src='STAR',outdir='.',append=''): 
    '''Make color-color plot.'''
    if src == 'STAR':
        xrange = xyrange['x_%s' % src.lower()]
        yrange = xyrange['y_%s' % src.lower()]
        xlab='r - z'
        ylab='g - r'
    elif src == 'ELG':
        xrange = xyrange['x_%s' % src.lower()]
        yrange = xyrange['y_%s' % src.lower()]
        xlab='r - z'
        ylab='g - r'
    elif src == 'LRG':
        xrange = xyrange['x_%s' % src.lower()]
        yrange = xyrange['y_%s' % src.lower()]
        xlab='r - z'
        ylab='r - w1'
    else: raise ValueError('src=%s not supported' % src)
    fig, ax = plt.subplots(1, 2, sharey=True,figsize=(8, 4))
    ax[0].plot(Xall[:,0],Xall[:,1], 'o', c='b', markersize=3)
    # Add ts box
    for i,title in zip(range(1),['Data']):
        xlab1=ax[i].set_xlabel(xlab)
        ax[i].set_xlim(xrange)
        ax[i].set_ylim(yrange)
        ti=ax[i].set_title(title)
    ylab1=ax[0].set_ylabel(ylab)
    fig.subplots_adjust(wspace=0) #, hspace=0.1)
    for i in range(2):
        rm_last_ticklabel(ax[i])
    name= os.path.join(outdir,'priors-%s%s.png' % (src,append))
    print('Writing {}'.format(name))
    plt.savefig(name, bbox_extra_artists=[xlab1,ylab1,ti], bbox_inches='tight',dpi=150)
    plt.close()


class ReadWrite(object):
    def read_fits(self,fn):
        return Table(fits.getdata(fn, 1))



class QSO(ReadWrite):
    def __init__(self):
        self.fn_cat= ''
        self.cat= self.read_fits(self.fn_cat)
    def plot(self):
        plt.plot(self.cat['ra'],self.cat['dec'])
        plt.savefig('test.png')
        plt.close()

class ELG(ReadWrite):
    def __init__(self):
        pass
        #self.Xall,self.cuts= elg_data_for_FDR()
        #self.fn_cat= ''
        #self.cat= self.read_fits(self.fn_cat)
    def get_FDR(self):
        '''version 3.0 of data discussed in
        https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=912'''
        zcat = self.read_fits('/project/projectdirs/desi/target/analysis/deep2/v3.0/deep2-field1-oii.fits.gz')
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
            
    def plot_FDR(self):
        Xall,cuts= self.get_FDR()
        # Object to add target selection box
        ts= TSBox(src='ELG')
        fig, ax = plt.subplots()
        # Add box
        xrange,yrange= [-0.5,2.2],[-0.3,2.]
        ts.add_ts_box(ax, xlim=xrange,ylim=yrange)
        # Add points
        b= cuts['loz']
        ax.scatter(Xall[:,0][b],Xall[:,1][b], marker='^', color='magenta', label=r'$z<0.6$')

        b=cuts['oiifaint']
        ax.scatter(Xall[:,0][b],Xall[:,1][b], marker='s', color='tan',
                        label=r'$z>0.6, [OII]<8\times10^{-17}$')

        b= cuts['oiibright_loz']
        ax.scatter(Xall[:,0][b],Xall[:,1][b], marker='o', color='powderblue',
                        label=r'$z>0.6, [OII]>8\times10^{-17}$')

        b=cuts['oiibright_hiz']
        ax.scatter(Xall[:,0][b],Xall[:,1][b], marker='o', color='powderblue', edgecolor='black',
                        label=r'$z>1.0, [OII]>8\times10^{-17}$')
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        xlab= ax.set_xlabel('r-z')
        ylab= ax.set_ylabel('g-r')
        ax.legend(loc='upper left', prop={'size': 14}, labelspacing=0.2,
                  markerscale=1.5)
        name='update_FDR_ELG.png'
        print('Wrote {}'.format(name))
        kwargs= dict(bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
        plt.savefig(name, **kwargs)
        plt.close()

    def plot(self):
        self.plot_FDR()
        sys.exit('stopping early')
        #plot_FDR(self.Xall,self.cuts,src='ELG')
        #b= self.cuts['any_elg']
        #color_color_plot(self.Xall[b,:],src='ELG',append='_FDR') #,extra=True)
        #Xall,cuts, morph= elg_data()
        #color_color_plot(Xall, src='ELG',append='_synth') #,extra=True)
        #b= cuts['has_morph']
        #color_color_plot(Xall[b,:],src='ELG',append='_synth+morph') #,extra=True)

class LRG(ReadWrite):
    def __init__(self):
        self.Xall,self.cuts= lrg_data_for_FDR()
        #self.fn_cat= ''
        #self.cat= self.read_fits(self.fn_cat)
    def get_FDR(self):
        # DR2 matched to sample below
        dr2=self.read_fits('/project/projectdirs/desi/target/analysis/truth/decals-dr2-cosmos-zphot.fits.gz')
        Z_FLUX = dr2['DECAM_FLUX'][:,4] / dr2['DECAM_MW_TRANSMISSION'][:,4]
        W1_FLUX = dr2['WISE_FLUX'][:,0] / dr2['WISE_MW_TRANSMISSION'][:,0]
        index={}
        index['decals']= np.all((Z_FLUX < 10**((22.5-20.46)/2.5),\
                                 W1_FLUX > 0.,\
                                 dr2['BRICK_PRIMARY'] == 'T'),axis=0)
        # http://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos_zphot_mag25_colDescriptions.html
        # http://irsa.ipac.caltech.edu/data/COSMOS/tables/redshift/cosmos_zphot_mag25.README
        spec=self.read_fits('/project/projectdirs/desi/target/analysis/truth/cosmos-zphot.fits.gz')
        for key in ['star','red_galaxy_lowz','red_galaxy_hiz','blue_galaxy']:
            if key == 'star': 
                index[key]= np.all((index['decals'],\
                                    spec['TYPE'] == 1),axis=0)
            elif key == 'blue_galaxy': 
                index[key]= np.all((index['decals'],\
                                    spec['TYPE'] == 0,\
                                    spec['MOD_GAL'] > 8),axis=0)
            elif key == 'red_galaxy_lowz': 
                index[key]= np.all((index['decals'],\
                                    spec['TYPE'] == 0,\
                                    spec['MOD_GAL'] <= 8,\
                                    spec['ZP_GAL'] <= 0.6),axis=0)
            elif key == 'red_galaxy_hiz': 
                index[key]= np.all((index['decals'],\
                                    spec['TYPE'] == 0,\
                                    spec['MOD_GAL'] <= 8,\
                                    spec['ZP_GAL'] > 0.6),axis=0)
        # return Mags
        rz,rW1={},{}
        R_FLUX = dr2['DECAM_FLUX'][:,2] / dr2['DECAM_MW_TRANSMISSION'][:,2]
        for key in ['star','red_galaxy_lowz','red_galaxy_hiz','blue_galaxy']:
            cut= index[key]
            rz[key]= flux2mag(R_FLUX[cut]) - flux2mag(Z_FLUX[cut])
            rW1[key]= flux2mag(R_FLUX[cut]) - flux2mag(W1_FLUX[cut])
        return rz,rW1
                                 
    def plot_FDR(self):
        rz,rW1= self.get_FDR()
        fig,ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(16,4))
        plt.subplots_adjust(wspace=0,hspace=0)
        rgb_cols=get_rgb_cols()
        for cnt,key,rgb in zip(range(4),rz.keys(),rgb_cols):
            rgb= (rgb[0]/255.,rgb[1]/255.,rgb[2]/255.)
            ax[cnt].scatter(rz[key],rW1[key],c=[rgb],edgecolors='none',marker='o',s=10.,rasterized=True)#,label=key)
            ti=ax[cnt].set_title(key)
            ax[cnt].set_xlim([0,2.5])
            ax[cnt].set_ylim([-2,6])
            xlab=ax[cnt].set_xlabel('r-z')
        ylab=ax[0].set_ylabel('r-W1')
        #handles,labels = ax.get_legend_handles_labels()
        #index=[0,1,2,3]
        #handles,labels= np.array(handles)[index],np.array(labels)[index]
        #leg=ax.legend(handles,labels,loc=(0,1.05),ncol=2,scatterpoints=1,markerscale=2)
        name='update_FDR_LRG.png'
        plt.savefig('update_FDR_LRG.png',\
                    bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
        plt.close()
        print('Wrote {}'.format(name))

    def plot(self):
        self.plot_FDR()
        #plot_FDR(self.Xall,self.cuts,src='LRG')
        #color_color_plot(self.Xall,src='LRG',append='cc') #,extra=True)
        #b= self.cuts['lrg']
        #color_color_plot(self.Xall[b,:],src='LRG') #,extra=True)

class STAR(ReadWrite):
    def __init__(self):
        self.Xall= star_data()
        #self.fn_cat= ''
        #self.cat= self.read_fits(self.fn_cat)
    def plot(self):
        color_color_plot(self.Xall, src='STAR') 
        #plt.plot(self.cat['ra'],self.cat['dec'])
        #plt.savefig('test.png')
        #plt.close()

class GalaxyPrior(object):
    def __init__(self):
        #self.qso= QSO()
        self.elg= ELG()
        self.lrg= LRG()
        self.star= STAR()
    def plot_all(self):
        #self.qso.plot()
        self.lrg.plot()
        self.elg.plot()
        #self.star.plot()

if __name__ == '__main__':
    gals=GalaxyPrior()
    gals.plot_all()
    print "gals.__dict__= ",gals.__dict__
