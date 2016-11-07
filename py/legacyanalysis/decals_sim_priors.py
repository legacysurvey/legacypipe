if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
#from astropy.table import vstack, Table
from astrometry.util.fits import fits_table, merge_tables
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


class ReadWrite(object):
    def read_fits(self,fn):
        #return Table(fits.getdata(fn, 1))
        return fits_table(fn)



class QSO(ReadWrite):
    def __init__(self):
        self.fn_cat= ''
        self.cat= self.read_fits(self.fn_cat)
    def plot(self,savefig=False):
        plt.plot(self.cat['ra'],self.cat['dec'])
        if savefig:
            name='test.png'
            plt.savefig(name)
            plt.close()
            print('Wrote {}'.format(name))

class ELG(ReadWrite):
    def __init__(self,DR=2,savefig=False):
        self.DR=DR
        self.savefig=savefig
        if self.DR == 2:
            self.truth_dir= '/project/projectdirs/desi/target/analysis/truth'
        elif self.DR == 3:
            self.truth_dir= '/project/projectdirs/desi/users/burleigh/desi/target/analysis/truth'
        else: raise ValueError()

    def get_FDR(self):
        '''version 3.0 of data discussed in
        https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=912'''
        rfaint = 23.4
        if self.DR == 2:
            zcat = self.read_fits(os.path.join(self.truth_dir,'../deep2/v3.0/','deep2-field1-oii.fits.gz'))
            G_MAG= zcat.get('cfhtls_g')
            R_MAG= zcat.get('cfhtls_r')
            Z_MAG= zcat.get('cfhtls_z')
            rmag_cut= R_MAG<rfaint 
        elif self.DR == 3:
            zcat = self.read_fits(os.path.join(self.truth_dir,'deep2f234-dr3matched.fits'))
            decals = self.read_fits(os.path.join(self.truth_dir,'dr3-deep2f234matched.fits'))
            # Add mag data 
            G_MAG= decals.get('decam_mag')[:,1]
            R_MAG= decals.get('decam_mag')[:,2]
            Z_MAG= decals.get('decam_mag')[:,4]
            rmag_cut= decals.get('decam_flux')[:,2] > 10**((22.5-rfaint)/2.5)
        # Cuts
        oiicut1 = 8E-17 # [erg/s/cm2]
        zmin = 0.6
        loz = np.all((zcat.get('zhelio')<zmin,\
                      rmag_cut),axis=0)
        oiifaint = np.all((zcat.get('zhelio')>zmin,\
                           rmag_cut,\
                           zcat.get('oii_3727_err')!=-2.0,\
                           zcat.get('oii_3727')<oiicut1),axis=0)
        oiibright_loz = np.all((zcat.get('zhelio')>zmin,\
                                zcat.get('zhelio')<1.0,\
                                rmag_cut,\
                                zcat.get('oii_3727_err')!=-2.0,\
                                zcat.get('oii_3727')>oiicut1),axis=0)
        oiibright_hiz = np.all((zcat.get('zhelio')>1.0,\
                                rmag_cut,\
                                zcat.get('oii_3727_err')!=-2.0,\
                                zcat.get('oii_3727')>oiicut1),axis=0)
        any_elg= np.all((rmag_cut,\
                         zcat.get('oii_3727_err')!=-2.0,\
                         zcat.get('oii_3727')>oiicut1),axis=0)
        # color data
        rz= (R_MAG - Z_MAG)
        gr= (G_MAG - R_MAG)
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
        xrange,yrange= xyrange['x_elg'],xyrange['y_elg']
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
        leg=ax.legend(loc=(0,1.05), ncol=2,prop={'size': 14}, labelspacing=0.2,
                  markerscale=1.5)
        name='update_FDR_ELG.png'
        kwargs= dict(bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight',dpi=150)
        if self.savefig:
            plt.savefig(name, **kwargs)
            plt.close()
            print('Wrote {}'.format(name))
 
    def plot_FDR_multipanel(self):
        Xall,cuts= self.get_FDR()
        ts= TSBox(src='ELG')
        xrange,yrange= xyrange['x_elg'],xyrange['y_elg']
        # Plot
        fig,ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(18,4))
        plt.subplots_adjust(wspace=0.1,hspace=0)
        for cnt,key,col,marker,ti in zip(range(4),\
                               ['loz','oiifaint','oiibright_loz','oiibright_hiz'],\
                               ['magenta','tan','powderblue','powderblue'],\
                               ['^','s','o','o'],\
                               [r'$z<0.6$',r'$z>0.6, [OII]<8\times10^{-17}$',r'$z>0.6, [OII]>8\times10^{-17}$',r'$z>1.0, [OII]>8\times10^{-17}$']):
            # Add box
            ts.add_ts_box(ax[cnt], xlim=xrange,ylim=yrange)
            # Add points
            b= cuts[key]
            ax[cnt].scatter(Xall[:,0][b],Xall[:,1][b], marker=marker,color=col)
            ti_loc=ax[cnt].set_title(ti)
            ax[cnt].set_xlim(xrange)
            ax[cnt].set_ylim(yrange)
            xlab= ax[cnt].set_xlabel('r-z')
            ylab= ax[cnt].set_ylabel('g-r')
            name='dr3_FDR_ELG_multipanel.png'
        kwargs= dict(bbox_extra_artists=[ti_loc,xlab,ylab], bbox_inches='tight',dpi=150)
        if self.savefig:
            plt.savefig(name, **kwargs)
            plt.close()
            print('Wrote {}'.format(name))


    def plot(self):
        self.plot_FDR()
        self.plot_FDR_multipanel()
        #plot_FDR(self.Xall,self.cuts,src='ELG')
        #b= self.cuts['any_elg']
        #color_color_plot(self.Xall[b,:],src='ELG',append='_FDR') #,extra=True)
        #Xall,cuts, morph= elg_data()
        #color_color_plot(Xall, src='ELG',append='_synth') #,extra=True)
        #b= cuts['has_morph']
        #color_color_plot(Xall[b,:],src='ELG',append='_synth+morph') #,extra=True)

class LRG(ReadWrite):
    def __init__(self,DR=2,savefig=False):
        self.DR=DR
        self.savefig=savefig
        if self.DR == 2:
            self.truth_dir= '/project/projectdirs/desi/target/analysis/truth'
        elif self.DR == 3:
            self.truth_dir= '/project/projectdirs/desi/users/burleigh/desi/target/analysis/truth'
        else: raise ValueError()
    
    def get_FDR(self):
        if self.DR == 2:
            decals=self.read_fits( os.path.join(self.truth_dir,'decals-dr2-cosmos-zphot.fits.gz') )
            spec=self.read_fits( os.path.join(self.truth_dir,'cosmos-zphot.fits.gz') )
            # brick_primary set to T and F not boolean
            new= np.zeros(len(decals.get('brick_primary'))).astype(bool)
            new[decals.get('brick_primary') == 'T']= True
            decals.set('brick_primary',new)
        elif self.DR == 3:
            decals=self.read_fits( os.path.join(self.truth_dir,'dr3-cosmoszphotmatched.fits') )
            spec=self.read_fits( os.path.join(self.truth_dir,'cosmos-zphot-dr3matched.fits') )
        # DECaLS
        Z_FLUX = decals.get('decam_flux'.lower())[:,4] / decals.get('decam_mw_transmission'.lower())[:,4]
        W1_FLUX = decals.get('wise_flux'.lower())[:,0] / decals.get('wise_mw_transmission'.lower())[:,0]
        index={}
        # BUG!!
        #index['decals']= np.all((Z_FLUX < 10**((22.5-20.46)/2.5),\
        index['decals']= np.all((Z_FLUX > 10**((22.5-20.46)/2.5),\
                                 W1_FLUX > 0.,\
                                 decals.get('brick_primary') == True),axis=0)
        # Cosmos
        # http://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos_zphot_mag25_colDescriptions.html
        # http://irsa.ipac.caltech.edu/data/COSMOS/tables/redshift/cosmos_zphot_mag25.README
        for key in ['star','red_galaxy_lowz','red_galaxy_hiz','blue_galaxy']:
            if key == 'star': 
                index[key]= np.all((index['decals'],\
                                    spec.get('type') == 1),axis=0)
            elif key == 'blue_galaxy': 
                index[key]= np.all((index['decals'],\
                                    spec.get('type') == 0,\
                                    spec.get('mod_gal') > 8),axis=0)
            elif key == 'red_galaxy_lowz': 
                index[key]= np.all((index['decals'],\
                                    spec.get('type') == 0,\
                                    spec.get('mod_gal') <= 8,\
                                    spec.get('zp_gal') <= 0.6),axis=0)
            elif key == 'red_galaxy_hiz': 
                index[key]= np.all((index['decals'],\
                                    spec.get('type') == 0,\
                                    spec.get('mod_gal') <= 8,\
                                    spec.get('zp_gal') > 0.6),axis=0)
        # return Mags
        rz,rW1={},{}
        R_FLUX = decals.get('decam_flux')[:,2] / decals.get('decam_mw_transmission')[:,2]
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
            # Add box
            ts= TSBox(src='LRG')
            xrange,yrange= xyrange['x_lrg'],xyrange['y_lrg']
            ts.add_ts_box(ax[cnt], xlim=xrange,ylim=yrange)
        ylab=ax[0].set_ylabel('r-W1')
        #handles,labels = ax.get_legend_handles_labels()
        #index=[0,1,2,3]
        #handles,labels= np.array(handles)[index],np.array(labels)[index]
        #leg=ax.legend(handles,labels,loc=(0,1.05),ncol=2,scatterpoints=1,markerscale=2)
        name='dr%d_FDR_LRG.png' % self.DR
        if self.savefig:
            plt.savefig(name,\
                        bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote {}'.format(name))

    def get_vipers(self):
        '''LRGs from VIPERS in CFHTLS W4 field (12 deg2)'''
        # DR2 matched
        if self.DR == 2:
            decals = self.read_fits(os.path.join(self.truth_dir,'decals-dr2-vipers-w4.fits.gz'))
            vip = self.read_fits(os.path.join(self.truth_dir,'vipers-w4.fits.gz'))
            # brick_primary set to T and F not boolean
            new= np.zeros(len(decals.get('brick_primary'))).astype(bool)
            new[decals.get('brick_primary') == 'T']= True
            decals.set('brick_primary',new)
        elif self.DR == 3:
            decals = self.read_fits(os.path.join(self.truth_dir,'dr3-vipersw1w4matched.fits'))
            vip = self.read_fits(os.path.join(self.truth_dir,'vipersw1w4-dr3matched.fits'))
        Z_FLUX = decals.get('decam_flux')[:,4] / decals.get('decam_mw_transmission')[:,4]
        W1_FLUX = decals.get('wise_flux')[:,0] / decals.get('wise_mw_transmission')[:,0]
        index={}
        index['decals']= np.all((Z_FLUX > 10**((22.5-20.46)/2.5),\
                                 W1_FLUX > 0.,\
                                 decals.get('brick_primary') == True),axis=0)
                                 #decals.get('DECAM_ANYMASK')[:,2] == 0,\
                                 #decals.get('DECAM_ANYMASK')[:,4] == 0),axis=0)  
        # VIPERS
        # https://arxiv.org/abs/1310.1008
        # https://arxiv.org/abs/1303.2623
        flag= vip.get('zflg').astype(int)
        index['good_z']= np.all((flag >= 2,\
                                 flag <= 9,\
                                 vip.get('zspec') < 9.9),axis=0) 
        # return Mags
        rz,rW1={},{}
        R_FLUX = decals.get('decam_flux')[:,2] / decals.get('decam_mw_transmission')[:,2]
        cut= np.all((index['decals'],\
                     index['good_z']),axis=0)
        rz= flux2mag(R_FLUX[cut]) - flux2mag(Z_FLUX[cut])
        rW1= flux2mag(R_FLUX[cut]) - flux2mag(W1_FLUX[cut])
        return rz,rW1
    
    def plot_vipers(self):
        rz,rW1= self.get_vipers()
        fig,ax = plt.subplots(figsize=(5,4))
        plt.subplots_adjust(wspace=0,hspace=0)
        rgb=get_rgb_cols()[0]
        rgb= (rgb[0]/255.,rgb[1]/255.,rgb[2]/255.)
        ax.scatter(rz,rW1,c=[rgb],edgecolors='none',marker='o',s=10.,rasterized=True)#,label=key)
        ax.set_xlim([0,2.5])
        ax.set_ylim([-2,6])
        xlab=ax.set_xlabel('r-z')
        ylab=ax.set_ylabel('r-W1')
        # Add box
        ts= TSBox(src='LRG')
        xrange,yrange= xyrange['x_lrg'],xyrange['y_lrg']
        ts.add_ts_box(ax, xlim=xrange,ylim=yrange)
        #handles,labels = ax.get_legend_handles_labels()
        #index=[0,1,2,3]
        #handles,labels= np.array(handles)[index],np.array(labels)[index]
        #leg=ax.legend(handles,labels,loc=(0,1.05),ncol=2,scatterpoints=1,markerscale=2)
        name='LRG_vipers.png'
        if self.savefig:
            plt.savefig(name,\
                        bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote {}'.format(name))


    def plot(self):
        self.plot_FDR()
        self.plot_vipers()
        #plot_FDR(self.Xall,self.cuts,src='LRG')
        #color_color_plot(self.Xall,src='LRG',append='cc') #,extra=True)
        #b= self.cuts['lrg')
        #color_color_plot(self.Xall[b,:],src='LRG') #,extra=True)



class STAR(ReadWrite):
    def __init__(self):
        pass
        #self.Xall= star_data()
        #self.fn_cat= ''
        #self.cat= self.read_fits(self.fn_cat)
    
    def get_dr2stars(self):
        '''Model the g-r, r-z color-color sequence for stars'''
        # Build a sample of stars with good photometry from a single sweep.
        rbright = 18
        rfaint = 19.5
        swp_dir='/global/project/projectdirs/cosmo/data/legacysurvey/dr2/sweep/2.0'
        sweep = self.read_fits(os.path.join(swp_dir,'sweep-340p000-350p005.fits'))
        keep = np.where((sweep.get('type') == 'PSF ')*
                        (np.sum((sweep.get('decam_flux')[:, [1,2,4]] > 0)*1, axis=1)==3)*
                        (np.sum((sweep.get('DECAM_ANYMASK')[:, [1,2,4]] > 0)*1, axis=1)==0)*
                        (np.sum((sweep.get('DECAM_FRACFLUX')[:, [1,2,4]] < 0.05)*1, axis=1)==3)*
                        (sweep.get('decam_flux')[:,2]<(10**(0.4*(22.5-rbright))))*
                        (sweep.get('decam_flux')[:,2]>(10**(0.4*(22.5-rfaint)))))[0]
        stars = sweep[keep]
        print('dr2stars sample: {}'.format(len(stars)))
        gg = 22.5-2.5*np.log10(stars.get('decam_flux')[:, 1])
        rr = 22.5-2.5*np.log10(stars.get('decam_flux')[:, 2])
        zz = 22.5-2.5*np.log10(stars.get('decam_flux')[:, 4])
        gr = gg - rr
        rz = rr - zz
        return np.array(rz),np.array(gr)

    def plot_dr2stars(self,savefig=False): 
        rz,gr= self.get_dr2stars()
        fig,ax = plt.subplots(figsize=(5,4))
        plt.subplots_adjust(wspace=0,hspace=0)
        rgb=get_rgb_cols()[0]
        rgb= (rgb[0]/255.,rgb[1]/255.,rgb[2]/255.)
        ax.scatter(rz,gr,c=[rgb],edgecolors='none',marker='o',s=10.,rasterized=True)#,label=key)
        ax.set_xlim([-0.5,2.2])
        ax.set_ylim([-0.3,2.])
        xlab=ax.set_xlabel('r-z')
        ylab=ax.set_ylabel('g-r')
        #handles,labels = ax.get_legend_handles_labels()
        #index=[0,1,2,3]
        #handles,labels= np.array(handles)[index],np.array(labels)[index]
        #leg=ax.legend(handles,labels,loc=(0,1.05),ncol=2,scatterpoints=1,markerscale=2)
        name='STAR_dr2.png'
        if savefig:
            plt.savefig(name,\
                        bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote {}'.format(name))
       
    def plot(self,savefig):
        self.plot_dr2stars(savefig=savefig) 
        #plt.plot(self.cat.get('ra'),self.cat.get('dec'))
        #plt.savefig('test.png')
        #plt.close()

class GalaxyPrior(object):
    def __init__(self):
        #self.qso= QSO()
        self.star= STAR()
        self.lrg= LRG()
        self.elg= ELG()
    def plot_all(self):
        #self.qso.plot()
        self.star.plot()
        self.lrg.plot()
        self.elg.plot()

if __name__ == '__main__':
    gals=GalaxyPrior()
    gals.plot_all()
    print "gals.__dict__= ",gals.__dict__
