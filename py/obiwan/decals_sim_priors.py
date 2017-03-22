'''
Uses intrinsic (dust removed) AB fluxes/mags to select ELGs,LRGs,QSOs,STARs
Makes color-color plots that reproduce the FDR
Single band, mag distributions, plotted using "as observed" AB fluxes/mags,
    since these are what one adds into a CP image
'''

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from astropy.io import fits
#from astropy.table import vstack, Table
from astrometry.util.fits import fits_table, merge_tables
import os
import sys
from glob import glob
from scipy.optimize import newton
from sklearn.neighbors import KernelDensity
import pickle

from theValidator.catalogues import CatalogueFuncs 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from theValidator.catalogues import CatalogueFuncs,Matcher

class EmptyClass(object):
    pass

class KernelOfTruth(object):
    '''Approximate color distributions with a Gaussian Kernel Density Estimator
    See: http://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html
    '''
    def __init__(self,data_list,labels,lims,\
                 bandwidth=0.05,kernel='gaussian',\
                 kdefn='kde.pickle',loadkde=False):
        '''
        array_list -- list of length nfeatures, each element is a numpy array of nsamples
        lims -- list of tuples giving low,hi limits
        '''
        self.kdefn= kdefn
        self.loadkde= loadkde
        self.kernel= kernel
        self.bandwidth= bandwidth
        self.labels= np.array(labels)
        # Data
        self.X= np.array(data_list).T
        assert(self.X.shape[1] == len(data_list))
        # Limits
        self.X_plot=[]
        for i in range(len(lims)):
            self.X_plot+= [np.linspace(lims[i][0],lims[i][1],1000)]
        self.X_plot= np.array(self.X_plot).T
        assert(self.X_plot.shape[1] == len(lims))
        # KDE
        self.kde= self.get_kde()

    def get_kde(self):
        if self.loadkde:
            fout=open(self.kdefn,'r')
            kde= pickle.load(fout)
            fout.close()
            print('loaded kde: %s' % self.kdefn)
            return kde
        else:
            print('fit kde')
            return KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(self.X)

    def save(self,name='kde.pickle'):
        fout=open(name,'w')
        pickle.dump(self.kde,fout)
        fout.close()
        print('Wrote %s' % name)

    # 1D histograms of each dim of KDE
    def plot_indiv_1d(self,lims=None, ndraws=1000,prefix=''):
        samp= self.kde.sample(n_samples=ndraws)
        for i,name in enumerate(self.labels):
            fig,ax= plt.subplots()
            # Data
            h,edges= np.histogram(self.X[:,i],bins=40,normed=True)
            binc= (edges[1:]+edges[:-1])/2.
            ax.step(binc,h,where='mid',lw=1,c='k',label='Data')
            # KDE distribution
            h,edges= np.histogram(samp[:,i],bins=40,normed=True)
            binc= (edges[1:]+edges[:-1])/2.
            ax.step(binc,h,where='mid',lw=1,c='b',label='KDE')
            xlab=ax.set_xlabel(name,fontsize='x-large')
            ylab=ax.set_ylabel('PDF')
            if lims:
                ax.set_xlim(lims[i])
            ax.legend(loc='upper right')
            savenm= 'kde_1d_%s_%s.png' % (prefix,name)
            plt.savefig(savenm,bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote %s' % savenm)

    # 2D scatterplot of selected dims of KDE
    def plot_indiv_2d(self,xy_names,xy_lims=None, ndraws=1000,prefix=''):
        samp= self.kde.sample(n_samples=ndraws)
        for i,_ in enumerate(xy_names):
            xname= xy_names[i][0]
            yname= xy_names[i][1]
            ix= np.where(self.labels == xname)[0][0]
            iy= np.where(self.labels == yname)[0][0]
            # Plot
            fig,ax= plt.subplots(1,2,figsize=(8,5))
            plt.subplots_adjust(wspace=0.2)
            ax[0].scatter(self.X[:,ix],self.X[:,iy],
                          c='k',marker='o',s=10.,rasterized=True,label='Data')
            ax[1].scatter(samp[:,ix],samp[:,iy],
                          c='b',marker='o',s=10.,rasterized=True,label='KDE')
            for cnt in range(2):
                xlab= ax[cnt].set_xlabel(xname)
                ylab= ax[cnt].set_ylabel(yname)
                ax[cnt].legend(loc='upper right')
                if xy_lims:
                    ax[cnt].set_xlim(xy_lims[i][0])
                    ax[cnt].set_ylim(xy_lims[i][1])
            ax[cnt].legend(loc='upper right')
            savenm= 'kde_2d_%s_%s_%s.png' % (prefix,xname,yname)
            plt.savefig(savenm,bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote %s' % savenm)

    def plot_FDR_using_kde(self,obj='LRG',ndraws=1000,prefix=''):
        assert(obj in ['LRG','ELG'])
        for use_data in [True,False]:
            fig,ax = plt.subplots()
            # Add box
            ts= TSBox(src=obj)
            xrange,yrange= xyrange['x_%s' % obj.lower()],xyrange['y_%s' % obj.lower()]
            ts.add_ts_box(ax, xlim=xrange,ylim=yrange)
            # xyrange dashed box
            #for i in range(2):
            #    ax[i].plot([xrange[i],xrange[i]],yrange,'k--')
            #    ax[i].plot(xrange,[yrange[i],yrange[i]],'k--')
            # KDE sample
            if obj == 'LRG':
                xname= 'rz'
                yname= 'rw1'
            elif obj == 'ELG':
                xname= 'rz'
                yname= 'gr'
            ix= np.where(self.labels == xname)[0][0]
            iy= np.where(self.labels == yname)[0][0]
            if use_data:
                ax.scatter(self.X[:,ix],self.X[:,iy],
                           c='k',marker='o',s=10.,rasterized=True,label='Data')
                savenm= 'kde_in_FDR_Data_%s_%s.png' % (prefix,obj)
            else:
                samp= self.kde.sample(n_samples=ndraws)
                ax.scatter(samp[:,ix],samp[:,iy],
                           c='b',marker='o',s=10.,rasterized=True,label='KDE')
                savenm= 'kde_in_FDR_KDE_%s_%s.png' % (prefix,obj)
            # finish
            #ax.set_xlim(xrange[0]-1,xrange[1]+1)
            #ax.set_ylim(yrange[0]-1,yrange[1]+1)
            ax.set_xlim(xrange)
            ax.set_ylim(yrange)
            ax.legend(loc='upper right')
            if obj == 'LRG':
                xlab=ax.set_xlabel('r-z')
                ylab=ax.set_ylabel('r-W1')
            elif obj == 'ELG':
                xlab=ax.set_xlabel('r-z')
                ylab=ax.set_ylabel('g-r')
            ax.set_aspect(1)
            plt.savefig(savenm,bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote %s' % savenm)


    def plot_1band_and_color(self, ndraws=1000,xylims=None,prefix=''):
        '''xylims -- dict of x1,y1,x2,y2,... where x1 is tuple of low,hi for first plot xaxis'''
        fig,ax= plt.subplots(2,2,figsize=(15,10))
        plt.subplots_adjust(wspace=0.2,hspace=0.2)
        # Data
        xyz= self.kde.sample(n_samples=ndraws)
        ax[0,0].hist(self.X[:,0],normed=True)
        ax[0,1].scatter(self.X[:,1],self.X[:,2],\
                      c='b',edgecolors='none',marker='o',s=10.,rasterized=True,alpha=0.2)
        # KDE distribution
        xyz= self.kde.sample(n_samples=ndraws)
        ax[1,0].hist(xyz[:,0],normed=True)
        ax[1,1].scatter(xyz[:,1],xyz[:,2],\
                      c='b',edgecolors='none',marker='o',s=10.,rasterized=True,alpha=0.2)
        for cnt in range(2):
            if xylims is not None:
                ax[cnt,0].set_xlim(xylims['x1'])
                ax[cnt,0].set_ylim(xylims['y1'])
                ax[cnt,1].set_xlim(xylims['x2'])
                ax[cnt,1].set_ylim(xylims['y2'])
            xlab=ax[cnt,0].set_xlabel(self.labels[0],fontsize='x-large')
            xlab=ax[cnt,1].set_xlabel(self.labels[1],fontsize='x-large')
            ylab=ax[cnt,1].set_ylabel(self.labels[2],fontsize='x-large')
        plt.savefig('%skde.png' % prefix,bbox_extra_artists=[xlab], bbox_inches='tight',dpi=150)
        plt.close()
        if prefix == 'lrg_':
            # plot g band distribution even though no Targeting cuts on g
            fig,ax= plt.subplots(2,1,figsize=(8,10))
            plt.subplots_adjust(hspace=0.2)
            ax[0].hist(self.X[:,3],normed=True)
            ax[1].hist(xyz[:,3],normed=True) 
            for cnt in range(2):
                if xylims is not None:
                    ax[cnt].set_xlim(xylims['x3'])
                    ax[cnt].set_ylim(xylims['y3'])
            xlab=ax[0].set_xlabel(self.labels[3],fontsize='x-large')
            plt.savefig('%sg_kde.png' % prefix,bbox_extra_artists=[xlab], bbox_inches='tight',dpi=150)
            plt.close()

    def plot_1band_color_and_redshift(self, ndraws=1000,xylims=None,prefix=''):
        '''xylims -- dict of x1,y1,x2,y2,... where x1 is tuple of low,hi for first plot xaxis'''
        fig,ax= plt.subplots(2,3,figsize=(20,10))
        plt.subplots_adjust(wspace=0.2,hspace=0.2)
        # Colormap the color-color plot by redshift
        cmap = mpl.colors.ListedColormap(['m','r', 'y', 'g','b', 'c'])
        bounds= np.linspace(xylims['x3'][0],xylims['x3'][1],num=6)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # Data
        ax[0,0].hist(self.X[:,0],normed=True)
        # color bar with color plot
        axobj= ax[0,1].scatter(self.X[:,1],self.X[:,2],c=self.X[:,3],\
                                 marker='o',s=10.,rasterized=True,lw=0,\
                                 cmap=cmap,norm=norm,\
                                 vmin=bounds.min(),vmax=bounds.max())
        divider3 = make_axes_locatable(ax[0,1])
        cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        cbar3 = plt.colorbar(axobj, cax=cax3,\
                             cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds)
        cbar3.set_label('redshift')
        #
        ax[0,2].hist(self.X[:,3],normed=True)
        #Xtest= np.linspace(-0.2,1.6,num=100)
        #log_dens = self.kde.score_samples(self.X)
        #ax[0,2].plot(self.X[:,3], np.exp(log_dens), 'y-')
        # KDE distribution
        samp= self.kde.sample(n_samples=ndraws)
        ax[1,0].hist(samp[:,0],normed=True)
        ax[1,1].scatter(samp[:,1],samp[:,2],\
                      c='b',edgecolors='none',marker='o',s=10.,rasterized=True,alpha=0.2)
        ax[1,2].hist(samp[:,3],normed=True)
        for cnt in range(2):
            if xylims is not None:
                ax[cnt,0].set_xlim(xylims['x1'])
                ax[cnt,0].set_ylim(xylims['y1'])
                ax[cnt,1].set_xlim(xylims['x2'])
                ax[cnt,1].set_ylim(xylims['y2'])
                ax[cnt,2].set_xlim(xylims['x3'])
                ax[cnt,2].set_ylim(xylims['y3'])
            xlab=ax[cnt,0].set_xlabel(self.labels[0],fontsize='x-large')
            xlab=ax[cnt,1].set_xlabel(self.labels[1],fontsize='x-large')
            ylab=ax[cnt,1].set_ylabel(self.labels[2],fontsize='x-large')
            xlab=ax[cnt,2].set_xlabel(self.labels[3],fontsize='x-large')
        sname='%skde.png' % prefix
        plt.savefig(sname,bbox_extra_artists=[xlab], bbox_inches='tight',dpi=150)
        plt.close()
        print('Wrote %s' % sname)
        if prefix == 'lrg_':
            # plot g band distribution even though no Targeting cuts on g
            fig,ax= plt.subplots(2,1,figsize=(8,10))
            plt.subplots_adjust(hspace=0.2)
            ax[0].hist(self.X[:,4],normed=True)
            ax[1].hist(samp[:,4],normed=True) 
            for cnt in range(2):
                if xylims is not None:
                    ax[cnt].set_xlim(xylims['x4'])
                    ax[cnt].set_ylim(xylims['y4'])
            xlab=ax[0].set_xlabel(self.labels[4],fontsize='x-large')
            sname='%sg_kde.png' % prefix
            plt.savefig(sname,bbox_extra_artists=[xlab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote %s' % sname)

    def plot_galaxy_shapes(self, ndraws=1000,xylims=None,name='kde.png'):
        '''xylims -- dict of x1,y1,x2,y2,... where x1 is tuple of low,hi for first plot xaxis'''
        fig,ax= plt.subplots(2,4,figsize=(20,10))
        plt.subplots_adjust(wspace=0.2,hspace=0.2)
        samp= self.kde.sample(n_samples=ndraws)
        # ba,pa can be slightly greater 1.,180
        samp[:,2][ samp[:,2] > 1. ]= 1.
        samp[:,3][ samp[:,3] > 180. ]= 180.
        # Physical values
        assert(np.all(samp[:,0] > 0))
        assert(np.all((samp[:,1] > 0)*\
                      (samp[:,1] < 10)))
        assert(np.all((samp[:,2] > 0)*\
                      (samp[:,2] <= 1.)))
        assert(np.all((samp[:,3] >= 0)*\
                      (samp[:,3] <= 180)))
        # plot
        for cnt in range(4):
            if cnt == 0:
                bins=np.linspace(0,80,num=20)
                # Data
                ax[0,cnt].hist(self.X[:,cnt],bins=bins,normed=True)
                # KDE distribution
                ax[1,cnt].hist(samp[:,cnt],bins=bins,normed=True)
            else:
                ax[0,cnt].hist(self.X[:,cnt],normed=True)
                ax[1,cnt].hist(samp[:,cnt],normed=True)
        # lims
        for row in range(2):
            for col in range(4):
                if xylims is not None:
                    ax[row,col].set_xlim(xylims['x%s' % str(col+1)])
                    #ax[cnt,1].set_xlim(xylims['x2'])
                    #ax[cnt,2].set_xlim(xylims['x3'])
                    xlab=ax[row,col].set_xlabel(self.labels[col],fontsize='x-large')
                    #xlab=ax[cnt,1].set_xlabel(self.labels[1],fontsize='x-large')
        plt.savefig(name,bbox_extra_artists=[xlab], bbox_inches='tight',dpi=150)
        plt.close()
        print('Wrote %s' % name)



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

#def flux2mag(nanoflux):
#    return 22.5-2.5*np.log10(nanoflux)

# Globals
xyrange=dict(x_star=[-0.5,2.2],\
             y_star=[-0.3,2.],\
             x_elg=[-0.5,2.2],\
             y_elg=[-0.3,2.],\
             x_lrg= [0, 3.],\
             y_lrg= [-2, 6],\
             x1_qso= [-0.5,3.],\
             y1_qso= [-0.5,2.5],\
             x2_qso= [-0.5,4.5],\
             y2_qso= [-2.5,3.5])

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



#def elg_data():
#    '''Use DEEP2 ELGs whose SEDs have been modeled.'''
#    elgs = fits.getdata('/project/projectdirs/desi/spectro/templates/basis_templates/v2.2/elg_templates_v2.0.fits', 1)
#    # Colors
#    gg = elgs['DECAM_G']
#    rr = elgs['DECAM_R']
#    zz = elgs['DECAM_Z']
#    gr = gg - rr
#    rz = rr - zz
#    Xall = np.array([rz, gr]).T
#    # Cuts
#    has_morph = elgs['radius_halflight'] > 0
#    cuts= dict(has_morph=has_morph) 
#    print('%d/%d of Fit Template Spectra ELGs have morphologies' % (len(elgs[has_morph]), len(elgs)))
#    # Morphology
#    morph= {}
#    morph['rz'] = rz[has_morph]
#    morph['gr'] = gr[has_morph]
#    morph['r50'] = elgs['RADIUS_HALFLIGHT'][has_morph] #arcsec
#    morph['n'] = elgs['SERSICN'][has_morph]                            
#    morph['ba'] = elgs['AXIS_RATIO'][has_morph] #minor/major
#    return Xall,cuts,morph                            


class ReadWrite(object):
    def read_fits(self,fn):
        #return Table(fits.getdata(fn, 1))
        return fits_table(fn)

#DR=2,savefig=False,alpha=1.,\
#brick_primary=True,anymask=False,allmask=False,fracflux=False):
class CommonInit(ReadWrite):
    def __init__(self,**kwargs):
        # Syntax is self.val2 = kwargs.get('val2',"default value")
        self.DR= kwargs.get('DR',2)
        self.savefig= kwargs.get('savefig',False)
        self.alpha= kwargs.get('alpha',1.)
        self.brick_primary= kwargs.get('brick_primary',True)
        self.anymask= kwargs.get('anymask',False)
        self.allmask= kwargs.get('allmask',False)
        self.fracflux= kwargs.get('fracflux',False)
        print('self.fracflux=',self.fracflux,'kwargs= ',kwargs)
        if self.DR == 2:
            self.truth_dir= '/project/projectdirs/desi/target/analysis/truth'
        elif self.DR == 3:
            self.truth_dir= '/project/projectdirs/desi/users/burleigh/desi/target/analysis/truth'
        else: raise valueerror()
        # KDE params
        self.kdefn= kwargs.get('kdefn','kde.pickle')
        self.loadkde= kwargs.get('loadkde',False)
        self.savekde= kwargs.get('savekde',False)

    def imaging_cut(self,data):
        '''data is a fits_table object with Tractor Catalogue columns'''
        cut=np.ones(len(data)).astype(bool)
        # Brick Primary
        if data.get('brick_primary').dtype == 'bool':
            cut*= data.get('brick_primary') == True
        elif data.get('brick_primary').dtype == 'S1':
            cut*= data.get('brick_primary') == 'T'
        else: 
            raise ValueError('brick_primary has type=',data.get('brick_primary').dtype)
        #if self.anymask:
        #    cut*= np.all((data.get('decam_anymask')[:, [1,2,4]] == 0),axis=1)
        # ALL Mask
        cut*= np.all((data.get('decam_allmask')[:, [1,2,4]] == 0),axis=1)
        # FracFlux
        cut*= np.all((data.get('decam_fracflux')[:, [1,2,4]] < 0.05),axis=1)
        return cut

    def std_star_cut(self,data):
        '''See: https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection#SpectrophotometricStandardStarsFSTD
           data is a fits_table object with Tractor Catalogue columns
        '''
        RFLUX_obs = data.get('decam_flux')[:,2]
        GFLUX = data.get('decam_flux_nodust')[:,1]
        RFLUX = data.get('decam_flux_nodust')[:,2]
        ZFLUX = data.get('decam_flux_nodust')[:,4]
        GRZSN = data.get('decam_flux')[:,[1,2,4]] * np.sqrt(data.get('decam_flux_ivar')[:,[1,2,4]])
        GRCOLOR = 2.5 * np.log10(RFLUX / GFLUX)
        RZCOLOR = 2.5 * np.log10(ZFLUX / RFLUX)
        cut= np.all((data.get('brick_primary') == True,\
                     data.get('type') == 'PSF',\
                     np.all((data.get('decam_allmask')[:, [1,2,4]] == 0),axis=1),\
                     np.all((data.get('decam_fracflux')[:, [1,2,4]] < 0.04),axis=1),\
                     np.all((GRZSN > 10),axis=1),\
                     RFLUX_obs < 10**((22.5-16.0)/2.5)),axis=0)
                     #np.power(GRCOLOR - 0.32,2) + np.power(RZCOLOR - 0.13,2) < 0.06**2,\
                     #RFLUX_obs > 10**((22.5-19.0)/2.5)),axis=0)
        return cut 

class CrossValidator():
	def kfold_indices(k,n_train):
		'''returns array of indices of shape (k,n_train/k) to grab the k bins of training data'''
		bucket_sz=int(n_train/float(k))
		ind= np.arange(k*bucket_sz) #robust for any k, even if does not divide evenly!
		np.random.shuffle(ind) 
		return np.reshape(ind,(k,bucket_sz))

	def kfold_cross_val(C=1.,kfolds=10,):
		'''C is parameter varying'''
		nsamples= self.X.shape[0]
		ind= kfold_indices(kfolds,nsamples)
		err=np.zeros(kfolds)-1
		keep_clf=dict(err=1.)
		for i in range(kfolds):
			ival= ind[i,:]
			itrain=np.array(list(set(ind.flatten())-set(ival)))
			assert(len(list(set(ival) | set(itrain))) == len(ind.flatten()))
			# Train
			clf = SVC(C=C,kernel='rbf',degree=3)
			clf.fit(self.X[itrain,:],self.y[itrain])
			#get error for this kth sample
			pred= clf.predict(self.X[ival,:])
			err[i],wrong= benchmark(pred, self.y[ival])
			if err[i] <= keep_clf['err']: 
				keep_clf['lsvc']=clf
				keep_clf['err']=err[i]
		return err,keep_clf

class RedshiftPredictor(CrossValidator):
    def __init__(self,X=None,y=None):
        # All NaNs must be removed
        X,y= self.remove_nans(X,y=y)
        self.Xtrain= X
        self.ytrain= y

    def svm(self,kernel='rbf',C=1.,degree=3):
        self.clf = SVC(kernel=kernel,C=C,degree=degree)
        self.clf.fit(self.Xtrain, self.ytrain)

    def nn(self,nn=3):
        self.clf= KNeighborsClassifier(n_neighbors=nn)
        self.clf.fit(self.Xtrain, self.ytrain)

    def predict(self,Xtest):
        Xtest= self.remove_nans(Xtest)
        return self.clf.predict(Xtest)

    def remove_nans(self,X,y=None):
        sh= X.shape
        assert(len(sh) == 2)
        assert(sh[1] == 2)
        keep= np.isfinite(X[:,0])*np.isfinite(X[:,1])    
        X= X[keep,:]
        if X.shape[0] < sh[0]:
            print('WARNING: %d/%d of X values were NANs, removed these' % \
                    (sh[0]-X.shape[0],X.shape[0]))
        if y is not None:
            y=y[keep]
            return X,y
        else:
            return X

   
    def cross_validate(self,typ='ELG'):
        k=5
        Cvals= np.logspace(-1,1,num=5)
        avgerr= np.zeros(len(Cvals))-1
        for cnt,C in enumerate(Cvals):
            print 'kfold cv on C= %g' % C
            err,junk= self.kfold_cross_val(C=C,kfolds=k)
            print 'err= ',err
            avgerr[cnt]= err.mean()
        print 'avg err= ',avgerr,'C_vals= ',Cvals
        fout=open('cross_validate_svm_%s.pickle' % typ,'w')
        pickle.dump((Cvals,avgerr),fout)
        fout.close()
        plt.plot(Cvals,avgerr,'k-')
        plt.scatter(Cvals,avgerr,s=50,color='b')
        plt.xlabel("C (regularization parameter)")
        plt.ylabel('Error rate')
        plt.title('Training')
        plt.xscale('log')
        plt.savefig('cross_validate_svm_%s.png' % typ)
        ibest= np.where(avgerr == avgerr.min())[0][0]
        print 'lowest cross valid error= ',avgerr[ibest],'for C = ',Cvals[ibest]	

def get_acs_six_col(self):
    savedir='/project/projectdirs/desi/users/burleigh/desi/target/analysis/truth'
    savenm= os.path.join(savedir,'acs_six_cols.fits')
    if os.path.exists(savenm):
        tab=fits_table(savenm)
    else:
        acsfn=os.path.join('/project/projectdirs/desi/users/burleigh/desi/target/analysis/truth','ACS-GC_published_catalogs','acs_public_galfit_catalog_V1.0.fits.gz')
        acs=fits_table(acsfn) 
        # Repackage
        tab= fits_table()
        for key in ['ra','dec','re_galfit_hi','n_galfit_hi','ba_galfit_hi','pa_galfit_hi']:
            tab.set(key, acs.get(key))
        # Cuts & clean
        tab.cut( tab.flag_galfit_hi == 0 )
        # -90,+90 --> 0,180 
        tab.pa_galfit_hi += 90. 
        # Save
        tab.writeto(savenm)
        print('Wrote %s' % savenm)
    return tab
 

class ELG(CommonInit):
    def __init__(self,**kwargs):
        super(ELG, self).__init__(**kwargs)
        self.rlimit= kwargs.get('rlimit',23.4)
        print('ELGs, self.rlimit= ',self.rlimit)
        # KDE params
        self.kdefn= 'elg-kde.pickle'
        self.kde_shapes_fn= 'elg-shapes-kde.pickle'
        self.kde_colors_shapes_fn= 'elg-colors-shapes-kde.pickle'

    def get_dr3_deep2(self):
        '''version 3.0 of data discussed in
        https://desi.lbl.gov/DocDB/cgi-bin/private/RetrieveFile?docid=912'''
        # Cut on DR3 rmag < self.rlimit
        if self.DR == 2:
            zcat = self.read_fits(os.path.join(self.truth_dir,'../deep2/v3.0/','deep2-field1-oii.fits.gz'))
            R_MAG= zcat.get('cfhtls_r')
            rmag_cut= R_MAG<self.rlimit 
        elif self.DR == 3:
            zcat = self.read_fits(os.path.join(self.truth_dir,'deep2f234-dr3matched.fits'))
            decals = self.read_fits(os.path.join(self.truth_dir,'dr3-deep2f234matched.fits'))
            # Add mag data 
            CatalogueFuncs().set_mags(decals)
            rflux= decals.get('decam_flux_nodust')[:,2]
            rmag_cut= rflux > 10**((22.5-self.rlimit)/2.5)
            rmag_cut*= self.imaging_cut(decals)
        zcat.cut(rmag_cut) 
        decals.cut(rmag_cut) 
        # color data
        if self.DR == 2:
            G_MAG= zcat.get('cfhtls_g')
            R_MAG= zcat.get('cfhtls_r')
            Z_MAG= zcat.get('cfhtls_z')
        elif self.DR == 3:
            G_MAG= decals.get('decam_mag_wdust')[:,1]
            R_MAG= decals.get('decam_mag_wdust')[:,2]
            Z_MAG= decals.get('decam_mag_wdust')[:,4]
            R_MAG_nodust= decals.get('decam_mag_nodust')[:,2]
            # Don't need w1, but might be useful
            W1_MAG = decals.get('wise_mag_wdust')[:,0]
        # Repackage
        tab= fits_table()
        # Deep2
        tab.set('ra', zcat.ra)
        tab.set('dec', zcat.dec)
        tab.set('zhelio', zcat.zhelio)
        tab.set('oii_3727', zcat.oii_3727)
        tab.set('oii_3727_err', zcat.oii_3727_err)
        # Decals
        tab.set('g_wdust', G_MAG)
        tab.set('r_wdust', R_MAG)
        tab.set('z_wdust', Z_MAG)
        tab.set('w1_wdust', W1_MAG) 
        tab.set('r_nodust', R_MAG_nodust)
        # DR3 shape info
        for key in ['type','shapeexp_r','shapedev_r']:
            tab.set(key, decals.get(key))
        return tab
 

    def get_FDR_cuts(self,tab):
        oiicut1 = 8E-17 # [erg/s/cm2]
        zmin = 0.6
        keep= {}
        keep['lowz'] = tab.zhelio < zmin
        keep['medz_lowO2'] = np.all((tab.zhelio > zmin,\
                                 tab.oii_3727_err != -2.0,\
                                 tab.oii_3727 < oiicut1), axis=0)
        keep['medz_hiO2'] = np.all((tab.zhelio > zmin,\
                                tab.zhelio < 1.0,\
                                tab.oii_3727_err != -2.0,\
                                tab.oii_3727 > oiicut1), axis=0)
        keep['hiz_hiO2'] = np.all((tab.zhelio > 1.0,\
                                   tab.oii_3727_err !=-2.0,\
                               tab.oii_3727 > oiicut1),axis=0)
        return keep 

    def get_obiwan_cuts(self,tab):
        return (tab.zhelio >= 0.8) * \
               (tab.zhelio <= 1.4) * \
               (tab.oii_3727 >= 0.) * \
               (tab.oii_3727_err > 0.)

    def fit_kde(self, use_acs=False):
        # Load Data
        if use_acs:
            print('Matching acs to dr3,deep2')
            deep= self.get_dr3_deep2()
            acs= get_acs_six_col()
            imatch,imiss,d2d= Matcher().match_within(deep,acs,dist=1./3600)
            deep.cut(imatch['ref'])
            acs.cut(imatch['obs'])
            # Remove ra,dec from acs, then merge
            for key in ['ra','dec']:
                acs.delete_column(key)
            tab= merge_tables([deep,acs], columns='fillzero')
        else:
            tab= self.get_dr3_deep2()
        print('dr3_deep2 %d' % len(tab))
        # Add tractor shapes
        dic= get_tractor_shapes(tab)
        tab.set('tractor_re', dic['re'])
        tab.set('tractor_n', dic['n'])
        # Cuts
        keep= self.get_obiwan_cuts(tab)
        tab.cut(keep)
        print('dr3_deep2, after obiwan cut %d' % len(tab))
        # Cut bad values
        keep= ( np.ones(len(tab),bool) )
        for col in tab.get_columns():
            if col == 'type':
                continue
            keep *= (np.isfinite(tab.get(col)))
        tab.cut(keep)
        print('dr3_deep2, after cut bad vals %d' % len(tab))
        # Sanity plots
        plot_tractor_shapes(tab,prefix='ELG_dr3deep2_expdev')
        print('size %d %d' % (len(tab),len(tab[tab.tractor_re > 0.])))
        tab.cut( tab.tractor_re > 0. )
        xy_names= [('tractor_re','zhelio'),
                   ('tractor_re','g_wdust'),
                   ('tractor_re','r_wdust'),
                   ('tractor_re','z_wdust'),
                   ('zhelio','g_wdust'),
                   ('zhelio','r_wdust'),
                   ('zhelio','z_wdust')]
        #xy_lims= [('tractor_re','zhelio'),
        #           ('tractor_re','r_wdust'),
        #           ('tractor_re','g_wdust'),
        #           ('tractor_re','z_wdust')]
        plot_indiv_2d(tab,xy_names=xy_names,xy_lims=None, ndraws=1000,prefix='ELG_dr3deep2')
        # KDE
        labels=['r_wdust','rz','gr','zhelio','tractor_re']
                #'re','n','ba','pa']
        lims= [(20.5,25.),(0,2),(-0.5,1.5),(0.6,1.6),(0.3,1.5)]
              # (0.,100.),(0.,10.),(0.2,0.9),(0.,180.)]
        kde_obj= KernelOfTruth([tab.r_wdust, tab.r_wdust - tab.z_wdust,
                                tab.g_wdust - tab.r_wdust, tab.zhelio,
                                tab.tractor_re],
                               labels,lims,\
                               bandwidth=0.05,kernel='tophat',\
                               kdefn=self.kde_shapes_fn,loadkde=self.loadkde)
        xy_names= [('rz','gr'),
                   ('tractor_re','gr'),
                   ('tractor_re','r_wdust'),
                   ('tractor_re','rz'),
                   ('zhelio','tractor_re'),
                   ('zhelio','gr'),
                   ('zhelio','r_wdust'),
                   ('zhelio','rz')]
        xy_lims= [([0,2],[-0.5,1.5]),
                  ([0.,2],[-0.5,1.5]),  
                  ([0.,2],[20.5,25.]),  
                  ([0.,2],[0,2]),  
                  ([0.6,1.6],[0,2]),  
                  ([0.6,1.6],[-0.5,1.5]),  
                  ([0.6,1.6],[20.5,25]),  
                  ([0.6,1.6],[0,2]),  
                 ]
        #kde_obj.plot_indiv_2d(xy_names,xy_lims=xy_lims, ndraws=10000,prefix='ELG_dr3deep2')
        kde_obj.plot_FDR_using_kde(obj='ELG',ndraws=10000,prefix='dr3deep2')
#        xylims=dict(x1=(20.5,25.5),y1=(0,0.8),\
#                    x2=xyrange['x_elg'],y2=xyrange['y_elg'],\
#                    x3=(0.6,1.6),y3=(0.,1.0),
#                    x4=(0,100),\
#                    x5=(0,10),\
#                    x6=(0,1),\
#                    x7=(0,180))
        #kde_obj.plot_1band_and_color(ndraws=1000,xylims=xylims,prefix='elg_')
        #kde_obj.plot_colors_shapes_z(ndraws=1000,xylims=xylims,name='elg_colors_shapes_z_kde.png')
        if self.savekde:
            if os.path.exists(self.kde_colors_shapes_fn):
                os.remove(self.kde_colors_shapes_fn)
            kde_obj.save(name=self.kde_colors_shapes_fn)

 
    def cross_validate_redshift(self):
        rz,gr,r_nodust,r_wdust,redshift= self.get_elgs_FDR_cuts()
        key='goodz_oiibright'
        categors= np.zeros(len(redshift[key])).astype(int)
        zbins=np.linspace(0.8,1.4,num=6)
        for lft,rt in zip(zbins[:-1],zbins[1:]):
            categors[key][ (redshift[key] > lft)*(redshift[key] <= rt) ]= cnt+1
        X= np.array([rz[key],gr[key]]).T
        obj= RedshiftPredictor(X=X,\
                               y=categors[key])
        obj.cross_validate(typ='ELG')
         

    def plot_redshift(self):
        rz,gr,r_nodust,r_wdust,redshift= self.get_elgs_FDR_cuts()
        ts= TSBox(src='ELG')
        xrange,yrange= xyrange['x_elg'],xyrange['y_elg']
        # Plot
        fig,ax = plt.subplots(3,2,sharex=True,sharey=True,figsize=(10,12))
        plt.subplots_adjust(wspace=0.25,hspace=0.15)
        # Plot data
        zbins=np.linspace(0.8,1.4,num=6)
        mysvm,mynn,categors={},{},{}
        # Plot, Predict, Predict
        # Discrete Color Map
        cmap = mpl.colors.ListedColormap(['m','r', 'y', 'g','b', 'c'])
        bounds = zbins
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # Make plots
        for cnt,key,ti in zip(range(2),
                              ['goodz_oiibright','goodz_oiifaint'],\
                              [r'$z=(0.8,1.4) [OII]>8\times10^{-17}$',r'$z=(0.8,1.4) [OII]<8\times10^{-17}$']):
            print('Learning redshifts for sample=%s' % key)
            # Add box
            ts.add_ts_box(ax[0,cnt], xlim=xrange,ylim=yrange)
            # Scatter
            axobj= ax[0,cnt].scatter(rz[key],gr[key],c=redshift[key],marker='o',\
                                     cmap=cmap,norm=norm,\
                                     vmin=bounds.min(),vmax=bounds.max())
            title=ax[0,cnt].set_title(ti)
            divider3 = make_axes_locatable(ax[0,cnt])
            cax3 = divider3.append_axes("right", size="5%", pad=0.1)
            cbar3 = plt.colorbar(axobj, cax=cax3,\
                                 cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds)
            cbar3.set_label('redshift')
            #cbar = fig.colorbar(axobj, orientation='vertical')
            # Train & Predict
            categors[key]= np.zeros(len(redshift[key])).astype(int)
            for ithbin,lft,rt in zip(range(len(zbins)-1),zbins[:-1],zbins[1:]):
                categors[key][ (redshift[key] > lft)*(redshift[key] <= rt) ]= ithbin+1
            X= np.array([rz[key],gr[key]]).T
            mysvm[key]= RedshiftPredictor(X=X, y=categors[key])
            mysvm[key].svm(kernel='rbf',C=1.,degree=3)
            # FIX!!!! ADD 5 color color bar, len(zbins)-1, 
            axobj= ax[1,cnt].scatter(rz[key],gr[key],c=mysvm[key].predict(X),\
                                     marker='o',\
                                     cmap=cmap,norm=norm,\
                                     vmin=bounds.min(),vmax=bounds.max())
            #cbar = fig.colorbar(axobj, orientation='vertical')
            divider3 = make_axes_locatable(ax[1,cnt])
            cax3 = divider3.append_axes("right", size="5%", pad=0.1)
            cbar3 = plt.colorbar(axobj, cax=cax3,\
                                 cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds)
            cbar3.set_label('redshift')
            # FIX add lines dividing the two samples, see sklearn doc plot examples
            # Train & Predict
            mynn[key]= RedshiftPredictor(X=X,y=categors[key])
            mynn[key].nn(nn=3)
            # FIX!!!! ADD 5 color color bar, len(zbins)-1, 
            axobj= ax[2,cnt].scatter(rz[key],gr[key],c=mynn[key].predict(X),\
                                     marker='o',\
                                     cmap=cmap,norm=norm,\
                                     vmin=bounds.min(),vmax=bounds.max())
            #cbar = fig.colorbar(axobj, orientation='vertical')
            divider3 = make_axes_locatable(ax[2,cnt])
            cax3 = divider3.append_axes("right", size="5%", pad=0.1)
            cbar3 = plt.colorbar(axobj, cax=cax3,\
                                 cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds)
            cbar3.set_label('redshift')
        # Finish axes
        for row in range(3):
            ylab= ax[row,0].set_ylabel('g-r')
            for col in range(2):
                ax[row,col].set_xlim(xrange)
                ax[row,col].set_ylim(yrange)
                xlab= ax[2,col].set_xlabel('r-z')
        # Save
        name='dr%d_ELG_redshifts.png' % self.DR
        kwargs= dict(bbox_extra_artists=[cbar3,title,xlab,ylab]) #, bbox_inches='tight',dpi=150)
        if self.savefig:
            plt.savefig(name, **kwargs)
            plt.close()
            print('Wrote {}'.format(name))

    def plot_FDR(self):
        tab= self.get_dr3_deep2()
        # Cuts 'lowz','medz_lowO2' ...
        keep= self.get_FDR_cuts(tab)
        # Plot
        fig, ax = plt.subplots()
        # Add box
        ts= TSBox(src='ELG')
        xrange,yrange= xyrange['x_elg'],xyrange['y_elg']
        ts.add_ts_box(ax, xlim=xrange,ylim=yrange)
        # Add points
        for cut_name,lab,color,marker in zip(['lowz','medz_lowO2','medz_hiO2','hiz_hiO2'],
                        [r'$z<0.6$',r'$z>0.6, [OII]<8\times10^{-17}$',
                         r'$z>0.6, [OII]>8\times10^{-17}$',r'$z>1.0, [OII]>8\times10^{-17}$'],
                                             ['magenta','tan','powderblue','blue'],
                                             ['^','s','o','o']):
            cut= keep[cut_name]
            ax.scatter(tab.rz[cut],tab.gr[cut], 
                       marker=marker, color=color, label=lab)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        xlab= ax.set_xlabel('r-z')
        ylab= ax.set_ylabel('g-r')
        leg=ax.legend(loc=(0,1.05), ncol=2,prop={'size': 14}, labelspacing=0.2,
                  markerscale=1.5)
        name='dr%d_FDR_ELG.png' % self.DR
        kwargs= dict(bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight',dpi=150)
        if self.savefig:
            plt.savefig(name, **kwargs)
            plt.close()
            print('Wrote {}'.format(name))

    def plot_FDR_multi(self):
        tab= self.get_dr3_deep2()
        # Cuts 'lowz','medz_lowO2' ...
        keep= self.get_FDR_cuts(tab)
        # Plot
        fig,ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(18,4))
        plt.subplots_adjust(wspace=0.1,hspace=0)
        ts= TSBox(src='ELG')
        xrange,yrange= xyrange['x_elg'],xyrange['y_elg']
        for cnt,cut_name,lab,color,marker in zip(range(4),
                        ['lowz','medz_lowO2','medz_hiO2','hiz_hiO2'],
                        [r'$z<0.6$',r'$z>0.6, [OII]<8\times10^{-17}$',
                            r'$z>0.6, [OII]>8\times10^{-17}$',r'$z>1.0, [OII]>8\times10^{-17}$'],
                        ['magenta','tan','powderblue','blue'],
                        ['^','s','o','o']):
            # Add box
            ts.add_ts_box(ax[cnt], xlim=xrange,ylim=yrange)
            # Add points
            cut= keep[cut_name]
            ax[cnt].scatter(tab.rz[cut],tab.gr[cut], marker=marker,color=color)
            ti_loc=ax[cnt].set_title(lab)
            ax[cnt].set_xlim(xrange)
            ax[cnt].set_ylim(yrange)
            xlab= ax[cnt].set_xlabel('r-z')
            ylab= ax[cnt].set_ylabel('g-r')
        name='dr%d_ELG_FDR_multi.png' % self.DR
        kwargs= dict(bbox_extra_artists=[ti_loc,xlab,ylab], bbox_inches='tight',dpi=150)
        if self.savefig:
            plt.savefig(name, **kwargs)
            plt.close()
            print('Wrote {}'.format(name))

    def plot_obiwan_multi(self):
        tab= self.get_dr3_deep2()
        # Cuts 'lowz','medz_lowO2' ...
        keep= self.get_obiwan_cuts(tab)
        # Plot
        fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(9,4))
        plt.subplots_adjust(wspace=0.1,hspace=0)
        ts= TSBox(src='ELG')
        xrange,yrange= xyrange['x_elg'],xyrange['y_elg']
        for cnt,thecut,lab,color in zip(range(2),
                            [keep, keep == False],
                            [r'$0.8<z<1.4, [OII] > 0$','Everything else'],
                            ['b','g']):
            # Add box
            ts.add_ts_box(ax[cnt], xlim=xrange,ylim=yrange)
            # Add points
            ax[cnt].scatter(tab.rz[thecut],tab.gr[thecut], marker='o',color=color)
            ti_loc=ax[cnt].set_title(lab)
            ax[cnt].set_xlim(xrange)
            ax[cnt].set_ylim(yrange)
            xlab= ax[cnt].set_xlabel('r-z')
            ylab= ax[cnt].set_ylabel('g-r')
        name='dr%d_ELG_obiwan_multi.png' % self.DR
        kwargs= dict(bbox_extra_artists=[ti_loc,xlab,ylab], bbox_inches='tight',dpi=150)
        if self.savefig:
            plt.savefig(name, **kwargs)
            plt.close()
            print('Wrote {}'.format(name))


    def plot_LRG_FDR_wELG_data(self):
        dic= self.get_elgs_FDR_cuts()
        ts= TSBox(src='LRG')
        xrange,yrange= xyrange['x_lrg'],xyrange['y_lrg']
        # Plot
        fig,ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(18,4))
        plt.subplots_adjust(wspace=0.1,hspace=0)
        for cnt,key,col,marker,ti in zip(range(4),\
                               ['loz','oiifaint','oiibright_loz','oiibright_hiz'],\
                               ['magenta','tan','powderblue','blue'],\
                               ['^','s','o','o'],\
                               [r'$z<0.6$',r'$z>0.6, [OII]<8\times10^{-17}$',r'$z>0.6, [OII]>8\times10^{-17}$',r'$z>1.0, [OII]>8\times10^{-17}$']):
            # Add box
            ts.add_ts_box(ax[cnt], xlim=xrange,ylim=yrange)
            # Add points
            b= key
            ax[cnt].scatter(dic['rz'][b],dic['rw1'][b], marker=marker,color=col)
            ti_loc=ax[cnt].set_title(ti)
            ax[cnt].set_xlim(xrange)
            ax[cnt].set_ylim(yrange)
            xlab= ax[cnt].set_xlabel('r-z')
            ylab= ax[cnt].set_ylabel('r-W1')
            name='dr%d_LRG_FDR_wELG_data.png' % self.DR
        kwargs= dict(bbox_extra_artists=[ti_loc,xlab,ylab], bbox_inches='tight',dpi=150)
        if self.savefig:
            plt.savefig(name, **kwargs)
            plt.close()
            print('Wrote {}'.format(name))

    def plot_FDR_mag_dist(self):
        tab= self.get_dr3_deep2()
        keep= self.get_FDR_cuts(tab)
        # Plot
        fig,ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(18,4))
        plt.subplots_adjust(wspace=0.1,hspace=0)
        for cnt,cut_name,lab in zip(range(4),
                        ['lowz','medz_lowO2','medz_hiO2','hiz_hiO2'],
                        [r'$z<0.6$',r'$z>0.6, [OII]<8\times10^{-17}$',
                            r'$z>0.6, [OII]>8\times10^{-17}$',r'$z>1.0, [OII]>8\times10^{-17}$']):
            # Mag data
            cut= keep[cut_name]
            mags=dict(r= tab.r_wdust[cut],
                      g= tab.gr[cut] + tab.r_wdust[cut],
                      z= tab.r_wdust[cut] - tab.rz[cut]) 
            for band,color in zip(['g','r','z'],['g','r','m']):
                # nans present
                mags[band]= mags[band][ np.isfinite(mags[band]) ]
                # histograms
                h,edges= np.histogram(mags[band],bins=40,normed=True)
                binc= (edges[1:]+edges[:-1])/2.
                ax[cnt].step(binc,h,where='mid',lw=1,c=color,label='%s' % band)
            ti_loc=ax[cnt].set_title(lab)
            ax[cnt].set_xlim([20,26])
            ax[cnt].set_ylim([0,0.9])
            xlab= ax[cnt].set_xlabel('AB mag')
            ylab= ax[cnt].set_ylabel('PDF')
            name='dr%d_ELG_mag_dist_FDR.png' % self.DR
        kwargs= dict(bbox_extra_artists=[ti_loc,xlab,ylab], bbox_inches='tight',dpi=150)
        if self.savefig:
            plt.savefig(name, **kwargs)
            plt.close()
            print('Wrote {}'.format(name))

    def plot_obiwan_mag_dist(self):
        tab= self.get_dr3_deep2()
        keep= self.get_obiwan_cuts(tab)
        # Plot
        fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(9,4))
        plt.subplots_adjust(wspace=0.1,hspace=0)
        for cnt,thecut,lab in zip(range(4),
                        [keep, keep == False],
                        [r'$0.8<z<1.4, [OII] > 0$','Everything else']):
            # Mag data
            mags=dict(r= tab.r_wdust[thecut],
                      g= tab.gr[thecut] + tab.r_wdust[thecut],
                      z= tab.r_wdust[thecut] - tab.rz[thecut]) 
            for band,color in zip(['g','r','z'],['g','r','m']):
                # nans present
                mags[band]= mags[band][ np.isfinite(mags[band]) ]
                # histograms
                h,edges= np.histogram(mags[band],bins=40,normed=True)
                binc= (edges[1:]+edges[:-1])/2.
                ax[cnt].step(binc,h,where='mid',lw=1,c=color,label='%s' % band)
            ti_loc=ax[cnt].set_title(lab)
            ax[cnt].set_xlim([20,26])
            ax[cnt].set_ylim([0,0.9])
            xlab= ax[cnt].set_xlabel('AB mag')
            ylab= ax[cnt].set_ylabel('PDF')
            name='dr%d_ELG_mag_dist_obiwan.png' % self.DR
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

    def plot_kde(self):
        rz,gr,r_nodust,r_wdust,redshift= self.get_elgs_FDR_cuts()
        x= r_wdust['med2hiz_oiibright']
        y= rz['med2hiz_oiibright']
        z= gr['med2hiz_oiibright']
        d4= redshift['med2hiz_oiibright']
        cut= (np.isfinite(x))* (np.isfinite(y))* (np.isfinite(z))
        x,y,z,d4= x[cut],y[cut],z[cut],d4[cut]
        labels=['r wdust','r-z','g-r','redshift']
        kde_obj= KernelOfTruth([x,y,z,d4],labels,\
                               [(20.5,25.),(0,2),(-0.5,1.5),(0.6,1.6)],\
                               bandwidth=0.05,
                               kdefn=self.kdefn,loadkde=self.loadkde)
        xylims=dict(x1=(20.5,25.5),y1=(0,0.8),\
                    x2=xyrange['x_elg'],y2=xyrange['y_elg'],\
                    x3=(0.6,1.6),y3=(0.,1.0))
        #kde_obj.plot_1band_and_color(ndraws=1000,xylims=xylims,prefix='elg_')
        kde_obj.plot_1band_color_and_redshift(ndraws=1000,xylims=xylims,prefix='elg_')
        if self.savekde:
            if os.path.exists(self.kdefn):
                os.remove(self.kdefn)
            kde_obj.save(name=self.kdefn)

    def plot_kde_shapes(self):
        re,n,ba,pa= self.get_acs_matched_deep2()
        pa+= 90. # 0-180 deg
        #cut= (np.isfinite(x))* (np.isfinite(y))* (np.isfinite(z))
        #x,y,z,d4= x[cut],y[cut],z[cut],d4[cut]
        # ba > 0
        labels=['re','n','ba','pa']
        kde_obj= KernelOfTruth([re,n,ba,pa],labels,\
                               [(0.,100.),(0.,10.),(0.2,0.9),(0.,180.)],\
                               bandwidth=0.05,kernel='tophat',\
                               kdefn=self.kde_shapes_fn,loadkde=self.loadkde)
        xylims=dict(x1=(0,100),\
                    x2=(0,10),\
                    x3=(0,1),\
                    x4=(0,180))
        #kde_obj.plot_1band_and_color(ndraws=1000,xylims=xylims,prefix='elg_')
        kde_obj.plot_galaxy_shapes(ndraws=1000,xylims=xylims,name='elg_shapes_kde.png')
        if self.savekde:
            if os.path.exists(self.kde_shapes_fn):
                os.remove(self.kde_shapes_fn)
            kde_obj.save(name=self.kde_shapes_fn)

# Returns re,n measured by tractor given a Tractor Cataluge
def get_tractor_shapes(cat):
    d= {}
    for key in ['re','n']:
        d[key]= np.zeros(len(cat))-1
    # SIMP
    #keep= (cat.type == 'SIMP') * (cat.shapeexp_r > 0.)
    #d['re'][keep]= cat.shapeexp_r[keep]
    #d['n'][keep]= 1.
    # EXP
    keep= (cat.type == 'EXP') * (cat.shapeexp_r > 0.)
    d['re'][keep]= cat.shapeexp_r[keep]
    d['n'][keep]= 1.
    # DEV
    keep= (cat.type == 'DEV') * (cat.shapedev_r > 0.)
    d['re'][keep]= cat.shapedev_r[keep]
    d['n'][keep]= 4.
    return d

def plot_tractor_shapes(cat,prefix=''):
    for name,rng in zip(['tractor_re','tractor_n'],
                        [(0,2),(1,4)]):
        fig,ax= plt.subplots()
        keep= cat.get(name) > 0.
        #h,edges= np.histogram(cat.get(name)[keep],bins=40,normed=True)
        #binc= (edges[1:]+edges[:-1])/2.
        #ax.step(binc,h,where='mid',lw=1,c='b')
        _=ax.hist(cat.get(name)[keep],bins=100,normed=True,range=rng)
        xlab= ax.set_xlabel(name)
        ylab= ax.set_ylabel('PDF')
        savenm= 'tractor_%s_%s.png' % (prefix,name)
        plt.savefig(savenm,bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
        plt.close()
        print('Wrote %s' % savenm)


def plot_tractor_galfit_shapes(cat,prefix=''):
    for name in ['re','n']:
        fig,ax= plt.subplots()
        keep= cat.get('tractor_'+name) > 0.
        ax.scatter(cat.get(name)[keep], cat.get('tractor_'+name)[keep],
                   c='b',marker='o',s=10.,rasterized=True)
        xlab= ax.set_xlabel('galfit_hi_%s' % name)
        ylab= ax.set_ylabel('tractor_%s' % name)
        savenm= 'tractor_galfit_%s_%s.png' % (prefix,name)
        plt.savefig(savenm,bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
        plt.close()
        print('Wrote %s' % savenm)
 
# 2D plots
def plot_indiv_2d(tab,xy_names=None,xy_lims=None, ndraws=1000,prefix=''):
    assert(xy_names)
    for i,_ in enumerate(xy_names):
        xname= xy_names[i][0]
        yname= xy_names[i][1]
        # Plot
        fig,ax= plt.subplots()
        ax.scatter(tab.get(xname),tab.get(yname),
                   c='b',marker='o',s=10.,rasterized=True)
        xlab= ax.set_xlabel(xname)
        ylab= ax.set_ylabel(yname)
        if xy_lims:
            ax.set_xlim(xy_lims[i][0])
            ax.set_ylim(xy_lims[i][1])
        savenm= 'plot_2d_%s_%s_%s.png' % (prefix,xname,yname)
        plt.savefig(savenm,bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
        plt.close()
        print('Wrote %s' % savenm)


class LRG(CommonInit):
    def __init__(self,**kwargs):
        super(LRG, self).__init__(**kwargs)
        self.zlimit= kwargs.get('zlimit',20.46)
        print('LRGs, self.zlimit= ',self.zlimit)
        # KDE params
        self.kdefn= 'lrg-kde.pickle'
        self.kde_shapes_fn= 'lrg-shapes-kde.pickle'

    def get_dr3_cosmos(self):
        # Cosmos
        # http://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos_zphot_mag25_colDescriptions.html
        # http://irsa.ipac.caltech.edu/data/COSMOS/tables/redshift/cosmos_zphot_mag25.README
        if self.DR == 2:
            decals=self.read_fits( os.path.join(self.truth_dir,'decals-dr2-cosmos-zphot.fits.gz') )
            spec=self.read_fits( os.path.join(self.truth_dir,'cosmos-zphot.fits.gz') )
        elif self.DR == 3:
            decals=self.read_fits( os.path.join(self.truth_dir,'dr3-cosmoszphotmatched.fits') )
            spec=self.read_fits( os.path.join(self.truth_dir,'cosmos-zphot-dr3matched.fits') )
        # DECaLS
        CatalogueFuncs().set_mags(decals)
        Z_FLUX = decals.get('decam_flux_nodust')[:,4]
        W1_FLUX = decals.get('wise_flux_nodust')[:,0]
        # Cuts
        # BUG!!
        keep= np.all((Z_FLUX > 10**((22.5-self.zlimit)/2.5),\
                      W1_FLUX > 0.),axis=0)
        keep *= self.imaging_cut(decals)
        decals.cut(keep) 
        spec.cut(keep)
        # Repackage
        tab= fits_table()
        # Cosmos
        tab.set('ra', spec.ra)
        tab.set('dec', spec.dec)
        tab.set('zp_gal', spec.zp_gal)
        tab.set('type_zphotcomos', spec.type)
        tab.set('mod_gal', spec.mod_gal)
        # DR3
        tab.set('g_wdust', decals.get('decam_mag_wdust')[:,1])
        tab.set('r_wdust', decals.get('decam_mag_wdust')[:,2])
        tab.set('z_wdust', decals.get('decam_mag_wdust')[:,4])
        tab.set('w1_wdust', decals.get('wise_mag_wdust')[:,0])
        tab.set('r_nodust', decals.get('decam_mag_nodust')[:,2])
        tab.set('z_nodust', decals.get('decam_mag_nodust')[:,4])
        # DR3 shape info
        for key in ['type','shapeexp_r','shapedev_r']:
            tab.set(key, decals.get(key))
        return tab
                                 
    
    def get_FDR_cuts(self,tab):
        keep={}  
        keep['star']= tab.type == 1
        keep['blue_galaxy']= (tab.type == 0) *\
                             (tab.mod_gal > 8)
        keep['red_galaxy_lowz']= (tab.type == 0) * \
                                 (tab.mod_gal <= 8) *\
                                 (tab.zp_gal <= 0.6)
        keep['red_galaxy_hiz']= (tab.type == 0) * \
                                 (tab.mod_gal <= 8) *\
                                 (tab.zp_gal > 0.6)
        return keep

    def get_obiwan_cuts(self,tab):
        # No redshift limits, just reddish galaxy
        return (tab.type_zphotcomos == 0) * \
               (tab.mod_gal <= 8) 
        return keep

    def fit_kde(self,use_acs=False,
                loadkde=False,savekde=False):
        '''No Targeting cuts on g band, but need to fit it so can insert in grz image'''
        # Load Data
        if use_acs:
            print('Matching acs to dr3,cosmos')
            cosmos= self.get_dr3_cosmos()
            acs= get_acs_six_col()
            imatch,imiss,d2d= Matcher().match_within(cosmos,acs,dist=1./3600)
            cosmos.cut(imatch['ref'])
            acs.cut(imatch['obs'])
            # Remove ra,dec from acs, then merge
            for key in ['ra','dec']:
                acs.delete_column(key)
            tab= merge_tables([cosmos,acs], columns='fillzero')
        else:
            tab= self.get_dr3_cosmos()
        print('dr3_cosmos %d' % len(tab))
        # Add tractor shapes
        dic= get_tractor_shapes(tab) 
        tab.set('tractor_re', dic['re'])
        tab.set('tractor_n', dic['n'])
        # Cuts
        keep= self.get_obiwan_cuts(tab)
        tab.cut(keep)
        print('dr3_cosmos after obiwan cuts: %d' % len(tab))
        # 9D space
        keep= np.ones(len(tab),bool) 
        #use_cols= ['g_wdust','r_wdust','z_wdust','w1_wdust',
        #           'zp_gal']
        for name in tab.get_columns():
            if name == 'type':
                continue
            keep *= (np.isfinite( tab.get(name) ))
        tab.cut(keep)
        print('dr3_cosmos after finite cuts: %d' % len(tab))
        # Redshift > bandwidth
        bandwidth=0.05
        tab.cut(tab.zp_gal - bandwidth >= 0.)
        print('dr3_cosmos after redshift > %f: %d' % (bandwidth,len(tab)))
        # Sanity plot
        plot_tractor_shapes(tab,prefix='LRG_dr3cosmos_expdev')
        print('size %d %d' % (len(tab),len(tab[tab.tractor_re > 0.])))
        #plot_tractor_galfit_shapes(tab,prefix='LRG_dr3cosmosacs')
        tab.cut(tab.tractor_re > 0.)
        xy_names= [('tractor_re','zp_gal'),
                   ('tractor_re','g_wdust'),
                   ('tractor_re','r_wdust'),
                   ('tractor_re','z_wdust')]
        plot_indiv_2d(tab,xy_names=xy_names, ndraws=1000,prefix='LRG')
        # KDE
        names= ['z_wdust','rz','rw1','zp_gal','g_wdust','tractor_re']
                #'re','n','ba','pa']
        fitlims= [(17.,22.),(0,2.5),(-2,5.),(0.,1.6),(17.,29),(0.3,1.5)] #,(0.,10.),(0.2,0.9),(0.,180.)]
        #tab.n, tab.ba, tab.pa
        kde_obj= KernelOfTruth([tab.z_wdust, tab.r_wdust - tab.z_wdust, 
                                tab.r_wdust - tab.w1_wdust, tab.zp_gal, 
                                tab.g_wdust, tab.tractor_re],
                                names,fitlims,\
                           bandwidth=bandwidth,kernel='tophat',\
                           kdefn=self.kdefn,loadkde=self.loadkde)
        #kde_obj.plot_indiv_1d(lims=plotlims, ndraws=1000,prefix='lrg_dr3cosmosacs')
        xy_names= [('rz','rw1'),
                   ('zp_gal','tractor_re'),
                   ('zp_gal','g_wdust'),
                   ('zp_gal','rz'),
                   ('zp_gal','z_wdust'),
                   ('zp_gal','rw1'),
                   ('tractor_re','g_wdust'),
                   ('tractor_re','rz'),
                   ('tractor_re','z_wdust'),
                   ('tractor_re','rw1')]
        xy_lims= [([0,2.5],[-2,5.]),
                  ([0.,1.6],[0.,2.]),  
                  ([0.,1.6],[17,29]),  
                  ([0.,1.6],[0,2.5]),  
                  ([0.,1.6],[17,22]),  
                  ([0.,1.6],[-2,5.]),  
                  ([0.,2.],[17,29]),  
                  ([0.,2.],[0,2.5]),  
                  ([0.,2.],[17,22]),  
                  ([0.,2.],[-2,5]) 
                 ]
        kde_obj.plot_indiv_2d(xy_names,xy_lims=xy_lims, ndraws=10000,prefix='lrg_dr3cosmosacs')
        kde_obj.plot_FDR_using_kde(obj='LRG',ndraws=10000,prefix='dr3cosmos')
        #plotlims= [(17.,22.),(0,2.5),(-2,5.),(0.,1.6),(17.,29),(-0.5,2.)] #,(-2,10.),(-0.2,1.2),(-20,200)]
        #kde_obj.plot_1band_and_color(ndraws=1000,xylims=xylims,prefix='lrg_')
        #kde_obj.plot_1band_color_and_redshift(ndraws=1000,xylims=xylims,prefix='lrg_')
        if self.savekde:
            if os.path.exists(self.kdefn):
                os.remove(self.kdefn)
            kde_obj.save(name=self.kdefn)
 
    def plot_FDR(self):
        tab= self.get_dr3_cosmos()
        keep= self.get_FDR_cuts(tab)
        # Plot
        fig,ax = plt.subplots()
        rgb_cols=get_rgb_cols()
        # Add box
        ts= TSBox(src='LRG')
        xrange,yrange= xyrange['x_lrg'],xyrange['y_lrg']
        ts.add_ts_box(ax, xlim=xrange,ylim=yrange)
        # Data
        for cnt,cut_name,rgb in zip(range(4),
                        ['star','red_galaxy_lowz','red_galaxy_hiz','blue_galaxy'],
                                    rgb_cols):
            rgb= (rgb[0]/255.,rgb[1]/255.,rgb[2]/255.)
            cut= keep[cut_name]
            ax.scatter(tab.rz[cut],tab.rW1[cut],c=[rgb],
                       edgecolors='none',marker='o',s=10.,rasterized=True,label=cut_name)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        xlab=ax.set_xlabel('r-z')
        ylab=ax.set_ylabel('r-W1')
        leg=ax.legend(loc=(0,1.05), ncol=2,prop={'size': 14}, labelspacing=0.2,\
                      markerscale=2,scatterpoints=1)
        #handles,labels = ax.get_legend_handles_labels()
        #index=[0,1,2,3]
        #handles,labels= np.array(handles)[index],np.array(labels)[index]
        #leg=ax.legend(handles,labels,loc=(0,1.05),ncol=2,scatterpoints=1,markerscale=2)
        name='dr%d_FDR_LRG.png' % self.DR
        if self.savefig:
            plt.savefig(name,\
                        bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote {}'.format(name))

    def plot_FDR_multi(self):
        tab= self.get_dr3_cosmos()
        keep= self.get_FDR_cuts(tab)
        # Plot
        fig,ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(16,4))
        plt.subplots_adjust(wspace=0.1,hspace=0)
        rgb_cols=get_rgb_cols()
        for cnt,cut_name,rgb in zip(range(4),   
                       ['star','red_galaxy_lowz','red_galaxy_hiz','blue_galaxy'],
                                    rgb_cols):
            rgb= (rgb[0]/255.,rgb[1]/255.,rgb[2]/255.)
            cut= keep[cut_name]
            ax[cnt].scatter(tab.rz[cut],tab.rW1[cut],c=[rgb],
                            edgecolors='none',marker='o',s=10.,rasterized=True)#,label=key)
            ti=ax[cnt].set_title(cut_name)
            ax[cnt].set_xlim([0,2.5])
            ax[cnt].set_ylim([-2,6])
            xlab=ax[cnt].set_xlabel('r-z')
            # Add box
            ts= TSBox(src='LRG')
            xrange,yrange= xyrange['x_lrg'],xyrange['y_lrg']
            ts.add_ts_box(ax[cnt], xlim=xrange,ylim=yrange)
            ylab=ax[cnt].set_ylabel('r-W1')
        #handles,labels = ax.get_legend_handles_labels()
        #index=[0,1,2,3]
        #handles,labels= np.array(handles)[index],np.array(labels)[index]
        #leg=ax.legend(handles,labels,loc=(0,1.05),ncol=2,scatterpoints=1,markerscale=2)
        name='dr%d_LRG_FDR_multi.png' % self.DR
        if self.savefig:
            plt.savefig(name,\
                        bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote {}'.format(name))

    def plot_obiwan_multi(self):
        tab= self.get_dr3_cosmos()
        keep= self.get_obiwan_cuts(tab)
        # Plot
        fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,4))
        plt.subplots_adjust(wspace=0.1,hspace=0)
        rgb_cols=get_rgb_cols()
        for cnt,thecut,cut_name,rgb in zip(range(2),   
                                  [keep, keep == False],
                                  ['red galaxy','everything else'],
                                  rgb_cols):
            rgb= (rgb[0]/255.,rgb[1]/255.,rgb[2]/255.)
            ax[cnt].scatter(tab.rz[thecut],tab.rW1[thecut],c=[rgb],
                            edgecolors='none',marker='o',s=10.,rasterized=True)#,label=key)
            ti=ax[cnt].set_title(cut_name)
            ax[cnt].set_xlim([0,2.5])
            ax[cnt].set_ylim([-2,6])
            xlab=ax[cnt].set_xlabel('r-z')
            # Add box
            ts= TSBox(src='LRG')
            xrange,yrange= xyrange['x_lrg'],xyrange['y_lrg']
            ts.add_ts_box(ax[cnt], xlim=xrange,ylim=yrange)
            ylab=ax[cnt].set_ylabel('r-W1')
        #handles,labels = ax.get_legend_handles_labels()
        #index=[0,1,2,3]
        #handles,labels= np.array(handles)[index],np.array(labels)[index]
        #leg=ax.legend(handles,labels,loc=(0,1.05),ncol=2,scatterpoints=1,markerscale=2)
        name='dr%d_LRG_obiwan_multi.png' % self.DR
        if self.savefig:
            plt.savefig(name,\
                        bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote {}'.format(name))


    def plot_LRGs_in_ELG_FDR(self):
        data= self.get_lrgs_FDR_cuts()
        fig,ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(16,4))
        plt.subplots_adjust(wspace=0.1,hspace=0)
        rgb_cols=get_rgb_cols()
        keys= ['star','red_galaxy_lowz','red_galaxy_hiz','blue_galaxy']
        for cnt,key,rgb in zip(range(4),keys,rgb_cols):
            rgb= (rgb[0]/255.,rgb[1]/255.,rgb[2]/255.)
            ax[cnt].scatter(data['rz'][key],data['g_wdust'][key]-data['r_wdust'][key],c=[rgb],
                            edgecolors='none',marker='o',s=10.,rasterized=True)#,label=key)
            ti=ax[cnt].set_title(key)
            xrange,yrange= xyrange['x_elg'],xyrange['y_elg']
            ax[cnt].set_xlim(xrange)
            ax[cnt].set_ylim(yrange)
            xlab=ax[cnt].set_xlabel('r-z')
            ylab=ax[cnt].set_ylabel('g-r')
            # Add box
            ts= TSBox(src='ELG')
            ts.add_ts_box(ax[cnt], xlim=xrange,ylim=yrange)
        #handles,labels = ax.get_legend_handles_labels()
        #index=[0,1,2,3]
        #handles,labels= np.array(handles)[index],np.array(labels)[index]
        #leg=ax.legend(handles,labels,loc=(0,1.05),ncol=2,scatterpoints=1,markerscale=2)
        name='dr%d_LRGs_in_ELG_FDR.png' % self.DR
        if self.savefig:
            plt.savefig(name,\
                        bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote {}'.format(name))

    def plot_FDR_mag_dist(self):
        tab= self.get_dr3_cosmos()
        keep= self.get_FDR_cuts(tab)
        fig,ax = plt.subplots(1,4,sharex=True,sharey=True,figsize=(16,4))
        plt.subplots_adjust(wspace=0.1,hspace=0)
        rgb_cols=get_rgb_cols()
        for cnt,cut_name,rgb in zip(range(4),
                    ['star','red_galaxy_lowz','red_galaxy_hiz','blue_galaxy'],
                                    rgb_cols):
            rgb= (rgb[0]/255.,rgb[1]/255.,rgb[2]/255.)
            # Mag data
            cut= keep[cut_name]
            mags=dict(r= tab.r_wdust[cut],
                      g= tab.g_wdust[cut],
                      z= tab.z_wdust[cut]) 
            for band,color in zip(['g','r','z'],['g','r','m']):
                # nans present
                mags[band]= mags[band][ np.isfinite(mags[band]) ]
                # histograms
                h,edges= np.histogram(mags[band],bins=40,normed=True)
                binc= (edges[1:]+edges[:-1])/2.
                ax[cnt].step(binc,h,where='mid',lw=1,c=color,label='%s' % band)
            ti=ax[cnt].set_title(cut_name)
            ax[cnt].set_xlim([16,26])
            ax[cnt].set_ylim([0,0.9])
            xlab=ax[cnt].set_xlabel('AB mags')
            ylab=ax[cnt].set_ylabel('PDF')
        #handles,labels = ax.get_legend_handles_labels()
        #index=[0,1,2,3]
        #handles,labels= np.array(handles)[index],np.array(labels)[index]
        #leg=ax.legend(handles,labels,loc=(0,1.05),ncol=2,scatterpoints=1,markerscale=2)
        name='dr%d_LRG_mag_dist_FDR.png' % self.DR
        if self.savefig:
            plt.savefig(name,\
                        bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote {}'.format(name))

    def plot_obiwan_mag_dist(self):
        tab= self.get_dr3_cosmos()
        keep= self.get_obiwan_cuts(tab)
        fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,4))
        plt.subplots_adjust(wspace=0.1,hspace=0)
        rgb_cols=get_rgb_cols()
        for cnt,thecut,lab,rgb in zip(range(2),
                    [keep, keep == False],
                    ['red galaxy','everything else'],
                                    rgb_cols):
            rgb= (rgb[0]/255.,rgb[1]/255.,rgb[2]/255.)
            # Mag data
            mags=dict(r= tab.r_wdust[thecut],
                      g= tab.g_wdust[thecut],
                      z= tab.z_wdust[thecut]) 
            for band,color in zip(['g','r','z'],['g','r','m']):
                # nans present
                mags[band]= mags[band][ np.isfinite(mags[band]) ]
                # histograms
                h,edges= np.histogram(mags[band],bins=40,normed=True)
                binc= (edges[1:]+edges[:-1])/2.
                ax[cnt].step(binc,h,where='mid',lw=1,c=color,label='%s' % band)
            ti=ax[cnt].set_title(lab)
            ax[cnt].set_xlim([16,26])
            ax[cnt].set_ylim([0,0.9])
            xlab=ax[cnt].set_xlabel('AB mags')
            ylab=ax[cnt].set_ylabel('PDF')
        #handles,labels = ax.get_legend_handles_labels()
        #index=[0,1,2,3]
        #handles,labels= np.array(handles)[index],np.array(labels)[index]
        #leg=ax.legend(handles,labels,loc=(0,1.05),ncol=2,scatterpoints=1,markerscale=2)
        name='dr%d_LRG_mag_dist_obiwan.png' % self.DR
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
        elif self.DR == 3:
            decals = self.read_fits(os.path.join(self.truth_dir,'dr3-vipersw1w4matched.fits'))
            vip = self.read_fits(os.path.join(self.truth_dir,'vipersw1w4-dr3matched.fits'))
        CatalogueFuncs().set_mags(decals)
        Z_FLUX = decals.get('decam_flux_nodust')[:,4]
        W1_FLUX = decals.get('wise_flux_nodust')[:,0]
        index={}
        index['decals']= np.all((Z_FLUX > 10**((22.5-self.zlimit)/2.5),\
                                 W1_FLUX > 0.),axis=0)
        index['decals']*= self.imaging_cut(decals)
        # VIPERS
        # https://arxiv.org/abs/1310.1008
        # https://arxiv.org/abs/1303.2623
        flag= vip.get('zflg').astype(int)
        index['good_z']= np.all((flag >= 2,\
                                 flag <= 9,\
                                 vip.get('zspec') < 9.9),axis=0) 
        # return Mags
        rz,rW1={},{}
        cut= np.all((index['decals'],\
                     index['good_z']),axis=0)
        rz= decals.get('decam_mag_nodust')[:,2][cut] - decals.get('decam_mag_nodust')[:,4][cut]
        rW1= decals.get('decam_mag_nodust')[:,2][cut] - decals.get('wise_mag_nodust')[:,0][cut]
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
        name='dr%d_LRG_vipers.png' % self.DR
        if self.savefig:
            plt.savefig(name,\
                        bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote {}'.format(name))


    def plot(self):
        self.plot_FDR()
        self.plot_FDR_multipanel()
        self.plot_vipers()
        #plot_FDR(self.Xall,self.cuts,src='LRG')
        #color_color_plot(self.Xall,src='LRG',append='cc') #,extra=True)
        #b= self.cuts['lrg')
        #color_color_plot(self.Xall[b,:],src='LRG') #,extra=True)

    def plot_kde(self,loadkde=False,savekde=False):
        '''No Targeting cuts on g band, but need to fit it so can insert in grz image'''
        rz,rW1,r_nodust,r_wdust,z_nodust,z_wdust,g_wdust,redshift= self.get_lrgs_FDR_cuts()
        x= z_wdust['red_galaxy']
        y= rz['red_galaxy']
        z= rW1['red_galaxy']
        d4= redshift['red_galaxy']
        M= g_wdust['red_galaxy']
        cut= (np.isfinite(x))* (np.isfinite(y))* (np.isfinite(z))* (np.isfinite(M))
        # Redshift > 0 given bandwidth
        bandwidth=0.05
        cut*= (d4 - bandwidth >= 0.)
        x,y,z,d4,M= x[cut],y[cut],z[cut],d4[cut],M[cut]
        labels=['z wdust','r-z','r-W1','redshift','g wdust']
        kde_obj= KernelOfTruth([x,y,z,d4,M],labels,\
                           [(17.,22.),(0,2.5),(-2,5.),(0.,1.6),(17.,29)],\
                           bandwidth=bandwidth,kernel='tophat',\
                           kdefn=self.kdefn,loadkde=self.loadkde)
        xylims=dict(x1=(17.,22.),y1=(0,0.7),\
                    x2=xyrange['x_lrg'],y2=xyrange['y_lrg'],\
                    x3=(0.,1.6),y3=(0,1.),\
                    x4=(17.,29),y4=(0,0.7))
        #kde_obj.plot_1band_and_color(ndraws=1000,xylims=xylims,prefix='lrg_')
        kde_obj.plot_1band_color_and_redshift(ndraws=1000,xylims=xylims,prefix='lrg_')
        if self.savekde:
            if os.path.exists(self.kdefn):
                os.remove(self.kdefn)
            kde_obj.save(name=self.kdefn)

    
    def plot_kde_shapes(self):
        re,n,ba,pa= self.get_acs_matched_cosmoszphot()
        pa+= 90. # 0-180 deg
        #cut= (np.isfinite(x))* (np.isfinite(y))* (np.isfinite(z))
        #x,y,z,d4= x[cut],y[cut],z[cut],d4[cut]
        # ba > 0
        labels=['re','n','ba','pa']
        kde_obj= KernelOfTruth([re,n,ba,pa],labels,\
                               [(0.,100.),(0.,10.),(0.2,0.9),(0.,180.)],\
                               bandwidth=0.05,kernel='tophat',\
                               kdefn=self.kde_shapes_fn,loadkde=self.loadkde)
        xylims=dict(x1=(-10,100),\
                    x2=(-2,10),\
                    x3=(-0.2,1.2),\
                    x4=(-20,200))
        #kde_obj.plot_1band_and_color(ndraws=1000,xylims=xylims,prefix='elg_')
        kde_obj.plot_galaxy_shapes(ndraws=10000,xylims=xylims,name='lrg_shapes_kde.png')
        if self.savekde:
            if os.path.exists(self.kde_shapes_fn):
                os.remove(self.kde_shapes_fn)
            kde_obj.save(name=self.kde_shapes_fn)


class STAR(CommonInit):
    def __init__(self,**kwargs):
        super(STAR,self).__init__(**kwargs)
        # KDE params
        self.kdefn= 'star-kde.pickle'
    
    def get_sweepstars(self):
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

    def plot_sweepstars(self): 
        rz,gr= self.get_sweepstars()
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
        if self.savefig:
            plt.savefig(name,\
                        bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote {}'.format(name))
      
    def get_purestars(self):
        # https://desi.lbl.gov/trac/wiki/TargetSelectionWG/TargetSelection#SpectrophotometricStandardStarsFSTD
        if self.DR == 2:
            stars=fits_table(os.path.join(self.truth_dir,'Stars_str82_355_4.DECaLS.dr2.fits'))
            CatalogueFuncs().set_mags(stars)
            stars.cut( self.std_star_cut(stars) )
            return stars
        elif self.DR == 3:
            raise ValueError()
 
    def plot(self):
        self.plot_sweepstars() 
        #plt.plot(self.cat.get('ra'),self.cat.get('dec'))
        #plt.savefig('test.png')
        #plt.close()

    def plot_kde(self,loadkde=False,savekde=False):
        stars= self.get_purestars()
        x=stars.get('decam_mag_wdust')[:,2]
        y=stars.get('decam_mag_nodust')[:,2]-stars.get('decam_mag_nodust')[:,4]
        z=stars.get('decam_mag_nodust')[:,1]-stars.get('decam_mag_nodust')[:,2]
        labels=['r wdust','r-z','g-r']
        kde_obj= KernelOfTruth([x,y,z],labels,\
                           [(15,24),(0,2),(0,1.5)],\
                           bandwidth=0.05,\
                           kdefn=self.kdefn,loadkde=self.loadkde)
        xylims=dict(x1=(15,24),y1=(0,0.3),\
                    x2=(-1,3.5),y2=(-0.5,2))
        kde_obj.plot_1band_and_color(ndraws=1000,xylims=xylims,prefix='star_')
        if self.savekde:
            if os.path.exists(self.kdefn):
                os.remove(self.kdefn)
            kde_obj.save(name=self.kdefn)

class QSO(CommonInit):
    def __init__(self,**kwargs):
        super(QSO,self).__init__(**kwargs)
        self.rlimit= kwargs.get('rlimit',22.7)
        print('QSOs, self.rlimit= ',self.rlimit)
        # KDE params
        self.kdefn= 'qso-kde.pickle'
  
    def get_qsos(self):
        if self.DR == 2:        
            qsos= self.read_fits( os.path.join(self.truth_dir,'AllQSO.DECaLS.dr2.fits') )
            # Add AB mags
            CatalogueFuncs().set_mags(qsos)
            qsos.cut( self.imaging_cut(qsos) )
            # r < 22.7, grz > 17
            GFLUX = qsos.get('decam_flux_nodust')[:,1] 
            RFLUX = qsos.get('decam_flux_nodust')[:,2]
            ZFLUX = qsos.get('decam_flux_nodust')[:,4] 
            GRZFLUX = (GFLUX + 0.8* RFLUX + 0.5* ZFLUX ) / 2.3
            cut= np.all((RFLUX > 10**((22.5-self.rlimit)/2.5),\
                         GRZFLUX < 10**((22.5-17.0)/2.5)),axis=0)
            qsos.cut(cut)
            return qsos
        elif self.DR == 3:
            raise ValueError('Not done yet')
            qsos=self.read_fits( os.path.join(self.truth_dir,'qso-dr3sweepmatched.fits') )
            decals=self.read_fits( os.path.join(self.truth_dir,'dr3-qsosweepmatched.fits') )
            CatalogueFuncs().set_mags(decals)
            decals.set('z',qsos.get('z'))
            decals.cuts( self.imaging_cut(decals) )
            return decals
        #G_FLUX= decals.get('decam_flux')[:,1]/decals.get('decam_mw_transmission')[:,1]
        #R_FLUX= decals.get('decam_flux')[:,2]/decals.get('decam_mw_transmission')[:,2]
        #Z_FLUX= decals.get('decam_flux')[:,4]/decals.get('decam_mw_transmission')[:,4]
        # DECaLS
        #index={}
        #rfaint=22.7
        #grzbright=17.
        #index['decals']= np.all((R_FLUX > 10**((22.5-rfaint)/2.5),\
        #                         G_FLUX < 10**((22.5-grzbright)/2.5),\
        #                         R_FLUX < 10**((22.5-grzbright)/2.5),\
        #                         Z_FLUX < 10**((22.5-grzbright)/2.5),\
        #                         decals.get('brick_primary') == True),axis=0)
        # QSO
        # Return
     
    def plot_FDR(self):
        # Data
        qsos= self.get_qsos()
        star_obj= STAR(DR=self.DR,savefig=False)
        stars= star_obj.get_purestars()
        hiz=2.1
        index={}
        index['hiz']= qsos.get('z') > hiz
        index['loz']= qsos.get('z') <= hiz
        # Plot
        fig,ax = plt.subplots(1,2,figsize=(10,4))
        plt.subplots_adjust(wspace=0.1,hspace=0)
        # Stars
        ax[0].scatter(stars.get('decam_mag_nodust')[:,2]-stars.get('decam_mag_nodust')[:,4],\
                      stars.get('decam_mag_nodust')[:,1]-stars.get('decam_mag_nodust')[:,2],\
                      c='b',edgecolors='none',marker='o',s=10.,rasterized=True, label='stars',alpha=self.alpha)
        W= 0.75*stars.get('wise_mag_nodust')[:,0]+ 0.25*stars.get('wise_mag_nodust')[:,1]
        ax[1].scatter(stars.get('decam_mag_nodust')[:,1]-stars.get('decam_mag_nodust')[:,4],\
                      stars.get('decam_mag_nodust')[:,2]-W,\
                      c='b',edgecolors='none',marker='o',s=10.,rasterized=True, label='stars',alpha=self.alpha)
        # QSOs
        for key,lab,col in zip(['loz','hiz'],['(z < 2.1)','(z > 2.1)'],['magenta','red']):
            i= index[key]
            ax[0].scatter(qsos.get('decam_mag_nodust')[:,2][i]-qsos.get('decam_mag_nodust')[:,4][i],\
                          qsos.get('decam_mag_nodust')[:,1][i]-qsos.get('decam_mag_nodust')[:,2][i],\
                          c=col,edgecolors='none',marker='o',s=10.,rasterized=True, label='qso '+lab,alpha=self.alpha)
            W= 0.75*qsos.get('wise_mag_nodust')[:,0]+ 0.25*qsos.get('wise_mag_nodust')[:,1]
            ax[1].scatter(qsos.get('decam_mag_nodust')[:,1][i]-qsos.get('decam_mag_nodust')[:,4][i],\
                          qsos.get('decam_mag_nodust')[:,2][i]-W[i],\
                          c=col,edgecolors='none',marker='o',s=10.,rasterized=True, label='qso '+lab,alpha=self.alpha)
        
        #for xlim,ylim,x_lab,y_lab in ax[0].set_xlim([-0.5,3.])
        ax[0].set_xlim(xyrange['x1_qso'])
        ax[1].set_xlim(xyrange['x2_qso'])
        ax[0].set_ylim(xyrange['y1_qso'])
        ax[1].set_ylim(xyrange['y2_qso'])
        xlab=ax[0].set_xlabel('r-z')
        xlab=ax[1].set_xlabel('g-z')
        ylab=ax[0].set_ylabel('g-r')
        ylab=ax[1].set_ylabel('r-W')
        leg=ax[0].legend(loc=(0,1.02),scatterpoints=1,ncol=3,markerscale=2)
        ## Add box
        #ts= TSBox(src='LRG')
        #xrange,yrange= xyrange['x_lrg'],xyrange['y_lrg']
        #ts.add_ts_box(ax[cnt], xlim=xrange,ylim=yrange)
        name='dr%d_FDR_QSO.png' % self.DR
        if self.savefig:
            plt.savefig(name,\
                        bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote {}'.format(name))

    def plot_FDR_multipanel(self):
        # Data
        qsos= self.get_qsos()
        star_obj= STAR(DR=self.DR,savefig=False)
        stars= star_obj.get_purestars()
        hiz=2.1
        index={}
        index['hiz']= qsos.get('z') > hiz
        index['loz']= qsos.get('z') <= hiz
        # Plot
        fig,ax = plt.subplots(3,2,figsize=(10,12))
        plt.subplots_adjust(wspace=0.2,hspace=0.1)
        # Stars top panel
        ax[0,0].scatter(stars.get('decam_mag_nodust')[:,2]-stars.get('decam_mag_nodust')[:,4],\
                      stars.get('decam_mag_nodust')[:,1]-stars.get('decam_mag_nodust')[:,2],\
                      c='b',edgecolors='none',marker='o',s=10.,rasterized=True, label='stars',alpha=self.alpha)
        W= 0.75*stars.get('wise_mag_nodust')[:,0]+ 0.25*stars.get('wise_mag_nodust')[:,1]
        ax[0,1].scatter(stars.get('decam_mag_nodust')[:,1]-stars.get('decam_mag_nodust')[:,4],\
                      stars.get('decam_mag_nodust')[:,2]-W,\
                      c='b',edgecolors='none',marker='o',s=10.,rasterized=True, label='stars',alpha=self.alpha)
        # QSOs loz middle, hiz bottom
        for cnt,key,lab,col in zip([1,2],['loz','hiz'],['(z < 2.1)','(z > 2.1)'],['magenta','red']):
            i= index[key]
            ax[cnt,0].scatter(qsos.get('decam_mag_nodust')[:,2][i]-qsos.get('decam_mag_nodust')[:,4][i],\
                          qsos.get('decam_mag_nodust')[:,1][i]-qsos.get('decam_mag_nodust')[:,2][i],\
                          c=col,edgecolors='none',marker='o',s=10.,rasterized=True, label='qso '+lab,alpha=self.alpha)
            W= 0.75*qsos.get('wise_mag_nodust')[:,0]+ 0.25*qsos.get('wise_mag_nodust')[:,1]
            ax[cnt,1].scatter(qsos.get('decam_mag_nodust')[:,1][i]-qsos.get('decam_mag_nodust')[:,4][i],\
                          qsos.get('decam_mag_nodust')[:,2][i]-W[i],\
                          c=col,edgecolors='none',marker='o',s=10.,rasterized=True, label='qso '+lab,alpha=self.alpha)
        
        for cnt in range(3):
            #for xlim,ylim,x_lab,y_lab in ax[0].set_xlim([-0.5,3.])
            ax[cnt,0].set_xlim(xyrange['x1_qso'])
            ax[cnt,1].set_xlim(xyrange['x2_qso'])
            ax[cnt,0].set_ylim(xyrange['y1_qso'])
            ax[cnt,1].set_ylim(xyrange['y2_qso'])
            xlab=ax[cnt,0].set_xlabel('r-z')
            xlab=ax[cnt,1].set_xlabel('g-z')
            ylab=ax[cnt,0].set_ylabel('g-r')
            ylab=ax[cnt,1].set_ylabel('r-W')
        #leg=ax[0,0].legend(loc=(0,1.02),scatterpoints=1,ncol=3,markerscale=2)
        ## Add box
        #ts= TSBox(src='LRG')
        #xrange,yrange= xyrange['x_lrg'],xyrange['y_lrg']
        #ts.add_ts_box(ax[cnt], xlim=xrange,ylim=yrange)
        name='dr%d_FDR_QSO_multi.png' % self.DR
        if self.savefig:
            plt.savefig(name,\
                        bbox_extra_artists=[leg,xlab,ylab], bbox_inches='tight',dpi=150)
            plt.close()
            print('Wrote {}'.format(name))


 
    def plot(self):
        self.plot_FDR()
        self.plot_FDR_multipanel()

    def plot_kde(self,loadkde=False,savekde=False):
        qsos= self.get_qsos()
        x= qsos.get('decam_mag_wdust')[:,2]
        y= qsos.get('decam_mag_wdust')[:,2]-qsos.get('decam_mag_wdust')[:,4]
        z= qsos.get('decam_mag_wdust')[:,1]-qsos.get('decam_mag_wdust')[:,2]
        d4= qsos.z
        hiz=2.1
        cut= (d4 <= hiz)*(d4 >= 0.)*\
             (np.isfinite(x))*(np.isfinite(y))*(np.isfinite(z))
        x,y,z,d4= x[cut],y[cut],z[cut],d4[cut]
        labels=['r wdust','r-z','g-r','redshift']
        kde_obj= KernelOfTruth([x,y,z,d4],labels,\
                           [(15.,24.),(-1.,1.5),(-1.,2.),(0.,hiz)],\
                           bandwidth=0.05,kernel='gaussian',\
                           kdefn=self.kdefn,loadkde=self.loadkde)
        xylims=dict(x1=(15.,24),y1=(0,0.5),\
                    x2=xyrange['x1_qso'],y2=xyrange['y1_qso'],\
                    x3=(0.,hiz+0.2),y3=(0.,1.))
        #kde_obj.plot_1band_and_color(ndraws=1000,xylims=xylims,prefix='qso_')
        kde_obj.plot_1band_color_and_redshift(ndraws=1000,xylims=xylims,prefix='qso_')
        if self.savekde:
            if os.path.exists(self.kdefn):
                os.remove(self.kdefn)
            kde_obj.save(name=self.kdefn)


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
    #gals=GalaxyPrior()
    #gals.plot_all()
    #print "gals.__dict__= ",gals.__dict__
    kwargs=dict(DR=2,savefig=True,alpha=0.25,
                loadkde=False,
                savekde=True)
    #star=STAR(**kwargs)
    #star.plot_kde()
    #star.plot()
    kwargs.update(dict(rlimit=22.7+1.))
    #qso=QSO(**kwargs)
    #qso.plot_kde()
    #qso.plot()
    kwargs.update(dict(DR=3, rlimit=23.4+1.))
    elg= ELG(**kwargs)
    elg.fit_kde(use_acs=False)
    #elg.plot_FDR()
    #elg.plot_FDR_multi()
    #elg.plot_obiwan_multi()
    #elg.plot_FDR_mag_dist()
    #elg.plot_obiwan_mag_dist()
    #elg.plot_LRG_FDR_wELG_data()
    #raise ValueError
    #elg.plot_FDR_multipanel()
    #elg.get_acs_matched_deep2()
    #elg.plot_dr3_acs_deep2()
    #elg.plot_kde_shapes()
    #elg.plot_kde()
    #elg.plot_redshift()
    #elg.cross_validate_redshift()

    #elg.plot_kde()
    #elg.plot()
    kwargs.update(dict(zlimit=20.46+1.))
    lrg= LRG(**kwargs)
    lrg.fit_kde(use_acs=False)
    raise ValueError
    #lrg.plot_dr3_cosmos_acs()
    #lrg.plot_FDR()
    #lrg.plot_FDR_multi()
    #lrg.plot_obiwan_multi()
    lrg.plot_FDR_mag_dist()
    lrg.plot_obiwan_mag_dist()
    lrg.plot_LRG_FDR_mag_dist()
    lrg.plot_LRGs_in_ELG_FDR()
    lrg.plot_FDR()
    lrg.plot_FDR_multipanel()
    raise ValueError
    #lrg.plot_kde_shapes()
    #lrg.plot_kde()
    #lrg.plot()
