import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


def add_scatter(ax,x,y,c='b',m='o',lab='',s=80,drawln=False,alpha=1):
    ax.scatter(x,y, s=s, lw=2.,facecolors='none',edgecolors=c, marker=m,label=lab,alpha=alpha)
    if drawln: ax.plot(x,y, c=c,ls='-')

def draw_unit_sphere(ramin=243.,ramax=246.,dcmin=7.,dcmax=10.,Nran=216000,seed=2015):
    '''# from https://github.com/desihub/imaginglss/master/scripts/imglss-mpi-make-random.py'''
    rng = np.random.RandomState(seed)
    u1,u2= rng.uniform(size=(2, Nran))
    #
    cmin = np.sin(dcmin*np.pi/180)
    cmax = np.sin(dcmax*np.pi/180)
    #
    RA   = ramin + u1*(ramax-ramin)
    DEC  = 90-np.arccos(cmin+u2*(cmax-cmin))*180./np.pi
    return RA,DEC


class QuickRandoms(object):
    '''Draw randomly from unit sphere
    Example:
    qran= QuickRandoms(ramin=243.,ramax=246.,dcmin=7.,dcmax=10.,Nran=216000)
    qran.get_randoms()
    # save and plot
    qran.save_randoms()
    qran.plot(xlim=(244.,244.1),ylim=(8.,8.1)) #,xlim=(244.,244.+10./360),ylim=(8.,8.+10./360.))
    '''
    def __init__(self,ramin=243.,ramax=246.,dcmin=7.,dcmax=10.,Nran=216000):
        self.ramin=ramin
        self.ramax=ramax
        self.dcmin=dcmin
        self.dcmax=dcmax
        self.Nran=Nran

    def get_randoms(self, fn='quick_randoms.pickle'):
        if os.path.exists(fn): 
            ra,dec= self.read_randoms()
        else:
            ra,dec=draw_unit_sphere(ramin=self.ramin,ramax=self.ramax,\
                                    dcmin=self.dcmin,dcmax=self.dcmax,Nran=self.Nran)
        self.ra,self.dec=ra,dec 

    def save_randoms(self,fn='quick_randoms.pickle'):
        if not os.path.exists(fn):
            fout=open(fn, 'w')
            pickle.dump((self.ra,self.dec),fout)
            fout.close() 
            print("Wrote randoms to: %s" % fn)
        else: 
            print("WARNING: %s exists, not overwritting it" % fn)

    def read_randoms(self,fn='quick_randoms.pickle'):
        print("Reading randoms from %s" % fn)
        fobj=open(fn, 'r')
        ra,dec= pickle.load(fobj)
        fobj.close()
        return ra,dec

    def plot(self,xlim=None,ylim=None,text=''):
        fig,ax=plt.subplots()
        add_scatter(ax,self.ra,self.dec,c='b',m='.',alpha=0.5)
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        if xlim is not None and ylim is not None: 
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            text='_xlim%0.5f_%.5f_ylim%.5f_%.5f' % (xlim[0],xlim[1],ylim[0],ylim[1])
        plt.savefig("quick_randoms%s.png" % text)
        plt.close()

class DesiRandoms(object):
    '''Draw randomly from unit sphere & provide 2 masks:
    mask1: inbricks -- indices where ra,dec pts are in LegacySurvey bricks
    mask2: inimages -- union with inbricks and where we have legacy survey imaging data at these ra,dec pts 
    Example:
    ran= DesiRandoms(ramin=243.,ramax=246.,dcmin=7.,dcmax=10.,Nran=216000)
    ran.get_randoms()
    # save randoms if file not exist and plot
    ran.save_randoms()
    ran.plot(xlim=(244.,244.1),ylim=(8.,8.1)) #,xlim=(244.,244.+10./360),ylim=(8.,8.+10./360.))
    '''
    def __init__(self,ramin=243.,ramax=246.,dcmin=7.,dcmax=10.,Nran=216000):
        self.ramin=ramin
        self.ramax=ramax
        self.dcmin=dcmin
        self.dcmax=dcmax
        self.Nran=Nran

    def get_randoms(self,fn='desi_randoms.pickle'):
        if os.path.exists(fn):
            self.ra,self.dec,self.i_inbricks,self.i_inimages= self.read_randoms()
        else: 
            self.ra,self.dec,self.i_inbricks,self.i_inimages= self.make_randoms()

    def save_randoms(self,fn='desi_randoms.pickle'):
        if not os.path.exists(fn):
            fout=open(fn, 'w')
            pickle.dump((self.ra,self.dec,self.i_inbricks,self.i_inimages),fout)
            fout.close() 
            print("Wrote: %s" % fn)
        else:
            print "WARNING: not saving randoms b/c file already exists: %s" % fn

    def make_randoms(self):
        '''Nran -- # of randoms'''
        import imaginglss
        from imaginglss.model import dataproduct
        from imaginglss.model.datarelease import contains
        import h5py
        print "Creating %d Randoms" % self.Nran
        # dr2 cache
        decals = imaginglss.DECALS('/project/projectdirs/desi/users/burleigh/dr3_testdir_for_bb/imaginglss/dr2.conf.py')
        foot= decals.datarelease.create_footprint((self.ramin,self.ramax,self.dcmin,self.dcmax))
        print('Total sq.deg. covered by Bricks= ',foot.area)
        # Sample full ra,dec box
        ra,dec=draw_unit_sphere(ramin=self.ramin,ramax=self.ramax,\
                                dcmin=self.dcmin,dcmax=self.dcmax,Nran=self.Nran)
        #randoms = np.empty(len(ra), dtype=dataproduct.RandomCatalogue)
        #randoms['RA'] = ra
        #randoms['DEC'] = dec
        # Get indices of inbrick points, copied from def filter()
        coord= np.array((ra,dec))
        bid = foot.brickindex.query_internal(coord)
        i_inbricks = contains(foot._covered_brickids, bid)
        i_inbricks = np.where(i_inbricks)[0]
        print('Number Density in bricks= ',len(ra)/foot.area)
        # Union of inbricks and have imaging data, evaluate depths at ra,dec
        coord= coord[:, i_inbricks]
        cat_lim = decals.datarelease.read_depths(coord, 'grz')
        depth= cat_lim['DECAM_DEPTH'] ** -0.5 / cat_lim['DECAM_MW_TRANSMISSION']
        nanmask= np.isnan(depth)
        nanmask=np.all(nanmask[:,[1,2,4]],axis=1) # shape (Nran,)
        i_inimages= i_inbricks[nanmask == False]
        print('ra.size=%d, i_inbricks.size=%d, i_inimages.size=%d' % (ra.size, i_inbricks.size, i_inimages.size))
        # We are not using Yu's randoms dtype=dataproduct.RandomCatalogue
        #randoms['INTRINSIC_NOISELEVEL'][:, :6] = (cat_lim['DECAM_DEPTH'] ** -0.5 / cat_lim['DECAM_MW_TRANSMISSION'])
        #randoms['INTRINSIC_NOISELEVEL'][:, 6:] = 0
        #nanmask = np.isnan(randoms['INTRINSIC_NOISELEVEL'])
        #randoms['INTRINSIC_NOISELEVEL'][nanmask] = np.inf
        print('Total sq.deg. where have imaging data approx.= ',foot.area*(len(ra[i_inimages]))/len(ra))
        print('Number Density for sources where have images= ',len(ra[i_inimages])/foot.area)
        # save ra,dec,mask to file
        #with h5py.File('eboss_randoms.hdf5', 'w') as ff:
        #    ds = ff.create_dataset('_HEADER', shape=(0,))
        #    ds.attrs['FootPrintArea'] = decals.datarelease.footprint.area
        #    ds.attrs['NumberDensity'] = 1.0 * len(randoms) / decals.datarelease.footprint.area
        #    for column in randoms.dtype.names:
        #        ds = ff.create_dataset(column, data=randoms[column])
        #    ds = ff.create_dataset('nanmask', data=nanmask)
        return ra,dec, i_inbricks,i_inimages

    def plot(self,name='desirandoms.png'):
        fig,ax=plt.subplots(1,3,sharey=True,sharex=True,figsize=(15,5))
        add_scatter(ax[0],self.ra, self.dec, c='b',m='o')
        add_scatter(ax[1],self.ra[self.i_inbricks], self.dec[self.i_inbricks], c='b',m='.')
        add_scatter(ax[2],self.ra[self.i_inimages], self.dec[self.i_inimages], c='b',m='.')
        for i,title in zip(range(3),['All','in Bricks','in Images']):
            ti=ax[i].set_title(title)
            xlab=ax[i].set_xlabel('ra')
            ax[i].set_ylim((self.dec.min(),self.dec.max()))
            ax[i].set_xlim((self.ra.min(),self.ra.max()))
        ylab=ax[0].set_ylabel('dec')
        plt.savefig(name, bbox_extra_artists=[ti,xlab,ylab], bbox_inches='tight',dpi=150)
        plt.close()
        print "wrote: %s" % name



class Angular_Correlator(object):
    '''Compute w(theta) from observed ra,dec and random ra,dec
    uses landy szalay estimator: DD - 2DR + RR / RR
    two numerical methods: 1) Yu Feng's kdcount, 2) astroML
    Example:
    ac= Angular_Correlator(gal_ra,gal_dec,ran_ra,ran_dec)
    ac.compute()
    ac.plot()
    '''
    def __init__(self,gal_ra,gal_dec,ran_ra,ran_dec,ncores=1):
        self.gal_ra=gal_ra
        self.gal_dec=gal_dec
        self.ran_ra=ran_ra
        self.ran_dec=ran_dec
        self.ncores=ncores

    def compute(self):
        self.theta,self.w={},{}
        for key in ['astroML','yu']:
            self.theta[key],self.w[key]= self.get_angular_corr(whos=key)
        self.plot()

    def get_angular_corr(self,whos='yu'):
        if whos == 'yu': return self.ac_yu()
        elif whos == 'astroML': return self.ac_astroML()
        else: raise ValueError()

    def ac_astroML(self):
        '''from two_point_angular() in astroML/correlation.py'''
        from astroML.correlation import two_point,ra_dec_to_xyz,angular_dist_to_euclidean_dist
        # 3d project
        data = np.asarray(ra_dec_to_xyz(self.gal_ra, self.gal_dec), order='F').T
        data_R = np.asarray(ra_dec_to_xyz(self.ran_ra, self.ran_dec), order='F').T
        # convert spherical bins to cartesian bins
        bins = 10 ** np.linspace(np.log10(1. / 60.), np.log10(6), 16)
        bins_transform = angular_dist_to_euclidean_dist(bins)
        w= two_point(data, bins_transform, method='landy-szalay',data_R=data_R)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        return bin_centers, w
    
    def ac_yu(self):
        from kdcount import correlate
        from kdcount import sphere
        abin = sphere.AngularBinning(np.logspace(-4, -2.6, 10))
        D = sphere.points(self.gal_ra, self.gal_dec)
        R = sphere.points(self.ran_ra, self.ran_dec) #weights=wt_array
        DD = correlate.paircount(D, D, abin, np=self.ncores)
        DR = correlate.paircount(D, R, abin, np=self.ncores)
        RR = correlate.paircount(R, R, abin, np=self.ncores)
        r = D.norm / R.norm
        w= (DD.sum1 - 2 * r * DR.sum1 + r ** 2 * RR.sum1) / (r ** 2 * RR.sum1)
        return abin.angular_centers,w

    def plot(self,name='wtheta.png'):
        fig,ax=plt.subplots()
        for key,col,mark in zip(['yu','astroML'],['g','b'],['o']*2):
            print "%s: theta,w" % key,self.theta[key],self.w[key]
            add_scatter(ax,self.theta[key], self.w[key], c=col,m=mark,lab=key,alpha=0.5)
        t = np.array([0.01, 10])
        plt.plot(t, 10 * (t / 0.01) ** -0.8, ':k', lw=1)
        ax.legend(loc='upper right',scatterpoints=1)
        xlab=ax.set_xlabel(r'$\theta$ (deg)')
        ylab=ax.set_ylabel(r'$\hat{w}(\theta)$')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.savefig(name, bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
        plt.close()
        print "wrote: %s" % name

def ac_unit_test():
    '''angular correlation func unit test'''
    qran= QuickRandoms(ramin=243.,ramax=246.,dcmin=7.,dcmax=10.,Nran=216000)
    qran.get_randoms()
    # subset
    index= np.all((qran.ra >= 244.,qran.ra <= 244.5,\
                   qran.dec >= 8.,qran.dec <= 8.5),axis=0)
    ra,dec= qran.ra[index],qran.dec[index]
    # use these as Ducks for DesiRandoms
    ran= DesiRandoms()
    ran.ra,ran.dec= ra,dec
    index= np.all((ran.ra >= 244.,ran.ra <= 244.25),axis=0)
    ran.i_inbricks= np.where(index)[0]
    index= np.all((index,ran.dec >= 8.1,ran.dec <= 8.4),axis=0)
    ran.i_inimages= np.where(index)[0]
    ran.plot()
    # wtheta
    ac= Angular_Correlator(ran.ra[ran.i_inimages],ran.dec[ran.i_inimages],ran.ra,ran.dec)
    ac.compute()
    ac.plot() 
    print 'finished unit_test'


if __name__ == '__main__':
    #ac_unit_test()
    #Nran=int(2.4e3*9.) 
    #ran= DesiRandoms(ramin=243.,ramax=246.,dcmin=7.,dcmax=10.,Nran=216000)
    Nran=int(2.4e3*1.e2) 
    ran= DesiRandoms(ramin=120.,ramax=130.,dcmin=20.,dcmax=30.,Nran=Nran)
    ran.get_randoms()
    # save randoms if file not exist and plot
    ran.save_randoms(fn='desi_randoms_qual.pickle')
    ran.plot() 
