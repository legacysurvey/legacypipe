import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imaginglss
from imaginglss.model import dataproduct
import numpy as np
import h5py
import os
import pickle

# Copied from 
# https://github.com/desihub/imaginglss/master/scripts/imglss-mpi-make-random.py
def fill_random(footprint, Nran, rng):
    """
    Generate uniformly distributed points within the boundary that lie in
    bricks.  We generate in the ra/dec area, then remove points not in any
    bricks.  This hugely increases the memory efficiency for footprints,
    like DR1, where the survey covers several disjoint patches of the sky.

    """


    coord = np.empty((2, Nran))

    ramin,ramax,dcmin,dcmax,area = footprint.range

    start = 0
    while start != Nran:
        # prepare for a parallel section.
        chunksize = 1024 * 512
        u1,u2= rng.uniform(size=(2, 1024 * 1024) )

        #
        cmin = np.sin(dcmin*np.pi/180)
        cmax = np.sin(dcmax*np.pi/180)
        #
        RA   = ramin + u1*(ramax-ramin)
        DEC  = 90-np.arccos(cmin+u2*(cmax-cmin))*180./np.pi
        # Filter out those not in any bricks: only very few points remain 
        coord1 = footprint.filter((RA, DEC))

        # Are we full?
        coord1 = coord1[:, :min(len(coord1.T), Nran - start)]
        sl = slice(start, start + len(coord1.T))
        coord[:, sl] = coord1
        start = start + len(coord1.T)
    
    return coord

def make_randoms(Nran=100):
    '''Nran -- # of randoms'''
    print "Creating %d Randoms" % Nran
    # dr2 cache
    decals = imaginglss.DECALS('/project/projectdirs/desi/users/burleigh/dr3_testdir_for_bb/imaginglss/dr2.conf.py')
    eBOSS= decals.datarelease.create_footprint((180.,210.,0.,60.))
    print('Total sq.deg. covered by Bricks= ',eBOSS.area)
    rng = np.random.RandomState(2015)  
    # randoms within eBOSS footprint & in a brick
    coord = fill_random(eBOSS, Nran, rng)
    randoms = np.empty(len(coord[0]), dtype=dataproduct.RandomCatalogue)
    randoms['RA'] = coord[0]
    randoms['DEC'] = coord[1]
    print('Number Density in bricks= ',len(randoms['RA'])/eBOSS.area)
    # keep if have imaging data there
    # set randoms['INTRINSIC_NOISELEVEL'] to np.inf where no imaging data exists
    cat_lim = decals.datarelease.read_depths(coord, 'grz')
    randoms['INTRINSIC_NOISELEVEL'][:, :6] = (cat_lim['DECAM_DEPTH'] ** -0.5 / cat_lim['DECAM_MW_TRANSMISSION'])
    randoms['INTRINSIC_NOISELEVEL'][:, 6:] = 0 # shape (Nran,10)
    nanmask = np.isnan(randoms['INTRINSIC_NOISELEVEL'])
    randoms['INTRINSIC_NOISELEVEL'][nanmask] = np.inf
    #randoms['INTRINSIC_NOISELEVEL'][nanmask] = np.inf
    nanmask=np.all(nanmask[:,[1,2,4]],axis=1) # shape (Nran,)
    print('Total sq.deg. where have imaging data approx.= ',eBOSS.area*(len(randoms['RA'][~nanmask]))/len(randoms['RA']))
    print('Number Density for sources where have images= ',len(randoms['RA'][~nanmask])/eBOSS.area)
    # save ra,dec,mask to file
    #with h5py.File('eboss_randoms.hdf5', 'w') as ff:
    #    ds = ff.create_dataset('_HEADER', shape=(0,))
    #    ds.attrs['FootPrintArea'] = decals.datarelease.footprint.area
    #    ds.attrs['NumberDensity'] = 1.0 * len(randoms) / decals.datarelease.footprint.area
    #    for column in randoms.dtype.names:
    #        ds = ff.create_dataset(column, data=randoms[column])
    #    ds = ff.create_dataset('nanmask', data=nanmask)
    #print("Wrote randoms to: %s" % 'eboss_randoms.hdf5')
    fout=open('eboss_randoms.pickle', 'w')
    pickle.dump((np.array(randoms['RA']),np.array(randoms['DEC']),nanmask),fout)
    fout.close() 
    print("Wrote randoms to: %s" % 'eboss_randoms.pickle')
    #return randoms

def get_randoms(Nran=100):
    if not os.path.exists('eboss_randoms.pickle'): #hdf5'):
        print('randoms file does not exitst yet: %s' % 'eboss_randoms.pickle') #hdf5')
        make_randoms(Nran=Nran)
    else:
        print('using existing randoms file: %s' % 'eboss_randoms.pickle') #hdf5')
    #return h5py.File('eboss_randoms.hdf5')
    fin=open('eboss_randoms.pickle', 'r')
    ra,dec,nanmask= pickle.load(fin)
    fin.close() 
    d= dict(ra=ra,dec=dec,nanmask=nanmask)
    return d

def plot_randoms(d,have_data=True):
    if have_data: keep = ~d['nanmask']
    else: keep = np.ones(len(d['ra'])).astype(bool)
    plt.scatter(d['ra'][keep],d['dec'][keep],c='b',alpha=0.5)
    plt.title("Cut to have_data: %r" % have_data)
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.savefig("randoms_havedata_%r.png" % have_data,dpi=150)
    plt.close()


d= get_randoms(Nran=10)
print('read in randoms')
plot_randoms(d,have_data=False)
plot_randoms(d,have_data=True)
print('done')

