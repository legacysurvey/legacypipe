import argparse
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import os
import pickle

from astroML.decorators import pickle_results
from astroML.correlation import two_point,ra_dec_to_xyz,angular_dist_to_euclidean_dist

from theValidator.catalogues import CatalogueFuncs,Matcher

from astrometry.util.fits import fits_table,merge_tables

# Identical to astroML.correlation.bootstrap_two_point_angular()
# but for data_R specified 
def new_bootstrap_two_point_angular(ra_G,dec_G, bins, method='standard',\
                                    ra_R=None,dec_R=None, Nbootstraps=10,\
                                    random_state=None):
    """Angular two-point correlation function

    A separate function is needed because angular distances are not
    euclidean, and random sampling needs to take into account the
    spherical volume element.

    Parameters
    ----------
    ra_G, dec_G : array_like
        Observed galaxy ra,dec
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1
    method : string
        "standard" or "landy-szalay".
    ra_R, dec_R : array_like
        Random catlogue of ra,dec
    Nbootstraps : int
        number of bootstrap resamples

    Returns
    -------
    corr : ndarray
        the estimate of the correlation function within each bin
        shape = Nbins
    dcorr : ndarray
        error estimate on dcorr (sample standard deviation of
        bootstrap resamples)
    bootstraps : ndarray
        The full sample of bootstraps used to compute corr and dcorr
    """
    data = np.asarray(ra_dec_to_xyz(ra_G, dec_G), order='F').T
    n_samples, n_features = data.shape
    if random_state is None:
        rng= np.random.RandomState()

    if ra_R is None or dec_R is None:
        print('Creating random sample')
        data_R = data.copy()
        for i in range(n_features - 1):
            rng.shuffle(data_R[:, i])
    else:
        data_R = np.asarray(ra_dec_to_xyz(ra_R, dec_R), order='F').T
        if (data_R.ndim != 2) or (data_R.shape[-1] != n_features):
            raise ValueError('data_R must have same n_features as data')
    bins = np.asarray(bins)
    Nbins = len(bins) - 1

    if method not in ['standard', 'landy-szalay']:
        raise ValueError("method must be 'standard' or 'landy-szalay'")

    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim != 2:
        raise ValueError("data should be 1D or 2D")

    # convert spherical bins to cartesian bins
    bins_transform = angular_dist_to_euclidean_dist(bins)

    bootstraps = []

    # Bootstrap sample data_R, hold data fixed 
    #R_samples, R_features = data_R.shape
    #for i in range(Nbootstraps):
    #    # Sample with replacement
    #    #inds= rng.randint(R_samples, size=R_samples) #np.random.randint
    #    ind = rng.randint(0, R_samples, R_samples)
    #    new_R= data_R[ind,:]
    #    print('WARNING: look in code, is new_R shape correct?')

    #    bootstraps.append(two_point(data, bins_transform, method=method,
    #                                data_R=new_R, random_state=rng))
    # Now, bootstrap sample data, hold data_R 
    for i in range(Nbootstraps):
        print('Computing bootstrap: %d' % (i+1,))
        ind = rng.randint(0, n_samples, n_samples)
        new_G= data[ind,:]
        bootstraps.append(two_point(new_G, bins_transform, method=method,
                                    data_R=data_R, random_state=rng))

    bootstraps = np.asarray(bootstraps)
    corr = np.mean(bootstraps, 0)
    corr_err = np.std(bootstraps, 0, ddof=1)

    return corr, corr_err, bootstraps



def compute_results(data,data_R,\
					Nbins=16, Nbootstraps=10,  method='landy-szalay', rseed=0,\
                    fn='correlation_functions.pkl'):
    if os.path.exists(fn):
        print('Loading correlation func from: %s' % fn)
        fin=open(fn,'r')
        results=pickle.load(fin)
        fin.close()
    else: 
        print('Computing correlation func, saving to: %s' % fn)
        np.random.seed(rseed)
        bins = 10 ** np.linspace(np.log10(1. / 60.), np.log10(6), 16)

        results = [bins]
        results += new_bootstrap_two_point_angular(\
                            data.ra,data.dec,\
                            bins=bins,method=method,\
                            ra_R=data_R.ra,dec_R=data_R.dec,\
                            Nbootstraps=Nbootstraps)
        # Save results
        fout=open(fn,'w')
        pickle.dump(results,fout)
        fout.close()
        print('Wrote %s' % fn)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a legacypipe-compatible CCDs file from a set of reduced imaging.')
    parser.add_argument('--type',choices=['elg','lrg','qso','star'],type=str,action='store',required=True)
    parser.add_argument('--brick_list',type=str,action='store',help='run on a single bircks data',default=False,required=True)
    parser.add_argument('--debug',action='store_true',help='run on a single bircks data',default=False,required=False)
    parser.add_argument('--nboots',type=int,action='store',default=10,help='number of bootstraps',required=False)
    args = parser.parse_args()

    # IDEA: recovered ra,dec is the imaging systematic correlation function, we want to remove that
    dr3dir='/global/project/projectdirs/cosmo/data/legacysurvey/dr3.0'
    basedir='/scratch2/scratchdirs/kaylanb/dr3-bootes'
    big_geom=fits_table()
    big_recover=fits_table()
    if args.debug:
        bricks=['2191p337']
        radec_fn='junk.pkl'
    else:
        bricks=np.loadtxt(args.brick_list,dtype=str)
        radec_fn='radec-%s.pkl' % args.brick_list
    # If pkl not exist, compute it
    if os.path.exists(radec_fn):
        print('Loading %s' % radec_fn)
        fin=open(radec_fn,'r')
        (big_geom,big_recover)=pickle.load(fin)
        fin.close()
    else: 
        print('Creating %s' % radec_fn)
        for nth,brick in enumerate(bricks):
            print('Stacking brick %d/%d' % (nth+1,len(bricks)))
            # Get lists of relevant files
            bri=brick[:3]
            recovered_fns=glob(os.path.join(basedir,'%s/%s/%s/*/tractor-*.fits' %\
                                    (args.type,bri,brick)))
            if len(recovered_fns) == 0:
                print('NO TRACTOR CAT, SKIPPING BRICK: %s' % brick)
                continue
            simcat_fns= np.char.replace(recovered_fns, 'tractor-','simcat-')
            dr3_fn= os.path.join(dr3dir,'tractor/%s/tractor-%s.fits' % \
                                    (bri,brick))
            # Combine the simcat and recovered files
            print('Stacking %d simcats' % (len(simcat_fns)))
            rand_geom= CatalogueFuncs().stack(simcat_fns,textfile=False)
            print('Stacking %d recovered tractors' % len(recovered_fns))
            rand_recover= CatalogueFuncs().stack(recovered_fns,textfile=False)
            dr3= fits_table(dr3_fn)
            # Cut to ra,dec cols to reduce file size
            small_geom=fits_table()
            small_recover=fits_table()
            for key in ['ra','dec']:
                small_geom.set(key, rand_geom.get(key))
                small_recover.set(key, rand_recover.get(key))
            # Remove DR3 sources from rand_geom_recover
            imatch,imiss,d2d= Matcher().match_within(dr3,small_recover,dist=1./3600)
            keep= np.arange(len(small_recover))
            keep= np.delete(keep,imatch['obs'] )
            small_recover.cut(keep)
            # Add small to big
            big_geom= merge_tables([small_geom,big_geom], columns='fillzero')
            big_recover= merge_tables([small_recover,big_recover], columns='fillzero')
        # Save results
        fout=open(radec_fn,'w')
        pickle.dump((big_geom,big_recover),fout)
        fout.close()
        print('Wrote %s' % radec_fn)

    # Compute corr funcs
    # data=big_recover, data_R=big_geom
    print('Computing corr func')
    (bins, corr, corr_err, bootstraps)= compute_results(big_recover,big_geom,\
                                                        Nbootstraps=args.nboots,\
                                        fn='corfunc-%d-bootes-%s.pickle' % (args.nboots,args.brick_list))
    print('Plotting')
    corr= [corr]
    corr_err=[corr_err]
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Plot the results
    # Ra,dec distribution
    ra1,ra2,dec1,dec2=None,None,None,None
    ra1,dec1= 216.5,33.5
    ra2,dec2=ra1+1./36,dec1+1./36
    if ra1 is not None:
        big_geom.cut( (big_geom.ra >= ra1)*(big_geom.ra <= ra2)*\
                      (big_geom.dec >= dec1)*(big_geom.dec <= dec2) )
        big_recover.cut( (big_recover.ra >= ra1)*(big_recover.ra <= ra2)*\
                      (big_recover.dec >= dec1)*(big_recover.dec <= dec2) )
    fig,ax= plt.subplots(1,2,figsize=(10,5))
    plt.subplots_adjust(wspace=0.1)
    for cnt,cat,name in zip(range(2),[big_geom,big_recover],['Random','Recovered']):
        ax[cnt].scatter(cat.get('ra'), cat.get('dec'), \
                        edgecolor='b',c='none',lw=1.)
        xlab=ax[cnt].set_xlabel('RA')
        ti=ax[cnt].set_title(name)
    ylab=ax[0].set_ylabel('DEC')
    for cnt in range(2):
        ax[cnt].set_xlim(ax[0].get_xlim())
    fn="radec-%s-%d-%s.png" % (args.type,args.nboots,args.brick_list)
    if ra1 is not None:
        fn=fn.replace('.png','-zoom.png'
    plt.savefig('%s' % fn,bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()
    raise ValueError
    labels = '%s: N=%i, R=%i' % (args.type,len(big_recover),len(big_geom))

    fig = plt.figure(figsize=(5, 2.5))
    fig.subplots_adjust(bottom=0.2, top=0.9,
                        left=0.13, right=0.95)

    for i in range(1):
        ax = fig.add_subplot(121 + i, xscale='log', yscale='log')

        ax.errorbar(bin_centers, corr[i], corr_err[i],
                    fmt='.k', ecolor='gray', lw=1)

        t = np.array([0.01, 10])
        ax.plot(t, 10 * (t / 0.01) ** -0.8, ':k', linewidth=1)

        ax.text(0.95, 0.95, labels[i],
                ha='right', va='top', transform=ax.transAxes)
        xlab=ax.set_xlabel(r'$\theta\ (deg)$')
        if i == 0:
            ylab=ax.set_ylabel(r'$\hat{w}(\theta)$')
    fn="corfun-%s-%d-%s.png" % (args.type,args.nboots,args.brick_list)
    plt.savefig(fn,bbox_extra_artists=[xlab,ylab], bbox_inches='tight',dpi=150)
    plt.close()
    print('Wrote %s' % fn)
