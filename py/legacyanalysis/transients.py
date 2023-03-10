import pylab as plt
import numpy as np
from astrometry.util.fits import *
from astrometry.libkd.spherematch import match_radec
import fitsio
import os
from collections import Counter
from scipy.ndimage.measurements import label, find_objects
import time

#import sys
#sys.path.append('legacypipe/py')

from legacypipe.survey import LegacySurveyData
from tractor import PointSource, RaDecPos, NanoMaggies, Tractor
import matplotlib.cm

import warnings
warnings.simplefilter("ignore")


def try_one_ccd(X):
    try:
        return one_ccd(X)
    except:
        print('try_one_ccd:', X)
        import traceback
        traceback.print_exc()
    (_,_,i,_) = X
    return i,None

def one_ccd(X):
    (survey, ccd, iccd, nccds) = X
    imgfn = ccd.image_filename.strip()
    #outfn = os.path.join('cosmo/data/legacysurvey/dr9/south/outlier-masks', imgfn).replace('.fits.fz', '-outlier.fits.fz')
    outfn = os.path.join(survey.survey_dir, 'outlier-masks', imgfn).replace('.fits.fz', '-outlier.fits.fz')
    if not os.path.exists(outfn):
        print('Missing:', outfn)
        return iccd,None
    #imgfn = os.path.join('cosmo/staging', imgfn)

    results = fits_table()
    results.image_filename = []
    results.image_hdu = []
    results.ccdname = []
    results.expnum = []
    results.filter = []
    results.mjd_obs = []
    results.npix = []
    results.ra = []
    results.dec = []
    results.chisqperpix = []
    results.fluxsn = []
    results.flux = []
    results.flux_err = []
    results.fracin = []

    #img = fitsio.read(imgfn, ext=ccd.ccdname)
    mask = fitsio.read(outfn, ext=ccd.ccdname)
    L,n = label(mask == 1)
    slcs = find_objects(L)
    im = survey.get_image_object(ccd)
    #wcs = im.get_wcs()

    #print('CCD', imgfn, 'ext', ccd.ccdname)
    print('CCD %i of %i:' % (iccd+1, nccds), im, ':', len(slcs), 'blobs of outlier pixels')
    for i,slc in enumerate(slcs):
        npix = np.sum(L[slc] == (i+1))
        #print('N pix', npix)
        #y0,y1,x0,y1 = slc[0].start, slc[0].stop, slc[1].start, slc[1].stop

        tim = im.get_tractor_image(slc=slc, tiny=5, trim_edges=False)
        if tim is None:
            print('Tiny tim for', im)
            continue
        #print('Read tim', tim.shape)
        #print('Slice shape', img[slc].shape)
        assert(tim.shape == L[slc].shape)
        tim.inverr[L[slc] != (i+1)] = 0.
        if np.all(tim.inverr == 0):
            print('No good pix in', im)
            continue
        th,tw = tim.shape

        imx = np.argmax(tim.getImage())
        ymax,xmax = np.unravel_index(imx, tim.shape)
        # Put a point source there!
        r,d = tim.getWcs().pixelToPosition(xmax, ymax)
        src = PointSource(RaDecPos(r, d), NanoMaggies(**{ccd.filter.strip(): 0.}))
        tr = Tractor([tim], [src])
        tr.freezeParam('images')
        # chi-squared without source
        #chisq = -2. * tr.getLogLikelihood()
        src.brightness.setParams([1.])
        #print('Optimizing source...')
        tr.optimize_loop()
        var = tr.optimize(variance=True, just_variance=True)
        if len(var) == 4:
            print('Failed to get variance for', im)
            continue
        #print('Source:', src)
        #print('Variance:', var)
        mod = tr.getModelImage(0)
        mod[tim.inverr == 0] = 0.
        chisq = np.sum(((tim.getImage() - mod) * tim.inverr)**2)
        #print('Chi-squared improvement:', chisq - chisq_2)
        cpp = chisq / np.sum(tim.inverr > 0)
        #print('Chi-squared per pixel:', cpp)

        for k in ['image_filename', 'image_hdu', 'ccdname', 'expnum', 'filter', 'mjd_obs']:
            results.get(k).append(getattr(ccd, k))
        results.ra.append(src.pos.ra)
        results.dec.append(src.pos.dec)
        results.npix.append(npix)
        results.chisqperpix.append(cpp)
        flux = src.brightness.getParams()[0]
        fluxvar = var[-1]
        fluxsig = np.sqrt(fluxvar)
        results.flux.append(flux)
        results.flux_err.append(fluxsig)
        results.fluxsn.append(flux / fluxsig)
        results.fracin.append(np.sum(mod) / flux)

        if False:
            cmap = matplotlib.cm.viridis
            cmap.set_bad(color='0.5')
            plt.subplot(1,3,1)
            ima = dict(interpolation='nearest', origin='lower', vmin=-2.*tim.sig1, vmax=5.*tim.sig1)
            pix = tim.getImage().copy()
            pix[tim.inverr == 0] = np.nan
            plt.imshow(pix, **ima)
            plt.subplot(1,3,2)
            plt.imshow(mod, **ima)
            plt.subplot(1,3,3)
            plt.imshow((tim.getImage() - mod) * tim.inverr, interpolation='nearest', origin='lower', vmin=-3, vmax=+3)
            plt.show()
        
    results.to_np_arrays()
    return iccd,results


def global_init():
    '''
    Global initialization routine called by mpi4py when each worker is started.
    '''
    import socket
    import os
    from mpi4py import MPI
    print('MPI4PY process starting on', socket.gethostname(), 'pid', os.getpid(),
          'MPI rank', MPI.COMM_WORLD.Get_rank())

def main():
    #survey = LegacySurveyData('cosmo/work/users/dstn/dr9')
    #survey = LegacySurveyData('cosmo/work/legacysurvey/dr9.1.1')
    survey = LegacySurveyData('/global/cfs/cdirs/cosmo/work/users/dstn/dr9.1.1')

    ccds = survey.get_ccds_readonly()
    len(ccds)

    # n = len(ccds)
    # for i,ccd in enumerate(ccds):
    #     if i < 180:
    #         continue
    #     one_ccd((survey, ccd, i, n))
    # import sys
    # sys.exit(0)

    
    #I = np.flatnonzero(ccds.filter == 'r')
    #ccds = ccds[I[:10]]

    from astrometry.util.multiproc import multiproc
    import multiprocessing

    mpi = True
    
    if mpi:
        from legacypipe.mpi_runbrick import MyMPIPool

        # The "initializer" arg is only available in mpi4py master
        pool = MyMPIPool(initializer=global_init)#, initargs=(,))
        u = int(os.environ.get('OMPI_UNIVERSE_SIZE', '0'))
        if u == 0:
            u = int(os.environ.get('MPICH_UNIVERSE_SIZE', '0'))
        if u == 0:
            from mpi4py import MPI
            u = MPI.COMM_WORLD.Get_size()
        print('Booting up MPI pool with', u, 'workers...')
        pool.bootup()
        print('Booted up MPI pool.')
        pool._processes = u
        mp = multiproc(None, pool=pool)

    else:
        mp = multiproc(32)

    n = len(ccds)
    riter = mp.imap_unordered(try_one_ccd, [(survey, ccd, i, n) for i,ccd in enumerate(ccds)])
    
    results = []
    nextwrite = 100
    while True:
        try:
            i,r = riter.next()
        except StopIteration:
            break
        except multiprocessing.TimeoutError:
            continue

        if r is None:
            print('Result', i, ': None')
        else:
            print('Result', i, ':', len(r), 'blobs')

        results.append(r)
        if len(results) >= nextwrite:
            R = merge_tables([r for r in results if r is not None])
            R.writeto('transients-interim.fits')
            print('Wrote', len(R), 'interim results')
            nextwrite *= 2

        #time.sleep(1)
    #results = mp.map(one_ccd, 
    results = merge_tables([r for r in results if r is not None])
    results.writeto('transients.fits')

    if mpi:
        print('Shutting down MPI pool...')
        pool.shutdown()
        print('Shut down MPI pool')
    
if __name__ == '__main__':
    main()
