from __future__ import print_function

import sys

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np
import fitsio
from astrometry.util.plotutils import *

def main():
    #img = fitsio.read('347736-S5.fits')
    img = fitsio.read('392772-N29.fits')
    print(img.shape)
    img = img.T.copy()
    mm = np.median(img)
    img -= mm

    ps = PlotSequence('sky')
    lo,hi = np.percentile(img, [20,80])
    ima = dict(vmin=lo, vmax=hi, interpolation='nearest', origin='lower',
               cmap='gray')
    plt.clf()
    plt.imshow(img, **ima)
    ps.savefig()

    # PAD
    padimg = np.zeros((2048, 4096))
    padimg[1:-1, 1:-1] = img
    img = padimg
    
    from tractor.splinesky import SplineSky
    from scipy.ndimage.morphology import binary_dilation
    from astrometry.util.util import median_smooth

    # # Estimate per-pixel noise via Blanton's 5-pixel MAD
    slice1 = (slice(0,-5,10),slice(0,-5,10))
    slice2 = (slice(5,None,10),slice(5,None,10))
    mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
    sig1 = 1.4826 * mad / np.sqrt(2.)
    print('sig1 estimate:', sig1)

    
    mask = np.zeros(img.shape, bool)
    mask[binary_dilation(img > 5*sig1, iterations=5)] = True

    for mm in [None, mask]:

        notmm = None
        if mm is not None:
            notmm = np.logical_not(mm)
        
        sky = SplineSky.BlantonMethod(img, notmm, 512)

        print('recovered grid?', sky.spl(sky.xgrid, sky.ygrid).T.astype(np.float32))
        print('vs grid', sky.gridvals)
        
        skyfn = 'sky-%s.fits' % (mm is not None and 'mask' or 'nomask')
        sky.write_fits(skyfn)

        from tractor.utils import get_class_from_name
        
        print('Reading sky model from', skyfn)
        hdr = fitsio.read_header(skyfn)
        skyclass = hdr['SKY']
        clazz = get_class_from_name(skyclass)

        if getattr(clazz, 'from_fits'):
            fromfits = getattr(clazz, 'from_fits')
            skyobj = fromfits(skyfn, hdr)
        else:
            fromfits = getattr(clazz, 'fromFitsHeader')
            skyobj = fromfits(hdr, prefix='SKY_')
        sky2 = skyobj
        print('sky2', sky2)
        sky2.write_fits(skyfn.replace('sky', 'sky2'))
        
        mod = np.zeros_like(img)
        sky.addTo(mod)

        plt.clf()
        plt.imshow(mod, **ima)
        plt.title('Blanton method')
        ps.savefig()

        plt.clf()
        plt.imshow(img - mod, **ima)
        plt.title('Blanton method (subtracted)')
        ps.savefig()
    
    
        grid = 512
        #grid = 256
        img = img.astype(np.float32)
        med = np.zeros_like(img)
        median_smooth(img, mm, grid/2, med)

        plt.clf()
        plt.imshow(med, **ima)
        plt.title('dmedsmooth')
        ps.savefig()

        plt.clf()
        plt.imshow(img - med, **ima)
        plt.title('dmedsmooth (subtracted)')
        ps.savefig()

    sys.exit(0)
    
    med2 = np.zeros_like(img)
    mask = np.zeros(img.shape, bool)
    mask[binary_dilation(img > 5*sig1, iterations=5)] = True
    median_smooth(img, mask, grid/2, med2)

    # UN-PAD
    img = img[1:-1, 1:-1]
    med = med[1:-1, 1:-1]
    med2 = med2[1:-1, 1:-1]
    mask = mask[1:-1, 1:-1]

    sub = img - med
    
    plt.clf()
    plt.imshow(sub, **ima)
    ps.savefig()

    plt.clf()
    plt.imshow(img - med2, **ima)
    ps.savefig()

    plt.clf()
    plt.imshow(img * (1-mask), **ima)
    ps.savefig()
    
    plt.clf()
    plt.imshow(med2, **ima)
    ps.savefig()

    
    lo2,hi2 = np.percentile(img, [5,95])

    ha = dict(bins=100, range=(lo2,hi2), log=True,
             histtype='step')
    plt.clf()
    n1,b,p = plt.hist(img.ravel(), color='r', **ha)
    n2,b,p = plt.hist(sub.ravel(), color='b', **ha)
    mx = max(max(n1), max(n2))
    plt.ylim(mx*0.1, mx)
    ps.savefig()

    ha = dict(bins=100, range=(lo2,hi2), histtype='step')
    plt.clf()
    n1,b,p = plt.hist(img.ravel(), color='r', **ha)
    n2,b,p = plt.hist(sub.ravel(), color='b', **ha)

    n3,b,p = plt.hist((img - sub).ravel(), color='m', **ha)

    #mx = max(max(n1), max(n2))
    #plt.ylim(0, mx)
    ps.savefig()


if __name__ == '__main__':
    main()
    
