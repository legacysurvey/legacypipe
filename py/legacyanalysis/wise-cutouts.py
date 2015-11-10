from __future__ import print_function
import numpy as np
import pylab as plt
from legacypipe.common import *
from astrometry.util.util import Tan
from astrometry.util.fits import *
from astrometry.util.resample import *
from astrometry.util.plotutils import *

def wise_cutouts(ra, dec, radius, ps, pixscale=2.75, tractor_base='.'):
    '''
    radius in arcsec.
    pixscale: WISE pixel scale in arcsec/pixel;
        make this smaller than 2.75 to oversample.
    '''

    npix = int(np.ceil(radius / pixscale))
    print('Image size:', npix)
    W = H = npix
    pix = pixscale / 3600.
    wcs = Tan(ra, dec, (W+1)/2., (H+1)/2., -pix, 0., 0., pix,float(W),float(H))
    
    # Find DECaLS bricks overlapping
    decals = Decals()
    B = bricks_touching_wcs(wcs, decals=decals)
    print('Found', len(B), 'bricks overlapping')

    TT = []
    for b in B.brickname:
        fn = os.path.join(tractor_base, 'tractor', b[:3],
                          'tractor-%s.fits' % b)
        T = fits_table(fn)
        print('Read', len(T), 'from', b)
        TT.append(T)
    T = merge_tables(TT)
    print('Total of', len(T), 'sources')
    T.cut(T.brick_primary)
    print(len(T), 'primary')
    margin = 20
    ok,xx,yy = wcs.radec2pixelxy(T.ra, T.dec)
    I = np.flatnonzero((xx > -margin) * (yy > -margin) *
                       (xx < W+margin) * (yy < H+margin))
    T.cut(I)
    print(len(T), 'within ROI')

    # Pull out DECaLS coadds (image, model, resid).
    dwcs = wcs.scale(2. * pixscale / 0.262)
    dh,dw = dwcs.shape
    print('DECaLS resampled shape:', dh,dw)
    tags = ['image', 'model', 'resid']
    coimgs = [np.zeros((dh,dw,3), np.uint8) for t in tags]
    coimgs2 = [np.zeros((dh,dw,3), np.uint8) for t in tags]

    for b in B.brickname:
        fn = os.path.join(tractor_base, 'coadd', b[:3], b,
                          'decals-%s-image-r.fits' % b)
        bwcs = Tan(fn)
        # try:
        #     Yo,Xo,Yi,Xi,nil = resample_with_wcs(dwcs, bwcs)
        # except ResampleError:
        #     continue
        # if len(Yo) == 0:
        #     continue
        # print('Resampling', len(Yo), 'pixels from', b)
        # for i,tag in enumerate(tags):
        #     fn = os.path.join(tractor_base, 'coadd', b[:3], b,
        #                       'decals-%s-%s.jpg' % (b, tag))
        #     img = plt.imread(fn)
        #     img = np.flipud(img)
        #     coimgs[i][Yo,Xo,:] = img[Yi,Xi,:]
        ims = []
        rgbims = []
        for i,tag in enumerate(tags):
            fn = os.path.join(tractor_base, 'coadd', b[:3], b,
                              'decals-%s-%s.jpg' % (b, tag))
            img = plt.imread(fn)
            img = np.flipud(img)
            ims.append(img)
            rgbims.append(img[:,:,0].astype(np.float32))
            rgbims.append(img[:,:,1].astype(np.float32))
            rgbims.append(img[:,:,2].astype(np.float32))
            
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(dwcs, bwcs, rgbims)
        except ResampleError:
            continue
        if len(Yo) == 0:
            continue
        print('Resampling', len(Yo), 'pixels from', b)
        for i,img in enumerate(ims):
            coimgs[i][Yo,Xo,:] = img[Yi,Xi,:]

        k = 0
        for i in range(len(coimgs2)):
            for j in range(3):
                coimgs2[i][Yo,Xo,j] = np.clip(np.round(rims[k]), 0, 255).astype(np.uint8)
                k += 1
                
    for img,tag in zip(coimgs, tags):
        plt.clf()
        dimshow(img, ticks=False)
        ps.savefig()

    for img,tag in zip(coimgs2, tags):
        plt.clf()
        dimshow(img, ticks=False)
        ps.savefig()
        
    
    # Find unWISE tiles overlapping

    # Cut out unWISE images

    # Render unWISE models & residuals

if __name__ == '__main__':

    ra,dec = 203.522, 20.232
    # arcsec
    radius = 90.

    ps = PlotSequence('cluster')
    
    wise_cutouts(ra, dec, radius, ps,
                 pixscale=2.75 / 2.,
                 tractor_base='cluster')
    

    
