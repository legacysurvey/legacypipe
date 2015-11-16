from __future__ import print_function
import numpy as np
import pylab as plt
import fitsio
from legacypipe.common import *
from astrometry.util.util import Tan
from astrometry.util.fits import *
from astrometry.util.resample import *
from astrometry.util.plotutils import *
from wise.forcedphot import unwise_tiles_touching_wcs
from wise.unwise import get_unwise_tractor_image
from legacypipe.desi_common import read_fits_catalog
from tractor.ellipses import EllipseE
from tractor import Tractor, NanoMaggies

# UGH, copy-n-pasted below...
#from decals_web.map.views import _unwise_to_rgb

def wise_cutouts(ra, dec, radius, ps, pixscale=2.75, tractor_base='.',
                 unwise_dir='unwise-coadds'):
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
        primhdr = fitsio.read_header(fn)
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

    for b in B.brickname:
        fn = os.path.join(tractor_base, 'coadd', b[:3], b,
                          'decals-%s-image-r.fits' % b)
        bwcs = Tan(fn)
        try:
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(dwcs, bwcs)
        except ResampleError:
            continue
        if len(Yo) == 0:
            continue
        print('Resampling', len(Yo), 'pixels from', b)
        xl,xh,yl,yh = Xi.min(), Xi.max(), Yi.min(), Yi.max()
        print('python legacypipe/runbrick.py -b %s --zoom %i %i %i %i --outdir cluster --pixpsf --splinesky --pipe --no-early-coadds' %
              (b, xl-5, xh+5, yl-5, yh+5) + ' -P \'pickles/cluster-%(brick)s-%%(stage)s.pickle\'')
        for i,tag in enumerate(tags):
            fn = os.path.join(tractor_base, 'coadd', b[:3], b,
                              'decals-%s-%s.jpg' % (b, tag))
            img = plt.imread(fn)
            img = np.flipud(img)
            coimgs[i][Yo,Xo,:] = img[Yi,Xi,:]

    tt = dict(image='Image', model='Model', resid='Resid')
    for img,tag in zip(coimgs, tags):
        plt.clf()
        dimshow(img, ticks=False)
        plt.title('DECaLS grz %s' % tt[tag])
        ps.savefig()

    # Find unWISE tiles overlapping
    tiles = unwise_tiles_touching_wcs(wcs)
    print('Cut to', len(tiles), 'unWISE tiles')

    # Here we assume the targetwcs is axis-aligned and that the
    # edge midpoints yield the RA,Dec limits (true for TAN).
    r,d = wcs.pixelxy2radec(np.array([1,   W,   W/2, W/2]),
                            np.array([H/2, H/2, 1,   H  ]))
    # the way the roiradec box is used, the min/max order doesn't matter
    roiradec = [r[0], r[1], d[2], d[3]]

    ra,dec = T.ra, T.dec

    T.shapeexp = np.vstack((T.shapeexp_r, T.shapeexp_e1, T.shapeexp_e2)).T
    T.shapedev = np.vstack((T.shapedev_r, T.shapedev_e1, T.shapedev_e2)).T
    srcs = read_fits_catalog(T, ellipseClass=EllipseE)

    wbands = [1,2]
    wanyband = 'w'

    for band in wbands:
        T.wise_flux[:, band-1] *= 10.**(primhdr['WISEAB%i' % band] / 2.5)

    coimgs = [np.zeros((H,W), np.float32) for b in wbands]
    comods = [np.zeros((H,W), np.float32) for b in wbands]
    con    = [np.zeros((H,W), np.uint8) for b in wbands]

    for iband,band in enumerate(wbands):
        print('Photometering WISE band', band)
        wband = 'w%i' % band

        for i,src in enumerate(srcs):
            #print('Source', src, 'brightness', src.getBrightness(), 'params', src.getBrightness().getParams())
            #src.getBrightness().setParams([T.wise_flux[i, band-1]])
            src.setBrightness(NanoMaggies(**{wanyband: T.wise_flux[i, band-1]}))
            # print('Set source brightness:', src.getBrightness())

        # The tiles have some overlap, so for each source, keep the
        # fit in the tile whose center is closest to the source.
        for tile in tiles:
            print('Reading tile', tile.coadd_id)

            tim = get_unwise_tractor_image(unwise_dir, tile.coadd_id, band,
                                           bandname=wanyband,
                                           roiradecbox=roiradec)
            if tim is None:
                print('Actually, no overlap with tile', tile.coadd_id)
                continue
            print('Read image with shape', tim.shape)

            # Select sources in play.
            wisewcs = tim.wcs.wcs
            H,W = tim.shape
            ok,x,y = wisewcs.radec2pixelxy(ra, dec)
            x = (x - 1.).astype(np.float32)
            y = (y - 1.).astype(np.float32)
            margin = 10.
            I = np.flatnonzero((x >= -margin) * (x < W+margin) *
                               (y >= -margin) * (y < H+margin))
            print(len(I), 'within the image + margin')

            subcat = [srcs[i] for i in I]
            tractor = Tractor([tim], subcat)
            mod = tractor.getModelImage(0)

            # plt.clf()
            # dimshow(tim.getImage(), ticks=False)
            # plt.title('WISE %s %s' % (tile.coadd_id, wband))
            # ps.savefig()

            # plt.clf()
            # dimshow(mod, ticks=False)
            # plt.title('WISE %s %s' % (tile.coadd_id, wband))
            # ps.savefig()

            try:
                Yo,Xo,Yi,Xi,nil = resample_with_wcs(wcs, tim.wcs.wcs)
            except ResampleError:
                continue
            if len(Yo) == 0:
                continue
            print('Resampling', len(Yo), 'pixels from WISE', tile.coadd_id,
                  band)

            coimgs[iband][Yo,Xo] += tim.getImage()[Yi,Xi]
            comods[iband][Yo,Xo] += mod[Yi,Xi]
            con   [iband][Yo,Xo] += 1

    for img,mod,n in zip(coimgs, comods, con):
        img /= np.maximum(n, 1)
        mod /= np.maximum(n, 1)

    for band,img,mod in zip(wbands, coimgs, comods):
        lo,hi = np.percentile(img, [25,99])
        plt.clf()
        dimshow(img, vmin=lo, vmax=hi, ticks=False)
        plt.title('WISE W%i Data' % band)
        ps.savefig()

        plt.clf()
        dimshow(mod, vmin=lo, vmax=hi, ticks=False)
        plt.title('WISE W%i Model' % band)
        ps.savefig()

        resid = img - mod
        mx = np.abs(resid).max()
        plt.clf()
        dimshow(resid, vmin=-mx, vmax=mx, ticks=False)
        plt.title('WISE W%i Resid' % band)
        ps.savefig()


    #kwa = dict(mn=-0.1, mx=2., arcsinh = 1.)
    kwa = dict(mn=-0.1, mx=2., arcsinh=None)
    rgb = _unwise_to_rgb(coimgs, **kwa)
    plt.clf()
    dimshow(rgb, ticks=False)
    plt.title('WISE W1/W2 Data')
    ps.savefig()

    rgb = _unwise_to_rgb(comods, **kwa)
    plt.clf()
    dimshow(rgb, ticks=False)
    plt.title('WISE W1/W2 Model')
    ps.savefig()

    kwa = dict(mn=-1, mx=1, arcsinh=None)
    rgb = _unwise_to_rgb([img-mod for img,mod in zip(coimgs,comods)], **kwa)
    plt.clf()
    dimshow(rgb, ticks=False)
    plt.title('WISE W1/W2 Resid')
    ps.savefig()


def _unwise_to_rgb(imgs, bands=[1,2], mn=-1, mx=100, arcsinh=1.):
    import numpy as np
    img = imgs[0]
    H,W = img.shape

    ## FIXME
    w1,w2 = imgs

    rgb = np.zeros((H, W, 3), np.uint8)

    scale1 = 50.
    scale2 = 50.

    #mn,mx = -3.,30.
    #arcsinh = None

    img1 = w1 / scale1
    img2 = w2 / scale2

    print('W1 99th', np.percentile(img1, 99))
    print('W2 99th', np.percentile(img2, 99))

    if arcsinh is not None:
        def nlmap(x):
            return np.arcsinh(x * arcsinh) / np.sqrt(arcsinh)
        #img1 = nlmap(img1)
        #img2 = nlmap(img2)
        mean = (img1 + img2) / 2.
        I = nlmap(mean)
        img1 = img1 / mean * I
        img2 = img2 / mean * I
        mn = nlmap(mn)
        mx = nlmap(mx)
    img1 = (img1 - mn) / (mx - mn)
    img2 = (img2 - mn) / (mx - mn)

    rgb[:,:,2] = (np.clip(img1, 0., 1.) * 255).astype(np.uint8)
    rgb[:,:,0] = (np.clip(img2, 0., 1.) * 255).astype(np.uint8)
    rgb[:,:,1] = rgb[:,:,0]/2 + rgb[:,:,2]/2

    return rgb

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--ra', type=float,  default=329.0358)
    parser.add_argument('-d', '--dec', type=float, default=  1.3909)
    parser.add_argument('-t', '--dir', default='dr2p')

    opt = parser.parse_args()

    #ra,dec = 203.522, 20.232
    #ra,dec = 329.0358,1.3909  # horrible fit
    ra,dec = 244.0424,6.9179

    # arcsec
    radius = 90.

    ps = PlotSequence('cluster')

    plt.figure(figsize=(4,4))
    plt.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)

    wise_cutouts(opt.ra, opt.dec, radius, ps,
                 pixscale=2.75 / 2.,
                 tractor_base=opt.dir)
                 #tractor_base='cluster')
