from __future__ import print_function
import numpy as np
import pylab as plt
import fitsio
from legacypipe.survey import *
from astrometry.util.util import Tan
from astrometry.util.fits import *
from astrometry.util.resample import *
from astrometry.util.plotutils import *
from wise.forcedphot import unwise_tiles_touching_wcs
from wise.unwise import get_unwise_tractor_image
from legacypipe.catalog import read_fits_catalog
from tractor.ellipses import EllipseE
from tractor import Tractor, NanoMaggies, LinearPhotoCal, ConstantFitsWcs, ConstantSky, NCircularGaussianPSF, NanoMaggies, Image

# UGH, copy-n-pasted below...
#from decals_web.map.views import _unwise_to_rgb

def wise_cutouts(ra, dec, radius, ps, pixscale=2.75, survey_dir=None,
                 unwise_dir=None):
    '''
    radius in arcsec.
    pixscale: WISE pixel scale in arcsec/pixel;
        make this smaller than 2.75 to oversample.
    '''

    if unwise_dir is None:
        unwise_dir = os.environ.get('UNWISE_COADDS_DIR')

    npix = int(np.ceil(radius / pixscale))
    print('Image size:', npix)
    W = H = npix
    pix = pixscale / 3600.
    wcs = Tan(ra, dec, (W+1)/2., (H+1)/2., -pix, 0., 0., pix,float(W),float(H))
    # Find DECaLS bricks overlapping
    survey = LegacySurveyData(survey_dir=survey_dir)
    B = bricks_touching_wcs(wcs, survey=survey)
    print('Found', len(B), 'bricks overlapping')

    TT = []
    for b in B.brickname:
        fn = survey.find_file('tractor', brick=b)
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

    #return wcs,T

    # Pull out DECaLS coadds (image, model, resid).
    dwcs = wcs.scale(2. * pixscale / 0.262)
    dh,dw = dwcs.shape
    print('DECaLS resampled shape:', dh,dw)
    tags = ['image', 'model', 'resid']
    coimgs = [np.zeros((dh,dw,3), np.uint8) for t in tags]

    for b in B.brickname:
        fn = survey.find_file('image', brick=b, band='r')
        bwcs = Tan(fn, 1) # ext 1: .fz
        try:
            Yo,Xo,Yi,Xi,nil = resample_with_wcs(dwcs, bwcs)
        except ResampleError:
            continue
        if len(Yo) == 0:
            continue
        print('Resampling', len(Yo), 'pixels from', b)
        xl,xh,yl,yh = Xi.min(), Xi.max(), Yi.min(), Yi.max()
        #print('python legacypipe/runbrick.py -b %s --zoom %i %i %i %i --outdir cluster --pixpsf --splinesky --pipe --no-early-coadds' %
        #      (b, xl-5, xh+5, yl-5, yh+5) + ' -P \'pickles/cluster-%(brick)s-%%(stage)s.pickle\'')
        for i,tag in enumerate(tags):
            fn = survey.find_file(tag+'-jpeg', brick=b)
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

    srcs = read_fits_catalog(T)

    wbands = [1,2]
    wanyband = 'w'

    for band in wbands:
        f = T.get('flux_w%i' % band)
        f *= 10.**(primhdr['WISEAB%i' % band] / 2.5)

    coimgs = [np.zeros((H,W), np.float32) for b in wbands]
    comods = [np.zeros((H,W), np.float32) for b in wbands]
    con    = [np.zeros((H,W), np.uint8) for b in wbands]

    for iband,band in enumerate(wbands):
        print('Photometering WISE band', band)
        wband = 'w%i' % band

        for i,src in enumerate(srcs):
            #print('Source', src, 'brightness', src.getBrightness(), 'params', src.getBrightness().getParams())
            #src.getBrightness().setParams([T.wise_flux[i, band-1]])
            src.setBrightness(NanoMaggies(**{wanyband: T.get('flux_w%i'%band)[i]}))
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

    return wcs, T


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

def ra_ranges_overlap(ralo, rahi, ra1, ra2):
    import numpy as np
    x1 = np.cos(np.deg2rad(ralo))
    y1 = np.sin(np.deg2rad(ralo))
    x2 = np.cos(np.deg2rad(rahi))
    y2 = np.sin(np.deg2rad(rahi))
    x3 = np.cos(np.deg2rad(ra1))
    y3 = np.sin(np.deg2rad(ra1))
    x4 = np.cos(np.deg2rad(ra2))
    y4 = np.sin(np.deg2rad(ra2))
    cw32 = x2*y3 - x3*y2
    cw41 = x1*y4 - x4*y1
    return np.logical_and(cw32 <= 0, cw41 >= 0)

def galex_rgb(imgs, bands, **kwargs):
    import numpy as np
    from scipy.ndimage.filters import uniform_filter, gaussian_filter
    nuv,fuv = imgs
    h,w = nuv.shape
    red = nuv * 0.206 * 2297
    blue = fuv * 1.4 * 1525
    #blue = uniform_filter(blue, 3)
    blue = gaussian_filter(blue, 1.)
    green = (0.2*blue + 0.8*red)

    red   *= 0.085
    green *= 0.095
    blue  *= 0.08
    nonlinearity = 2.5
    radius = red + green + blue
    val = np.arcsinh(radius * nonlinearity) / nonlinearity
    with np.errstate(divide='ignore', invalid='ignore'):
        red   = red   * val / radius
        green = green * val / radius
        blue  = blue  * val / radius
    mx = np.maximum(red, np.maximum(green, blue))
    mx = np.maximum(1., mx)
    red   /= mx
    green /= mx
    blue  /= mx
    rgb = np.clip(np.dstack((red, green, blue)), 0., 1.)
    return rgb

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--ra', type=float,  default=329.0358)
    parser.add_argument('-d', '--dec', type=float, default=  1.3909)
    parser.add_argument('--radius', type=float, default=90., help='Cutout radius (arcsec)')
    parser.add_argument('--survey-dir', help='Legacy Survey base directory')
    parser.add_argument('--base', help='Base filename for output plots', default='cutouts')
    parser.add_argument('--galex-dir', help='Try making GALEX cutouts too?')

    opt = parser.parse_args()

    #ra,dec = 203.522, 20.232
    #ra,dec = 329.0358,1.3909  # horrible fit
    #ra,dec = 244.0424,6.9179

    # arcsec
    radius = opt.radius

    ps = PlotSequence(opt.base)

    plt.figure(figsize=(4,4))
    plt.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)

    wcs,T = wise_cutouts(opt.ra, opt.dec, radius, ps,
                         pixscale=2.75 / 2.,
                         survey_dir=opt.survey_dir)


    H,W = wcs.shape
    ralo,declo = wcs.pixelxy2radec(W,1)
    rahi,dechi = wcs.pixelxy2radec(1,H)
    print('RA',  ralo,rahi)
    print('Dec', declo,dechi)

    if opt.galex_dir:
        fn = os.path.join(opt.galex_dir, 'galex-images.fits')
        print('Reading', fn)
        # galex "bricks" (actually just GALEX tiles)
        galex = fits_table(fn)

        galex.rename('ra_cent', 'ra')
        galex.rename('dec_cent', 'dec')
        galex.rename('have_n', 'has_n')
        galex.rename('have_f', 'has_f')
        cosd = np.cos(np.deg2rad(galex.dec))
        galex.ra1 = galex.ra - 3840*1.5/3600./2./cosd
        galex.ra2 = galex.ra + 3840*1.5/3600./2./cosd
        galex.dec1 = galex.dec - 3840*1.5/3600./2.
        galex.dec2 = galex.dec + 3840*1.5/3600./2.
        bricknames = []
        for tile,subvis in zip(galex.tilename, galex.subvis):
            if subvis == -999:
                bricknames.append(tile.strip())
            else:
                bricknames.append('%s_sg%02i' % (tile.strip(), subvis))
        galex.brickname = np.array(bricknames)

        # bricks_touching_radec_box(self, ralo, rahi, declo, dechi, scale=None):
        I, = np.nonzero((galex.dec1 <= dechi) * (galex.dec2 >= declo))
        ok = ra_ranges_overlap(ralo, rahi, galex.ra1[I], galex.ra2[I])
        I = I[ok]
        galex.cut(I)
        print('-> bricks', galex.brickname)

        gbands = ['n','f']
        coimgs = []
        comods = []

        srcs = read_fits_catalog(T)
        for src in srcs:
            src.freezeAllBut('brightness')


        for band in gbands:
            J = np.flatnonzero(galex.get('has_'+band))
            print(len(J), 'GALEX tiles have coverage in band', band)

            coimg = np.zeros((H,W), np.float32)
            comod = np.zeros((H,W), np.float32)
            cowt  = np.zeros((H,W), np.float32)

            for src in srcs:
                src.setBrightness(NanoMaggies(**{band: 1}))

            for j in J:
                brick = galex[j]
                fn = os.path.join(opt.galex_dir, brick.tilename.strip(),
                                  '%s-%sd-intbgsub.fits.gz' % (brick.brickname, band))
                print(fn)

                gwcs = Tan(*[float(f) for f in
                             [brick.crval1, brick.crval2, brick.crpix1, brick.crpix2,
                              brick.cdelt1, 0., 0., brick.cdelt2, 3840., 3840.]])
                img = fitsio.read(fn)
                print('Read', img.shape)

                try:
                    Yo,Xo,Yi,Xi,nil = resample_with_wcs(wcs, gwcs, [], 3)
                except OverlapError:
                    continue

                K = np.flatnonzero(img[Yi,Xi] != 0.)
                if len(K) == 0:
                    continue
                Yo = Yo[K]
                Xo = Xo[K]
                Yi = Yi[K]
                Xi = Xi[K]

                rimg = np.zeros((H,W), np.float32)
                rimg[Yo,Xo] = img[Yi,Xi]

                plt.clf()
                plt.imshow(rimg, interpolation='nearest', origin='lower')
                ps.savefig()

                wt = brick.get(band + 'exptime')
                coimg[Yo,Xo] += wt * img[Yi,Xi]
                cowt [Yo,Xo] += wt


                x0 = min(Xi)
                x1 = max(Xi)
                y0 = min(Yi)
                y1 = max(Yi)
                subwcs = gwcs.get_subimage(x0, y0, x1-x0+1, y1-y0+1)
                twcs = ConstantFitsWcs(subwcs)
                timg = img[y0:y1+1, x0:x1+1]
                tie = np.ones_like(timg)  ## HACK!
                #hdr = fitsio.read_header(fn)
                #zp = hdr['
                zps = dict(n=20.08, f=18.82)
                zp = zps[band]
                photocal = LinearPhotoCal(NanoMaggies.zeropointToScale(zp),
                                          band=band)
                tsky = ConstantSky(0.)

                # HACK -- circular Gaussian PSF of fixed size...
                # in arcsec
                #fwhms = dict(NUV=6.0, FUV=6.0)
                # -> sigma in pixels
                #sig = fwhms[band] / 2.35 / twcs.pixel_scale()
                sig = 6.0 / 2.35 / twcs.pixel_scale()
                tpsf = NCircularGaussianPSF([sig], [1.])

                tim = Image(data=timg, inverr=tie, psf=tpsf, wcs=twcs,
                            sky=tsky, photocal=photocal,
                            name='GALEX ' + band + brick.brickname)
                tractor = Tractor([tim], srcs)
                mod = tractor.getModelImage(0)

                print('Tractor image', tim.name)
                plt.clf()
                plt.imshow(timg, interpolation='nearest', origin='lower')
                ps.savefig()

                print('Tractor model', tim.name)
                plt.clf()
                plt.imshow(mod, interpolation='nearest', origin='lower')
                ps.savefig()

                tractor.freezeParam('images')

                print('Params:')
                tractor.printThawedParams()
            
                tractor.optimize_forced_photometry(priors=False, shared_params=False)

                mod = tractor.getModelImage(0)

                print('Tractor model (forced phot)', tim.name)
                plt.clf()
                plt.imshow(mod, interpolation='nearest', origin='lower')
                ps.savefig()

                comod[Yo,Xo] += wt * mod[Yi-y0,Xi-x0]


            coimg /= np.maximum(cowt, 1e-18)
            comod /= np.maximum(cowt, 1e-18)
            coimgs.append(coimg)
            comods.append(comod)

            print('Coadded image')
            plt.clf()
            plt.imshow(coimg, interpolation='nearest', origin='lower')
            ps.savefig()

            print('Coadded model')
            plt.clf()
            plt.imshow(comod, interpolation='nearest', origin='lower')
            ps.savefig()

        rgb = galex_rgb(coimgs, gbands)
        plt.clf()
        plt.imshow(rgb, interpolation='nearest', origin='lower')
        ps.savefig()

        print('Model RGB')
        rgb = galex_rgb(comods, gbands)
        plt.clf()
        plt.imshow(rgb, interpolation='nearest', origin='lower')
        ps.savefig()

