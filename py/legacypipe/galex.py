"""
LSLGA.galex
===========

Code to generate GALEX custom coadds / mosaics.

"""
import os, pdb
import numpy as np

from astrometry.util.util import Tan
from astrometry.util.fits import fits_table

import LSLGA.misc

def _ra_ranges_overlap(ralo, rahi, ra1, ra2):
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

def _galex_rgb_dstn(imgs, **kwargs):
    nuv,fuv = imgs
    h,w = nuv.shape
    myrgb = np.zeros((h,w,3), np.float32)
    lo,hi = -0.0005, 0.01
    myrgb[:,:,0] = myrgb[:,:,1] = np.clip((nuv - lo) / (hi - lo), 0., 1.)
    lo,hi = -0.00015, 0.003
    myrgb[:,:,2] = np.clip((fuv - lo) / (hi - lo), 0., 1.)
    myrgb[:,:,1] = np.clip((myrgb[:,:,0] + myrgb[:,:,2]*0.2), 0., 1.)
    return myrgb

def _galex_rgb_official(imgs, **kwargs):
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

def _galex_rgb_moustakas(imgs, **kwargs):
    #from scipy.ndimage.filters import uniform_filter, gaussian_filter
    nuv,fuv = imgs
    h,w = nuv.shape
    red = nuv * 0.206 * 2297
    blue = fuv * 1.4 * 1525
    #blue = uniform_filter(blue, 3)
    #blue = gaussian_filter(blue, 1.)
    green = (0.2*blue + 0.8*red)

    red   *= 0.085
    green *= 0.095
    blue  *= 0.08
    nonlinearity = 0.5 # 1.0 # 2.5
    radius = red + green + blue
    val = np.arcsinh(radius * nonlinearity) / nonlinearity
    with np.errstate(divide='ignore', invalid='ignore'):
        red   = red   * val / radius
        green = green * val / radius
        blue  = blue  * val / radius
    mx = np.maximum(red, np.maximum(green, blue))
    mx = np.maximum(1., mx)

    lo = -0.1
    red   = (red - lo) / (mx - lo)
    green = (green - lo) / (mx - lo)
    blue  = (blue - lo) / (mx - lo)
    #red   /= mx
    #green /= mx
    #blue  /= mx
    
    rgb = np.clip(np.dstack((red, green, blue)), 0., 1.)
    return rgb

def _read_galex_tiles(targetwcs, galex_dir, log=None, verbose=False):
    """Find and read the overlapping GALEX FUV/NUV tiles."""

    H, W = targetwcs.shape
    
    ralo, declo = targetwcs.pixelxy2radec(W, 1)
    rahi, dechi = targetwcs.pixelxy2radec(1, H)
    #print('RA',  ralo,rahi)
    #print('Dec', declo,dechi)

    fn = os.path.join(galex_dir, 'galex-images.fits')
    #print('Reading', fn)
    # galex "bricks" (actually just GALEX tiles)
    galex_tiles = fits_table(fn)
    galex_tiles.rename('ra_cent', 'ra')
    galex_tiles.rename('dec_cent', 'dec')
    galex_tiles.rename('have_n', 'has_n')
    galex_tiles.rename('have_f', 'has_f')
    
    cosd = np.cos(np.deg2rad(galex_tiles.dec))
    galex_tiles.ra1 = galex_tiles.ra - 3840*1.5/3600./2./cosd
    galex_tiles.ra2 = galex_tiles.ra + 3840*1.5/3600./2./cosd
    galex_tiles.dec1 = galex_tiles.dec - 3840*1.5/3600./2.
    galex_tiles.dec2 = galex_tiles.dec + 3840*1.5/3600./2.
    bricknames = []
    for tile, subvis in zip(galex_tiles.tilename, galex_tiles.subvis):
        if subvis == -999:
            bricknames.append(tile.strip())
        else:
            bricknames.append('%s_sg%02i' % (tile.strip(), subvis))
    galex_tiles.brickname = np.array(bricknames)

    # bricks_touching_radec_box(self, ralo, rahi, declo, dechi, scale=None):
    I, = np.nonzero((galex_tiles.dec1 <= dechi) * (galex_tiles.dec2 >= declo))
    ok = _ra_ranges_overlap(ralo, rahi, galex_tiles.ra1[I], galex_tiles.ra2[I])
    I = I[ok]
    galex_tiles.cut(I)
    if verbose:
        print('-> bricks', galex_tiles.brickname, flush=True, file=log)

    return galex_tiles

def galex_coadds(onegal, galaxy=None, radius_mosaic=30, radius_mask=None,
                 pixscale=1.5, ref_pixscale=0.262, output_dir=None, galex_dir=None,
                 log=None, centrals=True, verbose=False):
    '''Generate custom GALEX cutouts.
    
    radius_mosaic and radius_mask in arcsec
    
    pixscale: GALEX pixel scale in arcsec/pixel.

    '''
    import fitsio
    import matplotlib.pyplot as plt

    from astrometry.libkd.spherematch import match_radec
    from astrometry.util.resample import resample_with_wcs, OverlapError
    from tractor import (Tractor, NanoMaggies, Image, LinearPhotoCal,
                         NCircularGaussianPSF, ConstantFitsWcs, ConstantSky)

    from legacypipe.survey import imsave_jpeg
    from legacypipe.catalog import read_fits_catalog

    if galaxy is None:
        galaxy = 'galaxy'

    if galex_dir is None:
        galex_dir = os.environ.get('GALEX_DIR')

    if output_dir is None:
        output_dir = '.'

    if radius_mask is None:
        radius_mask = radius_mosaic
        radius_search = 5.0 # [arcsec]
    else:
        radius_search = radius_mask

    W = H = np.ceil(2 * radius_mosaic / pixscale).astype('int') # [pixels]
    targetwcs = Tan(onegal['RA'], onegal['DEC'], (W+1) / 2.0, (H+1) / 2.0,
                    -pixscale / 3600.0, 0.0, 0.0, pixscale / 3600.0, float(W), float(H))

    # Read the custom Tractor catalog
    tractorfile = os.path.join(output_dir, '{}-tractor.fits'.format(galaxy))
    if not os.path.isfile(tractorfile):
        print('Missing Tractor catalog {}'.format(tractorfile))
        return 0

    cat = fits_table(tractorfile)
    print('Read {} sources from {}'.format(len(cat), tractorfile), flush=True, file=log)

    keep = np.ones(len(cat)).astype(bool)
    if centrals:
        # Find the large central galaxy and mask out (ignore) all the models
        # which are within its elliptical mask.

        # This algorithm will have to change for mosaics not centered on large
        # galaxies, e.g., in galaxy groups.
        m1, m2, d12 = match_radec(cat.ra, cat.dec, onegal['RA'], onegal['DEC'],
                                  radius_search/3600.0, nearest=False)
        if len(m1) == 0:
            print('No central galaxies found at the central coordinates!', flush=True, file=log)
        else:
            pixfactor = ref_pixscale / pixscale # shift the optical Tractor positions
            for mm in m1:
                morphtype = cat.type[mm].strip()
                if morphtype == 'EXP' or morphtype == 'COMP':
                    e1, e2, r50 = cat.shapeexp_e1[mm], cat.shapeexp_e2[mm], cat.shapeexp_r[mm] # [arcsec]
                elif morphtype == 'DEV' or morphtype == 'COMP':
                    e1, e2, r50 = cat.shapedev_e1[mm], cat.shapedev_e2[mm], cat.shapedev_r[mm] # [arcsec]
                else:
                    r50 = None

                if r50:
                    majoraxis =  r50 * 5 / pixscale # [pixels]
                    ba, phi = LSLGA.misc.convert_tractor_e1e2(e1, e2)
                    these = LSLGA.misc.ellipse_mask(W / 2, W / 2, majoraxis, ba * majoraxis,
                                                    np.radians(phi), cat.bx*pixfactor, cat.by*pixfactor)
                    if np.sum(these) > 0:
                        #keep[these] = False
                        pass
                print('Hack!')
                keep[mm] = False

            #srcs = read_fits_catalog(cat)
            #_srcs = np.array(srcs)[~keep].tolist()
            #mod = LSLGA.misc.srcs2image(_srcs, ConstantFitsWcs(targetwcs), psf_sigma=3.0)
            #import matplotlib.pyplot as plt
            ##plt.imshow(mod, origin='lower') ; plt.savefig('junk.png')
            #plt.imshow(np.log10(mod), origin='lower') ; plt.savefig('junk.png')
            #pdb.set_trace()

    srcs = read_fits_catalog(cat)
    for src in srcs:
        src.freezeAllBut('brightness')
    #srcs_nocentral = np.array(srcs)[keep].tolist()
    
    # Find all overlapping GALEX tiles and then read the tims.
    galex_tiles = _read_galex_tiles(targetwcs, galex_dir, log=log, verbose=verbose)

    gbands = ['n','f']
    nicegbands = ['NUV', 'FUV']

    zps = dict(n=20.08, f=18.82)

    coimgs, comods, coresids, coimgs_central, comods_nocentral = [], [], [], [], []
    for niceband, band in zip(nicegbands, gbands):
        J = np.flatnonzero(galex_tiles.get('has_'+band))
        print(len(J), 'GALEX tiles have coverage in band', band)

        coimg = np.zeros((H, W), np.float32)
        comod = np.zeros((H, W), np.float32)
        cowt  = np.zeros((H, W), np.float32)

        comod_nocentral = np.zeros((H, W), np.float32)

        for src in srcs:
            src.setBrightness(NanoMaggies(**{band: 1}))

        for j in J:
            brick = galex_tiles[j]
            fn = os.path.join(galex_dir, brick.tilename.strip(),
                              '%s-%sd-intbgsub.fits.gz' % (brick.brickname, band))
            #print(fn)

            gwcs = Tan(*[float(f) for f in
                         [brick.crval1, brick.crval2, brick.crpix1, brick.crpix2,
                          brick.cdelt1, 0., 0., brick.cdelt2, 3840., 3840.]])
            img = fitsio.read(fn)
            #print('Read', img.shape)

            try:
                Yo, Xo, Yi, Xi, nil = resample_with_wcs(targetwcs, gwcs, [], 3)
            except OverlapError:
                continue

            K = np.flatnonzero(img[Yi, Xi] != 0.)
            if len(K) == 0:
                continue
            Yo, Xo, Yi, Xi = Yo[K], Xo[K], Yi[K], Xi[K]

            wt = brick.get(band + 'exptime')
            coimg[Yo, Xo] += wt * img[Yi, Xi]
            cowt [Yo, Xo] += wt

            x0, x1, y0, y1 = min(Xi), max(Xi), min(Yi), max(Yi)
            subwcs = gwcs.get_subimage(x0, y0, x1-x0+1, y1-y0+1)
            twcs = ConstantFitsWcs(subwcs)
            timg = img[y0:y1+1, x0:x1+1]

            tie = np.ones_like(timg)  ## HACK!
            #hdr = fitsio.read_header(fn)
            #zp = hdr['']
            zp = zps[band]
            photocal = LinearPhotoCal( NanoMaggies.zeropointToScale(zp), band=band)
            tsky = ConstantSky(0.0)
            
            # HACK -- circular Gaussian PSF of fixed size...
            # in arcsec
            #fwhms = dict(NUV=6.0, FUV=6.0)
            # -> sigma in pixels
            #sig = fwhms[band] / 2.35 / twcs.pixel_scale()
            sig = 6.0 / np.sqrt(8 * np.log(2)) / twcs.pixel_scale()
            tpsf = NCircularGaussianPSF([sig], [1.])

            tim = Image(data=timg, inverr=tie, psf=tpsf, wcs=twcs, sky=tsky,
                        photocal=photocal, name='GALEX ' + band + brick.brickname)

            ## Build the model image with and without the central galaxy model.
            tractor = Tractor([tim], srcs)
            mod = tractor.getModelImage(0)
            tractor.freezeParam('images')
            tractor.optimize_forced_photometry(priors=False, shared_params=False)
            mod = tractor.getModelImage(0)

            srcs_nocentral = np.array(srcs)[keep].tolist()
            #srcs_nocentral = np.array(srcs)[nocentral].tolist()
            tractor_nocentral = Tractor([tim], srcs_nocentral)
            mod_nocentral = tractor_nocentral.getModelImage(0)

            comod[Yo, Xo] += wt * mod[Yi-y0, Xi-x0]
            comod_nocentral[Yo, Xo] += wt * mod_nocentral[Yi-y0, Xi-x0]

        coimg /= np.maximum(cowt, 1e-18)
        comod /= np.maximum(cowt, 1e-18)
        comod_nocentral /= np.maximum(cowt, 1e-18)

        coresid = coimg - comod

        # Subtract the model image which excludes the central (comod_nocentral)
        # from the data (coimg) to isolate the light of the central
        # (coimg_central).
        coimg_central = coimg - comod_nocentral

        coimgs.append(coimg)
        comods.append(comod)
        coresids.append(coresid)

        comods_nocentral.append(comod_nocentral)
        coimgs_central.append(coimg_central)

        # Write out the final images with and without the central, making sure
        # to apply the zeropoint to go from counts/s to AB nanomaggies.
        # https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html
        for thisimg, imtype in zip( (coimg, comod, comod_nocentral),
                                ('image', 'model', 'model-nocentral') ):
            fitsfile = os.path.join(output_dir, '{}-{}-{}.fits'.format(galaxy, imtype, niceband))
            if verbose:
                print('Writing {}'.format(fitsfile))
            fitsio.write(fitsfile, thisimg * 10**(-0.4 * (zp - 22.5)), clobber=True)

    # Build a color mosaic (but note that the images here are in units of
    # background-subtracted counts/s).

    #_galex_rgb = _galex_rgb_moustakas
    #_galex_rgb = _galex_rgb_dstn
    _galex_rgb = _galex_rgb_official

    for imgs, imtype in zip( (coimgs, comods, coresids, comods_nocentral, coimgs_central),
                             ('image', 'model', 'resid', 'model-nocentral', 'image-central') ):
        rgb = _galex_rgb(imgs)
        jpgfile = os.path.join(output_dir, '{}-{}-FUVNUV.jpg'.format(galaxy, imtype))
        if verbose:
            print('Writing {}'.format(jpgfile))
        imsave_jpeg(jpgfile, rgb, origin='lower')

    return 1
