import numpy as np
import fitsio
import os

import logging
logger = logging.getLogger('legacypipe.outliers')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)


OUTLIER_POS = 1
OUTLIER_NEG = 2

def read_outlier_mask_file(survey, tims, brickname):
    from legacypipe.image import CP_DQ_BITS
    fn = survey.find_file('outliers_mask', brick=brickname, output=True)
    if not os.path.exists(fn):
        return False
    F = fitsio.FITS(fn)
    for tim in tims:
        extname = '%s-%s-%s' % (tim.camera, tim.expnum, tim.ccdname)
        if not extname in F:
            print('WARNING: Did not find extension', extname, 'in outlier-mask file', fn)
            return False
        mask = F[extname].read()
        hdr = F[extname].read_header()
        if mask.shape != tim.shape:
            print('Warning: Outlier mask', fn, 'does not match shape of tim', tim)
            return False
        x0 = hdr['X0']
        y0 = hdr['Y0']
        if x0 != tim.x0 or y0 != tim.y0:
            print('Warning: Outlier mask', fn, 'x0,y0 does not match that of tim', tim)
            return False
        # Apply this mask!
        tim.dq |= (mask > 0) * CP_DQ_BITS['outlier']
        tim.inverr[mask > 0] = 0.
    return True

def mask_outlier_pixels(survey, tims, bands, targetwcs, brickname, version_header,
                        mp=None, plots=False, ps=None, make_badcoadds=True):
    from legacypipe.image import CP_DQ_BITS
    if plots:
        import pylab as plt

    H,W = targetwcs.shape

    if make_badcoadds:
        badcoadds = []
    else:
        badcoadds = None

    args = []
    for iband,band in enumerate(bands):
        btims = [tim for tim in tims if tim.band == band]
        if len(btims) == 0:
            continue
        debug(len(btims), 'images for band', band)
        args.append((band, btims, targetwcs, make_badcoadds, plots, ps))

    band_masks = mp.map(bounce_outliers_one_band, args)

    with survey.write_output('outliers_mask', brick=brickname) as out:
        # empty Primary HDU
        out.fits.write(None, header=version_header)

        for (band,btims),masks in zip(args, band_masks):
            for tim, (mask,badcoadd) in zip(btims, masks):
                if make_badcoadds:
                    badcoadds.append(badcoadd)
                
                # Apply the mask!
                tim.inverr[mask > 0] = 0.
                tim.dq[mask > 0] |= CP_DQ_BITS['outlier']

                # copy version_header before modifying it.
                hdr = fitsio.FITSHDR()
                # Plug in the tim WCS header
                tim.subwcs.add_to_header(hdr)
                hdr.delete('IMAGEW')
                hdr.delete('IMAGEH')
                hdr.add_record(dict(name='IMTYPE', value='outlier_mask',
                                    comment='LegacySurvey image type'))
                hdr.add_record(dict(name='CAMERA',  value=tim.camera))
                hdr.add_record(dict(name='EXPNUM',  value=tim.expnum))
                hdr.add_record(dict(name='CCDNAME', value=tim.ccdname))
                hdr.add_record(dict(name='X0', value=tim.x0))
                hdr.add_record(dict(name='Y0', value=tim.y0))

                out.fits.write(mask, header=hdr, compress='HCOMPRESS')

    return badcoadds

def outliers_one_band(band, btims, targetwcs, make_badcoadds, plots, ps):
    import time
    from scipy.ndimage.filters import gaussian_filter
    from scipy.ndimage.morphology import binary_dilation
    from astrometry.util.resample import resample_with_wcs,OverlapError
    from legacypipe.image import CP_DQ_BITS
    
    H,W = targetwcs.shape
    # Build blurred reference image
    sigs = np.array([tim.psf_sigma for tim in btims])
    debug('PSF sigmas:', sigs)
    targetsig = max(sigs) + 0.5
    addsigs = np.sqrt(targetsig**2 - sigs**2)
    debug('Target sigma:', targetsig)
    debug('Blur sigmas:', addsigs)
    coimg = np.zeros((H,W), np.float32)
    cow   = np.zeros((H,W), np.float32)
    masks = np.zeros((H,W), np.int16)
    resams = []

    resam_time = 0.

    for tim,sig in zip(btims, addsigs):
        img = gaussian_filter(tim.getImage(), sig)
        try:
            t0 = time.clock()
            Yo,Xo,Yi,Xi,[rimg] = resample_with_wcs(
                targetwcs, tim.subwcs, [img], 3)
            print('Resample int types:', Yo.dtype, Xo.dtype, Yi.dtype, Xi.dtype)
            resam_time += (time.clock() - t0)
        except OverlapError:
            resams.append(None)
            continue
        del img
        blurnorm = 1./(2. * np.sqrt(np.pi) * sig)
        #print('Blurring "psf" norm', blurnorm)
        wt = tim.getInvvar()[Yi,Xi] / (blurnorm**2)
        coimg[Yo,Xo] += rimg * wt
        cow  [Yo,Xo] += wt
        masks[Yo,Xo] |= (tim.dq[Yi,Xi])
        resams.append([x.astype(np.int16) for x in [Yo,Xo,Yi,Xi]] + [rimg,wt])

    print('Total resampling time:', resam_time)
        
    #
    veto = np.logical_or(
        binary_dilation(masks & CP_DQ_BITS['bleed'], iterations=3),
        binary_dilation(masks & CP_DQ_BITS['satur'], iterations=10))
    del masks

    if plots:
        plt.clf()
        plt.imshow(veto, interpolation='nearest', origin='lower', cmap='gray')
        plt.title('SATUR, BLEED veto (%s band)' % band)
        ps.savefig()

    badcoadd = None
    if make_badcoadds:
        badcoadd = np.zeros((H,W), np.float32)
        badcon   = np.zeros((H,W), np.int16)

    band_masks = []

    # Compare against reference image...
    for tim,resam in zip(btims, resams):
        # create and store the result right away...
        maskedpix = np.zeros(tim.shape, np.uint8)
        band_masks.append(maskedpix)

        if resam is None:
            continue
        (Yo,Xo,Yi,Xi, rimg,wt) = resam

        # Subtract this image from the coadd
        otherwt = cow[Yo,Xo] - wt
        otherimg = (coimg[Yo,Xo] - rimg*wt) / np.maximum(otherwt, 1e-16)
        this_sig1 = 1./np.sqrt(np.median(wt[wt>0]))

        ## FIXME -- this image edges??

        # Compute the error on our estimate of (thisimg - co) =
        # sum in quadrature of the errors on thisimg and co.
        with np.errstate(divide='ignore'):
            diffvar = 1./wt + 1./otherwt
            sndiff = (rimg - otherimg) / np.sqrt(diffvar)

        with np.errstate(divide='ignore'):
            reldiff = ((rimg - otherimg) / np.maximum(otherimg, this_sig1))

        if plots:
            plt.clf()
            showimg = np.zeros((H,W),np.float32)
            showimg[Yo,Xo] = otherimg
            plt.subplot(2,3,1)
            plt.imshow(showimg, interpolation='nearest', origin='lower', vmin=-0.01, vmax=0.1,
                       cmap='gray')
            plt.title('other images')
            showimg[Yo,Xo] = otherwt
            plt.subplot(2,3,2)
            plt.imshow(showimg, interpolation='nearest', origin='lower', vmin=0)
            plt.title('other wt')
            showimg[Yo,Xo] = sndiff
            plt.subplot(2,3,3)
            plt.imshow(showimg, interpolation='nearest', origin='lower', vmin=0, vmax=10)
            plt.title('S/N diff')
            showimg[Yo,Xo] = rimg
            plt.subplot(2,3,4)
            plt.imshow(showimg, interpolation='nearest', origin='lower', vmin=-0.01, vmax=0.1,
                       cmap='gray')
            plt.title('this image')
            showimg[Yo,Xo] = wt
            plt.subplot(2,3,5)
            plt.imshow(showimg, interpolation='nearest', origin='lower', vmin=0)
            plt.title('this wt')
            plt.suptitle(tim.name)
            showimg[Yo,Xo] = reldiff
            plt.subplot(2,3,6)
            plt.imshow(showimg, interpolation='nearest', origin='lower', vmin=0, vmax=4)
            plt.title('rel diff')
            ps.savefig()

            from astrometry.util.plotutils import loghist
            plt.clf()
            loghist(sndiff.ravel(), reldiff.ravel(),
                    bins=100)
            plt.xlabel('S/N difference')
            plt.ylabel('Relative difference')
            plt.title('Outliers: ' + tim.name)
            ps.savefig()

        del otherimg

        # TEST:
        #sndiff = np.abs(sndiff)
        #reldiff = np.abs(reldiff)

        # Significant pixels
        hotpix = ((sndiff > 5.) * (reldiff > 2.) *
                  (otherwt > 1e-16) * (wt > 0.) *
                  (veto[Yo,Xo] == False))

        coldpix = ((sndiff < -5.) * (reldiff < -2.) *
                  (otherwt > 1e-16) * (wt > 0.) *
                  (veto[Yo,Xo] == False))

        del reldiff, otherwt

        if (not np.any(hotpix)) and (not np.any(coldpix)):
            ## FIXME
            #return hotpix
            continue

        hot = np.zeros((H,W), bool)
        hot[Yo,Xo] = hotpix
        cold = np.zeros((H,W), bool)
        cold[Yo,Xo] = coldpix

        del hotpix, coldpix

        snmap = np.zeros((H,W), np.float32)
        snmap[Yo,Xo] = sndiff

        hot = binary_dilation(hot, iterations=1)
        cold = binary_dilation(cold, iterations=1)
        if plots:
            heat = hot.astype(np.uint8)
        # "warm"
        hot = np.logical_or(hot,
                            binary_dilation(hot, iterations=5) * (snmap > 3.))
        hot = binary_dilation(hot, iterations=1)
        cold = np.logical_or(cold,
                            binary_dilation(cold, iterations=5) * (snmap < -3.))
        cold = binary_dilation(cold, iterations=1)

        if plots:
            heat += hot
        # "lukewarm"
        hot = np.logical_or(hot,
                            binary_dilation(hot, iterations=5) * (snmap > 2.))
        hot = binary_dilation(hot, iterations=3)
        cold = np.logical_or(cold,
                            binary_dilation(cold, iterations=5) * (snmap < -2.))
        cold = binary_dilation(cold, iterations=3)

        if plots:
            heat += hot
            plt.clf()
            plt.imshow(heat, interpolation='nearest', origin='lower', cmap='hot')
            plt.title(tim.name + ': outliers')
            ps.savefig()
            del heat

        del snmap

        bad, = np.nonzero(hot[Yo,Xo])

        if make_badcoadds:
            badcoadd[Yo[bad],Xo[bad]] += tim.getImage()[Yi[bad],Xi[bad]]
            badcon[Yo[bad],Xo[bad]] += 1

        # Actually do the masking!
        # Resample "hot" (in brick coords) back to tim coords.
        try:
            mYo,mXo,mYi,mXi,nil = resample_with_wcs(
                tim.subwcs, targetwcs, [], 3)
        except OverlapError:
            continue
        Ibad, = np.nonzero(hot[mYi,mXi])
        Ibad2, = np.nonzero(cold[mYi,mXi])
        info(tim, ': masking', len(Ibad), 'positive outlier pixels and', len(Ibad2), 'negative outlier pixels')
        #nz = np.sum(tim.getInvError() == 0)
        maskedpix[mYo[Ibad],  mXo[Ibad]]  = OUTLIER_POS
        maskedpix[mYo[Ibad2], mXo[Ibad2]] = OUTLIER_NEG

    if make_badcoadds:
        badcoadd /= np.maximum(badcon, 1)

    return band_masks, badcoadd

def bounce_outliers_one_band(X):
    (band, btims, targetwcs, make_badcoadds, plots, ps) = X
    try:
        return outliers_one_band(band, btims, targetwcs, make_badcoadds, plots, ps)
    except:
        import traceback
        traceback.print_exc()

def patch_from_coadd(coimgs, targetwcs, bands, tims, mp=None):
    from astrometry.util.resample import resample_with_wcs, OverlapError

    H,W = targetwcs.shape
    ibands = dict([(b,i) for i,b in enumerate(bands)])
    for tim in tims:
        ie = tim.getInvvar()
        img = tim.getImage()
        if np.any(ie == 0):
            # Patch from the coadd
            co = coimgs[ibands[tim.band]]
            # resample from coadd to img -- nearest-neighbour
            iy,ix = np.nonzero(ie == 0)
            if len(iy) == 0:
                continue
            ra,dec = tim.subwcs.pixelxy2radec(ix+1, iy+1)[-2:]
            ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)
            xx = np.round(xx-1).astype(np.int16)
            yy = np.round(yy-1).astype(np.int16)
            keep = (xx >= 0) * (xx < W) * (yy >= 0) * (yy < H)
            if not np.any(keep):
                continue
            img[iy[keep],ix[keep]] = coimgs[ibands[tim.band]][yy[keep],xx[keep]]
            del co

