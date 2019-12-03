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

from legacypipe.bits import OUTLIER_POS, OUTLIER_NEG

def get_bits_to_mask():
    return OUTLIER_POS | OUTLIER_NEG

def read_outlier_mask_file(survey, tims, brickname, subimage=True, output=True, ps=None):
    '''if subimage=True, assume that 'tims' are subimages, and demand that they have the same
    x0,y0 pixel offsets and size as the outlier mask files.

    if subimage=False, assume that 'tims' are full-CCD images, and
    apply the mask to the relevant subimage.

    *output* determines where we search for the file: treating it as output, or input?

    '''

    from legacypipe.bits import DQ_BITS
    fn = survey.find_file('outliers_mask', brick=brickname, output=output)
    if not os.path.exists(fn):
        print('Failed to apply outlier mask: No such file:', fn)
        return False
    F = fitsio.FITS(fn)
    for tim in tims:
        extname = '%s-%s-%s' % (tim.imobj.camera, tim.imobj.expnum, tim.imobj.ccdname)
        if not extname in F:
            print('WARNING: Did not find extension', extname, 'in outlier-mask file', fn)
            return False
        mask = F[extname].read()
        hdr = F[extname].read_header()
        if subimage:
            if mask.shape != tim.shape:
                print('Warning: Outlier mask', fn, 'does not match shape of tim', tim)
                return False
        x0 = hdr['X0']
        y0 = hdr['Y0']
        maskbits = get_bits_to_mask()
        if subimage:
            if x0 != tim.x0 or y0 != tim.y0:
                print('Warning: Outlier mask', fn, 'x0,y0 does not match that of tim', tim)
                return False
            # Apply this mask!
            tim.dq |= ((mask & maskbits) > 0) * DQ_BITS['outlier']
            tim.inverr[(mask & maskbits) > 0] = 0.
        else:

            from astrometry.util.miscutils import get_overlapping_region
            mh,mw = mask.shape
            th,tw = tim.shape
            my,ty = get_overlapping_region(tim.y0, tim.y0 + th - 1, y0, y0 + mh - 1)
            mx,tx = get_overlapping_region(tim.x0, tim.x0 + tw - 1, x0, x0 + mw - 1)
            # have to shift the "m" slices down by x0,y0
            my = slice(my.start - y0, my.stop - y0)
            mx = slice(mx.start - x0, mx.stop - x0)
            tim.dq[ty, tx] |= ((mask[my, mx] & maskbits) > 0) * DQ_BITS['outlier']
            tim.inverr[ty, tx][(mask[my, mx] & maskbits) > 0] = 0.

            if ps is not None:
                import pylab as plt
                print('Mask extent: x [%i, %i], vs tim extent x [%i, %i]' % (x0, x0+mw, tim.x0, tim.x0+tw))
                print('Mask extent: y [%i, %i], vs tim extent y [%i, %i]' % (y0, y0+mh, tim.y0, tim.y0+th))
                print('x slice: mask', mx, 'tim', tx)
                print('y slice: mask', my, 'tim', ty)
                print('tim shape:', tim.shape)
                print('mask shape:', mask.shape)
                
                newdq = np.zeros(tim.shape, bool)
                newdq[ty, tx] = ((mask[my, mx] & maskbits) > 0)
                print('Total of', np.sum(newdq), 'pixels masked')
                plt.clf()
                plt.imshow(tim.getImage(), interpolation='nearest', origin='lower', vmin=-2.*tim.sig1, vmax=5.*tim.sig1, cmap='gray')
                ax = plt.axis()
                from legacypipe.detection import plot_boundary_map
                plot_boundary_map(newdq, iterations=3, rgb=(0,128,255))
                plt.axis(ax)
                ps.savefig()

                plt.axis([tx.start, tx.stop, ty.start, ty.stop])
                ps.savefig()
            
    return True

def mask_outlier_pixels(survey, tims, bands, targetwcs, brickname, version_header,
                        mp=None, plots=False, ps=None, make_badcoadds=True,
                        refstars=None):
    from legacypipe.bits import DQ_BITS
    from legacypipe.reference import read_gaia
    from scipy.ndimage.filters import gaussian_filter
    from scipy.ndimage.morphology import binary_dilation
    from astrometry.util.resample import resample_with_wcs,OverlapError
    if plots:
        import pylab as plt

    H,W = targetwcs.shape

    if make_badcoadds:
        badcoadds_pos = []
        badcoadds_neg = []
    else:
        badcoadds_pos = None
        badcoadds_neg = None

    star_veto = np.zeros(targetwcs.shape, np.bool)
    if refstars:
        gaia = refstars[refstars.isgaia]
        # Not moving Gaia stars to epoch of individual images...
        ok,bx,by = targetwcs.radec2pixelxy(gaia.ra, gaia.dec)
        bx -= 1.
        by -= 1.
        # Radius to mask around Gaia stars, in arcsec
        radius = 1.0
        pixrad = radius / targetwcs.pixel_scale()
        for x,y in zip(bx,by):
            xlo = int(np.clip(np.floor(x - pixrad), 0, W-1))
            xhi = int(np.clip(np.ceil (x + pixrad), 0, W-1))
            ylo = int(np.clip(np.floor(y - pixrad), 0, H-1))
            yhi = int(np.clip(np.ceil (y + pixrad), 0, H-1))
            if xlo == xhi or ylo == yhi:
                continue
            r2 = (((np.arange(ylo,yhi+1) - y)**2)[:,np.newaxis] +
                  ((np.arange(xlo,xhi+1) - x)**2)[np.newaxis,:])
            star_veto[ylo:yhi+1, xlo:xhi+1] |= (r2 < pixrad)

    # if plots:
    #     plt.clf()
    #     plt.imshow(star_veto, interpolation='nearest', origin='lower',
    #                vmin=0, vmax=1, cmap='hot')
    #     ax = plt.axis()
    #     plt.plot(bx, by, 'r.')
    #     plt.axis(ax)
    #     plt.title('Star vetos')
    #     ps.savefig()

    with survey.write_output('outliers_mask', brick=brickname) as out:
        # empty Primary HDU
        out.fits.write(None, header=version_header)

        for iband,band in enumerate(bands):
            btims = [tim for tim in tims if tim.band == band]
            if len(btims) == 0:
                continue
            debug(len(btims), 'images for band', band)
            
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
    
            R = mp.map(blur_resample_one, [(tim,sig,targetwcs) for tim,sig in zip(btims,addsigs)])
            for tim,r in zip(btims, R):
                if r is None:
                    continue
                Yo,Xo,iacc,wacc,macc = r
                coimg[Yo,Xo] += iacc
                cow  [Yo,Xo] += wacc
                masks[Yo,Xo] |= macc
                del Yo,Xo,iacc,wacc,macc
            del r,R
    
            #
            veto = np.logical_or(star_veto,
                                 np.logical_or(
                binary_dilation(masks & DQ_BITS['bleed'], iterations=3),
                binary_dilation(masks & DQ_BITS['satur'], iterations=10)))
            del masks
        
            # if plots:
            #     plt.clf()
            #     plt.imshow(veto, interpolation='nearest', origin='lower', cmap='gray')
            #     plt.title('SATUR, BLEED veto (%s band)' % band)
            #     ps.savefig()
        
            R = mp.map(compare_one, [(tim, sig, targetwcs, coimg,cow, veto, make_badcoadds, plots,ps)
                                     for tim,sig in zip(btims,addsigs)])
            del coimg, cow, veto
    
            masks = []
            badcoadd_pos = None
            badcoadd_neg = None
            if make_badcoadds:
                badcoadd_pos = np.zeros((H,W), np.float32)
                badcon_pos   = np.zeros((H,W), np.int16)
                badcoadd_neg = np.zeros((H,W), np.float32)
                badcon_neg   = np.zeros((H,W), np.int16)

            for r,tim in zip(R, btims):
                if r is None:
                    # none masked
                    mask = np.zeros(tim.shape, np.uint8)
                else:
                    mask,badco = r
                    if make_badcoadds:
                        badhot, badcold = badco
                        yo,xo,bimg = badhot
                        badcoadd_pos[yo, xo] += bimg
                        badcon_pos  [yo, xo] += 1
                        yo,xo,bimg = badcold
                        badcoadd_neg[yo, xo] += bimg
                        badcon_neg  [yo, xo] += 1
                        del yo,xo,bimg, badhot,badcold
                    del badco


                # Apply the mask!
                maskbits = get_bits_to_mask()
                tim.inverr[(mask & maskbits) > 0] = 0.
                tim.dq[(mask & maskbits) > 0] |= DQ_BITS['outlier']

                # Write output!
                # copy version_header before modifying it.
                hdr = fitsio.FITSHDR()
                # Plug in the tim WCS header
                tim.subwcs.add_to_header(hdr)
                hdr.delete('IMAGEW')
                hdr.delete('IMAGEH')
                hdr.add_record(dict(name='IMTYPE', value='outlier_mask',
                                    comment='LegacySurvey image type'))
                hdr.add_record(dict(name='CAMERA',  value=tim.imobj.camera))
                hdr.add_record(dict(name='EXPNUM',  value=tim.imobj.expnum))
                hdr.add_record(dict(name='CCDNAME', value=tim.imobj.ccdname))
                hdr.add_record(dict(name='X0', value=tim.x0))
                hdr.add_record(dict(name='Y0', value=tim.y0))

                # HCOMPRESS;: 943k
                # GZIP_1: 4.4M
                # GZIP: 4.4M
                # RICE: 2.8M
                extname = '%s-%s-%s' % (tim.imobj.camera, tim.imobj.expnum, tim.imobj.ccdname)
                out.fits.write(mask, header=hdr, extname=extname, compress='HCOMPRESS')

            del r,R
        
            if make_badcoadds:
                badcoadd_pos /= np.maximum(badcon_pos, 1)
                badcoadd_neg /= np.maximum(badcon_neg, 1)
                badcoadds_pos.append(badcoadd_pos)
                badcoadds_neg.append(badcoadd_neg)

    return badcoadds_pos,badcoadds_neg

def compare_one(X):
    from scipy.ndimage.filters import gaussian_filter
    from scipy.ndimage.morphology import binary_dilation
    from astrometry.util.resample import resample_with_wcs,OverlapError

    (tim,sig,targetwcs, coimg,cow, veto, make_badcoadds, plots,ps) = X

    if plots:
        import pylab as plt

    H,W = targetwcs.shape

    img = gaussian_filter(tim.getImage(), sig)
    try:
        Yo,Xo,Yi,Xi,[rimg] = resample_with_wcs(
            targetwcs, tim.subwcs, [img], intType=np.int16)
    except OverlapError:
        return None
    del img
    blurnorm = 1./(2. * np.sqrt(np.pi) * sig)
    wt = tim.getInvvar()[Yi,Xi] / np.float32(blurnorm**2)
    if Xi.dtype != np.int16:
        Yi = Yi.astype(np.int16)
        Xi = Xi.astype(np.int16)

    # Compare against reference image...
    maskedpix = np.zeros(tim.shape, np.uint8)
    
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
        plt.imshow(showimg, interpolation='nearest', origin='lower', vmin=-10, vmax=10,cmap='RdBu_r')
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
        plt.imshow(showimg, interpolation='nearest', origin='lower', vmin=-4, vmax=4, cmap='RdBu_r')
        plt.title('rel diff')
        ps.savefig()

        # from astrometry.util.plotutils import loghist
        # plt.clf()
        # loghist(sndiff.ravel(), reldiff.ravel(),
        #         bins=100)
        # plt.xlabel('S/N difference')
        # plt.ylabel('Relative difference')
        # plt.title('Outliers: ' + tim.name)
        # ps.savefig()
    
    del otherimg
    
    # Significant pixels
    hotpix = ((sndiff > 5.) * (reldiff > 2.) *
              (otherwt > 1e-16) * (wt > 0.) *
              (veto[Yo,Xo] == False))
    
    coldpix = ((sndiff < -5.) * (reldiff < -2.) *
               (otherwt > 1e-16) * (wt > 0.) *
               (veto[Yo,Xo] == False))
    
    del reldiff, otherwt
    
    if (not np.any(hotpix)) and (not np.any(coldpix)):
        return None
    
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
        heat = np.zeros(hot.shape, np.int8)
        heat += hot
        heat += -1 * cold
    # "warm"
    hot = np.logical_or(hot,
                        binary_dilation(hot, iterations=5) * (snmap > 3.))
    hot = binary_dilation(hot, iterations=1)
    cold = np.logical_or(cold,
                         binary_dilation(cold, iterations=5) * (snmap < -3.))
    cold = binary_dilation(cold, iterations=1)
    
    if plots:
        heat += hot
        heat +- -1*cold
    # "lukewarm"
    hot = np.logical_or(hot,
                        binary_dilation(hot, iterations=5) * (snmap > 2.))
    hot = binary_dilation(hot, iterations=3)
    cold = np.logical_or(cold,
                         binary_dilation(cold, iterations=5) * (snmap < -2.))
    cold = binary_dilation(cold, iterations=3)
    
    if plots:
        heat += hot
        heat +- -1*cold
        plt.clf()
        plt.imshow(heat, interpolation='nearest', origin='lower', cmap='RdBu_r', vmin=-3, vmax=+3)
        plt.title(tim.name + ': outliers')
        ps.savefig()
        del heat

    del snmap

    badco = None
    if make_badcoadds:
        bad, = np.nonzero(hot[Yo,Xo])
        badhot = (Yo[bad], Xo[bad], tim.getImage()[Yi[bad],Xi[bad]])
        bad, = np.nonzero(cold[Yo,Xo])
        badcold = (Yo[bad], Xo[bad], tim.getImage()[Yi[bad],Xi[bad]])
        badco = badhot,badcold

    # Actually do the masking!
    # Resample "hot" (in brick coords) back to tim coords.
    try:
        mYo,mXo,mYi,mXi,nil = resample_with_wcs(
            tim.subwcs, targetwcs, intType=np.int16)
    except OverlapError:
        return None
    Ibad, = np.nonzero(hot[mYi,mXi])
    Ibad2, = np.nonzero(cold[mYi,mXi])
    info(tim, ': masking', len(Ibad), 'positive outlier pixels and', len(Ibad2), 'negative outlier pixels')
    maskedpix[mYo[Ibad],  mXo[Ibad]]  = OUTLIER_POS
    maskedpix[mYo[Ibad2], mXo[Ibad2]] = OUTLIER_NEG

    return maskedpix,badco


def blur_resample_one(X):
    from scipy.ndimage.filters import gaussian_filter
    from astrometry.util.resample import resample_with_wcs,OverlapError

    tim,sig,targetwcs = X

    img = gaussian_filter(tim.getImage(), sig)
    try:
        Yo,Xo,Yi,Xi,[rimg] = resample_with_wcs(
            targetwcs, tim.subwcs, [img], intType=np.int16)
    except OverlapError:
        return None
    del img
    blurnorm = 1./(2. * np.sqrt(np.pi) * sig)
    wt = tim.getInvvar()[Yi,Xi] / (blurnorm**2)
    return (Yo, Xo, rimg*wt, wt, tim.dq[Yi,Xi])

def patch_from_coadd(coimgs, targetwcs, bands, tims, mp=None):
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
            xx = (xx - 1. + 0.5).astype(np.int16)
            yy = (yy - 1. + 0.5).astype(np.int16)
            keep = (xx >= 0) * (xx < W) * (yy >= 0) * (yy < H)
            if not np.any(keep):
                continue
            img[iy[keep],ix[keep]] = coimgs[ibands[tim.band]][yy[keep],xx[keep]]
            del co
