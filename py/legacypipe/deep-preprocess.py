import sys
import warnings

import numpy as np

from scipy.ndimage import binary_dilation

import fitsio

from astrometry.util.ttime import Time
from astrometry.util.multiproc import multiproc

from legacypipe.runbrick import get_parser, get_runbrick_kwargs, run_brick
from legacypipe.survey import imsave_jpeg
from legacypipe.coadds import make_coadds
from legacypipe.outliers import patch_from_coadd, mask_outlier_pixels
from legacypipe.outliers import blur_resample_one
from legacypipe.outliers import OUTLIER_POS, OUTLIER_NEG
from legacypipe.bits import DQ_BITS
from legacypipe.utils import NothingToDoError, RunbrickError
from legacypipe.runbrick import stage_refs

import logging
logger = logging.getLogger('legacypipe.deep-preprocess')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

def formatwarning(message, category, filename, lineno, line=None):
    return 'Warning: %s (%s:%i)' % (message, filename, lineno)
    #return 'Warning: %s' % (message)
warnings.formatwarning = formatwarning

def stage_deep_preprocess(
        W=3600, H=3600, pixscale=0.262, brickname=None,
        survey=None,
        ra=None, dec=None,
        release=None,
        plots=False, ps=None,
        target_extent=None, program_name='runbrick.py',
        bands=None,
        do_calibs=True,
        old_calibs_ok=True,
        splinesky=True,
        subsky=True,
        gaussPsf=False, pixPsf=True, hybridPsf=True,
        normalizePsf=True,
        apodize=False,
        constant_invvar=False,
        read_image_pixels = True,
        min_mjd=None, max_mjd=None,
        gaia_stars=True,
        mp=None,
        record_event=None,
        unwise_dir=None,
        unwise_tr_dir=None,
        unwise_modelsky_dir=None,
        galex_dir=None,
        command_line=None,
        read_parallel=True,
        max_memory_gb=None,
        refstars=None,
        depth_margin=0.5,
        **kwargs):
    from legacypipe.survey import (
        get_git_version, get_version_header, get_dependency_versions,
        wcs_for_brick, read_one_tim)
    from legacypipe.depthcut import make_depth_cut

    from astrometry.util.starutil_numpy import ra2hmsstring, dec2dmsstring

    tlast = Time()
    record_event and record_event('stage_deep: starting')
    assert(survey is not None)

    info('Depth margin:', depth_margin)

    # Get brick object
    custom_brick = (ra is not None)
    if custom_brick:
        from legacypipe.survey import BrickDuck
        # Custom brick; create a fake 'brick' object
        brick = BrickDuck(ra, dec, brickname)
    else:
        brick = survey.get_brick_by_name(brickname)
        if brick is None:
            raise RunbrickError('No such brick: "%s"' % brickname)
    brickid = brick.brickid
    brickname = brick.brickname

    # Get WCS object describing brick
    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    if target_extent is not None:
        (x0,x1,y0,y1) = target_extent
        W = x1-x0
        H = y1-y0
        targetwcs = targetwcs.get_subimage(x0, y0, W, H)
    pixscale = targetwcs.pixel_scale()
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])
    # custom brick -- set RA,Dec bounds
    if custom_brick:
        brick.ra1,_  = targetwcs.pixelxy2radec(W, H/2)
        brick.ra2,_  = targetwcs.pixelxy2radec(1, H/2)
        _, brick.dec1 = targetwcs.pixelxy2radec(W/2, 1)
        _, brick.dec2 = targetwcs.pixelxy2radec(W/2, H)

    # Create FITS header with version strings
    gitver = get_git_version()

    version_header = get_version_header(program_name, survey.survey_dir, release,
                                        git_version=gitver)

    deps = get_dependency_versions(unwise_dir, unwise_tr_dir, unwise_modelsky_dir, galex_dir)
    for name,value,comment in deps:
        version_header.add_record(dict(name=name, value=value, comment=comment))
    if command_line is not None:
        version_header.add_record(dict(name='CMDLINE', value=command_line,
                                       comment='runbrick command-line'))
    version_header.add_record(dict(name='BRICK', value=brickname,
                                comment='LegacySurveys brick RRRr[pm]DDd'))
    version_header.add_record(dict(name='BRICKID' , value=brickid,
                                comment='LegacySurveys brick id'))
    version_header.add_record(dict(name='RAMIN'   , value=brick.ra1,
                                comment='Brick RA min (deg)'))
    version_header.add_record(dict(name='RAMAX'   , value=brick.ra2,
                                comment='Brick RA max (deg)'))
    version_header.add_record(dict(name='DECMIN'  , value=brick.dec1,
                                comment='Brick Dec min (deg)'))
    version_header.add_record(dict(name='DECMAX'  , value=brick.dec2,
                                comment='Brick Dec max (deg)'))
    # Add NOIRLab-requested headers
    version_header.add_record(dict(
        name='RA', value=ra2hmsstring(brick.ra, separator=':'), comment='Brick center RA (hms)'))
    version_header.add_record(dict(
        name='DEC', value=dec2dmsstring(brick.dec, separator=':'), comment='Brick center DEC (dms)'))
    version_header.add_record(dict(
        name='CENTRA', value=brick.ra, comment='Brick center RA (deg)'))
    version_header.add_record(dict(
        name='CENTDEC', value=brick.dec, comment='Brick center Dec (deg)'))
    for i,(r,d) in enumerate(targetrd[:4]):
        version_header.add_record(dict(
            name='CORN%iRA' %(i+1), value=r, comment='Brick corner RA (deg)'))
        version_header.add_record(dict(
            name='CORN%iDEC'%(i+1), value=d, comment='Brick corner Dec (deg)'))

    # Find CCDs
    ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)
    if ccds is None:
        raise NothingToDoError('No CCDs touching brick')
    debug(len(ccds), 'CCDs touching target WCS')
    survey.drop_cache()

    if 'ccd_cuts' in ccds.get_columns():
        ccds.cut(ccds.ccd_cuts == 0)
        debug(len(ccds), 'CCDs survive cuts')
    else:
        warnings.warn('Not applying CCD cuts')

    # Cut on bands to be used
    ccds.cut(np.array([b in bands for b in ccds.filter]))
    debug('Cut to', len(ccds), 'CCDs in bands', ','.join(bands))

    debug('Cutting on CCDs to be used for fitting...')
    I = survey.ccds_for_fitting(brick, ccds)
    if I is not None:
        debug('Cutting to', len(I), 'of', len(ccds), 'CCDs for fitting.')
        ccds.cut(I)

    if min_mjd is not None:
        ccds.cut(ccds.mjd_obs >= min_mjd)
        debug('Cut to', len(ccds), 'after MJD', min_mjd)
    if max_mjd is not None:
        ccds.cut(ccds.mjd_obs <= max_mjd)
        debug('Cut to', len(ccds), 'before MJD', max_mjd)

    # Create Image objects for each CCD
    ims = []
    info('Keeping', len(ccds), 'CCDs:')
    for ccd in ccds:
        im = survey.get_image_object(ccd)
        if survey.cache_dir is not None:
            im.check_for_cached_files(survey)
        ims.append(im)
        info(' ', im, im.band, 'expnum', im.expnum, 'exptime', im.exptime, 'propid', ccd.propid,
              'seeing %.2f' % (ccd.fwhm*im.pixscale), 'MJD %.3f' % ccd.mjd_obs,
              'object', getattr(ccd, 'object', '').strip(), '\n   ', im.print_imgpath)

    tnow = Time()
    debug('Finding images touching brick:', tnow-tlast)
    tlast = tnow

    if max_memory_gb:
        # Estimate total memory required for tim pixels
        mem = sum([im.estimate_memory_required(radecpoly=targetrd,
                                               mywcs=survey.get_approx_wcs(ccd))
                                               for im,ccd in zip(ims,ccds)])
        info('Estimated memory required: %.1f GB' % (mem/1e9))
        if mem / 1e9 > max_memory_gb:
            raise RuntimeError('Too much memory required: %.1f > %.1f GB' % (mem/1e9, max_memory_gb))

    keep,_,_ = make_depth_cut(survey, ccds, bands, targetrd, brick, W, H, pixscale,
                            plots, ps, splinesky, gaussPsf, pixPsf, normalizePsf,
                            do_calibs, gitver, targetwcs, old_calibs_ok,
                            margin=depth_margin)
    Ikeep = np.flatnonzero(keep)
    deepccds = ccds[Ikeep]
    deepims = [ims[i] for i in Ikeep]
    info('Cut to', len(deepccds), 'CCDs required to reach depth targets')

    # Read tims for 'deepccds'
    args = [(im, targetrd, dict(gaussPsf=gaussPsf, pixPsf=pixPsf,
                                hybridPsf=hybridPsf, normalizePsf=normalizePsf,
                                subsky=subsky,
                                apodize=apodize,
                                constant_invvar=constant_invvar,
                                pixels=read_image_pixels,
                                old_calibs_ok=old_calibs_ok))
                                for im in deepims]
    deeptims = list(mp.map(read_one_tim, args))

    # Drop any deepims and tims where the deeptim is None.
    deepims =  [di for di,dt in zip(deepims,deeptims) if dt is not None]
    deeptims = [dt for dt in deeptims if dt is not None]

    # Start outlier masking...
    deepC = make_coadds(deeptims, bands, targetwcs, mp=mp, sbscale=False,
                        allmasks=False, coweights=False)
    with survey.write_output('outliers-pre', brick=brickname) as out:
        rgb,kwa = survey.get_rgb(deepC.coimgs, bands)
        imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
        del rgb

    # Patch individual-CCD masked pixels from deep coadd
    patch_from_coadd(deepC.coimgs, targetwcs, bands, deeptims, mp=mp)
    del deepC

    # Find outliers in deep set...
    make_badcoadds = False
    mask_outlier_pixels(survey, deeptims, bands, targetwcs, brickname, version_header,
                        mp=mp, plots=plots, ps=ps, make_badcoadds=make_badcoadds,
                        refstars=refstars)
    deepC = make_coadds(deeptims, bands, targetwcs, mp=mp, sbscale=False,
                        allmasks=False, coweights=False)
    with survey.write_output('outliers-post', brick=brickname) as out:
        rgb,kwa = survey.get_rgb(deepC.coimgs, bands)
        imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
        del rgb
    del deepC

    keys = ['version_header', 'targetrd', 'pixscale', 'targetwcs', 'W','H',
            'ps', 'brickid', 'brickname', 'brick', 'custom_brick',
            'target_extent', 'ccds', 'bands', 'survey',
            'ims', 'deepims', 'deeptims',
    ]
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def stage_deep_preprocess_2(
        W=3600, H=3600, pixscale=0.262, brickname=None,
        survey=None,
        ra=None, dec=None,
        release=None,
        plots=False, ps=None,
        target_extent=None, program_name='runbrick.py',
        bands=None,
        do_calibs=True,
        old_calibs_ok=True,
        splinesky=True,
        subsky=True,
        gaussPsf=False, pixPsf=True, hybridPsf=True,
        normalizePsf=True,
        apodize=False,
        constant_invvar=False,
        read_image_pixels = True,
        min_mjd=None, max_mjd=None,
        gaia_stars=True,
        mp=None,
        record_event=None,
        unwise_dir=None,
        unwise_tr_dir=None,
        unwise_modelsky_dir=None,
        galex_dir=None,
        command_line=None,
        read_parallel=True,
        max_memory_gb=None,
        refstars=None,

        targetrd=None, targetwcs=None, brick=None, version_header=None,
        ccds=None,
        ims=None,
        deepims=None,
        deeptims=None,
        nsatur=None,
        star_halos=True,

        **kwargs):

    # Using 'deeptims', create template coadds for outlier detection, AND deep coadds.

    # coadds for patching images...
    info('Making coadds for patching images...')
    deepC = make_coadds(deeptims, bands, targetwcs, mp=mp, sbscale=False,
                        allmasks=False, coweights=False)

    star_veto = np.zeros(targetwcs.shape, bool)
    if refstars:
        gaia = refstars[refstars.isgaia]
        # Not moving Gaia stars to epoch of individual images...
        _,bx,by = targetwcs.radec2pixelxy(gaia.ra, gaia.dec)
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

    badcoadds_pos = []
    badcoadds_neg = []
    detmaps = []
    detivs  = []
    satmaps = []
    coadds = []
    coivs  = []
    have_bands = []

    with survey.write_output('outliers_mask', brick=brickname) as out:
        # empty Primary HDU
        out.fits.write(None, header=version_header)

        for iband,band in enumerate(bands):
            # Build blurred reference image (from 'deeptims') for outlier rejection
            btims = [tim for tim in deeptims if tim.band == band]
            if len(btims) == 0:
                continue
            info(len(btims), 'deep images for band', band)
            have_bands.append(band)

            info('Making blurred reference image...')
            H,W = targetwcs.shape
            sigs = np.array([tim.psf_sigma for tim in btims])
            debug('PSF sigmas:', sigs)
            targetsig = max(sigs) + 0.5
            addsigs = np.sqrt(targetsig**2 - sigs**2)
            debug('Target sigma:', targetsig)
            debug('Blur sigmas:', addsigs)
            coimg = np.zeros((H,W), np.float32)
            cowt  = np.zeros((H,W), np.float32)
            masks = np.zeros((H,W), np.int16)

            results = mp.imap_unordered(
                blur_resample_one, [(i_btim,tim,sig,targetwcs)
                                    for i_btim,(tim,sig) in enumerate(zip(btims,addsigs))])
            for _,r in results:
                if r is None:
                    continue
                Yo,Xo,iacc,wacc,macc = r
                coimg[Yo,Xo] += iacc
                cowt [Yo,Xo] += wacc
                masks[Yo,Xo] |= macc
                del Yo,Xo,iacc,wacc,macc
                del r
            del results
            #
            veto = np.logical_or(star_veto,
                                 np.logical_or(
                binary_dilation(masks & DQ_BITS['bleed'], iterations=3),
                binary_dilation(masks & DQ_BITS['satur'], iterations=10)))
            del masks

            coimg /= np.maximum(1e-16, cowt)
            refimg = coimg
            refiv  = cowt
            del coimg,cowt

            deep_sig = targetsig

            #info('Total of', len([im for im in ims if im.band == band]), 'images in', band, 'band')
            #bims = [im for im in ims if im.band == band and not im in deepims]
            #info('Running on', len(bims), 'individual images (not in deep set)...')

            # We actually still want to run all the processing steps
            # on the images in the deep set too!
            bims = [im for im in ims if im.band == band]
            info('Total of', len(bims), 'images in', band, 'band')
            if len(bims) == 0:
                continue

            patch_img = deepC.coimgs[iband]

            tim_kwargs = dict(gaussPsf=gaussPsf, pixPsf=pixPsf,
                              hybridPsf=hybridPsf, normalizePsf=normalizePsf,
                              subsky=subsky,
                              apodize=apodize,
                              constant_invvar=constant_invvar,
                              old_calibs_ok=old_calibs_ok)

            coimg = np.zeros((H,W), np.float32)
            coiv  = np.zeros((H,W), np.float32)
            coflat = np.zeros((H,W), np.float32)
            con    = np.zeros((H,W), np.uint16)
            detmap = np.zeros((H,W), np.float32)
            detiv  = np.zeros((H,W), np.float32)
            sattype = np.uint8
            satmax = 254
            satmap = np.zeros((H,W), sattype)
            BIG = 1e6
            badcoadd_pos = np.empty((H,W), np.float32)
            badcoadd_neg = np.empty((H,W), np.float32)
            badcoadd_pos[:,:] = -BIG
            badcoadd_neg[:,:] = +BIG

            # Prep halo subtraction
            halostars = None
            if star_halos and refstars:
                Igaia, = np.nonzero(refstars.isgaia * refstars.pointsource)
                info(len(Igaia), 'stars for halo subtraction')
                if len(Igaia):
                    halostars = refstars[Igaia]

            R = mp.imap_unordered(
                mask_and_coadd_one,
                [(i_bim, targetrd, tim_kwargs, im, targetwcs,
                  patch_img, refimg, refiv, veto, deep_sig, halostars, old_calibs_ok,
                  plots,ps)
                  for i_bim,im in enumerate(bims)])
            del refimg, refiv, veto, patch_img

            t0 = Time()

            nb = len(bims)
            n = 0

            for i_bim,res in R:
                im = bims[i_bim]
                n += 1
                if res is None:
                    # no overlap
                    continue
                (Yo,Xo, rimg, riv, dq, det, div, sat, badco,
                 outl_mask, x0, y0, hdr) = res
                info('Accumulating', n, 'of', nb, im, ':', Time()-t0)

                if badco is not None:
                    badhot, badcold = badco
                    yo,xo,bimg = badhot
                    badcoadd_pos[yo, xo] = np.maximum(badcoadd_pos[yo, xo], bimg)
                    yo,xo,bimg = badcold
                    badcoadd_neg[yo, xo] = np.minimum(badcoadd_neg[yo, xo], bimg)
                    del yo,xo,bimg, badhot,badcold
                    del badco

                if dq is None:
                    goodpix = 1
                else:
                    # include SATUR pixels if no other pixels exists
                    okbits = 0
                    for bitname in ['satur']:
                        okbits |= DQ_BITS[bitname]
                    brightpix = ((dq & okbits) != 0)
                    satur_val=10.
                    # force SATUR pix to be bright
                    rimg[brightpix] = satur_val
                    # Include these pixels if none other exist??
                    for bitname in ['interp']:
                        okbits |= DQ_BITS[bitname]
                    goodpix = ((dq & ~okbits) == 0)

                coimg [Yo,Xo] += rimg * riv
                coiv  [Yo,Xo] += riv
                coflat[Yo,Xo] += goodpix * rimg
                con   [Yo,Xo] += goodpix
                del rimg,riv

                detmap[Yo,Xo] += det * div
                detiv [Yo,Xo] += div
                del det,div

                if sat is not None:
                    satmap[Yo,Xo] = np.minimum(satmax, satmap[Yo,Xo] + (1*sat))
                del sat,Yo,Xo
                del res

                # Write output!
                hdr.add_record(dict(name='IMTYPE', value='outlier_mask',
                                    comment='LegacySurvey image type'))
                hdr.add_record(dict(name='CAMERA',  value=im.camera))
                hdr.add_record(dict(name='EXPNUM',  value=im.expnum))
                hdr.add_record(dict(name='CCDNAME', value=im.ccdname))
                hdr.add_record(dict(name='X0', value=x0))
                hdr.add_record(dict(name='Y0', value=y0))

                extname = '%s-%s-%s' % (im.camera, im.expnum, im.ccdname)
                out.fits.write(outl_mask.astype(np.uint8), header=hdr, extname=extname,
                               compress='HCOMPRESS')
            del R

            # Un-touched pixels -> 0
            badcoadd_pos[badcoadd_pos == -BIG] = 0.
            badcoadd_neg[badcoadd_neg == +BIG] = 0.

            badcoadds_pos.append(badcoadd_pos)
            badcoadds_neg.append(badcoadd_neg)
            detmaps.append(detmap)
            detmap /= np.maximum(detiv, 1e-16)
            detivs.append(detiv)
            if nsatur is None:
                nsatur = 1
            satmap = (satmap >= nsatur)
            satmaps.append(satmap)
            tinyw = 1e-30
            coimg /= np.maximum(coiv, tinyw)
            coflat /= np.maximum(con,1)
            # patch
            coimg[coiv == 0] = coflat[coiv == 0]
            coadds.append(coimg)
            coivs.append(coiv)

            del badcoadd_pos, badcoadd_neg, detmap, detiv, satmap, coimg, coiv, con, coflat

    with survey.write_output('outliers-masked-pos', brick=brickname) as out:
        rgb,kwa = survey.get_rgb(badcoadds_pos, bands)
        imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
        del rgb
    del badcoadds_pos
    with survey.write_output('outliers-masked-neg', brick=brickname) as out:
        rgb,kwa = survey.get_rgb(badcoadds_neg, bands)
        imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
        del rgb
    del badcoadds_neg

    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='ccdinfo',
                            comment='NOIRLab data product type'))

    # Write per-brick CCDs table
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=primhdr)

    from legacypipe.coadds import write_coadd_images
    co_sky = None
    for band,coimg,coiv in zip(bands, coadds, coivs):
        bandtims = [tim for tim in deeptims if tim.band == band]
        if len(bandtims) == 0:
            continue
        write_coadd_images(band, survey, brickname, version_header,
                           bandtims, targetwcs, co_sky,
                           cowimg=coimg, cow=coiv)

    with survey.write_output('image-jpeg', brick=brickname) as out:
        rgb,kwa = survey.get_rgb(coadds, bands)
        imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
        debug('Wrote', out.fn)
    #del coadds,coivs

    bands = have_bands
    keys = ['detmaps', 'detivs', 'satmaps', 'coadds', 'coivs', 'bands']
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def mask_and_coadd_one(X):
    from legacypipe.survey import read_one_tim
    from scipy.ndimage.filters import gaussian_filter
    from astrometry.util.resample import resample_with_wcs,OverlapError

    (i_bim, targetrd, tim_kwargs, im, targetwcs, patchimg, coimg, cow, veto,
     deep_sig, halostars, old_calibs_ok, plots, ps) = X

    # - read tim
    tim = read_one_tim((im, targetrd, tim_kwargs))
    if tim is None:
        return i_bim,None
    # - patch image
    patch_from_coadd([patchimg], targetwcs, [tim.band], [tim])

    # - compute blurring sigmas for image & reference
    targetsig = max(deep_sig, tim.psf_sigma + 0.5)
    deep_blursig = np.sqrt(targetsig**2 - deep_sig**2)
    tim_blursig  = np.sqrt(targetsig**2 - tim.psf_sigma**2)

    # Blur tim
    assert(tim_blursig > 0)
    blurimg = gaussian_filter(tim.getImage(), tim_blursig)
    blurnorm = 1./(2. * np.sqrt(np.pi) * tim_blursig)

    # Blur ref image
    if deep_blursig > 0:
        refimg = gaussian_filter(coimg, deep_blursig)
        refblurnorm = 1./(2. * np.sqrt(np.pi) * deep_blursig)
    else:
        refimg = coimg
        refblurnorm = 1.
    # Resample reference image to tim space.
    try:
        Yo,Xo,Yi,Xi,[rref] = resample_with_wcs(
            tim.subwcs, targetwcs, [refimg], intType=np.int16)
    except OverlapError:
        return i_bim,None
    del refimg
    refimg = rref
    refwt = cow[Yi,Xi] / (refblurnorm**2)
    vetopix = veto[Yi,Xi]
    del veto

    blurimg = blurimg[Yo,Xo]
    blurwt = tim.getInvvar()[Yo,Xo] / (blurnorm**2)
    blur_sig1 = tim.sig1 * blurnorm

    # Compute the error on our estimate of (blurimg - refimg) =
    # sum in quadrature of the errors on thisimg and co.
    with np.errstate(divide='ignore'):
        diffvar = 1./blurwt + 1./refwt
        sndiff = (blurimg - refimg) / np.sqrt(diffvar)
    with np.errstate(divide='ignore'):
        reldiff = ((blurimg - refimg) / np.maximum(refimg, blur_sig1))

    # Significant pixels
    hotpix = ((sndiff > 5.) * (reldiff > 2.) *
              (refwt > 1e-16) * (blurwt > 0.) * (vetopix == False))
    coldpix = ((sndiff < -5.) * (reldiff < -2.) *
               (refwt > 1e-16) * (blurwt > 0.) * (vetopix == False))
    del reldiff, refwt, vetopix

    if np.any(hotpix) or np.any(coldpix):
        # Plug hotpix,coldpix,snmap (which are vectors of pixels) back to tim-shaped images.
        hot = np.zeros(tim.shape, bool)
        hot[Yo,Xo] = hotpix
        cold = np.zeros(tim.shape, bool)
        cold[Yo,Xo] = coldpix
        del hotpix, coldpix
        snmap = np.zeros(tim.shape, np.float32)
        snmap[Yo,Xo] = sndiff
        del sndiff
        hot = binary_dilation(hot, iterations=1)
        cold = binary_dilation(cold, iterations=1)
        # "warm"
        hot = np.logical_or(hot,
                            binary_dilation(hot, iterations=5) * (snmap > 3.))
        hot = binary_dilation(hot, iterations=1)
        cold = np.logical_or(cold,
                             binary_dilation(cold, iterations=5) * (snmap < -3.))
        cold = binary_dilation(cold, iterations=1)
        # "lukewarm"
        hot = np.logical_or(hot,
                            binary_dilation(hot, iterations=5) * (snmap > 2.))
        hot = binary_dilation(hot, iterations=3)
        cold = np.logical_or(cold,
                             binary_dilation(cold, iterations=5) * (snmap < -2.))
        cold = binary_dilation(cold, iterations=3)
        del snmap

        print('Exposure', im.expnum, im.ccdname, 'image', im.name, ': masking %i hot and %i cold pixels' % (np.sum(hot), np.sum(cold)))
        # Set outlier mask bits (kind of irrelevant) and zero ivar.
        if tim.dq is not None:
            tim.dq |= tim.dq_type((hot | cold) * DQ_BITS['outlier'])
        tim.inverr[(hot | cold)] = 0.

        # For the bad coadds
        bad, = np.nonzero(hot[Yo,Xo])
        badhot = (Yi[bad], Xi[bad], tim.getImage()[Yo[bad],Xo[bad]])
        bad, = np.nonzero(cold[Yo,Xo])
        badcold = (Yi[bad], Xi[bad], tim.getImage()[Yo[bad],Xo[bad]])
        badco = badhot,badcold

        # returned outlier mask:
        outl_mask = np.zeros(tim.shape, np.uint8)
        outl_mask[hot]  |= OUTLIER_POS
        outl_mask[cold] |= OUTLIER_NEG

        # - patch again
        patch_from_coadd([patchimg], targetwcs, [tim.band], [tim])
    else:
        badco = None
        outl_mask = np.zeros(tim.shape, bool)

    # Do halo subtraction
    if halostars:
        from legacypipe.halos import subtract_halos
        mp = multiproc()
        subtract_halos([tim], halostars, [tim.band], mp, plots, ps, old_calibs_ok=old_calibs_ok)


    # - resample for regular coadd and detection map
    from legacypipe.detection import _detmap
    try:
        Yo,Xo,Yi,Xi,[rimg] = resample_with_wcs(
            targetwcs, tim.subwcs, [tim.getImage()], intType=np.int16)
    except OverlapError:
        return i_bim,None
    tim.resamp = (Yo,Xo,Yi,Xi)
    apodize = 10
    _,_,_,detim,detiv,sat = _detmap((tim, targetwcs, apodize))

    from legacypipe.utils import copy_header_with_wcs
    hdr = copy_header_with_wcs(None, tim.subwcs)

    if tim.dq is None:
        dq = None
    else:
        dq = tim.dq[Yi,Xi]
    iv = tim.getInvError()[Yi,Xi]**2
    x0,y0 = tim.x0, tim.y0
    del tim, Yi, Xi

    return i_bim, (Yo, Xo, rimg, iv, dq, detim, detiv, sat, badco,
                   outl_mask, x0, y0, hdr)

def stage_deep_preprocess_3(
        W=3600, H=3600, pixscale=0.262, brickname=None,
        survey=None,
        ra=None, dec=None,
        release=None,
        plots=False, ps=None,
        target_extent=None, program_name='runbrick.py',
        bands=None,
        do_calibs=True,
        old_calibs_ok=True,
        splinesky=True,
        subsky=True,
        gaussPsf=False, pixPsf=True, hybridPsf=True,
        normalizePsf=True,
        apodize=False,
        constant_invvar=False,
        read_image_pixels = True,
        min_mjd=None, max_mjd=None,
        gaia_stars=True,
        mp=None,
        record_event=None,
        unwise_dir=None,
        unwise_tr_dir=None,
        unwise_modelsky_dir=None,
        galex_dir=None,
        command_line=None,
        read_parallel=True,
        max_memory_gb=None,
        refstars=None,

        targetrd=None, targetwcs=None, brick=None, version_header=None,
        ccds=None,
        ims=None,
        deepims=None,
        deeptims=None,
        nsatur=None,
        star_halos=True,

        blob_dilate=None,
        nsigma=None,

        detmaps=None,
        detivs=None,
        satmaps=None,
        coadds=None,
        coivs=None,

        **kwargs):
    from legacypipe.blobmask import generate_blobmask

    info('Bands:', bands)
    info('Detivs:', len(detivs))
    if len(detivs) < len(bands):
        have_bands = []
        for iband,band in enumerate(bands):
            btims = [tim for tim in deeptims if tim.band == band]
            if len(btims) == 0:
                continue
            have_bands.append(band)
        bands = have_bands
        info('Actual bands:', bands)

    assert(len(bands) == len(detivs))

    hot, saturated_pix = generate_blobmask(
        survey, bands, nsigma, detmaps, detivs, satmaps, blob_dilate,
        version_header, targetwcs, brickname, record_event)
    del detmaps, detivs, satmaps

    from legacypipe.maskbits_light import write_maskbits_light
    write_maskbits_light(survey, brick, brickname, version_header,
                         targetwcs, W, H, refstars)

    keys = ['hot', 'saturated_pix', 'version_header', ]
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def main(args=None):
    import datetime
    from legacypipe.survey import get_git_version

    print()
    print('deep-preprocess.py starting at', datetime.datetime.now().isoformat())
    print('legacypipe git version:', get_git_version())
    if args is None:
        print('Command-line args:', sys.argv)
        cmd = 'python'
        for vv in sys.argv:
            cmd += ' {}'.format(vv)
        print(cmd)
    else:
        print('Args:', args)
    print()

    parser = get_parser()
    parser.add_argument('--depth-margin', type=float, default=0.5,
                        help='Set margin for the depth-cut code, beyond the DESI targets of g=24.0, r=23.4, i=23.0, z=22.5.  Default %(default)s')
    opt = parser.parse_args(args=args)
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    # default stage
    if len(opt.stage) == 0:
        opt.stage = ['deep_preprocess_3']

    optdict = vars(opt)
    verbose = optdict.pop('verbose')
    rgb_stretch = optdict.pop('rgb_stretch', None)

    survey, kwargs = get_runbrick_kwargs(**optdict)
    if kwargs in [-1, 0]:
        return kwargs
    kwargs.update(command_line=' '.join(sys.argv))

    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)
    # silence "findfont: score(<Font 'DejaVu Sans Mono' ...)" messages
    logging.getLogger('matplotlib.font_manager').disabled = True
    # route warnings through the logging system
    logging.captureWarnings(True)
    if opt.plots:
        import matplotlib
        matplotlib.use('Agg')
        import pylab as plt
        plt.figure(figsize=(12,9))
        plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.93,
                            hspace=0.2, wspace=0.05)

    if rgb_stretch is not None:
        import legacypipe.survey
        legacypipe.survey.rgb_stretch_factor = rgb_stretch

    kwargs.update(prereqs_update={ 'deep_preprocess': None,
                                   'refs': 'deep_preprocess',
                                   'deep_preprocess_2': 'refs',
                                   'deep_preprocess_3': 'deep_preprocess_2',
                                   })
    from astrometry.util.stages import CallGlobalTime
    stagefunc = CallGlobalTime('stage_%s', globals())
    kwargs.update(stagefunc=stagefunc)

    debug('kwargs:', kwargs)

    rtn = -1
    try:
        run_brick(opt.brick, survey, **kwargs)
        rtn = 0
    except NothingToDoError as e:
        print()
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        print()
        rtn = 0
    except RunbrickError as e:
        print()
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        print()
        rtn = -1

    return rtn

if __name__ == '__main__':
    from astrometry.util.ttime import MemMeas
    Time.add_measurement(MemMeas)
    sys.exit(main())
