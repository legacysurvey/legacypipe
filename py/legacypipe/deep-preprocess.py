import sys
import os
import warnings

import numpy as np

import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.ttime import Time

from legacypipe.runbrick import get_parser, get_runbrick_kwargs, run_brick
from legacypipe.survey import imsave_jpeg
from legacypipe.coadds import make_coadds
from legacypipe.outliers import patch_from_coadd, mask_outlier_pixels, read_outlier_mask_file

import logging
logger = logging.getLogger('legacypipe.deep-preprocess')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

def formatwarning(message, category, filename, lineno, line=None):
    return 'Warning: %s' % (message)
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
        **kwargs):
    from legacypipe.survey import (
        get_git_version, get_version_header, get_dependency_versions,
        wcs_for_brick, read_one_tim)
    from legacypipe.depthcut import make_depth_cut

    from astrometry.util.starutil_numpy import ra2hmsstring, dec2dmsstring

    tlast = Time()
    record_event and record_event('stage_deep: starting')
    assert(survey is not None)

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
    # Add NOAO-requested headers
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

    keep,_ = make_depth_cut(survey, ccds, bands, targetrd, brick, W, H, pixscale,
                            plots, ps, splinesky, gaussPsf, pixPsf, normalizePsf,
                            do_calibs, gitver, targetwcs, old_calibs_ok)
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
    make_badcoadds = True
    badcoaddspos, badcoaddsneg = mask_outlier_pixels(survey, deeptims, bands, targetwcs, brickname, version_header,
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

        **kwargs):

    # Using 'deeptims', create template coadds for outlier detection, AND deep coadds.

    # coadds for patching images...
    info('Making coadds for patching images...')
    deepC = make_coadds(deeptims, bands, targetwcs, mp=mp, sbscale=False,
                        allmasks=False, coweights=False)

    with survey.write_output('outliers_mask', brick=brickname) as out:
        # empty Primary HDU
        out.fits.write(None, header=version_header)

        for iband,band in enumerate(bands):
            btims = [tim for tim in deeptims if tim.band == band]
            if len(btims) == 0:
                continue
            info(len(btims), 'deep images for band', band)

            info('Making blurred reference image...')
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

            results = mp.imap_unordered(
                blur_resample_one, [(i_btim,tim,sig,targetwcs)
                                    for i_btim,(tim,sig) in enumerate(zip(btims,addsigs))])
            for i_btim,r in results:
                if r is None:
                    continue
                Yo,Xo,iacc,wacc,macc = r
                coimg[Yo,Xo] += iacc
                cow  [Yo,Xo] += wacc
                masks[Yo,Xo] |= macc
                del Yo,Xo,iacc,wacc,macc
                del r
            del results

            deep_sig = targetsig

            #
            veto = np.logical_or(star_veto,
                                 np.logical_or(
                binary_dilation(masks & DQ_BITS['bleed'], iterations=3),
                binary_dilation(masks & DQ_BITS['satur'], iterations=10)))
            del masks

            bims = [im for im in ims if im.band == band and not im in deepims]
            info('Running on', len(bims), 'individual images (not in deep set)...')
            if len(bims) == 0:
                continue

            patch_img = deepC.coimgs[iband]

            tim_kwargs = dict(gaussPsf=gaussPsf, pixPsf=pixPsf,
                              hybridPsf=hybridPsf, normalizePsf=normalizePsf,
                              subsky=subsky,
                              apodize=apodize,
                              constant_invvar=constant_invvar,
                              old_calibs_ok=old_calibs_ok)
            
            make_badcoadds=True
            R = mp.imap_unordered(
                mask_and_coadd_one, [(i_bim, survey, targetrd, tim_kwargs, im, targetwcs,
                                      patch_img, coimg, cow, veto, deep_sig, plots,ps)
                                     for i_bim,im in enumerate(bims)])
            del coimg, cow, veto


            #####


            
            badcoadd_pos = None
            badcoadd_neg = None
            if make_badcoadds:
                badcoadd_pos = np.zeros((H,W), np.float32)
                badcon_pos   = np.zeros((H,W), np.int16)
                badcoadd_neg = np.zeros((H,W), np.float32)
                badcon_neg   = np.zeros((H,W), np.int16)

            for i_btim,r in R:
                tim = btims[i_btim]
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
                del r

                # Apply the mask!
                maskbits = get_bits_to_mask()
                tim.inverr[(mask & maskbits) > 0] = 0.
                tim.dq[(mask & maskbits) > 0] |= tim.dq_type(DQ_BITS['outlier'])

                # Write output!
                from legacypipe.utils import copy_header_with_wcs
                hdr = copy_header_with_wcs(None, tim.subwcs)
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
            del R

            if make_badcoadds:
                badcoadd_pos /= np.maximum(badcon_pos, 1)
                badcoadd_neg /= np.maximum(badcon_neg, 1)
                badcoadds_pos.append(badcoadd_pos)
                badcoadds_neg.append(badcoadd_neg)


        
    
    
    
    return None

def mask_and_coadd_one(X):
    from legacypipe.survey import read_one_tim
    from scipy.ndimage.filters import gaussian_filter
    from astrometry.util.resample import resample_with_wcs,OverlapError

    (i_bim, survey, targetrd, tim_kwargs, im, targetwcs, patchimg, coimg, cow, veto,
     deep_sig, plots, ps) = X

    # - read tim
    tim = read_one_tim((im, targetrd, tim_kwargs))

    # - patch image
    patch_from_coadd([patchimg], targetwcs, ['x'], [tim])

    # - compute blurring sigmas for image & reference
    targetsig = max(deep_sig, tim.psf_sigma + 0.5)
    deep_blursig = np.sqrt(targetsig**2 - deep_sig**2)
    tim_blursig  = np.sqrt(targetsig**2 - tim.psf_sigma**2)
    
    # - blur & resample for masking coadd
    blurimg = gaussian_filter(tim.getImage(), tim_blursig)
    try:
        Yo,Xo,Yi,Xi,[rimg] = resample_with_wcs(
            targetwcs, tim.subwcs, [blurimg], intType=np.int16)
    except OverlapError:
        return i_bim, None
    del blurimg
    blurnorm = 1./(2. * np.sqrt(np.pi) * tim_blursig)
    wt = tim.getInvvar()[Yi,Xi] / (blurnorm**2)
    this_sig1 = 1./np.sqrt(np.median(wt[wt>0]))

    #return i_tim, (Yo, Xo, rimg*wt, wt, tim.dq[Yi,Xi])

    refimg = gaussian_filter(coimg, deep_blursig)[Yo,Xo]
    refblurnorm = 1./(2. * np.sqrt(np.pi) * deep_blursig)
    refwt = cow[Yo,Xo] / (refblurnorm**2)

    # Compare against reference image...
    maskedpix = np.zeros(tim.shape, np.uint8)

    # Compute the error on our estimate of (thisimg - co) =
    # sum in quadrature of the errors on thisimg and co.
    with np.errstate(divide='ignore'):
        diffvar = 1./wt + 1./refwt
        sndiff = (rimg - refimg) / np.sqrt(diffvar)
    with np.errstate(divide='ignore'):
        reldiff = ((rimg - refimg) / np.maximum(refimg, this_sig1))

    # Significant pixels
    hotpix = ((sndiff > 5.) * (reldiff > 2.) *
              (refwt > 1e-16) * (wt > 0.) *
              (veto[Yo,Xo] == False))
    coldpix = ((sndiff < -5.) * (reldiff < -2.) *
               (refwt > 1e-16) * (wt > 0.) *
               (veto[Yo,Xo] == False))
    del reldiff, refwt

    if np.any(hotpix) or np.any(coldpix):
        hot = np.zeros((H,W), bool)
        hot[Yo,Xo] = hotpix
        cold = np.zeros((H,W), bool)
        cold[Yo,Xo] = coldpix
        del hotpix, coldpix
        snmap = np.zeros((H,W), np.float32)
        snmap[Yo,Xo] = sndiff
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

        bad, = np.nonzero(hot[Yo,Xo])
        badhot = (Yo[bad], Xo[bad], tim.getImage()[Yi[bad],Xi[bad]])
        bad, = np.nonzero(cold[Yo,Xo])
        badcold = (Yo[bad], Xo[bad], tim.getImage()[Yi[bad],Xi[bad]])
        badco = badhot,badcold

        # Actually do the masking!
        # Resample "hot" (in brick coords) back to tim coords.
        try:
            mYo,mXo,mYi,mXi,_ = resample_with_wcs(
                tim.subwcs, targetwcs, intType=np.int16)
        except OverlapError:
            pass
        Ibad, = np.nonzero(hot[mYi,mXi])
        Ibad2, = np.nonzero(cold[mYi,mXi])
        info(tim, ': masking', len(Ibad), 'positive outlier pixels and', len(Ibad2), 'negative outlier pixels')
        maskedpix[mYo[Ibad],  mXo[Ibad]]  = OUTLIER_POS
        maskedpix[mYo[Ibad2], mXo[Ibad2]] = OUTLIER_NEG

    # - (patch again?)
    # - resample for regular coadd
    # - return: outlier mask, Yo,Xo,im,wts for coadd, Yo,Xo,im for badcoadd


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

    parser.set_defaults(stage=['deep_preprocess_2'])

    opt = parser.parse_args(args=args)
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

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
                                   'deep_preprocess_2': 'deep_preprocess' })
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


# 0436m002
