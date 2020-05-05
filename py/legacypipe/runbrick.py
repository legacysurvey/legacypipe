'''
Main "pipeline" script for the Legacy Survey (DECaLS, MzLS, BASS)
data reductions.

For calling from other scripts, see:

- :py:func:`run_brick`

Or for much more fine-grained control, see the individual stages:

- :py:func:`stage_tims`
- :py:func:`stage_refs`
- :py:func:`stage_outliers`
- :py:func:`stage_halos`
- :py:func:`stage_fit_on_coadds [optional]`
- :py:func:`stage_image_coadds`
- :py:func:`stage_srcs`
- :py:func:`stage_fitblobs`
- :py:func:`stage_coadds`
- :py:func:`stage_wise_forced`
- :py:func:`stage_writecat`

To see the code we run on each "blob" of pixels, see "oneblob.py".

- :py:func:`one_blob`

'''
from __future__ import print_function
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import sys
import os

import pylab as plt
import numpy as np

import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.plotutils import dimshow
from astrometry.util.ttime import Time

from legacypipe.survey import get_rgb, imsave_jpeg
from legacypipe.bits import DQ_BITS, MASKBITS
from legacypipe.utils import RunbrickError, NothingToDoError, iterwrapper, find_unique_pixels
from legacypipe.coadds import make_coadds, write_coadd_images, quick_coadds

from legacypipe.fit_on_coadds import stage_fit_on_coadds

import logging
logger = logging.getLogger('legacypipe.runbrick')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

def runbrick_global_init():
    from tractor.galaxy import disable_galaxy_cache
    info('Starting process', os.getpid(), Time()-Time())
    disable_galaxy_cache()

def stage_tims(W=3600, H=3600, pixscale=0.262, brickname=None,
               survey=None,
               survey_blob_mask=None,
               ra=None, dec=None,
               release=None,
               plots=False, ps=None,
               target_extent=None, program_name='runbrick.py',
               bands=None,
               do_calibs=True,
               old_calibs_ok=True,
               splinesky=True,
               subsky=True,
               gaussPsf=False, pixPsf=False, hybridPsf=False,
               normalizePsf=False,
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
               command_line=None,
               **kwargs):
    '''
    This is the first stage in the pipeline.  It
    determines which CCD images overlap the brick or region of
    interest, runs calibrations for those images if necessary, and
    then reads the images, creating `tractor.Image` ("tractor image"
    or "tim") objects for them.

    PSF options:

    - *gaussPsf*: boolean.  Single-component circular Gaussian, with
      width set from the header FWHM value.  Useful for quick
      debugging.

    - *pixPsf*: boolean.  Pixelized PsfEx model.

    - *hybridPsf*: boolean.  Hybrid Pixelized PsfEx / Gaussian approx model.

    Sky:

    - *splinesky*: boolean.  If we have to create sky calibs, create SplineSky model rather than ConstantSky?
    - *subsky*: boolean.  Subtract sky model from tims?

    '''
    from legacypipe.survey import (
        get_git_version, get_version_header, get_dependency_versions,
        wcs_for_brick, read_one_tim)
    from astrometry.util.starutil_numpy import ra2hmsstring, dec2dmsstring

    tlast = Time()
    record_event and record_event('stage_tims: starting')

    assert(survey is not None)

    if bands is None:
        bands = ['g','r','z']

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

    deps = get_dependency_versions(unwise_dir, unwise_tr_dir, unwise_modelsky_dir)
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

    if 'ccd_cuts' in ccds.get_columns():
        ccds.cut(ccds.ccd_cuts == 0)
        debug(len(ccds), 'CCDs survive cuts')
    else:
        print('WARNING: not applying CCD cuts')

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
        info('  ', os.path.basename(im.imgfn), im, im.band, 'exptime', im.exptime, 'propid', ccd.propid,
              'seeing %.2f' % (ccd.fwhm*im.pixscale), 'MJD %.3f' % ccd.mjd_obs,
              'object', getattr(ccd, 'object', '').strip())

    tnow = Time()
    debug('[serial tims] Finding images touching brick:', tnow-tlast)
    tlast = tnow

    if do_calibs:
        from legacypipe.survey import run_calibs
        record_event and record_event('stage_tims: starting calibs')
        kwa = dict(git_version=gitver, survey=survey,
                   old_calibs_ok=old_calibs_ok,
                   survey_blob_mask=survey_blob_mask)
        if gaussPsf:
            kwa.update(psfex=False)
        if splinesky:
            kwa.update(splinesky=True)
        if not gaia_stars:
            kwa.update(gaia=False)

        # Run calibrations
        args = [(im, kwa) for im in ims]
        mp.map(run_calibs, args)
        tnow = Time()
        debug('[parallel tims] Calibrations:', tnow-tlast)
        tlast = tnow

    # Read Tractor images
    args = [(im, targetrd, dict(gaussPsf=gaussPsf, pixPsf=pixPsf,
                                hybridPsf=hybridPsf, normalizePsf=normalizePsf,
                                subsky=subsky,
                                apodize=apodize,
                                constant_invvar=constant_invvar,
                                pixels=read_image_pixels,
                                old_calibs_ok=old_calibs_ok))
                                for im in ims]
    record_event and record_event('stage_tims: starting read_tims')
    tims = list(mp.map(read_one_tim, args))
    record_event and record_event('stage_tims: done read_tims')

    tnow = Time()
    debug('[parallel tims] Read', len(ccds), 'images:', tnow-tlast)
    tlast = tnow

    # Cut the table of CCDs to match the 'tims' list
    I = np.array([i for i,tim in enumerate(tims) if tim is not None])
    ccds.cut(I)
    tims = [tim for tim in tims if tim is not None]
    assert(len(ccds) == len(tims))
    if len(tims) == 0:
        raise NothingToDoError('No photometric CCDs touching brick.')

    # Check calibration product versions
    for tim in tims:
        for cal,ver in [('sky', tim.skyver), ('psf', tim.psfver)]:
            if tim.plver.strip() != ver[1].strip():
                print(('Warning: image "%s" PLVER is "%s" but %s calib was run'
                      +' on PLVER "%s"') % (str(tim), tim.plver, cal, ver[1]))

    # Add additional columns to the CCDs table.
    ccds.ccd_x0 = np.array([tim.x0 for tim in tims]).astype(np.int16)
    ccds.ccd_y0 = np.array([tim.y0 for tim in tims]).astype(np.int16)
    ccds.ccd_x1 = np.array([tim.x0 + tim.shape[1]
                            for tim in tims]).astype(np.int16)
    ccds.ccd_y1 = np.array([tim.y0 + tim.shape[0]
                            for tim in tims]).astype(np.int16)
    rd = np.array([[tim.subwcs.pixelxy2radec(1, 1)[-2:],
                    tim.subwcs.pixelxy2radec(1, y1-y0)[-2:],
                    tim.subwcs.pixelxy2radec(x1-x0, 1)[-2:],
                    tim.subwcs.pixelxy2radec(x1-x0, y1-y0)[-2:]]
                    for tim,x0,y0,x1,y1 in
                    zip(tims, ccds.ccd_x0+1, ccds.ccd_y0+1,
                        ccds.ccd_x1, ccds.ccd_y1)])
    _,x,y = targetwcs.radec2pixelxy(rd[:,:,0], rd[:,:,1])
    ccds.brick_x0 = np.floor(np.min(x, axis=1)).astype(np.int16)
    ccds.brick_x1 = np.ceil (np.max(x, axis=1)).astype(np.int16)
    ccds.brick_y0 = np.floor(np.min(y, axis=1)).astype(np.int16)
    ccds.brick_y1 = np.ceil (np.max(y, axis=1)).astype(np.int16)
    ccds.psfnorm = np.array([tim.psfnorm for tim in tims])
    ccds.galnorm = np.array([tim.galnorm for tim in tims])
    ccds.propid = np.array([tim.propid for tim in tims])
    ccds.plver  = np.array([tim.plver for tim in tims])
    ccds.skyver = np.array([tim.skyver[0] for tim in tims])
    ccds.psfver = np.array([tim.psfver[0] for tim in tims])
    ccds.skyplver = np.array([tim.skyver[1] for tim in tims])
    ccds.psfplver = np.array([tim.psfver[1] for tim in tims])

    # Cut "bands" down to just the bands for which we have images.
    timbands = [tim.band for tim in tims]
    bands = [b for b in bands if b in timbands]
    debug('Cut bands to', bands)

    if plots:
        from legacypipe.runbrick_plots import tim_plots
        tim_plots(tims, bands, ps)

    # Add header cards about which bands and cameras are involved.
    for band in 'grz':
        hasit = band in bands
        version_header.add_record(dict(
            name='BRICK_%s' % band.upper(), value=hasit,
            comment='Does band %s touch this brick?' % band))

        cams = np.unique([tim.imobj.camera for tim in tims
                          if tim.band == band])
        version_header.add_record(dict(
            name='CAMS_%s' % band.upper(), value=' '.join(cams),
            comment='Cameras contributing band %s' % band))
    version_header.add_record(dict(name='BANDS', value=''.join(bands),
                                   comment='Bands touching this brick'))
    version_header.add_record(dict(name='NBANDS', value=len(bands),
                                   comment='Number of bands in this catalog'))
    for i,band in enumerate(bands):
        version_header.add_record(dict(name='BAND%i' % i, value=band,
                                       comment='Band name in this catalog'))

    keys = ['version_header', 'targetrd', 'pixscale', 'targetwcs', 'W','H',
            'bands', 'tims', 'ps', 'brickid', 'brickname', 'brick', 'custom_brick',
            'target_extent', 'ccds', 'bands', 'survey']
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def _add_stage_version(version_header, short, stagename):
    from legacypipe.survey import get_git_version
    version_header.add_record(dict(name='VER_%s'%short, value=get_git_version(),
                                   help='legacypipe version for stage_%s'%stagename))

def stage_refs(survey=None,
               brick=None,
               brickname=None,
               brickid=None,
               pixscale=None,
               targetwcs=None,
               bands=None,
               version_header=None,
               tycho_stars=True,
               gaia_stars=True,
               large_galaxies=True,
               star_clusters=True,
               record_event=None,
               **kwargs):
    from legacypipe.reference import get_reference_sources

    record_event and record_event('stage_refs: starting')
    _add_stage_version(version_header, 'REFS', 'refs')
    refstars,refcat = get_reference_sources(survey, targetwcs, pixscale, bands,
                                            tycho_stars=tycho_stars,
                                            gaia_stars=gaia_stars,
                                            large_galaxies=large_galaxies,
                                            star_clusters=star_clusters)
    # "refstars" is a table
    # "refcat" is a list of tractor Sources
    # They are aligned
    T_donotfit = None
    T_clusters = None
    if refstars:
        assert(len(refstars) == len(refcat))
        # Pull out reference sources flagged do-not-fit; we add them
        # back in (much) later.  These are Gaia sources near the
        # centers of LSLGA large galaxies, so we want to propagate the
        # Gaia catalog information, but don't want to fit them.
        I, = np.nonzero(refstars.donotfit)
        if len(I):
            T_donotfit = refstars[I]
            I, = np.nonzero(np.logical_not(refstars.donotfit))
            refstars.cut(I)
            refcat = [refcat[i] for i in I]
            assert(len(refstars) == len(refcat))
        # Pull out star clusters too.
        I, = np.nonzero(refstars.iscluster)
        if len(I):
            T_clusters = refstars[I]
            I, = np.nonzero(np.logical_not(refstars.iscluster))
            refstars.cut(I)
            refcat = [refcat[i] for i in I]
            assert(len(refstars) == len(refcat))
        del I

    if refstars or T_donotfit or T_clusters:
        allrefs = merge_tables([t for t in [refstars, T_donotfit, T_clusters] if t],
                               columns='fillzero')
        with survey.write_output('ref-sources', brick=brickname) as out:
            allrefs.writeto(None, fits_object=out.fits, primheader=version_header)
        del allrefs

    if T_donotfit:
        # add columns for later...
        if not 'type' in T_donotfit.get_columns():
            T_donotfit.type = np.array(['DUP']*len(T_donotfit))
        else:
            for i in range(len(T_donotfit)):
                if len(T_donotfit.type[i].strip()) == 0:
                    T_donotfit.type[i] = 'DUP'
        T_donotfit.brickid = np.zeros(len(T_donotfit), np.int32) + brickid
        T_donotfit.brickname = np.array([brickname] * len(T_donotfit))

    keys = ['refstars', 'gaia_stars', 'T_donotfit', 'T_clusters', 'version_header',
            'refcat']
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def stage_outliers(tims=None, targetwcs=None, W=None, H=None, bands=None,
                   mp=None, nsigma=None, plots=None, ps=None, record_event=None,
                   survey=None, brickname=None, version_header=None,
                   refstars=None, outlier_mask_file=None,
                   outliers=True,
                   **kwargs):
    '''
    This pipeline stage tries to detect artifacts in the individual
    exposures, by blurring all images in the same band to the same PSF size,
    then searching for outliers.
    '''
    from legacypipe.outliers import patch_from_coadd, mask_outlier_pixels, read_outlier_mask_file

    record_event and record_event('stage_outliers: starting')
    _add_stage_version(version_header, 'OUTL', 'outliers')

    version_header.add_record(dict(name='OUTLIER',
                                   value=outliers,
                                   help='Are we applying outlier rejection?'))

    # Check for existing MEF containing masks for all the chips we need.
    if outliers and not read_outlier_mask_file(survey, tims, brickname, outlier_mask_file=outlier_mask_file):
        # Make before-n-after plots (before)
        C = make_coadds(tims, bands, targetwcs, mp=mp, sbscale=False)
        with survey.write_output('outliers-pre', brick=brickname) as out:
            imsave_jpeg(out.fn, get_rgb(C.coimgs, bands), origin='lower')

        # Patch individual-CCD masked pixels from a coadd
        patch_from_coadd(C.coimgs, targetwcs, bands, tims, mp=mp)
        del C

        make_badcoadds = True
        badcoaddspos, badcoaddsneg = mask_outlier_pixels(survey, tims, bands, targetwcs, brickname, version_header,
                                                         mp=mp, plots=plots, ps=ps, make_badcoadds=make_badcoadds,
                                                         refstars=refstars)

        # Make before-n-after plots (after)
        C = make_coadds(tims, bands, targetwcs, mp=mp, sbscale=False)
        with survey.write_output('outliers-post', brick=brickname) as out:
            imsave_jpeg(out.fn, get_rgb(C.coimgs, bands), origin='lower')
        with survey.write_output('outliers-masked-pos', brick=brickname) as out:
            imsave_jpeg(out.fn, get_rgb(badcoaddspos, bands), origin='lower')
        with survey.write_output('outliers-masked-neg', brick=brickname) as out:
            imsave_jpeg(out.fn, get_rgb(badcoaddsneg, bands), origin='lower')

    return dict(tims=tims, version_header=version_header)

def stage_halos(targetrd=None, pixscale=None, targetwcs=None,
                W=None,H=None,
                bands=None, ps=None, tims=None,
                plots=False, plots2=False,
                brickname=None,
                version_header=None,
                mp=None, nsigma=None,
                survey=None, brick=None,
                refstars=None,
                star_halos=True,
                record_event=None,
                **kwargs):
    record_event and record_event('stage_halos: starting')
    _add_stage_version(version_header, 'HALO', 'halos')

    # Subtract star halos?
    if star_halos and refstars:
        Igaia = []
        gaia = refstars
        Igaia, = np.nonzero(refstars.isgaia * refstars.pointsource)
        debug(len(Igaia), 'stars for halo subtraction')
        if len(Igaia):
            from legacypipe.halos import subtract_halos
            halostars = gaia[Igaia]

            if plots:
                from legacypipe.runbrick_plots import halo_plots_before, halo_plots_after
                coimgs = halo_plots_before(tims, bands, targetwcs, halostars, ps)

            subtract_halos(tims, halostars, bands, mp, plots, ps)

            if plots:
                halo_plots_after(tims, bands, targetwcs, halostars, coimgs, ps)

    return dict(tims=tims, version_header=version_header)

def stage_image_coadds(survey=None, targetwcs=None, bands=None, tims=None,
                       brickname=None, version_header=None,
                       plots=False, ps=None, coadd_bw=False, W=None, H=None,
                       brick=None, blobs=None, lanczos=True, ccds=None,
                       write_metrics=True,
                       mp=None, record_event=None,
                       co_sky=None,
                       **kwargs):
    record_event and record_event('stage_image_coadds: starting')
    '''
    Immediately after reading the images, we can create coadds of just
    the image products.  Later, full coadds including the models will
    be created (in `stage_coadds`).  But it's handy to have the coadds
    early on, to diagnose problems or just to look at the data.
    '''
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=version_header)

    C = make_coadds(tims, bands, targetwcs,
                    detmaps=True, ngood=True, lanczos=lanczos,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, tims,
                                   targetwcs, co_sky),
                    mp=mp, plots=plots, ps=ps)

    # Sims: coadds of galaxy sims only, image only
    if hasattr(tims[0], 'sims_image'):
        sims_coadd,_ = quick_coadds(
            tims, bands, targetwcs, images=[tim.sims_image for tim in tims])

    D = _depth_histogram(brick, targetwcs, bands, C.psfdetivs, C.galdetivs)
    with survey.write_output('depth-table', brick=brickname) as out:
        D.writeto(None, fits_object=out.fits)
    del D

    # Write per-brick CCDs table
    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='ccdinfo',
                            comment='NOAO data product type'))
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=primhdr)

    coadd_list= [('image', C.coimgs)]
    if hasattr(tims[0], 'sims_image'):
        coadd_list.append(('simscoadd', sims_coadd))

    for name,ims in coadd_list:
        rgb = get_rgb(ims, bands)
        kwa = {}
        if coadd_bw and len(bands) == 1:
            rgb = rgb.sum(axis=2)
            kwa = dict(cmap='gray')

        with survey.write_output(name + '-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
            debug('Wrote', out.fn)

        # Blob-outlined version
        if blobs is not None:
            from scipy.ndimage.morphology import binary_dilation
            outline = np.logical_xor(
                binary_dilation(blobs >= 0, structure=np.ones((3,3))),
                (blobs >= 0))
            # coadd_bw
            if len(rgb.shape) == 2:
                rgb = np.repeat(rgb[:,:,np.newaxis], 3, axis=2)
            # Outline in green
            rgb[:,:,0][outline] = 0
            rgb[:,:,1][outline] = 1
            rgb[:,:,2][outline] = 0

            with survey.write_output(name+'blob-jpeg', brick=brickname) as out:
                imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
                debug('Wrote', out.fn)

            # write out blob map
            if write_metrics:
                # copy version_header before modifying it.
                hdr = fitsio.FITSHDR()
                for r in version_header.records():
                    hdr.add_record(r)
                # Plug the WCS header cards into these images
                targetwcs.add_to_header(hdr)
                hdr.delete('IMAGEW')
                hdr.delete('IMAGEH')
                hdr.add_record(dict(name='IMTYPE', value='blobmap',
                                    comment='LegacySurveys image type'))
                with survey.write_output('blobmap', brick=brickname,
                                         shape=blobs.shape) as out:
                    out.fits.write(blobs, header=hdr)
        del rgb
    return None

def stage_srcs(targetrd=None, pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               brickname=None,
               version_header=None,
               mp=None, nsigma=None,
               saddle_fraction=None,
               saddle_min=None,
               survey=None, brick=None,
               refcat=None, refstars=None,
               T_donotfit=None, T_clusters=None,
               ccds=None,
               record_event=None,
               large_galaxies=True,
               gaia_stars=True,
               **kwargs):
    '''
    In this stage we run SED-matched detection to find objects in the
    images.  For each object detected, a `tractor` source object is
    created, initially a `tractor.PointSource`.  In this stage, the
    sources are also split into "blobs" of overlapping pixels.  Each
    of these blobs will be processed independently.
    '''
    from functools import reduce
    from tractor import PointSource, NanoMaggies, Catalog
    from legacypipe.detection import (detection_maps,
                        run_sed_matched_filters, segment_and_group_sources)
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.measurements import label

    record_event and record_event('stage_srcs: starting')
    _add_stage_version(version_header, 'SRCS', 'srcs')
    tlast = Time()

    if refstars:
        # Don't detect new sources where we already have reference stars
        avoid_x = refstars.ibx
        avoid_y = refstars.iby
        # Add a ~1" exclusion zone around reference stars and large galaxies
        avoid_r = np.zeros_like(avoid_x) + 4
        if T_clusters is not None:
            info('Avoiding source detection in', len(T_clusters), 'CLUSTERs')
            if len(T_clusters):
                avoid_x = np.append(avoid_x, T_clusters.ibx)
                avoid_y = np.append(avoid_y, T_clusters.iby)
                avoid_r = np.append(avoid_r, T_clusters.radius_pix)
                debug('CLUSTER pixel radii:', T_clusters.radius_pix)
    else:
        avoid_x, avoid_y, avoid_r = np.array([]), np.array([]), np.array([])

    record_event and record_event('stage_srcs: detection maps')
    tnow = Time()
    debug('Rendering detection maps...')
    detmaps, detivs, satmaps = detection_maps(tims, targetwcs, bands, mp,
                                              apodize=10)
    tnow = Time()
    debug('[parallel srcs] Detmaps:', tnow-tlast)
    tlast = tnow
    record_event and record_event('stage_srcs: sources')

    # Expand the mask around saturated pixels to avoid generating
    # peaks at the edge of the mask.
    saturated_pix = [binary_dilation(satmap > 0, iterations=4) for satmap in satmaps]

    # Formerly, we generated sources for each saturated blob, but since we now initialize
    # with Tycho-2 and Gaia stars and large galaxies, not needed.

    if plots:
        from legacypipe.runbrick_plots import detection_plots
        detection_plots(detmaps, detivs, bands, saturated_pix, tims,
                        targetwcs, refstars, large_galaxies, gaia_stars, ps)

    # SED-matched detections
    record_event and record_event('stage_srcs: SED-matched')

    debug('Running source detection at', nsigma, 'sigma')
    SEDs = survey.sed_matched_filters(bands)

    kwa = {}
    if plots:
        coims,_ = quick_coadds(tims, bands, targetwcs)
        kwa.update(rgbimg=get_rgb(coims, bands))

    Tnew,newcat,hot = run_sed_matched_filters(
        SEDs, bands, detmaps, detivs, (avoid_x,avoid_y,avoid_r), targetwcs,
        nsigma=nsigma, saddle_fraction=saddle_fraction, saddle_min=saddle_min,
        saturated_pix=saturated_pix, plots=plots, ps=ps, mp=mp, **kwa)

    #if Tnew is None:
    #    raise NothingToDoError('No sources detected.')

    if Tnew is not None:
        assert(len(Tnew) == len(newcat))
        Tnew.delete_column('peaksn')
        Tnew.delete_column('apsn')
        Tnew.ref_cat = np.array(['  '] * len(Tnew))
        Tnew.ref_id  = np.zeros(len(Tnew), np.int64)
    del detmaps
    del detivs

    # Merge newly detected sources with reference sources (Tycho2, Gaia, large galaxies)
    cats = []
    tables = []
    if Tnew is not None:
        cats.extend(newcat)
        tables.append(Tnew)
    if refstars and len(refstars):
        cats.extend(refcat)
        tables.append(refstars)
    T = merge_tables(tables, columns='fillzero')
    cat = Catalog(*cats)
    cat.freezeAllParams()
    # The tractor Source object list "cat" and the table "T" are row-aligned.
    assert(len(T) > 0)
    assert(len(cat) == len(T))

    tnow = Time()
    debug('[serial srcs] Peaks:', tnow-tlast)
    tlast = tnow

    if plots:
        from legacypipe.runbrick_plots import detection_plots_2
        detection_plots_2(tims, bands, targetwcs, refstars, Tnew, hot,
                          saturated_pix, ps)

    # Find "hot" pixels that are separated by masked pixels,
    # to connect blobs across, eg, bleed trails and saturated cores.
    if True:
        from scipy.ndimage.measurements import find_objects
        any_saturated = reduce(np.logical_or, saturated_pix)
        merging = np.zeros_like(any_saturated)
        _,w = any_saturated.shape
        # All our cameras have bleed trails that go along image rows.
        # We go column by column, checking whether blobs of "hot" pixels
        # get joined up when merged with SATUR pixels.
        for i in range(w):
            col = hot[:,i]
            cblobs,nc = label(col)
            col = np.logical_or(col, any_saturated[:,i])
            cblobs2,nc2 = label(col)
            if nc2 < nc:
                # at least one pair of blobs merged together
                # Find merged blobs:
                # "cblobs2" is a map from pixels to merged blob number.
                # look up which merged blob each un-merged blob belongs to.
                slcs = find_objects(cblobs)
                from collections import Counter
                counts = Counter()
                for slc in slcs:
                    (slc,) = slc
                    mergedblob = cblobs2[slc.start]
                    counts[mergedblob] += 1
                slcs2 = find_objects(cblobs2)
                for blob,n in counts.most_common():
                    if n == 1:
                        break
                    (slc,) = slcs2[blob-1]
                    merging[slc, i] = True
        hot |= merging

        if plots:
            plt.clf()
            plt.subplot(1,2,1)
            dimshow((hot*1) + (any_saturated*1), vmin=0, vmax=2, cmap='hot')
            plt.title('hot + saturated')
            ps.savefig()
            plt.clf()
            plt.subplot(1,2,1)
            dimshow(merging, vmin=0, vmax=1, cmap='hot')
            plt.title('merging')
            plt.subplot(1,2,2)
            dimshow(np.logical_or(hot, merging), vmin=0, vmax=1, cmap='hot')
            plt.title('merged')
            ps.savefig()

        del merging, any_saturated

    # Segment, and record which sources fall into each blob
    blobs,blobsrcs,blobslices = segment_and_group_sources(hot, T, name=brickname,
                                                          ps=ps, plots=plots)
    del hot

    tnow = Time()
    debug('[serial srcs] Blobs:', tnow-tlast)
    tlast = tnow

    sky_overlap = True
    ccds.co_sky = np.zeros(len(ccds), np.float32)
    if sky_overlap:
        '''
        A note about units here: we're passing 'sbscale=False' to the
        coadd function, so images are *not* getting scaled to constant
        surface-brightness -- so you don't want to mix-and-match
        cameras with different pixel scales within a band!  We're
        estimating the sky level as a surface brightness, in
        nanomaggies per pixel of the CCDs.
        '''
        debug('Creating coadd for sky overlap...')
        C = make_coadds(tims, bands, targetwcs, mp=mp, sbscale=False)
        co_sky = {}
        for band,co,cowt in zip(bands, C.coimgs, C.cowimgs):
            pix = co[(cowt > 0) * (blobs == -1)]
            if len(pix) == 0:
                continue
            cosky = np.median(pix)
            info('Median coadd sky for', band, ':', cosky)
            co_sky[band] = cosky
            for itim,tim in enumerate(tims):
                if tim.band != band:
                    continue
                tim.data -= cosky
                ccds.co_sky[itim] = cosky
    else:
        co_sky = None

    keys = ['T', 'tims', 'blobsrcs', 'blobslices', 'blobs', 'cat',
            'ps', 'saturated_pix', 'version_header', 'co_sky', 'ccds']
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def stage_fitblobs(T=None,
                   T_clusters=None,
                   T_donotfit=None,
                   brickname=None,
                   brickid=None,
                   brick=None,
                   version_header=None,
                   blobsrcs=None, blobslices=None, blobs=None,
                   cat=None,
                   targetwcs=None,
                   W=None,H=None,
                   bands=None, ps=None, tims=None,
                   survey=None,
                   plots=False, plots2=False,
                   nblobs=None, blob0=None, blobxy=None,
                   blobradec=None, blobid=None,
                   max_blobsize=None,
                   reoptimize=False,
                   iterative=False,
                   large_galaxies_force_pointsource=True,
                   less_masking=False,
                   use_ceres=True, mp=None,
                   checkpoint_filename=None,
                   checkpoint_period=600,
                   write_pickle_filename=None,
                   write_metrics=True,
                   get_all_models=False,
                   refstars=None,
                   bailout=False,
                   record_event=None,
                   custom_brick=False,
                   **kwargs):
    '''
    This is where the actual source fitting happens.
    The `one_blob` function is called for each "blob" of pixels with
    the sources contained within that blob.
    '''
    from tractor import Catalog

    record_event and record_event('stage_fitblobs: starting')
    _add_stage_version(version_header, 'FITB', 'fitblobs')
    tlast = Time()

    version_header.add_record(dict(name='GALFRPSF',
                                   value=large_galaxies_force_pointsource,
                                   help='Large galaxies force PSF?'))
    version_header.add_record(dict(name='LESSMASK',
                                   value=less_masking,
                                   help='Reduce masking behaviors?'))
    if plots:
        from legacypipe.runbrick_plots import fitblobs_plots
        fitblobs_plots(tims, bands, targetwcs, blobslices, blobsrcs, cat,
                       blobs, ps)

    T.orig_ra  = T.ra.copy()
    T.orig_dec = T.dec.copy()

    tnow = Time()
    debug('[serial fitblobs]:', tnow-tlast)
    tlast = tnow

    # Were we asked to only run a subset of blobs?
    keepblobs = None
    if blobradec is not None:
        # blobradec is a list like [(ra0,dec0), ...]
        rd = np.array(blobradec)
        _,x,y = targetwcs.radec2pixelxy(rd[:,0], rd[:,1])
        x = (x - 1).astype(int)
        y = (y - 1).astype(int)
        blobxy = list(zip(x, y))

    if blobxy is not None:
        # blobxy is a list like [(x0,y0), (x1,y1), ...]
        keepblobs = []
        for x,y in blobxy:
            x,y = int(x), int(y)
            if x < 0 or x >= W or y < 0 or y >= H:
                print('Warning: clipping blob x,y to brick bounds', x,y)
                x = np.clip(x, 0, W-1)
                y = np.clip(y, 0, H-1)
            blob = blobs[y,x]
            if blob >= 0:
                keepblobs.append(blob)
            else:
                print('WARNING: blobxy', x,y, 'is not in a blob!')
        keepblobs = np.unique(keepblobs)

    if blobid is not None:
        # comma-separated list of blob id numbers.
        keepblobs = np.array([int(b) for b in blobid.split(',')])

    if blob0 is not None or (nblobs is not None and nblobs < len(blobslices)):
        if blob0 is None:
            blob0 = 0
        if nblobs is None:
            nblobs = len(blobslices) - blob0
        keepblobs = np.arange(blob0, blob0+nblobs)

    # keepblobs can be None or empty list
    if keepblobs is not None and len(keepblobs):
        # 'blobs' is an image with values -1 for no blob, or the index
        # of the blob.  Create a map from old 'blob number+1' to new
        # 'blob number', keeping only blobs in the 'keepblobs' list.
        # The +1 is so that -1 is a valid index in the mapping.
        NB = len(blobslices)
        blobmap = np.empty(NB+1, int)
        blobmap[:] = -1
        blobmap[keepblobs + 1] = np.arange(len(keepblobs))
        # apply the map!
        blobs = blobmap[blobs + 1]
        # 'blobslices' and 'blobsrcs' are lists where the index
        # corresponds to the value in the 'blobs' map.
        blobslices = [blobslices[i] for i in keepblobs]
        blobsrcs   = [blobsrcs  [i] for i in keepblobs]
        # one more place where blob numbers are recorded...
        T.blob = blobs[np.clip(T.iby, 0, H-1), np.clip(T.ibx, 0, W-1)]

    # drop any cached data before we start pickling/multiprocessing
    survey.drop_cache()

    if plots and refstars:
        from legacypipe.runbrick_plots import fitblobs_plots_2
        fitblobs_plots_2(blobs, refstars, ps)

    skipblobs = []
    R = []
    # Check for existing checkpoint file.
    if checkpoint_filename and os.path.exists(checkpoint_filename):
        from astrometry.util.file import unpickle_from_file
        info('Reading', checkpoint_filename)
        try:
            R = unpickle_from_file(checkpoint_filename)
            debug('Read', len(R), 'results from checkpoint file', checkpoint_filename)
        except:
            import traceback
            print('Failed to read checkpoint file ' + checkpoint_filename)
            traceback.print_exc()
        keepR = _check_checkpoints(R, blobslices, brickname)
        info('Keeping', len(keepR), 'of', len(R), 'checkpointed results')
        R = keepR
        skipblobs = [r['iblob'] for r in R]

    bailout_mask = None
    T_refbail = None
    if bailout:
        bailout_mask = _get_bailout_mask(blobs, skipblobs, targetwcs, W, H, brick,
                                         blobslices)
        # skip all blobs!
        new_skipblobs = np.unique(blobs[blobs>=0])
        # Which blobs are we bailing out on?
        bailing = set(new_skipblobs) - set(skipblobs)
        info('Bailing out on blobs:', bailing)
        if len(bailing):
            Ibail = np.hstack([blobsrcs[b] for b in bailing])
            # Find reference sources in bailout blobs
            Irefbail = []
            for i in Ibail:
                if getattr(cat[i], 'is_reference_source', False):
                    Irefbail.append(i)
            if len(Irefbail):
                from legacypipe.catalog import _get_tractor_fits_values
                from legacypipe.oneblob import _convert_ellipses
                T_refbail = T[np.array(Irefbail)]
                cat_refbail = [cat[i] for i in Irefbail]
                # For LSLGA sources
                for src in cat_refbail:
                    _convert_ellipses(src)
                # Sets TYPE, etc for T_refbail table.
                _get_tractor_fits_values(T_refbail, cat_refbail, '%s')

            if T_refbail is not None:
                info('Found', len(T_refbail), 'reference sources in bail-out blobs')

        skipblobs = new_skipblobs
        # append empty results so that a later assert on the lengths will pass
        while len(R) < len(blobsrcs):
            R.append(dict(brickname=brickname, iblob=-1, result=None))

    refmap = get_blobiter_ref_map(refstars, T_clusters, less_masking, targetwcs)
    # Create the iterator over blobs to process
    blobiter = _blob_iter(brickname, blobslices, blobsrcs, blobs, targetwcs, tims,
                          cat, bands, plots, ps, reoptimize, iterative, use_ceres,
                          refmap, large_galaxies_force_pointsource, less_masking, brick,
                          skipblobs=skipblobs,
                          max_blobsize=max_blobsize, custom_brick=custom_brick)
    # to allow timingpool to queue tasks one at a time
    blobiter = iterwrapper(blobiter, len(blobsrcs))

    if checkpoint_filename is None:
        R = mp.map(_bounce_one_blob, blobiter)
    else:
        from astrometry.util.ttime import CpuMeas

        # Begin running one_blob on each blob...
        Riter = mp.imap_unordered(_bounce_one_blob, blobiter)
        # measure wall time and write out checkpoint file periodically.
        last_checkpoint = CpuMeas()
        n_finished = 0
        n_finished_total = 0
        while True:
            import multiprocessing
            # Time to write a checkpoint file? (And have something to write?)
            tnow = CpuMeas()
            dt = tnow.wall_seconds_since(last_checkpoint)
            if dt >= checkpoint_period and n_finished > 0:
                # Write checkpoint!
                debug('Writing', n_finished, 'new results; total for this run', n_finished_total)
                try:
                    _write_checkpoint(R, checkpoint_filename)
                    last_checkpoint = tnow
                    dt = 0.
                    n_finished = 0
                except:
                    print('Failed to rename checkpoint file', checkpoint_filename)
                    import traceback
                    traceback.print_exc()
            # Wait for results (with timeout)
            try:
                if mp.pool is not None:
                    timeout = max(1, checkpoint_period - dt)
                    r = Riter.next(timeout)
                else:
                    r = next(Riter)
                R.append(r)
                n_finished += 1
                n_finished_total += 1
            except StopIteration:
                debug('Done')
                break
            except multiprocessing.TimeoutError:
                # print('Timed out waiting for result')
                continue

        # Write checkpoint when done!
        _write_checkpoint(R, checkpoint_filename)

        debug('Got', n_finished_total, 'results; wrote', len(R), 'to checkpoint')

    debug('[parallel fitblobs] Fitting sources took:', Time()-tlast)

    # Repackage the results from one_blob...

    # one_blob can change the number and types of sources.
    # Reorder the sources:
    assert(len(R) == len(blobsrcs))
    # drop brickname,iblob
    R = [r['result'] for r in R]
    # Drop now-empty blobs.
    R = [r for r in R if r is not None and len(r)]
    if len(R) == 0:
        raise NothingToDoError('No sources passed significance tests.')
    # Sort results R by 'iblob'
    J = np.argsort([B.iblob for B in R])
    R = [R[j] for j in J]
    # Merge results R into one big table
    BB = merge_tables(R)
    del R
    # Pull out the source indices...
    II = BB.Isrcs
    newcat = BB.sources
    # ... and make the table T parallel with BB.
    # For iterative sources:
    n_iter = np.sum(II < 0)
    if n_iter:
        n_old = len(T)
        # first have to pad T with some new entries...
        Tnew = fits_table()
        Tnew.iterative = np.ones(n_iter, bool)
        Tnew.ref_cat = np.array(['  '] * len(Tnew))
        T = merge_tables([T, Tnew], columns='fillzero')
        # ... and then point II at them.
        II[II < 0] = n_old + np.arange(n_iter)
    else:
        T.iterative = np.zeros(len(T), bool)
    assert(np.all(II >= 0))
    assert(np.all(II < len(T)))
    assert(len(np.unique(II)) == len(II))
    T.cut(II)
    assert(len(T) == len(BB))
    del BB.Isrcs

    # so that iterative detections get a blob number.
    T.blob = BB.blob

    # Drop sources that exited the blob as a result of fitting.
    left_blob = np.logical_and(BB.started_in_blob,
                               np.logical_not(BB.finished_in_blob))
    I, = np.nonzero(np.logical_not(left_blob))
    if len(I) < len(BB):
        debug('Dropping', len(BB)-len(I), 'sources that exited their blobs during fitting')
        BB.cut(I)
        T.cut(I)
        newcat = [newcat[i] for i in I]
        assert(len(T) == len(BB))

    assert(len(T) == len(newcat))
    info('Old catalog:', len(cat))
    info('New catalog:', len(newcat))
    assert(len(newcat) > 0)
    cat = Catalog(*newcat)
    ns,nb = BB.fracflux.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = BB.fracmasked.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = BB.fracin.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = BB.rchisq.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = BB.dchisq.shape
    assert(ns == len(cat))
    assert(nb == 5) # psf, rex, dev, exp, ser

    # Renumber blobs to make them contiguous.
    oldblob = T.blob
    ublob,iblob = np.unique(T.blob, return_inverse=True)
    del ublob
    assert(len(iblob) == len(T))
    T.blob = iblob.astype(np.int32)

    # Build map from (old+1) to new blob numbers, for the blob image.
    blobmap = np.empty(blobs.max()+2, int)
    # make sure that dropped blobs -> -1
    blobmap[:] = -1
    # in particular,
    blobmap[0] = -1
    blobmap[oldblob + 1] = iblob
    blobs = blobmap[blobs+1]
    del blobmap

    # write out blob map
    if write_metrics:
        # copy version_header before modifying it.
        hdr = fitsio.FITSHDR()
        for r in version_header.records():
            hdr.add_record(r)
        # Plug the WCS header cards into these images
        targetwcs.add_to_header(hdr)
        hdr.delete('IMAGEW')
        hdr.delete('IMAGEH')
        hdr.add_record(dict(name='IMTYPE', value='blobmap',
                            comment='LegacySurveys image type'))
        hdr.add_record(dict(name='EQUINOX', value=2000.,
                            comment='Observation epoch'))
        with survey.write_output('blobmap', brick=brickname, shape=blobs.shape) as out:
            out.fits.write(blobs, header=hdr)
    del iblob, oldblob

    T.brickid = np.zeros(len(T), np.int32) + brickid
    T.brickname = np.array([brickname] * len(T))
    if len(T.brickname) == 0:
        T.brickname = T.brickname.astype('S8')
    T.objid = np.arange(len(T), dtype=np.int32)

    # How many sources in each blob?
    from collections import Counter
    ninblob = Counter(T.blob)
    T.ninblob = np.array([ninblob[b] for b in T.blob]).astype(np.int16)
    del ninblob

    # Copy blob results to table T
    for k in ['fracflux', 'fracin', 'fracmasked', 'rchisq',
              'cpu_arch', 'cpu_source', 'cpu_blob',
              'blob_width', 'blob_height', 'blob_npix',
              'blob_nimages', 'blob_totalpix',
              'blob_symm_width', 'blob_symm_height', 'blob_symm_npix',
              'blob_symm_nimages', 'brightblob', 'hit_limit', 'dchisq',
              'force_keep_source']:
        T.set(k, BB.get(k))

    # compute the pixel-space mask for *brightblob* values
    brightblobmask = refmap

    invvars = np.hstack(BB.srcinvvars)
    assert(cat.numberOfParams() == len(invvars))

    if T_donotfit:
        T_donotfit.objid = np.arange(len(T_donotfit), dtype=np.int32) + len(T)

    if write_metrics or get_all_models:
        from legacypipe.format_catalog import format_all_models
        # append our 'do not fit' sources so that the all-models file
        # matches the tractor catalog
        T2 = T
        cat2 = [src for src in newcat]
        if T_donotfit:
            T2 = merge_tables([T2, T_donotfit], columns='fillzero')
            cat2.extend([None] * len(T_donotfit))
        TT,hdr = format_all_models(T2, cat2, BB, bands, survey.allbands,
                                   force_keep=T2.force_keep_source)
        if get_all_models:
            all_models = TT
        if write_metrics:
            primhdr = fitsio.FITSHDR()
            for r in version_header.records():
                primhdr.add_record(r)
                primhdr.add_record(dict(name='PRODTYPE', value='catalog',
                                        comment='NOAO data product type'))
            with survey.write_output('all-models', brick=brickname) as out:
                TT.writeto(None, fits_object=out.fits, header=hdr,
                           primheader=primhdr)

    keys = ['cat', 'invvars', 'T', 'blobs', 'brightblobmask', 'version_header']
    if get_all_models:
        keys.append('all_models')
    if bailout:
        keys.extend(['bailout_mask', 'T_refbail'])
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

# Also called by farm.py
def get_blobiter_ref_map(refstars, T_clusters, less_masking, targetwcs):
    if refstars:
        from legacypipe.reference import get_reference_map
        refs = refstars[refstars.donotfit == False]
        if T_clusters is not None:
            refs = merge_tables([refs, T_clusters], columns='fillzero')

        if less_masking:
            # Reduce BRIGHT radius by 50%
            refs.radius_pix[refs.isbright] //= 2
            # (Also turn off special behavior for MEDIUM, in oneblob.py)

        refmap = get_reference_map(targetwcs, refs)
        del refs
    else:
        HH, WW = targetwcs.shape
        refmap = np.zeros((int(HH), int(WW)), np.uint8)
    return refmap

def _get_bailout_mask(blobs, skipblobs, targetwcs, W, H, brick, blobslices):
    maxblob = blobs.max()
    # mark all as bailed out...
    bmap = np.ones(maxblob+2, bool)
    # except no-blob
    bmap[0] = False
    # and blobs from the checkpoint file
    for i in skipblobs:
        bmap[i+1] = False
    # and blobs that are completely outside the primary region of this brick.
    U = find_unique_pixels(targetwcs, W, H, None,
                           brick.ra1, brick.ra2, brick.dec1, brick.dec2)
    for iblob in np.unique(blobs):
        if iblob == -1:
            continue
        if iblob in skipblobs:
            continue
        bslc  = blobslices[iblob]
        blobmask = (blobs[bslc] == iblob)
        if np.all(U[bslc][blobmask] == False):
            debug('Blob', iblob, 'is completely outside the PRIMARY region')
            bmap[iblob+1] = False
    bailout_mask = bmap[blobs+1]
    return bailout_mask

def _write_checkpoint(R, checkpoint_filename):
    from astrometry.util.file import pickle_to_file, trymakedirs
    d = os.path.dirname(checkpoint_filename)
    if len(d) and not os.path.exists(d):
        trymakedirs(d)
    fn = checkpoint_filename + '.tmp'
    pickle_to_file(R, fn)
    os.rename(fn, checkpoint_filename)
    debug('Wrote checkpoint to', checkpoint_filename)

def _check_checkpoints(R, blobslices, brickname):
    # Check that checkpointed blobids match our current set of blobs,
    # based on blob bounding-box.  This can fail if the code changes
    # between writing & reading the checkpoint, resulting in a
    # different set of detected sources.
    keepR = []
    for ri in R:
        brick = ri['brickname']
        iblob = ri['iblob']
        r = ri['result']

        if brick != brickname:
            print('Checkpoint brick mismatch:', brick, brickname)
            continue

        if r is None:
            pass
        else:
            if r.iblob != iblob:
                print('Checkpoint iblob mismatch:', r.iblob, iblob)
                continue
            if iblob >= len(blobslices):
                print('Checkpointed iblob', iblob, 'is too large! (>= %i)' % len(blobslices))
                continue
            if len(r) == 0:
                pass
            else:
                # expected bbox:
                sy,sx = blobslices[iblob]
                by0,by1,bx0,bx1 = sy.start, sy.stop, sx.start, sx.stop
                # check bbox
                rx0,ry0 = r.blob_x0[0], r.blob_y0[0]
                rx1,ry1 = rx0 + r.blob_width[0], ry0 + r.blob_height[0]
                if rx0 != bx0 or ry0 != by0 or rx1 != bx1 or ry1 != by1:
                    print('Checkpointed blob bbox', [rx0,rx1,ry0,ry1],
                          'does not match expected', [bx0,bx1,by0,by1], 'for iblob', iblob)
                    continue
        keepR.append(ri)
    return keepR

def _blob_iter(brickname, blobslices, blobsrcs, blobs, targetwcs, tims, cat, bands,
               plots, ps, reoptimize, iterative, use_ceres, refmap,
               large_galaxies_force_pointsource, less_masking,
               brick,
               skipblobs=None, max_blobsize=None, custom_brick=False):
    '''
    *blobs*: map, with -1 indicating no-blob, other values indexing *blobslices*,*blobsrcs*.
    '''
    from collections import Counter

    if skipblobs is None:
        skipblobs = []

    H,W = targetwcs.shape

    # sort blobs by size so that larger ones start running first
    blobvals = Counter(blobs[blobs>=0])
    blob_order = np.array([b for b,npix in blobvals.most_common()])
    del blobvals

    if custom_brick:
        U = None
    else:
        U = find_unique_pixels(targetwcs, W, H, None,
                               brick.ra1, brick.ra2, brick.dec1, brick.dec2)

    for nblob,iblob in enumerate(blob_order):
        if iblob in skipblobs:
            info('Skipping blob', iblob)
            continue

        bslc  = blobslices[iblob]
        Isrcs = blobsrcs  [iblob]
        assert(len(Isrcs) > 0)

        # blob bbox in target coords
        sy,sx = bslc
        by0,by1 = sy.start, sy.stop
        bx0,bx1 = sx.start, sx.stop
        blobh,blobw = by1 - by0, bx1 - bx0

        # Here we assume the "blobs" array has been remapped so that
        # -1 means "no blob", while 0 and up label the blobs, thus
        # iblob equals the value in the "blobs" map.
        blobmask = (blobs[bslc] == iblob)

        if U is not None:
            # If the blob is solely outside the unique region of this brick,
            # skip it!
            if np.all(U[bslc][blobmask] == False):
                info('Blob', nblob+1, 'is completely outside the unique region of this brick -- skipping')
                yield (brickname, iblob, None)
                continue

        # find one pixel within the blob, for debugging purposes
        onex = oney = None
        for y in range(by0, by1):
            ii = np.flatnonzero(blobmask[y-by0,:])
            if len(ii) == 0:
                continue
            onex = bx0 + ii[0]
            oney = y
            break

        npix = np.sum(blobmask)
        info(('Blob %i of %i, id: %i, sources: %i, size: %ix%i, npix %i, brick X: %i,%i, ' +
               'Y: %i,%i, one pixel: %i %i') %
              (nblob+1, len(blobslices), iblob, len(Isrcs), blobw, blobh, npix,
               bx0,bx1,by0,by1, onex,oney))

        if max_blobsize is not None and npix > max_blobsize:
            info('Number of pixels in blob,', npix, ', exceeds max blobsize', max_blobsize)
            yield (brickname, iblob, None)
            continue

        # Here we cut out subimages for the blob...
        rr,dd = targetwcs.pixelxy2radec([bx0,bx0,bx1,bx1],[by0,by1,by1,by0])
        subtimargs = []
        for tim in tims:
            h,w = tim.shape
            _,x,y = tim.subwcs.radec2pixelxy(rr,dd)
            sx0,sx1 = x.min(), x.max()
            sy0,sy1 = y.min(), y.max()
            #print('blob extent in pixel space of', tim.name, ': x',
            # (sx0,sx1), 'y', (sy0,sy1), 'tim shape', (h,w))
            if sx1 < 0 or sy1 < 0 or sx0 > w or sy0 > h:
                continue
            sx0 = np.clip(int(np.floor(sx0)), 0, w-1)
            sx1 = np.clip(int(np.ceil (sx1)), 0, w-1) + 1
            sy0 = np.clip(int(np.floor(sy0)), 0, h-1)
            sy1 = np.clip(int(np.ceil (sy1)), 0, h-1) + 1
            subslc = slice(sy0,sy1),slice(sx0,sx1)
            subimg = tim.getImage ()[subslc]
            subie  = tim.getInvError()[subslc]
            subdq  = tim.dq[subslc]
            subwcs = tim.getWcs().shifted(sx0, sy0)
            subsky = tim.getSky().shifted(sx0, sy0)
            subpsf = tim.getPsf().getShifted(sx0, sy0)
            subwcsobj = tim.subwcs.get_subimage(int(sx0), int(sy0),
                                                int(sx1-sx0), int(sy1-sy0))
            tim.imobj.psfnorm = tim.psfnorm
            tim.imobj.galnorm = tim.galnorm
            # FIXME -- maybe the cache is worth sending?
            if hasattr(tim.psf, 'clear_cache'):
                tim.psf.clear_cache()
            subtimargs.append((subimg, subie, subdq, subwcs, subwcsobj,
                               tim.getPhotoCal(),
                               subsky, subpsf, tim.name,
                               tim.band, tim.sig1, tim.imobj))

        yield (brickname, iblob,
               (nblob, iblob, Isrcs, targetwcs, bx0, by0, blobw, blobh,
               blobmask, subtimargs, [cat[i] for i in Isrcs], bands, plots, ps,
               reoptimize, iterative, use_ceres, refmap[bslc],
               large_galaxies_force_pointsource, less_masking))

def _bounce_one_blob(X):
    ''' This just wraps the one_blob function, for debugging &
    multiprocessing purposes.
    '''
    from legacypipe.oneblob import one_blob
    (brickname, iblob, X) = X
    try:
        result = one_blob(X)
        ### This defines the format of the results in the checkpoints files
        return dict(brickname=brickname, iblob=iblob, result=result)
    except:
        import traceback
        print('Exception in one_blob: brick %s, iblob %i' % (brickname, iblob))
        traceback.print_exc()
        raise

def _get_mod(X):
    from tractor import Tractor
    (tim, srcs) = X
    t0 = Time()
    tractor = Tractor([tim], srcs)
    mod = tractor.getModelImage(0)
    debug('Getting model for', tim, ':', Time()-t0)
    if hasattr(tim.psf, 'clear_cache'):
        tim.psf.clear_cache()
    return mod

def _get_both_mods(X):
    from astrometry.util.resample import resample_with_wcs, OverlapError
    from astrometry.util.miscutils import get_overlapping_region
    (tim, srcs, srcblobs, blobmap, targetwcs) = X
    t0 = Time()
    mod = np.zeros(tim.getModelShape(), np.float32)
    blobmod = np.zeros(tim.getModelShape(), np.float32)
    assert(len(srcs) == len(srcblobs))
    ### modelMasks during fitblobs()....?
    try:
        Yo,Xo,Yi,Xi,_ = resample_with_wcs(tim.subwcs, targetwcs)
    except OverlapError:
        return None,None
    timblobmap = np.empty(mod.shape, blobmap.dtype)
    timblobmap[:,:] = -1
    timblobmap[Yo,Xo] = blobmap[Yi,Xi]
    del Yo,Xo,Yi,Xi
    
    for src,srcblob in zip(srcs, srcblobs):
        patch = src.getModelPatch(tim)
        if patch is None:
            continue
        #patch.addTo(mod)
        # From patch.addTo()
        (ih, iw) = mod.shape
        (ph, pw) = patch.shape
        (outx, inx) = get_overlapping_region(
            patch.x0, patch.x0 + pw - 1, 0, iw - 1)
        (outy, iny) = get_overlapping_region(
            patch.y0, patch.y0 + ph - 1, 0, ih - 1)
        if inx == [] or iny == []:
            continue
        p = patch.patch[iny, inx]
        mod[outy, outx] += p
        # mask by blob map
        blobmod[outy, outx] += p * (timblobmap[outy,outx] == srcblob)
    if hasattr(tim.psf, 'clear_cache'):
        tim.psf.clear_cache()
    return mod, blobmod

def stage_coadds(survey=None, bands=None, version_header=None, targetwcs=None,
                 tims=None, ps=None, brickname=None, ccds=None,
                 custom_brick=False,
                 T=None, T_donotfit=None, T_refbail=None,
                 blobs=None,
                 cat=None, pixscale=None, plots=False,
                 coadd_bw=False, brick=None, W=None, H=None, lanczos=True,
                 co_sky=None,
                 saturated_pix=None,
                 brightblobmask=None,
                 bailout_mask=None,
                 mp=None,
                 record_event=None,
                 **kwargs):
    '''
    After the `stage_fitblobs` fitting stage, we have all the source
    model fits, and we can create coadds of the images, model, and
    residuals.  We also perform aperture photometry in this stage.
    '''
    from functools import reduce
    from legacypipe.survey import apertures_arcsec
    from legacypipe.bits import IN_BLOB
    record_event and record_event('stage_coadds: starting')
    _add_stage_version(version_header, 'COAD', 'coadds')
    tlast = Time()

    # Write per-brick CCDs table
    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='ccdinfo',
                            comment='NOAO data product type'))
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=primhdr)

    if plots:
        cat_init = [src for it,src in zip(T.iterative, cat) if not(it)]
        cat_iter = [src for it,src in zip(T.iterative, cat) if it]
        print(len(cat_init), 'initial sources and', len(cat_iter), 'iterative')
        mods_init = mp.map(_get_mod, [(tim, cat_init) for tim in tims])
        mods_iter = mp.map(_get_mod, [(tim, cat_iter) for tim in tims])
        coimgs_init,_ = quick_coadds(tims, bands, targetwcs, images=mods_init)
        coimgs_iter,_ = quick_coadds(tims, bands, targetwcs, images=mods_iter)
        coimgs,_ = quick_coadds(tims, bands, targetwcs)

        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('First-round data')
        ps.savefig()

        plt.clf()
        dimshow(get_rgb(coimgs_init, bands))
        plt.title('First-round model fits')
        ps.savefig()

        plt.clf()
        dimshow(get_rgb([img-mod for img,mod in zip(coimgs,coimgs_init)], bands))
        plt.title('First-round residuals')
        ps.savefig()

        plt.clf()
        dimshow(get_rgb(coimgs_iter, bands))
        plt.title('Iterative model fits')
        ps.savefig()

        plt.clf()
        dimshow(get_rgb([mod+mod2 for mod,mod2 in zip(coimgs_init, coimgs_iter)], bands))
        plt.title('Initial + Iterative model fits')
        ps.savefig()

        plt.clf()
        dimshow(get_rgb([img-mod-mod2 for img,mod,mod2 in zip(coimgs,coimgs_init,coimgs_iter)], bands))
        plt.title('Iterative model residuals')
        ps.savefig()

    tnow = Time()
    debug('[serial coadds]:', tnow-tlast)
    tlast = tnow
    # Render model images...
    record_event and record_event('stage_coadds: model images')
    #mods = mp.map(_get_mod, [(tim, cat) for tim in tims])
    bothmods = mp.map(_get_both_mods, [(tim, cat, T.blob, blobs, targetwcs) for tim in tims])

    mods = [m for m,b in bothmods]
    blobmods = [b for m,b in bothmods]
    del bothmods

    tnow = Time()
    debug('[parallel coadds] Getting model images:', tnow-tlast)
    tlast = tnow

    # Compute source pixel positions
    assert(len(T) == len(cat))
    ra  = np.array([src.getPosition().ra  for src in cat])
    dec = np.array([src.getPosition().dec for src in cat])

    # T_refbail and T_donotfit sources get the same treatment...
    if T_refbail is not None:
        if T_donotfit is not None:
            T_donotfit = merge_tables([T_donotfit, T_refbail], columns='fillzero')
        else:
            T_donotfit = T_refbail

    # We tag the "T_donotfit" sources on the end to get aperture phot
    # and other metrics.
    if T_donotfit:
        ra  = np.append(ra,  T_donotfit.ra)
        dec = np.append(dec, T_donotfit.dec)
    ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)

    # Get integer brick pixel coords for each source, for referencing maps
    T.out_of_bounds = reduce(np.logical_or, [xx < 0.5, yy < 0.5,
                                             xx > W+0.5, yy > H+0.5])[:len(T)]
    ixy = (np.clip(np.round(xx - 1), 0, W-1).astype(int),
           np.clip(np.round(yy - 1), 0, H-1).astype(int))
    # convert apertures to pixels
    apertures = apertures_arcsec / pixscale
    # Aperture photometry locations
    apxy = np.vstack((xx - 1., yy - 1.)).T
    del xx,yy,ok,ra,dec

    record_event and record_event('stage_coadds: coadds')
    C = make_coadds(tims, bands, targetwcs, mods=mods, blobmods=blobmods,
                    xy=ixy,
                    ngood=True, detmaps=True, psfsize=True, allmasks=True,
                    lanczos=lanczos,
                    apertures=apertures, apxy=apxy,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, tims,
                                   targetwcs, co_sky),
                    plots=plots, ps=ps, mp=mp)
    record_event and record_event('stage_coadds: extras')

    # Coadds of galaxy sims only, image only
    if hasattr(tims[0], 'sims_image'):
        sims_mods = [tim.sims_image for tim in tims]
        T_sims_coadds = make_coadds(tims, bands, targetwcs, mods=sims_mods,
                                    lanczos=lanczos, mp=mp)
        sims_coadd = T_sims_coadds.comods
        del T_sims_coadds
        image_only_mods= [tim.data-tim.sims_image for tim in tims]
        make_coadds(tims, bands, targetwcs, mods=image_only_mods,
                    lanczos=lanczos, mp=mp)
    ###

    # Save per-source measurements of the maps produced during coadding
    cols = ['nobs', 'anymask', 'allmask', 'psfsize', 'psfdepth', 'galdepth',
            'mjd_min', 'mjd_max']
    # store galaxy sim bounding box in Tractor cat
    if 'sims_xy' in C.T.get_columns():
        cols.append('sims_xy')

    Nno = T_donotfit and len(T_donotfit) or 0
    Nyes = len(T)
    for c in cols:
        val = C.T.get(c)
        T.set(c, val[:Nyes])
        # We appended T_donotfit; peel off those results
        if Nno:
            T_donotfit.set(c, val[Nyes:])
    assert(C.AP is not None)
    # How many apertures?
    A = len(apertures_arcsec)
    T.apflux        = np.zeros((len(T), len(bands), A), np.float32)
    T.apflux_ivar   = np.zeros((len(T), len(bands), A), np.float32)
    T.apflux_masked = np.zeros((len(T), len(bands), A), np.float32)
    T.apflux_resid  = np.zeros((len(T), len(bands), A), np.float32)
    T.apflux_blobresid = np.zeros((len(T), len(bands), A), np.float32)
    if Nno:
        T_donotfit.apflux        = np.zeros((Nno, len(bands), A), np.float32)
        T_donotfit.apflux_ivar   = np.zeros((Nno, len(bands), A), np.float32)
        T_donotfit.apflux_masked = np.zeros((Nno, len(bands), A), np.float32)
        T_donotfit.apflux_resid  = np.zeros((Nno, len(bands), A), np.float32)
        T_donotfit.apflux_blobresid = np.zeros((Nno, len(bands), A), np.float32)
    AP = C.AP
    for iband,band in enumerate(bands):
        T.apflux       [:,iband,:] = AP.get('apflux_img_%s'      % band)[:Nyes,:]
        T.apflux_ivar  [:,iband,:] = AP.get('apflux_img_ivar_%s' % band)[:Nyes,:]
        T.apflux_masked[:,iband,:] = AP.get('apflux_masked_%s'   % band)[:Nyes,:]
        T.apflux_resid [:,iband,:] = AP.get('apflux_resid_%s'    % band)[:Nyes,:]
        T.apflux_blobresid[:,iband,:] = AP.get('apflux_blobresid_%s'    % band)[:Nyes,:]
        if Nno:
            T_donotfit.apflux       [:,iband,:] = AP.get('apflux_img_%s'      % band)[Nyes:,:]
            T_donotfit.apflux_ivar  [:,iband,:] = AP.get('apflux_img_ivar_%s' % band)[Nyes:,:]
            T_donotfit.apflux_masked[:,iband,:] = AP.get('apflux_masked_%s'   % band)[Nyes:,:]
            T_donotfit.apflux_resid [:,iband,:] = AP.get('apflux_resid_%s'    % band)[Nyes:,:]
            T_donotfit.apflux_blobresid[:,iband,:] = AP.get('apflux_blobresid_%s'    % band)[Nyes:,:]
    del AP

    # Compute depth histogram
    D = _depth_histogram(brick, targetwcs, bands, C.psfdetivs, C.galdetivs)
    with survey.write_output('depth-table', brick=brickname) as out:
        D.writeto(None, fits_object=out.fits)
    del D

    coadd_list= [('image', C.coimgs, {}),
                 ('model', C.comods, {}),
                 ('blobmodel', C.coblobmods, {}),
                 ('resid', C.coresids, dict(resids=True))]
    ### blobresids??
    if hasattr(tims[0], 'sims_image'):
        coadd_list.append(('simscoadd', sims_coadd, {}))

    for name,ims,rgbkw in coadd_list:
        rgb = get_rgb(ims, bands, **rgbkw)
        kwa = {}
        if coadd_bw and len(bands) == 1:
            rgb = rgb.sum(axis=2)
            kwa = dict(cmap='gray')

        with survey.write_output(name + '-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
            info('Wrote', out.fn)
        del rgb

    # Construct a mask bits map
    maskbits = np.zeros((H,W), np.int16)
    # !PRIMARY
    if custom_brick:
        U = None
    else:
        U = find_unique_pixels(targetwcs, W, H, None,
                               brick.ra1, brick.ra2, brick.dec1, brick.dec2)
        maskbits += MASKBITS['NPRIMARY'] * np.logical_not(U).astype(np.int16)
        del U

    # BRIGHT
    if brightblobmask is not None:
        maskbits += MASKBITS['BRIGHT'] * ((brightblobmask & IN_BLOB['BRIGHT']) > 0)
        maskbits += MASKBITS['MEDIUM'] * ((brightblobmask & IN_BLOB['MEDIUM']) > 0)
        maskbits += MASKBITS['GALAXY'] * ((brightblobmask & IN_BLOB['GALAXY']) > 0)
        maskbits += MASKBITS['CLUSTER'] * ((brightblobmask & IN_BLOB['CLUSTER']) > 0)

    # SATUR
    saturvals = dict(g=MASKBITS['SATUR_G'], r=MASKBITS['SATUR_R'], z=MASKBITS['SATUR_Z'])
    if saturated_pix is not None:
        for b,sat in zip(bands, saturated_pix):
            maskbits += saturvals[b] * sat.astype(np.int16)

    # ALLMASK_{g,r,z}
    allmaskvals = dict(g=MASKBITS['ALLMASK_G'], r=MASKBITS['ALLMASK_R'],
                       z=MASKBITS['ALLMASK_Z'])
    for b,allmask in zip(bands, C.allmasks):
        if not b in allmaskvals:
            continue
        maskbits += allmaskvals[b]* (allmask > 0).astype(np.int16)

    # BAILOUT_MASK
    if bailout_mask is not None:
        maskbits += MASKBITS['BAILOUT'] * bailout_mask.astype(bool)

    # copy version_header before modifying it.
    hdr = fitsio.FITSHDR()
    for r in version_header.records():
        hdr.add_record(r)
    # Plug the WCS header cards into these images
    targetwcs.add_to_header(hdr)
    hdr.add_record(dict(name='EQUINOX', value=2000., comment='Observation epoch'))
    hdr.delete('IMAGEW')
    hdr.delete('IMAGEH')
    hdr.add_record(dict(name='IMTYPE', value='maskbits',
                        comment='LegacySurveys image type'))
    # NOTE that we pass the "maskbits" and "maskbits_header" variables
    # on to later stages, because we will add in the WISE mask planes
    # later (and write the result in the writecat stage. THEREFORE, if
    # you make changes to the bit mappings here, you MUST also adjust
    # the header values (and bit mappings for the WISE masks) in
    # stage_writecat.
    hdr.add_record(dict(name='NPRIMARY', value=MASKBITS['NPRIMARY'],
                        comment='Mask value for non-primary brick area'))
    hdr.add_record(dict(name='BRIGHT', value=MASKBITS['BRIGHT'],
                        comment='Mask value for bright star in blob'))
    hdr.add_record(dict(name='BAILOUT', value=MASKBITS['BAILOUT'],
                        comment='Mask value for bailed-out processing'))
    hdr.add_record(dict(name='MEDIUM', value=MASKBITS['MEDIUM'],
                        comment='Mask value for medium-bright star in blob'))
    hdr.add_record(dict(name='GALAXY', value=MASKBITS['GALAXY'],
                        comment='Mask value for LSLGA large galaxy'))
    hdr.add_record(dict(name='CLUSTER', value=MASKBITS['CLUSTER'],
                        comment='Mask value for Cluster'))
    keys = sorted(saturvals.keys())
    for b in keys:
        k = 'SATUR_%s' % b.upper()
        hdr.add_record(dict(name=k, value=MASKBITS[k],
                            comment='Mask value for saturated (& nearby) pixels in %s band' % b))
    keys = sorted(allmaskvals.keys())
    for b in keys:
        hdr.add_record(dict(name='ALLM_%s' % b.upper(), value=allmaskvals[b],
                            comment='Mask value for ALLMASK band %s' % b))
    maskbits_header = hdr

    if plots:
        plt.clf()
        ra  = np.array([src.getPosition().ra  for src in cat])
        dec = np.array([src.getPosition().dec for src in cat])
        ok,x0,y0 = targetwcs.radec2pixelxy(T.orig_ra, T.orig_dec)
        ok,x1,y1 = targetwcs.radec2pixelxy(ra, dec)
        dimshow(get_rgb(C.coimgs, bands))
        ax = plt.axis()
        #plt.plot(np.vstack((x0,x1))-1, np.vstack((y0,y1))-1, 'r-')
        I = np.flatnonzero(T.orig_ra != 0.)
        for xx0,yy0,xx1,yy1 in zip(x0[I],y0[I],x1[I],y1[I]):
            plt.plot([xx0-1,xx1-1], [yy0-1,yy1-1], 'r-')
        plt.plot(x1-1, y1-1, 'r.')
        plt.axis(ax)
        plt.title('Original to final source positions')
        ps.savefig()

        plt.clf()
        dimshow(get_rgb(C.coimgs, bands))
        ax = plt.axis()
        ps.savefig()

        for src,x,y,rr,dd in zip(cat, x1, y1, ra, dec):
            from tractor import PointSource
            from tractor.galaxy import DevGalaxy, ExpGalaxy
            from tractor.sersic import SersicGalaxy
            ee = []
            ec = []
            cc = None
            green = (0.2,1,0.2)
            if isinstance(src, PointSource):
                plt.plot(x, y, 'o', mfc=green, mec='k', alpha=0.6)
            elif isinstance(src, ExpGalaxy):
                ee = [src.shape]
                cc = '0.8'
                ec = [cc]
            elif isinstance(src, DevGalaxy):
                ee = [src.shape]
                cc = green
                ec = [cc]
            elif isinstance(src, SersicGalaxy):
                ee = [src.shape]
                cc = 'm'
                ec = [cc]
            else:
                print('Unknown type:', src)
                continue

            for e,c in zip(ee, ec):
                G = e.getRaDecBasis()
                angle = np.linspace(0, 2.*np.pi, 60)
                xy = np.vstack((np.append([0,0,1], np.sin(angle)),
                                np.append([0,1,0], np.cos(angle)))).T
                rd = np.dot(G, xy.T).T
                r = rr + rd[:,0] * np.cos(np.deg2rad(dd))
                d = dd + rd[:,1]
                ok,xx,yy = targetwcs.radec2pixelxy(r, d)
                xx -= 1.
                yy -= 1.
                x1,x2,x3 = xx[:3]
                y1,y2,y3 = yy[:3]
                plt.plot([x3, x1, x2], [y3, y1, y2], '-', color=c)
                plt.plot(x1, y1, '.', color=cc, ms=3, alpha=0.6)
                xx = xx[3:]
                yy = yy[3:]
                plt.plot(xx, yy, '-', color=c)
        plt.axis(ax)
        ps.savefig()

    tnow = Time()
    debug('[serial coadds] Aperture photometry, wrap-up', tnow-tlast)

    return dict(T=T, T_donotfit=T_donotfit, apertures_pix=apertures,
                apertures_arcsec=apertures_arcsec,
                maskbits=maskbits,
                maskbits_header=maskbits_header, version_header=version_header)

def get_fiber_fluxes(cat, T, targetwcs, H, W, pixscale, bands,
                     fibersize=1.5, seeing=1., year=2020.0,
                     plots=False, ps=None):
    from tractor import GaussianMixturePSF
    from legacypipe.survey import LegacySurveyWcs
    import astropy.time
    from tractor.tractortime import TAITime
    from tractor.image import Image
    from tractor.basics import LinearPhotoCal
    import photutils

    # Compute source pixel positions
    ra  = np.array([src.getPosition().ra  for src in cat])
    dec = np.array([src.getPosition().dec for src in cat])
    ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)
    del ok,ra,dec

    # Create a fake tim for each band to construct the models in 1" seeing
    # For Gaia stars, we need to give a time for evaluating the models.
    mjd_tai = astropy.time.Time(year, format='jyear').tai.mjd
    tai = TAITime(None, mjd=mjd_tai)
    # 1" FWHM -> pixels FWHM -> pixels sigma -> pixels variance
    v = ((seeing / pixscale) / 2.35)**2
    data = np.zeros((H,W), np.float32)
    inverr = np.ones((H,W), np.float32)
    psf = GaussianMixturePSF(1., 0., 0., v, v, 0.)
    wcs = LegacySurveyWcs(targetwcs, tai)
    faketim = Image(data=data, inverr=inverr, psf=psf,
                    wcs=wcs, photocal=LinearPhotoCal(1., bands[0]))

    # A model image (containing all sources) for each band
    modimgs = [np.zeros((H,W), np.float32) for b in bands]
    # A blank image that we'll use for rendering the flux from a single model
    onemod = data

    # Results go here!
    fiberflux    = np.zeros((len(cat),len(bands)), np.float32)
    fibertotflux = np.zeros((len(cat),len(bands)), np.float32)

    # Fiber diameter in arcsec -> radius in pix
    fiberrad = (fibersize / pixscale) / 2.

    # For each source, compute and measure its model, and accumulate
    for isrc,(src,sx,sy) in enumerate(zip(cat, xx-1., yy-1.)):
        # This works even if bands[0] has zero flux (or no overlapping
        # images)
        ums = src.getUnitFluxModelPatches(faketim)
        assert(len(ums) == 1)
        patch = ums[0]
        if patch is None:
            continue
        br = src.getBrightness()
        for iband,(modimg,band) in enumerate(zip(modimgs,bands)):
            flux = br.getFlux(band)
            flux_iv = T.flux_ivar[isrc, iband]
            if flux <= 0 or flux_iv <= 0:
                continue
            # Accumulate into image containing all models
            patch.addTo(modimg, scale=flux)
            # Add to blank image & photometer
            patch.addTo(onemod, scale=flux)
            aper = photutils.CircularAperture((sx, sy), fiberrad)
            p = photutils.aperture_photometry(onemod, aper)
            f = p.field('aperture_sum')[0]
            if not np.isfinite(f):
                # If the source is off the brick (eg, ref sources), can be NaN
                continue
            fiberflux[isrc,iband] = f
            # Blank out the image again
            x0,x1,y0,y1 = patch.getExtent()
            onemod[y0:y1, x0:x1] = 0.

    # Now photometer the accumulated images
    # Aperture photometry locations
    apxy = np.vstack((xx - 1., yy - 1.)).T
    aper = photutils.CircularAperture(apxy, fiberrad)
    for iband,modimg in enumerate(modimgs):
        p = photutils.aperture_photometry(modimg, aper)
        f = p.field('aperture_sum')
        # If the source is off the brick (eg, ref sources), can be NaN
        I = np.isfinite(f)
        if len(I):
            fibertotflux[I, iband] = f[I]

    if plots:
        for modimg,band in zip(modimgs, bands):
            plt.clf()
            plt.imshow(modimg, interpolation='nearest', origin='lower',
                       vmin=0, vmax=0.1, cmap='gray')
            plt.title('Fiberflux model for band %s' % band)
            ps.savefig()

        for iband,band in enumerate(bands):
            plt.clf()
            flux = [src.getBrightness().getFlux(band) for src in cat]
            plt.plot(flux, fiberflux[:,iband], 'b.', label='FiberFlux')
            plt.plot(flux, fibertotflux[:,iband], 'gx', label='FiberTotFlux')
            plt.plot(flux, T.apflux[:,iband, 1], 'r+', label='Apflux(1.5)')
            plt.legend()
            plt.xlabel('Catalog total flux')
            plt.ylabel('Aperture flux')
            plt.title('Fiberflux: %s band' % band)
            plt.xscale('symlog')
            plt.yscale('symlog')
            ps.savefig()

    return fiberflux, fibertotflux

def _depth_histogram(brick, targetwcs, bands, detivs, galdetivs):
    # Compute the brick's unique pixels.
    U = None
    if hasattr(brick, 'ra1'):
        debug('Computing unique brick pixels...')
        H,W = targetwcs.shape
        U = find_unique_pixels(targetwcs, W, H, None,
                               brick.ra1, brick.ra2, brick.dec1, brick.dec2)
        U = np.flatnonzero(U)
        debug(len(U), 'of', W*H, 'pixels are unique to this brick')

    # depth histogram bins
    depthbins = np.arange(20, 25.001, 0.1)
    depthbins[0] = 0.
    depthbins[-1] = 100.
    D = fits_table()
    D.depthlo = depthbins[:-1].astype(np.float32)
    D.depthhi = depthbins[1: ].astype(np.float32)

    for band,detiv,galdetiv in zip(bands,detivs,galdetivs):
        for det,name in [(detiv, 'ptsrc'), (galdetiv, 'gal')]:
            # compute stats for 5-sigma detection
            with np.errstate(divide='ignore'):
                depth = 5. / np.sqrt(det)
            # that's flux in nanomaggies -- convert to mag
            depth = -2.5 * (np.log10(depth) - 9)
            # no coverage -> very bright detection limit
            depth[np.logical_not(np.isfinite(depth))] = 0.
            if U is not None:
                depth = depth.flat[U]
            if len(depth):
                debug(band, name, 'band depth map: percentiles',
                      np.percentile(depth, np.arange(0,101, 10)))
            # histogram
            D.set('counts_%s_%s' % (name, band),
                  np.histogram(depth, bins=depthbins)[0].astype(np.int32))
    return D

def stage_wise_forced(
    survey=None,
    cat=None,
    T=None,
    targetwcs=None,
    targetrd=None,
    W=None, H=None,
    pixscale=None,
    brickname=None,
    unwise_dir=None,
    unwise_tr_dir=None,
    unwise_modelsky_dir=None,
    brick=None,
    wise_ceres=True,
    unwise_coadds=True,
    version_header=None,
    maskbits=None,
    mp=None,
    record_event=None,
    ps=None,
    plots=False,
    **kwargs):
    '''
    After the model fits are finished, we can perform forced
    photometry of the unWISE coadds.
    '''
    from legacypipe.unwise import unwise_phot, collapse_unwise_bitmask, unwise_tiles_touching_wcs
    from tractor import NanoMaggies

    record_event and record_event('stage_wise_forced: starting')
    _add_stage_version(version_header, 'WISE', 'wise_forced')

    if not plots:
        ps = None

    tiles = unwise_tiles_touching_wcs(targetwcs)
    info('Cut to', len(tiles), 'unWISE tiles')

    # the way the roiradec box is used, the min/max order doesn't matter
    roiradec = [targetrd[0,0], targetrd[2,0], targetrd[0,1], targetrd[2,1]]

    # Sources to photometer
    do_phot = np.ones(len(cat), bool)

    # Drop sources within the CLUSTER mask from forced photometry.
    Icluster = None
    if maskbits is not None:
        incluster = (maskbits & MASKBITS['CLUSTER'] > 0)
        if np.any(incluster):
            print('Checking for sources inside CLUSTER mask')
            ra  = np.array([src.getPosition().ra  for src in cat])
            dec = np.array([src.getPosition().dec for src in cat])
            ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)
            xx = np.round(xx - 1).astype(int)
            yy = np.round(yy - 1).astype(int)
            I = np.flatnonzero(ok * (xx >= 0)*(xx < W) * (yy >= 0)*(yy < H))
            if len(I):
                Icluster = I[incluster[yy[I], xx[I]]]
                print('Found', len(Icluster), 'of', len(cat), 'sources inside CLUSTER mask')
                do_phot[Icluster] = False
    Nskipped = len(do_phot) - np.sum(do_phot)

    wcat = []
    for i in np.flatnonzero(do_phot):
        src = cat[i]
        src = src.copy()
        src.setBrightness(NanoMaggies(w=1.))
        wcat.append(src)

    # use Aaron's WISE pixelized PSF model (unwise_psf repository)?
    wpixpsf = True

    # Create list of groups-of-tiles to photometer
    args = []
    # Skip if $UNWISE_COADDS_DIR or --unwise-dir not set.
    if unwise_dir is not None:
        wtiles = tiles.copy()
        wtiles.unwise_dir = np.array([unwise_dir]*len(tiles))
        for band in [1,2,3,4]:
            get_masks = targetwcs if (band == 1) else None
            args.append((wcat, wtiles, band, roiradec, wise_ceres, wpixpsf,
                         unwise_coadds, get_masks, ps, True,
                         unwise_modelsky_dir))

    # Add time-resolved WISE coadds
    # Skip if $UNWISE_COADDS_TIMERESOLVED_DIR or --unwise-tr-dir not set.
    eargs = []
    if unwise_tr_dir is not None:
        tdir = unwise_tr_dir
        TR = fits_table(os.path.join(tdir, 'time_resolved_atlas.fits'))
        debug('Read', len(TR), 'time-resolved WISE coadd tiles')
        TR.cut(np.array([t in tiles.coadd_id for t in TR.coadd_id]))
        debug('Cut to', len(TR), 'time-resolved vs', len(tiles), 'full-depth')
        assert(len(TR) == len(tiles))
        # Ugly -- we need to look up the "{ra,dec}[12]" fields from the non-TR
        # table to support unique areas of tiles.
        imap = dict((c,i) for i,c in enumerate(tiles.coadd_id))
        I = np.array([imap[c] for c in TR.coadd_id])
        for c in ['ra1','ra2','dec1','dec2', 'crpix_w1', 'crpix_w2']:
            TR.set(c, tiles.get(c)[I])
        # How big do we need to make the WISE time-resolved arrays?
        debug('TR epoch_bitmask:', TR.epoch_bitmask)
        # axis= arg to np.count_nonzero is new in numpy 1.12
        Nepochs = max(np.atleast_1d([np.count_nonzero(e)
                                     for e in TR.epoch_bitmask]))
        _,ne = TR.epoch_bitmask.shape
        info('Max number of time-resolved unWISE epochs for these tiles:', Nepochs)
        debug('epoch bitmask length:', ne)
        # Add time-resolved coadds
        for band in [1,2]:
            # W1 is bit 0 (value 0x1), W2 is bit 1 (value 0x2)
            bitmask = (1 << (band-1))
            # The epoch_bitmask entries are not *necessarily*
            # contiguous, and not necessarily aligned for the set of
            # overlapping tiles.  We will align the non-zero epochs of
            # the tiles.  (eg, brick 2437p425 vs coadds 2426p424 &
            # 2447p424 in NEO-2).

            # find the non-zero epochs for each overlapping tile
            epochs = np.empty((len(TR), Nepochs), int)
            epochs[:,:] = -1
            for i in range(len(TR)):
                ei = np.flatnonzero(TR.epoch_bitmask[i,:] & bitmask)
                epochs[i,:len(ei)] = ei

            for ie in range(Nepochs):
                # Which tiles have images for this epoch?
                I = np.flatnonzero(epochs[:,ie] >= 0)
                if len(I) == 0:
                    continue
                debug('Epoch index %i: %i tiles:' % (ie, len(I)), TR.coadd_id[I],
                      'epoch numbers', epochs[I,ie])
                eptiles = TR[I]
                eptiles.unwise_dir = np.array([os.path.join(tdir, 'e%03i'%ep)
                                              for ep in epochs[I,ie]])
                eargs.append((ie,(wcat, eptiles, band, roiradec,
                                  wise_ceres, wpixpsf, False, None, ps, False, unwise_modelsky_dir)))

    # Run the forced photometry!
    record_event and record_event('stage_wise_forced: photometry')
    phots = mp.map(unwise_phot, args + [a for ie,a in eargs])
    record_event and record_event('stage_wise_forced: results')

    # Unpack results...
    WISE = None
    wise_mask_maps = None
    if len(phots):
        # The "phot" results for the full-depth coadds are one table per
        # band.  Merge all those columns.
        wise_models = {}
        for i,p in enumerate(phots[:len(args)]):
            if p is None:
                (wcat,tiles,band) = args[i+1][:3]
                print('"None" result from WISE forced phot:', tiles, band)
                continue
            if unwise_coadds:
                wise_models.update(p.models)
            if p.maskmap is not None:
                wise_mask_maps = p.maskmap
            if WISE is None:
                WISE = p.phot
            else:
                p.phot.delete_column('wise_coadd_id') # duplicate
                WISE.add_columns_from(p.phot)

        if wise_mask_maps is not None:
            wise_mask_maps = [
                collapse_unwise_bitmask(wise_mask_maps, 1),
                collapse_unwise_bitmask(wise_mask_maps, 2)]

        if unwise_coadds:
            from legacypipe.coadds import UnwiseCoadd
            # Create the WCS into which we'll resample the tiles.
            # Same center as "targetwcs" but bigger pixel scale.
            wpixscale = 2.75
            wcoadds = UnwiseCoadd(targetwcs, W, H, pixscale, wpixscale)
            for tile in tiles.coadd_id:
                wcoadds.add(tile, wise_models)
            wcoadds.finish(survey, brickname, version_header)

        if Nskipped > 0:
            assert(len(WISE) == len(wcat))
            WISE = _fill_skipped_values(WISE, Nskipped, do_phot)
            assert(len(WISE) == len(cat))
            assert(len(WISE) == len(T))

        # Look up mask values for sources
        WISE.wise_mask = np.zeros((len(cat), 2), np.uint8)
        ra  = np.array([src.getPosition().ra  for src in cat])
        dec = np.array([src.getPosition().dec for src in cat])
        ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)
        xx = np.round(xx - 1).astype(int)
        yy = np.round(yy - 1).astype(int)
        I = np.flatnonzero(ok * (xx >= 0)*(xx < W) * (yy >= 0)*(yy < H))
        if len(I):
            WISE.wise_mask[I,0] = wise_mask_maps[0][yy[I], xx[I]]
            WISE.wise_mask[I,1] = wise_mask_maps[1][yy[I], xx[I]]

    # Unpack time-resolved results...
    WISE_T = None
    if len(phots) > len(args):
        WISE_T = True
    if WISE_T is not None:
        WISE_T = fits_table()
        phots = phots[len(args):]
        for (ie,_),r in zip(eargs, phots):
            debug('Epoch', ie, 'photometry:')
            if r is None:
                debug('Failed.')
                continue
            assert(ie < Nepochs)
            phot = r.phot
            phot.delete_column('wise_coadd_id')
            for c in phot.columns():
                if not c in WISE_T.columns():
                    x = phot.get(c)
                    WISE_T.set(c, np.zeros((len(x), Nepochs), x.dtype))
                X = WISE_T.get(c)
                X[:,ie] = phot.get(c)
        if Nskipped > 0:
            assert(len(wcat) == len(WISE_T))
            WISE_T = _fill_skipped_values(WISE_T, Nskipped, do_phot)
            assert(len(WISE_T) == len(cat))
            assert(len(WISE_T) == len(T))

    debug('Returning: WISE', WISE)
    debug('Returning: WISE_T', WISE_T)

    return dict(WISE=WISE, WISE_T=WISE_T, wise_mask_maps=wise_mask_maps,
                version_header=version_header)

def _fill_skipped_values(WISE, Nskipped, do_phot):
    # Fill in blank values for skipped (Icluster) sources
    # Append empty rows to the WISE results for !do_phot sources.
    Wempty = fits_table()
    Wempty.nil = np.zeros(Nskipped, bool)
    WISE = merge_tables([WISE, Wempty], columns='fillzero')
    WISE.delete_column('nil')
    # Reorder to match "cat" order.
    I = np.empty(len(WISE), int)
    I[:] = -1
    Ido, = np.nonzero(do_phot)
    I[Ido] = np.arange(len(Ido))
    Idont, = np.nonzero(np.logical_not(do_phot))
    I[Idont] = np.arange(len(Idont)) + len(Ido)
    assert(np.all(I > -1))
    WISE.cut(I)
    return WISE

def stage_writecat(
    survey=None,
    version_header=None,
    release=None,
    T=None,
    T_donotfit=None,
    WISE=None,
    WISE_T=None,
    maskbits=None,
    maskbits_header=None,
    wise_mask_maps=None,
    apertures_arcsec=None,
    cat=None, pixscale=None, targetwcs=None,
    W=None,H=None,
    bands=None, ps=None,
    plots=False,
    brickname=None,
    brickid=None,
    brick=None,
    invvars=None,
    gaia_stars=True,
    co_sky=None,
    record_event=None,
    **kwargs):
    '''
    Final stage in the pipeline: format results for the output
    catalog.
    '''
    from legacypipe.catalog import prepare_fits_catalog

    record_event and record_event('stage_writecat: starting')
    _add_stage_version(version_header, 'WCAT', 'writecat')

    if maskbits is not None:
        w1val = MASKBITS['WISEM1']
        w2val = MASKBITS['WISEM2']

        if wise_mask_maps is not None:
            # Add the WISE masks in!
            maskbits += w1val * (wise_mask_maps[0] != 0)
            maskbits += w2val * (wise_mask_maps[1] != 0)

        hdr = maskbits_header
        if hdr is not None:
            hdr.add_record(dict(name='WISEM1', value=w1val,
                                comment='Mask value for WISE W1 (all masks)'))
            hdr.add_record(dict(name='WISEM2', value=w2val,
                                comment='Mask value for WISE W2 (all masks)'))

        hdr.add_record(dict(name='BITNM0', value='NPRIMARY',
                            comment='maskbits bit 0: not-brick-primary'))
        hdr.add_record(dict(name='BITNM1', value='BRIGHT',
                            comment='maskbits bit 1: bright star in blob'))
        hdr.add_record(dict(name='BITNM2', value='SATUR_G',
                            comment='maskbits bit 2: g saturated + margin'))
        hdr.add_record(dict(name='BITNM3', value='SATUR_R',
                            comment='maskbits bit 3: r saturated + margin'))
        hdr.add_record(dict(name='BITNM4', value='SATUR_Z',
                            comment='maskbits bit 4: z saturated + margin'))
        hdr.add_record(dict(name='BITNM5', value='ALLMASK_G',
                            comment='maskbits bit 5: any ALLMASK_G bit set'))
        hdr.add_record(dict(name='BITNM6', value='ALLMASK_R',
                            comment='maskbits bit 6: any ALLMASK_R bit set'))
        hdr.add_record(dict(name='BITNM7', value='ALLMASK_Z',
                            comment='maskbits bit 7: any ALLMASK_Z bit set'))
        hdr.add_record(dict(name='BITNM8', value='WISEM1',
                            comment='maskbits bit 8: WISE W1 bright star mask'))
        hdr.add_record(dict(name='BITNM9', value='WISEM2',
                            comment='maskbits bit 9: WISE W2 bright star mask'))
        hdr.add_record(dict(name='BITNM10', value='BAILOUT',
                            comment='maskbits bit 10: Bailed out of processing'))
        hdr.add_record(dict(name='BITNM11', value='MEDIUM',
                            comment='maskbits bit 11: Medium-bright star'))
        hdr.add_record(dict(name='BITNM12', value='GALAXY',
                            comment='maskbits bit 12: LSLGA large galaxy'))
        hdr.add_record(dict(name='BITNM13', value='CLUSTER',
                            comment='maskbits bit 13: Cluster'))

        if wise_mask_maps is not None:
            wisehdr = fitsio.FITSHDR()
            wisehdr.add_record(dict(name='WBITNM0', value='BRIGHT',
                                    comment='Bright star core and wings'))
            wisehdr.add_record(dict(name='WBITNM1', value='SPIKE',
                                    comment='PSF-based diffraction spike'))
            wisehdr.add_record(dict(name='WBITNM2', value='GHOST',
                                    commet='Optical ghost'))
            wisehdr.add_record(dict(name='WBITNM3', value='LATENT',
                                    comment='First latent'))
            wisehdr.add_record(dict(name='WBITNM4', value='LATENT2',
                                    comment='Second latent image'))
            wisehdr.add_record(dict(name='WBITNM5', value='HALO',
                                    comment='AllWISE-like circular halo'))
            wisehdr.add_record(dict(name='WBITNM6', value='SATUR',
                                    comment='Bright star saturation'))
            wisehdr.add_record(dict(name='WBITNM7', value='SPIKE2',
                                    comment='Geometric diffraction spike'))

        with survey.write_output('maskbits', brick=brickname, shape=maskbits.shape) as out:
            out.fits.write(maskbits, header=hdr)
            if wise_mask_maps is not None:
                out.fits.write(wise_mask_maps[0], header=wisehdr)
                out.fits.write(wise_mask_maps[1], header=wisehdr)
        del wise_mask_maps

    TT = T.copy()
    for k in ['ibx','iby']:
        TT.delete_column(k)

    hdr = None
    T2,hdr = prepare_fits_catalog(cat, invvars, TT, hdr, bands, force_keep=TT.force_keep_source)

    # The "ra_ivar" values coming out of the tractor fits do *not*
    # have a cos(Dec) term -- ie, they give the inverse-variance on
    # the numerical value of RA -- so we want to make the ra_sigma
    #  values smaller by multiplying by cos(Dec); so invvars are /=
    #  cosdec^2
    T2.ra_ivar /= np.cos(np.deg2rad(T2.dec))**2

    # Compute fiber fluxes
    T2.fiberflux, T2.fibertotflux = get_fiber_fluxes(
        cat, T2, targetwcs, H, W, pixscale, bands, plots=plots, ps=ps)

    # For reference stars, plug in the reference-catalog inverse-variances.
    if 'ref_id' in T.get_columns() and 'ra_ivar' in T.get_columns():
        I, = np.nonzero(T.ref_id)
        if len(I):
            T2.ra_ivar [I] = T.ra_ivar[I]
            T2.dec_ivar[I] = T.dec_ivar[I]

    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='catalog',
                            comment='NOAO data product type'))

    if co_sky is not None:
        for band in bands:
            if band in co_sky:
                primhdr.add_record(dict(name='COSKY_%s' % band.upper(),
                                        value=co_sky[band],
                                        comment='Sky level estimated (+subtracted) from coadd'))

    for i,ap in enumerate(apertures_arcsec):
        primhdr.add_record(dict(name='APRAD%i' % i, value=ap,
                                comment='Aperture radius, in arcsec'))

    # Record the meaning of mask bits
    bits = list(DQ_BITS.values())
    bits.sort()
    bitmap = dict((v,k) for k,v in DQ_BITS.items())
    for i in range(16):
        bit = 1<<i
        if bit in bitmap:
            primhdr.add_record(dict(name='MASKB%i' % i, value=bitmap[bit],
                                    comment='ALLMASK/ANYMASK bit 2**%i=%i meaning' %
                                    (i, bit)))

    if WISE is not None:
        # Convert WISE fluxes from Vega to AB.
        # http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2ab
        vega_to_ab = dict(w1=2.699,
                          w2=3.339,
                          w3=5.174,
                          w4=6.620)

        for band in [1,2,3,4]:
            primhdr.add_record(dict(
                name='WISEAB%i' % band, value=vega_to_ab['w%i' % band],
                comment='WISE Vega to AB conv for band %i' % band))

        T2.wise_coadd_id = WISE.wise_coadd_id
        T2.wise_mask = WISE.wise_mask

        for band in [1,2,3,4]:
            # Apply the Vega-to-AB shift *while* copying columns from
            # WISE to T2.
            dm = vega_to_ab['w%i' % band]
            fluxfactor = 10.** (dm / -2.5)
            # fluxes
            c = 'w%i_nanomaggies' % band
            t = 'flux_w%i' % band
            T2.set(t, WISE.get(c) * fluxfactor)
            if WISE_T is not None and band <= 2:
                t = 'lc_flux_w%i' % band
                T2.set(t, WISE_T.get(c) * fluxfactor)
            # ivars
            c = 'w%i_nanomaggies_ivar' % band
            t = 'flux_ivar_w%i' % band
            T2.set(t, WISE.get(c) / fluxfactor**2)
            if WISE_T is not None and band <= 2:
                t = 'lc_flux_ivar_w%i' % band
                T2.set(t, WISE_T.get(c) / fluxfactor**2)
            # This is in 1/nanomaggies**2 units also
            c = 'w%i_psfdepth' % band
            t = 'psfdepth_w%i' % band
            T2.set(t, WISE.get(c) / fluxfactor**2)

        # Rename some WISE columns
        for cin,cout in [('w%i_nexp',        'nobs_w%i'),
                         ('w%i_profracflux', 'fracflux_w%i'),
                         ('w%i_prochi2',     'rchisq_w%i')]:
            for band in [1,2,3,4]:
                T2.set(cout % band, WISE.get(cin % band))

        if WISE_T is not None:
            for cin,cout in [('w%i_nexp',        'lc_nobs_w%i'),
                             ('w%i_profracflux', 'lc_fracflux_w%i'),
                             ('w%i_prochi2',     'lc_rchisq_w%i'),
                             ('w%i_mjd',         'lc_mjd_w%i'),]:
                for band in [1,2]:
                    T2.set(cout % band, WISE_T.get(cin % band))
        # Done with these now!
        WISE_T = None
        WISE = None

    if T_donotfit:
        T2 = merge_tables([T2, T_donotfit], columns='fillzero')

    # Brick pixel positions
    ok,bx,by = targetwcs.radec2pixelxy(T2.orig_ra, T2.orig_dec)
    # iterative sources
    bx[ok==False] = 1.
    by[ok==False] = 1.
    T2.bx0 = (bx - 1.).astype(np.float32)
    T2.by0 = (by - 1.).astype(np.float32)
    ok,bx,by = targetwcs.radec2pixelxy(T2.ra, T2.dec)
    T2.bx = (bx - 1.).astype(np.float32)
    T2.by = (by - 1.).astype(np.float32)

    T2.delete_column('orig_ra')
    T2.delete_column('orig_dec')

    T2.brick_primary = ((T2.ra  >= brick.ra1 ) * (T2.ra  < brick.ra2) *
                        (T2.dec >= brick.dec1) * (T2.dec < brick.dec2))
    H,W = maskbits.shape
    T2.maskbits = maskbits[np.clip(np.round(T2.by), 0, H-1).astype(int),
                           np.clip(np.round(T2.bx), 0, W-1).astype(int)]
    del maskbits

    # sigh, bytes vs strings.  In py3, T.type (dtype '|S3') are bytes.
    T2.sersic[np.array([t in ['DEV',b'DEV'] for t in T2.type])] = 4.0
    T2.sersic[np.array([t in ['EXP',b'EXP'] for t in T2.type])] = 1.0

    with survey.write_output('tractor-intermediate', brick=brickname) as out:
        T2.writeto(None, fits_object=out.fits, primheader=primhdr, header=hdr)

    # After writing tractor-i file, drop (reference) sources outside the brick.
    T2.cut((T2.bx >= -0.5) * (T2.bx <= W-0.5) *
           (T2.by >= -0.5) * (T2.by <= H-0.5))

    # The "format_catalog" code expects all lower-case column names...
    for c in T2.columns():
        if c != c.lower():
            T2.rename(c, c.lower())
    from legacypipe.format_catalog import format_catalog
    with survey.write_output('tractor', brick=brickname) as out:
        format_catalog(T2, hdr, primhdr, survey.allbands, None, release,
                       write_kwargs=dict(fits_object=out.fits),
                       N_wise_epochs=15, motions=gaia_stars, gaia_tagalong=True)

    # write fits file with galaxy-sim stuff (xy bounds of each sim)
    if 'sims_xy' in T.get_columns():
        sims_data = fits_table()
        sims_data.sims_xy = T.sims_xy
        with survey.write_output('galaxy-sims', brick=brickname) as out:
            sims_data.writeto(None, fits_object=out.fits)

    # produce per-brick checksum file.
    with survey.write_output('checksums', brick=brickname, hashsum=False) as out:
        f = open(out.fn, 'w')
        # Write our pre-computed hashcodes.
        for fn,hashsum in survey.output_file_hashes.items():
            f.write('%s *%s\n' % (hashsum, fn))
        f.close()

    record_event and record_event('stage_writecat: done')

    return dict(T2=T2, version_header=version_header)

def run_brick(brick, survey, radec=None, pixscale=0.262,
              width=3600, height=3600,
              survey_blob_mask=None,
              release=None,
              zoom=None,
              bands=None,
              allbands='grz',
              nblobs=None, blob=None, blobxy=None, blobradec=None, blobid=None,
              max_blobsize=None,
              nsigma=6,
              saddle_fraction=0.1,
              saddle_min=2.,
              reoptimize=False,
              iterative=False,
              wise=True,
              outliers=True,
              lanczos=True,
              early_coadds=False,
              blob_image=False,
              do_calibs=True,
              old_calibs_ok=False,
              write_metrics=True,
              gaussPsf=False,
              pixPsf=False,
              hybridPsf=False,
              normalizePsf=False,
              apodize=False,
              splinesky=True,
              subsky=True,
              constant_invvar=False,
              tycho_stars=True,
              gaia_stars=True,
              large_galaxies=True,
              large_galaxies_force_pointsource=True,
              fitoncoadds_reweight_ivar=True,
              less_masking=False,
              fit_on_coadds=False,
              min_mjd=None, max_mjd=None,
              unwise_coadds=True,
              bail_out=False,
              ceres=True,
              wise_ceres=True,
              unwise_dir=None,
              unwise_tr_dir=None,
              unwise_modelsky_dir=None,
              threads=None,
              plots=False, plots2=False, coadd_bw=False,
              plot_base=None, plot_number=0,
              command_line=None,
              record_event=None,
    # These are for the 'stages' infrastructure
              pickle_pat='pickles/runbrick-%(brick)s-%%(stage)s.pickle',
              stages=['writecat'],
              force=None, forceall=False, write_pickles=True,
              checkpoint_filename=None,
              checkpoint_period=None,
              prereqs_update=None,
              stagefunc = None,
              ):
    '''Run the full Legacy Survey data reduction pipeline.

    The pipeline is built out of "stages" that run in sequence.  By
    default, this function will cache the result of each stage in a
    (large) pickle file.  If you re-run, it will read from the
    prerequisite pickle file rather than re-running the prerequisite
    stage.  This can yield faster debugging times, but you almost
    certainly want to turn it off (with `writePickles=False,
    forceall=True`) in production.

    Parameters
    ----------
    brick : string
        Brick name such as '2090m065'.  Can be None if *radec* is given.
    survey : a "LegacySurveyData" object (see common.LegacySurveyData), which is in
        charge of the list of bricks and CCDs to be handled, and where output files
        should be written.
    radec : tuple of floats (ra,dec)
        RA,Dec center of the custom region to run.
    pixscale : float
        Brick pixel scale, in arcsec/pixel.  Default = 0.262
    width, height : integers
        Brick size in pixels.  Default of 3600 pixels (with the default pixel
        scale of 0.262) leads to a slight overlap between bricks.
    zoom : list of four integers
        Pixel coordinates [xlo,xhi, ylo,yhi] of the brick subimage to run.
    bands : string
        Filter (band) names to include; default is "grz".

    Notes
    -----
    You must specify the region of sky to work on, via one of:

    - *brick*: string, brick name such as '2090m065'
    - *radec*: tuple of floats; RA,Dec center of the custom region to run

    If *radec* is given, *brick* should be *None*.  If *brick* is given,
    that brick`s RA,Dec center will be looked up in the
    survey-bricks.fits file.

    You can also change the size of the region to reduce:

    - *pixscale*: float, brick pixel scale, in arcsec/pixel.
    - *width* and *height*: integers; brick size in pixels.  3600 pixels
      (with the default pixel scale of 0.262) leads to a slight overlap
      between bricks.
    - *zoom*: list of four integers, [xlo,xhi, ylo,yhi] of the brick
      subimage to run.

    If you want to measure only a subset of the astronomical objects,
    you can use:

    - *nblobs*: None or int; for debugging purposes, only fit the
       first N blobs.
    - *blob*: int; for debugging purposes, start with this blob index.
    - *blobxy*: list of (x,y) integer tuples; only run the blobs
      containing these pixels.
    - *blobradec*: list of (RA,Dec) tuples; only run the blobs
      containing these coordinates.

    Other options:

    - *max_blobsize*: int; ignore blobs with more than this many pixels

    - *nsigma*: float; detection threshold in sigmas.

    - *wise*: boolean; run WISE forced photometry?

    - *early_coadds*: boolean; generate the early coadds?

    - *do_calibs*: boolean; run the calibration preprocessing steps?

    - *old_calibs_ok*: boolean; allow/use old calibration frames?

    - *write_metrics*: boolean; write out a variety of useful metrics

    - *gaussPsf*: boolean; use a simpler single-component Gaussian PSF model?

    - *pixPsf*: boolean; use the pixelized PsfEx PSF model and FFT convolution?

    - *hybridPsf*: boolean; use combo pixelized PsfEx + Gaussian approx model

    - *normalizePsf*: boolean; make PsfEx model have unit flux

    - *splinesky*: boolean; use the splined sky model (default is constant)?

    - *subsky*: boolean; subtract the sky model when reading in tims (tractor images)?

    - *ceres*: boolean; use Ceres Solver when possible?

    - *wise_ceres*: boolean; use Ceres Solver for unWISE forced photometry?

    - *unwise_dir*: string; where to look for unWISE coadd files.
      This may be a colon-separated list of directories to search in
      order.

    - *unwise_tr_dir*: string; where to look for time-resolved
      unWISE coadd files.  This may be a colon-separated list of
      directories to search in order.

    - *unwise_modelsky_dir*: string; where to look for the unWISE sky background
      maps.  The default is to look in the "wise/modelsky" subdirectory of the
      calibration directory.

    - *threads*: integer; how many CPU cores to use

    Plotting options:

    - *coadd_bw*: boolean: if only one band is available, make B&W coadds?
    - *plots*: boolean; make a bunch of plots?
    - *plots2*: boolean; make a bunch more plots?
    - *plot_base*: string, default brick-BRICK, the plot filename prefix.
    - *plot_number*: integer, default 0, starting number for plot filenames.

    Options regarding the "stages":

    - *pickle_pat*: string; filename for 'pickle' files
    - *stages*: list of strings; stages (functions stage_*) to run.

    - *force*: list of strings; prerequisite stages that will be run
      even if pickle files exist.
    - *forceall*: boolean; run all stages, ignoring all pickle files.
    - *write_pickles*: boolean; write pickle files after each stage?

    Raises
    ------
    RunbrickError
        If an invalid brick name is given.
    NothingToDoError
        If no CCDs, or no photometric CCDs, overlap the given brick or region.

    '''
    from astrometry.util.stages import CallGlobalTime, runstage
    from astrometry.util.multiproc import multiproc
    from astrometry.util.plotutils import PlotSequence

    # *initargs* are passed to the first stage (stage_tims)
    # so should be quantities that shouldn't get updated from their pickled
    # values.
    initargs = {}
    # *kwargs* update the pickled values from previous stages
    kwargs = {}

    if force is None:
        force = []
    forceStages = [s for s in stages]
    forceStages.extend(force)
    if forceall:
        kwargs.update(forceall=True)

    if allbands is not None:
        survey.allbands = allbands

    if radec is not None:
        assert(len(radec) == 2)
        ra,dec = radec
        try:
            ra = float(ra)
        except:
            from astrometry.util.starutil_numpy import hmsstring2ra
            ra = hmsstring2ra(ra)
        try:
            dec = float(dec)
        except:
            from astrometry.util.starutil_numpy import dmsstring2dec
            dec = dmsstring2dec(dec)
        info('Parsed RA,Dec', ra,dec)
        initargs.update(ra=ra, dec=dec)
        if brick is None:
            brick = ('custom-%06i%s%05i' %
                         (int(1000*ra), 'm' if dec < 0 else 'p',
                          int(1000*np.abs(dec))))
    initargs.update(brickname=brick, survey=survey)

    if stagefunc is None:
        stagefunc = CallGlobalTime('stage_%s', globals())

    plot_base_default = 'brick-%(brick)s'
    if plot_base is None:
        plot_base = plot_base_default
    ps = PlotSequence(plot_base % dict(brick=brick))
    initargs.update(ps=ps)
    if plot_number:
        ps.skipto(plot_number)

    if release is None:
        release = survey.get_default_release()
        if release is None:
            release = 9999

    if fit_on_coadds:
        # Implied options!
        #subsky = False
        large_galaxies = True
        fitoncoadds_reweight_ivar = True
        large_galaxies_force_pointsource = False

    kwargs.update(ps=ps, nsigma=nsigma, saddle_fraction=saddle_fraction,
                  saddle_min=saddle_min,
                  survey_blob_mask=survey_blob_mask,
                  gaussPsf=gaussPsf, pixPsf=pixPsf, hybridPsf=hybridPsf,
                  release=release,
                  normalizePsf=normalizePsf,
                  apodize=apodize,
                  constant_invvar=constant_invvar,
                  splinesky=splinesky,
                  subsky=subsky,
                  tycho_stars=tycho_stars,
                  gaia_stars=gaia_stars,
                  large_galaxies=large_galaxies,
                  large_galaxies_force_pointsource=large_galaxies_force_pointsource,
                  fitoncoadds_reweight_ivar=fitoncoadds_reweight_ivar,
                  less_masking=less_masking,
                  min_mjd=min_mjd, max_mjd=max_mjd,
                  reoptimize=reoptimize,
                  iterative=iterative,
                  outliers=outliers,
                  use_ceres=ceres,
                  wise_ceres=wise_ceres,
                  unwise_coadds=unwise_coadds,
                  bailout=bail_out,
                  do_calibs=do_calibs,
                  old_calibs_ok=old_calibs_ok,
                  write_metrics=write_metrics,
                  lanczos=lanczos,
                  unwise_dir=unwise_dir,
                  unwise_tr_dir=unwise_tr_dir,
                  unwise_modelsky_dir=unwise_modelsky_dir,
                  command_line=command_line,
                  plots=plots, plots2=plots2, coadd_bw=coadd_bw,
                  force=forceStages, write=write_pickles,
                  record_event=record_event)

    if checkpoint_filename is not None:
        kwargs.update(checkpoint_filename=checkpoint_filename)
        if checkpoint_period is not None:
            kwargs.update(checkpoint_period=checkpoint_period)

    if threads and threads > 1:
        from astrometry.util.timingpool import TimingPool, TimingPoolMeas
        pool = TimingPool(threads, initializer=runbrick_global_init,
                          initargs=[])
        poolmeas = TimingPoolMeas(pool, pickleTraffic=False)
        StageTime.add_measurement(poolmeas)
        mp = multiproc(None, pool=pool)
    else:
        from astrometry.util.ttime import CpuMeas
        mp = multiproc(init=runbrick_global_init, initargs=[])
        StageTime.add_measurement(CpuMeas)
        pool = None
    kwargs.update(mp=mp)

    if nblobs is not None:
        kwargs.update(nblobs=nblobs)
    if blob is not None:
        kwargs.update(blob0=blob)
    if blobxy is not None:
        kwargs.update(blobxy=blobxy)
    if blobradec is not None:
        kwargs.update(blobradec=blobradec)
    if blobid is not None:
        kwargs.update(blobid=blobid)
    if max_blobsize is not None:
        kwargs.update(max_blobsize=max_blobsize)

    pickle_pat = pickle_pat % dict(brick=brick)

    prereqs = {
        'tims':None,
        'refs': 'tims',
        'outliers': 'refs',
        'halos': 'outliers',
        'srcs': 'halos',

        # fitblobs: see below

        'coadds': 'fitblobs',

        # wise_forced: see below

        'fitplots': 'fitblobs',
        'psfplots': 'tims',
        'initplots': 'srcs',

        }

    if 'image_coadds' in stages:
        early_coadds = True

    if early_coadds:
        if blob_image:
            prereqs.update({
                'image_coadds':'srcs',
                'fitblobs':'image_coadds',
                })
        else:
            prereqs.update({
                'image_coadds':'halos',
                'srcs':'image_coadds',
                'fitblobs':'srcs',
                })
    else:
        prereqs.update({
            'fitblobs':'srcs',
            })

    if wise:
        prereqs.update({
            'wise_forced': 'coadds',
            'writecat': 'wise_forced',
            })
    else:
        prereqs.update({
            'writecat': 'coadds',
            })

    if fit_on_coadds:
        prereqs.update({
            'fit_on_coadds': 'halos',
            'srcs': 'fit_on_coadds',
        })
        
    if prereqs_update is not None:
        prereqs.update(prereqs_update)

    initargs.update(W=width, H=height, pixscale=pixscale,
                    target_extent=zoom)
    if bands is not None:
        initargs.update(bands=bands)

    def mystagefunc(stage, mp=None, **kwargs):
        # Update the (pickled) survey output directory, so that running
        # with an updated --output-dir overrides the pickle file.
        picsurvey = kwargs.get('survey',None)
        if picsurvey is not None:
            picsurvey.output_dir = survey.output_dir

        flush()
        if mp is not None and threads is not None and threads > 1:
            # flush all workers too
            mp.map(flush, [[]] * threads)
        staget0 = StageTime()
        R = stagefunc(stage, mp=mp, **kwargs)
        flush()
        if mp is not None and threads is not None and threads > 1:
            mp.map(flush, [[]] * threads)
        info('Resources for stage', stage, ':', StageTime()-staget0)
        return R

    t0 = StageTime()
    R = None
    for stage in stages:
        R = runstage(stage, pickle_pat, mystagefunc, prereqs=prereqs,
                     initial_args=initargs, **kwargs)

    info('All done:', StageTime()-t0)
    return R

def flush(x=None):
    sys.stdout.flush()
    sys.stderr.flush()

class StageTime(Time):
    '''
    A Time subclass that reports overall CPU use, assuming multiprocessing.
    '''
    measurements = []
    @classmethod
    def add_measurement(cls, m):
        cls.measurements.append(m)
    def __init__(self):
        self.meas = [m() for m in self.measurements]

def get_parser():
    import argparse
    de = ('Main "pipeline" script for the Legacy Survey ' +
          '(DECaLS, MzLS, Bok) data reductions.')

    ep = '''
e.g., to run a small field containing a cluster:

python -u legacypipe/runbrick.py --plots --brick 2440p070 --zoom 1900 2400 450 950 -P pickles/runbrick-cluster-%%s.pickle

'''
    parser = argparse.ArgumentParser(description=de,epilog=ep)

    parser.add_argument('-r', '--run', default=None,
                        help='Set the run type to execute')

    parser.add_argument(
        '-f', '--force-stage', dest='force', action='append', default=[],
        help="Force re-running the given stage(s) -- don't read from pickle.")
    parser.add_argument('-F', '--force-all', dest='forceall',
                        action='store_true', help='Force all stages to run')
    parser.add_argument('-s', '--stage', dest='stage', default=[],
                        action='append', help="Run up to the given stage(s)")
    parser.add_argument('-n', '--no-write', dest='write', default=True,
                        action='store_false')
    parser.add_argument('-w', '--write-stage', action='append', default=None,
                        help='Write a pickle for a given stage: eg "tims", "image_coadds", "srcs"')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')

    parser.add_argument(
        '--checkpoint', dest='checkpoint_filename', default=None,
        help='Write to checkpoint file?')
    parser.add_argument(
        '--checkpoint-period', type=int, default=None,
        help='Period for writing checkpoint files, in seconds; default 600')

    parser.add_argument('-b', '--brick',
        help='Brick name to run; required unless --radec is given')

    parser.add_argument('--radec', nargs=2,
        help='RA,Dec center for a custom location (not a brick)')
    parser.add_argument('--pixscale', type=float, default=0.262,
                        help='Pixel scale of the output coadds (arcsec/pixel)')
    parser.add_argument('-W', '--width', type=int, default=3600,
                        help='Target image width, default %(default)i')
    parser.add_argument('-H', '--height', type=int, default=3600,
                        help='Target image height, default %(default)i')
    parser.add_argument('--zoom', type=int, nargs=4,
                        help='Set target image extent (default "0 3600 0 3600")')

    parser.add_argument('-d', '--outdir', dest='output_dir',
                        help='Set output base directory, default "."')

    parser.add_argument('--release', default=None, type=int,
                        help='Release code for output catalogs (default determined by --run)')

    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Override the $LEGACY_SURVEY_DIR environment variable')

    parser.add_argument('--blob-mask-dir', type=str, default=None,
                        help='The base directory to search for blob masks during sky model construction')

    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to search for cached files')

    parser.add_argument('--threads', type=int, help='Run multi-threaded')
    parser.add_argument('-p', '--plots', dest='plots', action='store_true',
                        help='Per-blob plots?')
    parser.add_argument('--plots2', action='store_true',
                        help='More plots?')

    parser.add_argument(
        '-P', '--pickle', dest='pickle_pat',
        help='Pickle filename pattern, default %(default)s',
        default='pickles/runbrick-%(brick)s-%%(stage)s.pickle')

    parser.add_argument('--plot-base',
                        help='Base filename for plots, default brick-BRICK')
    parser.add_argument('--plot-number', type=int, default=0,
                        help='Set PlotSequence starting number')

    parser.add_argument('--ceres', default=False, action='store_true',
                        help='Use Ceres Solver for all optimization?')

    parser.add_argument('--no-wise-ceres', dest='wise_ceres', default=True,
                        action='store_false',
                        help='Do not use Ceres Solver for unWISE forced phot')

    parser.add_argument('--nblobs', type=int,help='Debugging: only fit N blobs')
    parser.add_argument('--blob', type=int, help='Debugging: start with blob #')
    parser.add_argument('--blobid', help='Debugging: process this list of (comma-separated) blob ids.')
    parser.add_argument(
        '--blobxy', type=int, nargs=2, default=None, action='append',
        help=('Debugging: run the single blob containing pixel <bx> <by>; '+
              'this option can be repeated to run multiple blobs.'))
    parser.add_argument(
        '--blobradec', type=float, nargs=2, default=None, action='append',
        help=('Debugging: run the single blob containing RA,Dec <ra> <dec>; '+
              'this option can be repeated to run multiple blobs.'))

    parser.add_argument('--max-blobsize', type=int,
                        help='Skip blobs containing more than the given number of pixels.')

    parser.add_argument(
        '--check-done', default=False, action='store_true',
        help='Just check for existence of output files for this brick?')
    parser.add_argument('--skip', default=False, action='store_true',
                        help='Quit if the output catalog already exists.')
    parser.add_argument('--skip-coadd', default=False, action='store_true',
                        help='Quit if the output coadd jpeg already exists.')

    parser.add_argument(
        '--skip-calibs', dest='do_calibs', default=True, action='store_false',
        help='Do not run the calibration steps')

    parser.add_argument(
        '--old-calibs-ok', dest='old_calibs_ok', default=False, action='store_true',
        help='Allow old calibration files (where the data validation does not necessarily pass).')

    parser.add_argument('--skip-metrics', dest='write_metrics', default=True,
                        action='store_false',
                        help='Do not generate the metrics directory and files')

    parser.add_argument('--nsigma', type=float, default=6.0,
                        help='Set N sigma source detection thresh')

    parser.add_argument('--saddle-fraction', type=float, default=0.1,
                        help='Fraction of the peak height for selecting new sources.')

    parser.add_argument('--saddle-min', type=float, default=2.0,
                        help='Saddle-point depth from existing sources down to new sources (sigma).')

    parser.add_argument(
        '--reoptimize', action='store_true', default=False,
        help='Do a second round of model fitting after all model selections')

    parser.add_argument(
        '--no-iterative', dest='iterative', action='store_false', default=True,
        help='Turn off iterative source detection?')

    parser.add_argument('--no-wise', dest='wise', default=True,
                        action='store_false',
                        help='Skip unWISE forced photometry')

    parser.add_argument(
        '--unwise-dir', default=None,
        help='Base directory for unWISE coadds; may be a colon-separated list')
    parser.add_argument(
        '--unwise-tr-dir', default=None,
        help='Base directory for unWISE time-resolved coadds; may be a colon-separated list')

    parser.add_argument('--early-coadds', action='store_true', default=False,
                        help='Make early coadds?')
    parser.add_argument('--blob-image', action='store_true', default=False,
                        help='Create "imageblob" image?')

    parser.add_argument(
        '--no-lanczos', dest='lanczos', action='store_false', default=True,
        help='Do nearest-neighbour rather than Lanczos-3 coadds')

    parser.add_argument('--gpsf', action='store_true', default=False,
                        help='Use a fixed single-Gaussian PSF')

    parser.add_argument('--no-hybrid-psf', dest='hybridPsf', default=True,
                        action='store_false',
                        help="Don't use a hybrid pixelized/Gaussian PSF model")

    parser.add_argument('--no-normalize-psf', dest='normalizePsf', default=True,
                        action='store_false',
                        help='Do not normalize the PSF model to unix flux')

    parser.add_argument('--apodize', default=False, action='store_true',
                        help='Apodize image edges for prettier pictures?')

    parser.add_argument(
        '--coadd-bw', action='store_true', default=False,
        help='Create grayscale coadds if only one band is available?')

    parser.add_argument('--bands', default=None,
                        help='Set the list of bands (filters) that are included in processing: comma-separated list, default "g,r,z"')

    parser.add_argument('--no-tycho', dest='tycho_stars', default=True,
                        action='store_false',
                        help="Don't use Tycho-2 sources as fixed stars")

    parser.add_argument('--no-gaia', dest='gaia_stars', default=True,
                        action='store_false',
                        help="Don't use Gaia sources as fixed stars")

    parser.add_argument('--no-large-galaxies', dest='large_galaxies', default=True,
                        action='store_false', help="Don't seed (or mask in and around) large galaxies.")
    # HACK -- Default value for DR8 MJD cut
    # DR8 -- drop early data from before additional baffling was added to the camera.
    # 56730 = 2014-03-14
    parser.add_argument('--min-mjd', type=float,
                        help='Only keep images taken after the given MJD')
    parser.add_argument('--max-mjd', type=float,
                        help='Only keep images taken before the given MJD')

    parser.add_argument('--no-splinesky', dest='splinesky', default=True,
                        action='store_false', help='Use constant sky rather than spline.')
    parser.add_argument('--no-unwise-coadds', dest='unwise_coadds', default=True,
                        action='store_false', help='Turn off writing FITS and JPEG unWISE coadds?')
    parser.add_argument('--no-outliers', dest='outliers', default=True,
                        action='store_false', help='Do not compute or apply outlier masks')

    parser.add_argument('--bail-out', default=False, action='store_true',
                        help='Bail out of "fitblobs" processing, writing all blobs from the checkpoint and skipping any remaining ones.')

    parser.add_argument('--fit-on-coadds', default=False, action='store_true',
                        help='Fit to coadds rather than individual CCDs (e.g., large galaxies).')
    parser.add_argument('--no-ivar-reweighting', dest='fitoncoadds_reweight_ivar',
                        default=True, action='store_false',
                        help='Reweight the inverse variance when fitting on coadds.')
    parser.add_argument('--no-galaxy-forcepsf', dest='large_galaxies_force_pointsource',
                        default=True, action='store_false',
                        help='Do not force PSFs within galaxy mask.')

    parser.add_argument('--less-masking', default=False, action='store_true',
                        help='Reduce size of BRIGHT mask, and turn off MEDIUM mask behaviors.')

    return parser

def get_runbrick_kwargs(survey=None,
                        brick=None,
                        radec=None,
                        run=None,
                        survey_dir=None,
                        output_dir=None,
                        cache_dir=None,
                        check_done=False,
                        skip=False,
                        skip_coadd=False,
                        stage=[],
                        unwise_dir=None,
                        unwise_tr_dir=None,
                        unwise_modelsky_dir=None,
                        write_stage=None,
                        write=True,
                        gpsf=False,
                        bands=None,
                        **opt):
    if brick is not None and radec is not None:
        print('Only ONE of --brick and --radec may be specified.')
        return None, -1
    opt.update(radec=radec)

    if survey is None:
        from legacypipe.runs import get_survey
        survey = get_survey(run,
                            survey_dir=survey_dir,
                            output_dir=output_dir,
                            cache_dir=cache_dir)
        info(survey)

    blobdir = opt.pop('blob_mask_dir', None)
    if blobdir is not None:
        from legacypipe.survey import LegacySurveyData
        opt.update(survey_blob_mask=LegacySurveyData(blobdir))

    if check_done or skip or skip_coadd:
        if skip_coadd:
            fn = survey.find_file('image-jpeg', output=True, brick=brick)
        else:
            fn = survey.find_file('tractor', output=True, brick=brick)
        info('Checking for', fn)
        exists = os.path.exists(fn)
        if skip_coadd and exists:
            return survey,0
        if exists:
            try:
                T = fits_table(fn)
                info('Read', len(T), 'sources from', fn)
            except:
                print('Failed to read file', fn)
                import traceback
                traceback.print_exc()
                exists = False

        if skip:
            if exists:
                return survey,0
        elif check_done:
            if not exists:
                print('Does not exist:', fn)
                return survey,-1
            info('Found:', fn)
            return survey,0

    if len(stage) == 0:
        stage.append('writecat')

    opt.update(stages=stage)

    # Remove opt values that are None.
    toremove = [k for k,v in opt.items() if v is None]
    for k in toremove:
        del opt[k]

    if unwise_dir is None:
        unwise_dir = os.environ.get('UNWISE_COADDS_DIR', None)
    if unwise_tr_dir is None:
        unwise_tr_dir = os.environ.get('UNWISE_COADDS_TIMERESOLVED_DIR', None)
    if unwise_modelsky_dir is None:
        unwise_modelsky_dir = os.environ.get('UNWISE_MODEL_SKY_DIR', None)
        if unwise_modelsky_dir is not None and not os.path.exists(unwise_modelsky_dir):
            raise RuntimeError('The directory specified in $UNWISE_MODEL_SKY_DIR does not exist!')
    opt.update(unwise_dir=unwise_dir, unwise_tr_dir=unwise_tr_dir, unwise_modelsky_dir=unwise_modelsky_dir)

    # list of strings if -w / --write-stage is given; False if
    # --no-write given; True by default.
    if write_stage is not None:
        write_pickles = write_stage
    else:
        write_pickles = write
    opt.update(write_pickles=write_pickles)

    opt.update(gaussPsf=gpsf,
               pixPsf=not gpsf)

    if bands is not None:
        bands = bands.split(',')
    opt.update(bands=bands)
    return survey, opt

def main(args=None):
    import datetime
    from astrometry.util.ttime import MemMeas
    from legacypipe.survey import get_git_version

    print()
    print('runbrick.py starting at', datetime.datetime.now().isoformat())
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
    parser.add_argument(
        '--ps', help='Run "ps" and write results to given filename?')
    parser.add_argument(
        '--ps-t0', type=int, default=0, help='Unix-time start for "--ps"')

    opt = parser.parse_args(args=args)

    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    optdict = vars(opt)
    ps_file = optdict.pop('ps', None)
    ps_t0   = optdict.pop('ps_t0', 0)
    verbose = optdict.pop('verbose')

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

    Time.add_measurement(MemMeas)
    if opt.plots:
        plt.figure(figsize=(12,9))
        plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.93,
                            hspace=0.2, wspace=0.05)

    if ps_file is not None:
        import threading
        from collections import deque
        from legacypipe.utils import run_ps_thread
        ps_shutdown = threading.Event()
        ps_queue = deque()
        def record_event(msg):
            from time import time
            ps_queue.append((time(), msg))
        kwargs.update(record_event=record_event)
        if ps_t0 > 0:
            record_event('start')

        ps_thread = threading.Thread(
            target=run_ps_thread,
            args=(os.getpid(), os.getppid(), ps_file, ps_shutdown, ps_queue),
            name='run_ps')
        ps_thread.daemon = True
        print('Starting thread to run "ps"')
        ps_thread.start()

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

    if ps_file is not None:
        # Try to shut down ps thread gracefully
        ps_shutdown.set()
        print('Attempting to join the ps thread...')
        ps_thread.join(1.0)
        if ps_thread.isAlive():
            print('ps thread is still alive.')

    return rtn

if __name__ == '__main__':
    sys.exit(main())

# Test bricks & areas

# A single, fairly bright star
# python -u legacypipe/runbrick.py -b 1498p017 -P 'pickles/runbrick-z-%(brick)s-%%(stage)s.pickle' --zoom 1900 2000 2700 2800
# python -u legacypipe/runbrick.py -b 0001p000 -P 'pickles/runbrick-z-%(brick)s-%%(stage)s.pickle' --zoom 80 380 2970 3270
