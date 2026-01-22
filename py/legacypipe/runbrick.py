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
- :py:func:`stage_galex_forced` [optional]
- :py:func:`stage_writecat`

To see the code we run on each "blob" of pixels, see "oneblob.py".

- :py:func:`one_blob`

'''
import sys
import os
import time
import warnings

import numpy as np

import fitsio

from collections import Counter

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.ttime import Time

from legacypipe.survey import imsave_jpeg
from legacypipe.bits import DQ_BITS, FITBITS
from legacypipe.utils import RunbrickError, NothingToDoError, find_unique_pixels
from legacypipe.coadds import make_coadds, write_coadd_images, quick_coadds
from legacypipe.fit_on_coadds import stage_fit_on_coadds
from legacypipe.blobmask import stage_blobmask
from legacypipe.galex import stage_galex_forced
import multiprocessing
import queue
import traceback
import threading
import time
import datetime


_GLOBAL_LEGACYPIPE_CONTEXT = {'is_gpu_worker': False, 'gpu_device_id': None, 'gpumode': 0}

import logging
logger = logging.getLogger('legacypipe.runbrick')
def info(*args):
    from legacypipe.utils import log_info
    log_info(logger, args)
def debug(*args):
    from legacypipe.utils import log_debug
    log_debug(logger, args)

def formatwarning(message, category, filename, lineno, line=None):
    #return 'Warning: %s (%s:%i)' % (message, filename, lineno)
    return 'Warning: %s' % (message)

warnings.formatwarning = formatwarning

#New init method is cleaner - GPU assignment now occurs in _blobiter and _bounce_one_blob
def runbrick_global_init(shared_counter, shared_lock, available_gpu_ids_param,
                          ngpu_param, threads_per_gpu_param, gpumode_param):
    import os
    import time
    from tractor.galaxy import disable_galaxy_cache

    # Access the module-level global dictionary
    global _GLOBAL_LEGACYPIPE_CONTEXT

    pid = os.getpid()

    # 1. Store shared objects and config for the 'Lazy' phase
    _GLOBAL_LEGACYPIPE_CONTEXT['shared_counter'] = shared_counter
    _GLOBAL_LEGACYPIPE_CONTEXT['shared_lock'] = shared_lock
    _GLOBAL_LEGACYPIPE_CONTEXT['available_gpu_ids'] = available_gpu_ids_param
    _GLOBAL_LEGACYPIPE_CONTEXT['max_gpu_slots'] = ngpu_param * threads_per_gpu_param
    _GLOBAL_LEGACYPIPE_CONTEXT['target_gpumode'] = gpumode_param

    # 2. Initialize State Flags
    # These will be updated by the first task this worker receives
    _GLOBAL_LEGACYPIPE_CONTEXT['initialized'] = False
    _GLOBAL_LEGACYPIPE_CONTEXT['is_gpu_worker'] = False
    _GLOBAL_LEGACYPIPE_CONTEXT['gpu_device_id'] = None
    _GLOBAL_LEGACYPIPE_CONTEXT['gpumode'] = 0 # Default to CPU mode

    # 3. Perform necessary one-time CPU setup
    disable_galaxy_cache()

    # logging (using info if available in your scope, or print)
    print(f"Worker PID {pid}: Basic initialization complete. Awaiting first task...")


def stage_tims(W=3600, H=3600, pixscale=0.262, brickname=None,
               survey=None,
               survey_blob_mask=None,
               sky_subtract_large_galaxies=True,
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

    #ccds.writeto('ccds.fits')
        
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

    if do_calibs:
        from legacypipe.survey import run_calibs
        record_event and record_event('stage_tims: starting calibs')
        kwa = dict(git_version=gitver, survey=survey,
                   old_calibs_ok=old_calibs_ok,
                   survey_blob_mask=survey_blob_mask,
                   subtract_largegalaxies=sky_subtract_large_galaxies,
                   ps=(ps if plots else None),
                   splinesky=splinesky)
        if gaussPsf:
            kwa.update(psfex=False)
        if not gaia_stars:
            kwa.update(gaia=False)

        # Run calibrations
        args = [(im, kwa) for im in ims]
        mp.map(run_calibs, args)
        tnow = Time()
        debug('Calibrations:', tnow-tlast)
        tlast = tnow

    # Read Tractor images
    args = [(im, targetrd, dict(gaussPsf=gaussPsf, pixPsf=pixPsf,
                                hybridPsf=hybridPsf, normalizePsf=normalizePsf,
                                subsky=subsky,
                                apodize=apodize,
                                constant_invvar=constant_invvar,
                                pixels=read_image_pixels,
                                old_calibs_ok=old_calibs_ok,
                                plots=plots, ps=ps))
                                for im in ims]
    record_event and record_event('stage_tims: starting read_tims')
    if read_parallel:
        tims = list(mp.map(read_one_tim, args))
    else:
        tims = list(map(read_one_tim, args))
    record_event and record_event('stage_tims: done read_tims')

    tnow = Time()
    debug('Read', len(ccds), 'images:', tnow-tlast)
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
                warnings.warn(('Image "%s" PLVER is "%s" but %s calib was run'
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

    # Add header cards about the survey-ccds files that were used.
    fns = survey.find_file('ccd-kds')
    fns = survey.filter_ccd_kd_files(fns)
    if len(fns) == 0:
        fns = survey.find_file('ccds')
        fns.sort()
        fns = survey.filter_ccds_files(fns)
    for i,fn in enumerate(fns):
        version_header.add_record(dict(
            name='CCDFN_%i' % (i+1), value=fn,
            comment='survey-ccds file used'))

    # Add header cards about which bands and cameras are involved.
    for band in survey.allbands:
        hasit = band in bands
        version_header.add_record(dict(
            name='BRICK_%s' % band.upper(), value=hasit,
            comment='Does band %s touch this brick?' % band))

        cams = np.unique([tim.imobj.camera for tim in tims
                          if tim.band == band])
        version_header.add_record(dict(
            name='CAMS_%s' % band.upper(), value=' '.join(cams),
            comment='Cameras contributing band %s' % band))
    version_header.add_record(dict(name='BANDS', value=','.join(bands),
                                   comment='Bands touching this brick'))
    version_header.add_record(dict(name='NBANDS', value=len(bands),
                                   comment='Number of bands in this catalog'))
    for i,band in enumerate(bands):
        version_header.add_record(dict(name='BAND%i' % i, value=band,
                                       comment='Band name in this catalog'))

    _add_stage_version(version_header, 'TIMS', 'tims')
    keys = ['version_header', 'targetrd', 'pixscale', 'targetwcs', 'W','H',
            'tims', 'ps', 'brickid', 'brickname', 'brick', 'custom_brick',
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
               forced_bands=None,
               version_header=None,
               tycho_stars=True,
               gaia_stars=True,
               large_galaxies=True,
               star_clusters=True,
               plots=False, ps=None,
               record_event=None,
               tims=None,
               **kwargs):
    from legacypipe.reference import get_reference_sources

    record_event and record_event('stage_refs: starting')
    _add_stage_version(version_header, 'REFS', 'refs')

    thebands = bands
    if forced_bands is not None:
        thebands = thebands + forced_bands

    refobjs,refcat = get_reference_sources(survey, targetwcs, thebands,
                                            tycho_stars=tycho_stars,
                                            gaia_stars=gaia_stars,
                                            large_galaxies=large_galaxies,
                                            star_clusters=star_clusters,
                                            plots=plots, ps=ps)
    # "refobjs" is a table
    # "refcat" is a list of tractor Sources
    # They are aligned
    if refobjs:
        from legacypipe.units import get_units_for_columns
        assert(len(refobjs) == len(refcat))
        cols = ['ra', 'dec', 'ref_cat', 'ref_id', 'mag',
                'istycho', 'isgaia', 'islargegalaxy', 'iscluster',
                'isbright', 'ismedium', 'freezeparams', 'pointsource', 'ignore_source', 'in_bounds',
                'fitmode', 'isresolved', 'ismcloud',
                'ba', 'pa', 'decam_mag_g', 'decam_mag_r', 'decam_mag_i', 'decam_mag_z',
                'zguess', 'mask_mag', 'radius', 'keep_radius', 'radius_pix', 'ibx', 'iby',
                'ref_epoch', 'pmra', 'pmdec', 'parallax',
                'ra_ivar', 'dec_ivar', 'pmra_ivar', 'pmdec_ivar', 'parallax_ivar',
                # Gaia
                'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'phot_g_mean_flux_over_error',
                'phot_bp_mean_flux_over_error', 'phot_rp_mean_flux_over_error', 'phot_g_n_obs',
                'phot_bp_n_obs', 'phot_rp_n_obs', 'phot_variable_flag', 'astrometric_excess_noise',
                'astrometric_excess_noise_sig', 'astrometric_n_obs_al', 'astrometric_n_good_obs_al',
                'astrometric_weight_al', 'duplicated_source', 'a_g_val', 'e_bp_min_rp_val',
                'phot_bp_rp_excess_factor', 'astrometric_sigma5d_max', 'astrometric_params_solved',
                ]
        # Drop columns that don't exist (because one of the ref catalogs has no entries or is
        # not being used)
        refcols = refobjs.get_columns()
        cols = [c for c in cols if c in refcols]
        extra_units = dict(zguess='mag', pa='deg', radius='deg', keep_radius='deg')
        units = get_units_for_columns(cols, extras=extra_units)
        with survey.write_output('ref-sources', brick=brickname) as out:
            refobjs.writeto(None, fits_object=out.fits, primheader=version_header,
                             columns=cols, units=units, extname='REFERENCES')

    if plots and refobjs:
        import pylab as plt
        from tractor import Tractor
        for tim in tims:
            I = np.flatnonzero(refobjs.istycho | refobjs.isgaia)
            stars = refobjs[I]
            info(len(stars), 'ref stars')
            stars.index = I
            ok,xx,yy = tim.subwcs.radec2pixelxy(stars.ra, stars.dec)
            xx -= 1.
            yy -= 1.
            stars.xx = xx
            stars.yy = yy
            h,w = tim.shape
            edge = 25
            stars.cut((xx > edge) * (yy > edge) * (xx < w-1-edge) * (yy < h-1-edge))
            info(len(stars), 'are within tim', tim.name)
            K = np.argsort(stars.mag)
            stars.cut(K)
            plt.clf()
            for i in range(len(stars)):
                if i >= 5:
                    break
                src = refcat[stars.index[i]].copy()
                tr = Tractor([tim], [src])
                tr.freezeParam('images')
                src.freezeAllBut('brightness')
                src.getBrightness().freezeAllBut(tim.band)
                try:
                    from tractor.ceres_optimizer import CeresOptimizer
                    # import the sub-package that actually pulls in _ceres.so
                    from tractor.ceres import ceres_forced_phot
                    ceres_block = 8
                    tr.optimizer = CeresOptimizer(BW=ceres_block, BH=ceres_block)
                except ImportError:
                    from tractor.lsqr_optimizer import LsqrOptimizer
                    tr.optimizer = LsqrOptimizer()
                R = tr.optimize_forced_photometry(shared_params=False, wantims=True)
                src.thawAllParams()
                y = int(stars.yy[i])
                x = int(stars.xx[i])
                sz = edge
                sl = slice(y-sz, y+sz+1), slice(x-sz, x+sz+1)
                for data,mod,ie,chi,roi in getattr(R, 'ims1', []):
                    print('x,y', x, y, 'tim shape', tim.shape, 'slice', sl,
                          'roi', roi, 'data size', data.shape)
                    subimg = data[sl]
                    mn,mx = np.percentile(subimg.ravel(), [25,99])
                    mx = subimg.max()
                    ima = dict(origin='lower', interpolation='nearest', vmin=mn, vmax=mx)
                    plt.subplot(3,5, 1 + i)
                    plt.imshow(data[sl], **ima)
                    plt.subplot(3,5, 1 + 5 + i)
                    plt.imshow(mod[sl], **ima)
                    plt.subplot(3,5, 1 + 2*5 + i)
                    plt.imshow(chi[sl], origin='lower', interpolation='nearest', vmin=-5, vmax=+5)
            plt.suptitle('Ref stars: %s' % tim.name)
            ps.savefig()

    keys = ['refobjs', 'refcat', 'gaia_stars', 'version_header']
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def stage_outliers(tims=None, targetwcs=None, W=None, H=None, bands=None,
                   mp=None, nsigma=None, plots=None, ps=None, record_event=None,
                   survey=None, brickname=None, version_header=None,
                   refobjs=None, outlier_mask_file=None,
                   outliers=True, cache_outliers=False, remake_outlier_jpegs=False,
                   **kwargs):
    '''This pipeline stage tries to detect artifacts in the individual
    exposures, by blurring all images in the same band to the same PSF size,
    then searching for outliers.

    *cache_outliers*: bool: if the outliers-mask*.fits.fz file exists
    (from a previous run), use it.  We turn this off in production
    because we still want to create the JPEGs and the checksum entry
    for the outliers file.
    '''
    from legacypipe.outliers import patch_from_coadd, mask_outlier_pixels, read_outlier_mask_file

    record_event and record_event('stage_outliers: starting')
    _add_stage_version(version_header, 'OUTL', 'outliers')

    version_header.add_record(dict(name='OUTLIER',
                                   value=outliers,
                                   help='Are we applying outlier rejection?'))

    # Check for existing MEF containing masks for all the chips we need.
    if (outliers and
        (remake_outlier_jpegs or
         (not (cache_outliers and
               read_outlier_mask_file(survey, tims, brickname, outlier_mask_file=outlier_mask_file,
                                      output='both'))))):
        # Make before-n-after plots (before)
        t0 = Time()
        C = make_coadds(tims, bands, targetwcs, mp=mp, sbscale=False,
                        allmasks=False, coweights=False)
        with survey.write_output('outliers-pre', brick=brickname) as out:
            rgb,kwa = survey.get_rgb(C.coimgs, bands)
            imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
            del rgb
        info('"Before" coadds:', Time()-t0)

        # Patch individual-CCD masked pixels from a coadd
        patch_from_coadd(C.coimgs, targetwcs, bands, tims, mp=mp)
        del C

        run_outliers = not(remake_outlier_jpegs)

        if remake_outlier_jpegs:
            from legacypipe.outliers import recreate_outlier_jpegs
            ok = recreate_outlier_jpegs(survey, tims, bands, targetwcs, brickname)
            if not ok:
                run_outliers = True
        if run_outliers:
            t0 = Time()
            make_badcoadds = True
            badcoaddspos, badcoaddsneg = mask_outlier_pixels(
                survey, tims, bands, targetwcs, brickname, version_header,
                mp=mp, plots=plots, ps=ps, make_badcoadds=make_badcoadds, refobjs=refobjs)
            info('Masking outliers:', Time()-t0)

        # Make before-n-after plots (after)
        t0 = Time()
        C = make_coadds(tims, bands, targetwcs, mp=mp, sbscale=False,
                        allmasks=False, coweights=False)
        with survey.write_output('outliers-post', brick=brickname) as out:
            rgb,kwa = survey.get_rgb(C.coimgs, bands)
            imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
            del rgb
        del C
        if run_outliers:
            with survey.write_output('outliers-masked-pos', brick=brickname) as out:
                rgb,kwa = survey.get_rgb(badcoaddspos, bands)
                imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
                del rgb
            del badcoaddspos
            with survey.write_output('outliers-masked-neg', brick=brickname) as out:
                rgb,kwa = survey.get_rgb(badcoaddsneg, bands)
                imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
                del rgb
            del badcoaddsneg
        info('"After" coadds:', Time()-t0)

    return dict(tims=tims, version_header=version_header)

def stage_halos(pixscale=None, targetwcs=None,
                W=None,H=None,
                bands=None, ps=None, tims=None,
                plots=False, plots2=False,
                brickname=None,
                version_header=None,
                mp=None, nsigma=None,
                survey=None, brick=None,
                refobjs=None,
                star_halos=True,
                old_calibs_ok=True,
                record_event=None,
                **kwargs):
    record_event and record_event('stage_halos: starting')
    _add_stage_version(version_header, 'HALO', 'halos')

    # Subtract star halos?
    if star_halos and refobjs:
        Igaia, = np.nonzero(refobjs.isgaia * refobjs.pointsource)
        debug(len(Igaia), 'stars for halo subtraction')
        if len(Igaia):
            from legacypipe.halos import subtract_halos
            halostars = refobjs[Igaia]

            if plots:
                from legacypipe.runbrick_plots import halo_plots_before, halo_plots_after
                coimgs = halo_plots_before(tims, bands, targetwcs, halostars, ps)

            subtract_halos(tims, halostars, bands, mp, plots, ps, old_calibs_ok=old_calibs_ok)

            if plots:
                halo_plots_after(tims, bands, targetwcs, halostars, coimgs, ps)

    return dict(tims=tims, version_header=version_header)

def stage_image_coadds(survey=None, targetwcs=None, bands=None, tims=None,
                       brickname=None, version_header=None,
                       plots=False, ps=None, coadd_bw=False, W=None, H=None,
                       brick=None, blobmap=None, lanczos=True, ccds=None,
                       write_metrics=True,
                       minimal_coadds=False,
                       mp=None, record_event=None,
                       co_sky=None,
                       custom_brick=False,
                       refobjs=None,
                       saturated_pix=None,
                       less_masking=False,
                       **kwargs):
    from legacypipe.utils import copy_header_with_wcs
    record_event and record_event('stage_image_coadds: starting')
    '''
    Immediately after reading the images, we can create coadds of just
    the image products.  Later, full coadds including the models will
    be created (in `stage_coadds`).  But it's handy to have the coadds
    early on, to diagnose problems or just to look at the data.
    '''
    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='ccdinfo',
                            comment='NOIRLab data product type'))
    # Write per-brick CCDs table
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=primhdr, extname='CCDS')

    if plots:
        import pylab as plt
        assert(len(ccds) == len(tims))
        # Make per-exposure coadd jpeg
        expnums = np.unique(ccds.expnum)
        for e in expnums:
            I = np.flatnonzero(ccds.expnum == e)
            info('Coadding', len(I), 'exposures with expnum =', e)
            bb = [tims[I[0]].band]
            C = make_coadds([tims[i] for i in I], bb, targetwcs, lanczos=lanczos,
                            mp=mp, plots=False, ps=None, allmasks=False)
            rgb,kwa = survey.get_rgb(C.coimgs, bb, coadd_bw=True)
            plt.clf()
            plt.imshow(rgb, origin='lower', interpolation='nearest')
            plt.title('Expnum %s %s' % (e, ''.join(bb)))
            ps.savefig()

    kw = dict(ngood=True, coweights=False)
    if minimal_coadds:
        kw.update(allmasks=False)
    else:
        kw.update(detmaps=True)

    C = make_coadds(tims, bands, targetwcs, lanczos=lanczos,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, tims,
                                   targetwcs, co_sky),
                    mp=mp, plots=plots, ps=ps, **kw)

    if not minimal_coadds:
        # interim maskbits
        from legacypipe.bits import REF_MAP_BITS, maskbits_type
        from legacypipe.survey import clean_band_name

        MASKBITS = survey.get_maskbits()

        refmap = get_blobiter_ref_map(refobjs, less_masking, targetwcs)
        # Construct a mask bits map
        maskbits = np.zeros((H,W), maskbits_type)
        # !PRIMARY
        if not custom_brick:
            U = find_unique_pixels(targetwcs, W, H, None,
                                   brick.ra1, brick.ra2, brick.dec1, brick.dec2)
            maskbits |= MASKBITS['NPRIMARY'] * np.logical_not(U).astype(maskbits_type)
            del U
        # reference-catalog masks
        if refmap is not None:
            for key in ['BRIGHT', 'MEDIUM', 'GALAXY', 'CLUSTER', 'RESOLVED', 'MCLOUDS']:
                maskbits |= MASKBITS[key] * ((refmap & REF_MAP_BITS[key]) > 0)
            del refmap

        cleanbands = [clean_band_name(b).upper() for b in bands]
        # SATUR
        if saturated_pix is not None:
            for b, sat in zip(cleanbands, saturated_pix):
                bitname = 'SATUR_' + b
                if not bitname in MASKBITS:
                    warnings.warn('Skipping SATUR mask for band %s' % b)
                    continue
                maskbits |= (maskbits_type(MASKBITS[bitname]) * sat).astype(maskbits_type)
        # ALLMASK_{g,r,z}
        for b,allmask in zip(cleanbands, C.allmasks):
            bitname = 'ALLMASK_' + b
            if not bitname in MASKBITS:
                warnings.warn('Skipping ALLMASK for band %s' % b)
                continue
            maskbits |= (maskbits_type(MASKBITS[bitname]) * (allmask > 0)).astype(maskbits_type)

        # omitting WISE, BAILOUT, SUB_BLOB

        # Add the maskbits header cards to version_header
        hdr = copy_header_with_wcs(version_header, targetwcs)
        mbits = survey.get_maskbits_descriptions()
        hdr.add_record(dict(name='COMMENT', value='maskbits bits:'))
        _add_bit_description(hdr, MASKBITS, mbits, 'MB_%s', 'MBIT_%i', 'maskbit')
        with survey.write_output('maskbits', brick=brickname, shape=maskbits.shape) as out:
            out.fits.write(maskbits, header=hdr, extname='MASKBITS')

    # Sims: coadds of galaxy sims only, image only
    if hasattr(tims[0], 'sims_image'):
        sims_coadd,_ = quick_coadds(
            tims, bands, targetwcs, images=[tim.sims_image for tim in tims])

    if not minimal_coadds:
        D = _depth_histogram(brick, targetwcs, bands, C.psfdetivs, C.galdetivs)
        with survey.write_output('depth-table', brick=brickname) as out:
            D.writeto(None, fits_object=out.fits, extname='DEPTH')
        del D

    coadd_list= [('image', C.coimgs)]
    if hasattr(tims[0], 'sims_image'):
        coadd_list.append(('simscoadd', sims_coadd))

    for name,ims in coadd_list:
        rgb,kwa = survey.get_rgb(ims, bands)
        del ims
        with survey.write_output(name + '-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
            debug('Wrote', out.fn)

        # Blob-outlined version
        if blobmap is not None:
            from scipy.ndimage import binary_dilation
            outline = np.logical_xor(
                binary_dilation(blobmap >= 0, structure=np.ones((3,3))),
                (blobmap >= 0))
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
                hdr = copy_header_with_wcs(version_header, targetwcs)
                hdr.add_record(dict(name='IMTYPE', value='blobmap',
                                    comment='LegacySurveys image type'))
                with survey.write_output('blobmap', brick=brickname,
                                         shape=blobmap.shape) as out:
                    out.fits.write(blobmap, header=hdr)
        del rgb
    del coadd_list
    del C
    return None

def stage_srcs(pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               brickname=None,
               version_header=None,
               mp=None, nsigma=None,
               saddle_fraction=None,
               saddle_min=None,
               survey=None, brick=None,
               refcat=None, refobjs=None,
               ccds=None,
               ubercal_sky=False,
               nsatur=None,
               record_event=None,
               large_galaxies=True,
               gaia_stars=True,
               blob_dilate=None,
               **kwargs):
    '''
    In this stage we run SED-matched detection to find objects in the
    images.  For each object detected, a `tractor` source object is
    created, initially a `tractor.PointSource`.  In this stage, the
    sources are also split into "blobs" of overlapping pixels.  Each
    of these blobs will be processed independently.
    '''
    from functools import reduce
    from tractor import Catalog
    from legacypipe.detection import (detection_maps, merge_hot_satur,
                        run_sed_matched_filters, segment_and_group_sources)
    from legacypipe.reference import get_reference_map
    from scipy.ndimage import binary_dilation

    record_event and record_event('stage_srcs: starting')
    _add_stage_version(version_header, 'SRCS', 'srcs')
    tlast = Time()

    avoid_map = None
    avoid_xyr = []
    if refobjs:
        # Don't detect new sources where we already have reference stars
        # To treat fast-moving stars, we evaluate proper motions at each image
        # epoch and exclude the set of integer pixel locations.
        # Init with ref sources without proper motions:
        I = np.flatnonzero(refobjs.in_bounds * (refobjs.ref_epoch == 0) *
                           np.logical_not(refobjs.islargegalaxy))
        xy = set(zip(refobjs.ibx[I], refobjs.iby[I]))
        ns = len(xy)
        # For moving stars, evaluate position at epoch of each input image
        I = np.flatnonzero(refobjs.in_bounds * (refobjs.ref_epoch > 0) *
                           np.logical_not(refobjs.islargegalaxy))
        if len(I):
            from legacypipe.survey import radec_at_mjd
            for tim in tims:
                ra,dec = radec_at_mjd(
                    refobjs.ra[I], refobjs.dec[I], refobjs.ref_epoch[I].astype(float),
                    refobjs.pmra[I], refobjs.pmdec[I], refobjs.parallax[I],
                    tim.time.toMjd())
                _,xx,yy = targetwcs.radec2pixelxy(ra, dec)
                xy.update(zip(np.round(xx-1.).astype(int), np.round(yy-1.).astype(int)))
        debug('Avoiding', ns, 'stationary and', len(xy)-ns, '(from %i stars) pixels' % np.sum(refobjs.in_bounds * (refobjs.ref_epoch > 0)))
        # Add a ~1" exclusion zone around reference stars
        # (assuming pixel_scale ~ 0.25")
        r_excl = 4
        avoid_xyr.extend([(x,y,r_excl) for x,y in xy])

        # (We tried a larger exclusion radius on SGA sources, for
        # pre-burning SGA catalog; results were so-so)
        r_sga_excl = r_excl
        J = np.flatnonzero(refobjs.islargegalaxy * refobjs.in_bounds)
        avoid_xyr.extend([(x,y,r_sga_excl) for x,y in zip(refobjs.ibx[J], refobjs.iby[J])])
    avoid_xyr = np.array(avoid_xyr, dtype=np.int32)
    if len(avoid_xyr) > 0:
        avoid_x = avoid_xyr[:,0]
        avoid_y = avoid_xyr[:,1]
        avoid_r = avoid_xyr[:,2]
    else:
        avoid_x = avoid_y = avoid_r = np.array([], dtype=np.int32)
    del avoid_xyr

    if refobjs:
        # Reference objects that suppress new source detection.
        I_no_sourcedet = np.flatnonzero(
            reduce(np.logical_or, [
                # CLUSTER catalog
                refobjs.iscluster,
                # Magellanic clouds
                refobjs.ismcloud,
                # SGA frozen sources
                refobjs.freezeparams,
                ]))
        if len(I_no_sourcedet):
            info('Suppressing source detection in reference catalog masks:', Counter([str(r) for r in refobjs.ref_cat[I_no_sourcedet]]).most_common())
            refmap = get_reference_map(targetwcs, refobjs[I_no_sourcedet])
            from legacypipe.bits import REF_MAP_BITS
            for name,bitval in REF_MAP_BITS.items():
                n = np.sum((refmap & bitval) != 0)
                if n:
                    print('Bit', name, 'set for', n, 'pixels')
            avoid_map = (refmap != 0)
            #avoid_map = (get_reference_map(targetwcs, refs_no_sourcedet) != 0)

    record_event and record_event('stage_srcs: detection maps')
    tnow = Time()
    debug('Rendering detection maps...')
    detmaps, detivs, satmaps = detection_maps(tims, targetwcs, bands, mp,
                                              apodize=10, nsatur=nsatur)
    tnow = Time()
    debug('Detmaps:', tnow-tlast)
    tlast = tnow
    record_event and record_event('stage_srcs: sources')

    if plots:
        import pylab as plt
        for band,detmap,satmap in zip(bands, detmaps, satmaps):
            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(detmap, origin='lower', interpolation='nearest')
            plt.subplot(1,2,2)
            plt.imshow(satmap, origin='lower', interpolation='nearest', vmin=0, vmax=1, cmap='hot')
            plt.suptitle('%s detmap/satmap' % band)
            ps.savefig()
            
    # Expand the mask around saturated pixels to avoid generating
    # peaks at the edge of the mask.
    saturated_pix = [binary_dilation(satmap > 0, iterations=4) for satmap in satmaps]

    # Formerly, we generated sources for each saturated blob, but since we now initialize
    # with Tycho-2 and Gaia stars and large galaxies, not needed.

    if plots:
        from legacypipe.runbrick_plots import detection_plots
        detection_plots(detmaps, detivs, bands, saturated_pix, tims,
                        targetwcs, refobjs, large_galaxies, gaia_stars, ps)

    # SED-matched detections
    record_event and record_event('stage_srcs: SED-matched')
    debug('Running source detection at', nsigma, 'sigma')
    SEDs = survey.sed_matched_filters(bands)

    kwa = {}
    if plots:
        coims,_ = quick_coadds(tims, bands, targetwcs)
        rgb,_ = survey.get_rgb(coims, bands)
        kwa.update(rgbimg=rgb)

    Tnew,newcat,hot = run_sed_matched_filters(
        SEDs, bands, detmaps, detivs, (avoid_x,avoid_y,avoid_r), targetwcs,
        nsigma=nsigma, saddle_fraction=saddle_fraction, saddle_min=saddle_min,
        saturated_pix=saturated_pix, veto_map=avoid_map, blob_dilate=blob_dilate,
        plots=plots, ps=ps, mp=mp, **kwa)

    if Tnew is not None:
        assert(len(Tnew) == len(newcat))
        Tnew.delete_column('apsn')
        Tnew.ref_cat = np.array(['  '] * len(Tnew))
        Tnew.ref_id  = np.zeros(len(Tnew), np.int64)
    del detmaps
    del detivs

    # Merge newly detected sources with reference sources (Tycho2, Gaia, large galaxies)
    cats = []
    tables = []
    if Tnew is not None:
        for src,ix,iy in zip(newcat, Tnew.ibx, Tnew.iby):
            for satmap in saturated_pix:
                if satmap[iy, ix]:
                    src.needs_initial_flux = True
        cats.extend(newcat)
        tables.append(Tnew)

    # Hold aside "special" entries in the ref catalog: DUP, clusters, Magellanic clouds, ...
    if refobjs and len(refobjs):
        Ispecial = np.flatnonzero(refobjs.ignore_source)
        if len(Ispecial):
            Iregular = np.flatnonzero(~refobjs.ignore_source)
            T_special = refobjs[Ispecial]
            # note: leave the original "refobjs" alone, it gets used later, eg in
            # get_reference_map for oneblob.  (and may as well keep "refcat" matching)
            T_ref = refobjs[Iregular]
            cat_special = [refcat[i] for i in Ispecial]
            cat_ref = [refcat[i] for i in Iregular]
            del Iregular
            assert(len(refcat) == len(refobjs))
            assert(len(cat_ref) == len(T_ref))
            assert(len(cat_special) == len(T_special))
        else:
            T_special = None
            cat_special = None
            T_ref = refobjs
            cat_ref = refcat
        del Ispecial
        if len(cat_ref):
            cats.extend(cat_ref)
            tables.append(T_ref)

    T = merge_tables(tables, columns='fillzero')
    cat = Catalog(*cats)
    cat.freezeAllParams()
    # The tractor Source object list "cat" and the table "T" are row-aligned.
    assert(len(T) > 0)
    assert(len(cat) == len(T))

    tnow = Time()
    debug('Peaks:', tnow-tlast)
    tlast = tnow

    if plots:
        from legacypipe.runbrick_plots import detection_plots_2
        detection_plots_2(tims, bands, targetwcs, refobjs, Tnew, hot,
                          saturated_pix, ps)

    # Find "hot" pixels that are separated by masked pixels,
    # to connect blobs across, eg, bleed trails and saturated cores.
    hot = merge_hot_satur(hot, saturated_pix)

    # Segment, and record which sources fall into each blob
    blobmap,blobsrcs,blobslices = segment_and_group_sources(hot, T, name=brickname,
                                                          ps=ps, plots=plots)
    del hot

    tnow = Time()
    debug('Blobs:', tnow-tlast)
    tlast = tnow

    # DEBUG
    if False:
        BT = fits_table()
        BT.blob_pix = []
        BT.blob_srcs = []
        for blobid, (srcs, slc) in enumerate(zip(blobsrcs, blobslices)):
            BT.blob_pix.append(np.sum(blobmap[slc] == blobid))
            BT.blob_srcs.append(len(srcs))
        BT.to_np_arrays()
        BT.writeto('blob-stats-dilate%i.fits' % blob_dilate)
        sys.exit(0)

    ccds.co_sky = np.zeros(len(ccds), np.float32)
    if ubercal_sky:
        sky_overlap = False
    else:
        sky_overlap = True
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
            pix = co[(cowt > 0) * (blobmap == -1)]
            if len(pix) == 0:
                debug('Cosky band', band, ': no unmasked pixels outside blobs')
                continue
            cosky = np.median(pix)
            info('Median coadd sky for', band, ':', cosky)
            co_sky[band] = cosky
            for itim,tim in enumerate(tims):
                if tim.band != band:
                    continue
                goodpix = (tim.inverr > 0)
                tim.data[goodpix] -= cosky
                tim.setImage(tim.data) #Update GPU flag
                ccds.co_sky[itim] = cosky
    else:
        co_sky = None

    info('Sources detected:', len(T), 'in', len(blobslices), 'blobs')

    detstars = T.copy()
    detstars.blob = blobmap[np.clip(T.iby, 0, H-1), np.clip(T.ibx, 0, W-1)]
    # Append -1,-1,-1,-1 to fill in a value for (reference) sources with no blob.
    bb = np.array([[b[0].start, b[0].stop, b[1].start, b[1].stop] for b in blobslices] +
                  [[-1,-1,-1,-1]])
    detstars.blob_x0 = bb[detstars.blob, 2]
    detstars.blob_x1 = bb[detstars.blob, 3]
    detstars.blob_y0 = bb[detstars.blob, 0]
    detstars.blob_y1 = bb[detstars.blob, 1]
    for band in bands:
        bandflux = np.zeros(len(cat), 'f4')
        for isrc, src in enumerate(cat):
            if src:
                bandflux[isrc] = src.getBrightness().getFlux(band)
        detstars.set('flux_' + band, bandflux)
    with survey.write_output('detected-sources', brick=brickname) as out:
        detstars.writeto(None, fits_object=out.fits, primheader=version_header,
                         extname='DETECTIONS')
    del detstars
    if 'peaksn' in T.get_columns():
        # (can be missing if no sources are detected - only reference sources)
        T.delete_column('peaksn')

    keys = ['T', 'cat', 'T_special', 'cat_special', 'tims', 'blobsrcs', 'blobslices', 'blobmap',
            'ps', 'saturated_pix', 'version_header', 'co_sky', 'ccds']
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def stage_fitblobs(T=None,
                   brickname=None,
                   brickid=None,
                   brick=None,
                   version_header=None,
                   blobsrcs=None, blobslices=None, blobmap=None,
                   cat=None,
                   T_special=None,
                   cat_special=None,
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
                   iterative_nsigma=None,
                   nsigma=None,
                   large_galaxies_force_pointsource=True,
                   less_masking=False,
                   sub_blobs=False,
                   use_ceres=True, mp=None,
                   checkpoint_filename=None,
                   checkpoint_period=600,
                   write_pickle_filename=None,
                   write_metrics=True,
                   get_all_models=False,
                   refobjs=None,
                   bailout=False,
                   record_event=None,
                   custom_brick=False,
                   use_gpu=False,
                   gpumode=0,
                   ngpu=1,
                   gpu_ids=[0],
                   threads_per_gpu=16,
                   bid=None,
                   **kwargs):
    '''
    This is where the actual source fitting happens.
    The `one_blob` function is called for each "blob" of pixels with
    the sources contained within that blob.
    '''
    import time
    from tractor import Catalog
    from legacypipe.oneblob import MODEL_NAMES
    #return None

    record_event and record_event('stage_fitblobs: starting')
    _add_stage_version(version_header, 'FITB', 'fitblobs')
    tlast = Time()

    version_header.add_record(dict(name='GALFRPSF',
                                   value=large_galaxies_force_pointsource,
                                   help='Large galaxies force PSF?'))
    version_header.add_record(dict(name='LESSMASK',
                                   value=less_masking,
                                   help='Reduce masking behaviors?'))

    version_header.add_record(dict(name='COMMENT', value='DCHISQ array model names'))
    for i,mod in enumerate(MODEL_NAMES):
        version_header.add_record(dict(name='DCHISQ_%i' % i, value=mod.upper()))

    if plots:
        from legacypipe.runbrick_plots import fitblobs_plots
        fitblobs_plots(tims, bands, targetwcs, blobslices, blobsrcs, cat,
                       blobmap, ps)

    tnow = Time()
    debug('Fitblobs:', tnow-tlast)
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
                warnings.warn('Clipping blob x,y to brick bounds %i,%i' % (x,y))
                x = np.clip(x, 0, W-1)
                y = np.clip(y, 0, H-1)
            blob = blobmap[y,x]
            if blob >= 0:
                keepblobs.append(blob)
            else:
                warnings.warn('Blobxy %i,%i is not in a blob!' % (x,y))
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
        # 'blobmap' is an image with values -1 for no blob, or the index
        # of the blob.  Create a map from old 'blob number+1' to new
        # 'blob number', keeping only blobs in the 'keepblobs' list.
        # The +1 is so that -1 is a valid index in the mapping.
        NB = len(blobslices)
        remap = np.empty(NB+1, np.int32)
        remap[:] = -1
        remap[keepblobs + 1] = np.arange(len(keepblobs))
        # apply the map!
        blobmap = remap[blobmap + 1]
        # 'blobslices' and 'blobsrcs' are lists where the index
        # corresponds to the value in the 'blobs' map.
        blobslices = [blobslices[i] for i in keepblobs]
        blobsrcs   = [blobsrcs  [i] for i in keepblobs]

    # drop any cached data before we start pickling/multiprocessing
    survey.drop_cache()

    if plots and refobjs:
        from legacypipe.runbrick_plots import fitblobs_plots_2
        fitblobs_plots_2(blobmap, refobjs, ps)

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
        bailout_mask = _get_bailout_mask(blobmap, skipblobs, targetwcs, W, H, brick,
                                         blobslices)
        # skip all blobs!
        new_skipblobs = np.unique(blobmap[blobmap>=0])
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
                # For SGA sources
                for src in cat_refbail:
                    _convert_ellipses(src)
                # Sets TYPE, etc for T_refbail table.
                _get_tractor_fits_values(T_refbail, cat_refbail, '%s')
                for c,t in [('iterative', bool),
                            ('force_keep_source', bool),
                            ('forced_pointsource', bool),
                            ('fit_background', bool),
                            ('hit_r_limit', bool),
                            ('hit_ser_limit', bool),]:
                    T_refbail.set(c, np.zeros(len(T_refbail), dtype=t))
                T_refbail.dchisq     = np.zeros((len(T_refbail), 5),          np.float32)
                T_refbail.rchisq     = np.zeros((len(T_refbail), len(bands)), np.float32)
                T_refbail.fracflux   = np.zeros((len(T_refbail), len(bands)), np.float32)
                T_refbail.fracmasked = np.zeros((len(T_refbail), len(bands)), np.float32)
                T_refbail.fracin     = np.ones ((len(T_refbail), len(bands)), np.float32)
            if T_refbail is not None:
                info('Found', len(T_refbail), 'reference sources in bail-out blobs')
        skipblobs = new_skipblobs
        # append empty results so that a later assert on the lengths will pass
        while len(R) < len(blobsrcs):
            R.append(dict(brickname=brickname, iblob=-1, result=None))

    frozen_galaxies = get_frozen_galaxies(T, blobsrcs, blobmap, targetwcs, cat)
    refmap = get_blobiter_ref_map(refobjs, less_masking, targetwcs)
    # We pass this list in to _blob_iter; it appends any blob numbers
    # that were processed as sub-blobs.
    ran_sub_blobs = None
    if sub_blobs:
        ran_sub_blobs = []

    if iterative and (iterative_nsigma is None):
        assert(nsigma is not None)
        iterative_nsigma = nsigma

    job_id_map = {}
    gpu_tuple = (use_gpu, gpumode, ngpu, threads_per_gpu)

    # Create the iterator over blobs to process
    blobiter = _blob_iter(job_id_map,
                          brickname, blobslices, blobsrcs, blobmap, targetwcs, tims,
                          cat, T, bands, plots, ps, reoptimize, iterative, iterative_nsigma,
                          use_ceres,
                          refmap, large_galaxies_force_pointsource, less_masking, brick,
                          frozen_galaxies,
                          skipblobs=skipblobs,
                          single_thread=(mp is None or mp.pool is None),
                          max_blobsize=max_blobsize, custom_brick=custom_brick,
                          enable_sub_blobs=sub_blobs,
                          ran_sub_blobs=ran_sub_blobs,
                          gpu_tuple=gpu_tuple,bid=bid)

    if checkpoint_filename is None:
        # FIXME -- add worker-died checks & logging here
        print ("No checkpoint")
        R.extend(mp.map(_bounce_one_blob, blobiter))
    else:
        from astrometry.util.ttime import CpuMeas
        # Begin running one_blob on each blob...
        Riter = mp.imap_unordered(_bounce_one_blob, blobiter)
        # measure wall time and write out checkpoint file periodically.
        last_checkpoint = CpuMeas()
        n_finished = 0
        n_finished_total = 0
        procs_last = None
        last_printout = CpuMeas()

        while True:
            import time
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
                    print('Failed to write checkpoint file', checkpoint_filename)
                    import traceback
                    traceback.print_exc()

            dt = tnow.wall_seconds_since(last_printout)
            if dt > 60:
                last_printout = tnow
                if hasattr(Riter, 'get_running_jobs'):
                    print('Running:')
                    status = Riter.get_running_jobs()
                    #print('running job status:', status)
                    # other threads may try to update status during iteration
                    status = status.copy()
                    jmap = job_id_map.copy()
                    from legacypipe.utils import run_ps
                    pid = os.getpid()
                    if procs_last is None:
                        procs_last = run_ps(pid)
                        time.sleep(1.)
                    procs = run_ps(pid, last=procs_last)
                    #print('procs columns:', procs.get_columns())
                    tnow = time.time()
                    keys = list(status.keys())
                    keys.sort()
                    for jobid in keys:
                        s = status[jobid]
                        if not jobid in jmap:
                            print('trackingpool job id', jobid, 'not known')
                            continue
                        (brick,blob) = jmap[jobid]
                        pid = s['pid']
                        i = np.flatnonzero(procs.pid == pid)
                        if len(i) != 1:
                            print('Blob %10s' % blob, 'pid', pid, '"running" for %7.1f sec, but did not find PID %i: crashed?' %
                                  (tnow - s['time'], pid))
                        else:
                            i = i[0]
                            print('Blob %5s' % blob, 'pid %7i' % s['pid'],
                                  'total CPU %7.1f sec' % (tnow - s['time']),
                                  'CPU now %5.1f %%,' % procs.proc_icpu[i],
                                  'VMsize %5.1f GB,' % (procs.vsz[i] / (1024 * 1024)),
                                  'VMpeak %5.1f GB' % (procs.proc_vmpeak[i] / (1024 * 1024)))

            # Wait for results (with timeout)
            from legacypipe.trackingpool import PoolWorkerDiedException
            try:
                if mp.pool is not None:
                    timeout = max(1, checkpoint_period - dt)
                    timeout = min(10, timeout)
                    debug('Main thread: waiting for result...')
                    r = Riter.next(timeout)

                    rstr = ''
                    if r is None:
                        rstr = 'None'
                    else:
                        rr = r['result']
                        rrstr = '%i srcs'%len(rr) if rr is not None else 'none'
                        rstr = 'brick %s blob %s: %s' % (r['brickname'], r['iblob'], rrstr)
                    debug('Main thread: got result: [%s]' % rstr)

                else:
                    r = next(Riter)
                R.append(r)
                n_finished += 1
                n_finished_total += 1
            except StopIteration:
                print('Main thread: reached end of results')
                break
            except multiprocessing.TimeoutError:
                debug('Main thread: timed out waiting for result')
                continue
            except PoolWorkerDiedException as e:
                print('Main thread: worker died')
                print('A worker died (maybe OOM-killed?) while processing a blob.')
                print(e)
                workitem = 'unknown'
                index = e.index
                jmap = job_id_map.copy()
                if not index in jmap:
                    print('Unknown which blob that worker was processing.  index %i\n' % index)
                    workitem = 'unknown (index %i)' % index
                else:
                    (brick,blob) = jmap[index]
                    print('Worker was processing index %i: brick %s blob %s' % (index, brick, blob))
                    workitem = 'brick %s blob %s' % (brick, blob)

                print('Writing checkpoints before exiting...')
                _write_checkpoint(R, checkpoint_filename)
                print('Wrote checkpoints')
                print('terminating pool...')
                mp.pool.terminate()
                print('raising exception...')
                #raise e
                raise RunbrickError('Worker process died (OOM?) while processing: %s' % workitem)
            except Exception as e:
                print('Main thread: Exception fetching result:', e)
                import traceback
                traceback.print_exc()
                raise e

        # Write checkpoint when done!
        _write_checkpoint(R, checkpoint_filename)
        debug('Got', n_finished_total, 'results; wrote', len(R), 'to checkpoint')
    debug('Fitting sources:', Time()-tlast)

    # Repackage the results from one_blob...

    # check for any blobs that were processed as sub-blobs; mark them in the sub_blob_mask.
    sub_blob_mask = None
    if ran_sub_blobs is not None and len(ran_sub_blobs):
        # Create a 1-d array that will map from blob number (ie in "blobmap")
        # to the boolean mask value
        maxblob = blobmap.max()
        sbmap = np.zeros(maxblob+2, bool)
        sbmap[np.array(ran_sub_blobs)+1] = True
        sub_blob_mask = sbmap[blobmap+1]
        del sbmap

    # one_blob can change the number and types of sources.
    # Reorder the sources:
    # sub-blobs breaks this: MxN results R for one blob
    #assert(len(R) == len(blobsrcs))
    # Filter out total failures/skips (None values)
    R = [r for r in R if r is not None]
    # Extract the 'result' field from the dictionary
    # AND filter out results that are None or have 0 sources
    R = [r['result'] for r in R if r.get('result') is not None and len(r['result']) > 0]

    if len(R) == 0:
        if bailout:
            info('No sources, but continuing because of --bail-out')
            return dict(fitblobs=None)
        return None

    # BB = merge_tables(R) should now succeed
    BB = merge_tables(R)
    del R
    if len(BB):
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
        del BB.Isrcs, II
        assert(len(T) == len(BB))

        # Drop sources that exited the blob as a result of fitting.
        left_blob = np.logical_and(BB.started_in_blob,
                                   np.logical_not(BB.finished_in_blob))
        I, = np.nonzero(np.logical_not(left_blob))
        if len(I) < len(BB):
            debug('Dropping', len(BB)-len(I), 'sources that exited their blobs during fitting')
            BB.cut(I)
            T.cut(I)
            newcat = [newcat[i] for i in I]

    else:
        T.cut([])
        newcat = []
    assert(len(T) == len(BB))

    assert(len(T) == len(newcat))
    info('Old catalog:', len(cat))
    info('New catalog:', len(newcat))
    if len(newcat) > 0:
        ns,nb = BB.fracflux.shape
        assert(ns == len(newcat))
        assert(nb == len(bands))
        ns,nb = BB.fracmasked.shape
        assert(ns == len(newcat))
        assert(nb == len(bands))
        ns,nb = BB.fracin.shape
        assert(ns == len(newcat))
        assert(nb == len(bands))
        ns,nb = BB.rchisq.shape
        assert(ns == len(newcat))
        assert(nb == len(bands))
        ns,nb = BB.dchisq.shape
        assert(ns == len(newcat))
        assert(nb == 5) # psf, rex, dev, exp, ser

    # We want to order sources (and assign objids) so that sources outside the brick
    # are at the end

    # Copy source positions back to the object table
    T.ra  = np.array([src.getPosition().ra  for src in newcat], dtype=np.float64)
    T.dec = np.array([src.getPosition().dec for src in newcat], dtype=np.float64)

    if len(T):
        # Copy blob results to table T
        for k in ['fracflux', 'fracin', 'fracmasked', 'rchisq',
                  'cpu_arch', 'cpu_source', 'cpu_blob',
                  'blob_width', 'blob_height', 'blob_npix',
                  'blob_nimages', 'blob_totalpix',
                  'blob_symm_width', 'blob_symm_height', 'blob_symm_npix',
                  'blob_symm_nimages', 'bx0', 'by0',
                  'hit_limit', 'hit_ser_limit', 'hit_r_limit',
                  'dchisq',
                  'force_keep_source', 'fit_background', 'forced_pointsource']:
            T.set(k, BB.get(k))

    # Merge "special" sources and T_refbail back in.
    T.regular = np.ones(len(T), bool)
    T_all = [T]
    n_newcat = len(newcat)
    cat = newcat
    # T_special and T_refbail both get "regular" = False
    if T_special is not None:
        T_all.append(T_special)
        cat.extend(cat_special)
        print('cat_special:', cat_special)
    if T_refbail:
        T_all.append(T_refbail)
        cat.extend(cat_refbail)
        del cat_refbail

    if len(T_all) > 1:
        T = merge_tables(T_all, columns='fillzero')
    del T_all
    del T_refbail
    del T_special
    del cat_special
    del newcat

    assert(len(cat) == len(T))

    cat = Catalog(*cat)
    # freeze special catalog entries (so that number of catalog parameters is corrrect)
    for i in range(n_newcat, len(cat)):
        cat.freezeParam(i)

    if len(BB) == 0:
        invvars = None
    else:
        invvars = np.hstack(BB.srcinvvars)
        assert(cat.numberOfParams() == len(invvars))
    # NOTE that "BB" can now be shorter than cat and T.
    assert(np.sum(T.regular) == len(BB))
    # We assume below (when unpacking BB for all-models) that the
    # "regular" entries are at the beginning of T.
    
    _,bx,by = targetwcs.radec2pixelxy(T.ra, T.dec)
    T.bx = (bx - 1.).astype(np.float32)
    T.by = (by - 1.).astype(np.float32)
    T.ibx = np.round(T.bx).astype(np.int32)
    T.iby = np.round(T.by).astype(np.int32)
    T.in_bounds = ((T.ibx >= 0) * (T.iby >= 0) * (T.ibx < W) * (T.iby < H))
    # For --bail-out:
    if not 'bx0' in T.get_columns():
        T.bx0 = T.bx.copy()
        T.by0 = T.by.copy()

    # Order sources by RA.
    # (Here we're just setting 'objid', not actually reordering arrays.)
    # (put all the regular * in_bounds sources, then dup in-bound, then oob)
    I = np.argsort(T.ra + (-2000 * T.in_bounds) + (-1000 * T.regular))
    T.objid = np.empty(len(T), np.int32)
    T.objid[I] = np.arange(len(T))

    # Set blob numbers
    T.blob = np.empty(len(T), np.int32)
    T.blob[:] = -1
    T.blob[T.in_bounds] = blobmap[T.iby[T.in_bounds], T.ibx[T.in_bounds]]
    # Renumber blobs to make them contiguous.
    goodblobs = (T.blob > -1)
    oldblobs = T.blob[goodblobs]
    _,iblob = np.unique(oldblobs, return_inverse=True)
    T.blob[goodblobs] = iblob
    del goodblobs
    # Renumber blobmap to match T.blob
    remap = np.empty(blobmap.max() + 2, np.int32)
    # dropped blobs -> -1
    remap[:] = -1
    # (this +1 business is because we're using a numpy array for the map)
    remap[oldblobs + 1] = iblob
    blobmap = remap[blobmap+1]
    del iblob, oldblobs
    # Frozen galaxies: update blob numbers.
    # while remapping, flip from blob->[srcs] to src->[blobs].
    fro_gals = {}
    for b,gals in frozen_galaxies.items():
        for gal in gals:
            if not gal in fro_gals:
                fro_gals[gal] = []
            bnew = remap[b+1]
            if bnew != -1:
                fro_gals[gal].append(bnew)
    frozen_galaxies = fro_gals
    del remap

    # How many sources in each blob?
    ninblob = Counter(T.blob)
    ninblob[-1] = 0
    T.ninblob = np.array([ninblob[b] for b in T.blob]).astype(np.int32)
    del ninblob

    # write out blob map
    if write_metrics:
        from legacypipe.utils import copy_header_with_wcs
        hdr = copy_header_with_wcs(version_header, targetwcs)
        hdr.add_record(dict(name='IMTYPE', value='blobmap',
                            comment='LegacySurveys image type'))
        with survey.write_output('blobmap', brick=brickname, shape=blobmap.shape) as out:
            out.fits.write(blobmap, header=hdr, extname='BLOB-MAP')

    T.brickid = np.zeros(len(T), np.int32) + brickid
    T.brickname = np.array([brickname] * len(T))

    if (write_metrics or get_all_models) and len(BB):
        from legacypipe.format_catalog import format_all_models
        TT,hdr = format_all_models(T, cat, BB, bands, survey.allbands,
                                   force_keep=T.force_keep_source)
        if get_all_models:
            all_models = TT
        if write_metrics:
            primhdr = fitsio.FITSHDR()
            for r in version_header.records():
                primhdr.add_record(r)
                primhdr.add_record(dict(name='PRODTYPE', value='catalog',
                                        comment='NOIRLab data product type'))
            with survey.write_output('all-models', brick=brickname) as out:
                TT[np.argsort(TT.objid)].writeto(None, fits_object=out.fits, header=hdr,
                                                 primheader=primhdr, extname='ALL-MODELS')

    keys = ['cat', 'invvars', 'T', 'blobmap', 'refmap', 'version_header',
            'frozen_galaxies']
    if get_all_models:
        keys.append('all_models')
    if bailout:
        keys.append('bailout_mask')
    if sub_blob_mask is not None:
        keys.append('sub_blob_mask')
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

# Also called by farm.py
def get_blobiter_ref_map(refobjs, less_masking, targetwcs):
    refmap = None
    if refobjs:
        from legacypipe.reference import get_reference_map
        # ignore DUP entries
        I = np.flatnonzero(~refobjs.dup)
        if len(I):
            refmap = get_reference_map(targetwcs, refobjs)
    if refmap is None:
        HH, WW = targetwcs.shape
        refmap = np.zeros((int(HH), int(WW)), np.uint8)
    return refmap

# Also called by farm.py
def get_frozen_galaxies(T, blobsrcs, blobmap, targetwcs, cat):
    # Find reference (frozen) large galaxies that touch blobs that
    # they are not part of, to get their profiles subtracted.
    # Generate a blob -> [sources] mapping.
    frozen_galaxies = {}
    cols = T.get_columns()
    if not ('islargegalaxy' in cols and 'freezeparams' in cols):
        return frozen_galaxies
    Igals = np.flatnonzero(T.islargegalaxy * T.freezeparams)
    if len(Igals) == 0:
        return frozen_galaxies
    from legacypipe.reference import get_reference_map
    debug('Found', len(Igals), 'frozen large galaxies')
    # create map in pixel space for each one.
    for ii in Igals:
        # length-1 table
        refgal = T[np.array([ii])].copy()
        refgal.radius_pix *= 2
        galmap = get_reference_map(targetwcs, refgal)
        galblobs = set(blobmap[galmap > 0])
        debug('galaxy mask overlaps blobs:', galblobs)
        galblobs.discard(-1)
        debug('source:', cat[ii])
        if refgal.in_bounds:
            # If in-bounds, remove the blob that this source is
            # already part of, if it exists; it will get processed
            # within that blob.
            for ib,bsrcs in enumerate(blobsrcs):
                if ii in bsrcs:
                    if ib in galblobs:
                        debug('in bounds; removing frozen-galaxy entry for blob', ib, 'bsrcs', bsrcs)
                        galblobs.remove(ib)
        else:
            # Otherwise, remove this from any 'blobsrcs' members it is
            # part of -- this can happen when we clip a source
            # position outside the brick to the brick bounds and that
            # happens to touch a blob.
            for j,bsrcs in enumerate(blobsrcs):
                if ii in bsrcs:
                    blobsrcs[j] = bsrcs[bsrcs != ii]
                    debug('removed source', ii, 'from blob', j, 'blobsrcs', bsrcs, '->', blobsrcs[j])

        for blob in galblobs:
            if not blob in frozen_galaxies:
                frozen_galaxies[blob] = []
            frozen_galaxies[blob].append(cat[ii])
    return frozen_galaxies

def _get_bailout_mask(blobmap, skipblobs, targetwcs, W, H, brick, blobslices):
    # Create a 1-d array that will map from blob number (ie in "blobmap") to the boolean mask value
    maxblob = blobmap.max()
    # mark all as bailed out...
    bmap = np.ones(maxblob+2, bool)
    # except no-blob
    bmap[0] = False

    # Only normal blobs (not sub-blobs) are allowed
    keep_skipblobs = []
    # and blobs from the checkpoint file
    for i in skipblobs:
        try:
            i = int(i)
        except:
            # eg, sub-blobs with blobs like (1,1)
            continue
        bmap[i+1] = False
        keep_skipblobs.append(i)
    skipblobs = keep_skipblobs
    # and blobs that are completely outside the primary region of this brick.
    U = find_unique_pixels(targetwcs, W, H, None,
                           brick.ra1, brick.ra2, brick.dec1, brick.dec2)
    for iblob in np.unique(blobmap):
        if iblob == -1:
            continue
        if iblob in skipblobs:
            continue
        bslc  = blobslices[iblob]
        blobmask = (blobmap[bslc] == iblob)
        if np.all(U[bslc][blobmask] == False):
            debug('Blob', iblob, 'is completely outside the PRIMARY region')
            bmap[iblob+1] = False
    bailout_mask = bmap[blobmap+1]
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
        if ri is None:
            continue
        brick = ri['brickname']
        iblob = ri['iblob']
        r = ri['result']

        if brick != brickname:
            print('Checkpoint brick mismatch:', brick, brickname)
            continue

        if r is None:
            pass
        else:
            # sub-blobs break this!
            sub_blob = (type(iblob) is tuple)
            if sub_blob:
                iblob = r.iblob
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
                if sub_blob:
                    # check that it's a subset?
                    if rx0 < bx0 or ry0 < by0 or rx1 > bx1 or ry1 > by1:
                        print('Checkpointed sub-blob bbox', [rx0,rx1,ry0,ry1],
                              'is not inside expected', [bx0,bx1,by0,by1], 'for iblob', iblob)
                        continue
                else:
                    if rx0 != bx0 or ry0 != by0 or rx1 != bx1 or ry1 != by1:
                        print('Checkpointed blob bbox', [rx0,rx1,ry0,ry1],
                              'does not match expected', [bx0,bx1,by0,by1], 'for iblob', iblob)
                        continue
        keepR.append(ri)
    return keepR

def _blob_iter(job_id_map,
               brickname, blobslices, blobsrcs, blobmap, targetwcs, tims, cat, T, bands,
               plots, ps, reoptimize, iterative, iterative_nsigma, use_ceres, refmap,
               large_galaxies_force_pointsource, less_masking,
               brick, frozen_galaxies, single_thread=False,
               skipblobs=None, max_blobsize=None, custom_brick=False,
               enable_sub_blobs=False,
               ran_sub_blobs=None,
               gpu_tuple=None,
               bid=None):
    '''
    *blobmap*: integer image map, with -1 indicating no-blob, other values indexing
        into *blobslices*,*blobsrcs*.
    *blobsrcs*: a list of numpy arrays of integers -- indices into *cat* -- of the sources in
        this blob.
    *T*: a fits table parallel to *cat* with some extra info (very little used)
    '''
    from legacypipe.bits import REF_MAP_BITS
    from collections import Counter

    def get_subtim_args(tims, targetwcs, bx0,bx1, by0,by1, single_thread):
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
            sx0 = int(np.clip(int(np.floor(sx0 - 1)), 0, w-1))
            sx1 = int(np.clip(int(np.ceil (sx1 - 1)), 0, w-1)) + 1
            sy0 = int(np.clip(int(np.floor(sy0 - 1)), 0, h-1))
            sy1 = int(np.clip(int(np.ceil (sy1 - 1)), 0, h-1)) + 1
            subslc = slice(sy0,sy1),slice(sx0,sx1)
            subimg = tim.getImage   ()[subslc]
            subie  = tim.getInvError()[subslc]
            if tim.dq is None:
                subdq = None
            else:
                subdq  = tim.dq[subslc]
            subwcs = tim.getWcs().shifted(sx0, sy0)
            subsky = tim.getSky().shifted(sx0, sy0)
            subpsf = tim.getPsf().getShifted(sx0, sy0)
            subwcsobj = tim.subwcs.get_subimage(sx0, sy0, sx1-sx0, sy1-sy0)
            tim.imobj.psfnorm = tim.psfnorm
            tim.imobj.galnorm = tim.galnorm
            # FIXME -- maybe the cache is worth sending?
            if hasattr(tim.psf, 'clear_cache'):
                tim.psf.clear_cache()
            # Yuck!  If we're not running with --threads AND oneblob.py modifies the data,
            # bad things happen!
            if single_thread:
                subimg = subimg.copy()
                subie = subie.copy()
                subdq = subdq.copy()
            subtimargs.append((subimg, subie, subdq, subwcs, subwcsobj,
                               tim.getPhotoCal(),
                               subsky, subpsf, tim.name, tim.band, tim.sig1, tim.imobj))
        return subtimargs

    #New logic 1/17/25 - first figure out which blobs should be sent to GPU then yield in second loop
    # 1. Config & Initial Setup
    if gpu_tuple is not None:
        (use_gpu, gpumode, ngpu, threads_per_gpu) = gpu_tuple
    else:
        ngpu, use_gpu, gpumode, threads_per_gpu = 0, False, 0, 0

    num_gpu_slots = ngpu * threads_per_gpu
    skipblobset = set(skipblobs or [])

    blobvals = Counter(blobmap[blobmap >= 0])
    blob_order = np.array([b for b, npix in blobvals.most_common()])
    del blobvals

    H,W = targetwcs.shape
    if custom_brick:
        U = None
    else:
        H, W = targetwcs.shape
        U = find_unique_pixels(targetwcs, W, H, None,
                               brick.ra1, brick.ra2, brick.dec1, brick.dec2)

    all_tasks_metadata = []
    #print ("Blob order", blob_order)

    # 2. DISCOVERY PHASE (Original logging style preserved here)
    for nblob, iblob in enumerate(blob_order):
        bslc = blobslices[iblob]
        Isrcs = blobsrcs[iblob]
        sy, sx = bslc
        by0, by1, bx0, bx1 = sy.start, sy.stop, sx.start, sx.stop
        blobh, blobw = by1 - by0, bx1 - bx0
        blobmask = (blobmap[bslc] == iblob)

        # Skip checks
        if int(iblob) in skipblobset:
            all_tasks_metadata.append({'size': -1, 'iblob': iblob, 'nblob_idx': nblob+1})
            continue

        if U is not None and np.all(U[bslc][blobmask] == False):
            info(f'Blob {nblob+1} is completely outside the unique region of this brick -- skipping')
            all_tasks_metadata.append({'size': -1, 'iblob': iblob, 'nblob_idx': nblob+1})
            continue

        npix = np.sum(blobmask)

        # Find one pixel for debug (Legacy Style)
        onex = oney = None
        ii = np.flatnonzero(blobmask)
        if len(ii) > 0:
            local_y, local_x = np.unravel_index(ii[0], blobmask.shape)
            onex, oney = bx0 + local_x, by0 + local_y

        # LEGACY PRINT 1: Blob Info
        info(('Blob %i of %i, id: %i, sources: %i, size: %ix%i, npix %i, brick X: %i,%i, ' +
               'Y: %i,%i, one pixel: %i %i') %
              (nblob+1, len(blobslices), iblob, len(Isrcs), blobw, blobh, npix,
               bx0,bx1,by0,by1, onex,oney))

        if max_blobsize is not None and npix > max_blobsize:
            info('Number of pixels in blob, %i, exceeds max blobsize %i' % (npix, max_blobsize))
            all_tasks_metadata.append({'size': -1, 'iblob': iblob, 'nblob_idx': nblob+1})
            continue

        # Sub-blob logic
        do_sub_blobs = enable_sub_blobs or np.all((refmap[bslc][blobmask] & REF_MAP_BITS['CLUSTER']) != 0)

        if do_sub_blobs:
            # split into ~500-pixel sub-blobs. 
            # "overlap" is the duplicated / overlapping region between sub-blobs.
            overlap = 50
            # target sub-blob size for selecting number of sub-blobs
            # this yields  710 pixels -> 2 sub-blobs
            #             3600 pixels -> 8 sub-blobs (good for multi-processing!)
            target = 490

            # HACK - IBIS / XMM
            overlap = 32
            target = 256
            # Minimum size that will get split into 2 or more sub-blobs
            threshsize = 1.5 * (target - overlap) + overlap
        do_sub_blobs = do_sub_blobs and (blobw >= threshsize or blobh >= threshsize)

        if not do_sub_blobs:
            # Regular Blob Task
            all_tasks_metadata.append({
                'nsrcs': len(Isrcs),
                'size': npix,
                'iblob': iblob,
                'sub_idx': None,
                'coords': (bx0, bx1, by0, by1),
                'Isrcs': Isrcs,
                'blobmask': blobmask,
                'nblob_idx': nblob + 1,
                'onex': onex,
                'oney': oney,
            })
        else:
            # LEGACY PRINT 2: Split Info
            nsubx = int(max(1, np.round((blobw - overlap) / (target - overlap))))
            nsuby = int(max(1, np.round((blobh - overlap) / (target - overlap))))
            subw = (blobw + (nsubx-1)*overlap + nsubx-1) // nsubx
            subh = (blobh + (nsuby-1)*overlap + nsuby-1) // nsuby
            info('Will split into %i x %i sub-blobs of %i x %i pixels' % (nsubx, nsuby, subw, subh))

            uniqx = [0] + [n*(subw-overlap) + overlap//2 for n in range(1, nsubx)] + [blobw]
            uniqy = [0] + [n*(subh-overlap) + overlap//2 for n in range(1, nsuby)] + [blobh]

            for i in range(nsuby):
                s_y0, s_y1 = i*(subh-overlap), min(i*(subh-overlap)+subh, blobh)
                for j in range(nsubx):
                    sub_idx = i*nsubx + j
                    if (int(iblob), sub_idx) in skipblobset:
                        continue

                    s_x0, s_x1 = j*(subw-overlap), min(j*(subw-overlap)+subw, blobw)
                    s_bx0, s_bx1, s_by0, s_by1 = bx0+s_x0, bx0+s_x1, by0+s_y0, by0+s_y1

                    clipx = np.clip(T.ibx[Isrcs], 0, W-1)
                    clipy = np.clip(T.iby[Isrcs], 0, H-1)
                    Isubsrcs = Isrcs[(clipx >= s_bx0)*(clipx < s_bx1)*(clipy >= s_by0)*(clipy < s_by1)]

                    sub_blob_name = '%i-%i' % (nblob+1, 1+sub_idx)
                    info('%i of %i sources are within sub-blob %s' %
                         (len(Isubsrcs), len(Isrcs), sub_blob_name))

                    if len(Isubsrcs) == 0:
                        continue

                    sub_mask = blobmask[s_y0:s_y1, s_x0:s_x1]
                    all_tasks_metadata.append({
                        'nsrcs': len(Isubsrcs),
                        'size': np.sum(sub_mask),
                        'iblob': iblob,
                        'sub_idx': sub_idx,
                        'nblob_idx': nblob + 1,
                        'sub_name': sub_blob_name,
                        'coords': (s_bx0, s_bx1, s_by0, s_by1),
                        'Isrcs': Isubsrcs,
                        'blobmask': sub_mask,
                        'onex': s_bx0,
                        'oney': s_by0,
                        'unique_bounds': (bx0+uniqx[j], bx0+uniqx[j+1], by0+uniqy[i], by0+uniqy[i+1])
                    })

    # 3. GLOBAL DECISION: Determine the sorting metric
    # Determine sorting metric: 'nsrcs' if any sub-blobs exist, else 'size'
    any_sub = any(t.get('sub_idx') is not None for t in all_tasks_metadata if t['size'] != -1)
    sort_metric = 'nsrcs' if any_sub else 'size'

    # 4. GLOBAL SORT
    # We use a lambda to pull the chosen metric from the metadata dictionaries
    all_tasks_metadata.sort(key=lambda x: x.get(sort_metric, 0), reverse=True)

    # 5. YIELD PHASE (New concisely logged dispatch)
    job_id = 0
    total_tasks = len(all_tasks_metadata)
    for rank, task in enumerate(all_tasks_metadata):
        task_rank = rank + 1
        iblob = task['iblob']

        if task['size'] == -1:
            job_id_map[job_id] = (brickname, task['nblob_idx'])
            yield (brickname, iblob, None, None)
            job_id += 1
            continue

        is_priority = (rank < num_gpu_slots) if use_gpu else False
        bx0, bx1, by0, by1 = task['coords']
        Isrcs = task['Isrcs']
        sub_idx = task.get('sub_idx')
        display_name = task.get('sub_name', f"{task['nblob_idx']}")

        # DISPATCH LOG: Shows final order and priority
        p_str = "[GPU PRIORITY]" if is_priority else "[CPU]"
        val = task[sort_metric]
        if is_priority:
            info(f"Dispatching Rank {task_rank}/{total_tasks}: Blob {display_name} {p_str} ({val} {sort_metric}, size {task['size']} npix)")
        else:
            debug(f"Dispatching Rank {task_rank}/{total_tasks}: Blob {display_name} {p_str} ({val} {sort_metric}, size {task['size']} npix)")

        subtimargs = get_subtim_args(tims, targetwcs, bx0, bx1, by0, by1, single_thread)

        job_id_map[job_id] = (brickname, display_name)
        job_id += 1
        if sub_idx is not None and ran_sub_blobs is not None:
            ran_sub_blobs.append(int(iblob))

        yield (brickname, (iblob, sub_idx) if sub_idx is not None else iblob, task.get('unique_bounds'),
               (task_rank, iblob, Isrcs, targetwcs, bx0, by0, bx1-bx0, by1-by0,
                task['blobmask'], subtimargs, [cat[i] for i in Isrcs], bands, plots, ps,
                reoptimize, iterative, iterative_nsigma, use_ceres, refmap[by0:by1, bx0:bx1],
                large_galaxies_force_pointsource, less_masking, frozen_galaxies.get(iblob, []),
                use_gpu, gpumode, bid, is_priority))

def _bounce_one_blob(X):
    '''This wraps the one_blob function for multiprocessing purposes (and
    now also does some post-processing).
    '''
    from legacypipe.oneblob import one_blob
    global _GLOBAL_LEGACYPIPE_CONTEXT
    pid = os.getpid()

    # Unpack the high-level tuple
    (brickname, iblob, blob_unique, X_inner) = X

    if X_inner is None:
        debug(f"{iblob=}, worker {pid} has no sources.")
        return None

    # --- 1. INITIALIZATION & GPU ASSIGNMENT (Lazy & Task-Driven) ---
    # We only run this logic once per worker process.
    if not _GLOBAL_LEGACYPIPE_CONTEXT.get('initialized', False):
        # Indices relative to the inner tuple:
        # [..., use_gpu(-4), gpumode(-3), bid(-2), is_priority(-1)]
        is_priority = X_inner[-1]

        lock = _GLOBAL_LEGACYPIPE_CONTEXT['shared_lock']
        with lock:
            # Double-check inside the lock
            if not _GLOBAL_LEGACYPIPE_CONTEXT.get('initialized', False):
                counter = _GLOBAL_LEGACYPIPE_CONTEXT['shared_counter']
                available_ids = _GLOBAL_LEGACYPIPE_CONTEXT['available_gpu_ids']
                max_slots = _GLOBAL_LEGACYPIPE_CONTEXT['max_gpu_slots']

                # Only try to grab a GPU if this specific task is a priority blob
                if is_priority and available_ids and counter.value < max_slots:
                    try:
                        import cupy as cp
                        if cp.cuda.is_available():
                            gpu_id = available_ids[counter.value % len(available_ids)]
                            _GLOBAL_LEGACYPIPE_CONTEXT['is_gpu_worker'] = True
                            _GLOBAL_LEGACYPIPE_CONTEXT['gpu_device_id'] = gpu_id
                            # We keep the gpumode requested by main (e.g., 2)
                            _GLOBAL_LEGACYPIPE_CONTEXT['gpumode'] = _GLOBAL_LEGACYPIPE_CONTEXT['target_gpumode']
                            counter.value += 1
                            print(f"Worker PID {pid}: GRABBED GPU {gpu_id} using priority Blob id {iblob=}.")
                    except Exception as e:
                        print(f"Worker PID {pid}: GPU setup failed: {e}. Falling back to CPU.")

                # If we didn't get a GPU (either not priority or init failed), we are a CPU worker
                if not _GLOBAL_LEGACYPIPE_CONTEXT.get('is_gpu_worker', False):
                    _GLOBAL_LEGACYPIPE_CONTEXT['is_gpu_worker'] = False
                    _GLOBAL_LEGACYPIPE_CONTEXT['gpu_device_id'] = None
                    _GLOBAL_LEGACYPIPE_CONTEXT['gpumode'] = 0

                _GLOBAL_LEGACYPIPE_CONTEXT['initialized'] = True

    # --- 2. PREPARE PARAMETERS FOR THIS SPECIFIC BLOB ---
    is_gpu_worker = _GLOBAL_LEGACYPIPE_CONTEXT['is_gpu_worker']
    gpu_device_id = _GLOBAL_LEGACYPIPE_CONTEXT['gpu_device_id']
    worker_gpumode = _GLOBAL_LEGACYPIPE_CONTEXT['gpumode']

    # Unpack flags from X_inner to decide behavior
    # [..., use_gpu(-4), gpumode(-3), bid(-2), is_priority(-1)]
    task_wants_gpu = X_inner[-4]

    # Create the stripped list (remove 'is_priority' flag so one_blob is happy)
    # Indices in Xlist are now: [..., use_gpu(-3), gpumode(-2), bid(-1)]
    Xlist = list(X_inner[:-1])

    # If the task wants a GPU but this worker is CPU-only, force CPU mode
    if task_wants_gpu and not is_gpu_worker:
        Xlist[-3] = False # use_gpu = False
        Xlist[-2] = 0     # gpumode = 0
    else:
        # If we ARE a GPU worker, ensure the task uses the worker's assigned gpumode
        # (Usually 2, but we pass it explicitly to be sure)
        Xlist[-2] = worker_gpumode

    # Final conversion back to tuple for the call
    X_to_pass = tuple(Xlist)

    info(f"Worker PID {pid}: Context {is_gpu_worker=}, {gpu_device_id=} running {iblob=} at {datetime.datetime.now()}")

    # --- 3. EXECUTION ---
    try:
        if is_gpu_worker:
            import cupy as cp
            cp.cuda.Device(gpu_device_id).use()

            # Check Memory Availability
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            free_mem_g = free_mem / 1024.**3
            if free_mem_g < 1.0:
                print(f"Free memory under 1 GiB {free_mem_g=:.2f}; Running {pid} in CPU mode for Blob {iblob}.")
                # Temporarily override X for this call only
                temp_list = list(X_to_pass)
                temp_list[-3] = False # use_gpu
                temp_list[-2] = 0     # gpumode
                X_to_pass = tuple(temp_list)

        # Execute one_blob
        result = one_blob(X_to_pass)

        # Post-processing for sub-blobs (De-duplication)
        if result is not None and blob_unique is not None:
            x0, x1, y0, y1 = blob_unique
            ntot = len(result)
            if ntot > 0:
                # Keep only sources where the center falls within our unique sub-blob bounds
                result.cut((result.bx0 >= x0) * (result.bx0 < x1) *
                           (result.by0 >= y0) * (result.by0 < y1))
                # debug('Blob_unique cut kept', len(result), 'of', ntot, 'sources')

        return dict(brickname=brickname, iblob=iblob, result=result)

    except Exception:
        print(f'Exception in one_blob: brick {brickname}, iblob {iblob}')
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

def _get_both_mods(*X):
    from astrometry.util.resample import resample_with_wcs, OverlapError
    from astrometry.util.miscutils import get_overlapping_region
    (tim, targetwcs, srcs, srcblobs, blobmap, frozen_galaxies, ps, plots) = X
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

    srcs_blobs = list(zip(srcs, srcblobs))

    fro_rd = set()
    if frozen_galaxies is not None:
        from tractor.patch import ModelMask
        timblobs = set(timblobmap.ravel())
        timblobs.discard(-1)
        h,w = tim.shape
        mm = ModelMask(0, 0, w, h)
        for fro,bb in frozen_galaxies.items():
            # Does this source (which touches blobs bb) touch any blobs in this tim?
            touchedblobs = timblobs.intersection(bb)
            if len(touchedblobs) == 0:
                continue
            patch = fro.getModelPatch(tim, modelMask=mm)
            if patch is None:
                continue
            patch.addTo(mod)

            assert(patch.shape == mod.shape)
            # np.isin doesn't work with a *set* argument!
            blobmask = np.isin(timblobmap, list(touchedblobs))
            blobmod += patch.patch * blobmask

            if plots:
                import pylab as plt
                plt.clf()
                plt.imshow(blobmask, interpolation='nearest', origin='lower', vmin=0, vmax=1,
                           cmap='gray')
                plt.title('tim %s: frozen-galaxy blobmask' % tim.name)
                ps.savefig()
                plt.clf()
                plt.imshow(patch.patch, interpolation='nearest', origin='lower',
                           cmap='gray')
                plt.title('tim %s: frozen-galaxy patch' % tim.name)
                ps.savefig()

            # Drop this frozen galaxy from the catalog to render, if it is present
            # (ie, if it is in_bounds)
            fro_rd.add((fro.pos.ra, fro.pos.dec))

    NEA = []
    no_nea = [0.,0.,0.]
    pcal = tim.getPhotoCal()
    for src,srcblob in srcs_blobs:
        if src is None:
            NEA.append(no_nea)
            continue
        if (src.pos.ra, src.pos.dec) in fro_rd:
            # Skip frozen galaxy source (here we choose not to compute NEA)
            NEA.append(no_nea)
            continue
        patch = src.getModelPatch(tim)
        if patch is None:
            NEA.append(no_nea)
            continue
        # From patch.addTo() -- find pixel overlap region
        (ih, iw) = mod.shape
        (ph, pw) = patch.shape
        (outx, inx) = get_overlapping_region(
            patch.x0, patch.x0 + pw - 1, 0, iw - 1)
        (outy, iny) = get_overlapping_region(
            patch.y0, patch.y0 + ph - 1, 0, ih - 1)
        if inx == [] or iny == []:
            NEA.append(no_nea)
            continue
        # model image patch
        p = patch.patch[iny, inx]
        # add to model image
        mod[outy, outx] += p
        # mask by blob map
        maskedp = p * (timblobmap[outy,outx] == srcblob)
        # add to blob-masked image
        blobmod[outy, outx] += maskedp
        # per-image NEA computations
        # total flux
        flux = pcal.brightnessToCounts(src.brightness)
        # flux in patch
        pflux = np.sum(p)
        # weighting -- fraction of flux that is in the patch
        fracin = pflux / flux
        # nea
        if pflux == 0: # sum(p**2) can only be zero if all(p==0), and then pflux==0
            nea = 0.
        else:
            nea = pflux**2 / np.sum(p**2)
        mpsq = np.sum(maskedp**2)
        if mpsq == 0 or pflux == 0:
            mnea = 0.
        else:
            mnea = flux**2 / mpsq
        NEA.append([nea, mnea, fracin])

    if hasattr(tim.psf, 'clear_cache'):
        tim.psf.clear_cache()
    return [mod, blobmod], NEA

def stage_coadds(survey=None, bands=None, version_header=None, targetwcs=None,
                 tims=None, ps=None, brickname=None, ccds=None,
                 custom_brick=False,
                 T=None,
                 refobjs=None,
                 blobmap=None,
                 cat=None, pixscale=None, plots=False,
                 coadd_bw=False, brick=None, W=None, H=None, lanczos=True,
                 co_sky=None,
                 saturated_pix=None,
                 refmap=None,
                 frozen_galaxies=None,
                 bailout_mask=None,
                 sub_blob_mask=None,
                 coadd_headers={},
                 save_coadd_psf=False,
                 mp=None,
                 record_event=None,
                 **kwargs):
    '''
    After the `stage_fitblobs` fitting stage, we have all the source
    model fits, and we can create coadds of the images, model, and
    residuals.  We also perform aperture photometry in this stage.
    '''
    import time
    from functools import reduce
    from legacypipe.survey import apertures_arcsec
    from legacypipe.bits import REF_MAP_BITS, FITBITS_DESCRIPTIONS, maskbits_type
    from legacypipe.survey import clean_band_name
    #return None

    record_event and record_event('stage_coadds: starting')
    _add_stage_version(version_header, 'COAD', 'coadds')
    tlast = Time()

    # Write per-brick CCDs table
    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='ccdinfo',
                            comment='NOIRLab data product type'))
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=primhdr, extname='CCDS')

    if plots and False:
        import pylab as plt
        from astrometry.util.plotutils import dimshow
        cat_init = [src for it,src in zip(T.iterative, cat) if not(it)]
        cat_iter = [src for it,src in zip(T.iterative, cat) if it]
        info(len(cat_init), 'initial sources and', len(cat_iter), 'iterative')
        mods_init = mp.map(_get_mod, [(tim, cat_init) for tim in tims])
        mods_iter = mp.map(_get_mod, [(tim, cat_iter) for tim in tims])
        coimgs_init,_ = quick_coadds(tims, bands, targetwcs, images=mods_init)
        coimgs_iter,_ = quick_coadds(tims, bands, targetwcs, images=mods_iter)
        coimgs,_ = quick_coadds(tims, bands, targetwcs)
        plt.clf()
        rgb,kw = survey.get_rgb(coimgs, bands)
        dimshow(rgb, **kw)
        plt.title('First-round data')
        ps.savefig()
        plt.clf()
        rgb,kw = survey.get_rgb(coimgs_init, bands)
        dimshow(rgb, **kw)
        plt.title('First-round model fits')
        ps.savefig()
        plt.clf()
        rgb,kw = survey.get_rgb([img-mod for img,mod in zip(coimgs,coimgs_init)], bands)
        dimshow(rgb, **kw)
        plt.title('First-round residuals')
        ps.savefig()
        plt.clf()
        rgb,kw = survey.get_rgb(coimgs_iter, bands)
        dimshow(rgb, **kw)
        plt.title('Iterative model fits')
        ps.savefig()
        plt.clf()
        rgb,kw = survey.get_rgb([mod+mod2 for mod,mod2 in zip(coimgs_init, coimgs_iter)], bands)
        dimshow(rgb, **kw)
        plt.title('Initial + Iterative model fits')
        ps.savefig()
        plt.clf()
        rgb,kw = survey.get_rgb([img-mod-mod2 for img,mod,mod2 in zip(coimgs,coimgs_init,coimgs_iter)], bands)
        dimshow(rgb, **kw)
        plt.title('Iterative model residuals')
        ps.savefig()

    record_event and record_event('stage_coadds: coadds')

    # Re-add the blob that this galaxy is actually inside
    # (that blob got dropped way earlier, before fitblobs)
    if frozen_galaxies is not None:
        for src,bb in frozen_galaxies.items():
            _,xx,yy = targetwcs.radec2pixelxy(src.pos.ra, src.pos.dec)
            xx = int(xx-1)
            yy = int(yy-1)
            bh,bw = blobmap.shape
            if xx >= 0 and xx < bw and yy >= 0 and yy < bh:
                # in bounds!
                debug('Frozen galaxy', src, 'lands in blob', blobmap[yy,xx])
                if blobmap[yy,xx] != -1:
                    bb.append(blobmap[yy,xx])

    # Input args for _get_both_mods
    Ireg = np.flatnonzero(T.regular)
    Nreg = len(Ireg)
    reg_cat = [cat[i] for i in Ireg]
    reg_blob = T.blob[Ireg]
    both_args = [(reg_cat, reg_blob, blobmap, frozen_galaxies, ps, plots)
                 for tim in tims]

    # source pixel positions to probe depth maps, etc
    ixy = (np.clip(T.ibx, 0, W-1).astype(int), np.clip(T.iby, 0, H-1).astype(int))
    # convert apertures to pixels
    apertures = apertures_arcsec / pixscale
    # Aperture photometry locations
    apxy = np.vstack((T.bx, T.by)).T

    C = make_coadds(tims, bands, targetwcs,
                    xy=ixy,
                    ngood=True, detmaps=True, psfsize=True, allmasks=True,
                    lanczos=lanczos,
                    apertures=apertures, apxy=apxy,
                    psf_images=save_coadd_psf,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, tims,
                                   targetwcs, co_sky, coadd_headers),
                    mod_callback=_get_both_mods, mod_callback_args=both_args,
                    plots=plots, ps=ps, mp=mp)

    record_event and record_event('stage_coadds: extras')
    if save_coadd_psf:
        for band,psfimg in zip(bands, C.psf_imgs):
            with survey.write_output('copsf', brick=brickname, band=band) as out:
                out.fits.write(psfimg, header=version_header)

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
    cols = ['nobs', 'ngood', 'anymask', 'allmask', 'psfsize', 'psfdepth', 'galdepth',
            'mjd_min', 'mjd_max']
    # store galaxy sim bounding box in Tractor cat
    if 'sims_xy' in C.T.get_columns():
        cols.append('sims_xy')
    for c in cols:
        T.set(c, C.T.get(c))

    # average NEA stats per band -- after psfsize,psfdepth computed.
    # first init all bands expected by format_catalog
    for band in survey.allbands:
        T.set('nea_%s' % band, np.zeros(len(T), np.float32))
        T.set('blob_nea_%s' % band, np.zeros(len(T), np.float32))
    for iband,(band,cb_data) in enumerate(zip(bands, C.mod_callback_data)):
        num  = np.zeros(Nreg, np.float32)
        den  = np.zeros(Nreg, np.float32)
        bnum = np.zeros(Nreg, np.float32)
        btims = [tim for tim in tims if tim.band == band]
        assert(len(btims) == len(cb_data))
        # cb_data: list-of-lists, (ntim x nsrcs x 3)
        for tim,data in zip(btims, cb_data):
            data = np.array(data)
            assert(data.shape[-1] == 3)
            nea    = data[:,0]
            bnea   = data[:,1]
            nea_wt = data[:,2]
            iv = 1./(tim.sig1**2)
            I, = np.nonzero(nea)
            wt = nea_wt[I]
            num[I] += iv * wt * 1./(nea[I] * tim.imobj.pixscale**2)
            den[I] += iv * wt
            I, = np.nonzero(bnea)
            bnum[I] += iv * 1./bnea[I]
        # bden is the coadded per-pixel inverse variance derived from psfdepth and psfsize
        # this ends up in arcsec units, not pixels
        bden = T.psfdepth[Ireg,iband] * (4 * np.pi * (T.psfsize[Ireg,iband]/2.3548)**2)
        # numerator and denominator are for the inverse-NEA!
        with np.errstate(divide='ignore', invalid='ignore'):
            nea  = den  / num
            bnea = bden / bnum
        nea [np.logical_not(np.isfinite(nea ))] = 0.
        bnea[np.logical_not(np.isfinite(bnea))] = 0.
        # Set vals in T
        T.get('nea_%s' % band)[Ireg] = nea
        T.get('blob_nea_%s' % band)[Ireg] = bnea

    # Grab aperture fluxes
    assert(C.AP is not None)
    # How many apertures?
    A = len(apertures_arcsec)
    for src,dst in [('apflux_img_%s',       'apflux'),
                    ('apflux_img_ivar_%s',  'apflux_ivar'),
                    ('apflux_masked_%s',    'apflux_masked'),
                    ('apflux_resid_%s',     'apflux_resid'),
                    ('apflux_blobresid_%s', 'apflux_blobresid'),]:
        X = np.zeros((len(T), len(bands), A), np.float32)
        for iband,band in enumerate(bands):
            X[:,iband,:] = C.AP.get(src % band)
        T.set(dst, X)

    # Compute depth histogram
    D = _depth_histogram(brick, targetwcs, bands, C.psfdetivs, C.galdetivs)
    with survey.write_output('depth-table', brick=brickname) as out:
        D.writeto(None, fits_object=out.fits, extname='DEPTH')
    del D

    U = None
    # BRICK_PRIMARY pixel mask
    if not custom_brick:
        U = find_unique_pixels(targetwcs, W, H, None,
                               brick.ra1, brick.ra2, brick.dec1, brick.dec2)

    # Create JPEG coadds
    coadd_list= [('image', C.coimgs, {}, None),
                 ('model', C.comods, {}, None),
                 ('blobmodel', C.coblobmods, {}, None,),
                 ('resid', C.coresids, dict(resids=True), U)]
    if hasattr(tims[0], 'sims_image'):
        coadd_list.append(('simscoadd', sims_coadd, {}, None))

    record_event and record_event('stage_coadds: writing image files')
    for name,ims,rgbkw,mask in coadd_list:
        if mask is not None:
            # Update in-place!
            ims = [im * mask for im in ims]
            del mask
        rgb,kwa = survey.get_rgb(ims, bands, **rgbkw)
        with survey.write_output(name + '-jpeg', brick=brickname) as out:
            imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
            info('Wrote', out.fn)
        del rgb
        del ims
    del C.comods
    del C.coblobmods
    del C.coresids

    record_event and record_event('stage_coadds: maskbits')
    # Construct the maskbits map
    MASKBITS = survey.get_maskbits()
    maskbits = np.zeros((H,W), maskbits_type)
    if not custom_brick:
        # not BRICK_PRIMARY
        maskbits |= MASKBITS['NPRIMARY'] * np.logical_not(U).astype(maskbits_type)
        del U

    # reference-catalog masks
    if refmap is not None:
        for key in ['BRIGHT', 'MEDIUM', 'GALAXY', 'CLUSTER', 'RESOLVED', 'MCLOUDS']:
            maskbits |= (MASKBITS[key] * ((refmap & REF_MAP_BITS[key]) > 0)).astype(maskbits_type)
        del refmap

    cleanbands = [clean_band_name(b).upper() for b in bands]
    # SATUR
    if saturated_pix is not None:
        for b, sat in zip(cleanbands, saturated_pix):
            key = 'SATUR_' + b
            if key in MASKBITS:
                maskbits |= (maskbits_type(MASKBITS[key]) * sat).astype(maskbits_type)
            else:
                print('SATUR key %s not in MASKBITS' % key)
    # ALLMASK_{g,r,z}
    for b,allmask in zip(cleanbands, C.allmasks):
        key = 'ALLMASK_' + b
        if key in MASKBITS:
            maskbits |= (maskbits_type(MASKBITS[key]) * (allmask > 0)).astype(maskbits_type)
        else:
            print('ALLMASK key %s not in MASKBITS' % key)

    # BAILOUT
    if bailout_mask is not None:
        maskbits |= (MASKBITS['BAILOUT'] * bailout_mask.astype(bool)).astype(maskbits_type)

    # SUB_BLOB
    if sub_blob_mask is not None:
        maskbits |= (MASKBITS['SUB_BLOB'] * sub_blob_mask.astype(bool)).astype(maskbits_type)

    # Add the maskbits header cards to version_header
    mbits = survey.get_maskbits_descriptions()
    version_header.add_record(dict(name='COMMENT', value='maskbits bits:'))
    _add_bit_description(version_header, MASKBITS, mbits, 'MB_%s', 'MBIT_%i', 'maskbit')

    # Add the fitbits header cards to version_header
    version_header.add_record(dict(name='COMMENT', value='fitbits bits:'))
    _add_bit_description(version_header, FITBITS, FITBITS_DESCRIPTIONS,
                         'FB_%s', 'FBIT_%i', 'fitbit')

    if plots:
        import pylab as plt
        from astrometry.util.plotutils import dimshow
        plt.clf()
        ra  = np.array([src.getPosition().ra  for src in cat])
        dec = np.array([src.getPosition().dec for src in cat])
        x0,y0 = T.bx0, T.by0
        ok,x1,y1 = targetwcs.radec2pixelxy(ra, dec)
        x1 -= 1.
        y1 -= 1.
        rgb,kw = survey.get_rgb(C.coimgs, bands)
        dimshow(rgb, **kw)
        ax = plt.axis()
        for xx0,yy0,xx1,yy1 in zip(x0,y0,x1,y1):
            plt.plot([xx0,xx1], [yy0,yy1], 'r-')
        plt.plot(x1, y1, 'r.')
        plt.axis(ax)
        plt.title('Original to final source positions')
        ps.savefig()

        plt.clf()
        rgb,kw = survey.get_rgb(C.coimgs, bands)
        dimshow(rgb, **kw)
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
    debug('Aperture photometry wrap-up:', tnow-tlast)

    return dict(T=T, apertures_pix=apertures,
                apertures_arcsec=apertures_arcsec,
                maskbits=maskbits,
                version_header=version_header)

def _add_bit_description(header, BITS, bits, bnpat, bitpat, bitmapname):
    # BITS: name -> bit value (eg 0x400)
    # bits: [name, description] list
    # bnpat: eg "MB_%s"
    # bitpat: eg "MBIT_%s"
    for key,comm in bits:
        header.add_record(
            dict(name=bnpat % key,
                 value=BITS[key],
                 comment='%s: %s' % (bitmapname, comm)))
    revmap = dict([(bit,name) for name,bit in BITS.items()])
    commentmap = dict(bits)
    for bit in range(32):
        bitval = 1<<bit
        if not bitval in revmap:
            continue
        name = revmap[bitval]
        comment = commentmap.get(name, '')
        header.add_record(
            dict(name=bitpat % bit,
                 value=name,
                 comment='%s %i (0x%x): %s' % (bitmapname, bit, bitval, comment)))

def get_fiber_fluxes(cat, T, targetwcs, H, W, pixscale, bands,
                     fibersize=1.5, seeing=1., year=2020.0,
                     plots=False, ps=None):
    from tractor import GaussianMixturePSF
    from legacypipe.survey import LegacySurveyWcs
    import astropy.time
    from tractor.tractortime import TAITime
    from tractor.image import Image
    from tractor.basics import LinearPhotoCal
    from photutils.aperture import CircularAperture, aperture_photometry

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
    for isrc,src in enumerate(cat):
        if src is None:
            continue
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
            sx,sy = faketim.getWcs().positionToPixel(src.getPosition())
            aper = CircularAperture((sx, sy), fiberrad)
            p = aperture_photometry(onemod, aper)
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
    apxy = np.vstack((T.bx, T.by)).T
    aper = CircularAperture(apxy, fiberrad)
    for iband,modimg in enumerate(modimgs):
        p = aperture_photometry(modimg, aper)
        f = p.field('aperture_sum')
        # If the source is off the brick (eg, ref sources), can be NaN
        I = np.isfinite(f)
        if len(I):
            fibertotflux[I, iband] = f[I]

    if plots:
        import pylab as plt
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
    wise_ceres=True,
    unwise_coadds=True,
    version_header=None,
    maskbits=None,
    mp=None,
    record_event=None,
    wise_checkpoint_filename=None,
    wise_checkpoint_period=600,
    save_unwise_psf=False,
    ps=None,
    plots=False,
    **kwargs):
    '''
    After the model fits are finished, we can perform forced
    photometry of the unWISE coadds.
    '''
    from legacypipe.unwise import unwise_phot, collapse_unwise_bitmask, unwise_tiles_touching_wcs
    from legacypipe.survey import wise_apertures_arcsec
    from tractor import NanoMaggies

    record_event and record_event('stage_wise_forced: starting')
    _add_stage_version(version_header, 'WISE', 'wise_forced')
    version_header.add_record(dict(name='W_CERES', value=wise_ceres,
                                   comment='WISE forced phot: use Ceres optimizer?'))
    if not plots:
        ps = None

    tiles = unwise_tiles_touching_wcs(targetwcs)
    info('Cut to', len(tiles), 'unWISE tiles')

    # the way the roiradec box is used, the min/max order doesn't matter
    roiradec = [targetrd[0,0], targetrd[2,0], targetrd[0,1], targetrd[2,1]]

    # Sources to photometer
    do_phot = T.regular.copy()

    # Drop sources within the CLUSTER mask from forced photometry.
    Icluster = None
    if maskbits is not None:
        MASKBITS = survey.get_maskbits()
        incluster = (maskbits & MASKBITS['CLUSTER'] > 0)
        if np.any(incluster):
            info('Checking for sources inside CLUSTER mask')
            # With --bail-out, we can have (reference) sources set to None
            Igood = np.flatnonzero([src is not None for src in cat])
            # compute x,y for the good ones
            ra  = np.array([cat[i].getPosition().ra  for i in Igood])
            dec = np.array([cat[i].getPosition().dec for i in Igood])
            ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)
            xx = np.round(xx - 1).astype(int)
            yy = np.round(yy - 1).astype(int)
            xygood = (ok * (xx >= 0) * (xx < W) * (yy >= 0) * (yy < H))
            Igood = Igood[xygood]
            if len(Igood):
                Icluster = Igood[incluster[yy[xygood], xx[xygood]]]
                info('Found', len(Icluster), 'of', len(cat), 'sources inside CLUSTER mask')
                do_phot[Icluster] = False
            del Icluster,Igood,xygood,ra,dec,ok,xx,yy
        del incluster
    Nskipped = len(T) - np.sum(do_phot)

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
            args.append(((-1,band),
                         (wcat, wtiles, band, roiradec, wise_ceres, wpixpsf,
                          unwise_coadds, get_masks, ps, True,
                          unwise_modelsky_dir, 'Full-depth W%i' % (band))))

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
                eargs.append(((ie,band),
                              (wcat, eptiles, band, roiradec,
                               wise_ceres, wpixpsf, False, None, ps, False,
                               unwise_modelsky_dir, 'Epoch %i W%i' % (ie+1, band))))

    runargs = args + eargs
    info('unWISE forced phot: total of', len(runargs), 'images to photometer')
    photresults = {}
    # Check for existing checkpoint file.
    if wise_checkpoint_filename and os.path.exists(wise_checkpoint_filename):
        from astrometry.util.file import unpickle_from_file
        info('Reading', wise_checkpoint_filename)
        try:
            photresults = unpickle_from_file(wise_checkpoint_filename)
            info('Read', len(photresults), 'results from checkpoint file', wise_checkpoint_filename)
        except:
            import traceback
            print('Failed to read checkpoint file', wise_checkpoint_filename)
            traceback.print_exc()
        n_a = len(runargs)
        runargs = [(key,a) for (key,a) in runargs if not key in photresults]
        print('Running', len(runargs), 'of', n_a, 'images not in checkpoint')

    # Run the forced photometry!
    record_event and record_event('stage_wise_forced: photometry')
    #phots = mp.map(unwise_phot, args + eargs)

    if wise_checkpoint_filename is None or mp is None:
        res = mp.map(unwise_phot, runargs)
        for k,v in res:
            photresults[k] = v
        del res
    elif len(runargs) > 0:
        res = mp.imap_unordered(unwise_phot, runargs)
        from astrometry.util.ttime import CpuMeas
        import multiprocessing
        import concurrent.futures
        last_checkpoint = CpuMeas()
        n_finished = 0
        n_finished_total = 0
        while True:
            # Time to write a checkpoint file? (And have something to write?)
            tnow = CpuMeas()
            dt = tnow.wall_seconds_since(last_checkpoint)
            if dt >= wise_checkpoint_period and n_finished > 0:
                # Write checkpoint!
                info('Writing checkpoint:', n_finished, 'new results; total for this run', n_finished_total, 'total:', len(photresults))
                try:
                    _write_checkpoint(photresults, wise_checkpoint_filename)
                    last_checkpoint = tnow
                    dt = 0.
                    n_finished = 0
                except:
                    print('Failed to write checkpoint file', wise_checkpoint_filename)
                    import traceback
                    traceback.print_exc()
            # Wait for results (with timeout)
            try:
                info('waiting for result (%i to go)...' % (len(runargs)-n_finished_total))
                if mp.pool is not None:
                    timeout = max(1, wise_checkpoint_period - dt)
                    # If we don't have any new results to write, wait indefinitely
                    if n_finished == 0:
                        timeout = None
                    r = res.next(timeout)
                else:
                    r = next(res)
                k,v = r
                info('got result for epoch,band', k)
                photresults[k] = v
                n_finished += 1
                n_finished_total += 1
            except StopIteration:
                #info('got StopIteration')
                break
            except multiprocessing.TimeoutError:
                #info('got TimeoutError')
                continue
            except concurrent.futures.TimeoutError:
                #info('got MPI TimeoutError')
                continue
            except TimeoutError:
                continue
            except:
                import traceback
                traceback.print_exc()
        # Write checkpoint when done!
        _write_checkpoint(photresults, wise_checkpoint_filename)
        info('Computed', n_finished_total, 'new results; wrote', len(photresults), 'to checkpoint')

    phots = [photresults[k] for k,a in (args + eargs)]
    record_event and record_event('stage_wise_forced: results')

    # Unpack results...
    WISE = None
    wise_mask_maps = None
    if len(phots):
        # The "phot" results for the full-depth coadds are one table per
        # band.  Merge all those columns.
        wise_models = []
        psfs = {}
        for i,p in enumerate(phots[:len(args)]):
            key,theargs = args[i]
            epoch,band = key
            if p is None:
                (wcat,tiles) = theargs[:2]
                info('"None" result from WISE forced phot:', tiles, band, 'epoch', epoch)
                continue
            if unwise_coadds:
                wise_models.extend(p.models)
            if p.maskmap is not None:
                wise_mask_maps = p.maskmap
            if save_unwise_psf:
                if not band in psfs:
                    psfs[band] = []
                psfs[band].extend(p.psfs)
            if WISE is None:
                WISE = p.phot
            else:
                # remove duplicates
                p.phot.delete_column('wise_coadd_id')
                # (with move_crpix -- Aaron's update astrometry -- the
                # pixel positions can be *slightly* different per
                # band.  Ignoring that here.)
                p.phot.delete_column('wise_x')
                p.phot.delete_column('wise_y')
                WISE.add_columns_from(p.phot)

        if wise_mask_maps is not None:
            wise_mask_maps = [
                collapse_unwise_bitmask(wise_mask_maps, 1),
                collapse_unwise_bitmask(wise_mask_maps, 2)]

        if Nskipped > 0:
            assert(len(WISE) == len(wcat))
            WISE = _fill_skipped_values(WISE, Nskipped, do_phot)
            assert(len(WISE) == len(cat))
            assert(len(WISE) == len(T))

        if unwise_coadds:
            from legacypipe.coadds import UnwiseCoadd
            # Create the WCS into which we'll resample the tiles.
            # Same center as "targetwcs" but bigger pixel scale.
            wpixscale = 2.75
            rc,dc = targetwcs.radec_center()
            ww = int(W * pixscale / wpixscale)
            hh = int(H * pixscale / wpixscale)
            wcoadds = UnwiseCoadd(rc, dc, ww, hh, wpixscale)
            wcoadds.add(wise_models, unique=True)
            apphot = wcoadds.finish(survey, brickname, version_header,
                                    apradec=(T.ra,T.dec),
                                    apertures=wise_apertures_arcsec/wpixscale)
            api,apd,apr,_ = apphot
            for iband,band in enumerate([1,2,3,4]):
                WISE.set('apflux_w%i' % band, api[iband])
                WISE.set('apflux_resid_w%i' % band, apr[iband])
                d = apd[iband]
                iv = np.zeros_like(d)
                iv[d != 0.] = 1./(d[d != 0]**2)
                WISE.set('apflux_ivar_w%i' % band, iv)

        # Look up mask values for sources
        WISE.wise_mask = np.zeros((len(cat), 2), np.uint8)
        if wise_mask_maps is not None:
            WISE.wise_mask[T.in_bounds,0] = wise_mask_maps[0][T.iby[T.in_bounds], T.ibx[T.in_bounds]]
            WISE.wise_mask[T.in_bounds,1] = wise_mask_maps[1][T.iby[T.in_bounds], T.ibx[T.in_bounds]]

        if save_unwise_psf:
            for band,psflist in psfs.items():
                assert(len(psflist) > 0)
                psfimg = 0.
                for p in psflist:
                    psfimg = psfimg + p.getImage(0., 0.)
                psfimg /= len(psflist)
                with survey.write_output('copsf', brick=brickname, band='W%i' % band) as out:
                    out.fits.write(psfimg, header=version_header)

    # Unpack time-resolved results...
    WISE_T = None
    if len(phots) > len(args):
        WISE_T = True
    if WISE_T is not None:
        WISE_T = fits_table()
        phots = phots[len(args):]
        # eargs contains [ (key,args) ]
        for ((ie,_),_),r in zip(eargs, phots):
            debug('Epoch', ie, 'photometry:')
            if r is None:
                debug('Failed.')
                continue
            assert(ie < Nepochs)
            phot = r.phot
            phot.delete_column('wise_coadd_id')
            phot.delete_column('wise_x')
            phot.delete_column('wise_y')
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
                version_header=version_header,
                wise_apertures_arcsec=wise_apertures_arcsec)


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

def stage_forced_phot(survey=None, bands=None, forced_bands=None,
                      version_header=None, targetwcs=None,
                      targetrd=None,
                      release=None,
                      # TEMP
                      invvars=None,
                      gaia_stars=None,
                      # /TEMP
                      tims=None, ps=None, brickname=None, ccds=None,
                      custom_brick=False,
                      T=None,
                      refobjs=None,
                      blobmap=None,
                      cat=None, pixscale=None, plots=False,
                      gaussPsf=False, pixPsf=True, hybridPsf=True,
                      normalizePsf=True,
                      subsky=True,
                      apodize=False,
                      constant_invvar=False,
                      old_calibs_ok=True,
                      coadd_bw=False, brick=None, W=None, H=None, lanczos=True,
                      star_halos=True,
                      use_ceres=True,
                      co_sky=None,
                      saturated_pix=None,
                      refmap=None,
                      frozen_galaxies=None,
                      bailout_mask=None,
                      sub_blob_mask=None,
                      coadd_headers={},
                      mp=None,
                      record_event=None,
                      **kwargs):
    '''
    Read in images from the survey-ccds files in *forced_bands* bands.

    Perform forced photometry on them, and also accumulate a coadd.

    (Also run forced-photometry on the individual exposures in *tims* ??)
    '''
    from legacypipe.survey import clean_band_name
    from tractor import NanoMaggies
    from legacypipe.utils import run_ps
    from legacypipe.survey import read_one_tim, apertures_arcsec
    from legacypipe.halos import subtract_one
    from legacypipe.forced_photom import run_forced_phot, forced_phot_add_extra_fields
    from legacypipe.outliers import mask_outlier_pixels
    from legacypipe.units import get_units_for_columns

    if forced_bands is None:
        return dict()

    tlast = Time()
    record_event and record_event('stage_forced_phot: starting')

    # Before we begin, free the *tims* to reduce our memory use,
    # before reading in the *forced_bands* imaging data.

    pid = os.getpid()
    procs = run_ps(pid)
    print('PS:')
    procs.about()
    print('pmem:', procs.pmem)
    print('proc_vmpeak:', procs.proc_vmpeak)
    print('vsz:', procs.vsz)

    for i in range(len(tims)):
        tims[i] = None

    procs = run_ps(pid, last=procs)
    print('PS:')
    procs.about()
    print('pmem:', procs.pmem)
    print('proc_vmpeak:', procs.proc_vmpeak)
    print('vsz:', procs.vsz)

    print('T:')
    T.about()

    clean_bands = [clean_band_name(b) for b in forced_bands]
    clean_map = dict(list(zip(forced_bands, clean_bands)))

    # Copied code from stage_tims...
    ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)
    survey.drop_cache()
    if 'ccd_cuts' in ccds.get_columns():
        ccds.cut(ccds.ccd_cuts == 0)
        debug(len(ccds), 'CCDs survive cuts')
    else:
        warnings.warn('Not applying CCD cuts')
    # Cut on bands to be used
    ccds.cut(np.array([b in forced_bands for b in ccds.filter]))
    debug('Cut to', len(ccds), 'CCDs in bands', ','.join(forced_bands))

    # Not applying mjd_minmax or ccds_for_fitting cuts?

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

    # Skipping do_calibs

    # args for read_one_tim
    tim_args = dict(gaussPsf=gaussPsf, pixPsf=pixPsf,
                    hybridPsf=hybridPsf, normalizePsf=normalizePsf,
                    subsky=subsky,
                    apodize=apodize,
                    constant_invvar=constant_invvar,
                    pixels=True,
                    old_calibs_ok=old_calibs_ok)

    tims = list(mp.map(read_one_tim, [(im, targetrd, tim_args) for im in ims]))

    # Cut the table of CCDs to match the 'tims' list
    I = np.array([i for i,tim in enumerate(tims) if tim is not None])
    ccds.cut(I)
    tims = [tim for tim in tims if tim is not None]
    assert(len(ccds) == len(tims))
    if len(tims) == 0:
        return dict()

    # outliers

    # no before-n-after outlier mask plots?
    # no "Patch individual-CCD masked pixels from a coadd"
    print('Masking outlier pixels...')
    mask_outlier_pixels(
        survey, tims, forced_bands, targetwcs, brickname, version_header,
        mp=mp, plots=plots, ps=ps, make_badcoadds=False, refobjs=refobjs,
        write_mask_file=False)

    # from stage_halos...
    print('Subtracting stellar halos...')
    if star_halos and refobjs:
        Igaia, = np.nonzero(refobjs.isgaia * refobjs.pointsource)
        debug(len(Igaia), 'stars for halo subtraction')
        if len(Igaia):
            from legacypipe.halos import subtract_halos
            halostars = refobjs[Igaia]
            subtract_halos(tims, halostars, forced_bands, mp, plots, ps, old_calibs_ok=old_calibs_ok)
    # subtract SGA galaxies outside the chip?
    # (only if we have SGA photometry for this band...)

    set_brick_primary(T, brick)

    # Which sources to photometer... in forced_phot.py, we cut to brick_primary, but for consistency
    # with eg DR10, let's keep all the sources.
    do_phot = (#np.logical_or(T.brick_primary, T.ref_cat == 'L3') *
               (T.dup == False) * np.array([src is not None for src in cat]))
    # (we don't have T.type yet; this is equivalent to T.type not equal to 'DUP' or 'NUN')

    print('Objects to forced-photometer: catalog contains %i total.  %i are non-DUP.  %i have sources.  %i satisfy both.' % (len(T), np.sum(T.dup == False), len([src is not None for src in cat]), np.sum(do_phot)))

    # This will get multiprocessed...
    args = [list(a) + [cat, T, do_phot, release, use_ceres] for a in zip(ccds, ims, tims)]
    FF = mp.map(_forced_phot_one, args)
    FF = [F for F in FF if F is not None]
    F = merge_tables(FF, columns='fillzero')
    print('All forced photometry results:')
    F.about()
    del FF

    mag_unit = 'mag'
    pixel_unit = 'pixel'
    columns = F.get_columns()
    eunits = {'ccdzpt': mag_unit,
              'ccdphrms': mag_unit,
              'x': pixel_unit,
              'y': pixel_unit,
              }
    units = get_units_for_columns(columns, extras=eunits)
    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='forced',
                            comment='NOIRLab data product type'))
    # add forced_bands headers similar to those for bands
    primhdr.add_record(dict(name='FBANDS', value=','.join(forced_bands), comment='Forced-phot bands'))
    primhdr.add_record(dict(name='NFBANDS', value=len(forced_bands), comment='Number of forced-phot bands'))
    for i,band in enumerate(forced_bands):
        primhdr.add_record(dict(name='FBAND%i' % i, value=forced_bands[i], comment='Forced-phot band'))
    with survey.write_output('forced-brick', brick=brick.brickname) as out:
        F.writeto(None, fits_object=out.fits, primheader=primhdr,
                  units=units, columns=columns)

    # Create a fake table with just the forced-photometry results
    TF = fits_table()
    TF.brickid = T.brickid.copy()
    TF.objid = T.objid.copy()
    from legacypipe.forced_photom_brickwise import average_forced_phot
    forced_units = average_forced_phot(F, TF, prefix='')
    print('Averaged forced-phot results:')
    TF.about()
    del F

    # Create coadd
    print('Creating coadd...')
    # code from stage_coadds...
    # FIXME - ccds-table for forced-photometry results?

    # source pixel positions to probe depth maps, etc
    ixy = (np.clip(T.ibx, 0, W-1).astype(int), np.clip(T.iby, 0, H-1).astype(int))
    # convert apertures to pixels
    apertures = apertures_arcsec / pixscale
    # Aperture photometry locations
    apxy = np.vstack((T.bx, T.by)).T

    Ireg = np.flatnonzero(do_phot)
    Nreg = len(Ireg)
    # HACK -- replace the catalog brightnesses with the forced-photometry results!
    orig_bright = [cat[i].brightness for i in Ireg]
    for i in Ireg:
        br = dict([(b, TF.get('flux_%s' % b)[i]) for b in clean_bands])
        cat[i].brightness = NanoMaggies(**br)
    reg_cat = [cat[i] for i in Ireg]
    reg_blob = T.blob[Ireg]

    both_args = [(reg_cat, reg_blob, blobmap, frozen_galaxies, ps, plots)
                 for tim in tims]

    C = make_coadds(tims, forced_bands, targetwcs,
                    mods=None, xy=ixy, apertures=apertures, apxy=apxy,
                    ngood=True, detmaps=True, psfsize=True, allmasks=True,
                    mjdminmax=False,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, tims,
                                   targetwcs, co_sky, coadd_headers),
                    mod_callback=_get_both_mods, mod_callback_args=both_args,
                    mp=mp)

    # Revert catalog brightness
    for i,br in zip(Ireg, orig_bright):
        cat[i].brightness = br

    print('Coadd results contain:', dir(C))
    # 'AP', 'T', 'allmasks', 'coimgs', 'comods', 'coresids', 'cowimgs', 'galdetivs', 'maximgs', 'psfdetivs'
    print('Coadd Table contains:')
    C.T.about()

    # Save per-source measurements of the maps produced during coadding
    cols = ['nobs', 'ngood', 'anymask', 'allmask', 'psfsize', 'psfdepth', 'galdepth']
    for c in cols:
        X = C.T.get(c)
        print('Column', c, ': shape', X.shape, 'type', X.dtype)
        for i,band in enumerate(clean_bands):
            TF.set('%s_%s' % (c, band), X[:,i])

    # NEA
    # average NEA stats per band -- after psfsize,psfdepth computed.
    # first init all bands expected by format_catalog
    for band in clean_bands:
        T.set('nea_%s' % band, np.zeros(len(T), np.float32))
        T.set('blob_nea_%s' % band, np.zeros(len(T), np.float32))
    for iband,(band,cb_data) in enumerate(zip(forced_bands, C.mod_callback_data)):
        num  = np.zeros(Nreg, np.float32)
        den  = np.zeros(Nreg, np.float32)
        bnum = np.zeros(Nreg, np.float32)
        btims = [tim for tim in tims if tim.band == band]
        assert(len(btims) == len(cb_data))
        # cb_data: list-of-lists, (ntim x nsrcs x 3)
        for tim,data in zip(btims, cb_data):
            data = np.array(data)
            assert(data.shape[-1] == 3)
            nea    = data[:,0]
            bnea   = data[:,1]
            nea_wt = data[:,2]
            iv = 1./(tim.sig1**2)
            I, = np.nonzero(nea)
            wt = nea_wt[I]
            num[I] += iv * wt * 1./(nea[I] * tim.imobj.pixscale**2)
            den[I] += iv * wt
            I, = np.nonzero(bnea)
            bnum[I] += iv * 1./bnea[I]
        # bden is the coadded per-pixel inverse variance derived from psfdepth and psfsize
        # this ends up in arcsec units, not pixels
        bden = T.psfdepth[Ireg,iband] * (4 * np.pi * (T.psfsize[Ireg,iband]/2.3548)**2)
        # numerator and denominator are for the inverse-NEA!
        with np.errstate(divide='ignore', invalid='ignore'):
            nea  = den  / num
            bnea = bden / bnum
        nea [np.logical_not(np.isfinite(nea ))] = 0.
        bnea[np.logical_not(np.isfinite(bnea))] = 0.
        # Set vals in T
        T.get('nea_%s' % band)[Ireg] = nea
        T.get('blob_nea_%s' % band)[Ireg] = bnea

            
    # Grab aperture fluxes
    assert(C.AP is not None)

    print('Aperture flux table:')
    C.AP.about()
    # How many apertures?
    A = len(apertures_arcsec)
    for src,dst in [('apflux_img_%s',       'apflux'),
                    ('apflux_img_ivar_%s',  'apflux_ivar'),
                    ('apflux_masked_%s',    'apflux_masked'),
                    ('apflux_resid_%s',     'apflux_resid'),
                    #('apflux_blobresid_%s', 'apflux_blobresid'),
                    ]:
        for iband,band in enumerate(clean_bands):
            TF.set('%s_%s' % (dst, band), C.AP.get(src % band).astype(np.float32))

    cat2 = []
    for i,src in enumerate(cat):
        fluxes = {}
        for b in clean_bands:
            fluxes[b] = TF.get('flux_%s' % b)[i]
        src = src.copy()
        src.brightness = NanoMaggies(**fluxes)
        cat2.append(src)

    print('TF:', len(TF))
    TF.about()
    print('Catalog:', len(cat2))

    TF.bx = T.bx
    TF.by = T.by
    TF.flux_ivar = np.vstack([TF.get('flux_ivar_%s' % b) for b in clean_bands]).T
    ff, fftot = get_fiber_fluxes(
        cat2, TF, targetwcs, H, W, pixscale, clean_bands, plots=plots, ps=ps)
    for iband,band in enumerate(clean_bands):
        TF.set('fiberflux_%s'    % (band), ff   [:,iband])
        TF.set('fibertotflux_%s' % (band), fftot[:,iband])
    TF.delete_column('flux_ivar')

    print('After fiberfluxes:')
    TF.about()

    return dict(forced_T=TF, tims=tims)

def _forced_phot_one(args):
    from legacypipe.forced_photom import run_forced_phot, forced_phot_add_extra_fields
    from tractor import NanoMaggies

    ccd, im, tim, cat, T, do_phot, release, use_ceres = args

    print('Forced-photometering', tim)
    # Cut to sources within this chip
    chipwcs = tim.subwcs
    h,w = chipwcs.shape
    _,x,y = chipwcs.radec2pixelxy(T.ra, T.dec)
    # Same as get_catalog_in_wcs
    margin = 20
    H,W = tim.shape
    I = np.flatnonzero((x >= -margin) * (x <= (W+margin)) *
                       (y >= -margin) * (y <= (H+margin)) *
                       do_phot)
    # FIXME... this is going to drop SGA galaxies outside the margin
    # around this chip...

    # Note to self - when this gets fixed, make sure that special
    # sources like the MC (LMC/SMC), and RESOLVED sources, don't get
    # subtracted - they don't really have models.  Use the
    # *ignore_source* flag (which should be propagated from the ref
    # table to the sources).
    sub_cat = [cat[i] for i in I]

    print('Not subtracting SGA galaxies outside the image...')

    print('Forced-photometering %i sources in & near this image' % len(sub_cat))

    # create copies before modifying the Flux object
    sub_cat = [src.copy() for src in sub_cat]
    for src in sub_cat:
        fluxes = { tim.band : 1. }
        src.brightness = NanoMaggies(**fluxes)

    kwargs = {}
    #if plots:
    #kwargs.update(ps=ps)
    t0 = Time()
    F = run_forced_phot(sub_cat, tim,
                        ceres=use_ceres,
                        do_forced=True,
                        do_apphot=True,
                        full_position_fit=False,
                        windowed_peak=False,
                        timing=False, **kwargs)
    t1 = Time()
    print('run_forced_phot:', (t1-t0))

    if F is not None:
        derivs = False
        Tsub = T[I]
        Tsub.release = np.zeros(len(Tsub), np.int16) + release
        forced_phot_add_extra_fields(F, Tsub, ccd, im, tim, derivs)
    return F

def stage_writecat(
    survey=None,
    version_header=None,
    release=None,
    T=None,
    WISE=None,
    WISE_T=None,
    unwise_tr_dir=None,
    forced_T=None,
    forced_bands=None,
    maskbits=None,
    wise_mask_maps=None,
    apertures_arcsec=None,
    wise_apertures_arcsec=None,
    GALEX=None,
    galex_apertures_arcsec=None,
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
    import time
    from legacypipe.catalog import prepare_fits_catalog
    from legacypipe.utils import copy_header_with_wcs, add_bits
    from legacypipe.bits import WISE_MASK_BITS

    record_event and record_event('stage_writecat: starting')
    _add_stage_version(version_header, 'WCAT', 'writecat')

    assert(maskbits is not None)

    if wise_mask_maps is not None:
        # Add the WISE masks in!
        MASKBITS = survey.get_maskbits()
        maskbits |= MASKBITS['WISEM1'] * (wise_mask_maps[0] != 0)
        maskbits |= MASKBITS['WISEM2'] * (wise_mask_maps[1] != 0)

    version_header.add_record(dict(name='COMMENT', value='wisemask bits:'))
    bitvals = {}
    bitdescrs = []
    for bitnum,name,comment in WISE_MASK_BITS:
        bitvals[name] = 1 << bitnum
        bitdescrs.append((name, comment))
    _add_bit_description(version_header, bitvals, bitdescrs, 'WB_%s', 'WBIT_%s', 'wisemask')

    # Record the meaning of ALLMASK/ANYMASK bits
    add_bits(version_header, DQ_BITS, 'allmask/anymask', 'AM', 'A')

    # create maskbits header
    hdr = copy_header_with_wcs(version_header, targetwcs)
    hdr.add_record(dict(name='IMTYPE', value='maskbits',
                        comment='LegacySurveys image type'))
    with survey.write_output('maskbits', brick=brickname, shape=maskbits.shape) as out:
        out.fits.write(maskbits, header=hdr, extname='MASKBITS')
        if wise_mask_maps is not None:
            out.fits.write(wise_mask_maps[0], extname='WISEM1')
            out.fits.write(wise_mask_maps[1], extname='WISEM2')
        del wise_mask_maps

    T_orig = T.copy()

    T = prepare_fits_catalog(cat, invvars, T, bands, force_keep=T.force_keep_source)
    # Override type for DUP objects
    T.type[T.dup] = 'DUP'

    # The "ra_ivar" values coming out of the tractor fits do *not*
    # have a cos(Dec) term -- ie, they give the inverse-variance on
    # the numerical value of RA -- so we want to make the ra_sigma
    # values smaller by multiplying by cos(Dec); so invvars are /=
    # cosdec^2
    T.ra_ivar /= np.cos(np.deg2rad(T.dec))**2

    # Compute fiber fluxes
    debug('Computing fiber fluxes...')
    T.fiberflux, T.fibertotflux = get_fiber_fluxes(
        cat, T, targetwcs, H, W, pixscale, bands, plots=plots, ps=ps)

    # For reference *stars* only, plug in the reference-catalog inverse-variances.
    if 'ref_cat' in T.get_columns() and 'ra_ivar' in T_orig.get_columns():
        I = np.logical_or(T.isgaia, T.istycho)
        if len(I):
            T.ra_ivar [I] = T_orig.ra_ivar [I]
            T.dec_ivar[I] = T_orig.dec_ivar[I]

    # For "special" reference sources
    I = np.flatnonzero(np.logical_not(T.regular))
    T.bx0[I] = T.bx[I]
    T.by0[I] = T.by[I]

    # In oneblob.py we have a step where we zero out the fluxes for sources
    # with tiny "fracin" values.  Repeat that here, but zero out more stuff...
    for iband,band in enumerate(bands):
        # we could do this on the 2d arrays...
        I = np.flatnonzero(T.fracin[:,iband] < 1e-3)
        debug('Zeroing out', len(I), 'objs in', band, 'band with small fracin.')
        if len(I):
            # zero out:
            T.flux[I,iband] = 0.
            T.flux_ivar[I,iband] = 0.
            # zero out fracin itself??

    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='catalog',
                            comment='NOIRLab data product type'))

    if co_sky is not None:
        for band in bands:
            if band in co_sky:
                primhdr.add_record(dict(name='COSKY_%s' % band.upper(),
                                        value=co_sky[band],
                                        comment='Sky level estimated (+subtracted) from coadd'))

    for i,ap in enumerate(apertures_arcsec):
        primhdr.add_record(dict(name='APRAD%i' % i, value=ap,
                                comment='(optical) Aperture radius, in arcsec'))
    if wise_apertures_arcsec is not None:
        for i,ap in enumerate(wise_apertures_arcsec):
            primhdr.add_record(dict(name='WAPRAD%i' % i, value=ap,
                                    comment='(unWISE) Aperture radius, in arcsec'))
    if galex_apertures_arcsec is not None:
        for i,ap in enumerate(galex_apertures_arcsec):
            primhdr.add_record(dict(name='GAPRAD%i' % i, value=ap,
                                    comment='GALEX aperture radius, in arcsec'))

    if WISE is not None:
        copy_wise_into_catalog(T, WISE, WISE_T, primhdr)
        # Done with these now!
        WISE_T = None
        WISE = None

    if GALEX is not None:
        print('runbrick: GALEX table is:')
        GALEX.about()
        galex_cols = GALEX.get_columns()
        for c in ['flux_nuv', 'flux_ivar_nuv', 'flux_fuv', 'flux_ivar_fuv',
                  'apflux_nuv', 'apflux_resid_nuv', 'apflux_ivar_nuv',
                  'apflux_fuv', 'apflux_resid_fuv', 'apflux_ivar_fuv',
                  'psfdepth_nuv', 'psfdepth_fuv', 'nobs_nuv', 'nobs_fuv',
                  'rchisq_nuv', 'rchisq_fuv', 'fracflux_nuv', 'fracflux_fuv']:
            if not c in galex_cols:
                # Fill in missing columns (eg, if no FUV coverage)
                dtype = np.float32
                if c.startswith('nobs'):
                    dtype = np.int16
                T.set(c, np.zeros(len(T), dtype))
            else:
                T.set(c, GALEX.get(c))
        GALEX = None

    set_brick_primary(T, brick)

    H,W = maskbits.shape
    T.maskbits = maskbits[np.clip(T.iby, 0, H-1).astype(int),
                          np.clip(T.ibx, 0, W-1).astype(int)]
    del maskbits

    #if forced_T:
    #    # Add columns from forced photometry.

    # Set Sersic indices for all galaxy types.
    # sigh, bytes vs strings.  In py3, T.type (dtype '|S3') are bytes.
    T.sersic[np.array([t in ['DEV',b'DEV'] for t in T.type])] = 4.0
    T.sersic[np.array([t in ['EXP',b'EXP'] for t in T.type])] = 1.0
    T.sersic[np.array([t in ['REX',b'REX'] for t in T.type])] = 1.0

    T.fitbits = np.zeros(len(T), np.int16)
    T.fitbits[T.forced_pointsource] |= FITBITS['FORCED_POINTSOURCE']
    T.fitbits[T.fit_background]     |= FITBITS['FIT_BACKGROUND']
    T.fitbits[T.hit_r_limit]        |= FITBITS['HIT_RADIUS_LIMIT']
    T.fitbits[T.hit_ser_limit]      |= FITBITS['HIT_SERSIC_LIMIT']
    # WALKER/RUNNER
    moved = np.hypot(T.bx - T.bx0, T.by - T.by0)
    # radii in pixels:
    walk_radius = 1.  / pixscale
    run_radius  = 2.5 / pixscale
    T.fitbits[moved > walk_radius] |= FITBITS['WALKER']
    T.fitbits[moved > run_radius ] |= FITBITS['RUNNER']
    # do we have Gaia?
    if 'pointsource' in T.get_columns():
        T.fitbits[T.pointsource]   |= FITBITS['GAIA_POINTSOURCE']
    T.fitbits[T.iterative]         |= FITBITS['ITERATIVE']

    for col,bit in [('freezeparams',  'FROZEN'),
                    ('isbright',      'BRIGHT'),
                    ('ismedium',      'MEDIUM'),
                    ('isgaia',        'GAIA'),
                    ('istycho',       'TYCHO2'),
                    ('islargegalaxy', 'LARGEGALAXY')]:
        if not col in T.get_columns():
            continue
        T.fitbits[T.get(col)] |= FITBITS[bit]

    with survey.write_output('tractor-intermediate', brick=brickname) as out:
        T[np.argsort(T.objid)].writeto(None, fits_object=out.fits, primheader=primhdr,
                                       extname='CATALOG-INTERMEDIATE')

    # The "format_catalog" code expects all lower-case column names...
    for c in T.columns():
        if c != c.lower():
            T.rename(c, c.lower())

    from legacypipe.format_catalog import format_catalog

    # Reorder by objid
    Iorder = np.argsort(T.objid)
    T.cut(Iorder)
    if forced_T is not None:
        forced_T.cut(Iorder)

    allbands = survey.allbands
    if forced_bands is not None:
        # merge in forced_bands...
        band_order = ['u','g','r','i','z','y']
        allbands = [b for b in band_order if b in survey.allbands or b in forced_bands]
        # append any that aren't in band_order
        for b in survey.allbands + forced_bands:
            if not b in band_order:
                allbands.append(b)
    print('Allbands:', allbands)

    # ASSUME that the UNWISE_COADDS_TIMERESOLVED_DIR ends with "/neo#" and look up the target
    # array size based on that.
    dirname = None
    if unwise_tr_dir is not None:
        dirname = os.path.basename(unwise_tr_dir)
    wise_epochs_map = {
        # DR7/8
        'neo4' : 11,
        # DR9
        'neo6': 15,
        # DR10
        'neo7': 17,
        # DR11?
        'neo11': 25,
    }
    N_wise_epochs = wise_epochs_map.get(dirname)
    if N_wise_epochs is None:
        N_wise_epochs = 17
        warnings.warn('Could not determine array size for WISE time-resolved measurements; defaulting to %i' % N_wise_epochs)
    else:
        print('unWISE time-resolved version: %s - saving %i elements' % (dirname, N_wise_epochs))

    columns,units = format_catalog(T, bands, allbands, release,
                                   N_wise_epochs=N_wise_epochs, motions=gaia_stars, gaia_tagalong=True)

    if forced_bands is not None:
        from legacypipe.survey import clean_band_name
        unitmap = dict([(c,u) for c,u in zip(columns, units)])
        if forced_T is not None:
            # Add forced-photometry columns into the table "T"
            for c in ['brickid', 'objid', 'bx', 'by']:
                forced_T.delete_column(c)
            fc = forced_T.get_columns()
            print('Forced photometry columns:', fc)
            for c in fc:
                T.set(c, forced_T.get(c))
        # Re-align the units with the list of columns.
        units = [unitmap[c] for c in columns]

    # FIXME - maskbits, set i-band bits
    with survey.write_output('tractor', brick=brickname) as out:
        T.writeto(None, columns=columns, units=units, primheader=primhdr,
                  extname='CATALOG', fits_object=out.fits)

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
    return dict(T=T, version_header=version_header)

def set_brick_primary(T, brick):
    if brick.ra1 > brick.ra2: # wrap-around case
        T.brick_primary = (np.logical_or(T.ra >= brick.ra1, T.ra < brick.ra2) *
                           (T.dec >= brick.dec1) * (T.dec < brick.dec2))
    else:
        T.brick_primary = ((T.ra  >= brick.ra1 ) * (T.ra  < brick.ra2) *
                           (T.dec >= brick.dec1) * (T.dec < brick.dec2))

def copy_wise_into_catalog(T, WISE, WISE_T, primhdr):
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

    # Copy columns:
    for c in ['wise_coadd_id', 'wise_x', 'wise_y', 'wise_mask']:
        T.set(c, WISE.get(c))

    def get_or_zero(table, col, dt=np.float32):
        if col in table.get_columns():
            return table.get(col)
        else:
            return np.zeros(len(table), dtype=dt)

    for band in [1,2,3,4]:
        # Apply the Vega-to-AB shift *while* copying columns from
        # WISE to T.
        dm = vega_to_ab['w%i' % band]
        fluxfactor = 10.** (dm / -2.5)
        # fluxes
        c = t = 'flux_w%i' % band
        T.set(t, get_or_zero(WISE, c) * fluxfactor)
        if WISE_T is not None and band <= 2:
            t = 'lc_flux_w%i' % band
            T.set(t, get_or_zero(WISE_T, c) * fluxfactor)
        # ivars
        c = t = 'flux_ivar_w%i' % band
        T.set(t, get_or_zero(WISE, c) / fluxfactor**2)
        if WISE_T is not None and band <= 2:
            t = 'lc_flux_ivar_w%i' % band
            T.set(t, get_or_zero(WISE_T, c) / fluxfactor**2)
        # This is in 1/nanomaggies**2 units also
        c = t = 'psfdepth_w%i' % band
        T.set(t, get_or_zero(WISE, c) / fluxfactor**2)

        if 'apflux_w%i'%band in WISE.get_columns():
            t = c = 'apflux_w%i' % band
            T.set(t, get_or_zero(WISE, c) * fluxfactor)
            t = c = 'apflux_resid_w%i' % band
            T.set(t, get_or_zero(WISE, c) * fluxfactor)
            t = c = 'apflux_ivar_w%i' % band
            T.set(t, get_or_zero(WISE, c) / fluxfactor**2)

    # Copy/rename more columns
    for cin,cout,dt in [('nobs_w%i',        'nobs_w%i'    , np.int16),
                        ('profracflux_w%i', 'fracflux_w%i', np.float32),
                        ('prochi2_w%i',     'rchisq_w%i'  , np.float32)]:
        for band in [1,2,3,4]:
            T.set(cout % band, get_or_zero(WISE, cin % band, dt=dt))

    if WISE_T is not None:
        for cin,cout,dt in [('nobs_w%i',        'lc_nobs_w%i'    , np.int16),
                            ('profracflux_w%i', 'lc_fracflux_w%i', np.float32),
                            ('prochi2_w%i',     'lc_rchisq_w%i'  , np.float32),
                            ('mjd_w%i',         'lc_mjd_w%i'     , np.float64),]:
            for band in [1,2]:
                T.set(cout % band, get_or_zero(WISE_T, cin % band, dt=dt))

def stage_checksum(
        survey=None,
        brickname=None,
        **kwargs):
    '''
    For debugging / special-case processing, write out the current checksums file.
    '''
    # produce per-brick checksum file.
    with survey.write_output('checksums', brick=brickname, hashsum=False) as out:
        f = open(out.fn, 'w')
        # Write our pre-computed hashcodes.
        for fn,hashsum in survey.output_file_hashes.items():
            f.write('%s *%s\n' % (hashsum, fn))
        f.close()

def run_brick(brick, survey, radec=None, pixscale=0.262,
              width=3600, height=3600,
              survey_blob_mask=None,
              release=None,
              zoom=None,
              bands=None,
              forced_bands=None,
              nblobs=None, blob=None, blobxy=None, blobradec=None, blobid=None,
              max_blobsize=None,
              nsigma=6,
              saddle_fraction=0.1,
              saddle_min=2.,
              blob_dilate=None,
              subsky_radii=None,
              reoptimize=False,
              iterative=False,
              iterative_nsigma=False,
              wise=True,
              outliers=True,
              cache_outliers=False,
              remake_outlier_jpegs=False,
              lanczos=True,
              blob_image=False,
              blob_mask=False,
              sky_subtract_large_galaxies=True,
              minimal_coadds=False,
              do_calibs=True,
              old_calibs_ok=False,
              write_metrics=True,
              gaussPsf=False,
              pixPsf=True,
              hybridPsf=True,
              normalizePsf=True,
              apodize=False,
              splinesky=True,
              subsky=True,
              ubercal_sky=False,
              constant_invvar=False,
              tycho_stars=True,
              gaia_stars=True,
              large_galaxies=True,
              large_galaxies_force_pointsource=True,
              fitoncoadds_reweight_ivar=True,
              less_masking=False,
              sub_blobs=False,
              nsatur=None,
              fit_on_coadds=False,
              coadd_tiers=None,
              min_mjd=None, max_mjd=None,
              unwise_coadds=True,
              bail_out=False,
              use_gpu=False,
              gpumode=0,
              ngpu=1,
              gpu_ids=[0],
              threads_per_gpu=16,
              bid=None,
              ceres=True,
              wise_ceres=True,
              galex_ceres=True,
              unwise_dir=None,
              unwise_tr_dir=None,
              unwise_modelsky_dir=None,
              galex=False,
              galex_dir=None,
              save_unwise_psf=False,
              save_galex_psf=False,
              save_coadd_psf=False,
              threads=None,
              plots=False, plots2=False, coadd_bw=False,
              plot_base=None, plot_number=0,
              command_line=None,
              read_parallel=True,
              max_memory_gb=None,
              record_event=None,
    # These are for the 'stages' infrastructure
              pickle_pat='pickles/runbrick-%(brick)s-%%(stage)s.pickle',
              stages=None,
              force=None, forceall=False, write_pickles=True,
              checkpoint_filename=None,
              checkpoint_period=None,
              wise_checkpoint_filename=None,
              wise_checkpoint_period=None,
              prereqs_update=None,
              stagefunc = None,
              pool = None,
              **bonus_kwargs,
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

    - *galex_ceres*: boolean; use Ceres Solver for GALEX forced photometry?

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
    if stages is None:
        stages=['writecat']
    forceStages = [s for s in stages]
    forceStages.extend(force)
    if forceall:
        kwargs.update(forceall=True)

    if bands is None:
        bands = ['g','r','z']

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

    if remake_outlier_jpegs:
        cache_outliers = True

    kwargs.update(ps=ps, nsigma=nsigma, saddle_fraction=saddle_fraction,
                  forced_bands=forced_bands,
                  saddle_min=saddle_min,
                  blob_dilate=blob_dilate,
                  subsky_radii=subsky_radii,
                  survey_blob_mask=survey_blob_mask,
                  sky_subtract_large_galaxies=sky_subtract_large_galaxies,
                  gaussPsf=gaussPsf, pixPsf=pixPsf, hybridPsf=hybridPsf,
                  release=release,
                  normalizePsf=normalizePsf,
                  apodize=apodize,
                  constant_invvar=constant_invvar,
                  splinesky=splinesky,
                  subsky=subsky,
                  ubercal_sky=ubercal_sky,
                  tycho_stars=tycho_stars,
                  gaia_stars=gaia_stars,
                  large_galaxies=large_galaxies,
                  large_galaxies_force_pointsource=large_galaxies_force_pointsource,
                  fitoncoadds_reweight_ivar=fitoncoadds_reweight_ivar,
                  less_masking=less_masking,
                  sub_blobs=sub_blobs,
                  min_mjd=min_mjd, max_mjd=max_mjd,
                  coadd_tiers=coadd_tiers,
                  nsatur=nsatur,
                  reoptimize=reoptimize,
                  iterative=iterative,
                  iterative_nsigma=iterative_nsigma,
                  outliers=outliers,
                  cache_outliers=cache_outliers,
                  remake_outlier_jpegs=remake_outlier_jpegs,
                  use_ceres=ceres,
                  wise_ceres=wise_ceres,
                  galex_ceres=galex_ceres,
                  use_gpu=use_gpu,
                  gpumode=gpumode,
                  ngpu=ngpu,
                  gpu_ids=gpu_ids,
                  threads_per_gpu=threads_per_gpu,
                  bid=bid,
                  unwise_coadds=unwise_coadds,
                  bailout=bail_out,
                  minimal_coadds=minimal_coadds,
                  do_calibs=do_calibs,
                  old_calibs_ok=old_calibs_ok,
                  write_metrics=write_metrics,
                  lanczos=lanczos,
                  unwise_dir=unwise_dir,
                  unwise_tr_dir=unwise_tr_dir,
                  unwise_modelsky_dir=unwise_modelsky_dir,
                  galex=galex,
                  galex_dir=galex_dir,
                  save_unwise_psf=save_unwise_psf,
                  save_galex_psf=save_galex_psf,
                  save_coadd_psf=save_coadd_psf,
                  command_line=command_line,
                  read_parallel=read_parallel,
                  max_memory_gb=max_memory_gb,
                  plots=plots, plots2=plots2, coadd_bw=coadd_bw,
                  force=forceStages, write=write_pickles,
                  record_event=record_event)

    if checkpoint_filename is not None:
        kwargs.update(checkpoint_filename=checkpoint_filename)
        if checkpoint_period is not None:
            kwargs.update(checkpoint_period=checkpoint_period)
    if wise_checkpoint_filename is not None:
        kwargs.update(wise_checkpoint_filename=wise_checkpoint_filename)
        if wise_checkpoint_period is not None:
            kwargs.update(wise_checkpoint_period=wise_checkpoint_period)

    # --- Set up available GPU IDs for the initializer ---
    _available_gpu_ids = []
    if not use_gpu:
        #ngpu defaults to 1 but in CPU only mode we want this to be 0
        ngpu = 0
    _ngpu = ngpu
    _threads_per_gpu = threads_per_gpu
    if gpu_ids == '':
        gpu_ids = []
    else:
        gpu_ids = gpu_ids.split(',')
        try:
            for ig in range(len(gpu_ids)):
                gpu_ids[ig] = int(gpu_ids[ig])
        except Exception as ex:
            print("Exception processing gpu_ids: "+str(ex))
            traceback.print_exc()
            sys.exit(-1)
    if use_gpu and ngpu > 0:
        if len(gpu_ids) == 0:
            # This assumes CUDA_VISIBLE_DEVICES is set appropriately, or you want all detected GPUs
            _available_gpu_ids = list(range(ngpu))
            info(f"Main process: GPUs available for workers: {_available_gpu_ids}")
        elif len(gpu_ids) != ngpu:
            _available_gpu_ids = gpu_ids
            _ngpu = len(gpu_ids)
            info(f"Warning: {gpu_ids=} does not match {ngpu=}.  gpu_ids takes priority and {_ngpu=} will be used.")
            ngpu = _ngpu
            kwargs.update(ngpu=ngpu)
        else:
            _available_gpu_ids = gpu_ids
            info(f"Main process: GPUs available for workers: {_available_gpu_ids}")
        
    if len(_available_gpu_ids) == 0:
        info("Main process: No GPUs configured or CuPy not available; all workers will be CPU-only.")

    manager = multiprocessing.Manager()
    shared_counter = manager.Value('i', 0)
    shared_lock = manager.Lock()
    
    if pool or (threads and threads > 1):
        from astrometry.util.ttime import MemMeas
        from astrometry.util.ttime import MemMeas
        if pool is None:
            # from legacypipe.trackingpool import TrackingPool
            # pool = TrackingPool(threads,
            from astrometry.util.timingpool import TimingPool, TimingPoolMeas
            pool = TimingPool(threads,
                              initializer=runbrick_global_init,
                              initargs=(shared_counter, shared_lock, _available_gpu_ids, _ngpu, _threads_per_gpu, gpumode))
            poolmeas = TimingPoolMeas(pool, pickleTraffic=False)
            StageTime.add_measurement_once(poolmeas)
        StageTime.add_measurement_once(MemMeas)
        mp = multiproc(None, pool=pool)
    else:
        from astrometry.util.ttime import CpuMeas
        from astrometry.util.ttime import MemMeas
        mp = multiproc(init=runbrick_global_init, initargs=(shared_counter, shared_lock, _available_gpu_ids, _ngpu, _threads_per_gpu, gpumode))
        StageTime.add_measurement_once(CpuMeas)
        StageTime.add_measurement_once(MemMeas)
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

    # This exists for folks extending the code (eg, adding new stages that take
    # additional args)
    kwargs.update(bonus_kwargs)

    pickle_pat = pickle_pat % dict(brick=brick)

    prereqs = {
        'tims':None,
        'refs': 'tims',
        'outliers': 'refs',
        'halos': 'outliers',
        'srcs': 'halos',

        # fitblobs: see below
        'blobmask': 'halos',

        'coadds': 'fitblobs',

        # wise_forced: see below

        'fitplots': 'fitblobs',
        'psfplots': 'tims',
        'initplots': 'srcs',

        }

    if 'image_coadds' in stages:
        if blob_mask:
            prereqs.update({
                'image_coadds':'blobmask',
                'srcs': 'image_coadds',
                'fitblobs':'srcs',
            })
        elif blob_image:
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

    # not sure how to set up the prereqs here. --galex could always require --wise?
    if wise:
        if galex:
            prereqs.update({
                'wise_forced': 'coadds',
                'galex_forced': 'wise_forced',
                'writecat': 'galex_forced',
                })
        else:
            prereqs.update({
                'wise_forced': 'coadds',
                'writecat': 'wise_forced',
                })
    else:
        if galex:
            prereqs.update({
                'galex_forced': 'coadds',
                'writecat': 'galex_forced',
                })
        else:
            prereqs.update({
                'writecat': 'coadds',
                })

    if fit_on_coadds:
        prereqs.update({
            'fit_on_coadds': 'halos',
            'srcs': 'fit_on_coadds',
            'image_coadds': 'fit_on_coadds',
        })
        if blob_image:
            prereqs.update({'image_coadds':'srcs'})

    if forced_bands:
        prereqs.update({'forced_phot': prereqs['writecat'],
                        'writecat': 'forced_phot'})

    # HACK -- set the prereq to the stage after which you'd like to write out checksums.
    prereqs.update({'checksum': 'outliers'})

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
            picsurvey.allbands = survey.allbands
            picsurvey.coadd_bw = survey.coadd_bw
            # Also pick up maskbits updates, eg from --run
            picsurvey.maskbits_descriptions = survey.maskbits_descriptions
            picsurvey.maskbits = survey.maskbits

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
    try:
        for stage in stages:
            R = runstage(stage, pickle_pat, mystagefunc, prereqs=prereqs,
                         initial_args=initargs, **kwargs)
        info('All done:', StageTime()-t0)
    except Exception as e:
        print('runstage ... caught exception', e)
        if pool is not None:
            print('pool.terminate()...')
            pool.terminate()
        raise e
    finally:
        print('runstage ... finally clause')
        if pool is not None:
            print('pool.close()...')
            pool.close()
            print('pool.join()...')
            pool.join()
    print('run_brick returning...')

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
    @classmethod
    def add_measurement_once(cls, m):
        if not m in cls.measurements:
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

    parser.add_argument(
        '--wise-checkpoint', dest='wise_checkpoint_filename', default=None,
        help='Write WISE to checkpoint file?')
    parser.add_argument(
        '--wise-checkpoint-period', type=int, default=None,
        help='Period for writing WISE checkpoint files, in seconds; default 600')

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
    parser.add_argument('--prime-cache', default=False, action='store_true',
                        help='Copy image (ooi, ood, oow) files to --cache-dir before starting.')

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

    parser.add_argument('--use-gpu', default=False, action='store_true')
    parser.add_argument('--gpumode', default=0, type=int, help='Mode for GPU')
    parser.add_argument('--ngpu', default=1, type=int, help='Number of GPUs')
    parser.add_argument('--gpu-ids', default='', type=str, help='Comma separated list of GPU ids e.g. 0,1,2,3')
    parser.add_argument('--threads-per-gpu', default=16, type=int, help='Max threads per GPU - CPU will fulfill remaining threads')
    parser.add_argument('--bid', default=None, type=int, help='blob id')

    parser.add_argument('--ceres', default=False, action='store_true',
                        help='Use Ceres Solver for all optimization?')

    parser.add_argument('--no-wise-ceres', dest='wise_ceres', default=True,
                        action='store_false',
                        help='Do not use Ceres Solver for unWISE forced phot')

    parser.add_argument('--no-galex-ceres', dest='galex_ceres', default=True,
                        action='store_false',
                        help='Do not use Ceres Solver for GALEX forced phot')

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
    parser.add_argument('--blob-dilate', type=int, default=None,
                        help='How many pixels to dilate detection pixels (default: 8)')

    parser.add_argument(
        '--reoptimize', action='store_true', default=False,
        help='Do a second round of model fitting after all model selections')

    parser.add_argument(
        '--no-iterative', dest='iterative', action='store_false', default=True,
        help='Turn off iterative source detection?')
    parser.add_argument('--iterative-nsigma', type=float, default=None,
                        help='Set N sigma source detection thresh, for iterative detection')

    parser.add_argument('--no-wise', dest='wise', default=True,
                        action='store_false',
                        help='Skip unWISE forced photometry')

    parser.add_argument(
        '--unwise-dir', default=None,
        help='Base directory for unWISE coadds; may be a colon-separated list')
    parser.add_argument(
        '--unwise-tr-dir', default=None,
        help='Base directory for unWISE time-resolved coadds; may be a colon-separated list')

    parser.add_argument('--galex', dest='galex', default=False,
                        action='store_true',
                        help='Perform GALEX forced photometry')
    parser.add_argument(
        '--galex-dir', default=None,
        help='Base directory for GALEX coadds')

    parser.add_argument('--blob-image', action='store_true', default=False,
                        help='Create "imageblob" image?')
    parser.add_argument('--blob-mask', action='store_true', default=False,
                        help='With --stage image_coadds, also run the "blobmask" stage?')
    parser.add_argument('--minimal-coadds', action='store_true', default=False,
                        help='Only create image and invvar coadds in image_coadds stage')

    parser.add_argument('--sky-no-subtract-large-galaxies',
                        dest='sky_subtract_large_galaxies',
                        default=True, action='store_false',
                        help='For sky calibs: do not subtract large galaxies first')

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

    parser.add_argument('--forced-bands', default=None,
                        help='Set the list of bands (filters) that will get forced photometry')

    parser.add_argument('--no-tycho', dest='tycho_stars', default=True,
                        action='store_false',
                        help="Don't use Tycho-2 sources as fixed stars")

    parser.add_argument('--no-gaia', dest='gaia_stars', default=True,
                        action='store_false',
                        help="Don't use Gaia sources as fixed stars")

    parser.add_argument('--no-large-galaxies', dest='large_galaxies', default=True,
                        action='store_false', help="Don't seed (or mask in and around) large galaxies.")
    parser.add_argument('--min-mjd', type=float,
                        help='Only keep images taken after the given MJD')
    parser.add_argument('--max-mjd', type=float,
                        help='Only keep images taken before the given MJD')

    parser.add_argument('--no-splinesky', dest='splinesky', default=True,
                        action='store_false', help='Use constant sky rather than spline.')
    parser.add_argument('--no-subsky', dest='subsky', default=True,
                        action='store_false', help='Do not subtract the sky background.')
    parser.add_argument('--no-unwise-coadds', dest='unwise_coadds', default=True,
                        action='store_false', help='Turn off writing FITS and JPEG unWISE coadds?')
    parser.add_argument('--save-unwise-psf', default=False, action='store_true',
                        help='Save unWISE PSF models?')
    parser.add_argument('--save-galex-psf', default=False, action='store_true',
                        help='Save GALEX PSF models?')
    parser.add_argument('--save-coadd-psf', default=False, action='store_true',
                        help='Save optical coadded PSF models?')
    parser.add_argument('--no-outliers', dest='outliers', default=True,
                        action='store_false', help='Do not compute or apply outlier masks')
    parser.add_argument('--cache-outliers', default=False,
                        action='store_true', help='Use outlier-mask file if it exists?')
    parser.add_argument('--remake-outlier-jpegs', default=False,
                        action='store_true', help='Re-create outlier jpeg files (implies --cache-outliers)')
    parser.add_argument('--bail-out', default=False, action='store_true',
                        help='Bail out of "fitblobs" processing, writing all blobs from the checkpoint and skipping any remaining ones.')

    parser.add_argument('--sub-blobs', default=False, action='store_true',
                        help='Split large blobs into sub-blobs that can be processed in parallel.')

    parser.add_argument('--fit-on-coadds', default=False, action='store_true',
                        help='Fit to coadds rather than individual CCDs (e.g., large galaxies).')
    parser.add_argument('--coadd-tiers', default=None, type=int,
                        help='Split images into this many tiers of coadds (per band) by FWHW')

    parser.add_argument('--nsatur', default=None, type=int,
                        help='Demand that >= nsatur images per band are saturated before using saturated logic (eg, 2).')
    parser.add_argument('--no-ivar-reweighting', dest='fitoncoadds_reweight_ivar',
                        default=True, action='store_false',
                        help='Reweight the inverse variance when fitting on coadds.')
    parser.add_argument('--no-galaxy-forcepsf', dest='large_galaxies_force_pointsource',
                        default=True, action='store_false',
                        help='Do not force PSFs within galaxy mask.')
    parser.add_argument('--less-masking', default=False, action='store_true',
                        help='Turn off background fitting within MEDIUM mask.')

    parser.add_argument('--ubercal-sky', dest='ubercal_sky', default=False,
                        action='store_true', help='Use the ubercal sky-subtraction (only used with --fit-on-coadds and --no-subsky).')
    parser.add_argument('--subsky-radii', type=float, nargs='*', default=None,
                        help="""Sky-subtraction radii: rin, rout [arcsec] (only used with --fit-on-coadds and --no-subsky).
                        Image pixels r<rmask are fully masked and the pedestal sky background is estimated from an annulus
                        rin<r<rout on each CCD centered on the targetwcs.crval coordinates.""")
    parser.add_argument('--read-serial', dest='read_parallel', default=True,
                        action='store_false', help='Read images in series, not in parallel?')
    parser.add_argument('--max-memory-gb', type=float, default=None,
                        help='Maximum (estimated) memory to allow for tim pixels, in GB')
    parser.add_argument('--rgb-stretch', type=float, help='Stretch RGB jpeg plots by this factor.')
    return parser

def get_runbrick_kwargs(survey=None,
                        brick=None,
                        radec=None,
                        run=None,
                        survey_dir=None,
                        output_dir=None,
                        cache_dir=None,
                        prime_cache=False,
                        check_done=False,
                        skip=False,
                        skip_coadd=False,
                        stage=None,
                        unwise_dir=None,
                        unwise_tr_dir=None,
                        unwise_modelsky_dir=None,
                        galex_dir=None,
                        write_stage=None,
                        write=True,
                        gpsf=False,
                        bands=None,
                        allbands=None,
                        forced_bands=None,
                        coadd_bw=None,
                        **opt):
    if stage is None:
        stage = []
    if brick is not None and radec is not None:
        print('Only ONE of --brick and --radec may be specified.')
        return None, -1
    opt.update(radec=radec)

    if bands is None:
        bands = ['g','r','z']
    else:
        bands = bands.split(',')
    opt.update(bands=bands, coadd_bw=coadd_bw)

    if forced_bands is not None:
        forced_bands = forced_bands.split(',')
        opt.update(forced_bands=forced_bands)

    if allbands is None:
        allbands = bands
        # # Make sure at least 'bands' are in allbands.
        # allbands = ['g','r','z']
        # for b in bands:
        #     if not b in allbands:
        #         allbands.append(b)

    if survey is None:
        from legacypipe.runs import get_survey
        survey = get_survey(run,
                            survey_dir=survey_dir,
                            output_dir=output_dir,
                            cache_dir=cache_dir,
                            prime_cache=prime_cache,
                            allbands=allbands,
                            coadd_bw=coadd_bw)
        info(survey)

    survey.update_maskbits_bands(bands + (forced_bands or []))

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
            raise RuntimeError('The directory specified in $UNWISE_MODEL_SKY_DIR (%s) does not exist!' % unwise_modelsky_dir)
    if galex_dir is None:
        galex_dir = os.environ.get('GALEX_DIR', None)
    opt.update(unwise_dir=unwise_dir, unwise_tr_dir=unwise_tr_dir,
               unwise_modelsky_dir=unwise_modelsky_dir, galex_dir=galex_dir)

    # list of strings if -w / --write-stage is given; False if
    # --no-write given; True by default.
    if write_stage is not None:
        write_pickles = write_stage
    else:
        write_pickles = write
    opt.update(write_pickles=write_pickles)

    opt.update(gaussPsf=gpsf,
               pixPsf=not gpsf)

    return survey, opt

def main(args=None):
    import datetime
    from legacypipe.survey import get_git_version
    t = time.time()

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
    rgb_stretch = optdict.pop('rgb_stretch', None)

    survey, kwargs = get_runbrick_kwargs(**optdict)
    if kwargs in [-1, 0]:
        return kwargs
    kwargs.update(command_line=' '.join(sys.argv))

    # We really do not need to know about leap seconds
    import astropy.utils.iers
    c = astropy.utils.iers.Conf
    c.auto_download = False

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
            ps_queue.append((ps_t0, 'start'))
        ps_thread = threading.Thread(
            target=run_ps_thread,
            args=(os.getpid(), os.getppid(), ps_file, ps_shutdown, ps_queue),
            name='run_ps')
        ps_thread.daemon = True
        info('Starting thread to run "ps"')
        ps_thread.start()

    if rgb_stretch is not None:
        import legacypipe.survey
        legacypipe.survey.rgb_stretch_factor = rgb_stretch

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
        info('Attempting to join the ps thread...')
        ps_thread.join(1.0)
        if ps_thread.is_alive():
            info('ps thread is still alive.')

    print ("Total runtime:", time.time()-t)
    return rtn

if __name__ == '__main__':
    from astrometry.util.ttime import MemMeas
    Time.add_measurement(MemMeas)
    sys.exit(main())

# Test bricks & areas

# A single, fairly bright star
# python -u legacypipe/runbrick.py -b 1498p017 -P 'pickles/runbrick-z-%(brick)s-%%(stage)s.pickle' --zoom 1900 2000 2700 2800
# python -u legacypipe/runbrick.py -b 0001p000 -P 'pickles/runbrick-z-%(brick)s-%%(stage)s.pickle' --zoom 80 380 2970 3270
