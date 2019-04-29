'''
Main "pipeline" script for the Legacy Survey (DECaLS, MzLS, BASS)
data reductions.

For calling from other scripts, see:

- :py:func:`run_brick`

Or for much more fine-grained control, see the individual stages:

- :py:func:`stage_tims`
- :py:func:`stage_outliers`
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

# RGB image args used in the tile viewer:
rgbkwargs = dict(mnmx=(-3,300.), arcsinh=1.)
rgbkwargs_resid = dict(mnmx=(-5,5))

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
               depth_cut=None,
               read_image_pixels = True,
               min_mjd=None, max_mjd=None,
               gaia_stars=False,
               mp=None,
               record_event=None,
               unwise_dir=None,
               unwise_tr_dir=None,
               unwise_modelsky_dir=None,
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

    - *splinesky*: boolean.  Use SplineSky model, rather than ConstantSky?
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
        brick.ra1,nil  = targetwcs.pixelxy2radec(W, H/2)
        brick.ra2,nil  = targetwcs.pixelxy2radec(1, H/2)
        nil, brick.dec1 = targetwcs.pixelxy2radec(W/2, 1)
        nil, brick.dec2 = targetwcs.pixelxy2radec(W/2, H)

    # Create FITS header with version strings
    gitver = get_git_version()

    version_header = get_version_header(program_name, survey.survey_dir, release,
                                        git_version=gitver)

    deps = get_dependency_versions(unwise_dir, unwise_tr_dir, unwise_modelsky_dir)
    for name,value,comment in deps:
        version_header.add_record(dict(name=name, value=value, comment=comment))

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
    #version_header.add_record(dict(name='BRICKRA' , value=brick.ra,comment='[deg] Brick center'))
    #version_header.add_record(dict(name='BRICKDEC', value=brick.dec,comment='[deg] Brick center'))

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

    if depth_cut:
        # If we have many images, greedily select images until we have
        # reached our target depth
        debug('Cutting to CCDs required to hit our depth targets')
        # Previously, depth_cut was a boolean; I turned it into a float margin;
        # be a little backwards-compatible.
        margin = float(depth_cut)
        kwa = {}
        if margin >= 0:
            kwa.update(margin=depth_cut)
        keep_ccds,_ = make_depth_cut(survey, ccds, bands, targetrd, brick, W, H, pixscale,
                                     plots, ps, splinesky, gaussPsf, pixPsf, normalizePsf,
                                     do_calibs, gitver, targetwcs, old_calibs_ok, **kwa)
        ccds.cut(np.array(keep_ccds))
        debug('Cut to', len(ccds), 'CCDs required to reach depth targets')

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
        kwa = dict(git_version=gitver, survey=survey, old_calibs_ok=old_calibs_ok)
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
                                splinesky=splinesky,
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
        for cal,ver in [('sky', tim.skyver), ('wcs', tim.wcsver),
                        ('psf', tim.psfver)]:
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
    ok,x,y = targetwcs.radec2pixelxy(rd[:,:,0], rd[:,:,1])
    ccds.brick_x0 = np.floor(np.min(x, axis=1)).astype(np.int16)
    ccds.brick_x1 = np.ceil (np.max(x, axis=1)).astype(np.int16)
    ccds.brick_y0 = np.floor(np.min(y, axis=1)).astype(np.int16)
    ccds.brick_y1 = np.ceil (np.max(y, axis=1)).astype(np.int16)
    ccds.psfnorm = np.array([tim.psfnorm for tim in tims])
    ccds.galnorm = np.array([tim.galnorm for tim in tims])
    ccds.propid = np.array([tim.propid for tim in tims])
    ccds.plver  = np.array([tim.plver for tim in tims])
    ccds.skyver = np.array([tim.skyver[0] for tim in tims])
    ccds.wcsver = np.array([tim.wcsver[0] for tim in tims])
    ccds.psfver = np.array([tim.psfver[0] for tim in tims])
    ccds.skyplver = np.array([tim.skyver[1] for tim in tims])
    ccds.wcsplver = np.array([tim.wcsver[1] for tim in tims])
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

def make_depth_cut(survey, ccds, bands, targetrd, brick, W, H, pixscale,
                   plots, ps, splinesky, gaussPsf, pixPsf, normalizePsf, do_calibs,
                   gitver, targetwcs, old_calibs_ok, get_depth_maps=False, margin=0.5,
                   use_approx_wcs=False):
    # Add some margin to our DESI depth requirements
    target_depth_map = dict(g=24.0 + margin, r=23.4 + margin, z=22.5 + margin)

    # List extra (redundant) target percentiles so that increasing the depth at
    # any of these percentiles causes the image to be kept.
    target_percentiles = np.array(list(range(2, 10)) +
                                  list(range(10, 30, 5)) +
                                  list(range(30, 101, 10)))
    target_ddepths = np.zeros(len(target_percentiles), np.float32)
    target_ddepths[target_percentiles < 10] = -0.3
    target_ddepths[target_percentiles <  5] = -0.6
    #print('Target percentiles:', target_percentiles)
    #print('Target ddepths:', target_ddepths)

    cH,cW = H//10, W//10
    coarsewcs = targetwcs.scale(0.1)
    coarsewcs.imagew = cW
    coarsewcs.imageh = cH

    # Unique pixels in this brick (U: cH x cW boolean)
    U = find_unique_pixels(coarsewcs, cW, cH, None,
                           brick.ra1, brick.ra2, brick.dec1, brick.dec2)
    pixscale = 3600. * np.sqrt(np.abs(ccds.cd1_1*ccds.cd2_2 - ccds.cd1_2*ccds.cd2_1))
    seeing = ccds.fwhm * pixscale

    # Compute the rectangle in *coarsewcs* covered by each CCD
    slices = []
    overlapping_ccds = np.zeros(len(ccds), bool)
    for i,ccd in enumerate(ccds):
        wcs = survey.get_approx_wcs(ccd)
        hh,ww = wcs.shape
        rr,dd = wcs.pixelxy2radec([1,ww,ww,1], [1,1,hh,hh])
        ok,xx,yy = coarsewcs.radec2pixelxy(rr, dd)
        y0 = int(np.round(np.clip(yy.min(), 0, cH-1)))
        y1 = int(np.round(np.clip(yy.max(), 0, cH-1)))
        x0 = int(np.round(np.clip(xx.min(), 0, cW-1)))
        x1 = int(np.round(np.clip(xx.max(), 0, cW-1)))
        if y0 == y1 or x0 == x1:
            slices.append(None)
            continue
        # Check whether this CCD overlaps the unique area of this brick...
        if not np.any(U[y0:y1+1, x0:x1+1]):
            info('No overlap with unique area for CCD', ccd.expnum, ccd.ccdname)
            slices.append(None)
            continue
        overlapping_ccds[i] = True
        slices.append((slice(y0, y1+1), slice(x0, x1+1)))

    keep_ccds = np.zeros(len(ccds), bool)
    depthmaps = []

    for band in bands:
        # scalar
        target_depth = target_depth_map[band]
        # vector
        target_depths = target_depth + target_ddepths

        depthiv = np.zeros((cH,cW), np.float32)
        depthmap = np.zeros_like(depthiv)
        depthvalue = np.zeros_like(depthiv)
        last_pcts = np.zeros_like(target_depths)
        # indices of CCDs we still want to look at in the current band
        b_inds = np.where(ccds.filter == band)[0]
        info(len(b_inds), 'CCDs in', band, 'band')
        if len(b_inds) == 0:
            continue
        b_inds = np.array([i for i in b_inds if slices[i] is not None])
        info(len(b_inds), 'CCDs in', band, 'band overlap target')
        if len(b_inds) == 0:
            continue
        # CCDs that we will try before searching for good ones -- CCDs
        # from the same exposure number as CCDs we have chosen to
        # take.
        try_ccds = set()

        # Try DECaLS data first!
        Idecals = np.where(ccds.propid[b_inds] == '2014B-0404')[0]
        if len(Idecals):
            try_ccds.update(b_inds[Idecals])
        debug('Added', len(try_ccds), 'DECaLS CCDs to try-list')

        plot_vals = []

        if plots:
            plt.clf()
            for i in b_inds:
                sy,sx = slices[i]
                x0,x1 = sx.start, sx.stop
                y0,y1 = sy.start, sy.stop
                plt.plot([x0,x0,x1,x1,x0], [y0,y1,y1,y0,y0], 'b-', alpha=0.5)
            plt.title('CCDs overlapping brick: %i in %s band' % (len(b_inds), band))
            ps.savefig()

            nccds = np.zeros((cH,cW), np.int16)
            plt.clf()
            for i in b_inds:
                nccds[slices[i]] += 1
            plt.imshow(nccds, interpolation='nearest', origin='lower', vmin=0)
            plt.colorbar()
            plt.title('CCDs overlapping brick: %i in %s band (%i / %i / %i)' %
                      (len(b_inds), band, nccds.min(), np.median(nccds), nccds.max()))

            ps.savefig()
            #continue

        while len(b_inds):
            if len(try_ccds) == 0:
                # Choose the next CCD to look at in this band.

                # A rough point-source depth proxy would be:
                # metric = np.sqrt(ccds.extime[b_inds]) / seeing[b_inds]
                # If we want to put more weight on choosing good-seeing images, we could do:
                #metric = np.sqrt(ccds.exptime[b_inds]) / seeing[b_inds]**2

                # depth would be ~ 1 / (sig1 * seeing); we privilege good seeing here.
                metric = 1. / (ccds.sig1[b_inds] * seeing[b_inds]**2)

                # This metric is *BIG* for *GOOD* ccds!

                # Here, we try explicitly to include CCDs that cover
                # pixels that are still shallow by the largest amount
                # for the largest number of percentiles of interest;
                # note that pixels with no coverage get depth 0, so
                # score high in this metric.
                #
                # The value is the depth still required to hit the
                # target, summed over percentiles of interest
                # (for pixels unique to this brick)
                depthvalue[:,:] = 0.
                active = (last_pcts < target_depths)
                for d in target_depths[active]:
                    depthvalue += U * np.maximum(0, d - depthmap)
                ccdvalue = np.zeros(len(b_inds), np.float32)
                for j,i in enumerate(b_inds):
                    #ccdvalue[j] = np.sum(depthvalue[slices[i]])
                    # mean -- we want the most bang for the buck per pixel?
                    ccdvalue[j] = np.mean(depthvalue[slices[i]])
                metric *= ccdvalue

                # *ibest* is an index into b_inds
                ibest = np.argmax(metric)
                # *iccd* is an index into ccds.
                iccd = b_inds[ibest]
                ccd = ccds[iccd]
                debug('Chose best CCD: seeing', seeing[iccd], 'exptime', ccds.exptime[iccd], 'with value', ccdvalue[ibest])

            else:
                iccd = try_ccds.pop()
                ccd = ccds[iccd]
                debug('Popping CCD from use_ccds list')

            # remove *iccd* from b_inds
            b_inds = b_inds[b_inds != iccd]

            im = survey.get_image_object(ccd)
            debug('Band', im.band, 'expnum', im.expnum, 'exptime', im.exptime, 'seeing', im.fwhm*im.pixscale, 'arcsec, propid', im.propid)

            im.check_for_cached_files(survey)
            debug(im)

            if do_calibs:
                kwa = dict(git_version=gitver, old_calibs_ok=old_calibs_ok)
                if gaussPsf:
                    kwa.update(psfex=False)
                if splinesky:
                    kwa.update(splinesky=True)
                im.run_calibs(**kwa)

            if use_approx_wcs:
                debug('Using approximate (TAN) WCS')
                wcs = survey.get_approx_wcs(ccd)
            else:
                debug('Reading WCS from', im.imgfn, 'HDU', im.hdu)
                wcs = im.get_wcs()

            x0,x1,y0,y1,slc = im.get_image_extent(wcs=wcs, radecpoly=targetrd)
            if x0==x1 or y0==y1:
                debug('No actual overlap')
                continue
            wcs = wcs.get_subimage(int(x0), int(y0), int(x1-x0), int(y1-y0))

            if 'galnorm_mean' in ccds.get_columns():
                galnorm = ccd.galnorm_mean
                debug('Using galnorm_mean from CCDs table:', galnorm)
            else:
                psf = im.read_psf_model(x0, y0, gaussPsf=gaussPsf, pixPsf=pixPsf,
                                        normalizePsf=normalizePsf)
                psf = psf.constantPsfAt((x1-x0)//2, (y1-y0)//2)
                # create a fake tim to compute galnorm
                from tractor import PixPos, Flux, ModelMask, Image, NullWCS
                from legacypipe.survey import SimpleGalaxy

                h,w = 50,50
                gal = SimpleGalaxy(PixPos(w//2,h//2), Flux(1.))
                tim = Image(data=np.zeros((h,w), np.float32),
                            psf=psf, wcs=NullWCS(pixscale=im.pixscale))
                mm = ModelMask(0, 0, w, h)
                galmod = gal.getModelPatch(tim, modelMask=mm).patch
                galmod = np.maximum(0, galmod)
                galmod /= galmod.sum()
                galnorm = np.sqrt(np.sum(galmod**2))
            detiv = 1. / (im.sig1 / galnorm)**2
            galdepth = -2.5 * (np.log10(5. * im.sig1 / galnorm) - 9.)
            debug('Galnorm:', galnorm, 'sig1:', im.sig1, 'galdepth', galdepth)

            # Add this image the the depth map...
            from astrometry.util.resample import resample_with_wcs, OverlapError
            try:
                Yo,Xo,_,_,_ = resample_with_wcs(coarsewcs, wcs)
                debug(len(Yo), 'of', (cW*cH), 'pixels covered by this image')
            except OverlapError:
                debug('No overlap')
                continue
            depthiv[Yo,Xo] += detiv

            # compute the new depth map & percentiles (including the proposed new CCD)
            depthmap[:,:] = 0.
            depthmap[depthiv > 0] = 22.5 - 2.5*np.log10(5./np.sqrt(depthiv[depthiv > 0]))
            depthpcts = np.percentile(depthmap[U], target_percentiles)

            for i,(p,d,t) in enumerate(zip(target_percentiles, depthpcts, target_depths)):
                info('  pct % 3i, prev %5.2f -> %5.2f vs target %5.2f %s' % (p, last_pcts[i], d, t, ('ok' if d >= t else '')))

            keep = False
            # Did we increase the depth of any target percentile that did not already exceed its target depth?
            if np.any((depthpcts > last_pcts) * (last_pcts < target_depths)):
                keep = True

            # Add any other CCDs from this same expnum to the try_ccds list.
            # (before making the plot)
            I = np.where(ccd.expnum == ccds.expnum[b_inds])[0]
            try_ccds.update(b_inds[I])
            debug('Adding', len(I), 'CCDs with the same expnum to try_ccds list')

            if plots:
                cc = '1' if keep else '0'
                xx = [Xo.min(), Xo.min(), Xo.max(), Xo.max(), Xo.min()]
                yy = [Yo.min(), Yo.max(), Yo.max(), Yo.min(), Yo.min()]
                plot_vals.append(((xx,yy,cc),(last_pcts,depthpcts,keep),im.ccdname))

            if plots and (
                (len(try_ccds) == 0) or np.all(depthpcts >= target_depths)):
                plt.clf()

                plt.subplot2grid((2,2),(0,0))
                plt.imshow(depthvalue, interpolation='nearest', origin='lower',
                           vmin=0)
                plt.xticks([]); plt.yticks([])
                plt.colorbar()
                plt.title('heuristic value')

                plt.subplot2grid((2,2),(0,1))
                plt.imshow(depthmap, interpolation='nearest', origin='lower',
                           vmin=target_depth - 2, vmax=target_depth + 0.5)
                ax = plt.axis()
                for (xx,yy,cc) in [p[0] for p in plot_vals]:
                    plt.plot(xx,yy, '-', color=cc, lw=3)
                plt.axis(ax)
                plt.xticks([]); plt.yticks([])
                plt.colorbar()
                plt.title('depth map')

                plt.subplot2grid((2,2),(1,0), colspan=2)
                ax = plt.gca()
                plt.plot(target_percentiles, target_depths, 'ro', label='Target')
                plt.plot(target_percentiles, target_depths, 'r-')
                for (lp,dp,k) in [p[1] for p in plot_vals]:
                    plt.plot(target_percentiles, lp, 'k-',
                             label='Previous percentiles')
                for (lp,dp,k) in [p[1] for p in plot_vals]:
                    cc = 'b' if k else 'r'
                    plt.plot(target_percentiles, dp, '-', color=cc,
                             label='Depth percentiles')
                ccdnames = ','.join([p[2] for p in plot_vals])
                plot_vals = []

                plt.ylim(target_depth - 2, target_depth + 0.5)
                plt.xscale('log')
                plt.xlabel('Percentile')
                plt.ylabel('Depth')
                plt.title('depth percentiles')
                plt.suptitle('%s %i-%s, exptime %.0f, seeing %.2f, band %s' %
                             (im.camera, im.expnum, ccdnames, im.exptime,
                              im.pixscale * im.fwhm, band))
                ps.savefig()

            if keep:
                info('Keeping this exposure')
            else:
                info('Not keeping this exposure')
                depthiv[Yo,Xo] -= detiv
                continue

            keep_ccds[iccd] = True
            last_pcts = depthpcts

            if np.all(depthpcts >= target_depths):
                info('Reached all target depth percentiles for band', band)
                break

        if get_depth_maps:
            if np.any(depthiv > 0):
                depthmap[:,:] = 0.
                depthmap[depthiv > 0] = 22.5 -2.5*np.log10(5./np.sqrt(depthiv[depthiv > 0]))
                depthmap[np.logical_not(U)] = np.nan
                depthmaps.append((band, depthmap.copy()))

        if plots:
            I = np.where(ccds.filter == band)[0]
            plt.clf()
            plt.plot(seeing[I], ccds.exptime[I], 'k.')
            # which CCDs from this band are we keeping?
            kept, = np.nonzero(keep_ccds)
            if len(kept):
                kept = kept[ccds.filter[kept] == band]
                plt.plot(seeing[kept], ccds.exptime[kept], 'ro')
            plt.xlabel('Seeing (arcsec)')
            plt.ylabel('Exptime (sec)')
            plt.title('CCDs kept for band %s' % band)
            plt.ylim(0, np.max(ccds.exptime[I]) * 1.1)
            ps.savefig()

    if get_depth_maps:
        return (keep_ccds, overlapping_ccds, depthmaps)
    return keep_ccds, overlapping_ccds

def stage_outliers(tims=None, targetwcs=None, W=None, H=None, bands=None,
                    mp=None, nsigma=None, plots=None, ps=None, record_event=None,
                    survey=None, brickname=None, version_header=None,
                    gaia_stars=False,
                    **kwargs):
    '''
    This pipeline stage tries to detect artifacts in the individual
    exposures, by blurring all images in the same band to the same PSF size,
    then searching for outliers.
    '''
    from legacypipe.outliers import patch_from_coadd, mask_outlier_pixels, read_outlier_mask_file

    record_event and record_event('stage_outliers: starting')

    # Check for existing MEF containing masks for all the chips we need.
    if not read_outlier_mask_file(survey, tims, brickname):
        from astrometry.util.file import trymakedirs

        # Make before-n-after plots (before)
        C = make_coadds(tims, bands, targetwcs, mp=mp, sbscale=False)
        outdir = os.path.join(survey.output_dir, 'metrics', brickname[:3])
        trymakedirs(outdir)
        outfn = os.path.join(outdir, 'outliers-pre-%s.jpg' % brickname)
        imsave_jpeg(outfn, get_rgb(C.coimgs, bands), origin='lower')

        # Patch individual-CCD masked pixels from a coadd
        patch_from_coadd(C.coimgs, targetwcs, bands, tims, mp=mp)
        del C

        make_badcoadds = True
        badcoadds = mask_outlier_pixels(survey, tims, bands, targetwcs, brickname, version_header,
                                        mp=mp, plots=plots, ps=ps, make_badcoadds=make_badcoadds,
            gaia_stars=gaia_stars)

        # Make before-n-after plots (after)
        C = make_coadds(tims, bands, targetwcs, mp=mp, sbscale=False)
        outfn = os.path.join(outdir, 'outliers-post-%s.jpg' % brickname)
        imsave_jpeg(outfn, get_rgb(C.coimgs, bands), origin='lower')
        outfn = os.path.join(outdir, 'outliers-masked-%s.jpg' % brickname)
        imsave_jpeg(outfn, get_rgb(badcoadds, bands), origin='lower')

    return dict(tims=tims)

def stage_image_coadds(survey=None, targetwcs=None, bands=None, tims=None,
                       brickname=None, version_header=None,
                       plots=False, ps=None, coadd_bw=False, W=None, H=None,
                       brick=None, blobs=None, lanczos=True, ccds=None,
                       rgb_kwargs=None,
                       write_metrics=True,
                       mp=None, record_event=None,
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
                                   targetwcs),
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

    if rgb_kwargs is None:
        rgb_kwargs = {}

    coadd_list= [('image', C.coimgs, rgb_kwargs)]
    if hasattr(tims[0], 'sims_image'):
        coadd_list.append(('simscoadd', sims_coadd, rgb_kwargs))

    for name,ims,rgbkw in coadd_list:
        #rgb = get_rgb(ims, bands, **rgbkw)
        # kwargs used for the SDSS layer in the viewer.
        #sdss_map_kwargs = dict(scales={'g':(2,2.5), 'r':(1,1.5), 'i':(0,1.0),
        #                               'z':(0,0.4)}, m=0.02)
        #rgb = sdss_rgb(ims, bands, **sdss_map_kwargs)
        rgb = sdss_rgb(ims, bands, **rgbkw)

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

def sdss_rgb(imgs, bands, scales=None, m=0.03, Q=20):
    rgbscales=dict(g=(2, 6.0),
                   r=(1, 3.4),
                   i=(0, 3.0),
                   z=(0, 2.2))
    # rgbscales = {'u': 1.5, #1.0,
    #              'g': 2.5,
    #              'r': 1.5,
    #              'i': 1.0,
    #              'z': 0.4, #0.3
    #              }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = np.clip((img * scale + m) * fI / I, 0, 1)
    return rgb

def stage_srcs(targetrd=None, pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               brickname=None,
               version_header=None,
               mp=None, nsigma=None,
               survey=None, brick=None,
               tycho_stars=False,
               gaia_stars=False,
               large_galaxies=False,
               star_clusters=True,
               star_halos=False,
               record_event=None,
               **kwargs):
    '''
    In this stage we run SED-matched detection to find objects in the
    images.  For each object detected, a `tractor` source object is
    created, initially a `tractor.PointSource`.  In this stage, the
    sources are also split into "blobs" of overlapping pixels.  Each
    of these blobs will be processed independently.
    '''
    from functools import reduce
    from tractor import PointSource, NanoMaggies, RaDecPos, Catalog
    from legacypipe.detection import (detection_maps,
                        run_sed_matched_filters, segment_and_group_sources)
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.measurements import label, center_of_mass
    from legacypipe.reference import get_reference_sources

    record_event and record_event('stage_srcs: starting')

    tlast = Time()

    refstars, refcat = get_reference_sources(survey, targetwcs, pixscale, bands,
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

    if refstars:
        # Don't detect new sources where we already have reference stars
        avoid_x = refstars.ibx[refstars.in_bounds]
        avoid_y = refstars.iby[refstars.in_bounds]
    else:
        avoid_x, avoid_y = np.array([]), np.array([])

    # Subtract star halos?
    if star_halos and gaia_stars:
        Igaia = []
        gaia = refstars
        Igaia, = np.nonzero(np.logical_or(refstars.isbright, refstars.ismedium) *
                            np.logical_not(refstars.iscluster) * refstars.pointsource)
        Igaia = Igaia[np.argsort(gaia.phot_g_mean_mag[Igaia])]
        debug(len(Igaia), 'stars for halo fitting')
        if len(Igaia):
            from legacypipe.halos import fit_halos, subtract_halos
            # FIXME -- another coadd...
            coimgs,cons = quick_coadds(tims, bands, targetwcs)
            fluxes,haloimgs = fit_halos(coimgs, cons, H, W, targetwcs, pixscale, bands,
                                        gaia[Igaia], plots, ps)
            init_fluxes = [(f and f[0] or None) for f in fluxes]

            if plots:
                plt.clf()
                dimshow(get_rgb(coimgs, bands, **rgbkwargs))
                plt.title('data')
                ps.savefig()
                plt.clf()
                dimshow(get_rgb(haloimgs, bands, **rgbkwargs))
                plt.title('fit profiles')
                ps.savefig()
                plt.clf()
                dimshow(get_rgb([c-h for c,h in zip(coimgs,haloimgs)], bands,
                                **rgbkwargs))
                plt.title('data - fit profiles')
                ps.savefig()

            # Subtract first-round halos from coadd
            co2 = [c - h for c,h in zip(coimgs,haloimgs)]
            del haloimgs

            fluxes2,haloimgs2 = fit_halos(co2, cons, H, W, targetwcs, pixscale, bands,
                                          gaia[Igaia], plots, ps,
                                          init_fluxes=init_fluxes)
            del co2, cons
            fluxarray = np.array([f[0] for f in fluxes2])

            for iband,b in enumerate(bands):
                haloflux = np.zeros(len(refstars))
                haloflux[Igaia] = fluxarray[:,iband]
                refstars.set('star_halo_flux_%s' % b, haloflux)

            if plots:
                plt.clf()
                dimshow(get_rgb(haloimgs2, bands, **rgbkwargs))
                plt.title('second-round fit profiles')
                ps.savefig()
                plt.clf()
                dimshow(get_rgb([c-h2 for c,h2 in zip(coimgs,haloimgs2)], bands,
                                **rgbkwargs))
                plt.title('second-round data - fit profiles')
                ps.savefig()

            del haloimgs2
            del coimgs

            # Actually subtract the halos from the tims!
            subtract_halos(tims, gaia[Igaia], fluxarray, pixscale, bands, plots, ps,mp)

            if plots:
                coimgs,_ = quick_coadds(tims, bands, targetwcs)
                plt.clf()
                dimshow(get_rgb(coimgs, bands, **rgbkwargs))
                plt.title('halos subtracted')
                ps.savefig()

    if refstars or T_donotfit or T_clusters:
        allrefs = merge_tables([t for t in [refstars, T_donotfit, T_clusters] if t],
                               columns='fillzero')
        with survey.write_output('ref-sources', brick=brickname) as out:
            allrefs.writeto(None, fits_object=out.fits, primheader=version_header)
        del allrefs

    record_event and record_event('stage_srcs: detection maps')

    tnow = Time()
    info('Rendering detection maps...')
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
        rgb = get_rgb(detmaps, bands, **rgbkwargs)
        plt.clf()
        dimshow(rgb)
        plt.title('detmaps')
        ps.savefig()

        for i,satpix in enumerate(saturated_pix):
            rgb[:,:,2-i][satpix] = 1
        plt.clf()
        dimshow(rgb)
        plt.title('detmaps & saturated')
        ps.savefig()

        coimgs,cons = quick_coadds(tims, bands, targetwcs, fill_holes=False)

        if refstars:
            plt.clf()
            dimshow(get_rgb(coimgs, bands, **rgbkwargs))
            ax = plt.axis()
            lp,lt = [],[]
            tycho = refstars[refstars.isbright]
            if len(tycho):
                ok,ix,iy = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
                p = plt.plot(ix-1, iy-1, 'o', mew=3, ms=14, mec='r', mfc='none')
                lp.append(p)
                lt.append('Tycho-2 only')
            if gaia_stars:
                gaia = refstars[refstars.ref_cat == 'G2']
            if gaia_stars and len(gaia):
                ok,ix,iy = targetwcs.radec2pixelxy(gaia.ra, gaia.dec)
                p = plt.plot(ix-1, iy-1, 'o', mew=3, ms=10, mec='c', mfc='none')
                lp.append(p)
                lt.append('Gaia')
            # star_clusters?
            if large_galaxies:
                galaxies = refstars[refstars.islargegalaxy]
            if large_galaxies and len(galaxies):
                ok,ix,iy = targetwcs.radec2pixelxy(galaxies.ra, galaxies.dec)
                p = plt.plot(ix-1, iy-1, 'o', mew=3, ms=14, mec=(0,1,0), mfc='none')
                lp.append(p)
                lt.append('Galaxies')
            plt.axis(ax)
            plt.title('Ref sources')
            plt.figlegend([p[0] for p in lp], lt)
            ps.savefig()

        if gaia_stars and len(gaia):
            ok,ix,iy = targetwcs.radec2pixelxy(gaia.ra, gaia.dec)
            for x,y,g in zip(ix,iy,gaia.phot_g_mean_mag):
                plt.text(x, y, '%.1f' % g, color='k',
                         bbox=dict(facecolor='w', alpha=0.5))
            plt.axis(ax)
            ps.savefig()

    # SED-matched detections
    record_event and record_event('stage_srcs: SED-matched')
    info('Running source detection at', nsigma, 'sigma')
    SEDs = survey.sed_matched_filters(bands)

    # Add a ~1" exclusion zone around reference stars and large galaxies
    avoid_r = np.zeros_like(avoid_x) + 4
    Tnew,newcat,hot = run_sed_matched_filters(
        SEDs, bands, detmaps, detivs, (avoid_x,avoid_y,avoid_r), targetwcs,
        nsigma=nsigma, saturated_pix=saturated_pix, plots=plots, ps=ps, mp=mp)
    if Tnew is None:
        raise NothingToDoError('No sources detected.')
    assert(len(Tnew) == len(newcat))
    Tnew.delete_column('peaksn')
    Tnew.delete_column('apsn')
    del detmaps
    del detivs
    Tnew.ref_cat = np.array(['  '] * len(Tnew))
    Tnew.ref_id  = np.zeros(len(Tnew), np.int64)

    # Merge newly detected sources with reference sources (Tycho2, Gaia, large galaxies)
    cats = newcat
    tables = [Tnew]
    if refstars and len(refstars):
        tables.append(refstars)
        cats += refcat
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
        coimgs,cons = quick_coadds(tims, bands, targetwcs)
        crossa = dict(ms=10, mew=1.5)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('Detections')
        ps.savefig()
        ax = plt.axis()
        if len(refstars):
            I, = np.nonzero([r[0] == 'T' for r in refstars.ref_cat])
            if len(I):
                plt.plot(refstars.ibx[I], refstars.iby[I], '+', color=(0,1,1),
                         label='Tycho-2', **crossa)
            I, = np.nonzero([r[0] == 'G' for r in refstars.ref_cat])
            if len(I):
                plt.plot(refstars.ibx[I], refstars.iby[I], '+',
                         color=(0.2,0.2,1), label='Gaia', **crossa)
            I, = np.nonzero([r[0] == 'L' for r in refstars.ref_cat])
            if len(I):
                plt.plot(refstars.ibx[I], refstars.iby[I], '+',
                         color=(0.6,0.6,0.2), label='Large Galaxy', **crossa)
        plt.plot(Tnew.ibx, Tnew.iby, '+', color=(0,1,0),
                 label='New SED-matched detections', **crossa)
        plt.axis(ax)
        plt.title('Detections')
        plt.legend(loc='upper left')
        ps.savefig()

        plt.clf()
        plt.subplot(1,2,1)
        dimshow(hot, vmin=0, vmax=1, cmap='hot')
        plt.title('hot')
        plt.subplot(1,2,2)
        rgb = np.zeros((H,W,3))
        for i,satpix in enumerate(saturated_pix):
            rgb[:,:,2-i] = satpix
        dimshow(rgb)
        plt.title('saturated_pix')
        ps.savefig()

    # Find "hot" pixels that are separated by masked pixels?
    if False:
        from scipy.ndimage.measurements import find_objects
        any_saturated = reduce(np.logical_or, saturated_pix)
        merging = np.zeros_like(any_saturated)
        h,w = any_saturated.shape
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
                    #print('slice', slc, 'cblobs2 shape', cblobs2.shape,
                    #      'blob index', cblobs2[slc.start])
                    mergedblob = cblobs2[slc.start]
                    counts[mergedblob] += 1
                slcs2 = find_objects(cblobs2)
                for blob,n in counts.most_common():
                    if n == 1:
                        break
                    #print('Index', index)
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

    keys = ['T', 'tims', 'blobsrcs', 'blobslices', 'blobs', 'cat',
            'ps', 'refstars', 'gaia_stars', 'saturated_pix',
            'T_donotfit', 'T_clusters']
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

def stage_fitblobs(T=None,
                   T_clusters=None,
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
                   nblobs=None, blob0=None, blobxy=None, blobradec=None, blobid=None,
                   max_blobsize=None,
                   simul_opt=False, use_ceres=True, mp=None,
                   checkpoint_filename=None,
                   checkpoint_period=600,
                   write_pickle_filename=None,
                   write_metrics=True,
                   get_all_models=False,
                   refstars=None,
                   rex=False,
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
    tlast = Time()

    # How far down to render model profiles
    minsigma = 0.1
    for tim in tims:
        tim.modelMinval = minsigma * tim.sig1

    if plots:
        coimgs,_ = quick_coadds(tims, bands, targetwcs)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0],'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % i,
                     ha='center', va='bottom', color='r')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

        for i,Isrcs in enumerate(blobsrcs):
            for isrc in Isrcs:
                src = cat[isrc]
                ra,dec = src.getPosition().ra, src.getPosition().dec
                ok,x,y = targetwcs.radec2pixelxy(ra, dec)
                plt.text(x, y, 'b%i/s%i' % (i,isrc),
                         ha='center', va='bottom', color='r')
        plt.axis(ax)
        plt.title('Blobs + Sources')
        ps.savefig()

        plt.clf()
        dimshow(blobs)
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0,bx0, bx1, bx1, bx0], [by0, by1, by1, by0, by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % i,
                     ha='center', va='bottom', color='r')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

        plt.clf()
        dimshow(blobs != -1)
        ax = plt.axis()
        for i,bs in enumerate(blobslices):
            sy,sx = bs
            by0,by1 = sy.start, sy.stop
            bx0,bx1 = sx.start, sx.stop
            plt.plot([bx0, bx0, bx1, bx1, bx0], [by0, by1, by1, by0,by0], 'r-')
            plt.text((bx0+bx1)/2., by0, '%i' % i,
                     ha='center', va='bottom', color='r')
        plt.axis(ax)
        plt.title('Blobs')
        ps.savefig()

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
        ok,x,y = targetwcs.radec2pixelxy(rd[:,0], rd[:,1])
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
        # 'blobslices' and 'blobsrcs' are lists where the index corresponds to the
        # value in the 'blobs' map.
        blobslices = [blobslices[i] for i in keepblobs]
        blobsrcs   = [blobsrcs  [i] for i in keepblobs]
        # one more place where blob numbers are recorded...
        T.blob = blobs[np.clip(T.iby, 0, H-1), np.clip(T.ibx, 0, W-1)]

    # drop any cached data before we start pickling/multiprocessing
    survey.drop_cache()

    if plots and refstars:
        plt.clf()
        dimshow(blobs>=0, vmin=0, vmax=1)
        ax = plt.axis()
        plt.plot(refstars.ibx, refstars.iby, 'ro')
        for x,y,mag in zip(refstars.ibx,refstars.iby,refstars.mag):
            plt.text(x, y, '%.1f' % (mag),
                     color='r', fontsize=10,
                     bbox=dict(facecolor='w', alpha=0.5))
        plt.axis(ax)
        plt.title('Reference stars')
        ps.savefig()

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
    if bailout:
        bailout_mask = _get_bailout_mask(blobs, skipblobs, targetwcs, W, H, brick,
                                         blobslices)
        # skip all blobs!
        skipblobs = np.unique(blobs[blobs>=0])
        # append empty results so that a later assert on the lengths will pass
        while len(R) < len(blobsrcs):
            R.append(dict(brickname=brickname, iblob=-1, result=None))

    if refstars:
        from legacypipe.oneblob import get_inblob_map
        refs = refstars[refstars.donotfit == False]
        if T_clusters is not None:
            refs = merge_tables([refs, T_clusters], columns='fillzero')
        refmap = get_inblob_map(targetwcs, refs)
        del refs
    else:
        HH, WW = targetwcs.shape
        refmap = np.zeros((int(HH), int(WW)), np.uint8)

    # Create the iterator over blobs to process
    blobiter = _blob_iter(brickname, blobslices, blobsrcs, blobs, targetwcs, tims,
                          cat, bands, plots, ps, simul_opt, use_ceres,
                          refmap, brick, rex,
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

    # one_blob can reduce the number and change the types of sources.
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
    # FIXME -- Dustin
    T.cut(II)
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
    assert(nb == 5) # ptsrc, rex, dev, exp, comp

    # Renumber blobs to make them contiguous.
    oldblob = T.blob
    ublob,iblob = np.unique(T.blob, return_inverse=True)
    del ublob
    assert(len(iblob) == len(T))
    T.blob = iblob.astype(np.int32)

    # write out blob map
    if write_metrics:
        # Build map from (old+1) to new blob numbers, for the blob image.
        blobmap = np.empty(blobs.max()+2, int)
        # make sure that dropped blobs -> -1
        blobmap[:] = -1
        # in particular,
        blobmap[0] = -1
        blobmap[oldblob + 1] = iblob
        blobs = blobmap[blobs+1]
        del blobmap

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
        hdr.add_record(dict(name='EQUINOX', value=2000.,comment='Observation epoch'))

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
    for k in ['fracflux', 'fracin', 'fracmasked', 'rchisq', 'cpu_source',
              'cpu_blob', 'blob_width', 'blob_height', 'blob_npix',
              'blob_nimages', 'blob_totalpix',
              'blob_symm_width', 'blob_symm_height',
              'blob_symm_npix', 'blob_symm_nimages', 'brightblob',
              'hit_limit', 'dchisq']:
        T.set(k, BB.get(k))

    # compute the pixel-space mask for *brightblob* values
    brightblobmask = refmap

    # Comment this out if you need to save the 'blobs' map for later (eg, sky fibers)
    blobs = None

    invvars = np.hstack(BB.srcinvvars)
    assert(cat.numberOfParams() == len(invvars))

    if write_metrics or get_all_models:
        TT,hdr = _format_all_models(T, newcat, BB, bands, rex)
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

    keys = ['cat', 'invvars', 'T', 'blobs', 'brightblobmask']
    if get_all_models:
        keys.append('all_models')
    if bailout:
        keys.append('bailout_mask')
    L = locals()
    rtn = dict([(k,L[k]) for k in keys])
    return rtn

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

def _format_all_models(T, newcat, BB, bands, rex):
    from legacypipe.catalog import prepare_fits_catalog, fits_typemap
    from tractor import Catalog

    TT = fits_table()
    # Copy only desired columns...
    for k in ['blob', 'brickid', 'brickname', 'dchisq', 'objid',
              'ra','dec',
              'cpu_source', 'cpu_blob', 'ninblob',
              'blob_width', 'blob_height', 'blob_npix', 'blob_nimages',
              'blob_totalpix',
              'blob_symm_width', 'blob_symm_height',
              'blob_symm_npix', 'blob_symm_nimages',
              'hit_limit']:
        TT.set(k, T.get(k))
    TT.type = np.array([fits_typemap[type(src)] for src in newcat])

    hdr = fitsio.FITSHDR()

    if rex:
        simpname = 'rex'
    else:
        simpname = 'simple'
    srctypes = ['ptsrc', simpname, 'dev','exp','comp']

    for srctype in srctypes:
        # Create catalog with the fit results for each source type
        xcat = Catalog(*[m.get(srctype,None) for m in BB.all_models])
        # NOTE that for Rex, the shapes have been converted to EllipseE
        # and the e1,e2 params are frozen.

        namemap = dict(ptsrc='psf', simple='simp')
        prefix = namemap.get(srctype,srctype)

        allivs = np.hstack([m.get(srctype,[]) for m in BB.all_model_ivs])
        assert(len(allivs) == xcat.numberOfParams())

        TT,hdr = prepare_fits_catalog(xcat, allivs, TT, hdr, bands, None,
                                      prefix=prefix+'_')
        TT.set('%s_cpu' % prefix,
               np.array([m.get(srctype,0)
                         for m in BB.all_model_cpu]).astype(np.float32))
        TT.set('%s_hit_limit' % prefix,
               np.array([m.get(srctype,0)
                         for m in BB.all_model_hit_limit]).astype(bool))

    # remove silly columns
    for col in TT.columns():
        # all types
        if '_type' in col:
            TT.delete_column(col)
            continue
        # shapes for shapeless types
        if (('psf_' in col or 'simp_' in col) and
            ('shape' in col or 'fracDev' in col)):
            TT.delete_column(col)
            continue
        # shapeDev for exp sources, vice versa
        if (('exp_' in col and 'Dev' in col) or
            ('dev_' in col and 'Exp' in col) or
            ('rex_' in col and 'Dev' in col)):
            TT.delete_column(col)
            continue
    TT.delete_column('dev_fracDev')
    TT.delete_column('dev_fracDev_ivar')
    if rex:
        TT.delete_column('rex_shapeExp_e1')
        TT.delete_column('rex_shapeExp_e2')
        TT.delete_column('rex_shapeExp_e1_ivar')
        TT.delete_column('rex_shapeExp_e2_ivar')
    return TT,hdr

def _blob_iter(brickname, blobslices, blobsrcs, blobs, targetwcs, tims, cat, bands,
               plots, ps, simul_opt, use_ceres, refmap,
               brick, rex,
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
    blob_order = np.array([i for i,npix in blobvals.most_common()])
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
            ok,x,y = tim.subwcs.radec2pixelxy(rr,dd)
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
            subwcs = tim.getWcs().shifted(sx0, sy0)
            # Note that we *don't* shift the PSF here -- we do that
            # in the one_blob code.
            subsky = tim.getSky().shifted(sx0, sy0)
            tim.imobj.psfnorm = tim.psfnorm
            tim.imobj.galnorm = tim.galnorm
            # FIXME -- maybe the cache is worth sending?
            if hasattr(tim.psf, 'clear_cache'):
                tim.psf.clear_cache()
            subtimargs.append((subimg, subie, subwcs, tim.subwcs,
                               tim.getPhotoCal(),
                               subsky, tim.psf, tim.name, sx0, sx1, sy0, sy1,
                               tim.band, tim.sig1, tim.modelMinval,
                               tim.imobj))

        yield (brickname, iblob,
               (nblob, iblob, Isrcs, targetwcs, bx0, by0, blobw, blobh,
               blobmask, subtimargs, [cat[i] for i in Isrcs], bands, plots, ps,
               simul_opt, use_ceres, rex, refmap[bslc]))

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

    if hasattr(tim, 'modelMinval'):
        debug('tim modelMinval', tim.modelMinval)
    else:
        # this doesn't really help when using pixelized PSFs / FFTs
        tim.modelMinval = minval = tim.sig * 0.1
    mod = tractor.getModelImage(0)
    debug('Getting model for', tim, ':', Time()-t0)
    return mod

def stage_coadds(survey=None, bands=None, version_header=None, targetwcs=None,
                 tims=None, ps=None, brickname=None, ccds=None,
                 custom_brick=False,
                 T=None, T_donotfit=None,
                 cat=None, pixscale=None, plots=False,
                 coadd_bw=False, brick=None, W=None, H=None, lanczos=True,
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
    tlast = Time()
    record_event and record_event('stage_coadds: starting')

    # Write per-brick CCDs table
    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='ccdinfo',
                            comment='NOAO data product type'))
    with survey.write_output('ccds-table', brick=brickname) as out:
        ccds.writeto(None, fits_object=out.fits, primheader=primhdr)

    tnow = Time()
    debug('[serial coadds]:', tnow-tlast)
    tlast = tnow
    # Render model images...
    record_event and record_event('stage_coadds: model images')
    mods = mp.map(_get_mod, [(tim, cat) for tim in tims])

    tnow = Time()
    debug('[parallel coadds] Getting model images:', tnow-tlast)
    tlast = tnow

    # Compute source pixel positions
    assert(len(T) == len(cat))
    ra  = np.array([src.getPosition().ra  for src in cat])
    dec = np.array([src.getPosition().dec for src in cat])
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
    C = make_coadds(tims, bands, targetwcs, mods=mods, xy=ixy,
                    ngood=True, detmaps=True, psfsize=True, allmasks=True,
                    lanczos=lanczos,
                    apertures=apertures, apxy=apxy,
                    callback=write_coadd_images,
                    callback_args=(survey, brickname, version_header, tims,
                                   targetwcs),
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
        T_image_coadds = make_coadds(tims, bands, targetwcs,
                                     mods=image_only_mods,
                                     lanczos=lanczos, mp=mp)
        image_coadd= T_image_coadds.comods
        del T_image_coadds
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
    T.apflux       = np.zeros((len(T), len(bands), A), np.float32)
    T.apflux_ivar  = np.zeros((len(T), len(bands), A), np.float32)
    T.apflux_resid = np.zeros((len(T), len(bands), A), np.float32)
    if Nno:
        T_donotfit.apflux       = np.zeros((Nno, len(bands), A), np.float32)
        T_donotfit.apflux_ivar  = np.zeros((Nno, len(bands), A), np.float32)
        T_donotfit.apflux_resid = np.zeros((Nno, len(bands), A), np.float32)
    AP = C.AP
    for iband,band in enumerate(bands):
        T.apflux      [:,iband,:] = AP.get('apflux_img_%s'      % band)[:Nyes,:]
        T.apflux_ivar [:,iband,:] = AP.get('apflux_img_ivar_%s' % band)[:Nyes,:]
        T.apflux_resid[:,iband,:] = AP.get('apflux_resid_%s'    % band)[:Nyes,:]
        if Nno:
            T_donotfit.apflux      [:,iband,:] = AP.get('apflux_img_%s'      % band)[Nyes:,:]
            T_donotfit.apflux_ivar [:,iband,:] = AP.get('apflux_img_ivar_%s' % band)[Nyes:,:]
            T_donotfit.apflux_resid[:,iband,:] = AP.get('apflux_resid_%s'    % band)[Nyes:,:]
    del AP

    # Compute depth histogram
    D = _depth_histogram(brick, targetwcs, bands, C.psfdetivs, C.galdetivs)
    with survey.write_output('depth-table', brick=brickname) as out:
        D.writeto(None, fits_object=out.fits)
    del D

    coadd_list= [('image', C.coimgs,   rgbkwargs),
                 ('model', C.comods,   rgbkwargs),
                 ('resid', C.coresids, rgbkwargs_resid)]
    if hasattr(tims[0], 'sims_image'):
        coadd_list.append(('simscoadd', sims_coadd, rgbkwargs))

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
        dimshow(get_rgb(C.coimgs, bands, **rgbkwargs))
        ax = plt.axis()
        #plt.plot(np.vstack((x0,x1))-1, np.vstack((y0,y1))-1, 'r-')
        for xx0,yy0,xx1,yy1 in zip(x0,y0,x1,y1):
            plt.plot([xx0-1,xx1-1], [yy0-1,yy1-1], 'r-')
        plt.plot(x1-1, y1-1, 'r.')
        plt.axis(ax)
        plt.title('Original to final source positions')
        ps.savefig()

        plt.clf()
        dimshow(get_rgb(C.coimgs, bands, **rgbkwargs))
        ax = plt.axis()
        ps.savefig()

        for i,(src,x,y,rr,dd) in enumerate(zip(cat, x1, y1, ra, dec)):
            from tractor import PointSource
            from tractor.galaxy import DevGalaxy, ExpGalaxy, FixedCompositeGalaxy
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
            elif isinstance(src, FixedCompositeGalaxy):
                ee = [src.shapeExp, src.shapeDev]
                cc = 'm'
                ec = ['m', 'c']
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
    return dict(T=T, apertures_pix=apertures,
                apertures_arcsec=apertures_arcsec,
                maskbits=maskbits,
                maskbits_header=maskbits_header)

def get_fiber_fluxes(cat, T, targetwcs, H, W, pixscale, bands,
                     fibersize=1.5, seeing=1., year=2020.0,
                     plots=False, ps=None):
    from tractor import GaussianMixturePSF
    from legacypipe.survey import LegacySurveyWcs
    import astropy.time
    from tractor.tractortime import TAITime
    from tractor.image import Image
    from tractor.basics import NanoMaggies, LinearPhotoCal
    from astrometry.util.util import Tan
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
        #print('Source', src)
        # This works even if bands[0] has zero flux (or no overlapping
        # images)
        ums = src.getUnitFluxModelPatches(faketim)
        #print('ums', ums)
        assert(len(ums) == 1)
        patch = ums[0]
        if patch is None:
            continue
        #print('sum', patch.patch.sum())
        br = src.getBrightness()
        for iband,(modimg,band) in enumerate(zip(modimgs,bands)):
            flux = br.getFlux(band)
            flux_iv = T.flux_ivar[isrc, iband]
            #print('Band', band, 'flux', flux, 'iv', flux_iv)
            if flux > 0 and flux_iv > 0:
                # Accumulate
                patch.addTo(modimg, scale=flux)
                # Add to blank image & photometer
                patch.addTo(onemod, scale=flux)
                aper = photutils.CircularAperture((sx, sy), fiberrad)
                p = photutils.aperture_photometry(onemod, aper)
                f = p.field('aperture_sum')[0]
                fiberflux[isrc,iband] = f
                #print('Aperture flux:', f)
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
        fibertotflux[:, iband] = f

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
    unwise_coadds=False,
    version_header=None,
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

    if not plots:
        ps = None

    tiles = unwise_tiles_touching_wcs(targetwcs)
    info('Cut to', len(tiles), 'unWISE tiles')

    # the way the roiradec box is used, the min/max order doesn't matter
    roiradec = [targetrd[0,0], targetrd[2,0], targetrd[0,1], targetrd[2,1]]

    wcat = []
    for src in cat:
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
            args.append((wcat, wtiles, band, roiradec,
                         wise_ceres, wpixpsf, unwise_coadds, get_masks, ps, True, unwise_modelsky_dir))

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
        nil,ne = TR.epoch_bitmask.shape
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
        for (ie,a),r in zip(eargs, phots):
            debug('Epoch', ie, 'photometry:')
            if r is None:
                debug('Failed.')
                continue
            assert(ie < Nepochs)
            phot = r.phot
            #phot.about()
            phot.delete_column('wise_coadd_id')
            for c in phot.columns():
                if not c in WISE_T.columns():
                    x = phot.get(c)
                    WISE_T.set(c, np.zeros((len(x), Nepochs), x.dtype))
                X = WISE_T.get(c)
                X[:,ie] = phot.get(c)

    debug('Returning: WISE', WISE)
    debug('Returning: WISE_T', WISE_T)

    return dict(WISE=WISE, WISE_T=WISE_T, wise_mask_maps=wise_mask_maps)

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
    gaia_stars=False,
    record_event=None,
    **kwargs):
    '''
    Final stage in the pipeline: format results for the output
    catalog.
    '''
    from legacypipe.catalog import prepare_fits_catalog

    record_event and record_event('stage_writecat: starting')

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

    #print('Catalog table contents:')
    #TT.about()

    hdr = fs = None
    T2,hdr = prepare_fits_catalog(cat, invvars, TT, hdr, bands, fs)

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

    # print('T2:')
    # T2.about()

    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='catalog',
                            comment='NOAO data product type'))

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
        T_donotfit.type = np.array(['DUP']*len(T_donotfit))
        T2 = merge_tables([T2, T_donotfit], columns='fillzero')

    # Brick pixel positions
    ok,bx,by = targetwcs.radec2pixelxy(T2.orig_ra, T2.orig_dec)
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
    T2.maskbits = maskbits[np.clip(T2.by, 0, H-1).astype(int),
                           np.clip(T2.bx, 0, W-1).astype(int)]
    del maskbits

    with survey.write_output('tractor-intermediate', brick=brickname) as out:
        T2.writeto(None, fits_object=out.fits, primheader=primhdr, header=hdr)

    # The "format_catalog" code expects all lower-case column names...
    for c in T2.columns():
        if c != c.lower():
            T2.rename(c, c.lower())
    from legacypipe.format_catalog import format_catalog
    with survey.write_output('tractor', brick=brickname) as out:
        format_catalog(T2, hdr, primhdr, survey.allbands, None, release,
                       write_kwargs=dict(fits_object=out.fits),
                       N_wise_epochs=11, motions=gaia_stars, gaia_tagalong=True)

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

    return dict(T2=T2)

def run_brick(brick, survey, radec=None, pixscale=0.262,
              width=3600, height=3600,
              release=None,
              zoom=None,
              bands=None,
              allbands='grz',
              depth_cut=None,
              nblobs=None, blob=None, blobxy=None, blobradec=None, blobid=None,
              max_blobsize=None,
              nsigma=6,
              simul_opt=False,
              wise=True,
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
              rgb_kwargs=None,
              rex=False,
              splinesky=True,
              subsky=True,
              constant_invvar=False,
              tycho_stars=False,
              gaia_stars=False,
              large_galaxies=False,
              min_mjd=None, max_mjd=None,
              unwise_coadds=False,
              bail_out=False,
              ceres=True,
              wise_ceres=True,
              unwise_dir=None,
              unwise_tr_dir=None,
              unwise_modelsky_dir=None,
              threads=None,
              plots=False, plots2=False, coadd_bw=False,
              plot_base=None, plot_number=0,
              record_event=None,
    # These are for the 'stages' infrastructure
              pickle_pat='pickles/runbrick-%(brick)s-%%(stage)s.pickle',
              stages=['writecat'],
              force=[], forceall=False, write_pickles=True,
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

    - *simul_opt*: boolean; during fitting, if a blob contains multiple
      sources, run a step of fitting the sources simultaneously?

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

    # print('Total Memory Available to Job:')
    # get_ulimit()

    # *initargs* are passed to the first stage (stage_tims)
    # so should be quantities that shouldn't get updated from their pickled
    # values.
    initargs = {}
    # *kwargs* update the pickled values from previous stages
    kwargs = {}

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
            release = 8888

    kwargs.update(ps=ps, nsigma=nsigma,
                  gaussPsf=gaussPsf, pixPsf=pixPsf, hybridPsf=hybridPsf,
                  release=release,
                  normalizePsf=normalizePsf,
                  apodize=apodize,
                  rgb_kwargs=rgb_kwargs,
                  rex=rex,
                  constant_invvar=constant_invvar,
                  depth_cut=depth_cut,
                  splinesky=splinesky,
                  subsky=subsky,
                  tycho_stars=tycho_stars,
                  gaia_stars=gaia_stars,
                  large_galaxies=large_galaxies,
                  min_mjd=min_mjd, max_mjd=max_mjd,
                  simul_opt=simul_opt,
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
        'outliers': 'tims',
        'srcs': 'outliers',

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
                'image_coadds':'outliers',
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

    parser.add_argument(
        '--simul-opt', action='store_true', default=False,
        help='Do simultaneous optimization after model selection')

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

    parser.add_argument('--simp', dest='rex', default=True,
                        action='store_false',
                        help='Use SIMP rather than REX')
    parser.add_argument(
        '--coadd-bw', action='store_true', default=False,
        help='Create grayscale coadds if only one band is available?')

    parser.add_argument('--bands', default=None,
                        help='Set the list of bands (filters) that are included in processing: comma-separated list, default "g,r,z"')

    parser.add_argument('--depth-cut', default=None, type=float,
                        help='Margin in mags to use to cut to the set of CCDs required to reach our depth target + margin')

    parser.add_argument('--no-tycho', dest='tycho_stars', default=True,
                        action='store_false',
                        help="Don't use Tycho-2 sources as fixed stars")

    parser.add_argument('--no-gaia', dest='gaia_stars', default=True,
                        action='store_false',
                        help="Don't use Gaia sources as fixed stars")

    parser.add_argument('--no-large-galaxies', dest='large_galaxies', default=True,
                        action='store_false', help="Don't do the default large-galaxy magic.")
    # HACK -- Default value for DR8 MJD cut
    # DR8 -- drop early data from before additional baffling was added to the camera.
    # 56730 = 2014-03-14
    parser.add_argument('--min-mjd', type=float, default=56730.,
                        help='Only keep images taken after the given MJD')
    parser.add_argument('--max-mjd', type=float,
                        help='Only keep images taken before the given MJD')

    parser.add_argument('--no-splinesky', dest='splinesky', default=True,
                        action='store_false', help='Use constant sky rather than spline.')
    parser.add_argument('--unwise-coadds', default=False,
                        action='store_true', help='Write FITS and JPEG unWISE coadds?')

    parser.add_argument('--bail-out', default=False, action='store_true',
                        help='Bail out of "fitblobs" processing, writing all blobs from the checkpoint and skipping any remaining ones.')

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
        unwise_modelsky_dir = os.path.join(survey.get_calib_dir(), 'wise', 'modelsky')
        if not os.path.exists(unwise_modelsky_dir):
            print('WARNING: no WISE sky background maps in {}'.format(unwise_modelsky_dir))
            unwise_modelsky_dir = None
        else:
            unwise_modelsky_dir = os.path.realpath(unwise_modelsky_dir) # follow the soft link
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
    #opt.update(splinesky=True)
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
    cid = os.environ.get('SLURM_CLUSTER_NAME', 'none')
    jid = os.environ.get('SLURM_JOB_ID', 'none')
    aid = os.environ.get('ARRAY_TASK_ID', 'none')
    print('Slurm cluster/job/array:', cid, '/', jid, '/', aid)
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

    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

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
