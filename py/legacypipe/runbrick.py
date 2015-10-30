'''
Main "pipeline" script for the Dark Energy Camera Legacy Survey (DECaLS)
data reductions.

For calling from other scripts, see:

- :py:func:`run_brick`

Or for much more fine-grained control, see the individual stages:

- :py:func:`stage_tims`
- :py:func:`stage_image_coadds`
- :py:func:`stage_srcs`
- :py:func:`stage_fitblobs`
- :py:func:`stage_fitblobs_finish`
- :py:func:`stage_coadds`
- :py:func:`stage_wise_forced`
- :py:func:`stage_writecat`

To see the code we run on each "blob" of pixels,

- :py:func:`_one_blob`

'''

from __future__ import print_function

# python -u legacypipe/runbrick.py -b 2437p082 --zoom 2575 2675 400 500 -P "pickles/zoom2-%(brick)s-%%(stage)s.pickle" > log 2>&1 &

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import pylab as plt
import numpy as np
import sys
import os

import fitsio

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.file import trymakedirs
from astrometry.util.plotutils import PlotSequence, dimshow
from astrometry.util.resample import resample_with_wcs, OverlapError
from astrometry.util.ttime import Time
from astrometry.util.starutil_numpy import (radectoxyz, ra2hmsstring,
                                            dec2dmsstring)
from astrometry.util.miscutils import patch_image

from tractor import Tractor, PointSource, Image, NanoMaggies
from tractor.ellipses import EllipseESoft, EllipseE
from tractor.galaxy import DevGalaxy, ExpGalaxy, FixedCompositeGalaxy, SoftenedFracDev, FracDev, disable_galaxy_cache

# Argh, can't do relative imports if this script is to be runnable.
from legacypipe.common import *
from legacypipe.utils import RunbrickError, NothingToDoError, iterwrapper
from legacypipe.runbrick_plots import _plot_mods


## GLOBALS!  Oh my!
nocache = True

# RGB image args used in the tile viewer:
rgbkwargs = dict(mnmx=(-1,100.), arcsinh=1.)
rgbkwargs_resid = dict(mnmx=(-5,5))

def runbrick_global_init():
    if nocache:
        disable_galaxy_cache()

def stage_tims(W=3600, H=3600, pixscale=0.262, brickname=None,
               decals=None,
               ra=None, dec=None,
               plots=False, ps=None, decals_dir=None,
               target_extent=None, pipe=False, program_name='runbrick.py',
               bands='grz',
               do_calibs=True,
               const2psf=False, gaussPsf=False, pixPsf=False,
               splinesky=False,
               use_blacklist = True,
               mp=None,
               rsync=False,
               **kwargs):
    '''
    This is the first stage in the pipeline.  It
    determines which CCD images overlap the brick or region of
    interest, runs calibrations for those images if necessary, and
    then reads the images, creating `tractor.Image` ("tractor image"
    or "tim") objects for them.

    PSF options:

    - *const2psf*: boolean.  2-component general Gaussian mixture, fit
      to PsfEx model at the center of the image.  Used in DR1.

    - *gaussPsf*: boolean.  Single-component circular Gaussian, with
      width set from the header FWHM value.  Useful for quick
      debugging.

    - *pixPsf*: boolean.  Pixelized PsfEx model, evaluated at the
      image center.  Uses the FFT-based galaxy convolution code.

    - *splinesky*: boolean.  Use SplineSky model, rather than ConstantSky?

    '''
    t0 = tlast = Time()

    # early fail for mysterious "ImportError: c.so.6: cannot open
    # shared object file: No such file or directory" seen in DR1 on
    # Edison.
    from tractor.mix import c_gauss_2d_grid

    if decals is None:
        decals = Decals(decals_dir)

    if ra is not None:
        # Custom brick; create a fake 'brick' object
        brick = BrickDuck()
        brick.ra  = ra
        brick.dec = dec
        brickid = brick.brickid = -1
        brick.brickname = brickname
    else:
        brick = decals.get_brick_by_name(brickname)
        if brick is None:
            raise RunbrickError('No such brick: "%s"' % brickname)

        print('Chosen brick:')
        brick.about()
        brickid = brick.brickid
        brickname = brick.brickname

    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    if target_extent is not None:
        (x0,x1,y0,y1) = target_extent
        W = x1-x0
        H = y1-y0
        targetwcs = targetwcs.get_subimage(x0, y0, W, H)
    pixscale = targetwcs.pixel_scale()
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])

    if ra is not None:
        brick.ra1,nil  = targetwcs.pixelxy2radec(W, H/2)
        brick.ra2,nil  = targetwcs.pixelxy2radec(1, H/2)
        nil, brick.dec1 = targetwcs.pixelxy2radec(W/2, 1)
        nil, brick.dec2 = targetwcs.pixelxy2radec(W/2, H)
        #print('RA1,RA2', brick.ra1, brick.ra2)
        #print('Dec1,Dec2', brick.dec1, brick.dec2)

    gitver = get_git_version()

    version_hdr = get_version_header(program_name, decals.decals_dir,
                                     git_version=gitver)
    for i,dep in enumerate(['numpy', 'scipy', 'wcslib', 'astropy', 'photutils',
                            'ceres', 'sextractor', 'psfex', 'astrometry_net',
                            'tractor', 'fitsio', 'unwise_coadds']):
        # Look in the OS environment variables for modules-style
        # $scipy_VERSION => 0.15.1_5a3d8dfa-7.1
        default_ver = 'UNAVAILABLE'
        verstr = os.environ.get('%s_VERSION' % dep, default_ver)
        if verstr == default_ver:
            print('Warning: failed to get version string for "%s"' % dep)
        version_hdr.add_record(dict(name='DEPNAM%02i' % i, value=dep,
                                    comment='Name of dependency product'))
        version_hdr.add_record(dict(name='DEPVER%02i' % i, value=verstr,
                                    comment='Version of dependency product'))

    version_hdr.add_record(dict(name='BRICKNAM', value=brickname,
                                comment='DECaLS brick RRRr[pm]DDd'))
    version_hdr.add_record(dict(name='BRICKID' , value=brickid,
                                comment='DECaLS brick id'))
    version_hdr.add_record(dict(name='RAMIN'   , value=brick.ra1,
                                comment='Brick RA min'))
    version_hdr.add_record(dict(name='RAMAX'   , value=brick.ra2,
                                comment='Brick RA max'))
    version_hdr.add_record(dict(name='DECMIN'  , value=brick.dec1,
                                comment='Brick Dec min'))
    version_hdr.add_record(dict(name='DECMAX'  , value=brick.dec2,
                                comment='Brick Dec max'))
    version_hdr.add_record(dict(name='BRICKRA' , value=brick.ra,
                                comment='Brick center'))
    version_hdr.add_record(dict(name='BRICKDEC', value=brick.dec,
                                comment='Brick center'))

    # NOAO-requested headers
    version_hdr.add_record(dict(
        name='RA', value=ra2hmsstring(brick.ra, separator=':'),
        comment='[h] RA Brick center'))
    version_hdr.add_record(dict(
        name='DEC', value=dec2dmsstring(brick.dec, separator=':'),
        comment='[deg] Dec Brick center'))

    version_hdr.add_record(dict(
        name='CENTRA', value=brick.ra, comment='[deg] Brick center RA'))
    version_hdr.add_record(dict(
        name='CENTDEC', value=brick.dec, comment='[deg] Brick center Dec'))

    for i,(r,d) in enumerate(targetrd[:4]):
        version_hdr.add_record(dict(
            name='CORN%iRA' % (i+1), value=r, comment='[deg] Brick corner RA'))
        version_hdr.add_record(dict(
            name='CORN%iDEC' %(i+1), value=d, comment='[deg] Brick corner Dec'))

    # print('Version header:')
    # print(version_hdr)

    ccds = decals.ccds_touching_wcs(targetwcs, ccdrad=None)
    if ccds is None:
        raise NothingToDoError('No CCDs touching brick')
    print(len(ccds), 'CCDs touching target WCS')

    blacklist = [
        '2012B-0003', # labeled as "DES SV", but appears to exclusively be DES deep fields taken during SV, through Jan 2013.
        '2013A-0351', # lots of deep data on COSMOS
        '2014A-0339', # two strips of sky
        '2013A-0360', # 9 fields total
        '2013A-0614', # 2 fields
        '2013A-0717', # 2 fields
        '2013B-0502', # 3 fields
        '2014A-0239', # 1 field
        '2014A-0429', # 2 fields
        '2013A-0611', # many 900-sec exposures in EDR region
        '2013A-0737', # 10 fields
        '2013A-0719', # 8 fields
        '2013A-9999', # 11 fields
        '2013A-0716', # 3 fields
        '2013A-0529', # 2 fields
        '2013A-0613', # 40 exposures of 600 sec in g,r and nothing else in DR2
        ]

    if use_blacklist:
        keep = np.array([propid not in blacklist for propid in ccds.propid])
        ccds.cut(keep)
        print(len(ccds), 'CCDs not in blacklisted propids (too many exposures!)')

    # Sort images by band -- this also eliminates images whose
    # *image.filter* string is not in *bands*.
    ccds.cut(np.hstack([np.flatnonzero(ccds.filter == band) for band in bands]))

    print('Cutting out non-photometric CCDs...')
    I = decals.photometric_ccds(ccds)
    print(len(I), 'of', len(ccds), 'CCDs are photometric')
    ccds.cut(I)

    ims = []
    for ccd in ccds:
        im = decals.get_image_object(ccd)
        ims.append(im)
        print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid)

    tnow = Time()
    print('[serial tims] Finding images touching brick:', tnow-tlast)
    tlast = tnow

    if rsync:
        # Check for existence of calibration files & rsync missing ones.
        reqd = []
        for im in ims:
            if not gaussPsf:
                reqd.append(im.psffn)
            if splinesky:
                reqd.append(im.splineskyfn)
            else:
                reqd.append(im.skyfn)
            reqd.append(im.pvwcsfn)
        # Just request missing files?
        reqd = [fn for fn in reqd if not os.path.exists(fn)]
        print('Required calib files:', reqd)
        print('Calib dir:', decals.get_calib_dir())
        caldir = decals.get_calib_dir() + '/'
        reqd = [fn.replace(caldir, '') for fn in reqd]

        cmd = 'rsync -LRrv edison:/scratch1/scratchdirs/desiproc/decals-dir/calib/./"{%s}" %s' % (','.join(reqd), caldir)
        print(cmd)
        os.system(cmd)

        # Also grab image files
        reqd = []
        for im in ims:
            reqd.extend([im.imgfn, im.dqfn, im.wtfn])
        reqd = [fn for fn in reqd if not os.path.exists(fn)]
        print('Required image files:', reqd)
        imgdir = decals.get_image_dir() + '/'
        reqd = [fn.replace(imgdir, '') for fn in reqd]

        cmd = 'rsync -LRrv edison:/scratch1/scratchdirs/desiproc/images/./"{%s}" %s' % (','.join(reqd), imgdir)
        print(cmd)
        os.system(cmd)

    if do_calibs:
        kwa = dict(git_version=gitver)
        if gaussPsf:
            kwa.update(psfex=False)
        if splinesky:
            kwa.update(splinesky=True)

        # Run calibrations
        args = [(im, kwa) for im in ims]
        mp.map(run_calibs, args)
        tnow = Time()
        print('[parallel tims] Calibrations:', tnow-tlast)
        tlast = tnow

    if False and plots:
        # Here, we read full images (not just subimages) to plot their
        # pixel distributions
        sig1s = dict([(b,[]) for b in bands])
        allpix = []
        for im in ims:
            subtim = im.get_tractor_image(splinesky=True, gaussPsf=True, subsky=False,
                                          radecpoly=targetrd)
            if subtim is None:
                continue
            fulltim = im.get_tractor_image(splinesky=True, gaussPsf=True)
            sig1s[fulltim.band].append(fulltim.sig1)
            allpix.append((fulltim.band, fulltim.name, im.exptime,
                           fulltim.getImage()[fulltim.getInvError() > 0]))
        sig1s = dict([(b, np.median(sig1s[b])) for b in sig1s.keys()])
        for b in bands:
            plt.clf()
            lp,lt = [],[]
            s = sig1s[b]
            for band,name,exptime,pix in allpix:
                if band != b:
                    continue
                # broaden range to encompass most pixels... only req'd when sky is bad
                lo,hi = -5.*s, 5.*s
                lo = min(lo, np.percentile(pix, 5))
                hi = max(hi, np.percentile(pix, 95))
                n,bb,p = plt.hist(pix, range=(lo, hi), bins=50, histtype='step',
                                 alpha=0.5)
                lp.append(p[0])
                lt.append('%s: %.0f s' % (name, exptime))
            plt.legend(lp, lt)
            plt.xlabel('Pixel values')
            plt.title('Pixel distributions: %s band' % b)
            ps.savefig()


    # Read Tractor images
    args = [(im, targetrd, dict(gaussPsf=gaussPsf, const2psf=const2psf,
                                pixPsf=pixPsf, splinesky=splinesky)) for im in ims]
    tims = mp.map(read_one_tim, args)

    # Cut the table of CCDs to match the 'tims' list
    I = np.flatnonzero(np.array([tim is not None for tim in tims]))
    ccds.cut(I)
    tims = [tim for tim in tims if tim is not None]
    assert(len(ccds) == len(tims))

    tnow = Time()
    print('[parallel tims] Read', len(ccds), 'images:', tnow-tlast)
    tlast = tnow

    if len(tims) == 0:
        raise NothingToDoError('No photometric CCDs touching brick.')

    for tim in tims:
        for cal,ver in [('sky', tim.skyver), ('wcs', tim.wcsver), ('psf', tim.psfver)]:
            if tim.plver != ver[1]:
                print('Warning: image "%s" PLVER is "%s" but %s calib was run on PLVER "%s"' %
                      (str(tim), tim.plver, cal, ver[1]))

    if plots:
        # Pixel histograms of subimages.
        for b in bands:
            sig1 = np.median([tim.sig1 for tim in tims if tim.band == b])
            plt.clf()
            for tim in tims:
                if tim.band != b:
                    continue
                # broaden range to encompass most pixels... only req'd when sky is bad
                lo,hi = -5.*s, 5.*s
                pix = tim.getImage()[tim.getInvError() > 0]
                lo = min(lo, np.percentile(pix, 5))
                hi = max(hi, np.percentile(pix, 95))
                plt.hist(pix, range=(lo, hi), bins=50, histtype='step',
                         alpha=0.5, label='%s: %.0f s' % (name, exptime))
            plt.legend()
            plt.xlabel('Pixel values')
            plt.title('Pixel distributions: %s band' % b)
            ps.savefig()

    if plots:
        for tim in tims:
            plt.clf()
            dimshow(tim.getImage(), **tim.ima)
            plt.title(tim.name)
            ps.savefig()

    if not pipe:
        # save resampling params
        tims_compute_resamp(mp, tims, targetwcs)
        tnow = Time()
        print('Computing resampling:', tnow-tlast)
        tlast = tnow
        # Produce per-band coadds, for plots
        coimgs,cons = compute_coadds(tims, bands, targetwcs)
        tnow = Time()
        print('Coadds:', tnow-tlast)
        tlast = tnow

    # Cut "bands" down to just the bands for which we have images.
    timbands = [tim.band for tim in tims]
    bands = [b for b in bands if b in timbands]
    print('Cut bands to', bands)

    for band in 'grz':
        hasit = band in bands
        version_hdr.add_record(dict(name='BRICK_%s' % band, value=hasit,
                                    comment='Does band %s touch this brick?' % band))
    version_hdr.add_record(dict(name='BRICKBND', value=''.join(bands),
                                comment='Bands touching this brick'))

    version_header = version_hdr

    keys = ['version_header', 'targetrd', 'pixscale', 'targetwcs', 'W','H',
            'bands', 'tims', 'ps', 'brickid', 'brickname', 'brick',
            'target_extent', 'ccds', 'bands', 'decals']
    if not pipe:
        keys.extend(['coimgs', 'cons'])
    rtn = dict()
    for k in keys:
        rtn[k] = locals()[k]
    return rtn

def _coadds(tims, bands, targetwcs,
            mods=None, xy=None, apertures=None, apxy=None,
            ngood=False, detmaps=False, psfsize=False,
            callback=None, callback_args=[],
            plots=False, ps=None):
    class Duck(object):
        pass
    C = Duck()

    W = targetwcs.get_width()
    H = targetwcs.get_height()

    # always, for patching SATUR, etc pixels?
    unweighted=True

    if not xy:
        psfsize = False
    
    C.coimgs = []
    if detmaps:
        C.galdetivs = []
        C.detivs = []
    if mods:
        C.comods = []
        C.coresids = []
        
    if apertures is not None:
        unweighted = True
        C.AP = fits_table()

    if xy:
        ix,iy = xy
        C.T = fits_table()
        C.T.nobs    = np.zeros((len(ix), len(bands)), np.uint8)
        C.T.anymask = np.zeros((len(ix), len(bands)), np.int16)
        C.T.allmask = np.zeros((len(ix), len(bands)), np.int16)
        if psfsize:
            C.T.psfsize = np.zeros((len(ix), len(bands)), np.float32)

    tinyw = 1e-30
    for iband,band in enumerate(bands):
        print('Computing coadd for band', band)

        cow    = np.zeros((H,W), np.float32)
        cowimg = np.zeros((H,W), np.float32)

        kwargs = dict(cowimg=cowimg, cow=cow)

        if detmaps:
            detiv = np.zeros((H,W), np.float32)
            C.detivs.append(detiv)
            kwargs.update(detiv=detiv)
            galdetiv = np.zeros((H,W), np.float32)
            C.galdetivs.append(galdetiv)
            kwargs.update(galdetiv=galdetiv)

        if mods:
            cowmod = np.zeros((H,W), np.float32)
            cochi2 = np.zeros((H,W), np.float32)
            kwargs.update(cowmod=cowmod, cochi2=cochi2)

        if unweighted:
            coimg  = np.zeros((H,W), np.float32)
            if mods:
                comod  = np.zeros((H,W), np.float32)
            con    = np.zeros((H,W), np.uint8)
            coiv   = np.zeros((H,W), np.float32)
            kwargs.update(coimg=coimg, coiv=coiv)

        # Note that we have 'congood' as well as 'nobs'.
        # 'congood' is used for the 'nexp' *image*
        # 'nobs' is used for the per-source measurement (you want to know the number
        #        of observations within the source footprint, not just the peak pixel
        #        which may be saturated, etc.)

        if ngood:
            congood = np.zeros((H,W), np.uint8)
            kwargs.update(congood=congood)

        if xy:
            # These match the type of the "DQ" images.
            ormask  = np.zeros((H,W), np.int16)
            andmask = np.empty((H,W), np.int16)
            allbits = reduce(np.bitwise_or, CP_DQ_BITS.values())
            andmask[:,:] = allbits
            nobs  = np.zeros((H,W), np.uint8)
            kwargs.update(ormask=ormask, andmask=andmask, nobs=nobs)

        if psfsize:
            psfsizemap = np.zeros((H,W), np.float32)

        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            R = tim_get_resamp(tim, targetwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R
            iv = tim.getInvvar()[Yi,Xi]
            im = tim.getImage ()[Yi,Xi]

            # invvar-weighted image
            cowimg[Yo,Xo] += iv * im
            cow   [Yo,Xo] += iv

            if unweighted:
                dq = tim.dq[Yi,Xi]
                # include BLEED, SATUR, INTERP pixels if no other pixels exists
                # (do this by eliminating all other CP flags)
                badbits = 0
                for bitname in ['badpix', 'cr', 'trans', 'edge', 'edge2']:
                    badbits |= CP_DQ_BITS[bitname]
                goodpix = ((dq & badbits) == 0)

                coimg[Yo,Xo] += goodpix * im
                con  [Yo,Xo] += goodpix
                coiv [Yo,Xo] += goodpix * 1./tim.sig1**2  # ...ish
                del dq

            if xy:
                dq = tim.dq[Yi,Xi]
                ormask [Yo,Xo] |= dq
                andmask[Yo,Xo] &= dq
                del dq
                # raw exposure count
                nobs[Yo,Xo] += 1

            if psfsize:
                # psfnorm is in units of 1/pixels.
                # (eg, psfnorm for a gaussian is ~ 1/psf_sigma)
                # Neff is in pixels**2
                neff = 1./tim.psfnorm**2
                # Narcsec is in arcsec**2
                narcsec = neff * tim.wcs.pixel_scale()**2
                psfsizemap[Yo,Xo] += iv * (1. / narcsec)

                #print('Tim', tim.name, 'FWHM', tim.psf_fwhm, 'pixels; sigma',
                #  tim.psf_sigma, 'FWHM arcsec:', tim.psf_fwhm * 0.262)
                # print('Narcsec', narcsec)
                # see2 = np.sqrt(narcsec) / (2.*np.sqrt(np.pi)) * 2.35
                #print('Tim', tim.name, 'PSF FWHM->', tim.psf_fwhm * 0.262,
                # 'arcsec vs', see2)
                
            if detmaps:
                # point-source depth
                detsig1 = tim.sig1 / tim.psfnorm
                detiv[Yo,Xo] += (iv > 0) * (1. / detsig1**2)

                # Galaxy detection
                gdetsig1 = tim.sig1 / tim.galnorm
                galdetiv[Yo,Xo] += (iv > 0) * (1. / gdetsig1**2)

            if ngood:
                congood[Yo,Xo] += (iv > 0)

            if mods:
                mo = mods[itim][Yi,Xi]
                # straight-up
                comod[Yo,Xo] += goodpix * mo
                # invvar-weighted
                cowmod[Yo,Xo] += iv * mo
                # chi-squared
                cochi2[Yo,Xo] += iv * (im - mo)**2
                del mo
                del goodpix

            del Yo,Xo,Yi,Xi,im,iv
            # END of loop over tims

        cowimg /= np.maximum(cow, tinyw)
        C.coimgs.append(cowimg)
        if mods:
            cowmod  /= np.maximum(cow, tinyw)
            C.comods.append(cowmod)
            coresid = cowimg - cowmod
            coresid[cow == 0] = 0.
            C.coresids.append(coresid)

        if unweighted:
            coimg  /= np.maximum(con, 1)
            del con
            cowimg[cow == 0] = coimg[cow == 0]
            if mods:
                cowmod[cow == 0] = comod[cow == 0]

        if plots:
            plt.clf()
            dimshow(nobs, vmin=0, vmax=max(1,nobs.max()), cmap='jet')
            plt.colorbar()
            plt.title('Number of observations, %s band' % band)
            ps.savefig()

            for k,v in CP_DQ_BITS.items():
                plt.clf()
                dimshow(ormask & v, vmin=0, vmax=v)
                plt.title('OR mask, %s band: %s' % (band, k))
                ps.savefig()

                plt.clf()
                dimshow(andmask & v, vmin=0, vmax=v)
                plt.title('AND mask, %s band: %s' % (band,k))
                ps.savefig()

        if xy:
            C.T.nobs [:,iband] = nobs[iy,ix]
            C.T.anymask[:,iband] =  ormask [iy,ix]
            C.T.allmask[:,iband] =  andmask[iy,ix]
            # unless there were no images there...
            C.T.allmask[nobs[iy,ix] == 0, iband] = 0

        if psfsize:
            # We're summing this across bands....
            wt = cow[iy,ix]
            # psfsizemap is in units of iv * (1 / arcsec**2)
            sz = psfsizemap[iy,ix]
            sz /= np.maximum(wt, tinyw)
            sz[wt == 0] = 0.
            # Back to units of linear arcsec.
            sz = 1. / np.sqrt(sz)
            sz[wt == 0] = 0.
            # Correction factor to get back to equivalent of Gaussian sigma
            sz /= (2. * np.sqrt(np.pi))
            # Conversion factor to FWHM (2.35)
            sz *= 2. * np.sqrt(2. * np.log(2.))
            C.T.psfsize[:,iband] = sz
            del psfsizemap

        if apertures is not None:
            import photutils

            # Aperture photometry, using the unweighted "coimg" and "coiv" arrays.
            with np.errstate(divide='ignore'):
                imsigma = 1.0/np.sqrt(coiv)
                imsigma[coiv == 0] = 0

            apimg = []
            apimgerr = []
            if mods:
                apres = []

            for rad in apertures:
                aper = photutils.CircularAperture(apxy, rad)
                p = photutils.aperture_photometry(coimg, aper, error=imsigma)
                apimg.append(p.field('aperture_sum'))
                apimgerr.append(p.field('aperture_sum_err'))
                if mods:
                    p = photutils.aperture_photometry(coresid, aper)
                    apres.append(p.field('aperture_sum'))
            ap = np.vstack(apimg).T
            ap[np.logical_not(np.isfinite(ap))] = 0.
            C.AP.set('apflux_img_%s' % band, ap)
            ap = 1./(np.vstack(apimgerr).T)**2
            ap[np.logical_not(np.isfinite(ap))] = 0.
            C.AP.set('apflux_img_ivar_%s' % band, ap)
            if mods:
                ap = np.vstack(apres).T
                ap[np.logical_not(np.isfinite(ap))] = 0.
                C.AP.set('apflux_resid_%s' % band, ap)
                del apres
            del apimg,apimgerr,ap

        if callback is not None:
            callback(band, *callback_args, **kwargs)
        # END of loop over bands

    return C


def _write_band_images(band,
                       brickname, version_header, tims, targetwcs, basedir,
                       cowimg=None, cow=None, cowmod=None, cochi2=None,
                       detiv=None, galdetiv=None, congood=None, **kwargs):

    # copy version_header before modifying...
    hdr = fitsio.FITSHDR()
    for r in version_header.records():
        hdr.add_record(r)
    # Grab these keywords from all input files for this band...
    keys = ['TELESCOP','OBSERVAT','OBS-LAT','OBS-LONG','OBS-ELEV',
            'INSTRUME','FILTER']
    vals = set()
    for tim in tims:
        if tim.band != band:
            continue
        v = []
        for key in keys:
            v.append(tim.primhdr.get(key,''))
        vals.add(tuple(v))
    for i,v in enumerate(vals):
        for ik,key in enumerate(keys):
            if i == 0:
                kk = key
            else:
                kk = key[:7] + '%i'%i
            hdr.add_record(dict(name=kk, value=v[ik]))
    hdr.add_record(dict(name='FILTERX', value=band))

    # DATE-OBS converted to TAI.
    # print('Times:', [tim.time for tim in tims if tim.band == band])
    mjds = [tim.time.toMjd() for tim in tims if tim.band == band]
    minmjd = min(mjds)
    maxmjd = max(mjds)
    #print('MJDs', mjds, 'range', minmjd, maxmjd)
    # back to date string in UTC...
    import astropy.time
    tt = [astropy.time.Time(mjd, format='mjd', scale='tai').utc.isot
          for mjd in [minmjd, maxmjd]]
    hdr.add_record(dict(
        name='DATEOBS1', value=tt[0],
        comment='DATE-OBS for the first image in the stack (UTC)'))
    hdr.add_record(dict(
        name='DATEOBS2', value=tt[1],
        comment='DATE-OBS for the last  image in the stack (UTC)'))

    # Plug the WCS header cards into these images
    targetwcs.add_to_header(hdr)
    hdr.delete('IMAGEW')
    hdr.delete('IMAGEH')
    hdr.add_record(dict(name='EQUINOX', value=2000.))

    imgs = [
        ('image', 'image',  cowimg,  False),
        ]
    if congood is not None:
        imgs.append(
            ('nexp',   'expmap',   congood, True ),
            )
    # if detiv is not None:
    if cowmod is not None:
        imgs.extend([
                ('invvar',   'wtmap',    cow,      False),
                ('model',    'model',    cowmod,   True ),
                ('chi2',     'chi2',     cochi2,   False),
                ('depth',    'psfdepth', detiv,    True ),
                ('galdepth', 'galdepth', galdetiv, True ),
                ])
    for name,prodtype,img,gzip in imgs:
        # Make a copy, because each image has different values for
        # these headers...
        hdr2 = MyFITSHDR()
        for r in hdr.records():
            hdr2.add_record(r)
        hdr2.add_record(dict(name='IMTYPE', value=name,
                             comment='DECaLS image type'))
        hdr2.add_record(dict(name='PRODTYPE', value=prodtype,
                             comment='NOAO image type'))
        if name in ['image', 'model']:
            hdr2.add_record(dict(name='MAGZERO', value=22.5,
                                 comment='Magnitude zeropoint'))
            hdr2.add_record(dict(name='BUNIT', value='nanomaggy',
                                 comment='AB mag = 22.5 - 2.5*log10(nanomaggy)'))
        if name in ['invvar', 'depth']:
            hdr2.add_record(dict(name='BUNIT', value='1/nanomaggy^2',
                                 comment='Ivar of ABmag=22.5-2.5*log10(nmgy)'))

        fn = os.path.join(basedir,
                          'decals-%s-%s-%s.fits' % (brickname, name, band))
        if gzip:
            fn += '.gz'
        fitsio.write(fn, img, clobber=True, header=hdr2)
        print('Wrote', fn)


def stage_mask_junk(tims=None, targetwcs=None, W=None, H=None, bands=None,
                    mp=None, nsigma=None, plots=None, ps=None, **kwargs):
    from scipy.ndimage.filters import gaussian_filter
    from scipy.ndimage.morphology import binary_fill_holes
    from scipy.ndimage.measurements import label, find_objects, center_of_mass
    from scipy.linalg import svd

    if plots:
        coimgs,cons = compute_coadds(tims, bands, targetwcs, fill_holes=False)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('Before')
        ps.savefig()

    allss = []
    for tim in tims:
        # detection map
        det = tim.data * (tim.inverr > 0)
        det = gaussian_filter(det, tim.psf_sigma) / tim.psfnorm**2
        detsig1 = tim.sig1 / tim.psfnorm
        det = (det > (nsigma * detsig1))
        det = binary_fill_holes(det)
        timblobs,timnblobs = label(det)
        timslices = find_objects(timblobs)

        if plots:
            zeroed = np.zeros(tim.shape, bool)

        for i,slc in enumerate(timslices):
            inblob = timblobs[slc]
            inblob = (inblob == (i+1))
            cy,cx = center_of_mass(inblob)
            bh,bw = inblob.shape
            xx = np.arange(bw)
            yy = np.arange(bh)
            ninblob = float(np.sum(inblob))
            cxx = np.sum(((xx - cx)**2)[np.newaxis,:] * inblob) / ninblob
            cyy = np.sum(((yy - cy)**2)[:,np.newaxis] * inblob) / ninblob
            cxy = np.sum((yy - cy)[:,np.newaxis] *
                         (xx - cx)[np.newaxis,:] * inblob) / ninblob
            C = np.array([[cxx, cxy],[cxy, cyy]])
            u,s,v = svd(C)
            allss.append(np.sqrt(np.abs(s)))

            # For an ellipse, this gives the major and minor axes
            # (ie, end to end, not semi-major/minor)
            ss = np.sqrt(s)
            major = 4. * ss[0]
            minor = 4. * ss[1]
            if not (major > 200 and minor/major < 0.1):
                continue
            # Zero it out!
            tim.inverr[slc] *= np.logical_not(inblob)
            # Add to dq mask bits
            tim.dq[slc] |= CP_DQ_BITS['longthin']

            ra,dec = tim.wcs.pixelToPosition(cx, cy)
            ok,bx,by = targetwcs.radec2pixelxy(ra, dec)
            print('Zeroing out a source with major/minor axis', major, '/',
                  minor, 'at centroid RA,Dec=(%.4f,%.4f), brick coords %i,%i'
                  % (ra, dec, bx, by))

            if plots:
                zeroed[slc] = np.logical_not(inblob)

        if plots:
            if np.sum(zeroed) > 0:
                plt.clf()
                from scipy.ndimage.morphology import binary_dilation
                zout = (binary_dilation(zeroed, structure=np.ones((3,3)))
                        - zeroed)
                dimshow(tim.getImage(), **tim.ima)
                outline = np.zeros(zout.shape+(4,), np.uint8)
                outline[:,:,1] = zout*255
                outline[:,:,3] = zout*255
                dimshow(outline)
                plt.title('Masked: ' + tim.name)
                ps.savefig()

    if plots:
        coimgs,cons = compute_coadds(tims, bands, targetwcs, fill_holes=False)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('After')
        ps.savefig()

        allss = np.array(allss)
        mx = max(allss.max(), 100) * 1.1
        plt.clf()
        plt.plot([0, mx], [0, mx], 'k-', alpha=0.2)
        plt.plot(allss[:,1], allss[:,0], 'b.')
        plt.ylabel('Major axis size (pixels)')
        plt.xlabel('Minor axis size (pixels)')
        plt.axis('scaled')
        plt.axis([0, mx, 0, mx])
        ps.savefig()

        plt.clf()
        plt.plot(allss[:,0], (allss[:,1]+1.) / (allss[:,0]+1.), 'b.')
        plt.xlabel('Major axis size (pixels)')
        plt.ylabel('Axis ratio')
        plt.axis([0, mx, 0, 1])
        plt.axhline(0.1, color='r', alpha=0.2)
        plt.axvline(200, color='r', alpha=0.2)
        ps.savefig()

    return dict(tims=tims)

def stage_image_coadds(targetwcs=None, bands=None, tims=None, outdir=None,
                       brickname=None, version_header=None,
                       plots=False, ps=None, coadd_bw=False, W=None, H=None,
                       brick=None, blobs=None,
                       **kwargs):
    '''
    Immediately after reading the images, we
    create coadds of just the image products.  Later, full coadds
    including the models will be created (in `stage_coadds`).  But
    it's handy to have the coadds early on, to diagnose problems or
    just to look at the data.

    '''

    if plots and False:

        for band in bands:

            # Plot sky-subtracted traces (projections)
            plt.clf()

            rr = []
            lp,lt = [],[]

            ppp = []

            cc = ['b','r','g','m','k']
            ci = 0
            for j,tim in enumerate(tims):
                if tim.band != band:
                    continue
                R = tim_get_resamp(tim, targetwcs)
                if R is None:
                    continue
                Yo,Xo,Yi,Xi = R

                proj = np.zeros((H,W), np.float32)
                haveproj = np.zeros((H,W), bool)

                proj[Yo,Xo] = tim.data[Yi,Xi]
                haveproj[Yo,Xo] = (tim.inverr[Yi,Xi] > 0)

                xx,ylo,ymed,yhi = [],[],[],[]
                for i in range(W):
                    I = np.flatnonzero(haveproj[:,i])
                    if len(I) == 0:
                        continue
                    xx.append(i)
                    y = proj[I,i]
                    lo,m,hi = np.percentile(y, [16, 50, 84])
                    ylo.append(lo)
                    ymed.append(m)
                    yhi.append(hi)

                if len(xx):
                    color = cc[ci % len(cc)]
                    pargs = dict(color=color)
                    ci += 1
                    p = plt.plot(xx, ymed, '-', alpha=0.5, zorder=10, **pargs)
                    plt.plot([xx[0], xx[-1]], [ymed[0], ymed[-1]], 'o', zorder=20, **pargs)
                    plt.plot(xx, ylo, '-', alpha=0.25, zorder=5, **pargs)
                    plt.plot(xx, yhi, '-', alpha=0.25, zorder=5, **pargs)

                    lp.append(p[0])
                    lt.append(tim.name)

                    rr.extend([h-l for h,l in zip(yhi,ylo)])

                    ppp.append((xx, ymed, color, tim.name))

            yrange = np.median(rr)
            plt.ylim(-yrange, +yrange)
            plt.xlim(0, W)
            plt.title('Per-column medians for %s band' % band)
            plt.legend(lp, lt, loc='upper right')
            ps.savefig()

            plt.clf()
            nplots = len(ppp)
            for i,(xx, ymed, color, name) in enumerate(ppp):
                dy = 0.25 * (i - float(nplots)/2) * yrange/2.
                plt.plot(xx, ymed + dy, color=color)
                plt.axhline(dy, color='k', alpha=0.1)
                plt.text(np.median(xx), np.median(ymed)+dy, name, ha='center', va='bottom', color='k',
                         bbox=dict(facecolor='w', edgecolor='none', alpha=0.5))
            plt.ylim(-yrange, +yrange)
            plt.title('Per-column medians for %s band' % band)
            ps.savefig()

    if outdir is None:
        outdir = '.'
    basedir = os.path.join(outdir, 'coadd', brickname[:3], brickname)
    trymakedirs(basedir)

    C = _coadds(tims, bands, targetwcs,
                #############
                detmaps=True,
                callback=_write_band_images,
                callback_args=(brickname, version_header, tims, targetwcs, basedir))

    if True:
        # Compute the brick's unique pixels.
        U = None
        if hasattr(brick, 'ra1'):
            print('Computing unique brick pixels...')
            xx,yy = np.meshgrid(np.arange(W), np.arange(H))
            rr,dd = targetwcs.pixelxy2radec(xx+1, yy+1)
            U = np.flatnonzero((rr >= brick.ra1 ) * (rr < brick.ra2 ) *
                               (dd >= brick.dec1) * (dd < brick.dec2))
            print(len(U), 'of', W*H, 'pixels are unique to this brick')

        # depth histogram bins
        depthbins = np.arange(19.9, 25.101, 0.1)
        depthbins[0] = 0.
        depthbins[-1] = 100.
        D = fits_table()
        D.depthlo = depthbins[:-1].astype(np.float32)
        D.depthhi = depthbins[1: ].astype(np.float32)

        for band,detiv,galdetiv in zip(bands, C.detivs, C.galdetivs):
            for det,name in [(detiv, 'ptsrc'), (galdetiv, 'gal')]:
                # compute stats for 5-sigma detection
                depth = 5. / np.sqrt(np.maximum(det, 1e-20))
                # that's flux in nanomaggies -- convert to mag
                depth = -2.5 * (np.log10(depth) - 9)
                # no coverage -> very bright detection limit
                depth[det == 0] = 0.
                if U is not None:
                    depth = depth.flat[U]
                print(band, name, 'band depth map: deciles',
                      np.percentile(depth, np.arange(0, 101, 10)))
                # histogram
                D.set('counts_%s_%s' % (name, band),
                      np.histogram(depth, bins=depthbins)[0].astype(np.int32))

        del U
        del depth
        del det
        del detiv
        del galdetiv
        fn = os.path.join(basedir, 'decals-%s-depth.fits' % (brickname))
        D.writeto(fn)
        print('Wrote', fn)
        del D

    #rgbkwargs2 = dict(mnmx=(-3., 3.))
    #rgbkwargs2 = dict(mnmx=(-2., 10.))

    tmpfn = create_temp(suffix='.png')
    for name,ims,rgbkw in [('image',C.coimgs,rgbkwargs),
        #('image2',C.coimgs,rgbkwargs2),
                           ]:
        rgb = get_rgb(ims, bands, **rgbkw)
        kwa = {}
        if coadd_bw and len(bands) == 1:
            i = 'zrg'.index(bands[0])
            rgb = rgb[:,:,i]
            kwa = dict(cmap='gray')
        plt.imsave(tmpfn, rgb, origin='lower', **kwa)
        jpegfn = os.path.join(basedir, 'decals-%s-%s.jpg' % (brickname, name))
        cmd = 'pngtopnm %s | pnmtojpeg -quality 90 > %s' % (tmpfn, jpegfn)
        os.system(cmd)
        os.unlink(tmpfn)
        print('Wrote', jpegfn)

        # Blob-outlined version
        if blobs is not None:
            from scipy.ndimage.morphology import binary_dilation
            outline = (binary_dilation(blobs >= 0, structure=np.ones((3,3)))
                       - (blobs >= 0))
            # coadd_bw?
            if len(rgb.shape) == 2:
                rgb = np.repeat(rgb[:,:,np.newaxis], 3, axis=2)
            # Outline in green
            rgb[:,:,0][outline] = 0
            rgb[:,:,1][outline] = 1
            rgb[:,:,2][outline] = 0
            plt.imsave(tmpfn, rgb, origin='lower', **kwa)
            jpegfn = os.path.join(basedir, 'decals-%s-%s.jpg' % (brickname, name+'blob'))
            cmd = 'pngtopnm %s | pnmtojpeg -quality 90 > %s' % (tmpfn, jpegfn)
            os.system(cmd)
            os.unlink(tmpfn)
            print('Wrote', jpegfn)

    return None

def _median_smooth_detmap(X):
    (detmap, detiv, binning) = X
    from scipy.ndimage.filters import median_filter
    #from astrometry.util.util import median_smooth
    #smoo = np.zeros_like(detmap)
    #median_smooth(detmap, detiv>0, 100, smoo)
    #smoo = median_filter(detmap, (50,50))
    # Bin down before median-filtering, for speed.
    binned,nil = bin_image(detmap, detiv, binning)
    smoo = median_filter(binned, (50,50))
    return smoo

def stage_srcs(coimgs=None, cons=None,
               targetrd=None, pixscale=None, targetwcs=None,
               W=None,H=None,
               bands=None, ps=None, tims=None,
               plots=False, plots2=False,
               pipe=False, brickname=None,
               mp=None, outdir=None, nsigma=5,
               no_sdss=False, on_bricks=False,
               decals=None, brick=None,
               **kwargs):
    '''
    In this stage we read SDSS catalog objects overlapping
    our region of interest (if enabled), and also run SED-match
    detection to find faint objects in the images.  For each object
    detected, a `tractor` source object is created: a
    `tractor.PointSource`, `tractor.ExpGalaxy`, `tractor.DevGalaxy`,
    or `tractor.FixedCompositeGalaxy` object.  In this stage, the
    sources are also split into "blobs" of overlapping pixels.  Each
    of these blobs will be processed independently.

    '''
    from legacypipe.detection import (detection_maps, sed_matched_filters,
                                      run_sed_matched_filters)

    tlast = Time()
    if not no_sdss:
        from legacypipe.sdss import get_sdss_sources

        # Read SDSS sources
        cols = ['parent', 'tai', 'mjd', 'psf_fwhm', 'objc_flags2', 'flags2',
                'devflux_ivar', 'expflux_ivar', 'calib_status', 'raerr',
                'decerr']
        sdsscat,Tsdss = get_sdss_sources(bands, targetwcs, extracols=cols,
                                 ellipse=LegacyEllipseWithPriors.fromRAbPhi)
        tnow = Time()
        print('[serial srcs] SDSS sources:', tnow-tlast)
        tlast = tnow
    else:
        sdsscat = []
        Tsdss = None

    avoid_xy = []
    if Tsdss is not None:
        # SDSS RAERR, DECERR are in arcsec.  Convert to deg.
        err = Tsdss.raerr / 3600.
        Tsdss.ra_ivar  = 1./err**2
        err = Tsdss.decerr / 3600.
        Tsdss.dec_ivar  = 1./err**2
        Tsdss.delete_column('raerr')
        Tsdss.delete_column('decerr')

        for c in Tsdss.columns():
            if c in ['itx','ity']:
                continue
            print('Renaming', c)
            Tsdss.rename(c, 'sdss_c')

    if on_bricks:
        from legacypipe.desi_common import read_fits_catalog
        # Find nearby bricks from earlier brick phases
        bricks = decals.get_bricks_readonly()
        print(len(bricks), 'bricks')
        bricks = bricks[bricks.brickq < brick.brickq]
        print(len(bricks), 'from phases before this brickq:', brick.brickq)
        if len(bricks):
            from astrometry.libkd.spherematch import match_radec

            bricks.cut(np.abs(brick.dec - bricks.dec) < 1)
            print(len(bricks), 'within 1 degree of Dec')
            radius = np.sqrt(2.) * np.abs(brick.dec2 - brick.dec1)
            I,J,d = match_radec(brick.ra, brick.dec, bricks.ra, bricks.dec,
                                radius)
            bricks.cut(J)
            print(len(bricks), 'within', radius, 'degrees')
        else:
            bricks = []
        B = []
        if outdir is None:
            outdir = '.'
        for b in bricks:
            fn = os.path.join(outdir, 'tractor', b.brickname[:3],
                              'tractor-%s.fits' % b.brickname)
            print('Looking for', fn)
            B.append(fits_table(fn))
        B = merge_tables(B)
        print('Total of', len(B), 'sources from neighbouring bricks')
        # Keep only sources that are primary in their own brick
        B.cut(B.brick_primary)
        print(len(B), 'are BRICK_PRIMARY')
        # HACK -- Keep only sources within a margin of this brick
        ok,B.xx,B.yy = targetwcs.radec2pixelxy(B.ra, B.dec)
        margin = 20
        B.cut((B.xx >= 1-margin) * (B.xx < W+margin) *
              (B.yy >= 1-margin) * (B.yy < H+margin))
        print(len(B), 'are within this image + margin')
        B.cut((B.out_of_bounds == False) * (B.left_blob == False))
        print(len(B), 'do not have out_of_bounds or left_blob set')

        # Note that we shouldn't need to drop sources that are within this
        # current brick's unique area, because we cut to sources that are
        # BRICK_PRIMARY within their own brick.

        # Create sources for these catalog entries
        ### see forced-photom-decam.py for some additional patchups?

        B.shapeexp = np.vstack((B.shapeexp_r, B.shapeexp_e1, B.shapeexp_e2)).T
        B.shapedev = np.vstack((B.shapedev_r, B.shapedev_e1, B.shapedev_e2)).T
        bcat = read_fits_catalog(B, ellipseClass=EllipseE)
        print('Created', len(bcat), 'tractor catalog objects')

        # Trim off SDSS sources that overlap this brick.
        if Tsdss is not None:
            keep_sdss = np.ones(len(Tsdss), bool)
            for brick in bricks:
                # Drop SDSS sources within the BRICK_PRIMARY region of the
                # neighbouring brick.
                keep_sdss[(Tsdss.ra  >= brick.ra1 ) * (Tsdss.ra  < brick.ra2 ) *
                          (Tsdss.dec >= brick.dec1) * (Tsdss.dec < brick.dec2)] = False
            if sum(keep_sdss) < len(keep_sdss):
                print('Trimming', len(keep_sdss)-sum(keep_sdss),
                      'SDSS sources within neighbouring bricks')
                Tsdss.cut(keep_sdss)
                sdsscat = Catalog(*[src for src,keep in zip(sdsscat,keep_sdss) if keep])

        # Add the new sources to the 'avoid_xy' list, which are
        # existing sources that should be avoided when detecting new
        # faint sources.
        ax = np.round(B.xx - 1).astype(int)
        ay = np.round(B.yy - 1).astype(int)
        avoid_xy.extend(zip(ax, ay))

        print('Subtracting tractor-on-bricks sources belonging to other bricks')
        ## HACK -- note that this is going to screw up fracflux and
        ## other metrics for sources in this brick that overlap
        ## subtracted sources.
        if plots:
            mods = []
            # Before...
            coimgs,cons = compute_coadds(tims, bands, targetwcs)
            plt.clf()
            dimshow(get_rgb(coimgs, bands))
            plt.title('Before subtracting tractor-on-bricks marginal sources')
            ps.savefig()

        for i,src in enumerate(bcat):
            print(' ', i, src)

        tlast = Time()
        mods = mp.map(_get_mod, [(tim, bcat) for tim in tims])
        tnow = Time()
        print('[parallel srcs] Getting tractor-on-bricks model images:', tnow-tlast)
        tlast = tnow
        for tim,mod in zip(tims, mods):
            tim.data -= mod
        if not plots:
            del mods

        if plots:
            coimgs,cons = compute_coadds(tims, bands, targetwcs, images=mods)
            plt.clf()
            dimshow(get_rgb(coimgs, bands))
            plt.title('Marginal sources subtracted off')
            ps.savefig()
            coimgs,cons = compute_coadds(tims, bands, targetwcs)
            plt.clf()
            dimshow(get_rgb(coimgs, bands))
            plt.title('After subtracting off marginal sources')
            ps.savefig()


    if Tsdss is not None:
        avoid_xy.extend(zip(Tsdss.itx, Tsdss.ity))

    if plots and False:
        for tim in tims:
            ivx = tim.imobj.read_invvar(clip=False, slice=tim.slice)
            plt.clf()
            plt.subplot(2,2,1)
            dimshow(tim.getInvvar(), vmin=-0.1 / tim.sig1**2,
                    vmax=2. / tim.sig1**2, ticks=False)
            plt.title('Tim invvar')
            plt.subplot(2,2,2)
            dimshow(tim.getInvvar() == 0, vmin=0, vmax=1, ticks=False)
            plt.title('Tim invvar == 0')
            plt.subplot(2,2,3)
            dimshow(ivx * tim.zpscale**2, vmin=-0.1 / tim.sig1**2,
                    vmax=2. / tim.sig1**2, ticks=False)
            plt.title('Orig invvar')
            plt.subplot(2,2,4)
            dimshow(np.logical_or(ivx == 0, tim.dq != 0),
                    vmin=0, vmax=1, ticks=False)
            plt.title('Orig invvar == 0 or DQ')
            #dimshow(tim.dq != 0, vmin=0, vmax=1, ticks=False)
            #plt.title('DQ')
            plt.suptitle('Tim ' + tim.name)
            ps.savefig()

        for tim in tims:
            plt.clf()
            plt.subplot(2,2,1)
            dimshow(tim.getInvvar(), vmin=-0.1 / tim.sig1**2,
                    vmax=2. / tim.sig1**2, ticks=False)
            h,w = tim.shape
            rgba = np.zeros((h,w,4), np.uint8)
            rgba[:,:,0][tim.inverr == 0] = 255
            rgba[:,:,3][tim.inverr == 0] = 255
            dimshow(rgba)
            plt.title('Tim invvar')
            plt.subplot(2,2,2)
            print('Image max:', tim.getImage().max(), 'satval', tim.satval)
            mx = max(tim.getImage().max(), tim.satval) * 1.1
            ima = dict(vmin=-2.*tim.sig1, vmax=mx,
                       ticks=False)
            dimshow(tim.getImage(), **ima)
            plt.title('Tim image')
            plt.subplot(2,2,3)
            dimshow((tim.dq & tim.dq_bits['satur'] > 0), vmin=0, vmax=1,
                    ticks=False)
            plt.title('SATUR')
            plt.subplot(2,2,4)
            img = tim.getImage().copy()
            img[(tim.dq & tim.dq_bits['satur']) > 0] = tim.satval
            dimshow(img, **ima)
            plt.title('Patched image')
            plt.suptitle('Tim ' + tim.name)
            ps.savefig()

    print('Rendering detection maps...')
    detmaps, detivs, satmap = detection_maps(tims, targetwcs, bands, mp)
    tnow = Time()
    print('[parallel srcs] Detmaps:', tnow-tlast)

    tlast = tnow
    # Median-smooth detection maps
    binning = 4
    smoos = mp.map(_median_smooth_detmap,
                   [(m,iv,binning) for m,iv in zip(detmaps, detivs)])
    tnow = Time()
    print('[parallel srcs] Median-filter detmaps:', tnow-tlast)
    tlast = tnow

    for i,(detmap,detiv,smoo) in enumerate(zip(detmaps, detivs, smoos)):
        # Subtract binned median image.
        S = binning
        for ii in range(S):
            for jj in range(S):
                sh,sw = detmap[ii::S, jj::S].shape
                detmap[ii::S, jj::S] -= smoo[:sh,:sw]

        if plots:
            sig1 = 1./np.sqrt(np.median(detiv[detiv > 0]))
            kwa = dict(vmin=-2.*sig1, vmax=10.*sig1)
            kwa2 = dict(vmin=-2.*sig1, vmax=50.*sig1)

            subbed = detmap.copy()
            S = binning
            for ii in range(S):
                for jj in range(S):
                    subbed[ii::S, jj::S] -= smoo

            plt.clf()
            plt.subplot(2,3,1)
            dimshow(detmap, **kwa)
            plt.subplot(2,3,2)
            dimshow(smoo, **kwa)
            plt.subplot(2,3,3)
            dimshow(subbed, **kwa)
            plt.subplot(2,3,4)
            dimshow(detmap, **kwa2)
            plt.subplot(2,3,5)
            dimshow(smoo, **kwa2)
            plt.subplot(2,3,6)
            dimshow(subbed, **kwa2)
            plt.suptitle('Median filter of detection map: %s band' %
                         bands[i])
            ps.savefig()

    # Handle the margin of interpolated (masked) pixels around
    # saturated pixels
    from scipy.ndimage.morphology import binary_dilation

    saturated_pix = binary_dilation(satmap > 0, iterations=10)

    if Tsdss is not None:
        #saturated_pix = np.zeros(detmaps[0].shape, bool)
        for band,detmap,detiv in zip(bands, detmaps, detivs):
            I = np.flatnonzero(detiv[Tsdss.ity, Tsdss.itx] == 0.)
            print(len(I), 'SDSS sources have detiv = 0 in band', band)
            if len(I) == 0:
                continue

            from scipy.ndimage.morphology import binary_propagation

            # Set the central pixel of the detmap
            saturated_pix[Tsdss.ity[I], Tsdss.itx[I]] = True
            # Spread the True pixels wherever detiv==0
            binary_propagation(saturated_pix, mask=(detiv == 0),
                               output=saturated_pix)

    # Saturated blobs -- create a source for each?!
    from scipy.ndimage.measurements import label, find_objects, center_of_mass

    satblobs,nsat = label(satmap > 0)
    satyx = center_of_mass(satmap, labels=satblobs, index=np.arange(nsat)+1)
    # NOTE, satyx is in y,x order (center_of_mass)
    satx = np.array([x for y,x in satyx]).astype(int)
    saty = np.array([y for y,x in satyx]).astype(int)
    del satyx

    if len(satx):
        avoid_xy.extend(zip(satx, saty))

        Tsat = fits_table()
        Tsat.tx = Tsat.itx = satx
        Tsat.ty = Tsat.ity = saty
        Tsat.ra,Tsat.dec = targetwcs.pixelxy2radec(satx+1, saty+1)

        satcat = []
        for r,d in zip(Tsat.ra, Tsat.dec):
            # ??!
            fluxes = dict([(band, 1.) for band in bands])
            satcat.append(PointSource(RaDecPos(r, d),
                                      NanoMaggies(order=bands, **fluxes)))

    else:
        Tsat = None
        satcat = []

    if plots:
        plt.clf()
        dimshow(satmap)
        plt.title('satmap')
        ps.savefig()

        rgb = get_rgb(detmaps, bands)
        plt.clf()
        dimshow(rgb)
        plt.title('detmaps')
        ps.savefig()

        print('rgb', rgb.dtype)
        rgb[:,:,0][saturated_pix] = 0
        rgb[:,:,1][saturated_pix] = 1
        rgb[:,:,2][saturated_pix] = 0
        plt.clf()
        dimshow(rgb)
        ax = plt.axis()
        plt.plot(satx, saty, 'ro')
        plt.axis(ax)
        plt.title('detmaps & saturated')
        ps.savefig()

    # SED-matched detections
    print('Running source detection at', nsigma, 'sigma')
    if len(avoid_xy) == 0:
        avoid_xy = None
    else:
        avoid_xy = np.vstack(avoid_xy)
        avoid_xy = avoid_xy[:,0], avoid_xy[:,1]
    SEDs = sed_matched_filters(bands)
    Tnew,newcat,hot = run_sed_matched_filters(
        SEDs, bands, detmaps, detivs, avoid_xy, targetwcs,
        nsigma=nsigma, saturated_pix=saturated_pix, plots=plots, ps=ps, mp=mp)

    peaksn = Tnew.peaksn
    apsn = Tnew.apsn
    peakx,peaky = Tnew.tx, Tnew.ty

    Tnew.delete_column('peaksn')
    Tnew.delete_column('apsn')

    TT = []
    cats = []
    if Tsdss is not None:
        TT.append(Tsdss)
        cats.extend(sdsscat)
    if Tsat is not None:
        TT.append(Tsat)
        cats.extend(satcat)
    TT.append(Tnew)
    cats.extend(newcat)

    T = merge_tables(TT, columns='fillzero')
    cat = Catalog(*cats)
    del TT
    del cats

    if pipe:
        del detmaps
        del detivs

    tnow = Time()
    print('[serial srcs] Peaks:', tnow-tlast)
    tlast = tnow

    if plots:
        if False and not plots:
            plt.figure(figsize=(18,18))
            plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.95,
                                hspace=0.2, wspace=0.05)
            if outdir is None:
                outdir = '.'
            outdir = os.path.join(outdir, 'metrics', brickname[:3])
            trymakedirs(outdir)
            fn = os.path.join(outdir, 'sources-%s' % brickname)
            ps = PlotSequence(fn)

        if coimgs is None:
            coimgs,cons = compute_coadds(tims, bands, targetwcs)
        crossa = dict(ms=10, mew=1.5)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        plt.title('Catalog + SED-matched detections')
        ps.savefig()

        ax = plt.axis()
        p1 = plt.plot(T.tx, T.ty, 'r+', **crossa)
        p2 = plt.plot(peakx, peaky, '+', color=(0,1,0), **crossa)
        plt.axis(ax)
        plt.title('Catalog + SED-matched detections')
        plt.figlegend((p1[0], p2[0]), ('SDSS', 'New'), 'upper left')
        ps.savefig()
        # for x,y in zip(peakx,peaky):
        #     plt.text(x+5, y, '%.1f' % (hot[np.clip(int(y),0,H-1),
        #                                  np.clip(int(x),0,W-1)]), color='r',
        #              ha='left', va='center')
        # ps.savefig()

    # Segment, and record which sources fall into each blob
    blobs,blobsrcs,blobslices = segment_and_group_sources(hot, T, name=brickname,
                                                          ps=ps, plots=plots)
    del hot

    for i,Isrcs in enumerate(blobsrcs):
        if not (Isrcs.dtype in [int, np.int64]):
            print('Isrcs dtype', Isrcs.dtype)
            print('i:', i)
            print('Isrcs:', Isrcs)
            print('blobslice:', blobslices[i])

    cat.freezeAllParams()

    tnow = Time()
    print('[serial srcs] Blobs:', tnow-tlast)
    tlast = tnow

    keys = ['T', 'tims',
            'blobsrcs', 'blobslices', 'blobs',
            'cat', 'ps']
    if not pipe:
        keys.extend(['detmaps', 'detivs'])

    if False:
        for b,detmap,detiv in zip(bands, detmaps, detivs):
            fitsio.write('detmap-%s-%s.fits' % (brickname, b),
                         detmap)
            fitsio.write('detiv-%s-%s.fits' % (brickname, b),
                         detiv)
            fitsio.write('detsn-%s-%s.fits' % (brickname, b),
                         detmap * np.sqrt(detiv))

    rtn = dict()
    for k in keys:
        rtn[k] = locals()[k]
    return rtn

def set_source_radii(bands, tims, cat, minsigma, minradius=3):
    # FIXME -- set source radii crudely, based on the maximum of the
    # PSF profiles in all images (!) -- should have a source x image
    # structure -- *and* based on SDSS fluxes.
    profiles = []
    R = 100
    minsig1s = dict([(band,1e100) for band in bands])
    for tim in tims:
        minsig1s[tim.band] = min(minsig1s[tim.band], tim.sig1)
        th,tw = tim.shape

        if hasattr(tim.psf, 'getMixtureOfGaussians'):
            mog = tim.psf.getMixtureOfGaussians(px=tw/2., py=th/2.)
            profiles.extend([
                    mog.evaluate_grid(0, R, 0, 1, 0., 0.).patch.ravel(),
                    mog.evaluate_grid(-(R-1), 1, 0, 1, 0., 0.).patch.ravel()[-1::-1],
                    mog.evaluate_grid(0, 1, 0, R, 0., 0.).patch.ravel(),
                    mog.evaluate_grid(0, 1, -(R-1), 1, 0., 0.).patch.ravel()[-1::-1]])
        else:
            patch = tim.psf.getPointSourcePatch(px=tw/2., py=th/2.)
            ph,pw = patch.shape
            profiles.extend([
                    patch.patch[ph/2    , pw/2:   ],
                    patch.patch[ph/2    , pw/2::-1],
                    patch.patch[ph/2:   , pw/2    ],
                    patch.patch[ph/2::-1, pw/2    ],
                    ])

    profiles = np.array(profiles)

    minradius = 3
    pro = np.max(profiles, axis=0)
    for src in cat:
        if not isinstance(src, PointSource):
            continue
        nsigmas = 0.
        bright = src.getBrightness()
        for band in bands:
            nsigmas = max(nsigmas, bright.getFlux(band) / minsig1s[band])
        if nsigmas <= 0:
            continue
        ii = np.flatnonzero(pro > (minsigma / nsigmas))
        if len(ii) == 0:
            continue
        src.fixedRadius = max(minradius, 1 + ii[-1])

def _write_fitblobs_pickle(fn, data):
    from astrometry.util.file import pickle_to_file
    tmpfn = fn + '.tmp'
    pickle_to_file(data, tmpfn)
    os.rename(tmpfn, fn)
    print('Wrote', fn)
        
def stage_fitblobs(T=None,
                   blobsrcs=None, blobslices=None, blobs=None,
                   cat=None, targetrd=None, pixscale=None,
                   targetwcs=None,
                   W=None,H=None,
                   bands=None, ps=None, tims=None,
                   decals=None,
                   plots=False, plots2=False,
                   nblobs=None, blob0=None, blobxy=None,
                   simul_opt=False, use_ceres=True, mp=None,
                   checkpoint_filename=None,
                   checkpoint_period=600,
                   write_pickle_filename=None,
                   **kwargs):
    '''
    This is where the actual source fitting happens.
    The `_one_blob` function is called for each "blob" of pixels with
    the sources contained within that blob.  The results are assembled
    in the next stage, `stage_fitblobs_finish`.
    '''
    tlast = Time()
    for tim in tims:
        assert(np.all(np.isfinite(tim.getInvError())))

    if write_pickle_filename is not None:
        #import multiprocessing
        #write_pool = multiprocessing.Pool(1)
        import threading
        keys = ['T', 'blobsrcs', 'blobslices', 'blobs', 'cat', 'targetrd',
                'pixscale', 'targetwcs', 'W', 'H', 'bands', 'tims', 'decals',]
        L = locals()
        vals = dict([(k, L[k]) for k in keys])
        write_thread = threading.Thread(
            target=_write_fitblobs_pickle,
            args=(write_pickle_filename, vals), name='write_pickle')
        print('Starting thread to write fitblobs pickle')
        write_thread.start()
        
    # How far down to render model profiles
    minsigma = 0.1
    for tim in tims:
        tim.modelMinval = minsigma * tim.sig1

    set_source_radii(bands, tims, cat, minsigma)

    if plots and False:
        coimgs,cons = compute_coadds(tims, bands, targetwcs)
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
    print('[serial fitblobs]:', tnow-tlast)
    tlast = tnow

    keepblobs = None

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

    if blob0 is not None or (nblobs is not None and nblobs < len(blobslices)):
        if blob0 is None:
            blob0 = 0
        if nblobs is None:
            nblobs = len(blobslices) - blob0
        keepblobs = np.arange(blob0, blob0+nblobs)

    if keepblobs is not None:
        # 'blobs' is an image with values -1 for no blob, or the index of the
        # blob.  Create a map from old 'blobs+1' to new 'blobs+1'.  +1 so that
        # -1 is a valid index.
        NB = len(blobslices)
        blobmap = np.empty(NB+1, int)
        blobmap[:] = -1
        blobmap[keepblobs + 1] = np.arange(len(keepblobs))
        # apply the map!
        blobs = blobmap[blobs + 1]

        # 'blobslices' and 'blobsrcs' are lists
        blobslices = [blobslices[i] for i in keepblobs]
        blobsrcs   = [blobsrcs  [i] for i in keepblobs]

        # one more place where blob numbers are recorded...
        T.blob = blobs[T.ity, T.itx]

    # drop any cached data before we start pickling/multiprocessing
    decals.drop_cache()

    tycho = fits_table('tycho2.fits')
    print('Read', len(tycho), 'Tycho-2 stars')
    ok,tx,ty = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
    margin = 100
    tycho.cut(ok * (tx > -margin) * (tx < W+margin) *
              (ty > -margin) * (ty < H+margin))
    print('Cut to', len(tycho), 'Tycho-2 stars within brick')
    del ok,tx,ty

    if plots:
        ok,tx,ty = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
        plt.clf()
        dimshow(blobs>=0, vmin=0, vmax=1)
        ax = plt.axis()
        plt.plot(tx-1, ty-1, 'ro')
        for x,y,mb,mv,mh in zip(tx,ty,tycho.mag_bt,tycho.mag_vt,tycho.mag_hp):
            plt.text(x, y, '%.1f/%.1f/%.1f' % (mb,mv,mh),
                     color='r', fontsize=10,
                     bbox=dict(facecolor='w', alpha=0.5))
        plt.axis(ax)
        plt.title('Tycho-2 stars')
        ps.savefig()

    if checkpoint_filename is None:
        blobiter = _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims,
                              cat, bands, plots, ps, simul_opt, use_ceres,
                              tycho)
        # to allow timingpool to queue tasks one at a time
        blobiter = iterwrapper(blobiter, len(blobsrcs))
        R = mp.map(_bounce_one_blob, blobiter)
    else:
        from astrometry.util.ttime import CpuMeas

        # Check for existing checkpoint file.
        R = []
        if os.path.exists(checkpoint_filename):
            from astrometry.util.file import unpickle_from_file
            print('Reading', checkpoint_filename)
            try:
                R = unpickle_from_file(checkpoint_filename)
                print('Read', len(R), 'results from checkpoint file',
                      checkpoint_filename)
            except:
                import traceback
                print('Failed to read checkpoint file ' + checkpoint_filename)
                traceback.print_exc()
                R = []

        skipblobs = [B.iblob for B in R]
        blobiter = _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims,
                              cat, bands, plots, ps, simul_opt, use_ceres,
                              tycho, skipblobs=skipblobs)
        # to allow timingpool to queue tasks one at a time
        blobiter = iterwrapper(blobiter, len(blobsrcs))

        Riter = mp.imap_unordered(_bounce_one_blob, blobiter)
        # we'll actually measure wall time -- CpuMeas is just mis-named
        last_checkpoint = CpuMeas()
        while True:
            import multiprocessing
            from astrometry.util.file import pickle_to_file

            d = os.path.dirname(checkpoint_filename)
            if len(d) and not os.path.exists(d):
                trymakedirs(d)
            
            tnow = CpuMeas()
            dt = tnow.wall_seconds_since(last_checkpoint)
            if dt >= checkpoint_period:
                # Write checkpoint!
                fn = checkpoint_filename + '.tmp'
                # (this happens out here in the main process, while the worker
                # processes continue in the background.)
                pickle_to_file(R, fn)
                os.rename(fn, checkpoint_filename)
                print('Wrote checkpoint to', checkpoint_filename)
                last_checkpoint = tnow
                dt = 0.
            try:
                if mp.is_multiproc():
                    timeout = max(1, checkpoint_period - dt)
                    r = Riter.next(timeout)
                else:
                    r = Riter.next()
                R.append(r)
            except StopIteration:
                print('Done')
                break
            except multiprocessing.TimeoutError:
                print('Timed out waiting for result')
                continue


    print('[parallel fitblobs] Fitting sources took:', Time()-tlast)

    return dict(fitblobs_R=R, tims=tims, ps=ps, blobs=blobs,
                blobslices=blobslices, blobsrcs=blobsrcs)

def stage_fitblobs_finish(
    brickname=None,
    brickid=None,
    version_header=None,
    T=None, blobsrcs=None, blobslices=None, blobs=None,
    cat=None, targetrd=None, pixscale=None,
    targetwcs=None,
    W=None,H=None,
    bands=None, ps=None, tims=None,
    plots=False, plots2=False,
    fitblobs_R=None,
    outdir=None,
    write_metrics=True,
    get_all_models=False,
    allbands = 'ugrizY',
    **kwargs):
    '''
    This is a "glue" stage to repackage the results from the
    `stage_fitblobs` stage.
    '''

    # one_blob can reduce the number and change the types of sources!
    # Reorder the sources here...
    R = fitblobs_R
    del fitblobs_R
    assert(len(R) == len(blobsrcs))

    # Drop now-empty blobs.
    R = [r for r in R if r is not None and len(r)]
    if len(R) > 0:
        J = np.argsort([B.iblob for B in R])
        R = [R[j] for j in J]
        BB = merge_tables(R)
    else:
        BB = fits_table()
        BB.Isrcs = []
        BB.sources = []
        nb = len(bands)
        BB.fracflux   = np.zeros((0,nb))
        BB.fracmasked = np.zeros((0,nb))
        BB.fracin     = np.zeros((0,nb))
        BB.rchi2      = np.zeros((0,nb))
        BB.dchisqs    = np.zeros((0,5))
        BB.flags      = np.zeros((0,5))
        BB.all_models = np.array([])
        BB.all_model_flags   = np.array([])
        BB.all_model_fluxivs = np.array([])
        BB.started_in_blob  = []
        BB.finished_in_blob = []
        BB.hastycho         = []
        BB.srcinvvars = [np.zeros((0,))]
    del R
    II = BB.Isrcs
    newcat = BB.sources
    T.cut(II)

    assert(len(T) == len(newcat))
    print('Old catalog:', len(cat))
    print('New catalog:', len(newcat))
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
    ns,nb = BB.rchi2.shape
    assert(ns == len(cat))
    assert(nb == len(bands))
    ns,nb = BB.dchisqs.shape
    assert(ns == len(cat))
    assert(nb == 5) # ptsrc, simple, dev, exp, comp
    assert(len(BB.flags) == len(cat))

    # Renumber blobs to make them contiguous.
    ublob,iblob = np.unique(T.blob, return_inverse=True)
    del ublob
    assert(len(iblob) == len(T))
    T.blob = iblob.astype(np.int32)

    # write out blob map
    if write_metrics:
        if outdir is None:
            outdir = '.'
        metricsdir = os.path.join(outdir, 'metrics', brickname[:3])
        trymakedirs(metricsdir)
        fn = os.path.join(metricsdir, 'blobs-%s.fits.gz' % brickname)

        # Build map from (old+1) to new blob numbers, for the blob image.
        blobmap = np.empty(blobs.max()+2, int)
        # make sure that dropped blobs -> -1
        blobmap[:] = -1
        # in particular,
        blobmap[0] = -1
        blobmap[T.blob + 1] = iblob
        blobs = blobmap[blobs+1]

        fitsio.write(fn, blobs, header=version_header, clobber=True)
        print('Wrote', fn)
        del blobmap
    del iblob
    blobs = None
    
    T.brickid   = np.zeros(len(T), np.int32) + brickid
    T.brickname = np.array([brickname] * len(T))
    T.objid     = np.arange(len(T)).astype(np.int32)

    # How many sources in each blob?
    from collections import Counter
    ninblob = Counter(T.blob)
    T.ninblob = np.array([ninblob[b] for b in T.blob]).astype(np.int16)
    T.tycho2inblob = BB.hastycho
    T.dchisq       = BB.dchisqs.astype(np.float32)

    if write_metrics or get_all_models:
        from desi_common import prepare_fits_catalog, fits_typemap
        from astrometry.util.file import pickle_to_file

        TT = fits_table()
        # Copy only desired columns...
        for k in ['blob', 'brickid', 'brickname', 'dchisq', 'objid']:
            TT.set(k, T.get(k))
        TT.type = np.array([fits_typemap[type(src)] for src in newcat])

        hdr = fitsio.FITSHDR()
        for srctype in ['ptsrc', 'simple', 'dev','exp','comp']:
            xcat = Catalog(*[m.get(srctype,None) for m in BB.all_models])
            # Convert shapes to EllipseE types
            if srctype in ['dev','exp']:
                for src in xcat:
                    if src is None:
                        continue
                    src.shape = src.shape.toEllipseE()
            elif srctype == 'comp':
                for src in xcat:
                    if src is None:
                        continue
                    src.shapeDev = src.shapeDev.toEllipseE()
                    src.shapeExp = src.shapeExp.toEllipseE()
                    src.fracDev = FracDev(src.fracDev.clipped())

            xcat.thawAllRecursive()

            namemap = dict(ptsrc='psf', simple='simp')
            prefix = namemap.get(srctype,srctype)

            TT,hdr = prepare_fits_catalog(xcat, None, TT, hdr, bands, None,
                                          allbands=allbands, prefix=prefix+'_',
                                          save_invvars=False)
            TT.set('%s_flags' % prefix,
                   np.array([m.get(srctype,0)
                             for m in BB.all_model_flags]))
            fluxivs = np.zeros((len(TT), len(allbands)), np.float32)
            bandmap = np.array([allbands.index(b) for b in bands])
            for j in range(len(xcat)):
                f = BB.all_model_fluxivs[j].get(srctype, None)
                if f is None:
                    continue
                fluxivs[j, bandmap] = f
            TT.set('%s_decam_flux_ivar' % prefix, fluxivs)

        TT.delete_column('psf_shapeExp')
        TT.delete_column('psf_shapeDev')
        TT.delete_column('psf_fracDev')
        TT.delete_column('psf_type')
        TT.delete_column('simp_shapeExp')
        TT.delete_column('simp_shapeDev')
        TT.delete_column('simp_fracDev')
        TT.delete_column('simp_type')
        TT.delete_column('dev_shapeExp')
        TT.delete_column('dev_fracDev')
        TT.delete_column('dev_type')
        TT.delete_column('exp_shapeDev')
        TT.delete_column('exp_fracDev')
        TT.delete_column('exp_type')
        TT.delete_column('comp_type')
        # Unpack ellipses
        TT.dev_shape_r  = TT.dev_shapeDev[:,0]
        TT.dev_shape_e1 = TT.dev_shapeDev[:,1]
        TT.dev_shape_e2 = TT.dev_shapeDev[:,2]
        TT.exp_shape_r  = TT.exp_shapeExp[:,0]
        TT.exp_shape_e1 = TT.exp_shapeExp[:,1]
        TT.exp_shape_e2 = TT.exp_shapeExp[:,2]
        TT.comp_shapeDev_r  = TT.comp_shapeDev[:,0]
        TT.comp_shapeDev_e1 = TT.comp_shapeDev[:,1]
        TT.comp_shapeDev_e2 = TT.comp_shapeDev[:,2]
        TT.comp_shapeExp_r  = TT.comp_shapeExp[:,0]
        TT.comp_shapeExp_e1 = TT.comp_shapeExp[:,1]
        TT.comp_shapeExp_e2 = TT.comp_shapeExp[:,2]
        TT.delete_column('dev_shapeDev')
        TT.delete_column('exp_shapeExp')
        TT.delete_column('comp_shapeDev')
        TT.delete_column('comp_shapeExp')

        if get_all_models:
            all_models = TT
        if write_metrics:
            primhdr = fitsio.FITSHDR()
            for r in version_header.records():
                primhdr.add_record(r)
                primhdr.add_record(dict(name='ALLBANDS', value=allbands,
                                        comment='Band order in array values'))
                primhdr.add_record(dict(name='PRODTYPE', value='catalog',
                                        comment='NOAO data product type'))
            fn = os.path.join(metricsdir, 'all-models-%s.fits' % brickname)
            TT.writeto(fn, header=hdr)
            del TT
            print('Wrote', fn)

    T.decam_flags = BB.flags
    T.fracflux    = BB.fracflux
    T.fracin      = BB.fracin
    T.fracmasked  = BB.fracmasked
    T.rchi2       = BB.rchi2
    T.left_blob   = np.logical_and(BB.started_in_blob,
                                   np.logical_not(BB.finished_in_blob))
    invvars = np.hstack(BB.srcinvvars)
    assert(cat.numberOfParams() == len(invvars))

    rtn = dict(fitblobs_R = None)
    for k in ['cat', 'invvars', 'T', 'allbands', 'blobs']:
        rtn[k] = locals()[k]
    if get_all_models:
        rtn['all_models'] = all_models
    return rtn

def _blob_iter(blobslices, blobsrcs, blobs, targetwcs, tims, cat, bands,
               plots, ps, simul_opt, use_ceres, tycho, skipblobs=[]):

    ok,tx,ty = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
    tx = np.round(tx-1).astype(int)
    ty = np.round(ty-1).astype(int)
    # FIXME -- Cut or clip to targetwcs region?
    H,W = targetwcs.shape
    tx = np.clip(tx, 0, W-1)
    ty = np.clip(ty, 0, H-1)
    tychoblobs = set(blobs[ty, tx])
    # Remove -1 = no blob from set; not strictly necessary, just cosmetic
    try:
        tychoblobs.remove(-1)
    except:
        pass
    print('Blobs containing Tycho-2 stars:', tychoblobs)

    # sort blobs by size so that larger ones start running first
    from collections import Counter
    blobvals = Counter(blobs[blobs>=0])
    blob_order = np.array([i for i,npix in blobvals.most_common()])

    for nblob,iblob in enumerate(blob_order):
        if iblob in skipblobs:
            print('Skipping blob', iblob)
            continue

        bslc  = blobslices[iblob]
        Isrcs = blobsrcs  [iblob]
        assert(len(Isrcs) > 0)

        tblob = Time()
        # blob bbox in target coords
        sy,sx = bslc
        by0,by1 = sy.start, sy.stop
        bx0,bx1 = sx.start, sx.stop
        blobh,blobw = by1 - by0, bx1 - bx0

        # Here we assume the "blobs" array has been remapped so that
        # -1 means "no blob", while 0 and up label the blobs, thus
        # iblob equals the value in the "blobs" map.
        blobmask = (blobs[bslc] == iblob)

        # find one pixel within the blob, for debugging purposes
        onex = oney = None
        for y in range(by0, by1):
            ii = np.flatnonzero(blobmask[y-by0,:])
            if len(ii) == 0:
                continue
            onex = bx0 + ii[0]
            oney = y
            break

        hastycho = iblob in tychoblobs

        print('Blob', nblob+1, 'of', len(blobslices), ': blob', iblob,
              len(Isrcs), 'sources, size', blobw, 'x', blobh,
              'center', (bx0+bx1)/2, (by0+by1)/2, 'npix', np.sum(blobmask),
              'one pixel:', onex,oney, 'has Tycho-2 star:', hastycho)

        rr,dd = targetwcs.pixelxy2radec([bx0,bx0,bx1,bx1],[by0,by1,by1,by0])
        subtimargs = []
        for itim,tim in enumerate(tims):
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
            # in the _one_blob code.
            subsky = tim.getSky().shifted(sx0, sy0)
            tim.imobj.psfnorm = tim.psfnorm
            tim.imobj.galnorm = tim.galnorm
            # FIXME -- maybe the cache is worth sending?
            tim.psf.clear_cache()
            subtimargs.append((subimg, subie, subwcs, tim.subwcs,
                               tim.getPhotoCal(),
                               subsky, tim.psf, tim.name, sx0, sx1, sy0, sy1,
                               tim.band, tim.sig1, tim.modelMinval,
                               tim.imobj))

        yield (nblob, iblob, Isrcs, targetwcs, bx0, by0, blobw, blobh,
               blobmask, subtimargs, [cat[i] for i in Isrcs], bands, plots, ps,
               simul_opt, use_ceres, hastycho)

def _bounce_one_blob(X):
    # iblob = X[0]
    # fn = 'blob-%i.pickle' % iblob
    # from astrometry.util.file import pickle_to_file
    # pickle_to_file(X, fn)
    # print('Wrote', fn)

    try:
        return _one_blob(X)
    except:
        import traceback
        print('Exception in _one_blob: (iblob = %i)' % (X[0]))
        traceback.print_exc()
        #print 'CARRYING ON...'
        #return None
        raise

def _clip_model_to_blob(mod, sh, ie):
    '''
    mod: Patch
    sh: tim shape
    ie: tim invError
    Returns: new Patch
    '''
    mslc,islc = mod.getSlices(sh)
    sy,sx = mslc
    mod = Patch(mod.x0 + sx.start, mod.y0 + sy.start,
                mod.patch[mslc] * (ie[islc]>0))
    return mod

FLAG_CPU_A   = 1
FLAG_STEPS_A = 2
FLAG_CPU_B   = 4
FLAG_STEPS_B = 8
FLAG_TRIED_C = 0x10
FLAG_CPU_C   = 0x20

def _select_model(chisqs, nparams, galaxy_margin):
    '''
    Returns keepmod
    '''
    keepmod = 'none'

    # This is our "detection threshold": 5-sigma in
    # *parameter-penalized* units; ie, ~5.2-sigma for point sources
    cut = 5.**2
    # Take the best of all models computed
    diff = max([chisqs[name] - nparams[name] for name in chisqs.keys()
                if name != 'none'])

    if diff < cut:
        return keepmod

    # We're going to keep this source!
    if chisqs['ptsrc'] > chisqs['simple']:
        #print('Keeping source; PTSRC is better than SIMPLE')
        keepmod = 'ptsrc'
    else:
        #print('Keeping source; SIMPLE is better than PTSRC')
        keepmod = 'simple'

    if not 'exp' in chisqs:
        return keepmod

    # This is our "upgrade" threshold: how much better a galaxy
    # fit has to be versus ptsrc, and comp versus galaxy.
    cut = galaxy_margin

    # This is the "fractional" upgrade threshold for ptsrc/simple->dev/exp:
    # 2% of ptsrc vs nothing
    fcut = 0.02 * chisqs['ptsrc']
    #print('Cut: max of', cut, 'and', fcut, ' (fraction of chisq_psf=%.1f)'
    # % chisqs['ptsrc'])
    cut = max(cut, fcut)

    expdiff = chisqs['exp'] - chisqs[keepmod]
    devdiff = chisqs['dev'] - chisqs[keepmod]

    #print('EXP vs', keepmod, ':', expdiff)
    #print('DEV vs', keepmod, ':', devdiff)

    if not (expdiff > cut or devdiff > cut):
        #print('Keeping', keepmod)
        return keepmod

    if expdiff > devdiff:
        #print('Upgrading from PTSRC to EXP: diff', expdiff)
        keepmod = 'exp'
    else:
        #print('Upgrading from PTSRC to DEV: diff', expdiff)
        keepmod = 'dev'

    if not 'comp' in chisqs:
        return keepmod

    diff = chisqs['comp'] - chisqs[keepmod]
    #print('Comparing', keepmod, 'to comp.  cut:', cut, 'comp:', diff)
    if diff < cut:
        return keepmod

    #print('Upgrading from dev/exp to composite.')
    keepmod = 'comp'
    return keepmod


def _chisq_improvement(src, chisqs, chisqs_none):
    '''
    chisqs, chisqs_none: dict of band->chisq
    '''
    bright = src.getBrightness()
    bands = chisqs.keys()
    fluxes = dict([(b, bright.getFlux(b)) for b in bands])
    dchisq = 0.
    for b in bands:
        flux = fluxes[b]
        if flux == 0:
            continue
        # this will be positive for an improved model
        d = chisqs_none[b] - chisqs[b]
        if flux > 0:
            dchisq += d
        else:
            dchisq -= np.abs(d)
    return dchisq

def _per_band_chisqs(srctractor, bands):
    chisqs = dict([(b,0) for b in bands])
    for img in srctractor.images:
        chi = srctractor.getChiImage(img=img)
        chisqs[img.band] = chisqs[img.band] + (chi ** 2).sum()
    return chisqs

def _fit_fluxes(cat, tims, bands, use_ceres, alphas):
    # Try fitting fluxes first?
    cat.thawAllRecursive()
    for src in cat:
        src.freezeAllBut('brightness')
    for b in bands:
        for src in cat:
            src.getBrightness().freezeAllBut(b)
        # Images for this band
        btims = [tim for tim in tims if tim.band == b]

        btr = Tractor(btims, cat)
        btr.freezeParam('images')
        done = False
        if use_ceres:
            from tractor.ceres_optimizer import CeresOptimizer
            orig_opt = btr.optimizer
            btr.optimizer = CeresOptimizer(BW=8, BH=8)
            try:
                btr.optimize_forced_photometry(shared_params=False,
                                               wantims=False)
                done = True
            except:
                import traceback
                print('Warning: Optimize_forced_photometry with Ceres failed:')
                traceback.print_exc()
                print('Falling back to LSQR')
            btr.optimizer = orig_opt
        if not done:
            try:
                btr.optimize_forced_photometry(
                    alphas=alphas, shared_params=False, wantims=False)
            except:
                import traceback
                print('Warning: Optimize_forced_photometry failed:')
                traceback.print_exc()
                # carry on


class SourceModels(object):
    '''
    This class maintains a list of the model patches for a set of sources
    in a set of images.
    '''
    def save_images(self, tims):
        self.orig_images = [tim.getImage() for tim in tims]
        for tim,img in zip(tims, self.orig_images):
            tim.data = img.copy()

    def restore_images(self, tims):
        for tim,img in zip(tims, self.orig_images):
            tim.data = img

    def create(self, tims, srcs, subtract=False):
        '''
        Note that this modifies the *tims* if subtract=True.
        '''
        self.models = []
        for tim in tims:
            mods = []
            sh = tim.shape
            ie = tim.getInvError()
            for src in srcs:
                mod = src.getModelPatch(tim)
                if mod is not None and mod.patch is not None:
                    if not np.all(np.isfinite(mod.patch)):
                        print('Non-finite mod patch')
                        print('source:', src)
                        print('tim:', tim)
                        print('PSF:', tim.getPsf())
                    assert(np.all(np.isfinite(mod.patch)))
                    mod = _clip_model_to_blob(mod, sh, ie)
                    if subtract:
                        mod.addTo(tim.getImage(), scale=-1)
                mods.append(mod)
            self.models.append(mods)

    def add(self, i, tims):
        '''
        Adds the models for source *i* back into the tims.
        '''
        for tim,mods in zip(tims, self.models):
            mod = mods[i]
            if mod is not None:
                mod.addTo(tim.getImage())

    def update_and_subtract(self, i, src, tims):
        for tim,mods in zip(tims, self.models):
            #mod = srctractor.getModelPatch(tim, src)
            if src is None:
                mod = None
            else:
                mod = src.getModelPatch(tim)
            if mod is not None:
                mod.addTo(tim.getImage(), scale=-1)
            mods[i] = mod

    def model_masks(self, i, src):
        modelMasks = []
        for mods in self.models:
            d = dict()
            modelMasks.append(d)
            mod = mods[i]
            if mod is not None:
                d[src] = Patch(mod.x0, mod.y0, mod.patch != 0)
        return modelMasks


def _one_blob(X):
    '''
    Fits sources contained within a "blob" of pixels.
    '''
    (nblob, iblob, Isrcs, brickwcs, bx0, by0, blobw, blobh, blobmask, timargs,
     srcs, bands, plots, ps, simul_opt, use_ceres, hastycho) = X

    print('Fitting blob number', nblob, 'val', iblob, ':', len(Isrcs),
          'sources, size', blobw, 'x', blobh, len(timargs), 'images')
    plots2 = False
    tlast = Time()
    alphas = [0.1, 0.3, 1.0]
    optargs = dict(priors=True, shared_params=False, alphas=alphas)
    bigblob = (blobw * blobh) > 100*100

    # 50 CCDs is over 90th percentile of bricks in DR2.
    many_exposures = len(timargs) >= 50

    blobwcs = brickwcs.get_subimage(bx0, by0, blobw, blobh)
    ok,x0,y0 = blobwcs.radec2pixelxy(
        np.array([src.getPosition().ra  for src in srcs]),
        np.array([src.getPosition().dec for src in srcs]))
    started_in_blob = blobmask[np.clip(np.round(y0-1).astype(int), 0,blobh-1),
                               np.clip(np.round(x0-1).astype(int), 0,blobw-1)]

    tims = []
    for (img, inverr, twcs, wcs, pcal, sky, psf, name, sx0, sx1, sy0, sy1,
         band, sig1, modelMinval, imobj) in timargs:

        # Mask out inverr for pixels that are not within the blob.
        subwcs = wcs.get_subimage(int(sx0), int(sy0),
                                  int(sx1-sx0), int(sy1-sy0))
        try:
            Yo,Xo,Yi,Xi,rims = resample_with_wcs(subwcs, blobwcs, [], 2)
        except OverlapError:
            #print('No overlap')
            continue
        if len(Yo) == 0:
            continue
        inverr2 = np.zeros_like(inverr)
        I = np.flatnonzero(blobmask[Yi,Xi])
        inverr2[Yo[I],Xo[I]] = inverr[Yo[I],Xo[I]]
        inverr = inverr2

        # If the subimage (blob) is small enough, instantiate a
        # constant PSF model in the center.
        if sy1-sy0 < 400 and sx1-sx0 < 400:
            subpsf = psf.constantPsfAt((sx0+sx1)/2., (sy0+sy1)/2.)
        else:
            # Otherwise, instantiate a (shifted) spatially-varying
            # PsfEx model.
            subpsf = psf.getShifted(sx0, sy0)

        tim = Image(data=img, inverr=inverr, wcs=twcs,
                    psf=subpsf, photocal=pcal, sky=sky, name=name)
        tim.band = band
        tim.sig1 = sig1
        tim.modelMinval = modelMinval
        tim.subwcs = subwcs
        tim.meta = imobj
        tims.append(tim)

        if plots:
            try:
                Yo,Xo,Yi,Xi,rims = resample_with_wcs(blobwcs, subwcs,[],2)
            except OverlapError:
                continue
            tim.resamp = (Yo, Xo, Yi, Xi)
            if False:
                plt.clf()
                plt.subplot(1,2,1)
                dimshow(img, vmin=-2.*sig1, vmax=5.*sig1)
                plt.subplot(1,2,2)
                dimshow(inverr, vmin=0, vmax=1.1/sig1)
                plt.suptitle('Subimage: ' + name)
                ps.savefig()

    if plots:
        coimgs,cons = compute_coadds(tims, bands, blobwcs, fill_holes=False)
        plt.clf()
        dimshow(get_rgb(coimgs, bands))
        ps.savefig()

        plt.clf()
        ccmap = dict(g='g', r='r', z='m')
        for tim in tims:
            chi = (tim.data * tim.inverr)[tim.inverr > 0]
            plt.hist(chi.ravel(), range=(-5,10), bins=100, histtype='step',
                     color=ccmap[tim.band])
        plt.xlabel('signal/noise per pixel')
        ps.savefig()

    cat = Catalog(*srcs)
    tr = Tractor(tims, cat)
    tr.freezeParam('images')

    #print('Tims:', [s.shape for s in tims])

    _fit_fluxes(cat, tims, bands, use_ceres, alphas)
    cat.thawAllRecursive()

    if plots:
        bslc = (slice(by0, by0+blobh), slice(bx0, bx0+blobw))
        plotmods = []
        plotmodnames = []
        plotmods.append(list(tr.getModelImages()))
        plotmodnames.append('Initial models')
        _plot_mods(tims, plotmods, plotmodnames, bands, None, None,
                   bslc, blobw, blobh, ps, chi_plots=False)

    # Optimize individual sources in order of flux
    fluxes = []
    for src in cat:
        # HACK -- here we just *sum* the nanomaggies in each band.  Bogus!
        br = src.getBrightness()
        flux = sum([br.getFlux(band) for band in bands])
        fluxes.append(flux)
    Ibright = np.argsort(-np.array(fluxes))

    if len(cat) > 1:
        # -Remember the original images
        # -Compute initial models for each source (in each tim)
        # -Subtract initial models from images
        # -During fitting, for each source:
        #   -add back in the source's initial model (to each tim)
        #   -fit, with Catalog([src])
        #   -subtract final model (from each tim)
        # -Replace original images

        models = SourceModels()
        # Remember original tim images
        models.save_images(tims)
        # Create & subtract initial models for each tim x each source
        models.create(tims, srcs, subtract=True)

        # For sources, in decreasing order of brightness
        for numi,i in enumerate(Ibright):
            #tsrc = Time()
            print('Fitting source', i, '(%i of %i in blob)' %
                  (numi, len(Ibright)))
            src = cat[i]

            # Add this source's initial model back in.
            models.add(i, tims)

            if bigblob:
                # Create super-local sub-sub-tims around this source

                # Make the subimages the same size as the modelMasks.
                #tbb0 = Time()
                mods = [mod[i] for mod in models.models]
                srctims,modelMasks = _get_subimages(tims, mods, src)
                #print('Creating srctims:', Time()-tbb0)

                if plots and (numi < 3 or numi >= len(Ibright)-3):
                    bx1 = bx0 + blobw
                    by1 = by0 + blobh
                    plt.clf()
                    coimgs,cons = compute_coadds(tims, bands, blobwcs,
                                                 fill_holes=False)
                    dimshow(get_rgb(coimgs, bands), extent=(bx0,bx1,by0,by1))
                    # plt.plot([bx0,bx0,bx1,bx1,bx0],[by0,by1,by1,by0,by0],
                    #          'r-')
                    # for tim in srctims:
                    #     h,w = tim.shape
                    #     tx,ty = [0,0,w,w,0], [0,h,h,0,0]
                    #     rd = [tim.getWcs().pixelToPosition(xi,yi)
                    #           for xi,yi in zip(tx,ty)]
                    #     ra  = [p.ra  for p in rd]
                    #     dec = [p.dec for p in rd]
                    #     ok,x,y = brickwcs.radec2pixelxy(ra, dec)
                    #     plt.plot(x, y, 'g-')
                    #
                    #     ra,dec = tim.subwcs.pixelxy2radec(tx, ty)
                    #     ok,x,y = brickwcs.radec2pixelxy(ra, dec)
                    #     plt.plot(x, y, 'm-')
                    for tim in tims:
                        h,w = tim.shape
                        tx,ty = [0,0,w,w,0], [0,h,h,0,0]
                        rd = [tim.getWcs().pixelToPosition(xi,yi)
                              for xi,yi in zip(tx,ty)]
                        ra  = [p.ra  for p in rd]
                        dec = [p.dec for p in rd]
                        ok,x,y = brickwcs.radec2pixelxy(ra, dec)
                        plt.plot(x, y, 'b-')

                        ra,dec = tim.subwcs.pixelxy2radec(tx, ty)
                        ok,x,y = brickwcs.radec2pixelxy(ra, dec)
                        plt.plot(x, y, 'c-')
                    plt.title('source %i of %i' % (numi, len(Ibright)))
                    ps.savefig()

            else:
                srctims = tims
                modelMasks = models.model_masks(i, src)

            srctractor = Tractor(srctims, [src])
            srctractor.freezeParams('images')
            srctractor.setModelMasks(modelMasks)

            if plots and False:
                spmods,spnames = [],[]
                spallmods,spallnames = [],[]
                if numi == 0:
                    spallmods.append(list(tr.getModelImages()))
                    spallnames.append('Initial (all)')
                spmods.append(list(srctractor.getModelImages()))
                spnames.append('Initial')

            # First-round optimization
            for step in range(50):
                dlnp,X,alpha = srctractor.optimize(**optargs)
                # print('dlnp:', dlnp, 'src', src)
                if dlnp < 0.1:
                    break

            if plots and False:
                spmods.append(list(srctractor.getModelImages()))
                spnames.append('Fit')
                spallmods.append(list(tr.getModelImages()))
                spallnames.append('Fit (all)')

            if plots and False:
                tims_compute_resamp(None, srctractor.getImages(), brickwcs)
                tims_compute_resamp(None, tims, brickwcs)
                plt.figure(1, figsize=(8,6))
                plt.subplots_adjust(left=0.01, right=0.99, top=0.95,
                                    bottom=0.01, hspace=0.1, wspace=0.05)
                #plt.figure(2, figsize=(3,3))
                #plt.subplots_adjust(left=0.005, right=0.995,
                #                    top=0.995,bottom=0.005)
                #_plot_mods(tims, spmods, spnames, bands, None, None, bslc,
                #           blobw, blobh, ps, chi_plots=plots2)
                plt.figure(2, figsize=(3,3.5))
                plt.subplots_adjust(left=0.005, right=0.995,
                                    top=0.88, bottom=0.005)
                plt.suptitle('Blob %i' % iblob)
                tempims = [tim.getImage() for tim in tims]

                _plot_mods(list(srctractor.getImages()), spmods, spnames,
                           bands, None, None, bslc, blobw, blobh, ps,
                           chi_plots=plots2, rgb_plots=True, main_plot=False,
                           rgb_format=('spmods Blob %i, src %i: %%s' %
                                       (iblob, i)))
                _plot_mods(tims, spallmods, spallnames, bands, None, None,
                           bslc, blobw, blobh, ps,
                           chi_plots=plots2, rgb_plots=True, main_plot=False,
                           rgb_format=('spallmods Blob %i, src %i: %%s' %
                                       (iblob, i)))

                models.restore_images(tims)
                _plot_mods(tims, spallmods, spallnames, bands, None, None,
                           bslc, blobw, blobh, ps,
                           chi_plots=plots2, rgb_plots=True, main_plot=False,
                           rgb_format='Blob %i, src %i: %%s' % (iblob, i))
                for tim,im in zip(tims, tempims):
                    tim.data = im

            # Re-remove the final fit model for this source
            models.update_and_subtract(i, src, tims)

            srctractor.setModelMasks(None)
            disable_galaxy_cache()

            #print('Fitting source took', Time()-tsrc)
            #print(src)

        models.restore_images(tims)
        del models

    else:
        # Single source (though this is coded to handle multiple sources)
        # Fit sources one at a time, but don't subtract other models
        cat.freezeAllParams()

        models = SourceModels()
        models.create(tims, [src])
        modelMasks = models.model_masks(0, src)

        tr.setModelMasks(modelMasks)
        enable_galaxy_cache()

        for numi,i in enumerate(Ibright):
            #tsrc = Time()
            #print('Fitting source', i, '(%i of %i in blob)' %
            #  (numi, len(Ibright)))
            cat.freezeAllBut(i)

            for step in range(50):
                dlnp,X,alpha = tr.optimize(**optargs)
                # print('dlnp:', dlnp)
                if dlnp < 0.1:
                    break
            #print('Fitting source took', Time()-tsrc)
            # print(cat[i])

        tr.setModelMasks(None)
        disable_galaxy_cache()

    if plots and False:
        plotmods.append(list(tr.getModelImages()))
        plotmodnames.append('Per Source')

    if len(srcs) > 1 and len(srcs) <= 10:
        #tfit = Time()
        # Optimize all at once?
        cat.thawAllParams()
        #print('Optimizing:', tr)
        # tr.printThawedParams()
        for step in range(20):
            dlnp,X,alpha = tr.optimize(**optargs)
            if dlnp < 0.1:
                break
        #print('Simultaneous fit took:', Time()-tfit)

        if plots and False:
            plotmods.append(list(tr.getModelImages()))
            plotmodnames.append('All Sources')

    if plots and False:
        _plot_mods(tims, plotmods, plotmodnames, bands, None, None,
                   bslc, blobw, blobh, ps)

    # FIXME -- for large blobs, fit strata of sources simultaneously?

    print('Blob finished fitting:', Time()-tlast)
    tlast = Time()

    # Next, model selections: point source vs dev/exp vs composite.

    # FIXME -- render initial models and find significant flux overlap
    # (product)??  (Could use the same logic above!)  This would give
    # families of sources to fit simultaneously.  (The
    # not-friends-of-friends version of blobs!)

    # We repeat the "compute & subtract initial models" logic from above.
    # -Remember the original images
    # -Compute initial models for each source (in each tim)
    # -Subtract initial models from images
    # -During fitting, for each source:
    #   -add back in the source's initial model (to each tim)
    #   -fit, with Catalog([src])
    #   -subtract final model (from each tim)
    # -Replace original images

    models = SourceModels()
    # Remember original tim images
    models.save_images(tims)
    # Create initial models for each tim x each source
    # tt = Time()
    models.create(tims, cat, subtract=True)
    # print('Subtracting initial models:', Time()-tt)

    # table of per-source measurements for this blob.
    B = fits_table()
    B.flags = np.zeros(len(Isrcs), np.uint16)
    B.dchisqs = np.zeros((len(Isrcs), 5), np.float32)
    B.sources = srcs
    B.Isrcs = Isrcs
    B.started_in_blob = started_in_blob
    B.all_models        = np.array([{} for i in range(len(Isrcs))])
    B.all_model_fluxivs = np.array([{} for i in range(len(Isrcs))])
    B.all_model_flags   = np.array([{} for i in range(len(Isrcs))])

    del srcs
    del Isrcs
    del started_in_blob

    # Model selection for sources, in decreasing order of brightness
    for numi,i in enumerate(Ibright):

        src = cat[i]
        print('Model selection for source %i of %i in blob' %
              (numi, len(Ibright)))
        #tsel = Time()

        # Add this source's initial model back in.
        models.add(i, tims)

        if bigblob:
            mods = [mod[i] for mod in models.models]
            srctims,modelMasks = _get_subimages(tims, mods, src)
            # Create a little local WCS subregion for this source, by
            # resampling non-zero inverrs from the srctims into blobwcs
            insrc = np.zeros((blobh,blobw), bool)
            for tim in srctims:
                try:
                    Yo,Xo,Yi,Xi,nil = resample_with_wcs(blobwcs, tim.subwcs,
                                                        [],2)
                except:
                    continue
                insrc[Yo,Xo] |= (tim.inverr[Yi,Xi] > 0)
            yin = np.max(insrc, axis=1)
            xin = np.max(insrc, axis=0)
            yl,yh = np.flatnonzero(yin)[np.array([0,-1])]
            xl,xh = np.flatnonzero(xin)[np.array([0,-1])]
            srcwcs = blobwcs.get_subimage(xl, yl, 1+xh-xl, 1+yh-yl)
            # A mask for which pixels in the 'srcwcs' square are occupied.
            srcpix = insrc[yl:yh+1, xl:xh+1]
            from scipy.ndimage.morphology import binary_erosion
            srcpix2 = binary_erosion(srcpix)
        else:
            modelMasks = models.model_masks(i, src)
            srctims = tims
            srcwcs = blobwcs
            srcpix = None

        srctractor = Tractor(srctims, [src])
        srctractor.freezeParams('images')
        srctractor.setModelMasks(modelMasks)
        enable_galaxy_cache()

        if plots:
            tims_compute_resamp(None, srctims, blobwcs)
            plt.clf()
            coimgs,cons = compute_coadds(srctims, bands, blobwcs,
                                         fill_holes=False)
            dimshow(get_rgb(coimgs, bands))#, extent=(bx0,bx1,by0,by1))
            plt.title('Model selection: stage1 data')
            ps.savefig()

            for tim in srctims:
                del tim.resamp
            tims_compute_resamp(None, srctims, srcwcs)
            plt.clf()
            coimgs,cons = compute_coadds(srctims, bands, srcwcs,
                                         fill_holes=False)
            dimshow(get_rgb(coimgs, bands))
            plt.title('Model selection: stage1 data')
            ps.savefig()

            # srch,srcw = srcwcs.shape
            # _plot_mods(srctims, [list(srctractor.getModelImages())],
            #            ['Model selection init'], bands, None, None,
            #            None, srch,srcw, ps, chi_plots=False)

        # use log likelihood rather than log prior because we use
        # priors on, eg, the ellipticity to avoid crazy fits.  Here we
        # sort of want to just evaluate the fit quality regardless of
        # priors on parameters...?

        srccat = srctractor.getCatalog()

        # Compute the log-likehood without a source here.
        srccat[0] = None
        chisqs_none = _per_band_chisqs(srctractor, bands)

        nparams = dict(ptsrc=2, simple=2, exp=5, dev=5, comp=9)
        # This is our "upgrade" threshold: how much better a galaxy
        # fit has to be versus ptsrc, and comp versus galaxy.
        galaxy_margin = 3.**2 + (nparams['exp'] - nparams['ptsrc'])

        # *chisqs* is actually chi-squared improvement vs no source;
        # larger is a better fit.
        chisqs = dict(none=0)

        oldmodel, ptsrc, simple, dev, exp, comp = _initialize_models(src)

        trymodels = [('ptsrc', ptsrc), ('simple', simple)]

        if oldmodel == 'ptsrc':
            # Try galaxy models if simple > ptsrc, or if bright.
            # The 'gals' model is just a marker
            trymodels.extend([('gals', None)])
        else:
            trymodels.extend([('dev', dev), ('exp', exp), ('comp', comp)])

        # If lots of exposures, cut to a subset that reach the DECaLS
        # depth goals and use those in an initial round?
        if many_exposures:
            timsubset = set()

            for band in bands:
                # Order to try them: first, DECaLS data (our propid),
                # then in point-source depth (* npix?) order.
                otims = []
                value = []
                for tim in srctims:
                    if tim.band != band:
                        continue
                    otims.append(tim)

                    detsig1 = tim.sig1 / tim.meta.galnorm
                    tim.detiv1 = 1./detsig1**2
                    h,w = tim.shape
                    propid = tim.meta.propid
                    value.append((propid == DECALS_PROPID) * 1e12 +
                                 tim.detiv1 * h*w)

                t1,t2,t3 = dict(g=(24.0, 23.7, 23.4),
                                r=(23.4, 23.1, 22.8),
                                z=(22.5, 22.2, 21.9))[band]
                Nsigma = 5.
                sig = NanoMaggies.magToNanomaggies(t1) / Nsigma
                target1 = 1./sig**2
                sig = NanoMaggies.magToNanomaggies(t2) / Nsigma
                target2 = 1./sig**2
                sig = NanoMaggies.magToNanomaggies(t3) / Nsigma
                target3 = 1./sig**2

                detiv = np.zeros(srcwcs.shape, np.float32)
                I = np.argsort(-np.array(value))
                for i in I:
                    tim = otims[i]
                    try:
                        Yo,Xo,Yi,Xi,nil = resample_with_wcs(
                            srcwcs, tim.subwcs, [], 2)
                    except:
                        continue
                    if len(Yo) == 0:
                        continue
                    detiv[Yo,Xo] += tim.detiv1
                    timsubset.add(tim.name)

                    print('Tim:', tim.name, 'exptime', tim.meta.exptime,
                          'FWHM', tim.meta.fwhm)
                    print('Tim: detiv', tim.detiv1, 'depth mag',
                          NanoMaggies.nanomaggiesToMag(np.sqrt(1./tim.detiv1)
                                                       * Nsigma))
                    print('overlapping pixels:', len(Yo))

                    # Hit DECaLS depth targets?
                    pctiles = [100-90, 100-95, 100-98]
                    if srcpix is None:
                        p1,p2,p3 = np.percentile(detiv, pctiles)
                    else:
                        p1,p2,p3 = np.percentile(detiv[srcpix2], pctiles)

                    m1 = NanoMaggies.nanomaggiesToMag(np.sqrt(1./p1) * Nsigma)
                    m2 = NanoMaggies.nanomaggiesToMag(np.sqrt(1./p2) * Nsigma)
                    m3 = NanoMaggies.nanomaggiesToMag(np.sqrt(1./p3) * Nsigma)
                    print('Added image', tim.name, 'and got depths (mag)',
                          '%.2f, %.2f, %.2f' % (m1,m2,m3),
                          'vs target mags', t1, t2, t3)
                    print('  detivs', p1, p2, p3, 'vs targets',
                          target1, target2, target3)

                    if p1 >= target1 and p2 >= target2 and p3 >= target3:
                        # Got enough depth, thank you!
                        print('Reached target depth!')
                        break

            if plots:
                dtims = [tim for tim in srctims if tim.name in timsubset]
                plt.clf()
                coimgs,cons = compute_coadds(dtims, bands, srcwcs,
                                             fill_holes=False)
                dimshow(get_rgb(coimgs, bands))
                plt.title('To-depth data')
                ps.savefig()

        allflags = {}
        for name,newsrc in trymodels:

            if name == 'gals':
                # If 'simple' was better than 'ptsrc', or the source is
                # bright, try the galaxy models.
                if ((chisqs['simple'] > chisqs['ptsrc']) or
                    (chisqs['ptsrc'] > 400)):

                    if hastycho:
                        print('Not computing galaxy models: Tycho-2 star in '+
                              'blob')
                        continue

                    trymodels.extend([
                        ('dev', dev), ('exp', exp), ('comp', comp)])
                continue

            if name == 'comp' and newsrc is None:
                # Compute the comp model if exp or dev would be accepted
                if (max(chisqs['dev'], chisqs['exp']) <
                    (chisqs['ptsrc'] + galaxy_margin)):
                    #print('dev/exp not much better than ptsrc; not computing
                    # comp model.')
                    continue
                newsrc = comp = FixedCompositeGalaxy(
                    src.getPosition(), src.getBrightness(),
                    SoftenedFracDev(0.5), exp.getShape(),
                    dev.getShape()).copy()
            #print('New source:', newsrc)
            srccat[0] = newsrc

            # Use the same modelMask shapes as the original source ('src').
            # Need to create newsrc->mask mappings though:
            mm = []
            for mim in modelMasks:
                d = dict()
                mm.append(d)
                try:
                    d[newsrc] = mim[src]
                except KeyError:
                    pass
            srctractor.setModelMasks(mm)
            enable_galaxy_cache()

            # Save these modelMasks for later...
            newsrc_mm = mm

            #lnp = srctractor.getLogProb()
            #print('Initial log-prob:', lnp)
            #print('vs original src: ', lnp - lnp0)
            if plots and False:
                # Grid of derivatives.
                _plot_derivs(tims, newsrc, srctractor, ps)
            if plots:
                mods = list(srctractor.getModelImages())
                plt.clf()
                coimgs,cons = compute_coadds(srctims, bands, srcwcs,
                                             images=mods, fill_holes=False)
                dimshow(get_rgb(coimgs, bands))
                plt.title('Initial: ' + name)
                ps.savefig()

            if many_exposures:
                # Run a quick round of optimization with our to-depth subset
                dtims = []
                dmm = []
                for tim,mim in zip(srctims, modelMasks):
                    if not tim.name in timsubset:
                        continue
                    dtims.append(tim)
                    d = dict()
                    dmm.append(d)
                    try:
                        d[newsrc] = mim[src]
                    except KeyError:
                        pass

                dtractor = Tractor(dtims, [newsrc])
                dtractor.freezeParams('images')
                dtractor.setModelMasks(dmm)
                enable_galaxy_cache()

                for step in range(50):
                    #print('optimizing:', newsrc)
                    dlnp,X,alpha = dtractor.optimize(**optargs)
                    #print('  dlnp:', dlnp, 'new src', newsrc)
                    if dlnp < 0.1:
                        break

                # print('Mod', name, 'round0 opt', Time()-t0)
                # print('New source (after to-depth round optimization):',
                #   newsrc)

                if plots:
                    plt.clf()
                    modimgs = list(dtractor.getModelImages())
                    comods,nil = compute_coadds(dtims, bands, srcwcs,
                                                images=modimgs)
                    dimshow(get_rgb(comods, bands))
                    plt.title('To-depth opt: ' + name)
                    ps.savefig()

            # First-round optimization (during model selection)
            thisflags = 0
            for step in range(50):
                #print('optimizing:', newsrc)
                dlnp,X,alpha = srctractor.optimize(**optargs)
                #print('  dlnp:', dlnp, 'new src', newsrc)
                if dlnp < 0.1:
                    break
            else:
                thisflags |= FLAG_STEPS_A

            # print('Mod', name, 'round1 opt', Time()-t0)
            #print('New source (after first round optimization):', newsrc)

            if plots:
                # _plot_mods(srctims, [list(srctractor.getModelImages())],
                #            ['Model selection: ' + name], bands, None, None,
                #            None, srch,srcw, ps, chi_plots=False)
                plt.clf()
                modimgs = list(srctractor.getModelImages())
                comods,nil = compute_coadds(srctims, bands, srcwcs,
                                            images=modimgs)
                dimshow(get_rgb(comods, bands))
                plt.title('First-round opt: ' + name)
                ps.savefig()

            srctractor.setModelMasks(None)
            disable_galaxy_cache()

            # Recompute modelMasks in the original tims

            ## FIXME -- avoid huge patches?  Clip to significant
            ## pixels in model?  Via minval, I guess; appealing to use
            ## same cut as determining the blobs, but that's in brick
            ## coadd space.

            tim = tims[0]
            from tractor.galaxy import ProfileGalaxy
            if isinstance(newsrc, ProfileGalaxy):
                px,py = tim.wcs.positionToPixel(newsrc.getPosition())
                h = newsrc._getUnitFluxPatchSize(tim, px, py, tim.modelMinval)
                MAXHALF = 128
                if h > MAXHALF:
                    print('halfsize', h,'for',newsrc,'-> setting to',MAXHALF)
                    newsrc.halfsize = MAXHALF

            if hastycho:
                modtims = None
            elif bigblob:
                mods = []
                for tim in tims:
                    mod = newsrc.getModelPatch(tim)
                    if mod is not None:
                        h,w = tim.shape
                        mod.clipTo(w,h)
                        if mod.patch is None:
                            mod = None
                    mods.append(mod)
                modtims,mm = _get_subimages(tims, mods, newsrc)
            else:
                mm = []
                modtims = []
                for tim in tims:
                    d = dict()
                    mod = newsrc.getModelPatch(tim)
                    if mod is None:
                        continue
                    #print('After first-round fit: model is', mod.shape)
                    mod = _clip_model_to_blob(mod, tim.shape,tim.getInvError())
                    d[newsrc] = Patch(mod.x0, mod.y0, mod.patch != 0)
                    modtims.append(tim)
                    mm.append(d)

            if modtims is not None:
                modtractor = Tractor(modtims, [newsrc])
                modtractor.freezeParams('images')
                modtractor.setModelMasks(mm)
                enable_galaxy_cache()

                #t0 = Time()

                # Run another round of opt.
                for step in range(50):
                    dlnp,X,alpha = modtractor.optimize(**optargs)
                    #print('  dlnp:', dlnp, 'new src', newsrc)
                    if dlnp < 0.1:
                        break
                else:
                    thisflags |= FLAG_STEPS_B

                # print('Mod', name, 'round2 opt', Time()-t0)

                if plots:
                    plt.clf()
                    modimgs = list(modtractor.getModelImages())
                    tims_compute_resamp(None, modtims, srcwcs)
                    comods,nil = compute_coadds(modtims, bands, srcwcs,
                                                images=modimgs)
                    dimshow(get_rgb(comods, bands))
                    plt.title('Second-round opt: ' + name)
                    ps.savefig()
            else:
                # Tycho-2 star; set modtractor = srctractor for the ivars
                srctractor.setModelMasks(newsrc_mm)
                modtractor = srctractor

            # Compute FLUX inverse-variances for each source.
            # This uses the second-round modelMasks.
            newsrc.freezeAllBut('brightness')
            allderivs = modtractor.getDerivs()
            ivs = np.zeros(len(bands), np.float32)
            B.all_model_fluxivs[i][name] = ivs
            for iparam,derivs in enumerate(allderivs):
                chisq = 0
                for deriv,tim in derivs:
                    h,w = tim.shape
                    deriv.clipTo(w,h)
                    ie = tim.getInvError()
                    slc = deriv.getSlice(ie)
                    chi = deriv.patch * ie[slc]
                    chisq += (chi**2).sum()
                ivs[iparam] = chisq
            newsrc.thawAllParams()

            # Use the original 'srctractor' here so that the different
            # models are evaluated on the same pixels.
            # ---> AND with the same modelMasks as the original source...
            # FIXME -- it is not clear that this is what we want!!!
            srctractor.setModelMasks(newsrc_mm)
            ch = _per_band_chisqs(srctractor, bands)
            chisqs[name] = _chisq_improvement(newsrc, ch, chisqs_none)
            B.all_models[i][name] = newsrc.copy()
            B.all_model_flags[i][name] = thisflags

        # if plots:
        #    _plot_mods(tims, plotmods, plotmodnames, bands, None, None,
        #               bslc, blobw, blobh, ps)

        if plots:
            from collections import OrderedDict
            plt.clf()
            rows,cols = 2, 6
            mods = OrderedDict([('none',None), ('ptsrc',ptsrc),
                                ('simple',simple),
                                ('dev',dev), ('exp',exp), ('comp',comp)])
            for imod,modname in enumerate(mods.keys()):

                if mod != 'none' and not modname in chisqs:
                    continue

                srccat[0] = mods[modname]

                print('Plotting model for blob', iblob, 'source', i,
                      ':', modname)
                print(srccat[0])

                srctractor.setModelMasks(None)

                plt.subplot(rows, cols, imod+1)

                if modname != 'none':
                    modimgs = list(srctractor.getModelImages())
                    comods,nil = compute_coadds(srctims, bands, srcwcs,
                                                images=modimgs)
                    dimshow(get_rgb(comods, bands), ticks=False)
                    plt.title(modname)
                    chis = [((tim.getImage() - mod) * tim.getInvError())**2
                            for tim,mod in zip(srctims, modimgs)]
                    res = [(tim.getImage() - mod) for tim,mod in
                           zip(srctims, modimgs)]
                else:
                    coimgs, cons = compute_coadds(srctims, bands, srcwcs)
                    dimshow(get_rgb(coimgs, bands))
                    ax = plt.axis()
                    ok,x,y = blobwcs.radec2pixelxy(src.getPosition().ra,
                                                     src.getPosition().dec)
                    plt.plot(x-1, y-1, 'r+')
                    plt.axis(ax)
                    plt.title('Image')
                    chis = [((tim.getImage()) * tim.getInvError())**2
                              for tim in srctims]
                    res = [tim.getImage() for tim in srctims]

                if False:
                    cochisqs,nil = compute_coadds(tims, bands, blobwcs,
                                                  images=chis)
                    cochisq = reduce(np.add, cochisqs)
                    plt.subplot(rows, cols, imod+1+cols)
                    dimshow(cochisq, vmin=0, vmax=25)

                else:
                    # residuals
                    coresids,nil = compute_coadds(srctims, bands, srcwcs,
                                                  images=res)
                    plt.subplot(rows, cols, imod+1+cols)
                    dimshow(get_rgb(coresids, bands, **rgbkwargs_resid),
                            ticks=False)
                plt.title('chisq %.0f' % chisqs[modname], fontsize=8)
            plt.suptitle('Blob %i, source %i: was: \n%s' %
                         (iblob, i, str(src)), fontsize=10)
            ps.savefig()

        # This determines the order of the elements in the DCHISQ
        # column of the catalog.
        modnames = ['ptsrc', 'simple', 'dev', 'exp', 'comp']

        keepmod = _select_model(chisqs, nparams, galaxy_margin)

        keepsrc = dict(none=None, ptsrc=ptsrc, simple=simple,
                       dev=dev, exp=exp, comp=comp)[keepmod]

        B.dchisqs[i, :] = np.array([chisqs.get(k,0) for k in modnames])
        B.flags[i] = allflags.get(keepmod, 0)
        B.sources[i] = keepsrc
        cat[i] = keepsrc

        # Re-remove the final fit model for this source.
        models.update_and_subtract(i, keepsrc, tims)

        #print('Keeping model:', keepmod)
        #print('Keeping source:', keepsrc)
        #print(Time() - tsel)

    models.restore_images(tims)
    del models

    print('Blob finished model selection:', Time()-tlast)
    tlast = Time()

    if plots:
        plotmods, plotmodnames = [],[]
        plotmods.append(list(tr.getModelImages()))
        plotmodnames.append('All model selection')
        _plot_mods(tims, plotmods, plotmodnames, bands, None, None, bslc,
                   blobw, blobh, ps)

    I = np.array([i for i,s in enumerate(cat) if s is not None])
    B.cut(I)

    cat = Catalog(*B.sources)
    tr.catalog = cat

    # Do another quick round of flux-only fitting?
    # This does horribly -- fluffy galaxies go out of control because
    # they're only constrained by pixels within this blob.
    #_fit_fluxes(cat, tims, bands, use_ceres, alphas)

    # print('After cutting sources:')
    # for src,dchi in zip(B.sources, B.dchisqs):
    #     print('  source', src, 'max dchisq', max(dchi), 'dchisqs', dchi)

    ### Simultaneous re-opt?
    if simul_opt and len(cat) > 1 and len(cat) <= 10:
        #tfit = Time()
        cat.thawAllParams()
        #print('Optimizing:', tr)
        #tr.printThawedParams()
        flags |= FLAG_TRIED_C
        max_cpu = 300.
        cpu0 = time.clock()
        for step in range(50):
            dlnp,X,alpha = tr.optimize(**optargs)
            cpu = time.clock()
            if cpu-cpu0 > max_cpu:
                print('Warning: Exceeded maximum CPU time for source')
                flags |= FLAG_CPU_C
                break
            if dlnp < 0.1:
                break
        #print('Simultaneous fit took:', Time()-tfit)

    # Compute variances on all parameters for the kept model
    B.srcinvvars = [[] for i in range(len(B))]
    cat.thawAllRecursive()
    cat.freezeAllParams()
    for isub in range(len(B.sources)):
        cat.thawParam(isub)
        src = cat[isub]
        if src is None:
            cat.freezeParam(isub)
            continue
        # Convert to "vanilla" ellipse parameterization
        if isinstance(src, (DevGalaxy, ExpGalaxy)):
            src.shape = src.shape.toEllipseE()
        elif isinstance(src, FixedCompositeGalaxy):
            src.shapeExp = src.shapeExp.toEllipseE()
            src.shapeDev = src.shapeDev.toEllipseE()
            src.fracDev = FracDev(src.fracDev.clipped())

        #print('Computing variances for source:', src)
        #tr.printThawedParams()
        #print('halfsize:', getattr(src, 'halfsize', None))

        allderivs = tr.getDerivs()
        for iparam,derivs in enumerate(allderivs):
            chisq = 0
            for deriv,tim in derivs:
                h,w = tim.shape
                deriv.clipTo(w,h)
                ie = tim.getInvError()
                slc = deriv.getSlice(ie)
                chi = deriv.patch * ie[slc]
                chisq += (chi**2).sum()
            B.srcinvvars[isub].append(chisq)
        assert(len(B.srcinvvars[isub]) == cat[isub].numberOfParams())
        cat.freezeParam(isub)
    #print('Blob variances:', Time()-tlast)
    #tlast = Time()

    # Check for sources with zero inverse-variance -- I think these
    # can be generated during the "Simultaneous re-opt" stage above --
    # sources can get scattered outside the blob.
    # Arbitrarily look at the first element (RA)
    I = np.flatnonzero(np.array([iv[0] for iv in B.srcinvvars]))
    if len(I) < len(B):
        print('Keeping', len(I), 'of', len(B), 'sources with non-zero ivar')
        B.cut(I)
        cat = Catalog(*B.sources)
        tr.catalog = cat

    M = _compute_source_metrics(B.sources, tims, bands, tr)
    for k,v in M.items():
        B.set(k, v)

    ok,x1,y1 = blobwcs.radec2pixelxy(
        np.array([src.getPosition().ra  for src in B.sources]),
        np.array([src.getPosition().dec for src in B.sources]))
    B.finished_in_blob = blobmask[
        np.clip(np.round(y1-1).astype(int), 0, blobh-1),
        np.clip(np.round(x1-1).astype(int), 0, blobw-1)]
    assert(len(B.finished_in_blob) == len(B))
    assert(len(B.finished_in_blob) == len(B.started_in_blob))

    B.hastycho = np.zeros(len(B), bool)
    if hastycho:
        B.hastycho[:] = True

    print('Blob number', nblob, 'iblob', iblob, 'finished:', Time()-tlast)

    B.iblob = iblob

    return B




def _compute_source_metrics(srcs, tims, bands, tr):
    # rchi2 quality-of-fit metric
    rchi2_num    = np.zeros((len(srcs),len(bands)), np.float32)
    rchi2_den    = np.zeros((len(srcs),len(bands)), np.float32)

    # fracflux degree-of-blending metric
    fracflux_num = np.zeros((len(srcs),len(bands)), np.float32)
    fracflux_den = np.zeros((len(srcs),len(bands)), np.float32)

    # fracin flux-inside-blob metric
    fracin_num = np.zeros((len(srcs),len(bands)), np.float32)
    fracin_den = np.zeros((len(srcs),len(bands)), np.float32)

    # fracmasked: fraction of masked pixels metric
    fracmasked_num = np.zeros((len(srcs),len(bands)), np.float32)
    fracmasked_den = np.zeros((len(srcs),len(bands)), np.float32)

    for iband,band in enumerate(bands):
        for tim in tims:
            if tim.band != band:
                continue
            mod = np.zeros(tim.getModelShape(), tr.modtype)
            srcmods = [None for src in srcs]
            counts = np.zeros(len(srcs))
            pcal = tim.getPhotoCal()

            # For each source, compute its model and record its flux
            # in this image.  Also compute the full model *mod*.
            for isrc,src in enumerate(srcs):
                patch = tr.getModelPatch(tim, src, minsb=tim.modelMinval)
                if patch is None or patch.patch is None:
                    continue
                counts[isrc] = np.sum([np.abs(pcal.brightnessToCounts(b))
                                              for b in src.getBrightnesses()])
                if counts[isrc] == 0:
                    continue
                H,W = mod.shape
                patch.clipTo(W,H)
                srcmods[isrc] = patch
                patch.addTo(mod)

            # Now compute metrics for each source
            for isrc,patch in enumerate(srcmods):
                if patch is None:
                    continue
                if patch.patch is None:
                    continue
                if counts[isrc] == 0:
                    continue
                if np.sum(patch.patch**2) == 0:
                    continue
                slc = patch.getSlice(mod)
                patch = patch.patch

                # print('fracflux: band', band, 'isrc', isrc, 'tim', tim.name)
                # print('src:', srcs[isrc])
                # print('patch sum', np.sum(patch),'abs',np.sum(np.abs(patch)))
                # print('counts:', counts[isrc])
                # print('mod slice sum', np.sum(mod[slc]))
                # print('mod[slc] - patch:', np.sum(mod[slc] - patch))

                # (mod - patch) is flux from others
                # (mod - patch) / counts is normalized flux from others
                # We take that and weight it by this source's profile;
                #  patch / counts is unit profile
                # But this takes the dot product between the profiles,
                # so we have to normalize appropriately, ie by
                # (patch**2)/counts**2; counts**2 drops out of the
                # denom.  If you have an identical source with twice the flux,
                # this results in fracflux being 2.0

                # fraction of this source's flux that is inside this patch.
                # This can be < 1 when the source is near an edge, or if the
                # source is a huge diffuse galaxy in a small patch.
                fin = np.abs(np.sum(patch) / counts[isrc])

                # print('fin:', fin)
                # print('fracflux_num: fin *',
                #      np.sum((mod[slc] - patch) * np.abs(patch)) /
                #      np.sum(patch**2))

                fracflux_num[isrc,iband] += (fin *
                    np.sum((mod[slc] - patch) * np.abs(patch)) /
                    np.sum(patch**2))
                fracflux_den[isrc,iband] += fin
                
                fracmasked_num[isrc,iband] += (
                    np.sum((tim.getInvError()[slc] == 0) * np.abs(patch)) /
                    np.abs(counts[isrc]))
                    
                fracmasked_den[isrc,iband] += fin

                fracin_num[isrc,iband] += np.abs(np.sum(patch))
                fracin_den[isrc,iband] += np.abs(counts[isrc])

            tim.getSky().addTo(mod)
            chisq = ((tim.getImage() - mod) * tim.getInvError())**2

            for isrc,patch in enumerate(srcmods):
                if patch is None or patch.patch is None:
                    continue
                if counts[isrc] == 0:
                    continue
                slc = patch.getSlice(mod)
                # We compute numerator and denom separately to handle
                # edge objects, where sum(patch.patch) < counts.
                # Also, to normalize by the number of images.  (Being
                # on the edge of an image is like being in half an
                # image.)
                rchi2_num[isrc,iband] += (np.sum(chisq[slc] * patch.patch) / 
                                          counts[isrc])
                # If the source is not near an image edge,
                # sum(patch.patch) == counts[isrc].
                rchi2_den[isrc,iband] += np.sum(patch.patch) / counts[isrc]

    #print('Fracflux_num:', fracflux_num)
    #print('Fracflux_den:', fracflux_den)
                
    fracflux   = fracflux_num   / fracflux_den
    rchi2      = rchi2_num      / rchi2_den
    fracmasked = fracmasked_num / fracmasked_den

    #print('Fracflux:', B.fracflux)
    
    # fracin_{num,den} are in flux * nimages units
    tinyflux = 1e-9
    fracin     = fracin_num     / np.maximum(tinyflux, fracin_den)

    return dict(fracin=fracin, fracflux=fracflux, rchi2=rchi2,
                fracmasked=fracmasked)


def _initialize_models(src):
    if isinstance(src, PointSource):
        ptsrc = src.copy()
        simple = SimpleGalaxy(src.getPosition(), src.getBrightness()).copy()
        # logr, ee1, ee2
        shape = LegacyEllipseWithPriors(-1., 0., 0.)
        dev = DevGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
        exp = ExpGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
        comp = None
        oldmodel = 'ptsrc'

    elif isinstance(src, DevGalaxy):
        ptsrc = PointSource(src.getPosition(), src.getBrightness()).copy()
        simple = SimpleGalaxy(src.getPosition(), src.getBrightness()).copy()
        dev = src.copy()
        exp = ExpGalaxy(src.getPosition(), src.getBrightness(),
                        src.getShape()).copy()
        comp = None
        oldmodel = 'dev'

    elif isinstance(src, ExpGalaxy):
        ptsrc = PointSource(src.getPosition(), src.getBrightness()).copy()
        simple = SimpleGalaxy(src.getPosition(), src.getBrightness()).copy()
        dev = DevGalaxy(src.getPosition(), src.getBrightness(),
                        src.getShape()).copy()
        exp = src.copy()
        comp = None
        oldmodel = 'exp'

    elif isinstance(src, FixedCompositeGalaxy):
        ptsrc = PointSource(src.getPosition(), src.getBrightness()).copy()
        simple = SimpleGalaxy(src.getPosition(), src.getBrightness()).copy()
        frac = src.fracDev.clipped()
        if frac > 0:
            shape = src.shapeDev
        else:
            shape = src.shapeExp
        dev = DevGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
        if frac < 1:
            shape = src.shapeExp
        else:
            shape = src.shapeDev
        exp = ExpGalaxy(src.getPosition(), src.getBrightness(), shape).copy()
        comp = src.copy()
        oldmodel = 'comp'

    return oldmodel, ptsrc, simple, dev, exp, comp


def _get_subimages(tims, mods, src):
    srctims = []
    modelMasks = []
    #print('Big blob: trimming:')
    for tim,mod in zip(tims, mods):
        if mod is None:
            continue
        mh,mw = mod.shape
        if mh == 0 or mw == 0:
            continue
        # for modelMasks
        d = { src: Patch(0, 0, mod.patch != 0) }
        modelMasks.append(d)

        x0,y0 = mod.x0 , mod.y0
        x1,y1 = x0 + mw, y0 + mh
        slc = slice(y0,y1), slice(x0, x1)

        subimg = tim.getImage()[slc]
        if subimg.shape != (mh,mw):
            print('Subimage shape:', subimg.shape, 'image shape',
                  tim.getImage().shape, 'slice y', y0,y1, 'x', x0,x1,
                  'mod shape', mh,mw)
        # print('  srctim: x0,y0', x0,y0, 'shape', (y1-y0,x1-x0))
        subpsf = tim.psf.constantPsfAt((x0+x1)/2., (y0+y1)/2.)
        srctim = Image(data=subimg,
                       inverr=tim.getInvError()[slc],
                       wcs=tim.wcs.shifted(x0, y0),
                       psf=subpsf,
                       photocal=tim.getPhotoCal(),
                       sky=tim.sky.shifted(x0, y0),
                       name=tim.name)
        srctim.subwcs = tim.subwcs.get_subimage(x0, y0, mw, mh)
        srctim.band = tim.band
        srctim.sig1 = tim.sig1
        srctim.modelMinval = tim.modelMinval
        srctim.x0 = x0
        srctim.y0 = y0
        srctim.meta = tim.meta
        srctims.append(srctim)
        #print('  ', tim.shape, 'to', srctim.shape)
    return srctims, modelMasks

def _get_mod(X):
    (tim, srcs) = X
    t0 = Time()
    tractor = Tractor([tim], srcs)

    if hasattr(tim, 'modelMinval'):
        print('tim modelMinval', tim.modelMinval)
        minval = tim.modelMinval
    else:
        # this doesn't really help when using pixelized PSFs / FFTs
        tim.modelMinval = minval = tim.sig * 0.1

    for src in srcs:
        from tractor.galaxy import ProfileGalaxy
        if not isinstance(src, ProfileGalaxy):
            continue
        px,py = tim.wcs.positionToPixel(src.getPosition())
        h = src._getUnitFluxPatchSize(tim, px, py, minval)
        if h > 512:
            print('halfsize', h, 'for', src)
            src.halfsize = 512

    mod = tractor.getModelImage(0)
    print('Getting model for', tim, ':', Time()-t0)

    #######
    # from tractor.galaxy import fft_timing
    # from astrometry.util.file import pickle_to_file
    # for row in fft_timing:
    #     print(row)
    # pickle_to_file(fft_timing,
    #                ('fft-timing-%s.pickle' % 
    # str(tim).replace('Image ','').replace(' ','')))
    #######

    return mod

def stage_coadds(bands=None, version_header=None, targetwcs=None,
                 tims=None, ps=None, brickname=None, ccds=None,
                 outdir=None, T=None, cat=None, pixscale=None, plots=False,
                 coadd_bw=False, brick=None, W=None, H=None,
                 mp=None,
                 **kwargs):
    '''
    After the `stage_fitblobs` fitting stage (and
    `stage_fitblobs_finish` reformatting stage), we have all the
    source model fits, and we can create coadds of the images, model,
    and residuals.  We also perform aperture photometry in this stage.
    '''
    tlast = Time()

    if outdir is None:
        outdir = '.'
    basedir = os.path.join(outdir, 'coadd', brickname[:3], brickname)
    trymakedirs(basedir)
    fn = os.path.join(basedir, 'decals-%s-ccds.fits' % brickname)
    #
    ccds.ccd_x0 = np.array([tim.x0 for tim in tims]).astype(np.int16)
    ccds.ccd_x1 = np.array([tim.x0 + tim.shape[1]
                            for tim in tims]).astype(np.int16)
    ccds.ccd_y0 = np.array([tim.y0 for tim in tims]).astype(np.int16)
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
    ccds.sig1 = np.array([tim.sig1 for tim in tims])
    ccds.psfnorm = np.array([tim.psfnorm for tim in tims])
    ccds.galnorm = np.array([tim.galnorm for tim in tims])
    ccds.propid = np.array([tim.propid for tim in tims])
    ccds.plver = np.array([tim.plver for tim in tims])
    ccds.skyver = np.array([tim.skyver[0] for tim in tims])
    ccds.wcsver = np.array([tim.wcsver[0] for tim in tims])
    ccds.psfver = np.array([tim.psfver[0] for tim in tims])
    ccds.skyplver = np.array([tim.skyver[1] for tim in tims])
    ccds.wcsplver = np.array([tim.wcsver[1] for tim in tims])
    ccds.psfplver = np.array([tim.psfver[1] for tim in tims])

    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)
    primhdr.add_record(dict(name='PRODTYPE', value='ccdinfo',
                            comment='NOAO data product type'))
    ccds.writeto(fn, primheader=primhdr)
    print('Wrote', fn)

    tnow = Time()
    print('[serial coadds]:', tnow-tlast)
    tlast = tnow
    mods = mp.map(_get_mod, [(tim, cat) for tim in tims])
    tnow = Time()
    print('[parallel coadds] Getting model images:', tnow-tlast)
    tlast = tnow

    W = targetwcs.get_width()
    H = targetwcs.get_height()

    # Look up number of images overlapping each source's position.
    assert(len(T) == len(cat))
    rr = np.array([s.getPosition().ra  for s in cat])
    dd = np.array([s.getPosition().dec for s in cat])
    ok,ix,iy = targetwcs.radec2pixelxy(rr, dd)
    T.oob = reduce(np.logical_or, [ix < 0.5, iy < 0.5, ix > W+0.5, iy > H+0.5])
    ix = np.clip(np.round(ix - 1), 0, W-1).astype(int)
    iy = np.clip(np.round(iy - 1), 0, H-1).astype(int)

    # convert apertures to pixels
    apertures = apertures_arcsec / pixscale

    # Aperture photometry locations
    ra  = np.array([src.getPosition().ra  for src in cat])
    dec = np.array([src.getPosition().dec for src in cat])
    ok,xx,yy = targetwcs.radec2pixelxy(ra, dec)
    apxy = np.vstack((xx - 1., yy - 1.)).T
    del xx,yy,ok,ra,dec

    C = _coadds(tims, bands, targetwcs, mods=mods, xy=(ix,iy),
                ngood=True, detmaps=True, psfsize=True,
                apertures=apertures, apxy=apxy,
                callback=_write_band_images,
                callback_args=(brickname, version_header, tims, targetwcs,
                               basedir),
                plots=False, ps=ps)

    for c in ['nobs', 'anymask', 'allmask', 'psfsize']:
        T.set(c, C.T.get(c))

    # Compute the brick's unique pixels.
    U = None
    if hasattr(brick, 'ra1'):
        print('Computing unique brick pixels...')
        xx,yy = np.meshgrid(np.arange(W), np.arange(H))
        rr,dd = targetwcs.pixelxy2radec(xx+1, yy+1)
        U = np.flatnonzero((rr >= brick.ra1 ) * (rr < brick.ra2 ) *
                           (dd >= brick.dec1) * (dd < brick.dec2))
        print(len(U), 'of', W*H, 'pixels are unique to this brick')

    # depth histogram bins
    depthbins = np.arange(20, 25.001, 0.1)
    depthbins[0] = 0.
    depthbins[-1] = 100.
    D = fits_table()
    D.depthlo = depthbins[:-1].astype(np.float32)
    D.depthhi = depthbins[1: ].astype(np.float32)

    for band,detiv,galdetiv in zip(bands, C.detivs, C.galdetivs):
        for det,name in [(detiv, 'ptsrc'), (galdetiv, 'gal')]:
            # compute stats for 5-sigma detection
            depth = 5. / np.sqrt(det)
            # that's flux in nanomaggies -- convert to mag
            depth = -2.5 * (np.log10(depth) - 9)
            # no coverage -> very bright detection limit
            depth[np.logical_not(np.isfinite(depth))] = 0.
            if U is not None:
                depth = depth.flat[U]
            print(band, name, 'band depth map: percentiles',
                  np.percentile(depth, np.arange(0,101, 10)))
            # histogram
            D.set('counts_%s_%s' % (name, band),
                  np.histogram(depth, bins=depthbins)[0].astype(np.int32))

    del U
    del depth
    del det
    del detiv
    del galdetiv
    fn = os.path.join(basedir, 'decals-%s-depth.fits' % (brickname))
    D.writeto(fn)
    print('Wrote', fn)
    del D

    tmpfn = create_temp(suffix='.png')
    for name,ims,rgbkw in [('image', C.coimgs, rgbkwargs),
                           ('model', C.comods, rgbkwargs),
                           ('resid', C.coresids, rgbkwargs_resid),
                           ]:
        rgb = get_rgb(ims, bands, **rgbkw)
        kwa = {}
        if coadd_bw and len(bands) == 1:
            i = 'zrg'.index(bands[0])
            rgb = rgb[:,:,i]
            kwa = dict(cmap='gray')
        plt.imsave(tmpfn, rgb, origin='lower', **kwa)
        del rgb
        jpegfn = os.path.join(basedir, 'decals-%s-%s.jpg' % (brickname, name))
        cmd = ('pngtopnm %s | pnmtojpeg -quality 90 > %s' % (tmpfn, jpegfn))
        os.system(cmd)
        os.unlink(tmpfn)
        print('Wrote', jpegfn)

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
        ps.savefig()

        plt.clf()
        dimshow(get_rgb(C.coimgs, bands, **rgbkwargs))
        ax = plt.axis()
        ps.savefig()

        for i,(src,x,y,rr,dd) in enumerate(zip(cat, x1, y1, ra, dec)):
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
    print('[serial coadds] Aperture photometry, wrap-up', tnow-tlast)

    return dict(T=T, AP=C.AP, apertures_pix=apertures,
                apertures_arcsec=apertures_arcsec)


def stage_wise_forced(
    cat=None,
    T=None,
    targetwcs=None,
    W=None, H=None,
    brickname=None,
    unwise_dir=None,
    unwise_w12_dir=None,
    unwise_w34_dir=None,
    brick=None,
    outdir=None,
    use_ceres=True,
    mp=None,
    **kwargs):
    '''
    After the model fits are finished, we can perform forced
    photometry of the unWISE coadds.
    '''
    from wise.forcedphot import unwise_tiles_touching_wcs

    if unwise_dir is None:
        unwise_dir = 'unwise-coadds'
    if unwise_w12_dir is None:
        unwise_w12_dir = unwise_dir
    if unwise_w34_dir is None:
        unwise_w34_dir = unwise_dir

    # Here we assume the targetwcs is axis-aligned and that the
    # edge midpoints yield the RA,Dec limits (true for TAN).
    r,d = targetwcs.pixelxy2radec(np.array([1,   W,   W/2, W/2]),
                                  np.array([H/2, H/2, 1,   H  ]))
    # the way the roiradec box is used, the min/max order doesn't matter
    roiradec = [r[0], r[1], d[2], d[3]]

    tiles = unwise_tiles_touching_wcs(targetwcs)
    print('Cut to', len(tiles), 'unWISE tiles')

    wcat = []
    for src in cat:
        src = src.copy()
        src.setBrightness(NanoMaggies(w=1.))
        wcat.append(src)

    args = []
    for band in [1,2]:
        args.append((wcat, tiles, band, roiradec, unwise_w12_dir, use_ceres))
    for band in [3,4]:
        args.append((wcat, tiles, band, roiradec, unwise_w34_dir, use_ceres))

    phots = mp.map(_unwise_phot, args)
    WISE = phots[0]
    for p in phots[1:]:
        WISE.add_columns_from(p)

    WISE.rename('tile', 'unwise_tile')
    return dict(WISE=WISE)

def _unwise_phot(X):
    from wise.forcedphot import unwise_forcedphot

    (wcat, tiles, band, roiradec, unwise_dir, use_ceres) = X

    try:
        W = unwise_forcedphot(wcat, tiles, roiradecbox=roiradec, bands=[band],
                              unwise_dir=unwise_dir, use_ceres=use_ceres)
    except:
        import traceback
        print('unwise_forcedphot failed:')
        traceback.print_exc()

        if use_ceres:
            print('Trying without Ceres...')
            W = unwise_forcedphot(wcat, tiles, roiradecbox=roiradec,
                                  bands=[band], unwise_dir=unwise_dir,
                                  use_ceres=False)
    return W


'''
Write catalog output
'''
def stage_writecat(
    version_header=None,
    T=None,
    WISE=None,
    no_sdss=False,
    AP=None,
    apertures_arcsec=None,
    cat=None, targetrd=None, pixscale=None, targetwcs=None,
    W=None,H=None,
    bands=None, ps=None,
    plots=False,
    brickname=None,
    brickid=None,
    brick=None,
    invvars=None,
    catalogfn=None,
    outdir=None,
    write_catalog=True,
    allbands=None,
    **kwargs):
    '''
    Final stage in the pipeline: format results for the output
    catalog.
    '''

    from desi_common import prepare_fits_catalog
    fs = None
    TT = T.copy()
    for k in ['itx','ity','index']:
        if k in TT.get_columns():
            TT.delete_column(k)
    TT.tx = TT.tx.astype(np.float32)
    TT.ty = TT.ty.astype(np.float32)

    TT.decam_rchi2      = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_fracflux   = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_fracmasked = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_fracin     = np.zeros((len(TT), len(allbands)), np.float32)
    TT.decam_nobs       = np.zeros((len(TT), len(allbands)), np.uint8)
    TT.decam_anymask    = np.zeros((len(TT), len(allbands)), TT.anymask.dtype)
    TT.decam_allmask    = np.zeros((len(TT), len(allbands)), TT.allmask.dtype)
    TT.decam_psfsize    = np.zeros((len(TT), len(allbands)), np.float32)
    B = np.array([allbands.index(band) for band in bands])
    TT.decam_rchi2     [:,B] = TT.rchi2
    TT.decam_fracflux  [:,B] = TT.fracflux
    TT.decam_fracmasked[:,B] = TT.fracmasked
    TT.decam_fracin    [:,B] = TT.fracin
    TT.decam_nobs      [:,B] = TT.nobs
    TT.decam_anymask   [:,B] = TT.anymask
    TT.decam_allmask   [:,B] = TT.allmask
    TT.decam_psfsize   [:,B] = TT.psfsize
    TT.delete_column('rchi2')
    TT.delete_column('fracflux')
    TT.delete_column('fracin')
    TT.delete_column('nobs')
    TT.delete_column('anymask')
    TT.delete_column('allmask')
    TT.rename('oob', 'out_of_bounds')

    if AP is not None:
        # How many apertures?
        ap = AP.get('apflux_img_%s' % bands[0])
        n,A = ap.shape
        TT.decam_apflux = np.zeros((len(TT), len(allbands), A), np.float32)
        TT.decam_apflux_ivar  = np.zeros((len(TT), len(allbands), A),
                                         np.float32)
        TT.decam_apflux_resid = np.zeros((len(TT), len(allbands), A),
                                         np.float32)
        for iband,band in enumerate(bands):
            i = allbands.index(band)
            TT.decam_apflux      [:,i,:] = AP.get('apflux_img_%s'      % band)
            TT.decam_apflux_ivar [:,i,:] = AP.get('apflux_img_ivar_%s' % band)
            TT.decam_apflux_resid[:,i,:] = AP.get('apflux_resid_%s'    % band)

    cat.thawAllRecursive()
    hdr = None
    T2,hdr = prepare_fits_catalog(cat, invvars, TT, hdr, bands, fs,
                                  allbands=allbands)

    # mod
    T2.ra += (T2.ra <   0) * 360.
    T2.ra -= (T2.ra > 360) * 360.

    primhdr = fitsio.FITSHDR()
    for r in version_header.records():
        primhdr.add_record(r)

    primhdr.add_record(dict(name='ALLBANDS', value=allbands,
                            comment='Band order in array values'))

    primhdr.add_record(dict(name='PRODTYPE', value='catalog',
                            comment='NOAO data product type'))

    if AP is not None:
        for i,ap in enumerate(apertures_arcsec):
            primhdr.add_record(dict(name='APRAD%i' % i, value=ap,
                                    comment='Aperture radius, in arcsec'))

    bits = CP_DQ_BITS.values()
    bits.sort()
    bitmap = dict((v,k) for k,v in CP_DQ_BITS.items())
    for i in range(16):
        bit = 1<<i
        if bit in bitmap:
            primhdr.add_record(dict(name='MASKB%i' % i, value=bitmap[bit],
                                    comment='Mask bit 2**%i=%i meaning' %
                                    (i, bit)))

    ok,bx,by = targetwcs.radec2pixelxy(T2.orig_ra, T2.orig_dec)
    T2.bx0 = (bx - 1.).astype(np.float32)
    T2.by0 = (by - 1.).astype(np.float32)
    ok,bx,by = targetwcs.radec2pixelxy(T2.ra, T2.dec)
    T2.bx = (bx - 1.).astype(np.float32)
    T2.by = (by - 1.).astype(np.float32)

    T2.brick_primary = ((T2.ra  >= brick.ra1 ) * (T2.ra  < brick.ra2) *
                        (T2.dec >= brick.dec1) * (T2.dec < brick.dec2))
    T2.ra_ivar  = T2.ra_ivar .astype(np.float32)
    T2.dec_ivar = T2.dec_ivar.astype(np.float32)

    # Unpack shape columns
    T2.shapeExp_r  = T2.shapeExp[:,0]
    T2.shapeExp_e1 = T2.shapeExp[:,1]
    T2.shapeExp_e2 = T2.shapeExp[:,2]
    T2.shapeDev_r  = T2.shapeDev[:,0]
    T2.shapeDev_e1 = T2.shapeDev[:,1]
    T2.shapeDev_e2 = T2.shapeDev[:,2]
    T2.shapeExp_r_ivar  = T2.shapeExp_ivar[:,0]
    T2.shapeExp_e1_ivar = T2.shapeExp_ivar[:,1]
    T2.shapeExp_e2_ivar = T2.shapeExp_ivar[:,2]
    T2.shapeDev_r_ivar  = T2.shapeDev_ivar[:,0]
    T2.shapeDev_e1_ivar = T2.shapeDev_ivar[:,1]
    T2.shapeDev_e2_ivar = T2.shapeDev_ivar[:,2]

    # For sources that had DECam flux initialization from SDSS but no
    # overlapping images (hence decam_flux_ivar = 0), zero out the DECam flux.
    T2.decam_flux[T2.decam_flux_ivar == 0] = 0.

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

        for band in [1,2,3,4]:
            dm = vega_to_ab['w%i' % band]
            fluxfactor = 10.** (dm / -2.5)
            c = 'w%i_nanomaggies' % band
            WISE.set(c, WISE.get(c) * fluxfactor)
            c = 'w%i_nanomaggies_ivar' % band
            WISE.set(c, WISE.get(c) / fluxfactor**2)

        T2.wise_flux = np.vstack([
            WISE.w1_nanomaggies, WISE.w2_nanomaggies,
            WISE.w3_nanomaggies, WISE.w4_nanomaggies]).T
        T2.wise_flux_ivar = np.vstack([
            WISE.w1_nanomaggies_ivar, WISE.w2_nanomaggies_ivar,
            WISE.w3_nanomaggies_ivar, WISE.w4_nanomaggies_ivar]).T
        T2.wise_nobs = np.vstack([
            WISE.w1_nexp, WISE.w2_nexp, WISE.w3_nexp, WISE.w4_nexp]).T
        T2.wise_fracflux = np.vstack([
            WISE.w1_profracflux,WISE.w2_profracflux,
            WISE.w3_profracflux,WISE.w4_profracflux]).T
        T2.wise_rchi2 = np.vstack([
            WISE.w1_prochi2, WISE.w2_prochi2,
            WISE.w3_prochi2, WISE.w4_prochi2]).T

    if catalogfn is not None:
        fn = catalogfn
    else:
        if outdir is None:
            outdir = '.'
        outdir = os.path.join(outdir, 'tractor', brickname[:3])
        fn = os.path.join(outdir, 'tractor-%s.fits' % brickname)
    dirnm = os.path.dirname(fn)
    trymakedirs(dirnm)

    print('Reading SFD maps...')
    sfd = SFDMap()
    filts = ['%s %s' % ('DES', f) for f in allbands]
    wisebands = ['WISE W1', 'WISE W2', 'WISE W3', 'WISE W4']
    ebv,ext = sfd.extinction(filts + wisebands, T2.ra, T2.dec, get_ebv=True)
    ext = ext.astype(np.float32)
    decam_extinction = ext[:,:len(allbands)]
    wise_extinction = ext[:,len(allbands):]
    T2.ebv = ebv.astype(np.float32)
    T2.decam_mw_transmission = 10.**(-decam_extinction / 2.5)
    if WISE is not None:
        T2.wise_mw_transmission  = 10.**(-wise_extinction  / 2.5)

    cols = [
        'brickid', 'brickname', 'objid', 'brick_primary', 'blob', 'ninblob',
        'tycho2inblob', 'type', 'ra', 'ra_ivar', 'dec', 'dec_ivar',
        'bx', 'by', 'bx0', 'by0', 'left_blob', 'out_of_bounds',
        'dchisq', 'ebv',
        'decam_flux', 'decam_flux_ivar' ]

    if AP is not None:
        cols.extend(['decam_apflux', 'decam_apflux_resid','decam_apflux_ivar'])

    cols.extend(['decam_mw_transmission', 'decam_nobs',
        'decam_rchi2', 'decam_fracflux', 'decam_fracmasked', 'decam_fracin',
        'decam_anymask', 'decam_allmask', 'decam_psfsize' ])

    if WISE is not None:
        cols.extend([
            'wise_flux', 'wise_flux_ivar',
            'wise_mw_transmission', 'wise_nobs', 'wise_fracflux',
            'wise_rchi2'])

    cols.extend([
        'fracdev', 'fracDev_ivar', 'shapeexp_r', 'shapeexp_r_ivar',
        'shapeexp_e1', 'shapeexp_e1_ivar',
        'shapeexp_e2', 'shapeexp_e2_ivar',
        'shapedev_r',  'shapedev_r_ivar',
        'shapedev_e1', 'shapedev_e1_ivar',
        'shapedev_e2', 'shapedev_e2_ivar',])

    if not no_sdss:
        cols.extend(['sdss_'+c for c in [
            'run', 'camcol', 'field', 'id', 'objid', 'parent', 'nchild',
            'objc_type', 'objc_flags', 'objc_flags2', 'flags', 'flags2',
            'tai', 'ra',  'ra_ivar', 'dec', 'dec_ivar', 'psf_fwhm', 'mjd',
            'theta_dev', 'theta_deverr', 'ab_dev', 'ab_deverr',
            'theta_exp', 'theta_experr', 'ab_exp', 'ab_experr',
            'fracdev', 'phi_dev_deg', 'phi_exp_deg',
            'psfflux',    'psfflux_ivar',   'cmodelflux', 'cmodelflux_ivar',
            'modelflux',  'modelflux_ivar', 'devflux',    'devflux_ivar',
            'expflux',    'expflux_ivar',
            'extinction', 'calib_status', 'resolve_status']])

    # TUNIT cards.
    deg='deg'
    degiv='1/deg^2'
    flux = 'nanomaggy'
    fluxiv = '1/nanomaggy^2'
    units = dict(
        ra=deg, dec=deg, ra_ivar=degiv, dec_ivar=degiv, ebv='mag',
        decam_flux=flux, decam_flux_ivar=fluxiv,
        decam_apflux=flux, decam_apflux_ivar=fluxiv, decam_apflux_resid=flux,
        wise_flux=flux, wise_flux_ivar=fluxiv,
        shapeexp_r='arcsec', shapeexp_r_ivar='1/arcsec^2',
        shapedev_r='arcsec', shapedev_r_ivar='1/arcsec^2')
    if not no_sdss:
        units.update(dict([('sdss_'+k, v) for k,v in dict(
            ra=deg, ra_ivar=degiv, dec=deg, dec_ivar=degiv,
            tai='sec', psf_fwhm='arcsec', mjd='days',
            theta_dev='arcsec', theta_exp='arcsec',
            theta_deverr='1/arcsec', theta_experr='1/arcsec',
            phi_dev_deg=deg, phi_exp_deg=deg,
            psfflux=flux, psfflux_ivar=fluxiv,
            cmodelflux=flux, cmodelflux_ivar=fluxiv,
            modelflux=flux, modelflux_ivar=fluxiv,
            devflux=flux, devflux_ivar=fluxiv,
            expflux=flux, expflux_ivar=fluxiv,
            extinction='mag').items()]))

    for i,col in enumerate(cols):
        if col in units:
            hdr.add_record(dict(name='TUNIT%i' % (i+1), value=units[col]))

    # match case to T2.
    cc = T2.get_columns()
    cclower = [c.lower() for c in cc]
    for i,c in enumerate(cols):
        if (not c in cc) and c in cclower:
            j = cclower.index(c)
            cols[i] = cc[j]

    if not no_sdss:
        _fill_sdss_columns(T2, cols)

        # Blank out all SDSS fields for sources that have moved too much.
        xyz1 = radectoxyz(T2.ra, T2.dec)
        xyz2 = radectoxyz(T2.sdss_ra, T2.sdss_dec)
        d2 = np.sum((xyz2-xyz1)**2, axis=1)
        # 1.5 arcsec
        maxd2 = np.deg2rad(1.5 / 3600.)**2
        blankout = np.flatnonzero((T2.sdss_ra != 0) * (d2 > maxd2))
        print('Blanking out', len(blankout), 'SDSS no-longer-matches')
        if len(blankout):
            Tcols = T2.get_columns()
            for c in Tcols:
                if c.startswith('sdss'):
                    x = T2.get(c)
                    x[blankout] = 0

    if write_catalog:
        T2.writeto(fn, primheader=primhdr, header=hdr, columns=cols)
        print('Wrote', fn)

    return dict(T2=T2)


def _fill_sdss_columns(T2, cols):
    # fill in any empty columns (eg, SDSS columns in areas outside the
    # footprint).  FITS types:
    fB = np.uint8
    fI = np.int16
    fJ = np.int32
    fK = np.int64
    fD = float
    fE = np.float32
    # sdss_*
    coltypes = dict(
        run=fI, camcol=fB, field=fI, id=fI, objid=fK, parent=fI, nchild=fI,
        objc_type=fJ, objc_flags=fJ, objc_flags2=fJ, ra=fD, ra_ivar=fD, dec=fD,
        dec_ivar=fD, mjd=fJ, resolve_status=fJ)
    arrtypes = dict(
        flags=fJ, flags2=fJ, tai=fD, psf_fwhm=fE, theta_dev=fE,
        theta_deverr=fE, ab_dev=fE, ab_deverr=fE, theta_exp=fE,
        theta_experr=fE, ab_exp=fE, ab_experr=fE, fracdev=fE, phi_dev_deg=fE,
        phi_exp_deg=fE, psfflux=fE, psfflux_ivar=fE, cmodelflux=fE,
        cmodelflux_ivar=fE, modelflux=fE, modelflux_ivar=fE, devflux=fE,
        devflux_ivar=fE, expflux=fE, expflux_ivar=fE, extinction=fE,
        calib_status=fJ)
    Tcols = T2.get_columns()
    for c in cols:
        if not c in Tcols:
            print('Filling in missing column', c)
            sc = c.replace('sdss_','')
            if sc in coltypes:
                T2.set(c, np.zeros(len(T2), coltypes[sc]))
            if sc in arrtypes:
                T2.set(c, np.zeros((len(T2),5), arrtypes[sc]))

def _bounce_tim_get_resamp(X):
    (tim, targetwcs) = X
    return tim_get_resamp(tim, targetwcs)

def tims_compute_resamp(mp, tims, targetwcs):
    if mp is None:
        from utils import MyMultiproc
        mp = MyMultiproc()
    R = mp.map(_bounce_tim_get_resamp, [(tim,targetwcs) for tim in tims])
    for tim,r in zip(tims, R):
        tim.resamp = r

# Pretty much only used for plots; the real deal is _coadds().
def compute_coadds(tims, bands, targetwcs, images=None,
                   get_cow=False, get_n2=False, fill_holes=True):

    W = targetwcs.get_width()
    H = targetwcs.get_height()

    coimgs = []
    cons = []
    if get_n2:
        cons2 = []
    if get_cow:
        # moo
        cowimgs = []
        wimgs = []

    for ib,band in enumerate(bands):
        coimg = np.zeros((H,W), np.float32)
        coimg2 = np.zeros((H,W), np.float32)
        con   = np.zeros((H,W), np.uint8)
        con2  = np.zeros((H,W), np.uint8)
        if get_cow:
            cowimg = np.zeros((H,W), np.float32)
            wimg  = np.zeros((H,W), np.float32)
        for itim,tim in enumerate(tims):
            if tim.band != band:
                continue
            R = tim_get_resamp(tim, targetwcs)
            if R is None:
                continue
            (Yo,Xo,Yi,Xi) = R
            nn = (tim.getInvError()[Yi,Xi] > 0)
            if images is None:
                coimg [Yo,Xo] += tim.getImage()[Yi,Xi] * nn
                coimg2[Yo,Xo] += tim.getImage()[Yi,Xi]
            else:
                coimg [Yo,Xo] += images[itim][Yi,Xi] * nn
                coimg2[Yo,Xo] += images[itim][Yi,Xi]
            con   [Yo,Xo] += nn
            if get_cow:
                cowimg[Yo,Xo] += tim.getInvvar()[Yi,Xi] * tim.getImage()[Yi,Xi]
                wimg  [Yo,Xo] += tim.getInvvar()[Yi,Xi]
            con2  [Yo,Xo] += 1
        coimg /= np.maximum(con,1)
        if fill_holes:
            coimg[con == 0] = coimg2[con == 0] / np.maximum(1, con2[con == 0])
        if get_cow:
            cowimg /= np.maximum(wimg, 1e-16)
            cowimg[wimg == 0] = coimg[wimg == 0]
            cowimgs.append(cowimg)
            wimgs.append(wimg)
        coimgs.append(coimg)
        cons.append(con)
        if get_n2:
            cons2.append(con2)

    rtn = [coimgs,cons]
    if get_cow:
        rtn.extend([cowimgs, wimgs])
    if get_n2:
        rtn.append(cons2)
    return rtn

def run_brick(brick, radec=None, pixscale=0.262,
              width=3600, height=3600,
              zoom=None,
              bands=None,
              blacklist=True,
              nblobs=None, blob=None, blobxy=None,
              pv=True, pipe=True, nsigma=6,
              simulOpt=False,
              wise=True,
              sdssInit=True,
              early_coadds=True,
              do_calibs=True,
              write_metrics=True,
              on_bricks=False,
              gaussPsf=False,
              pixPsf=False,
              splinesky=False,
              ceres=True,
              outdir=None,
              decals=None, decals_dir=None,
              unwise_dir=None,
              threads=None,
              plots=False, plots2=False, coadd_bw=False,
              plotbase=None, plotnumber=0,
              rsync=False,
              picklePattern='pickles/runbrick-%(brick)s-%%(stage)s.pickle',
              stages=['writecat'],
              force=[], forceAll=False, writePickles=True,
              checkpoint_filename=None,
              checkpoint_period=None):
    '''
    Run the full DECaLS data reduction pipeline.

    The pipeline is built out of "stages" that run in sequence.  By
    default, this function will cache the result of each stage in a
    (large) pickle file.  If you re-run, it will read from the
    prerequisite pickle file rather than re-running the prerequisite
    stage.  This can yield faster debugging times, but you almost
    certainly want to turn it off (with `writePickles=False,
    forceAll=True`) in production.

    Parameters
    ----------
    brick : string
        Brick name such as '2090m065'.  Can be None if *radec* is given.
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
    that brick's RA,Dec center will be looked up in the
    "decals-bricks.fits" file.

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

    Other options:

    - *pv*: boolean; use the Community Pipeline's WCS headers, with
      astrometric shifts from the zeropoints.fits file, converted from
      their native PV format into SIP format.

    - *pipe*: boolean; "pipeline mode"; avoid computing non-essential
      things.

    - *nsigma*: float; detection threshold in sigmas.

    - *simulOpt*: boolean; during fitting, if a blob contains multiple
      sources, run a step of fitting the sources simultaneously?

    - *wise*: boolean; run WISE forced photometry?

    - *sdssInit*: boolean; initialize sources from the SDSS catalogs?

    - *early_coadds*: boolean; generate the early coadds?

    - *do_calibs*: boolean; run the calibration preprocessing steps?

    - *write_metrics*: boolean; write out a variety of useful metrics

    - *on_bricks*: boolean; tractor-on-bricks?

    - *gaussPsf*: boolean; use a simpler single-component Gaussian PSF model?

    - *pixPsf*: boolean; use the pixelized PsfEx PSF model and FFT convolution?

    - *splinesky*: boolean; use the splined sky model (default is constant)?

    - *ceres*: boolean; use Ceres Solver when possible?

    - *outdir*: string; base directory for output files; default "."

    - *decals*: a "Decals" object (see common.Decals), which is in
      charge of the list of bricks and CCDs to be handled, and also
      creates DecamImage objects.

    - *decals_dir*: string; default $DECALS_DIR environment variable;
      where to look for files including calibration files, tables of
      CCDs and bricks, image data, etc.

    - *unwise_dir*: string; default unwise-coadds; where to look for
      unWISE coadd files.  This may be a colon-separated list of
      directories to search in order.

    - *threads*: integer; how many CPU cores to use

    Plotting options:

    - *coadd_bw*: boolean: if only one band is available, make B&W coadds?
    - *plots*: boolean; make a bunch of plots?
    - *plots2*: boolean; make a bunch more plots?
    - *plotbase*: string, default brick-BRICK, the plot filename prefix.
    - *plotnumber*: integer, default 0, starting number for plot filenames.

    Options regarding the "stages":

    - *picklePattern*: string; filename for 'pickle' files
    - *stages*: list of strings; stages (functions stage_*) to run.

    - *force*: list of strings; prerequisite stages that will be run
      even if pickle files exist.
    - *forceAll*: boolean; run all stages, ignoring all pickle files.
    - *writePickles*: boolean; write pickle files after each stage?

    Raises
    ------
    RunbrickError
        If an invalid brick name is given.
    NothingToDoError
        If no CCDs, or no photometric CCDs, overlap the given brick or region.

    '''

    from astrometry.util.stages import CallGlobalTime, runstage
    from astrometry.util.multiproc import multiproc

    from legacypipe.utils import MyMultiproc

    initargs = {}
    kwargs = {}

    forceStages = [s for s in stages]
    forceStages.extend(force)

    if forceAll:
        kwargs.update(forceall=True)

    if radec is not None:
        print('RA,Dec:', radec)
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
        print('Parsed RA,Dec', ra,dec)
        initargs.update(ra=ra, dec=dec)
        if brick is None:
            brick = ('custom-%06i%s%05i' %
                         (int(1000*ra), 'm' if dec < 0 else 'p',
                          int(1000*np.abs(dec))))
    initargs.update(brickname=brick,
                    decals=decals)

    stagefunc = CallGlobalTime('stage_%s', globals())

    plot_base_default = 'brick-%(brick)s'
    if plotbase is None:
        plotbase = plot_base_default
    ps = PlotSequence(plotbase % dict(brick=brick))
    initargs.update(ps=ps)

    if plotnumber:
        ps.skipto(plotnumber)

    kwargs.update(ps=ps, nsigma=nsigma, gaussPsf=gaussPsf, pixPsf=pixPsf,
                  use_blacklist=blacklist,
                  splinesky=splinesky,
                  simul_opt=simulOpt, pipe=pipe,
                  no_sdss=not(sdssInit),
                  use_ceres=ceres,
                  do_calibs=do_calibs,
                  write_metrics=write_metrics,
                  on_bricks=on_bricks,
                  outdir=outdir, decals_dir=decals_dir, unwise_dir=unwise_dir,
                  plots=plots, plots2=plots2, coadd_bw=coadd_bw,
                  rsync=rsync,
                  force=forceStages, write=writePickles)

    if checkpoint_filename is not None:
        kwargs.update(checkpoint_filename=checkpoint_filename)
        if checkpoint_period is not None:
            kwargs.update(checkpoint_period=checkpoint_period)

    if threads and threads > 1:
        from astrometry.util.timingpool import TimingPool, TimingPoolMeas
        pool = TimingPool(threads, initializer=runbrick_global_init,
                          initargs=[])
        Time.add_measurement(TimingPoolMeas(pool, pickleTraffic=False))
        mp = MyMultiproc(None, pool=pool)
    else:
        mp = MyMultiproc(init=runbrick_global_init, initargs=[])
    kwargs.update(mp=mp)

    if nblobs is not None:
        kwargs.update(nblobs=nblobs)
    if blob is not None:
        kwargs.update(blob0=blob)
    if blobxy is not None:
        kwargs.update(blobxy=blobxy)

    picklePattern = picklePattern % dict(brick=brick)

    prereqs = {
        'tims':None,
        'mask_junk': 'tims',
        'srcs': 'mask_junk',

        # fitblobs: see below

        'fitblobs_finish':'fitblobs',
        'coadds': 'fitblobs_finish',

        # wise_forced: see below

        'fitplots': 'fitblobs_finish',
        'psfplots': 'tims',
        'initplots': 'srcs',

        }

    if early_coadds:
        prereqs.update({
            'image_coadds':'srcs',
            'fitblobs':'image_coadds',
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

    initargs.update(W=width, H=height, pixscale=pixscale,
                    target_extent=zoom)
    if bands is not None:
        initargs.update(bands=bands)

    t0 = Time()

    for stage in stages:
        runstage(stage, picklePattern, stagefunc, prereqs=prereqs,
                 initial_args=initargs, **kwargs)

    print('All done:', Time()-t0)
    mp.report(threads)


def get_parser():
    import argparse
    de = ('Main "pipeline" script for the Dark Energy Camera Legacy Survey ' +
          '(DECaLS) data reductions.')

    ep = '''
e.g., to run a small field containing a cluster:

python -u legacypipe/runbrick.py --plots --brick 2440p070 --zoom 1900 2400 450 950 -P pickles/runbrick-cluster-%%s.pickle

'''
    parser = argparse.ArgumentParser(description=de,epilog=ep)
    parser.add_argument(
        '-f', '--force-stage', dest='force', action='append', default=[],
        help="Force re-running the given stage(s) -- don't read from pickle.")
    parser.add_argument('-F', '--force-all', dest='forceall',
                        action='store_true', help='Force all stages to run')
    parser.add_argument('-s', '--stage', dest='stage', default=[],
                        action='append', help="Run up to the given stage(s)")
    parser.add_argument('-n', '--no-write', dest='write', default=True,
                        action='store_false')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')

    parser.add_argument('--checkpoint', default=None,
                        help='Write to checkpoint file?')
    parser.add_argument(
        '--checkpoint-period', type=int, default=None,
        help='Period for writing checkpoint files, in seconds; default 600')

    parser.add_argument('-b', '--brick',
        help='Brick name to run; required unless --radec is given')

    parser.add_argument(
        '--radec', nargs=2,
        help='RA,Dec center for a custom location (not a brick)')
    parser.add_argument('--pixscale', type=float, default=0.262,
                        help='Pixel scale of the output coadds (arcsec/pixel)')

    parser.add_argument('-d', '--outdir', help='Set output base directory')
    parser.add_argument('--decals-dir', type=str, default=None,
                        help='Overwrite the $DECALS_DIR environment variable')

    parser.add_argument('--threads', type=int, help='Run multi-threaded')
    parser.add_argument('-p', '--plots', dest='plots', action='store_true',
                        help='Per-blob plots?')
    parser.add_argument('--plots2', action='store_true',
                        help='More plots?')

    parser.add_argument(
        '-P', '--pickle', dest='picklepat',
        help='Pickle filename pattern, default %(default)s',
        default='pickles/runbrick-%(brick)s-%%(stage)s.pickle')

    parser.add_argument('--plot-base',
                        help='Base filename for plots, default brick-BRICK')
    parser.add_argument('--plot-number', type=int, default=0,
                        help='Set PlotSequence starting number')

    parser.add_argument('-W', '--width', type=int, default=3600,
                        help='Target image width, default %(default)i')
    parser.add_argument('-H', '--height', type=int, default=3600,
                        help='Target image height, default %(default)i')

    parser.add_argument(
        '--zoom', type=int, nargs=4,
        help='Set target image extent (default "0 3600 0 3600")')

    parser.add_argument('--no-ceres', dest='ceres', default=True,
                        action='store_false', help='Do not use Ceres Solver')

    parser.add_argument('--nblobs', type=int,help='Debugging: only fit N blobs')
    parser.add_argument('--blob', type=int, help='Debugging: start with blob #')
    parser.add_argument(
        '--blobxy', type=int, nargs=2, default=None, action='append',
        help=('Debugging: run the single blob containing pixel <bx> <by>; '+
              'this option can be repeated to run multiple blobs.'))

    parser.add_argument(
        '--no-pv', dest='pv', default='True', action='store_false',
        help=('Do not use Community Pipeline WCS with PV distortion terms '+
              '-- solve using Astrometry.net'))

    parser.add_argument('--pipe', default=False, action='store_true',
                        help='"pipeline" mode')

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

    parser.add_argument('--skip-metrics', dest='write_metrics', default=True,
                        action='store_false',
                        help='Do not generate the metrics directory and files')

    parser.add_argument('--nsigma', type=float,
                        help='Set N sigma source detection thresh')

    parser.add_argument(
        '--simul-opt', action='store_true', default=False,
        help='Do simultaneous optimization after model selection')

    parser.add_argument('--no-wise', action='store_true', default=False,
                        help='Skip unWISE forced photometry')

    parser.add_argument(
        '--unwise-dir', default=None,
        help='Base directory for unWISE coadds; may be a colon-separated list')

    parser.add_argument('--no-sdss', action='store_true', default=False,
                        help='Do not initialize from SDSS')

    parser.add_argument('--no-early-coadds', action='store_true', default=False,
                        help='Skip making the early coadds')

    parser.add_argument('--gpsf', action='store_true', default=False,
                        help='Use a fixed single-Gaussian PSF')
    parser.add_argument(
        '--pixpsf', action='store_true', default=False,
        help='Use the pixelized PsfEx PSF model, and FFT convolution')

    parser.add_argument('--splinesky', action='store_true', default=False,
                        help='Use flexible sky model?')

    parser.add_argument(
        '--coadd-bw', action='store_true', default=False,
        help='Create grayscale coadds if only one band is available?')

    parser.add_argument('--on-bricks', action='store_true', default=False,
                        help='Tractor-on-bricks?')

    parser.add_argument(
        '--no-blacklist', dest='blacklist', default=True, action='store_false',
        help='Do not blacklist some proposals?')

    parser.add_argument(
        '--rsync', default=False, action='store_true',
        help=('Rather than running calibrations, rsync from NERSC.  Also '+
              'rsync missing image file inputs'))
    return parser

def get_runbrick_kwargs(opt):
    if opt.brick is not None and opt.radec is not None:
        print('Only ONE of --brick and --radec may be specified.')
        return -1

    if opt.check_done or opt.skip or opt.skip_coadd:
        outdir = opt.outdir
        if outdir is None:
            outdir = '.'
        brickname = opt.brick
        if opt.skip_coadd:
            fn = os.path.join(outdir, 'coadd', brickname[:3], brickname,
                              'decals-%s-image.jpg' % brickname)
        else:
            fn = os.path.join(outdir, 'tractor', brickname[:3],
                              'tractor-%s.fits' % brickname)
        print('Checking for', fn)
        exists = os.path.exists(fn)
        if opt.skip_coadd and exists:
            return 0
        if exists:
            try:
                T = fits_table(fn)
                print('Read', len(T), 'sources from', fn)
            except:
                print('Failed to read file', fn)
                exists = False

        if opt.skip:
            if exists:
                return 0
        elif opt.check_done:
            if not exists:
                print('Does not exist:', fn)
                return -1
            print('Found:', fn)
            return 0

    if len(opt.stage) == 0:
        opt.stage.append('writecat')

    kwa = {}
    if opt.nsigma:
        kwa.update(nsigma=opt.nsigma)
    if opt.no_sdss:
        kwa.update(sdssInit=False)
    if opt.no_wise:
        kwa.update(wise=False)
    if opt.no_early_coadds:
        kwa.update(early_coadds=False)

    if opt.unwise_dir is None:
        opt.unwise_dir = os.environ.get('UNWISE_COADDS_DIR', None)

    kwa.update(
        radec=opt.radec, pixscale=opt.pixscale,
        width=opt.width, height=opt.height, zoom=opt.zoom,
        blacklist=opt.blacklist,
        pv=opt.pv,
        threads=opt.threads, ceres=opt.ceres,
        do_calibs=opt.do_calibs,
        write_metrics=opt.write_metrics,
        on_bricks=opt.on_bricks,
        gaussPsf=opt.gpsf, pixPsf=opt.pixpsf, splinesky=opt.splinesky,
        simulOpt=opt.simul_opt,
        nblobs=opt.nblobs, blob=opt.blob, blobxy=opt.blobxy,
        pipe=opt.pipe, outdir=opt.outdir, decals_dir=opt.decals_dir,
        unwise_dir=opt.unwise_dir,
        plots=opt.plots, plots2=opt.plots2,
        coadd_bw=opt.coadd_bw,
        plotbase=opt.plot_base, plotnumber=opt.plot_number,
        force=opt.force, forceAll=opt.forceall,
        stages=opt.stage, writePickles=opt.write,
        picklePattern=opt.picklepat,
        rsync=opt.rsync,
        checkpoint_filename=opt.checkpoint,
        checkpoint_period=opt.checkpoint_period,
        )
    return kwa
    
def main():
    import logging
    from astrometry.util.ttime import MemMeas, CpuMeas
    import datetime

    print()
    print('runbrick.py starting at', datetime.datetime.now().isoformat())
    print('Command-line args:', sys.argv)
    print()

    parser = get_parser()
    opt = parser.parse_args()
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1
    kwargs = get_runbrick_kwargs(opt)
    if kwargs == -1:
        return -1

    if opt.verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)

    Time.add_measurement(MemMeas)
    plt.figure(figsize=(12,9))
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.95,
                        hspace=0.2, wspace=0.05)

    try:
        run_brick(opt.brick, **kwargs)
    except NothingToDoError as e:
        print()
        print(e.message)
        print()
        return 0
    except RunbrickError as e:
        print()
        print(e.message)
        print()
        return -1
    return 0

def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace

if __name__ == '__main__':
    #sys.settrace(trace)
    sys.exit(main())


# Test bricks & areas

# A single, fairly bright star
# python -u legacypipe/runbrick.py -b 1498p017 -P 'pickles/runbrick-z-%(brick)s-%%(stage)s.pickle' --zoom 1900 2000 2700 2800 --pixpsf --splinesky --no-sdss

# python -u legacypipe/runbrick.py -b 0001p000 -P 'pickles/runbrick-z-%(brick)s-%%(stage)s.pickle' --zoom 80 380 2970 3270 --pixpsf --splinesky --no-sdss

# 80, 380, 2970, 3270

    # not (380, 480, 0, 100)
# decmin 0.0851497
# decmax 0.106983
# ramin 0.228344
# ramax 0.250178
