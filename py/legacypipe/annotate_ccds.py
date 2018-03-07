from __future__ import print_function
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.starutil_numpy import degrees_between
from astrometry.util.util import Tan
from astrometry.util.miscutils import polygon_area
from legacypipe.survey import LegacySurveyData
import tractor
import tractor.sfd

'''
Note: can parallelize this via:

-create script like mzls.sh:

#! /bin/bash
export LEGACY_SURVEY_DIR=/global/cscratch1/sd/desiproc/dr4/dr4_fixes/legacypipe-dir/
python -u legacypipe/annotate-ccds.py --mzls --ccds /global/projecta/projectdirs/cosmo/work/dr4/survey-ccds-mzls.fits.gz --threads 1 --piece $1

- seq 0 805 | qdo load mzls -
- qdo launch mzls 1 --cores_per_worker=1 --keep_env --batchqueue shared --walltime 4:00:00 --script ./mzls.sh --batchopts "-a 0-31"
- once finished, run annotate-ccds.py again to merge the files together:

LEGACY_SURVEY_DIR=/global/cscratch1/sd/desiproc/dr4/dr4_fixes/legacypipe-dir/
python -u legacypipe/annotate-ccds.py --mzls --ccds /global/projecta/projectdirs/cosmo/work/dr4/survey-ccds-mzls.fits.gz --threads 1

Can add kd-tree data structure to this resulting annotated-ccds file like this:
# Astrometry.net's 'startree' program:
# > startree -i annotated-ccds.fits -o /tmp/ann.kd -P -k -n ccds
# > fitsgetext -i /tmp/ann.kd -o annotated-ccds.kd.fits -e 0 -e 6 -e 1 -e 2 -e 3 -e 4 -e 5

'''

def annotate(ccds, survey, mzls=False, normalizePsf=False):
    # File from the "observing" svn repo:
    if mzls:
        # https://desi.lbl.gov/svn/decam/code/mosaic3/trunk
        tiles = fits_table('mosaic-tiles_obstatus.fits')
    else:
        # https://desi.lbl.gov/svn/decam/code/observing/trunk
        tiles = fits_table('decam-tiles_obstatus.fits')

    # Map tile IDs back to index in the obstatus file.
    tileid_to_index = np.empty(max(tiles.tileid)+1, int)
    tileid_to_index[:] = -1
    tileid_to_index[tiles.tileid] = np.arange(len(tiles))

    assert('ccd_cuts' in ccds.get_columns())

    gaussgalnorm = np.zeros(len(ccds), np.float32)

    for iccd,ccd in enumerate(ccds):
        print('Reading CCD %i of %i:' % (iccd+1, len(ccds)), 'file', ccd.image_filename, 'CCD', ccd.expnum, ccd.ccdname)
        try:
            im = survey.get_image_object(ccd)
        except:
            print('Failed to get_image_object()')
            import traceback
            traceback.print_exc()
            continue
        print('Reading CCD %i of %i:' % (iccd+1, len(ccds)), im, 'file', ccd.image_filename, 'CCD', ccd.ccdname)

        X = im.get_good_image_subregion()
        for i,x in enumerate(X):
            if x is not None:
                ccds.good_region[iccd,i] = x

        W,H = ccd.width, ccd.height

        kwargs = dict(pixPsf=True, splinesky=True, subsky=False,
                      pixels=False, dq=False, invvar=False,
                      normalizePsf=normalizePsf)

        psf = None
        wcs = None
        sky = None
        try:
            tim = im.get_tractor_image(**kwargs)

        except:
            print('Failed to get_tractor_image')
            import traceback
            traceback.print_exc()
            continue

        if tim is None:
            continue

        psf = tim.psf
        wcs = tim.wcs.wcs
        sky = tim.sky
        hdr = tim.primhdr

        # print('CCD fwhm:', ccd.fwhm)
        # print('im fwhm:', im.fwhm)
        # print('tim psf_fwhm', tim.psf_fwhm)
        # print('tim psf_sigma:', tim.psf_sigma)
        # print('Got PSF', psf)
        # print('Got sky', type(sky))
        # print('Got WCS', wcs)
        # print('sig1', tim.sig1)

        ccds.humidity[iccd] = hdr.get('HUMIDITY')
        ccds.outtemp[iccd]  = hdr.get('OUTTEMP')

        ccds.sig1[iccd] = tim.sig1
        ccds.plver[iccd] = tim.plver

        # parse 'DECaLS_15150_r' to get tile number
        obj = ccd.object.strip()
        words = obj.split('_')
        tile = None
        if len(words) == 3 and (
            ((not mzls) and (words[0] == 'DECaLS')) or
            (     mzls  and (words[0] in ['MzLS','MOSAIC']))):
            try:
                tileid = int(words[1])
                tile = tiles[tileid_to_index[tileid]]
                assert(tile.tileid == tileid)
            except:
                import traceback
                traceback.print_ext()
                pass

        if tile is not None:
            ccds.tileid  [iccd] = tile.tileid
            ccds.tilepass[iccd] = tile.get('pass')
            ccds.tileebv [iccd] = tile.ebv_med

        # Instantiate PSF on a grid
        S = 32
        xx = np.linspace(1+S, W-S, 5)
        yy = np.linspace(1+S, H-S, 5)
        xx,yy = np.meshgrid(xx, yy)
        psfnorms = []
        galnorms = []
        for x,y in zip(xx.ravel(), yy.ravel()):

            # HACK -- DR4 PSF sampling issue
            tim.psf = psf.constantPsfAt(x, y)

            p = im.psf_norm(tim, x=x, y=y)
            g = im.galaxy_norm(tim, x=x, y=y)
            psfnorms.append(p)
            galnorms.append(g)

        tim.psf = psf

        ccds.psfnorm_mean[iccd] = np.mean(psfnorms)
        ccds.psfnorm_std [iccd] = np.std (psfnorms)
        ccds.galnorm_mean[iccd] = np.mean(galnorms)
        ccds.galnorm_std [iccd] = np.std (galnorms)

        #print('psf norm', ccds.psfnorm_mean[iccd])
        #print('gal norm', ccds.galnorm_mean[iccd])

        # PSF in center of field
        cx,cy = (W+1)/2., (H+1)/2.
        p = psf.getPointSourcePatch(cx, cy).patch
        ph,pw = p.shape
        px,py = np.meshgrid(np.arange(pw), np.arange(ph))
        psum = np.sum(p)
        # print('psum', psum)
        p /= psum
        # centroids
        cenx = np.sum(p * px)
        ceny = np.sum(p * py)
        # print('cenx,ceny', cenx,ceny)
        # second moments
        x2 = np.sum(p * (px - cenx)**2)
        y2 = np.sum(p * (py - ceny)**2)
        xy = np.sum(p * (px - cenx)*(py - ceny))
        # semi-major/minor axes and position angle
        theta = np.rad2deg(np.arctan2(2 * xy, x2 - y2) / 2.)
        theta = np.abs(theta) * np.sign(xy)
        s = np.sqrt(((x2 - y2)/2.)**2 + xy**2)
        a = np.sqrt((x2 + y2) / 2. + s)
        b = np.sqrt((x2 + y2) / 2. - s)
        ell = 1. - b/a

        # print('PSF second moments', x2, y2, xy)
        # print('PSF position angle', theta)
        # print('PSF semi-axes', a, b)
        # print('PSF ellipticity', ell)

        ccds.psf_mx2[iccd] = x2
        ccds.psf_my2[iccd] = y2
        ccds.psf_mxy[iccd] = xy
        ccds.psf_a[iccd] = a
        ccds.psf_b[iccd] = b
        ccds.psf_theta[iccd] = theta
        ccds.psf_ell  [iccd] = ell

        #print('Computing Gaussian approximate PSF quantities...')
        # Galaxy norm using Gaussian approximation of PSF.
        realpsf = tim.psf

        #print('FWHM', ccds.fwhm[i])
        #print('-> sigma', ccds.fwhm[i] / 2.35)
        #print('tim.PSF sigma', tim.psf_sigma)

        tim.psf = im.read_psf_model(0, 0, gaussPsf=True,
                                    psf_sigma=tim.psf_sigma)
        gaussgalnorm[iccd] = im.galaxy_norm(tim, x=cx, y=cy)

        psfnorm = im.psf_norm(tim)
        pnorm = 1./(2. * np.sqrt(np.pi) * tim.psf_sigma)
        #print('Gaussian PSF norm:', psfnorm, 'vs analytic', pnorm)
        #print('Gaussian gal norm:', gaussgalnorm[iccd])

        tim.psf = realpsf
        
        has_skygrid = hasattr(sky, 'evaluateGrid')

        # Sky -- evaluate on a grid (every ~10th pixel)
        if has_skygrid:
            skygrid = sky.evaluateGrid(np.linspace(0, ccd.width-1,  int(1+ccd.width/10)),
                                       np.linspace(0, ccd.height-1, int(1+ccd.height/10)))
            ccds.meansky[iccd] = np.mean(skygrid)
            ccds.stdsky[iccd]  = np.std(skygrid)
            ccds.maxsky[iccd]  = skygrid.max()
            ccds.minsky[iccd]  = skygrid.min()
        else:
            skyval = sky.getConstant()
            ccds.meansky[iccd] = skyval
            ccds.stdsky[iccd]  = 0.
            ccds.maxsky[iccd]  = skyval
            ccds.minsky[iccd]  = skyval

        # WCS
        ccds.ra0[iccd],ccds.dec0[iccd] = wcs.pixelxy2radec(1, 1)
        ccds.ra1[iccd],ccds.dec1[iccd] = wcs.pixelxy2radec(1, H)
        ccds.ra2[iccd],ccds.dec2[iccd] = wcs.pixelxy2radec(W, H)
        ccds.ra3[iccd],ccds.dec3[iccd] = wcs.pixelxy2radec(W, 1)

        midx, midy = (W+1)/2., (H+1)/2.
        rc,dc  = wcs.pixelxy2radec(midx, midy)
        ra,dec = wcs.pixelxy2radec([1,W,midx,midx], [midy,midy,1,H])
        ccds.dra [iccd] = max(degrees_between(ra, dc+np.zeros_like(ra),
                                              rc, dc))
        ccds.ddec[iccd] = max(degrees_between(rc+np.zeros_like(dec), dec,
                                              rc, dc))
        ccds.ra_center [iccd] = rc
        ccds.dec_center[iccd] = dc

        # Compute scale change across the chip
        # how many pixels to step
        step = 10
        xx = np.linspace(1+step, W-step, 5)
        yy = np.linspace(1+step, H-step, 5)
        xx,yy = np.meshgrid(xx, yy)
        pixscale = []
        for x,y in zip(xx.ravel(), yy.ravel()):
            sx = [x-step, x-step, x+step, x+step, x-step]
            sy = [y-step, y+step, y+step, y-step, y-step]
            sr,sd = wcs.pixelxy2radec(sx, sy)
            rc,dc = wcs.pixelxy2radec(x, y)
            # project around a tiny little TAN WCS at (x,y), with 1" pixels
            locwcs = Tan(rc, dc, 0., 0., 1./3600, 0., 0., 1./3600, 1., 1.)
            ok,lx,ly = locwcs.radec2pixelxy(sr, sd)
            #print('local x,y:', lx, ly)
            A = polygon_area((lx, ly))
            pixscale.append(np.sqrt(A / (2*step)**2))
        # print('Pixel scales:', pixscale)
        ccds.pixscale_mean[iccd] = np.mean(pixscale)
        ccds.pixscale_min[iccd] = min(pixscale)
        ccds.pixscale_max[iccd] = max(pixscale)
        ccds.pixscale_std[iccd] = np.std(pixscale)

        ccds.annotated[iccd] = True

    sfd = tractor.sfd.SFDMap()
    allbands = 'ugrizY'
    filts = ['%s %s' % ('DES', f) for f in allbands]
    wisebands = ['WISE W1', 'WISE W2', 'WISE W3', 'WISE W4']
    ebv,ext = sfd.extinction(filts + wisebands, ccds.ra_center,
                             ccds.dec_center, get_ebv=True)

    ext[np.logical_not(ccds.annotated),:] = 0.
    ebv[np.logical_not(ccds.annotated)] = 0.

    ext = ext.astype(np.float32)
    ccds.ebv = ebv.astype(np.float32)
    ccds.decam_extinction = ext[:,:len(allbands)]
    ccds.wise_extinction = ext[:,len(allbands):]

    # Depth
    detsig1 = ccds.sig1 / ccds.psfnorm_mean
    depth = 5. * detsig1
    # that's flux in nanomaggies -- convert to mag
    ccds.psfdepth = -2.5 * (np.log10(depth) - 9)

    detsig1 = ccds.sig1 / ccds.galnorm_mean
    depth = 5. * detsig1
    # that's flux in nanomaggies -- convert to mag
    ccds.galdepth = -2.5 * (np.log10(depth) - 9)
    
    # Depth using Gaussian FWHM.
    psf_sigma = ccds.fwhm / 2.35
    gnorm = 1./(2. * np.sqrt(np.pi) * psf_sigma)
    detsig1 = ccds.sig1 / gnorm
    depth = 5. * detsig1
    # that's flux in nanomaggies -- convert to mag
    ccds.gausspsfdepth = -2.5 * (np.log10(depth) - 9)

    # Gaussian galaxy depth
    detsig1 = ccds.sig1 / gaussgalnorm
    depth = 5. * detsig1
    # that's flux in nanomaggies -- convert to mag
    ccds.gaussgaldepth = -2.5 * (np.log10(depth) - 9)

    # NaN depths -> 0
    for X in [ccds.psfdepth, ccds.galdepth, ccds.gausspsfdepth, ccds.gaussgaldepth]:
        X[np.logical_not(np.isfinite(X))] = 0.


def init_annotations(ccds):
    ccds.annotated = np.zeros(len(ccds), bool)

    ccds.good_region = np.empty((len(ccds), 4), np.int16)
    ccds.good_region[:,:] = -1

    ccds.ra0  = np.zeros(len(ccds), np.float64)
    ccds.dec0 = np.zeros(len(ccds), np.float64)
    ccds.ra1  = np.zeros(len(ccds), np.float64)
    ccds.dec1 = np.zeros(len(ccds), np.float64)
    ccds.ra2  = np.zeros(len(ccds), np.float64)
    ccds.dec2 = np.zeros(len(ccds), np.float64)
    ccds.ra3  = np.zeros(len(ccds), np.float64)
    ccds.dec3 = np.zeros(len(ccds), np.float64)

    ccds.dra  = np.zeros(len(ccds), np.float32)
    ccds.ddec = np.zeros(len(ccds), np.float32)
    ccds.ra_center  = np.zeros(len(ccds), np.float64)
    ccds.dec_center = np.zeros(len(ccds), np.float64)

    ccds.sig1 = np.zeros(len(ccds), np.float32)

    ccds.meansky = np.zeros(len(ccds), np.float32)
    ccds.stdsky  = np.zeros(len(ccds), np.float32)
    ccds.maxsky  = np.zeros(len(ccds), np.float32)
    ccds.minsky  = np.zeros(len(ccds), np.float32)

    ccds.pixscale_mean = np.zeros(len(ccds), np.float32)
    ccds.pixscale_std  = np.zeros(len(ccds), np.float32)
    ccds.pixscale_max  = np.zeros(len(ccds), np.float32)
    ccds.pixscale_min  = np.zeros(len(ccds), np.float32)

    ccds.psfnorm_mean = np.zeros(len(ccds), np.float32)
    ccds.psfnorm_std  = np.zeros(len(ccds), np.float32)
    ccds.galnorm_mean = np.zeros(len(ccds), np.float32)
    ccds.galnorm_std  = np.zeros(len(ccds), np.float32)

    # 2nd moments
    ccds.psf_mx2 = np.zeros(len(ccds), np.float32)
    ccds.psf_my2 = np.zeros(len(ccds), np.float32)
    ccds.psf_mxy = np.zeros(len(ccds), np.float32)
    #
    ccds.psf_a = np.zeros(len(ccds), np.float32)
    ccds.psf_b = np.zeros(len(ccds), np.float32)
    ccds.psf_theta = np.zeros(len(ccds), np.float32)
    ccds.psf_ell   = np.zeros(len(ccds), np.float32)

    ccds.humidity = np.zeros(len(ccds), np.float32)
    ccds.outtemp  = np.zeros(len(ccds), np.float32)

    ccds.tileid   = np.zeros(len(ccds), np.int32)
    ccds.tilepass = np.zeros(len(ccds), np.uint8)
    ccds.tileebv  = np.zeros(len(ccds), np.float32)

    ccds.plver = np.array([' '*6] * len(ccds))


def main(outfn='ccds-annotated.fits', ccds=None, **kwargs):
    survey = LegacySurveyData(ccds=ccds)
    if ccds is None:
        ccds = survey.get_ccds()

    # Set to True if we successfully read the calibration products and computed
    # annotated values
    init_annotations(ccds)

    annotate(ccds, **kwargs)

    
    print('Writing to', outfn)
    ccds.writeto(outfn)
    print('Wrote', outfn)

def _bounce_main(X):
    (name, i, ccds, force, mzls, normalizePsf) = X
    try:
        outfn = 'ccds-annotated/ccds-annotated-%s-%03i.fits' % (name, i)
        if (not force) and os.path.exists(outfn):
            print('Already exists:', outfn)
            return
        main(outfn=outfn, ccds=ccds, mzls=mzls, normalizePsf=normalizePsf)
    except:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Produce annotated CCDs file by reading CCDs file + calibration products')
    parser.add_argument('--part', action='append', help='CCDs file to read, survey-ccds-X.fits.gz, default: ["decals","nondecals","extra"].  Can be repeated.', default=[])
    parser.add_argument('--ccds', action='append', help='CCDs file to read; can be repeated', default=[])
    parser.add_argument('--mzls', action='store_true', default=False, help='MzLS (default: DECaLS')

    parser.add_argument('--normalize-psf', dest='normalizePsf', action='store_true', default=False)

    parser.add_argument('--force', action='store_true', default=False,
                        help='Ignore ccds-annotated/* files and re-run')
    parser.add_argument('--threads', type=int, help='Run multi-threaded', default=4)

    parser.add_argument('--piece', type=int, help='Run only a single subset of CCDs',
                        default=None)
    parser.add_argument('--update',
                        help='Read an existing annotated-CCDs file and update rows where annotated==False')

    opt = parser.parse_args()
    survey = LegacySurveyData()

    if opt.update is not None:
        fn = opt.update
        ccds = fits_table(fn)
        print('Read', len(ccds), 'from', fn)
        I = np.flatnonzero(ccds.annotated == False)
        print(len(I), 'CCDs have annotated==False')

        upccds = ccds[I]
        annotate(upccds, mzls=opt.mzls, normalizePsf=opt.normalizePsf)
        ccds[I] = upccds

        fn = 'updated-' + os.path.basename(opt.update)
        ccds.writeto(fn)
        print('Wrote', fn)

        sys.exit(0)

    from astrometry.util.multiproc import *
    mp = multiproc(opt.threads)
    N = 100
    
    if len(opt.part) == 0 and len(opt.ccds) == 0:
        opt.part.append('decals')
        opt.part.append('nondecals')
        opt.part.append('extra')

    ccdfns = opt.ccds
    for p in opt.part:
        ccdfns.append(os.path.join(survey.survey_dir, 'survey-ccds-%s.fits.gz' % p))

    for fn in ccdfns:
        print()
        print('Reading', fn)
        print()

        name = os.path.basename(fn)
        name = name.replace('survey-ccds-', '')
        name = name.replace('.fits', '')
        name = name.replace('.gz', '')
        print('Name', name)

        ## Quick check for existing output filename
        if opt.piece is not None and not opt.force:
            outfn = 'ccds-annotated/ccds-annotated-%s-%03i.fits' % (name, opt.piece)
            if os.path.exists(outfn):
                print('Already exists:', outfn)
                sys.exit(0)

        args = []
        i = 0
        ccds = fits_table(fn)
        ccds = survey.cleanup_ccds_table(ccds)

        if opt.piece is not None:
            c = ccds[opt.piece*N:]
            c = c[:N]
            _bounce_main((name, opt.piece, c, opt.force, opt.mzls, opt.normalizePsf))
            sys.exit(0)

        while len(ccds):
            c = ccds[:N]
            ccds = ccds[N:]
            args.append((name, i, c, opt.force, opt.mzls, opt.normalizePsf))
            i += 1
        print('Split CCDs file into', len(args), 'pieces')
        print('sizes:', [len(a[2]) for a in args])
        mp.map(_bounce_main, args)

        # reassemble outputs
        TT = [fits_table('ccds-annotated/ccds-annotated-%s-%03i.fits' % (name,i))
              for name,i,nil,nil,nil,nil in args]
        T = merge_tables(TT)

        T.object = T.object.astype('S37')

        T.writeto('survey-ccds-annotated-%s.fits' % name)

