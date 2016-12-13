from __future__ import print_function

import os
import numpy as np
import tempfile

import fitsio

from astrometry.util.fits import fits_table
from astrometry.util.file import trymakedirs
from astrometry.util.util import Tan

from legacypipe.survey import LegacySurveyData, wcs_for_brick

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='This script creates small self-contained data sets that '
        'are useful for test cases of the pipeline codes.')

    parser.add_argument('ccds', help='CCDs table describing region to grab')
    parser.add_argument('outdir', help='Output directory name')
    parser.add_argument('brick', help='Brick containing these images')

    parser.add_argument('--wise', help='For WISE outputs, give the path to a WCS file describing the sub-brick region of interest, eg, a coadd image')
    parser.add_argument('--fpack', action='store_true', default=False)
    parser.add_argument('--pad', action='store_true', default=False,
                        help='Keep original image size, but zero out pixels outside ROI')
    
    args = parser.parse_args()

    C = fits_table(args.ccds)
    print(len(C), 'CCDs in', args.ccds)
    C.camera = np.array([c.strip() for c in C.camera])
    
    survey = LegacySurveyData()
    bricks = survey.get_bricks_readonly()
    outbricks = bricks[np.array([n == args.brick for n in bricks.brickname])]
    assert(len(outbricks) == 1)
    
    outsurvey = LegacySurveyData(survey_dir = args.outdir)
    trymakedirs(args.outdir)
    outbricks.writeto(os.path.join(args.outdir, 'survey-bricks.fits.gz'))

    targetwcs = wcs_for_brick(outbricks[0])
    H,W = targetwcs.shape
    
    tycho = fits_table(os.path.join(survey.get_survey_dir(), 'tycho2.fits.gz'))
    print('Read', len(tycho), 'Tycho-2 stars')
    ok,tx,ty = targetwcs.radec2pixelxy(tycho.ra, tycho.dec)
    margin = 100
    tycho.cut(ok * (tx > -margin) * (tx < W+margin) *
              (ty > -margin) * (ty < H+margin))
    print('Cut to', len(tycho), 'Tycho-2 stars within brick')
    del ok,tx,ty
    tycho.writeto(os.path.join(args.outdir, 'tycho2.fits.gz'))
    
    outccds = C.copy()
    for c in ['ccd_x0', 'ccd_x1', 'ccd_y0', 'ccd_y1',
              'brick_x0', 'brick_x1', 'brick_y0', 'brick_y1',
              'plver', 'skyver', 'wcsver', 'psfver', 'skyplver', 'wcsplver',
              'psfplver' ]:
        outccds.delete_column(c)
    outccds.image_hdu[:] = 1

    # Convert to list to avoid truncating filenames
    outccds.image_filename = [fn for fn in outccds.image_filename]
    
    for iccd,ccd in enumerate(C):

        decam = (ccd.camera.strip() == 'decam')
        bok   = (ccd.camera.strip() == '90prime')

        im = survey.get_image_object(ccd)
        print('Got', im)
        slc = (slice(ccd.ccd_y0, ccd.ccd_y1), slice(ccd.ccd_x0, ccd.ccd_x1))
        tim = im.get_tractor_image(slc, pixPsf=True, splinesky=True,
                                   subsky=False, nanomaggies=False)
        print('Tim:', tim.shape)

        psf = tim.getPsf()
        print('PSF:', psf)
        psfex = psf.psfex
        print('PsfEx:', psfex)

        outim = outsurvey.get_image_object(ccd)
        print('Output image:', outim)

        print('Image filename:', outim.imgfn)
        trymakedirs(outim.imgfn, dir=True)

        imgdata = tim.getImage()
        dqdata = tim.dq
        if decam:
            # DECam-specific code remaps the DQ codes... re-read raw
            print('Reading data quality from', im.dqfn, 'hdu', im.hdu)
            dqdata = im._read_fits(im.dqfn, im.hdu, slice=tim.slice)
        ivdata = tim.getInvvar()

        if args.pad:
            # Create zero image of full size, copy in data.
            fullsize = np.zeros((ccd.height, ccd.width), imgdata.dtype)
            fullsize[slc] = imgdata
            imgdata = fullsize

            fullsize = np.zeros((ccd.height, ccd.width), dqdata.dtype)
            fullsize[slc] = dqdata
            dqdata = fullsize

            fullsize = np.zeros((ccd.height, ccd.width), ivdata.dtype)
            fullsize[slc] = ivdata
            ivdata = fullsize
            
        else:
            # Adjust the header WCS by x0,y0
            crpix1 = tim.hdr['CRPIX1']
            crpix2 = tim.hdr['CRPIX2']
            tim.hdr['CRPIX1'] = crpix1 - ccd.ccd_x0
            tim.hdr['CRPIX2'] = crpix2 - ccd.ccd_y0

        # Add image extension to filename
        # fitsio doesn't compress .fz by default, so drop .fz suffix
        
        outim.imgfn = outim.imgfn.replace('.fits', '-%s.fits' % im.ccdname)
        if not args.fpack:
            outim.imgfn = outim.imgfn.replace('.fits.fz', '.fits')

        # if bok:
        #     outim.whtfn  = outim.whtfn .replace('.wht.fits', '-%s.wht.fits' % im.ccdname)
        #     if not args.fpack:
        #         outim.whtfn  = outim.whtfn .replace('.fits.fz', '.fits')
        # else:
        if True:
            outim.wtfn  = outim.wtfn .replace('.fits', '-%s.fits' % im.ccdname)
            if not args.fpack:
                outim.wtfn  = outim.wtfn .replace('.fits.fz', '.fits')

        if outim.dqfn is not None:
            outim.dqfn  = outim.dqfn .replace('.fits', '-%s.fits' % im.ccdname)
            if not args.fpack:
                outim.dqfn  = outim.dqfn .replace('.fits.fz', '.fits')

        if bok:
            outim.psffn = outim.psffn.replace('.psf', '-%s.psf' % im.ccdname)

        ccdfn = outim.imgfn
        ccdfn = ccdfn.replace(outsurvey.get_image_dir(),'')
        if ccdfn.startswith('/'):
            ccdfn = ccdfn[1:]
        outccds.image_filename[iccd] = ccdfn

        print('Changed output filenames to:')
        print(outim.imgfn)
        print(outim.dqfn)

        ofn = outim.imgfn
        if args.fpack:
            f,ofn = tempfile.mkstemp(suffix='.fits')
            os.close(f)
        fitsio.write(ofn, None, header=tim.primhdr, clobber=True)
        fitsio.write(ofn, imgdata, header=tim.hdr, extname=ccd.ccdname)

        if args.fpack:
            cmd = 'fpack -qz 8 -S %s > %s && rm %s' % (ofn, outim.imgfn, ofn)
            print('Running:', cmd)
            rtn = os.system(cmd)
            assert(rtn == 0)

        h,w = tim.shape
        if not args.pad:
            outccds.width[iccd] = w
            outccds.height[iccd] = h
            outccds.crpix1[iccd] = crpix1 - ccd.ccd_x0
            outccds.crpix2[iccd] = crpix2 - ccd.ccd_y0

        wcs = Tan(*[float(x) for x in
                    [ccd.crval1, ccd.crval2, ccd.crpix1, ccd.crpix2,
                     ccd.cd1_1, ccd.cd1_2, ccd.cd2_1, ccd.cd2_2, ccd.width, ccd.height]])

        if args.pad:
            subwcs = wcs
        else:
            subwcs = wcs.get_subimage(ccd.ccd_x0, ccd.ccd_y0, w, h)
            outccds.ra[iccd],outccds.dec[iccd] = subwcs.radec_center()

        #if not bok:
        if True:
            print('Weight filename:', outim.wtfn)
            wfn = outim.wtfn
        # else:
        #     print('Weight filename:', outim.whtfn)
        #     wfn = outim.whtfn

        trymakedirs(wfn, dir=True)

        ofn = wfn
        if args.fpack:
            f,ofn = tempfile.mkstemp(suffix='.fits')
            os.close(f)

        fitsio.write(ofn, None, header=tim.primhdr, clobber=True)
        fitsio.write(ofn, ivdata, header=tim.hdr, extname=ccd.ccdname)

        if args.fpack:
            cmd = 'fpack -qz 8 -S %s > %s && rm %s' % (ofn, wfn, ofn)
            print('Running:', cmd)
            rtn = os.system(cmd)
            assert(rtn == 0)

        if outim.dqfn is not None:
            print('DQ filename', outim.dqfn)
            trymakedirs(outim.dqfn, dir=True)

            ofn = outim.dqfn
            if args.fpack:
                f,ofn = tempfile.mkstemp(suffix='.fits')
                os.close(f)

            fitsio.write(ofn, None, header=tim.primhdr, clobber=True)
            fitsio.write(ofn, dqdata, header=tim.hdr, extname=ccd.ccdname)

            if args.fpack:
                cmd = 'fpack -g -q 0 -S %s > %s && rm %s' % (ofn, outim.dqfn, ofn)
                print('Running:', cmd)
                rtn = os.system(cmd)
                assert(rtn == 0)

        print('PSF filename:', outim.psffn)
        trymakedirs(outim.psffn, dir=True)
        psfex.writeto(outim.psffn)

        if not bok:
            print('Sky filename:', outim.splineskyfn)
            sky = tim.getSky()
            print('Sky:', sky)
            trymakedirs(outim.splineskyfn, dir=True)
            sky.write_fits(outim.splineskyfn)

    outccds.writeto(os.path.join(args.outdir, 'survey-ccds-1.fits.gz'))

    # WISE
    if args.wise is not None:
        from wise.forcedphot import unwise_tiles_touching_wcs
        from wise.unwise import (unwise_tile_wcs, unwise_tiles_touching_wcs,
                                 get_unwise_tractor_image, get_unwise_tile_dir)
        # Read WCS...
        print('Reading TAN wcs header from', args.wise)
        targetwcs = Tan(args.wise)
        tiles = unwise_tiles_touching_wcs(targetwcs)
        print('Cut to', len(tiles), 'unWISE tiles')
        H,W = targetwcs.shape
        r,d = targetwcs.pixelxy2radec(np.array([1,   W,   W/2, W/2]),
                                      np.array([H/2, H/2, 1,   H  ]))
        roiradec = [r[0], r[1], d[2], d[3]]

        unwise_dir = os.environ['UNWISE_COADDS_DIR']
        wise_out = os.path.join(args.outdir, 'images', 'unwise')
        print('Will write WISE outputs to', wise_out)

        unwise_tr_dir = os.environ['UNWISE_COADDS_TIMERESOLVED_DIR']
        wise_tr_out = os.path.join(args.outdir, 'images', 'unwise-tr')
        print('Will write WISE time-resolved outputs to', wise_tr_out)

        W = fits_table(os.path.join(unwise_tr_dir, 'time_resolved_neo1-atlas.fits'))
        print('Read', len(W), 'time-resolved WISE coadd tiles')
        W.cut(np.array([t in tiles.coadd_id for t in W.coadd_id]))
        print('Cut to', len(W), 'time-resolved vs', len(tiles), 'full-depth')

        # Write the time-resolved index subset.
        W.writeto(os.path.join(wise_tr_out, 'time_resolved_neo1-atlas.fits'))

        # this ought to be enough for anyone =)
        Nepochs = 5

        wisedata = []

        # full depth
        for band in [1,2,3,4]:
            wisedata.append((unwise_dir, wise_out, tiles.coadd_id, band))

        # time-resolved
        for band in [1,2]:
            # W1 is bit 0 (value 0x1), W2 is bit 1 (value 0x2)
            bitmask = (1 << (band-1))
            for e in range(Nepochs):
                # Which tiles have images for this epoch?
                I = np.flatnonzero(W.epoch_bitmask[:,e] & bitmask)
                if len(I) == 0:
                    continue
                print('Epoch %i: %i tiles:' % (e, len(I)), W.coadd_id[I])
                edir = os.path.join(unwise_tr_dir, 'e%03i' % e)
                eoutdir = os.path.join(wise_tr_out, 'e%03i' % e)
                wisedata.append((edir, eoutdir, tiles.coadd_id[I], band))

        wrote_masks = set()

        for indir, outdir, tiles, band in wisedata:
            for tile in tiles:
                wanyband = 'w'
                tim = get_unwise_tractor_image(indir, tile, band,
                                               bandname=wanyband, roiradecbox=roiradec)
                print('Got unWISE tim', tim)
                print(tim.shape)
                
                thisdir = get_unwise_tile_dir(outdir, tile)
                print('Directory for this WISE tile:', thisdir)
                base = os.path.join(thisdir, 'unwise-%s-w%i-' % (tile, band))
                print('Base filename:', base)

                masked = True
                mu = 'm' if masked else 'u'

                imfn = base + 'img-%s.fits'       % mu
                ivfn = base + 'invvar-%s.fits.gz' % mu
                nifn = base + 'n-%s.fits.gz'      % mu
                nufn = base + 'n-u.fits.gz'

                #print('WISE image header:', tim.hdr)

                # Adjust the header WCS by x0,y0
                wcs = tim.wcs.wcs
                tim.hdr['CRPIX1'] = wcs.crpix[0]
                tim.hdr['CRPIX2'] = wcs.crpix[1]

                H,W = tim.shape
                tim.hdr['IMAGEW'] = W
                tim.hdr['IMAGEH'] = H

                print('WCS:', wcs)
                print('Header CRPIX', tim.hdr['CRPIX1'], tim.hdr['CRPIX2'])

                trymakedirs(imfn, dir=True)
                fitsio.write(imfn, tim.getImage(), header=tim.hdr, clobber=True)
                print('Wrote', imfn)
                fitsio.write(ivfn, tim.getInvvar(), header=tim.hdr, clobber=True)
                print('Wrote', ivfn)
                fitsio.write(nifn, tim.nims, header=tim.hdr, clobber=True)
                print('Wrote', nifn)
                fitsio.write(nufn, tim.nuims, header=tim.hdr, clobber=True)
                print('Wrote', nufn)

                if not (indir,tile) in wrote_masks:
                    print('Looking for mask file for', indir, tile)
                    # record that we tried this dir/tile combo
                    wrote_masks.add((indir,tile))
                    for idir in indir.split(':'):
                        tdir = get_unwise_tile_dir(idir, tile)
                        maskfn = 'unwise-%s-msk.fits.gz' % tile
                        fn = os.path.join(tdir, maskfn)
                        print('Mask file:', fn)
                        if os.path.exists(fn):
                            print('Reading', fn)
                            (x0,x1,y0,y1) = tim.roi
                            roislice = (slice(y0,y1), slice(x0,x1))
                            F = fitsio.FITS(fn)[0]
                            hdr = F.read_header()
                            M = F[roislice]
                            outfn = os.path.join(thisdir, maskfn)
                            fitsio.write(outfn, M, header=tim.hdr, clobber=True)
                            print('Wrote', outfn)
                            break

    outC = outsurvey.get_ccds_readonly()
    for iccd,ccd in enumerate(outC):
        outim = outsurvey.get_image_object(ccd)
        print('Got output image:', outim)
        otim = outim.get_tractor_image(pixPsf=True, splinesky=True)
        print('Got output tim:', otim)
    
if __name__ == '__main__':
    main()


