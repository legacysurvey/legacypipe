from __future__ import print_function
import numpy as np
import fitsio
from astrometry.util.fits import fits_table
from legacypipe.common import *
from legacypipe.coadds import _resample_one
from legacypipe.cpimage import CP_DQ_BITS
from legacypipe.runbrick import rgbkwargs

def main():
    #brickname = '0362m045'
    #brickname = '0359m047'
    brickname = '0359m045'
    W = H = 3600
    pixscale = 0.262
    bands = 'grz'
    lanczos = True
    
    survey = LegacySurveyData()
    brick = survey.get_brick_by_name(brickname)
    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    pixscale = targetwcs.pixel_scale()
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])

    # where to measure the depth
    probe_ra = brick.ra
    probe_dec = brick.dec
    
    #ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)

    ccds = survey.get_annotated_ccds()
    I = ccds_touching_wcs(targetwcs, ccds)
    ccds.cut(I)

    print(len(ccds), 'CCDs touching target WCS')

    #I = survey.apply_blacklist(ccds)
    #ccds.cut(I)
    #print(len(ccds), 'CCDs not in blacklisted propids (too many exposures!)')

    # Sort images by band -- this also eliminates images whose
    # *image.filter* string is not in *bands*.
    print('Unique filters:', np.unique(ccds.filter))
    ccds.cut(np.hstack([np.flatnonzero(ccds.filter == band) for band in bands]))
    print('Cut on filter:', len(ccds), 'CCDs remain.')

    print('Cutting out non-photometric CCDs...')
    I = survey.photometric_ccds(ccds)
    print(len(I), 'of', len(ccds), 'CCDs are photometric')
    ccds.cut(I)
    
    fn = 'coadd-%s-ccds.fits' % brickname
    ccds.writeto(fn)
    print('Wrote', fn)

    psfdepths = dict([(b,0.) for b in bands])
    ims = []
    for ccd in ccds:
        im = survey.get_image_object(ccd)
        ims.append(im)
        print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid)

        wcs = survey.get_approx_wcs(ccd)
        if wcs.is_inside(probe_ra, probe_dec):
            # Point-source detection
            detsig1 = ccd.sig1 / ccd.psfnorm_mean
            psfdepths[im.band] += (1. / detsig1**2)

    for band in bands:
        sig1 = np.sqrt(1. / psfdepths[band])
        depth = 5. * sig1
        mag = -2.5 * (np.log10(depth) - 9)
        print('PSF 5-sigma depth:', mag)

    coimgs = []
    coimgs2 = []
    
    for band in bands:
        print('Computing coadd for band', band)

        hdr = fitsio.FITSHDR()
        hdr.add_record(dict(name='FILTER', value=band))
        hdr.add_record(dict(name='BRICK', value=brickname))
        # Plug the WCS header cards into these images
        targetwcs.add_to_header(hdr)
        hdr.delete('IMAGEW')
        hdr.delete('IMAGEH')
        hdr.add_record(dict(name='EQUINOX', value=2000.))
        
        # coadded weight map (moo)
        cow    = np.zeros((H,W), np.float32)
        # coadded weighted image map
        cowimg = np.zeros((H,W), np.float32)
        # unweighted image
        coimg  = np.zeros((H,W), np.float32)
        # number of exposures
        con    = np.zeros((H,W), np.uint8)

        # coadded weight map (moo)
        cow2    = np.zeros((H,W), np.float32)
        # coadded weighted image map
        cowimg2 = np.zeros((H,W), np.float32)
        # unweighted image
        coimg2  = np.zeros((H,W), np.float32)
        # number of exposures
        con2    = np.zeros((H,W), np.uint8)
        
        tinyw = 1e-30

        I = np.flatnonzero(ccds.filter == band)
        medsee = np.median(ccds.seeing[I])

        for ccd in ccds[I]:
            im = survey.get_image_object(ccd)
            tim = im.get_tractor_image(radecpoly=targetrd, splinesky=True, gaussPsf=True)
            if tim is None:
                continue
            print('Reading', tim.name)

            # surface-brightness correction
            tim.sbscale = (targetwcs.pixel_scale() / tim.subwcs.pixel_scale())**2

            R = _resample_one((0, tim, None, lanczos, targetwcs))
            if R is None:
                continue
            itim,Yo,Xo,iv,im,mo,dq = R

            goodsee = (ccd.seeing < medsee)
            
            # invvar-weighted image
            cowimg[Yo,Xo] += iv * im
            cow   [Yo,Xo] += iv

            if goodsee:
                cowimg2[Yo,Xo] += iv * im
                cow2   [Yo,Xo] += iv

            if dq is None:
                goodpix = 1
            else:
                # include BLEED, SATUR, INTERP pixels if no other
                # pixels exists (do this by eliminating all other CP
                # flags)
                badbits = 0
                for bitname in ['badpix', 'cr', 'trans', 'edge', 'edge2']:
                    badbits |= CP_DQ_BITS[bitname]
                goodpix = ((dq & badbits) == 0)
                
            coimg[Yo,Xo] += goodpix * im
            con  [Yo,Xo] += goodpix
            if goodsee:
                coimg2[Yo,Xo] += goodpix * im
                con2  [Yo,Xo] += goodpix
            
        # Per-band:
        cowimg /= np.maximum(cow, tinyw)
        coimg  /= np.maximum(con, 1)
        cowimg[cow == 0] = coimg[cow == 0]
        
        fn = 'coadd-%s-image-%s.fits' % (brickname, band)
        fitsio.write(fn, cowimg, clobber=True, header=hdr)
        print('Wrote', fn)
        fn = 'coadd-%s-invvar-%s.fits' % (brickname, band)
        fitsio.write(fn, cow, clobber=True, header=hdr)
        print('Wrote', fn)
        fn = 'coadd-%s-n-%s.fits' % (brickname, band)
        fitsio.write(fn, con, clobber=True, header=hdr)
        print('Wrote', fn)

        coimgs.append(cowimg)

        cowimg2 /= np.maximum(cow2, tinyw)
        coimg2  /= np.maximum(con2, 1)
        cowimg2[cow2 == 0] = coimg2[cow2 == 0]
        
        fn = 'coadd-%s-image2-%s.fits' % (brickname, band)
        fitsio.write(fn, cowimg2, clobber=True, header=hdr)
        print('Wrote', fn)
        fn = 'coadd-%s-invvar2-%s.fits' % (brickname, band)
        fitsio.write(fn, cow2, clobber=True, header=hdr)
        print('Wrote', fn)
        fn = 'coadd-%s-n2-%s.fits' % (brickname, band)
        fitsio.write(fn, con2, clobber=True, header=hdr)
        print('Wrote', fn)

        coimgs2.append(cowimg2)

        
    rgb = get_rgb(coimgs, bands, **rgbkwargs)
    kwa = {}
    imsave_jpeg('coadd-%s-image.jpg' % brickname, rgb, origin='lower', **kwa)

    rgb = get_rgb(coimgs2, bands, **rgbkwargs)
    imsave_jpeg('coadd-%s-image2.jpg' % brickname, rgb, origin='lower', **kwa)
            
if __name__ == '__main__':
    main()
    
