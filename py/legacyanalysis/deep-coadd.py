from __future__ import print_function
import numpy as np
import fitsio
from astrometry.util.fits import fits_table
from legacypipe.common import *
from legacypipe.coadds import _resample_one
from legacypipe.cpimage import CP_DQ_BITS
from legacypipe.runbrick import rgbkwargs

def main():
    brickname = '0362m045'
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

    ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)
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
    
    ims = []
    for ccd in ccds:
        im = survey.get_image_object(ccd)
        ims.append(im)
        print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid)

    coimgs = []
    
    for band in bands:
        print('Computing coadd for band', band)

        # coadded weight map (moo)
        cow    = np.zeros((H,W), np.float32)
        # coadded weighted image map
        cowimg = np.zeros((H,W), np.float32)
        # unweighted image
        coimg  = np.zeros((H,W), np.float32)
        # number of exposures
        con    = np.zeros((H,W), np.uint8)
        
        unweighted=True
        tinyw = 1e-30

        N = 0
        
        for iim,im in enumerate(ims):
            if im.band != band:
                continue
            tim = im.get_tractor_image(radecpoly=targetrd, splinesky=True, gaussPsf=True)
            if tim is None:
                continue
            print('Reading', tim.name)

            N += 1
            #if N == 10:
            #    break
            
            # surface-brightness correction
            tim.sbscale = (targetwcs.pixel_scale() / tim.subwcs.pixel_scale())**2

            R = _resample_one((0, tim, None, lanczos, targetwcs))
            if R is None:
                continue
            itim,Yo,Xo,iv,im,mo,dq = R

            # invvar-weighted image
            cowimg[Yo,Xo] += iv * im
            cow   [Yo,Xo] += iv
            
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

        # Per-band:
        cowimg /= np.maximum(cow, tinyw)
        coimg  /= np.maximum(con, 1)
        cowimg[cow == 0] = coimg[cow == 0]

        fn = 'coadd-%s-image-%s.fits' % (brickname, band)
        fitsio.write(fn, cowimg, clobber=True)
        print('Wrote', fn)
        fn = 'coadd-%s-invvar-%s.fits' % (brickname, band)
        fitsio.write(fn, cow, clobber=True)
        print('Wrote', fn)
        fn = 'coadd-%s-n-%s.fits' % (brickname, band)
        fitsio.write(fn, con, clobber=True)
        print('Wrote', fn)

        coimgs.append(cowimg)
    
    rgb = get_rgb(coimgs, bands, **rgbkwargs)
    kwa = {}
    imsave_jpeg('coadd-%s-image.jpg' % brickname, rgb, origin='lower', **kwa)

        
            
            
if __name__ == '__main__':
    main()
    
