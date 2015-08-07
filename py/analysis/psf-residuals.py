#!/usr/bin/env python

"""Check the PSFs

Get PS1 stars for this brick:
bb = mrdfits('~/dr1/decals-bricks.fits',1)
bb = bb[where(bb.brickname eq '2402p062')]
print, bb.ra1, bb.ra2, bb.dec1, bb.dec2
       240.08380       240.33520       6.1250000       6.3750000
on edison:
pp = read_ps1cat([240.08380D,240.33520D],[6.1250000D,6.3750000D])
mwrfits, pp, 'ps1cat-all-2402p062.fits', /create

jj = mrdfits('ps1cat-all-2402p062.fits',1)
tt = mrdfits('vanilla/tractor/240/tractor-2402p062.fits',1)
spherematch, jj.ra, jj.dec, tt.ra, tt.dec, 1D/3600, m1, m2
mwrfits, jj[m1], 'ps1cat-2402p062.fits', /create

djs_plot, tt.ra, tt.dec, psym=3, ysty=3
djs_oplot, jj.ra, jj.dec, psym=6, color='orange'
djs_oplot, jj[m1].ra, jj[m1].dec, psym=6, color='blue'

"""

from __future__ import division, print_function

import os
import sys
import logging
import argparse
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from astropy.io import fits

from tractor.psfex import PsfEx
from astrometry.util.fits import fits_table
from legacypipe.common import Decals, DecamImage, LegacySurveyImage

flogging.basicConfig(format='%(message)s',level=logging.INFO,stream=sys.stdout)
log = logging.getLogger('check-psfs')

def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--expnum', type=str, default='396086', metavar='', 
                        help='exposure number')
    parser.add_argument('-c', '--ccdname', type=str, default='S31', metavar='', 
                        help='CCD name')

    args = parser.parse_args()
    if args.expnum is None or args.ccdname is None:
        parser.print_help()
        sys.exit(1)

    #ccd = decals.find_ccds(expnum=args.expnum,ccdname=args.ccdname)

    import numpy as np
    import matplotlib.pyplot as plt
    from tractor.psfex import PsfEx
    from tractor.basics import GaussianMixtureEllipsePSF, PixelizedPSF, RaDecPos
    from astrometry.util.fits import fits_table
    from legacypipe.common import Decals, DecamImage
    
    decals = Decals()
    ccd = decals.find_ccds(expnum=396086,ccdname='S31')[0]
    ccd.about()

    band = ccd.filter
    im = DecamImage(decals,ccd)
    print('Band: ', band)
    print('Reading: ', im.imgfn)

    iminfo = im.get_image_info()
    print('img: ', iminfo)
    H,W = iminfo['dims']

    xpos, ypos = (1501.3, 1800.2)
    
    psfex = PsfEx(im.psffn, W, H, ny=13, nx=7, K=2,
                  psfClass=GaussianMixtureEllipsePSF)
    psfim = PsfEx.instantiateAt(psfmog,xpos,ypos)

    mog = psfim[5:-5,5:-5] # trim
    #mog1 = GaussianMixtureEllipsePSF.fromStamp(mog,N=2)

    # pixelized PSF
    psfpix = PixelizedPSF(psfim)

    im = decals.get_image_object(ccd)
    
    # use the 2-component Gaussian mixture fit to the PsfEx model, as in DR1
    tim = im.get_tractor_image(const2psf=True)
    gpsf = tim.getPsf()
    psfex = tim.psfex

    # Instantiate the PsfEx model at image center
    imh,imw = im.get_image_shape()
    psfexim = psfex.instantiateAt(imw/2,imh/2)
    print 'PsfEx patch:', psfexim.shape

    ## Instantiate the Gaussian mixture PSF model.  'gpatch' is a Patch object,
    ## gpatch.patch is a numpy image
    #gpatch = gpsf.getPointSourcePatch(0.0,0.0)
    #gpatch = gpatch.patch
    #print 'Gaussian patch:', gpatch.shape
    
    # If you want to insert the Gaussian model into an image the same shape as
    # the PsfEx stamp, you can do that too:
    stamph, stampw = psfexim.shape
    gpatch = gpsf.getPointSourcePatch(stampw/2, stamph/2)
    gim = np.zeros((stamph, stampw))
    gpatch.addTo(gim)

    ps1 = fits_table('tmp/ps1cat-2402p062.fits')
    ra, dec = ps1.ra[0], ps1.dec[0]
    xx, yy = tim.getWcs().positionToPixel(RaDecPos(ra,dec)))




    plt.imshow(mog) ; plt.show()


    slc = slice(1000,1100), slice(1500,1600)
    tim = legacy.get_tractor_image(slc=slc,gaussPsf=True)

    

    
    brickwcs = wcs_for_brick(decals.get_brick_by_name(brickname))
    ccdinfo = decals.ccds_touching_wcs(brickwcs)


    # Read the WCS


    xxyy = brickwcs.radec2pixelxy(ps1.ra,ps1.dec)    
    

    # Get cutouts of the missing sources
    chunksuffix = '00'
    imfile = os.path.join(decals_sim_dir,'qa-'+brickname+'-'+lobjtype+
                          '-image-'+chunksuffix+'.jpg')
    hw = 30 # half-width [pixels]
    ncols = 5
    nrows = 5
    nthumb = ncols*nrows
    dims = (ncols*hw*2,nrows*hw*2)
    mosaic = Image.new('RGB',dims)

    miss = missing[np.argsort(simcat['r'][missing])]
    print(simcat['r'][miss])
    
    xpos, ypos = np.meshgrid(np.arange(0,dims[0],hw*2,dtype='int'),
                             np.arange(0,dims[1],hw*2,dtype='int'))
    im = Image.open(imfile)
    sz = im.size
    iobj = 0
    for ic in range(ncols):
        for ir in range(nrows):
            mm = miss[iobj]
            xx = int(simcat['X'][mm])
            yy = int(sz[1]-simcat['Y'][mm])
            crop = (xx-hw,yy-hw,xx+hw,yy+hw)
            box = (xpos[ir,ic],ypos[ir,ic])
            thumb = im.crop(crop)
            mosaic.paste(thumb,box)
            iobj = iobj+1

    # Add a border
    draw = ImageDraw.Draw(mosaic)
    for ic in range(ncols):
        for ir in range(nrows):
            draw.rectangle([(xpos[ir,ic],ypos[ir,ic]),
                            (xpos[ir,ic]+hw*2,ypos[ir,ic]+hw*2)])
    qafile = os.path.join(decals_sim_dir,'qa-'+brickname+'-'+lobjtype+'-missing.png')
    log.info('Writing {}'.format(qafile))
    mosaic.save(qafile)


    
if __name__ == "__main__":
    main()
