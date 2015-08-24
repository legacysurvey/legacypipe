#!/usr/bin/env python

"""
Try to hack decals_sim so we don't have to make copies of the data.

from legacypipe.common import Decals, DecamImage, wcs_for_brick, ccds_touching_wcs

brickname = '2428p117'
decals = Decals()
brickinfo = decals.get_brick_by_name(brickname)
brickwcs = wcs_for_brick(brickinfo)
ccdinfo = decals.ccds_touching_wcs(brickwcs)
im = DecamImage(decals,ccdinfo[10])
tim = im.get_tractor_image(const2psf=True)
"""

import os
import sys
import logging

import galsim
import numpy as np
import matplotlib.pyplot as plt

from tractor.engine import Patch
from tractor.psfex import PsfEx, PsfExModel
from tractor.basics import GaussianMixtureEllipsePSF, RaDecPos
from astrometry.util.fits import fits_table, merge_tables

from legacypipe.runbrick import run_brick
from legacypipe.common import Decals, DecamImage, wcs_for_brick, ccds_touching_wcs

logging.basicConfig(format='%(message)s',level=logging.INFO,stream=sys.stdout)
log = logging.getLogger('decals_sim')

class SimDecals(Decals):
    def __init__(self, decals_dir=None, sim_sources=None):
        super(SimDecals, self).__init__(decals_dir=decals_dir)
        self.sim_sources = sim_sources

    def get_image_object(self, t):
        return SimImage(self, t)

class SimImage(DecamImage):
    def __init__(self, decals, t):
        super(SimImage, self).__init__(decals, t)
        self.t = t

    def get_tractor_image(self, **kwargs):

        tim = super(SimImage, self).get_tractor_image(**kwargs)
        #help(tim)

        image = galsim.Image(tim.getImage())
        invvar = galsim.Image(tim.getInvvar())

        psf = tim.getPsf()
        wcs = tim.wcs
        twcs = self.get_wcs()

        # Get the Galsim-compatible WCS and the magnitude zeropoint for this CCD.
        #wcs, origin = galsim.wcs.readFromFitsHeader(
        #    galsim.fits.FitsHeader(self.pvwcsfn))
        magzpt = self.t.ccdzpt+2.5*np.log10(self.t.exptime)
        gain = self.t.arawgain

        # QAplot
        #from PIL import Image, ImageDraw
        #rad = 15
        #im = Image.fromarray(image.array).convert('RGB')
        #sz = im.size
        #draw = ImageDraw.Draw(im)

        for obj in self.decals.sim_sources:
        #for iobj in range(5):
            #obj = self.decals.sim_sources[iobj]

            if twcs.is_inside(obj.ra,obj.dec):
                objstamp = build_stamp(obj,psf=psf,wcs=wcs,
                                       band=self.t.filter,magzpt=magzpt)
                stamp = objstamp.star()
                #print(stamp.bounds, image.bounds)
                #plt.imshow(stamp.array) ; plt.show()

                overlap = stamp.bounds & image.bounds
                if (overlap.xmax>=0 and overlap.ymax>=0 and
                    overlap.xmin<=image.bounds.xmax and
                    overlap.ymin<=image.bounds.ymax and overlap.area()>0):

                    # Add Poisson noise
                    stamp = stamp[overlap]            # [ADU]
                    ivarstamp = invvar[overlap].copy() # [1/ADU^2]
                    stamp, ivarstamp = objstamp.addnoise(stamp,ivarstamp,gain)

                    image[overlap] += stamp
                    invvar[overlap] = ivarstamp

                    #impatch = Patch(overlap.xmin,overlap.ymin,stamp)
                    #impatch.addTo(image.array)
                    #varpatch = Patch(overlap.xmin,overlap.ymin,varstamp)
                    #varpatch.addTo(invvar.array)

                    # QAplot
                    #plt.imshow(image.array, **tim.ima) ; plt.show()
                    #draw.ellipse((obj.x-rad, sz[1]-obj.y-rad,obj.x+rad,
                    #              sz[1]-obj.y+rad))

                    #break

            tim.data = image.array
            tim.inverr = np.sqrt(invvar.array)

            #outfile = 'test-{}-{}.fits'.format(self.expnum,self.ccdname)
            #print('Writing {}'.format(outfile))
            #image.write(outfile,clobber=True)

            #tim.Image(data=image,invvar=invvar)
            #im.save('test.jpg')

#           patch = src.getModelPatch(tim)
#           if patch is None:
#               continue
#           print 'Adding', src
#           patch.addTo(tim.getImage())
        return tim

class build_stamp():
    def __init__(self,obj,psf=None,wcs=None,band=None,magzpt=None):
        """Build stamps of different types of objects."""
        #self.objtype = obj.objtype

        self.band = band.strip().lower()
        self.magzpt = magzpt

        xx, yy = wcs.positionToPixel(RaDecPos(obj.ra,obj.dec))
        self.pos = galsim.PositionD(xx,yy)
        #self.pos = wcs.toImage(galsim.CelestialCoord(obj.ra*galsim.degrees,
        #                                             obj.dec*galsim.degrees))
        self.xpos = int(self.pos.x)
        self.ypos = int(self.pos.y)
        self.offset = galsim.PositionD(self.pos.x-self.xpos,self.pos.y-self.ypos)

        self.gsparams = galsim.GSParams(maximum_fft_size=2L**30L)

        # Get the local WCS
        #self.localwcs = wcs.local(image_pos=self.pos)
        #pixscale, shear, theta, flip = self.localwcs.getDecomposition() # get the pixel scale
        #self.pixscale = pixscale
        self.pixscale = wcs.pixel_scale()
        #print('Pixel scale!', self.pixscale)

        # Get the local PSF.  Need to update this to the latest PSFEx models!
        #psfmodel = PsfExModel(psffile).at(self.xpos, self.ypos)[5:-5, 5:-5]
        psfim = psf.constantPsfAt(self.xpos,self.ypos)
        psfim = psfim.getPointSourcePatch(0,0).getImage()
        #self.localpsf = galsim.InterpolatedImage(galsim.Image(psfim),scale=pixscale,flux=1.0)
        self.localpsf = galsim.InterpolatedImage(galsim.Image(psfim),flux=1.0,
                                                 scale=self.pixscale)
        #self.localpsf = galsim.InterpolatedImage(galsim.Image(psfim),flux=1.0,
        #                                         wcs=self.localwcs)

        #psfex = PsfExModel(psffile)
        #psfim = psfex.at(self.xpos, self.ypos)
        #psfim = psfim[5:-5, 5:-5]
        #psfim = GaussianMixtureEllipsePSF.fromStamp(psfim, N=2)
        #psfim = psfim.getPointSourcePatch(51,51).getPatch()
        #psf = galsim.InterpolatedImage(galsim.Image(psfim),scale=pixscale,flux=1.0)

        #psf = galsim.InterpolatedImage(galsim.Image(psfmodel),scale=pixscale,flux=1.0)
        #print('Hack!!!!!  Setting negative pixels to zero!')
        #psf.array[psf.array<0] = 0.0
        #psf_centroid = psf.centroid()
        #self.localpsf = psf.shift(-psf_centroid.x,-psf_centroid.y)

        # Get the integrated object flux in the appropriate band in ADU.
        flux = obj.to_dict()[self.band+'flux']
        flux *= 10**(-0.4*(self.magzpt-22.5)) # [ADU]
        self.objflux = flux

    def convolve_and_draw(self,obj):
        """Convolve the object with the PSF and then draw it."""
        obj = galsim.Convolve([obj,self.localpsf])
        stamp = obj.drawImage(offset=self.offset,wcs=self.localwcs,method='no_pixel')
        stamp.setCenter(self.xpos,self.ypos)
        return stamp

    def addnoise(self,stamp,ivarstamp,gain=4.0):
        # Invert the inverse variance stamp, cleaning up zeros.
        varstamp = ivarstamp.copy()
        varstamp.invertSelf() # [ADU^2]

        medvar = np.median(varstamp.array[varstamp.array>0])
        #print('Median variance ', medvar)
        varstamp.array[varstamp.array<=0] = medvar

        # Convert to electrons.
        stamp *= gain         # [electron]
        varstamp *= (gain**2) # [electron^2]

        # Add Poisson noise
        #plt.imshow(varstamp.array) ; plt.show()
        #print('Turning off adding noise!')
        #stamp.addNoise(galsim.VariableGaussianNoise(galsim.BaseDeviate(),varstamp))
        #stamp.addNoise(galsim.VariableGaussianNoise(galsim.BaseDeviate(),stamp))
        #varstamp += stamp
        #plt.imshow(varstamp.array) ; plt.show()

        # Convert back to ADU
        stamp /= gain          # [ADU]
        varstamp /= (gain**2)  # [ADU^2]

        ivarstamp = varstamp.copy()
        ivarstamp.invertSelf()  # [1/ADU^2]

        return stamp, ivarstamp

    def star(self):
        """Render a star (PSF)."""
        psf = self.localpsf.withFlux(self.objflux)
        stamp = psf.drawImage(offset=self.offset,scale=self.pixscale,method='no_pixel')
        #stamp = psf.drawImage(offset=self.offset,wcs=self.localwcs,method='no_pixel')
        stamp.setCenter(self.xpos,self.ypos)
        #print(psf.getFlux(), stamp.center())
        return stamp

    def elg(self,objinfo,siminfo):
        """Create an ELG (disk-like) galaxy."""
        obj = galsim.Sersic(float(objinfo['SERSICN_1']),half_light_radius=
                            float(objinfo['R50_1']),
                            flux=self.objflux,gsparams=siminfo.gsparams)
        obj = obj.shear(q=float(objinfo['BA_1']),beta=
                        float(objinfo['PHI_1'])*galsim.degrees)
        stamp = self.convolve_and_draw(obj)
        return stamp

    def lrg(self,objinfo,siminfo):
        """Create an LRG (spheroidal) galaxy."""
        obj = galsim.Sersic(float(objinfo['SERSICN_1']),half_light_radius=
                            float(objinfo['R50_1']),
                            flux=self.objflux,gsparams=siminfo.gsparams)
        obj = obj.shear(q=float(objinfo['BA_1']),beta=
                        float(objinfo['PHI_1'])*galsim.degrees)
        stamp = self.convolve_and_draw(obj)
        return stamp

def main():

    brickname = '2428p117'
    objtype = 'STAR'
    lobjtype = objtype.lower()

    decals_sim_dir = '/Users/ioannis/decals_sim_dir/star/2428p117'

    zoom = [1800,2400,1800,2400]
    #zoom = [1800,2000,1800,2000]
    #zoom = None

    decals = Decals()
    brickinfo = decals.get_brick_by_name(brickname)
    brickwcs = wcs_for_brick(brickinfo)
    W, H, pixscale = brickwcs.get_width(), brickwcs.get_height(), brickwcs.pixel_scale()

    if zoom is not None:
        # See also runbrick.stage_tims()
        #(x0,x1,y0,y1) = args.zoom
        (x0,x1,y0,y1) = zoom
        W = x1-x0
        H = y1-y0
        targetwcs = brickwcs.get_subimage(x0, y0, W, H)

    bounds = brickwcs.radec_bounds()
    ra_range = bounds[1]-bounds[0]
    dec_range = bounds[3]-bounds[2]
    radec_center = brickwcs.radec_center()
    print(radec_center, ra_range, dec_range)
        
    corners = np.array([brickwcs.pixelxy2radec(x,y) for x,y in
                        [(1,1),(W,1),(W,H),(1,H),(1,1)]])
    

    # Identify the CCDs in the region of interest.
    ccdinfo = decals.ccds_touching_wcs(brickwcs)
    ccdinfo.about()
    log.info('Got {} CCDs'.format(len(ccdinfo)))

    # Generate the catalog of simulated sources here!
    simcat = fits_table('simcat-{}-{}-00.fits'.format(brickname,lobjtype))
    #simcat = fits.getdata('simcat-{}-{}-00.fits'.format(brickname,lobjtype))
    simcat.about()
    blobxy = zip(simcat.x,simcat.y)

    simdecals = SimDecals(sim_sources=simcat)

    # For testing!
    #sim = SimImage(simdecals,ccdinfo[0])
    #tim = sim.get_tractor_image(const2psf=True)

    run_brick(brickname, decals=simdecals, outdir=decals_sim_dir,
              threads=8, zoom=zoom, wise=False, sdssInit=False,
              forceAll=True, writePickles=False, blobxy=blobxy,
              pixPsf=False, stages=['tims','srcs','fitblobs',
                                    'fitblobs_finish','coadds',
                                    'writecat'])

if __name__ == '__main__':
    main()
