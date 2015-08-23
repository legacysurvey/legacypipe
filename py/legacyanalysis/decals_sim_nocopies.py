#!/usr/bin/env python

"""
Try to hack decals_sim so we don't have to make copies of the data.

"""

import os
import sys
import logging

import galsim
import numpy as np
import matplotlib.pyplot as plt

from tractor.psfex import PsfEx, PsfExModel
from tractor.basics import GaussianMixtureEllipsePSF
from astrometry.util.fits import fits_table, merge_tables

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

        image = galsim.Image(tim.getImage())
        invvar = galsim.Image(tim.getInvvar())
        twcs = self.get_wcs()

        # Get the Galsim-compatible WCS and the magnitude zeropoint for this CCD.
        wcs, origin = galsim.wcs.readFromFitsHeader(
            galsim.fits.FitsHeader(self.pvwcsfn))
        magzpt = self.t.ccdzpt+2.5*np.log10(self.t.exptime)

        for obj in self.decals.sim_sources:
        #for iobj in range(5):
#           obj = self.decals.sim_sources[iobj]

            if twcs.is_inside(obj.ra,obj.dec):
                objstamp = build_stamp(obj,psffile=self.psffn,wcs=wcs,
                                       band=self.t.filter,magzpt=magzpt)
                stamp = objstamp.star()
                print(stamp.bounds, image.bounds)
                plt.imshow(stamp) ; plt.show()

                overlap = stamp.bounds & image.bounds
                if (overlap.xmax>=0 and overlap.ymax>=0 and
                    overlap.xmin<=image.bounds.xmax and
                    overlap.ymin<=image.bounds.ymax and overlap.area()>0):

                    # Add Poisson noise
                    stamp = stamp[overlap]            # [ADU]
                    varstamp = invvar[overlap].copy() # [1/ADU^2]

                    stamp, varstamp = objstamp.addnoise(stamp,varstamp,siminfo)
                    image[overlap] += stamp
                    invvar[overlap] = varstamp

#           patch = src.getModelPatch(tim)
#           if patch is None:
#               continue
#           print 'Adding', src
#           patch.addTo(tim.getImage())
        return tim

class build_stamp():
    def __init__(self,obj,psffile=None,wcs=None,band=None,magzpt=None):
        """Build stamps of different types of objects."""
        #self.objtype = obj.objtype
        self.band = band.strip().upper()
        self.magzpt = magzpt
        self.pos = wcs.toImage(galsim.CelestialCoord(obj.ra*galsim.degrees,obj.dec*galsim.degrees))
        self.xpos = int(self.pos.x)
        self.ypos = int(self.pos.y)
        self.offset = galsim.PositionD(self.pos.x-self.xpos,self.pos.y-self.ypos)

        # Get the local WCS
        self.localwcs = wcs.local(image_pos=self.pos)
        pixscale, shear, theta, flip = self.localwcs.getDecomposition() # get the pixel scale
        self.pixscale = pixscale

        # Get the local PSF.  Need to update this to the latest PSFEx models!
        psfmodel = PsfExModel(psffile).at(self.xpos, self.ypos)[5:-5, 5:-5]
        #psf = GaussianMixtureEllipsePSF.fromStamp(psfmodel, N=2)

        psf = galsim.InterpolatedImage(galsim.Image(psfmodel),scale=pixscale,flux=1.0)
        psf_centroid = psf.centroid()
        self.localpsf = psf.shift(-psf_centroid.x,-psf_centroid.y)

        # Get the integrated object flux in the appropriate band in ADU.
        flux = obj[self.band+'FLUX']
        flux *= 10**(0.4*(self.magzpt-22.5)) # [ADU]
        self.objflux = flux

    def convolve_and_draw(self,obj):
        """Convolve the object with the PSF and then draw it."""
        obj = galsim.Convolve([obj,self.localpsf])
        stamp = obj.drawImage(offset=self.offset,wcs=self.localwcs,method='no_pixel')
        stamp.setCenter(self.xpos,self.ypos)
        return stamp

    def addnoise(self,stamp,varstamp,siminfo):
        varstamp.invertSelf()            # [ADU^2]
        medvar = np.median(varstamp.array[varstamp.array>0])
        varstamp.array[varstamp.array<(0.2*medvar)] = medvar

        # Convert to electrons
        stamp *= siminfo.gain         # [electron]
        varstamp *= (siminfo.gain**2) # [electron^2]

        stamp.addNoise(galsim.VariableGaussianNoise(galsim.BaseDeviate(),varstamp))
        varstamp += stamp
                
        stamp /= siminfo.gain          # [ADU]
        varstamp /= (siminfo.gain**2)  # [ADU^2]
        varstamp.invertSelf()          # [1/ADU^2]

        return stamp, varstamp

    def star(self):
        """Render a star (PSF)."""
        psf = self.localpsf.withFlux(self.objflux)
        stamp = psf.drawImage(offset=self.offset,wcs=self.localwcs,method='no_pixel')
        stamp.setCenter(self.xpos,self.ypos)
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

    zoom = [1800,2000,1800,2000]

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

    simdecals = SimDecals(sim_sources=simcat)

    sim = SimImage(simdecals,ccdinfo[0])
    tim = sim.get_tractor_image(const2psf=True)
    
    
#    # Our list of fake sources to insert into images...
#    decals.fake_sources = [PointSource(RaDecPos(0., 0.),
#                                       NanoMaggies(g=1000, r=1000, z=1000))]
#
#    run_brick(brickname, decals=simdecals, outdir=decals_sim_dir,
#              threads=8, zoom=zoom, wise=False, sdssInit=False,
#              forceAll=True, writePickles=False)
#
#
#
#    # Loop on each CCD and add each object
#    for ccd in ccdinfo:
#        # Gather some basic info on this CCD and then read the data, the WCS
#        # info, and initialize the PSF.
#        im = DecamImage(decals,ccd)
#        iminfo = im.get_image_info()
#
#        wcs = im.get_wcs()
#
#        # Identify the objects that are on this CCD.
#        onccd = []
#        for iobj, cat in enumerate(simcat):
#            if wcs.is_inside(cat.ra,cat.dec)==1:
#                onccd.append(iobj)
#        nobj = len(onccd)
#
#        # Need to pass the ra,dec positions of all the fake sources so that only
#        # those blobs are processed.
#
#    decals = FakeDecals()
#
#    # Our list of fake sources to insert into images...
#    decals.fake_sources = [PointSource(RaDecPos(0., 0.),
#                                       NanoMaggies(g=1000, r=1000, z=1000))]
#
#    run_brick(brickname, decals=decals, outdir=decals_sim_dir,
#              threads=8, zoom=zoom, wise=False, sdssInit=False,
#              forceAll=True, writePickles=False)

#    plt.scatter(simcat.ra,simcat.dec) ; plt.scatter(simcat.ra[onccd],simcat.dec[onccd],color='orange') ; plt.show()
#
#        if nobj>0:
#            log.info('Adding {} objects to HDU {}'.format(nobj,ccd.image_hdu))
#
#            for objinfo in simcat[onccd]:
#
#                # get the local coordinate, WCS, and PSF and then build the stamp
#                objstamp.getlocal(objinfo,siminfo)
#                if objtype=='STAR':
#                    stamp = objstamp.star()
#                if objtype=='ELG':
#                    stamp = objstamp.elg(objinfo,siminfo)
#
#                overlap = stamp.bounds & image.bounds
#                if (overlap.xmax>=0 and overlap.ymax>=0 and
#                    overlap.xmin<=image.bounds.xmax and
#                    overlap.ymin<=image.bounds.ymax and overlap.area()>0):
#
#                    # Add Poisson noise
#                    stamp = stamp[overlap]            # [ADU]
#                    varstamp = invvar[overlap].copy() # [1/ADU^2]
#
#                    stamp, varstamp = objstamp.addnoise(stamp,varstamp,siminfo)
#                    image[overlap] += stamp
#                    invvar[overlap] = varstamp
#
#            log.info('Writing {}[{}]'.format(siminfo.imfile_root,siminfo.hdu))
#            #fits.update(siminfo.imfile,image.array,ext=siminfo.hdu,
#            #            header=fits.Header(imhdr.items()))
#            #fits.update(siminfo.ivarfile,invvar.array,ext=siminfo.hdu,
#            #            header=fits.Header(ivarhdr.items()))
    
    


#    # Now, we can either call run_brick()....
#    run_brick(None, decals=decals, radec=(0.,0.), width=100, height=100,
#              stages=['image_coadds'])
    
    
#   # ... or the individual stages in the pipeline:
#   if False:
#       mp = multiproc()
#       kwa = dict(W = 100, H=100, ra=0., dec=0., mp=mp, brickname='fakey')
#       kwa.update(decals=decals)
#       R = stage_tims(**kwa)
#       kwa.update(R)
#       R = stage_image_coadds(**kwa)
#       # ...

if __name__ == '__main__':
    main()
