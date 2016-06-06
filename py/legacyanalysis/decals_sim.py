#!/usr/bin/env python

"""Try to hack decals_sim so we don't have to make copies of the data.

3216p000

nohup python ${LEGACYPIPE}/py/legacyanalysis/decals_sim.py -n 32 -c 10 -b 2428p117 -o STAR --zoom 500 1400 600 1300 > log2 2>&1 &
nohup python ${LEGACYPIPE}/py/legacyanalysis/decals_sim.py -n 5000 -c 500 -b 3216p000 -o STAR > & 3216p000.log &

import numpy as np
from legacypipe.common import Decals, DecamImage, wcs_for_brick, ccds_touching_wcs

brickname = '2428p117'
decals = Decals()
brickinfo = decals.get_brick_by_name(brickname)
brickwcs = wcs_for_brick(brickinfo)
ccdinfo = decals.ccds_touching_wcs(brickwcs)
im = DecamImage(decals,ccdinfo[19])
targetrd = np.array([[ 242.98072831,   11.61900584],
                     [ 242.71332268,   11.61900584],
                     [ 242.71319548,   11.88093189],
                     [ 242.98085551,   11.88093189],
                     [ 242.98072831,   11.61900584]])
tim = im.get_tractor_image(const2psf=True, radecpoly=targetrd)

~1 target per square arcminute in DESI, so the random should have ~50 sources
per square arcminute: 

IDL> print, 50*3600.0*0.25^2
      11250.0 ; sources over a whole brick
"""
from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg')

import os
import sys
import shutil
import logging
import argparse

import fitsio
import galsim
import numpy as np
import matplotlib.pyplot as plt

# from pydl.pydlutils.spheregroup import spheregroup
from astropy.table import Table, Column, vstack

from tractor.psfex import PsfEx, PsfExModel
from tractor.basics import GaussianMixtureEllipsePSF, RaDecPos

from legacypipe.runbrick import run_brick
from legacypipe.decam import DecamImage
from legacypipe.common import LegacySurveyData, wcs_for_brick, ccds_touching_wcs

#from thesis_code.galsim import plots

class SimDecals(LegacySurveyData):
    def __init__(self, survey_dir=None, output_dir=None,
                 metacat=None, simcat=None):
        super(SimDecals, self).__init__(survey_dir=survey_dir,
                                        output_dir=output_dir)
        self.metacat = metacat
        self.simcat = simcat
        #add +fakes string to camera to signal SimDecas

    def get_image_object(self, t):
        return SimImage(self, t)

class SimImage(DecamImage):
    def __init__(self, survey, t):
        super(SimImage, self).__init__(survey, t)
        self.t = t

    def get_tractor_image(self, **kwargs):
        print('get_tractor_image kwargs= ',kwargs)
        print('dir(self)= ',dir(self))
        print('dir(self.t)= ',dir(self.t))
        print('dir(self.survey)= ',dir(self.survey))
        print('self.survey.output_dir= ',self.survey.output_dir)
        #print('being called in SimImage, quitting')
        #print('self.imgfn=',self.imgfn)
        #sys.exit()
        tim = super(SimImage, self).get_tractor_image(**kwargs)
        print('dir(tim)= ',dir(tim))
        if tim is None: # this can be None when the edge of a CCD overlaps
            return tim

        # Initialize the object stamp class
        objtype = self.survey.metacat['objtype']
        objstamp = BuildStamp(tim,gain=self.t.arawgain)

        # Grab the data and inverse variance images [nanomaggies!]
        image = galsim.Image(tim.getImage())
        invvar = galsim.Image(tim.getInvvar())
        sims_image= image.copy() #sum of sims image
        sims_image.fill(0.) 
        sims_ivar= sims_image.copy() 
        tim.sims_xylim= np.empty((len(self.survey.simcat),4))+np.nan #N sims, 4 box corners (xmin,xmax,ymin,ymax) for each sim
        #sys.exit(1)

        #store simulated galaxy images in tim object 
        # Loop on each object.
        for ii, obj in enumerate(self.survey.simcat):
            #print(obj)
            if objtype=='STAR':
                stamp = objstamp.star(obj)
            elif objtype=='ELG':
                stamp = objstamp.elg(obj)
            elif objtype=='LRG':
                stamp = objstamp.lrg(obj)

            # Make sure the object falls on the image and then add Poisson noise.
            overlap = stamp.bounds & image.bounds
            print('obj=%d, image.bounds=' % ii,image.bounds,'stamp.bounds=',stamp.bounds)
            if (overlap.area()>0):
                stamp = stamp[overlap] #image bounds the stamp region, not vice versa     
                ivarstamp = invvar[overlap]
                stamp, ivarstamp = objstamp.addnoise(stamp,ivarstamp)
 
                #image[overlap] += stamp
                #plots.one_image(image.array,name='tim.getImage_obj%d.png' % ii)
                sims_image[overlap] += stamp #just sim, they are ADDED over the whole image
                sims_ivar[overlap] += ivarstamp
                tim.sims_xylim[ii,:]= [overlap.xmin-1,overlap.xmax-1,overlap.ymin-1,overlap.ymax-1] #-1 b/c galsim 1st index is 1
                #print('obj= %d, img= %s' % (ii,os.path.basename(self.imgfn)))
                #print('image.bounds=',image.bounds,'stamp.bounds=',stamp.bounds,'overlap=',overlap)
                #plots.image_plus_stamp(image.array,[overlap.xmin,overlap.xmax],\
                #                                    [overlap.ymin,overlap.ymax],'yellow_box.png')
                #print('dir(time)',dir(tim))
                #print('dir(self)',dir(self))

                if np.min(invvar.array)<0:
                    print('Negative invvar!')
                    sys.exit(1)
        assert(sims_image.array.shape == image.array.shape)
        assert(sims_ivar.array.shape == invvar.array.shape)
        tim.sims_image= sims_image.array
        tim.sims_inverr= np.sqrt(sims_ivar.array)
        tim.sims_xylim= tim.sims_xylim.astype(int)
        tim.data = image.array + sims_image.array
        tim.inverr = np.sqrt(invvar.array + sims_ivar.array)
        #plot image,image regions where have sims, just sims as 3 plot panel with yellow boxes
        #basename= plots.get_basename(self.imgfn)
        #plots.image_v_stamp([tim.data,tim.data-tim.sims_image,tim.sims_image], \
        #                    xy_lim= tim.sims_xylim, name=os.path.join(self.survey.output_dir,"image_v_stamp_%s.png" % basename))
        #plots.image_v_stamp([np.power(tim.inverr,-1),np.power(tim.sims_inverr,-1)], \
        #                    xy_lim= tim.sims_xylim, titles=['image_std','sims_std'],\
                            #name=os.path.join(self.survey.output_dir,"std_%s.png" % basename))
        #print('exiting early')
        #sys.exit()
        return tim

class BuildStamp():
    def __init__(self,tim,gain=4.0):
        """Initialize the BuildStamp object with the CCD-level properties we need."""

        self.gsparams = galsim.GSParams(maximum_fft_size=2L**30L)
        self.band = tim.band.strip().lower()

        self.wcs = tim.getWcs()
        self.psf = tim.getPsf()

        # zpscale equivalent to magzpt = self.t.ccdzpt+2.5*np.log10(self.t.exptime)
        self.zpscale = tim.zpscale      # nanomaggies-->ADU conversion factor
        self.nano2e = self.zpscale*gain # nanomaggies-->electrons conversion factor

    def setlocal(self,obj):
        """Get the pixel positions, local pixel scale, and local PSF.""" 

        xx, yy = self.wcs.positionToPixel(RaDecPos(obj['ra'],obj['dec']))
        self.pos = galsim.PositionD(xx,yy)
        self.xpos = int(self.pos.x)
        self.ypos = int(self.pos.y)
        self.offset = galsim.PositionD(self.pos.x-self.xpos,self.pos.y-self.ypos)

        # Get the local pixel scale [arcsec/pixel]
        cd = self.wcs.cdAtPixel(self.pos.x, self.pos.y)
        self.pixscale = np.sqrt(np.linalg.det(cd))*3600.0

        # Get the local PSF
        psfim = self.psf.getPointSourcePatch(self.xpos, self.ypos).getImage()
        #plt.imshow(psfim) ; plt.show()
        
        self.localpsf = galsim.InterpolatedImage(galsim.Image(psfim),scale=self.pixscale)

    def addnoise(self,stamp,ivarstamp):
        """Add noise to the object postage stamp.  Remember that STAMP and IVARSTAMP
        are in units of nanomaggies and 1/nanomaggies**2, respectively.

        """
        varstamp = ivarstamp.copy()
        varstamp.invertSelf()
        if np.min(varstamp.array)<0:
            print(np.min(varstamp.array))
            #sys.exit(1)
            
        # Add the variance of the object to the variance image (in electrons).
        stamp *= self.nano2e       # [electron]
        #stamp.array = np.abs(stamp.array)
        st = np.abs(stamp.array)
        stamp = galsim.Image(st)
        varstamp *= self.nano2e**2 # [electron^2]
        firstvarstamp = varstamp + stamp

        # Add Poisson noise
        stamp.addNoise(galsim.VariableGaussianNoise(
            galsim.BaseDeviate(),firstvarstamp))

        # ensure the Poisson variance from the object is >0 (see Galsim.demo13)
        objvar = galsim.Image(np.sqrt(stamp.array**2),scale=stamp.scale) 
        objvar.setOrigin(galsim.PositionI(stamp.xmin, stamp.ymin))
        varstamp += objvar

        # Convert back to [nanomaggies]
        stamp /= self.nano2e      
        varstamp /= self.nano2e**2

        ivarstamp = varstamp.copy()
        ivarstamp.invertSelf()

        return stamp, ivarstamp

    def convolve_and_draw(self,obj):
        """Convolve the object with the PSF and then draw it."""
        obj = galsim.Convolve([obj,self.localpsf])
        stamp = obj.drawImage(offset=self.offset,wcs=self.localwcs,method='no_pixel')
        stamp.setCenter(self.xpos,self.ypos)
        return stamp

    def star(self,obj):
        """Render a star (PSF)."""

#       import pdb ; pdb.set_trace()
        self.setlocal(obj)

        flux = obj[self.band+'flux'] # [nanomaggies]
        psf = self.localpsf.withFlux(flux)

        stamp = psf.drawImage(offset=self.offset,scale=self.pixscale,method='no_pixel')
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

def build_simcat(nobj=None,brickname=None,brickwcs=None,meta=None,seed=None):
    """Build the simulated object catalog, which depends on OBJTYPE."""

    rand = np.random.RandomState(seed)

    # Assign central coordinates uniformly but remove simulated sources which
    # are too near to one another.  Iterate until we have the requisite number
    # of objects.
    bounds = brickwcs.radec_bounds()
    ra = rand.uniform(bounds[0],bounds[1],nobj)
    dec = rand.uniform(bounds[2],bounds[3],nobj)
    #ra = np.random.uniform(bounds[0],bounds[1],nobj)
    #dec = np.random.uniform(bounds[2],bounds[3],nobj)

    #ngood = 0
    #while ngood<nobj:
    #    print(nobj, ngood)
    #    #print(ra)
    #    gg = spheregroup(ra,dec,5.0/3600.0)
    #
    #    nperbin = np.bincount(gg[0],minlength=nobj)
    #    good = np.delete(np.arange(nobj),np.where(nperbin!=1))
    #    ngood = len(good)
    #
    #    #plt.scatter(ra,dec) ; plt.scatter(ra[good],dec[good],color='orange') ; plt.show()
    #
    #    ra = np.append(ra[good],np.random.uniform(bounds[0],bounds[1],nobj))
    #    dec = np.append(dec[good],np.random.uniform(bounds[2],bounds[3],nobj))
    #    print(len(ra))
    #    #np.append(ra[good],rand.uniform(bounds[0],bounds[1],nobj))
    #    #np.append(dec[good],rand.uniform(bounds[2],bounds[3],nobj))
    #
    #ra = ra[:nobj]
    #dec = dec[:nobj]
    #
    #test = spheregroup(ra,dec,10.0/3600.0)
    #print(np.bincount(test[0],minlength=nobj))
    #sys.exit(1)

    xxyy = brickwcs.radec2pixelxy(ra,dec)

    cat = Table()
    cat['id'] = Column(np.arange(nobj,dtype='i4'))
    cat['x'] = Column(xxyy[1][:],dtype='f4')
    cat['y'] = Column(xxyy[2][:],dtype='f4')
    cat['ra'] = Column(ra,dtype='f8')
    cat['dec'] = Column(dec,dtype='f8')

    if meta['objtype']=='ELG':
        gr_range = [-0.3,0.5]
        rz_range = [0.0,1.5]

        sersicn_1_range = [1.0,1.0]
        r50_1_range = [0.5,2.5]
        ba_1_range = [0.2,1.0]

        cat['sersicn_1'] = Column(sersicn_1,dtype='f4')
        cat['r50_1'] = Column(r50_1,dtype='f4')
        cat['ba_1'] = Column(ba_1,dtype='f4')
        cat['phi_1'] = Column(phi_1,dtype='f4')

        ## Bulge parameters
        #bulge_r50_range = [0.1,1.0]
        #bulge_n_range = [3.0,5.0]
        #bdratio_range = [0.0,1.0] # bulge-to-disk ratio

        cat = np.concatenate((cat,elgcat))

    if meta['objtype']=='STAR':
        gr_range = [-0.2,1.0]
        rz_range = [-0.2,1.5]

    # For convenience, also store the grz fluxes in nanomaggies.
    rmag_range = meta['rmag_range'][0]
    rmag = rand.uniform(rmag_range[0],rmag_range[1],nobj)
    gr = rand.uniform(gr_range[0],gr_range[1],nobj)
    rz = rand.uniform(rz_range[0],rz_range[1],nobj)
    
    cat['r'] = rmag
    cat['gr'] = gr
    cat['rz'] = rz
    cat['gflux'] = 1E9*10**(-0.4*(rmag+gr)) # [nanomaggies]
    cat['rflux'] = 1E9*10**(-0.4*rmag)      # [nanomaggies]
    cat['zflux'] = 1E9*10**(-0.4*(rmag-rz)) # [nanomaggies]

    return cat

def main():
    """Main routine which parses the optional inputs."""

    parser = argparse.ArgumentParser(formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter,
                                     description='DECaLS simulations.')
    parser.add_argument('-n', '--nobj', type=long, default=None, metavar='', 
                        help='number of objects to simulate (required input)')
    parser.add_argument('-c', '--chunksize', type=long, default=500, metavar='', 
                        help='divide NOBJ into CHUNKSIZE chunks')
    parser.add_argument('-b', '--brick', type=str, default='2428p117', metavar='', 
                        help='simulate objects in this brick')
    parser.add_argument('-o', '--objtype', type=str, choices=['STAR','ELG', 'LRG', 'QSO', 'LSB'],default='STAR', metavar='', 
                        help='insert these into images') 
    parser.add_argument('-t', '--threads', type=int, default=8, metavar='', 
                        help='number of threads to use when calling The Tractor')
    parser.add_argument('-s', '--seed', type=long, default=None, metavar='', 
                        help='random number seed')
    parser.add_argument('-z', '--zoom', nargs=4, type=int, metavar='', 
                        help='see runbrick.py; (default is 0 3600 0 3600)')
    parser.add_argument('-survey_dir', '--survey_dir', metavar='', 
                        help='location of survey-ccds*.fits.gz')
    parser.add_argument('--rmag-range', nargs=2, type=float, default=(18,26), metavar='', 
                        help='r-band magnitude range')
    parser.add_argument('--stage', choices=['tims','image_coadds','srcs','fitblobs','image_coadds'], default='writecat',metavar='', 
                        help='Run up to the given stage')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='toggle on verbose output')

    args = parser.parse_args()
    if args.nobj is None:
        parser.print_help()
        sys.exit(1)

    # Set the debugging level
    if args.verbose:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO
    logging.basicConfig(format='%(message)s',level=lvl,stream=sys.stdout)
    log = logging.getLogger('__name__')

    brickname = args.brick
    objtype = args.objtype.upper()
    lobjtype = objtype.lower()

    if objtype=='LRG':
        log.warning('{} objtype not yet supported!'.format(objtype))
        sys.exit(1)
    elif objtype=='LSB':
        log.warning('{} objtype not yet supported!'.format(objtype))
        sys.exit(1)
    elif objtype=='QSO':
        log.warning('{} objtype not yet supported!'.format(objtype))
        sys.exit(1)

    # Determine how many chunks we need
    nobj = args.nobj
    chunksize = args.chunksize
    nchunk = long(np.ceil(nobj/chunksize))

    log.info('Object type = {}'.format(objtype))
    log.info('Number of objects = {}'.format(nobj))
    log.info('Chunksize = {}'.format(chunksize))
    log.info('Number of chunks = {}'.format(nchunk))

    # Optionally zoom into a portion of the brick
    survey = LegacySurveyData()
    brickinfo = survey.get_brick_by_name(brickname)
    brickwcs = wcs_for_brick(brickinfo)
    W, H, pixscale = brickwcs.get_width(), brickwcs.get_height(), brickwcs.pixel_scale()
    print(W, H, pixscale)

    log.info('Brick = {}'.format(brickname))
    if args.zoom is not None: # See also runbrick.stage_tims()
        (x0,x1,y0,y1) = args.zoom
        W = x1-x0
        H = y1-y0
        brickwcs = brickwcs.get_subimage(x0, y0, W, H)
        log.info('Zoom (pixel boundaries) = {}'.format(args.zoom))
    targetrd = np.array([brickwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])
 
    radec_center = brickwcs.radec_center()
    log.info('RA, Dec center = {}'.format(radec_center))
    log.info('Brick = {}'.format(brickname))

    # Pack the input parameters into a meta-data table.
    metacat = Table()
    metacat['brickname'] = Column([brickname],dtype='S10')
    metacat['objtype'] = Column([objtype],dtype='S10')
    metacat['nobj'] = Column([args.nobj],dtype='i4')
    metacat['chunksize'] = Column([args.chunksize],dtype='i2')
    metacat['nchunk'] = Column([nchunk],dtype='i2')
    #metacat['RA'] = Column([ra_range],dtype='f8')
    #metacat['DEC'] = Column([dec_range],dtype='f8')
    if args.zoom is None:
        metacat['zoom'] = Column([[0,3600,0,3600]],dtype='i4')
    else:
        metacat['zoom'] = Column([args.zoom],dtype='i4')
    metacat['rmag_range'] = Column([args.rmag_range],dtype='f4')

    # Deal with the random seed
    if args.seed is not None:
        log.info('Random seed = {}'.format(args.seed))
        metacat['seed'] = Column([args.seed],dtype='i4')
    rand = np.random.RandomState(args.seed)
    seeds = rand.random_integers(0,2**18,nchunk)
    
    if 'DECALS_SIM_DIR' in os.environ:
        decals_sim_dir = os.getenv('DECALS_SIM_DIR')
    else:
        decals_sim_dir = '.'
    outdir = os.path.join(decals_sim_dir, brickname)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    metafile = os.path.join(outdir,'metacat-'+brickname+'-'+lobjtype+'.fits')
    log.info('Writing {}'.format(metafile))
    if os.path.isfile(metafile):
        os.remove(metafile)
    metacat.write(metafile)

    # Work in chunks.
    for ichunk in range(nchunk):
        log.info('Working on chunk {:02d}/{:02d}'.format(ichunk+1,nchunk))
        chunksuffix = '{:02d}'.format(ichunk)

        # There's probably a smarter way to do this
        nobjchunk = np.min((nobj,chunksize))
        if (nchunk>1) and (ichunk==(nchunk-1)):
            nobjchunk = np.max((nobjchunk,nobj%((nchunk-1)*chunksize)))

        # Build and write out the simulated object catalog.
        simcat = build_simcat(nobjchunk,brickname,brickwcs,metacat,seeds[ichunk])
        simcatfile = os.path.join(outdir,'simcat-'+brickname+'-'+
                                  lobjtype+'.fits')
        log.info('Writing {}'.format(simcatfile))
        if os.path.isfile(simcatfile):
            os.remove(simcatfile)
        simcat.write(simcatfile)

#       # Use Tractor to just process the blobs containing the simulated sources. 
        simdecals = SimDecals(metacat=metacat, simcat=simcat,
                              survey_dir=args.survey_dir, output_dir=outdir)
        blobxy = zip(simcat['x'],simcat['y'])
        print('args.stage=',args.stage)
        run_brick(brickname, survey=simdecals, 
                  threads=args.threads, zoom=args.zoom, wise=False,
                  forceAll=True, writePickles=False, do_calibs=True,
                  write_metrics=False, pixPsf=True, blobxy=blobxy, 
                  early_coadds=False, splinesky=True) #stages=[args.stage], splinesky=True)
        log.info('Cleaning up...')
        #mv tractor catalogue, coadd/image.jpg and resid.jpg to outdir/
        #shutil.move(os.path.join(outdir,'tractor',brickname[:3],
        #                         'tractor-'+brickname+'.fits'),
        #            os.path.join(outdir,'tractor-'+brickname+'-'+
        #                         lobjtype+'-'+chunksuffix+'.fits'))
        #shutil.move(os.path.join(outdir,'coadd',brickname[:3],brickname,
        #                         'legacysurvey-'+brickname+'-image.jpg'),
        #            os.path.join(outdir,'qa-'+brickname+'-'+lobjtype+
        #                         '-image-'+chunksuffix+'.jpg'))
        #shutil.move(os.path.join(outdir,'coadd',brickname[:3],brickname,
        #                         'legacysurvey-'+brickname+'-resid.jpg'),
        #            os.path.join(outdir,'qa-'+brickname+'-'+lobjtype+
        #                         '-resid-'+chunksuffix+'.jpg'))

        # Write a log file

    log.info('All done!')
        
if __name__ == '__main__':
    main()
