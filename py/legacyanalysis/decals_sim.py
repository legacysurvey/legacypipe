#!/usr/bin/env python

"""Try to hack decals_sim so we don't have to make copies of the data.

decals_sim -b 2428p117 -n 2000 --chunksize 500 -o STAR --seed 7618 --threads 15 > ~/2428p117.log 2>&1 & 

2428p117
3216p000
1315p240

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

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
import os
import sys
import shutil
import logging
import argparse
import pdb
import photutils
import galsim

import numpy as np
import matplotlib.pyplot as plt
from pkg_resources import resource_filename

from astropy.table import Table, Column, vstack

from astrometry.libkd.spherematch import match_radec

from tractor.psfex import PsfEx, PsfExModel
from tractor.basics import GaussianMixtureEllipsePSF, RaDecPos

from legacypipe.runbrick import run_brick
from legacypipe.decam import DecamImage
from legacypipe.common import LegacySurveyData, wcs_for_brick, ccds_touching_wcs

from pickle import dump

class SimDecals(LegacySurveyData):
    def __init__(self, survey_dir=None, metacat=None, simcat=None, output_dir=None,\
                       add_sim_noise=False):
        super(SimDecals, self).__init__(survey_dir=survey_dir, output_dir=output_dir)
        self.metacat = metacat
        self.simcat = simcat
        self.add_sim_noise= add_sim_noise

    def get_image_object(self, t):
        return SimImage(self, t)


class SimImage(DecamImage):
    def __init__(self, survey, t):
        super(SimImage, self).__init__(survey, t)
        self.t = t

    def get_tractor_image(self, **kwargs):
        tim = super(SimImage, self).get_tractor_image(**kwargs)
        if tim is None: # this can be None when the edge of a CCD overlaps
            return tim

        # Initialize the object stamp class
        if 'SEED' in self.survey.metacat.columns:
            seed = self.survey.metacat['SEED']
        else:
            seed = None

        objtype = self.survey.metacat['OBJTYPE']
        objstamp = BuildStamp(tim, gain=self.t.arawgain, seed=seed)

        # Grab the data and inverse variance images [nanomaggies!]
        image = galsim.Image(tim.getImage())
        invvar = galsim.Image(tim.getInvvar())
        # Also store galaxy sims and sims invvar
        sims_image = image.copy() 
        sims_image.fill(0.0)
        sims_ivar = sims_image.copy()
        # N sims, 4 box corners (xmin,xmax,ymin,ymax) for each sim
        tim.sims_xy = np.empty((len(self.survey.simcat),4))+np.nan 

        # Store simulated galaxy images in tim object 
        # Loop on each object.
        ######
        #in_flux= np.zeros(len(self.survey.simcat))+np.nan
        #ap_flux= in_flux.copy()
        #added_flux= in_flux.copy()
        #####
        for ii, obj in enumerate(self.survey.simcat):
            if objtype == 'STAR':
                #stamp, in_flux[ii],ap_flux[ii],added_flux[ii] = objstamp.star(obj)
                stamp = objstamp.star(obj)
            elif objtype == 'ELG':
                stamp = objstamp.elg(obj)
            elif objtype == 'LRG':
                stamp = objstamp.lrg(obj)
            
            # Make sure the object falls on the image and then add Poisson noise.
            overlap = stamp.bounds & image.bounds
            if (overlap.area() > 0):
                stamp = stamp[overlap]      
                ivarstamp = invvar[overlap]
                # Add noise to simulated source
                if self.survey.add_sim_noise:
                    stamp, ivarstamp = objstamp.addnoise(stamp, ivarstamp)
                # Only where stamps will in inserted
                sims_image[overlap] += stamp 
                sims_ivar[overlap] += ivarstamp
                tim.sims_xy[ii, :] = [overlap.xmin-1, overlap.xmax-1,
                                      overlap.ymin-1, overlap.ymax-1] # galsim 1st index is 1

                image[overlap] += stamp
                invvar[overlap] = ivarstamp

                #print('HACK!!!')
                #galsim.fits.write(stamp, 'stamp-{:02d}.fits'.format(ii), clobber=True)
                #galsim.fits.write(ivarstamp, 'ivarstamp-{:02d}.fits'.format(ii), clobber=True)

                if np.min(sims_ivar.array) < 0:
                    print('Negative invvar!')
                    import pdb ; pdb.set_trace()
        tim.sims_image = sims_image.array
        tim.sims_inverr = np.sqrt(sims_ivar.array)
        tim.sims_xy = tim.sims_xy.astype(int)
        tim.data = image.array
        tim.inverr = np.sqrt(invvar.array)
        ##########
        #print('objstamp.gsparams=', objstamp.gsparams) 
        #print('self.survey.simcat STAMP_%sFLUX/%sFLUX=' % (objstamp.band,objstamp.band),\
        #         self.survey.simcat[objstamp.band+'FLUX']/self.survey.simcat['STAMP_'+objstamp.band+'FLUX']) 
        #print('exiting early')
        #sys.exit()
        ##########
        #print('min,max,median apflux/influx= ',np.min(ap_flux/in_flux),np.max(ap_flux/in_flux),np.median(ap_flux/in_flux))
        #print('min,max,median added_flux/apflux= ',np.min(added_flux/ap_flux),np.max(added_flux/ap_flux),np.median(added_flux/ap_flux))
        #print('exiting before write aper.pickle')
        #sys.exit()
        #stamp_c= np.array(stamp_c)
        #fout=open('aper.pickle','w')
        #dump((tim.data,tim.sims_image,self.survey.simcat,stamp_c),fout)
        #fout.close()
        #print('exiting after writig aper.pickle')
        #sys.exit()
        ##########

        #print('HACK!!!')
        #galsim.fits.write(invvar, 'invvar.fits'.format(ii), clobber=True)
        #import pdb ; pdb.set_trace()
        return tim

def perc_diff(test,answer):
    '''returns percentage that test and answer differ by'''
    return 100.*abs(test-answer)/answer

class BuildStamp():
    def __init__(self,tim, gain=4.0, seed=None):
        """Initialize the BuildStamp object with the CCD-level properties we need."""

        self.band = tim.band.strip().upper()
        # Folding_threshold chosen so that percent diff between stamp.added_flux and requested AB mag flux is < 1e-2%
        self.gsparams = galsim.GSParams(maximum_fft_size=2L**30L,\
                                        folding_threshold=1.e-6) 
        #print('FIX ME!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.gsdeviate = galsim.BaseDeviate()
        #if seed is None:
        #    self.gsdeviate = galsim.BaseDeviate()
        #else:
        #    self.gsdeviate = galsim.BaseDeviate(seed)

        self.wcs = tim.getWcs()
        self.psf = tim.getPsf()

        ## Get the Galsim-compatible WCS as well.
        #import pdb ; pdb.set_trace()
        #galsim_wcs, _ = galsim.wcs.readFromFitsHeader(
        #    galsim.fits.FitsHeader(tim.pvwcsfn))
        #self.galsim_wcs = galsim_wcs

        # zpscale equivalent to magzpt = self.t.ccdzpt+2.5*np.log10(self.t.exptime)
        self.zpscale = tim.zpscale      # nanomaggies-->ADU conversion factor
        self.nano2e = self.zpscale*gain # nanomaggies-->electrons conversion factor

    def setlocal(self,obj):
        """Get the pixel positions, local pixel scale, and local PSF.""" 

        xx, yy = self.wcs.positionToPixel(RaDecPos(obj['RA'], obj['DEC']))
        self.pos = galsim.PositionD(xx, yy)
        self.xpos = int(self.pos.x)
        self.ypos = int(self.pos.y)
        self.offset = galsim.PositionD(self.pos.x-self.xpos, self.pos.y-self.ypos)

        # Get the local pixel scale [arcsec/pixel] and the local Galsim WCS
        # object.
        cd = self.wcs.cdAtPixel(self.pos.x, self.pos.y)
        self.pixscale = np.sqrt(np.linalg.det(cd))*3600.0
        #self.localwcs = self.galsim_wcs.local(image_pos=self.pos)
        
        # Get the local PSF
        psfim = self.psf.getPointSourcePatch(self.xpos, self.ypos).getImage()
        #plt.imshow(psfim) ; plt.show()
        
        # make galsim PSF object
        self.localpsf = galsim.InterpolatedImage(galsim.Image(psfim),scale=self.pixscale,\
                                    gsparams=self.gsparams)

    def addnoise(self, stamp, ivarstamp):
        """Add noise to the object postage stamp.  Remember that STAMP and IVARSTAMP
        are in units of nanomaggies and 1/nanomaggies**2, respectively.

        """
        # Get the set of pixels masked in the input inverse variance image/stamp.
        mask = ivarstamp.copy()
        mask = mask.array
        mask[mask != 0] = 1
        
        varstamp = ivarstamp.copy()
        varstamp.invertSelf() # input data, convert to variance
        varstamp *= self.nano2e**2 # [electron^2]
            
        # Add the variance of the object to the variance image (in electrons).
        stamp *= self.nano2e       # [noiseless stamp, electron]
        objvar = galsim.Image(np.sqrt(stamp.array**2), scale=stamp.scale) 
        objvar.setOrigin(galsim.PositionI(stamp.xmin, stamp.ymin))

        # Add Poisson noise
        stamp.addNoise(galsim.VariableGaussianNoise( # pass the random seed
            self.gsdeviate, objvar))
        varstamp += objvar

        # Convert back to [nanomaggies]
        stamp /= self.nano2e      
        varstamp /= self.nano2e**2

        ivarstamp = varstamp.copy()
        ivarstamp.invertSelf()

        # Remask pixels that were masked in the original inverse variance stamp.
        ivarstamp *= mask

        return stamp, ivarstamp

    def convolve_and_draw(self,obj):
        """Convolve the object with the PSF and then draw it."""
        obj = galsim.Convolve([obj, self.localpsf])
        stamp = obj.drawImage(offset=self.offset, wcs=self.localwcs,
                              method='no_pixel')
        stamp.setCenter(self.xpos, self.ypos)
        return stamp

    def star(self,obj):
        """Render a star (PSF)."""
        # Set total stamp flux to input flux
        self.setlocal(obj)
        #flux = obj[self.band+'FLUX'] # [nanomaggies]
        #psf = self.localpsf.withFlux(flux)
        psf = self.localpsf.withFlux(1.)
        stamp = psf.drawImage(offset=self.offset, scale=self.pixscale, method='no_pixel')
        # Get flux in 7'' diameter aperture
        diam = 7/self.pixscale
        # If aperture larger than stamp, Redraw with stamp 2 pix larger than aperture
        width= stamp.bounds.xmax-stamp.bounds.xmin
        height= stamp.bounds.ymax-stamp.bounds.ymin
        if diam > width and diam > height:
            nxy= int(diam)+2
            stamp = psf.drawImage(nx=nxy,ny=nxy, offset=self.offset, scale=self.pixscale, method='no_pixel')
        assert(diam <= float(stamp.bounds.xmax-stamp.bounds.xmin))
        assert(diam <= float(stamp.bounds.ymax-stamp.bounds.ymin))
        # Aperture fits on stamp
        apers= photutils.CircularAperture((stamp.trueCenter().x,stamp.trueCenter().y), r=diam/2)
        apy_table = photutils.aperture_photometry(stamp.array, apers)
        apflux= np.array(apy_table['aperture_sum'])[0]
        # Set total flux to aperture flux
        #psf = self.localpsf.withFlux(flux*flux/apflux) #apflux
        flux = obj[self.band+'FLUX'] # [nanomaggies]
        psf = self.localpsf.withFlux(flux*(2.-apflux/stamp.added_flux))
        stamp = psf.drawImage(offset=self.offset, scale=self.pixscale, method='no_pixel')
        #stamp looses less than 0.01% of requested flux
        assert( perc_diff(stamp.added_flux,psf.getFlux()) < 1.e-2) 
        # Convert stamp's center to its corresponding center on full tractor image
        stamp.setCenter(self.xpos, self.ypos)
        
        #print('pixelscale=%.5f, input flux=%.3f, apflux=%.3f, stamp flux=%.3f' % (self.pixscale,flux,apflux,stamp.added_flux))
        return stamp #, flux,apflux,stamp.added_flux

    def elg(self,objinfo):
        """Create an ELG (disk-like) galaxy."""

        self.setlocal(objinfo)

        objflux = objinfo[self.band+'FLUX'] # [nanomaggies]
        obj = galsim.Sersic(float(objinfo['SERSICN_1']), half_light_radius=
                            float(objinfo['R50_1']),
                            flux=objflux,gsparams=self.gsparams)
        obj = obj.shear(q=float(objinfo['BA_1']), beta=
                        float(objinfo['PHI_1'])*galsim.degrees)
        stamp = self.convolve_and_draw(obj)
        return stamp

    def lrg(self,objinfo):
        """Create an LRG (spheroidal) galaxy."""
        obj = galsim.Sersic(float(objinfo['SERSICN_1']),half_light_radius=
                            float(objinfo['R50_1']),
                            flux=self.objflux,gsparams=self.gsparams)
        obj = obj.shear(q=float(objinfo['BA_1']),beta=
                        float(objinfo['PHI_1'])*galsim.degrees)
        stamp = self.convolve_and_draw(obj)
        return stamp

class _GaussianMixtureModel():
    """Read and sample from a pre-defined Gaussian mixture model.

    """
    def __init__(self, weights, means, covars, covtype):
        self.weights = weights
        self.means = means
        self.covars = covars
        self.covtype = covtype
        self.n_components, self.n_dimensions = self.means.shape
    
    @staticmethod
    def save(model, filename):
        from astropy.io import fits
        hdus = fits.HDUList()
        hdr = fits.Header()
        hdr['covtype'] = model.covariance_type
        hdus.append(fits.ImageHDU(model.weights_, name='weights', header=hdr))
        hdus.append(fits.ImageHDU(model.means_, name='means'))
        hdus.append(fits.ImageHDU(model.covars_, name='covars'))
        hdus.writeto(filename, clobber=True)
        
    @staticmethod
    def load(filename):
        from astropy.io import fits
        hdus = fits.open(filename, memmap=False)
        hdr = hdus[0].header
        covtype = hdr['covtype']
        model = _GaussianMixtureModel(
            hdus['weights'].data, hdus['means'].data, hdus['covars'].data, covtype)
        hdus.close()
        return model
    
    def sample(self, n_samples=1, random_state=None):
        
        if self.covtype != 'full':
            return NotImplementedError(
                'covariance type "{0}" not implemented yet.'.format(self.covtype))
        
        # Code adapted from sklearn's GMM.sample()
        if random_state is None:
            random_state = np.random.RandomState()

        weight_cdf = np.cumsum(self.weights)
        X = np.empty((n_samples, self.n_dimensions))
        rand = random_state.rand(n_samples)
        # decide which component to use for each sample
        comps = weight_cdf.searchsorted(rand)
        # for each component, generate all needed samples
        for comp in range(self.n_components):
            # occurrences of current component in X
            comp_in_X = (comp == comps)
            # number of those occurrences
            num_comp_in_X = comp_in_X.sum()
            if num_comp_in_X > 0:
                X[comp_in_X] = random_state.multivariate_normal(
                    self.means[comp], self.covars[comp], num_comp_in_X)
        return X

def no_overlapping_radec(ra,dec, bounds, random_state=None, dist=5.0/3600):
    '''resamples ra,dec where they are within dist of each other 
    ra,dec -- numpy arrays [degrees]
    bounds-- list of [ramin,ramax,decmin,decmax]
    random_state-- np.random.RandomState() object
    dist-- separation in degrees, 
           sqrt(x^2+y^2) where x=(dec2-dec1), y=cos(avg(dec2,dec1))*(ra2-ra1)
    '''
    if random_state is None:
        random_state = np.random.RandomState()
    # ra,dec indices of just neighbors within "dist" away, just nerest neighbor of those
    m1, m2, d12 = match_radec(ra.copy(),dec.copy(), ra.copy(),dec.copy(),\
                              dist, nearest=True,notself=True) 

    cnt = 1
    print("after iter=%d, have overlapping ra,dec %d/%d" % (cnt, len(m2),ra.shape[0]))
    while len(m2) > 0:
        ra[m2]= random_state.uniform(bounds[0], bounds[1], len(m2))
        dec[m2]= random_state.uniform(bounds[2], bounds[3], len(m2))
        m1, m2, d12 = match_radec(ra.copy(),dec.copy(), ra.copy(),dec.copy(),\
                                  dist, nearest=True,notself=True) 
        cnt += 1
        print("after iter=%d, have overlapping ra,dec %d/%d" % (cnt, len(m2),ra.shape[0]))
        if cnt > 30:
            print('Crash, could not get non-overlapping ra,dec in 30 iterations')
            raise ValueError
    return ra, dec

def build_simcat(nobj=None, brickname=None, brickwcs=None, meta=None, seed=None, noOverlap=True):
    """Build the simulated object catalog, which depends on OBJTYPE."""

    rand = np.random.RandomState(seed)

    # Assign central coordinates uniformly but remove simulated sources which
    # are too near to one another.  Iterate until we have the requisite number
    # of objects.
    bounds = brickwcs.radec_bounds()
    ra = rand.uniform(bounds[0], bounds[1], nobj)
    dec = rand.uniform(bounds[2], bounds[3], nobj)
    if noOverlap:
        ra, dec= no_overlapping_radec(ra,dec, bounds,
                                      random_state=rand,
                                      dist=5./3600)
    xxyy = brickwcs.radec2pixelxy(ra, dec)

    cat = Table()
    cat['ID'] = Column(np.arange(nobj, dtype='i4'))
    cat['RA'] = Column(ra, dtype='f8')
    cat['DEC'] = Column(dec, dtype='f8')
    cat['X'] = Column(xxyy[1][:], dtype='f4')
    cat['Y'] = Column(xxyy[2][:], dtype='f4')

    if meta['OBJTYPE'] == 'STAR':
        # Read the MoG file and sample from it.
        mogfile = resource_filename('legacypipe', os.path.join('data', 'star_colors_mog.fits'))
        mog = _GaussianMixtureModel.load(mogfile)
        grzsample = mog.sample(nobj, random_state=rand)
        rz = grzsample[:, 0]
        gr = grzsample[:, 1]

    elif meta['OBJTYPE'] == 'ELG':
        gr_range = [-0.3, 0.5]
        rz_range = [0.0, 1.5]
        sersicn_1_range = [1.0, 1.0]
        r50_1_range = [0.5, 2.5]
        ba_1_range = [0.2, 1.0]

        gr = rand.uniform(gr_range[0], gr_range[1], nobj)
        rz = rand.uniform(rz_range[0], rz_range[1], nobj)
        sersicn_1 = rand.uniform(sersicn_1_range[0], sersicn_1_range[1], nobj)
        r50_1 = rand.uniform(r50_1_range[0], r50_1_range[1], nobj)
        ba_1 = rand.uniform(ba_1_range[0], ba_1_range[1], nobj)
        phi_1 = rand.uniform(0.0, 180.0, nobj)

        cat['SERSICN_1'] = Column(sersicn_1, dtype='f4')
        cat['R50_1'] = Column(r50_1, dtype='f4')
        cat['BA_1'] = Column(ba_1, dtype='f4')
        cat['PHI_1'] = Column(phi_1, dtype='f4')

        ## Bulge parameters
        #bulge_r50_range = [0.1,1.0]
        #bulge_n_range = [3.0,5.0]
        #bdratio_range = [0.0,1.0] # bulge-to-disk ratio

    else:
        print('Unrecognized OBJTYPE!')
        return 0

    # For convenience, also store the grz fluxes in nanomaggies.
    rmag_range = np.squeeze(meta['RMAG_RANGE'])
    rmag = rand.uniform(rmag_range[0], rmag_range[1], nobj)
    #gr = rand.uniform(gr_range[0], gr_range[1], nobj)
    #rz = rand.uniform(rz_range[0], rz_range[1], nobj)

    cat['R'] = rmag
    cat['GR'] = gr
    cat['RZ'] = rz
    cat['GFLUX'] = 1E9*10**(-0.4*(rmag+gr)) # [nanomaggies]
    cat['RFLUX'] = 1E9*10**(-0.4*rmag)      # [nanomaggies]
    cat['ZFLUX'] = 1E9*10**(-0.4*(rmag-rz)) # [nanomaggies]

    return cat



def get_parser():
    '''return parser object, tells it what options to look for
    options can come from a list of strings or command line'''
    parser = argparse.ArgumentParser(formatter_class=argparse.
                                     ArgumentDefaultsHelpFormatter,
                                     description='DECaLS simulations.')
    parser.add_argument('-n', '--nobj', type=long, default=500, metavar='', 
                        help='number of objects to simulate (required input)')
    parser.add_argument('-ic', '--ith_chunk', type=long, default=None, metavar='', 
                        help='run the ith chunk, 0-999')
    parser.add_argument('-c', '--nchunk', type=long, default=1, metavar='', 
                        help='run chunks 0 to nchunk')
    parser.add_argument('-b', '--brick', type=str, default='2428p117', metavar='', 
                        help='simulate objects in this brick')
    parser.add_argument('-o', '--objtype', type=str, choices=['STAR','ELG', 'LRG', 'QSO', 'LSB'], default='STAR', metavar='', 
                        help='insert these into images') 
    parser.add_argument('-t', '--threads', type=int, default=1, metavar='', 
                        help='number of threads to use when calling The Tractor')
    parser.add_argument('-s', '--seed', type=long, default=None, metavar='', 
                        help='random number seed, determines chunk seeds 0-999')
    parser.add_argument('-z', '--zoom', nargs=4, default=(0, 3600, 0, 3600), type=int, metavar='', 
                        help='see runbrick.py; (default is 0 3600 0 3600)')
    parser.add_argument('-survey-dir', '--survey_dir', metavar='', 
                        help='Location of survey-ccds*.fits.gz')
    parser.add_argument('--rmag-range', nargs=2, type=float, default=(18, 26), metavar='', 
                        help='r-band magnitude range')
    parser.add_argument('--add_sim_noise', action="store_true", help="set to add noise to simulated sources")
    parser.add_argument('--all-blobs', action='store_true', 
                        help='Process all the blobs, not just those that contain simulated sources.')
    parser.add_argument('--stage', choices=['tims', 'image_coadds', 'srcs', 'fitblobs', 'coadds'],
                        type=str, default='writecat', metavar='', help='Run up to the given stage')
    parser.add_argument('--early_coadds', action='store_true',
                        help='add this option to make the JPGs before detection/model fitting')
    parser.add_argument('-v', '--verbose', action='store_true', help='toggle on verbose output')
    return parser
   
def get_metadata_other(args=None):
    '''Parses input and returns dict containing 
    metatable,seeds,brickwcs, other goodies'''
    # Tell parser options to look for
    parser= get_parser()    
    # If args is not None the input args are used instead of command line
    args = parser.parse_args(args=args)
    print('decals_sim.py args: ',args)

    max_nobj=500
    max_nchunk=1000
    if args.ith_chunk is not None: assert(args.ith_chunk <= max_nchunk-1)
    assert(args.nchunk <= max_nchunk)
    assert(args.nobj <= max_nobj)
    if args.ith_chunk is not None: 
        assert(args.nchunk == 1) #if choose a chunk, only doing 1 chunk
    if args.nobj is None:
        parser.print_help()
        sys.exit(1)

    # Set the debugging level
    if args.verbose:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO
    logging.basicConfig(format='%(message)s', level=lvl, stream=sys.stdout)
    log = logging.getLogger('__name__')

    brickname = args.brick
    objtype = args.objtype.upper()
    lobjtype = objtype.lower()

    for obj in ('LRG', 'LSB', 'QSO'):
        if objtype == obj:
            log.warning('{} objtype not yet supported!'.format(objtype))
            return 0

    # Deal with the paths.
    if 'DECALS_SIM_DIR' in os.environ:
        decals_sim_dir = os.getenv('DECALS_SIM_DIR')
    else:
        decals_sim_dir = '.'
        
    nobj = args.nobj
    nchunk = args.nchunk
    rand = np.random.RandomState(args.seed) # determines seed for all chunks
    seeds = rand.random_integers(0,2**18, max_nchunk)

    log.info('Object type = {}'.format(objtype))
    log.info('Number of objects = {}'.format(nobj))
    log.info('Number of chunks = {}'.format(nchunk))

    # Optionally zoom into a portion of the brick
    survey = LegacySurveyData()
    brickinfo = survey.get_brick_by_name(brickname)
    brickwcs = wcs_for_brick(brickinfo)
    W, H, pixscale = brickwcs.get_width(), brickwcs.get_height(), brickwcs.pixel_scale()

    log.info('Brick = {}'.format(brickname))
    if args.zoom is not None: # See also runbrick.stage_tims()
        (x0, x1, y0, y1) = args.zoom
        W = x1 - x0
        H = y1 - y0
        brickwcs = brickwcs.get_subimage(x0, y0, W, H)
        log.info('Zoom (pixel boundaries) = {}'.format(args.zoom))
    targetrd = np.array([brickwcs.pixelxy2radec(x, y) for x, y in
                         [(1,1), (W,1), (W,H), (1,H), (1,1)]])
 
    radec_center = brickwcs.radec_center()
    log.info('RA, Dec center = {}'.format(radec_center))
    log.info('Brick = {}'.format(brickname))

    # Pack the input parameters into a meta-data table and write out.
    metacols = [
        ('BRICKNAME', 'S10'),
        ('OBJTYPE', 'S10'),
        ('NOBJ', 'i4'),
        ('CHUNKSIZE', 'i2'),
        ('NCHUNK', 'i2'),
        ('ZOOM', 'i4', (4,)),
        ('SEED', 'S20'),
        ('RMAG_RANGE', 'f4', (2,))]
    metacat = Table(np.zeros(1, dtype=metacols))

    metacat['BRICKNAME'] = brickname
    metacat['OBJTYPE'] = objtype
    metacat['NOBJ'] = args.nobj
    metacat['NCHUNK'] = nchunk
    metacat['ZOOM'] = args.zoom
    metacat['RMAG_RANGE'] = args.rmag_range
    if not args.seed:
        log.info('Random seed = {}'.format(args.seed))
        metacat['SEED'] = args.seed
   
    metacat_dir = os.path.join(decals_sim_dir, brickname,lobjtype)    
    if not os.path.exists(metacat_dir): 
        os.makedirs(metacat_dir)
    
    metafile = os.path.join(metacat_dir, 'metacat-{}-{}.fits'.format(brickname, lobjtype))
    log.info('Writing {}'.format(metafile))
    if os.path.isfile(metafile):
        os.remove(metafile)
    metacat.write(metafile)
    
    # With this can get simcat for each chunk
    return dict(seeds=seeds,\
                brickname=brickname, \
                metacat=metacat,\
                decals_sim_dir= decals_sim_dir,\
                brickwcs= brickwcs, \
                lobjtype=lobjtype,\
                nobj=nobj,\
                log=log,\
                nchunk=nchunk,\
                metacat_dir=metacat_dir,\
                args=args)

def get_ith_simcat(ith_chunk, d=None):
    '''add simcat, simcat_dir to dict d
    simcat -- contains randomized ra,dec and PDF fluxes etc for ith chunk
    d -- dict returned by get_metadata_others()'''
    assert(d is not None)
    chunksuffix = '{:02d}'.format(ith_chunk)
    # Build and write out the simulated object catalog.
    seed= d['seeds'][ith_chunk]
    simcat = build_simcat(d['nobj'], d['brickname'], d['brickwcs'], d['metacat'], seed)
    # Save file
    simcat_dir = os.path.join(d['metacat_dir'],'%3.3d' % ith_chunk)    
    if not os.path.exists(simcat_dir): 
        os.makedirs(simcat_dir)
    simcatfile = os.path.join(simcat_dir, 'simcat-{}-{}-{}.fits'.format(d['brickname'], d['lobjtype'], chunksuffix))
    d['log'].info('Writing {}'.format(simcatfile))
    if os.path.isfile(simcatfile):
        os.remove(simcatfile)
    simcat.write(simcatfile)
    # add to dict
    d['simcat']= simcat
    d['simcat_dir']= simcat_dir

def do_one_chunk(d=None):
    '''Run tractor
    Can be run as 1) a loop over nchunks or 2) for one chunk
    d -- dict returned by get_metadata_others() AND added to by get_ith_simcat()'''
    assert(d is not None)
    simdecals = SimDecals(metacat=d['metacat'], simcat=d['simcat'], output_dir=d['simcat_dir'], \
                          add_sim_noise=d['args'].add_sim_noise)
    # Use Tractor to just process the blobs containing the simulated sources.
    if d['args'].all_blobs:
        blobxy = None
    else:
        blobxy = zip(d['simcat']['X'], d['simcat']['Y'])

    run_brick(d['brickname'], simdecals, threads=d['args'].threads, zoom=d['args'].zoom,
              wise=False, forceAll=True, writePickles=False, do_calibs=False,
              write_metrics=False, pixPsf=True, blobxy=blobxy, early_coadds=d['args'].early_coadds,
              splinesky=True, ceres=False, stages=[ d['args'].stage ], plots=False,
              plotbase='sim')

def do_ith_cleanup(ith_chunk=None, d=None):
    '''for each chunk that finishes running, 
    Remove unecessary files and give unique names to all others
    d -- dict returned by get_metadata_others() AND added to by get_ith_simcat()'''
    assert(ith_chunk is not None and d is not None) 
    d['log'].info('Cleaning up...')
    chunksuffix = '{:02d}'.format(ith_chunk)
    brickname= d['brickname']
    output_dir= d['simcat_dir']
    lobjtype= d['lobjtype'] 
    shutil.copy(os.path.join(output_dir, 'tractor', brickname[:3],
                             'tractor-{}.fits'.format(brickname)),
                os.path.join(output_dir, 'tractor-{}-{}-{}.fits'.format(
                    brickname, lobjtype, chunksuffix)))
    for suffix in ('image', 'model', 'resid', 'simscoadd'):
        shutil.copy(os.path.join(output_dir,'coadd', brickname[:3], brickname,
                                 'legacysurvey-{}-{}.jpg'.format(brickname, suffix)),
                                 os.path.join(output_dir, 'qa-{}-{}-{}-{}.jpg'.format(
                                     brickname, lobjtype, suffix, chunksuffix)))
    shutil.rmtree(os.path.join(output_dir, 'coadd'))
    shutil.rmtree(os.path.join(output_dir, 'tractor'))
    d['log'].info("Finished chunk %3.3d" % ith_chunk)


def main(args=None):
    """Main routine which parses the optional inputs."""
    # Get info that is independent of chunks
    kwargs= get_metadata_other(args=args)
    # do chunks
    if kwargs['args'].ith_chunk is not None: 
        chunk_list= [kwargs['args'].ith_chunk]
    else: 
        chunk_list= range(kwargs['nchunk'])
    for ith_chunk in chunk_list:
        kwargs['log'].info('Working on chunk {:02d}/{:02d}'.format(ith_chunk,kwargs['nchunk']-1))
        # Random ra,dec and source properties
        get_ith_simcat(ith_chunk, d=kwargs)
        # Run tractor
        do_one_chunk(d=kwargs)
        # Clean up output
        do_ith_cleanup(ith_chunk=ith_chunk, d=kwargs)
    kwargs['log'].info('All done!')
     
if __name__ == '__main__':
    main()
