from __future__ import print_function
import os
import numpy as np
import fitsio
from tractor.utils import get_class_from_name
from tractor.basics import NanoMaggies, ConstantFitsWcs, LinearPhotoCal
from tractor.image import Image
from tractor.sky import ConstantSky
from tractor.tractortime import TAITime
from astrometry.util.file import trymakedirs
from astrometry.util.fits import fits_table
from legacypipe.survey import SimpleGalaxy

'''
Generic image handling code.
'''

from astrometry.util.plotutils import PlotSequence
psgalnorm = PlotSequence('norms')

class LegacySurveyImage(object):
    '''A base class containing common code for the images we handle.

    You probably shouldn't need to directly instantiate this class,
    but rather use the recipe described in the __init__ method.

    Objects of this class represent the metadata we have on an image,
    and are used to handle some of the details of going from an entry
    in the CCDs table to a tractor Image object.

    '''

    def __init__(self, survey, ccd):
        '''
        Create a new LegacySurveyImage object, from a LegacySurveyData object,
        and one row of a CCDs fits_table object.

        You may not need to instantiate this class directly, instead using
        survey.get_image_object():

            survey = LegacySurveyData()
            # targetwcs = ....
            # ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)
            ccds = survey.get_ccds()
            im = survey.get_image_object(ccds[0])
            # which does the same thing as:
            im = DecamImage(survey, ccds[0])

        Or, if you have a Community Pipeline-processed input file and
        FITS HDU extension number:

            survey = LegacySurveyData()
            ccds = exposure_metadata([filename], hdus=[hdu])
            im = DecamImage(survey, ccds[0])

        Perhaps the most important method in this class is
        *get_tractor_image*.

        '''
        self.survey = survey

        imgfn = ccd.image_filename.strip()
        if os.path.exists(imgfn):
            self.imgfn = imgfn
        else:
            self.imgfn = os.path.join(self.survey.get_image_dir(), imgfn)

        self.hdu     = ccd.image_hdu
        self.expnum  = ccd.expnum
        self.ccdname = ccd.ccdname.strip()
        self.band    = ccd.filter.strip()
        self.exptime = ccd.exptime
        self.camera  = ccd.camera.strip()

        # Photometric and astrometric zeropoints
        self.ccdzpt = ccd.ccdzpt
        self.dradec = (ccd.ccdraoff / 3600., ccd.ccddecoff / 3600.)
        
        self.fwhm    = ccd.fwhm
        self.propid  = ccd.propid
        # in arcsec/pixel
        self.pixscale = 3600. * np.sqrt(np.abs(ccd.cd1_1 * ccd.cd2_2 -
                                               ccd.cd1_2 * ccd.cd2_1))
        self.mjdobs = ccd.mjd_obs
        self.width  = ccd.width
        self.height = ccd.height

        super(LegacySurveyImage, self).__init__()
        
    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def photometric_ccds(self, survey, ccds):
        '''
        Returns an index array for the members of the table 'ccds' that are
        photometric.

        Default is to return None, meaning keep all CCDs.
        '''
        return None

    @classmethod
    def ccd_cuts(self, survey, ccds):
        return np.zeros(len(ccds), np.int32)
    
    def get_good_image_slice(self, extent, get_extent=False):
        '''
        extent = None or extent = [x0,x1,y0,y1]

        If *get_extent* = True, returns the new [x0,x1,y0,y1] extent.

        Returns a new pair of slices, or *extent* if the whole image is good.
        '''
        gx0,gx1,gy0,gy1 = self.get_good_image_subregion()
        if gx0 is None and gx1 is None and gy0 is None and gy1 is None:
            return extent
        if extent is None:
            imh,imw = self.get_image_shape()
            extent = (0, imw, 0, imh)
        x0,x1,y0,y1 = extent
        if gx0 is not None:
            x0 = max(x0, gx0)
        if gy0 is not None:
            y0 = max(y0, gy0)
        if gx1 is not None:
            x1 = min(x1, gx1)
        if gy1 is not None:
            y1 = min(y1, gy1)
        if get_extent:
            return (x0,x1,y0,y1)
        return slice(y0,y1), slice(x0,x1)

    def get_good_image_subregion(self):
        '''
        Returns x0,x1,y0,y1 of the good region of this chip,
        or None if no cut should be applied to that edge; returns
        (None,None,None,None) if the whole chip is good.

        This cut is applied in addition to any masking in the mask or
        invvar map.
        '''
        return None,None,None,None

    def get_tractor_image(self, slc=None, radecpoly=None,
                          gaussPsf=False, pixPsf=False, hybridPsf=False,
                          splinesky=False,
                          nanomaggies=True, subsky=True, tiny=10,
                          dq=True, invvar=True, pixels=True,
                          constant_invvar=False):
        '''
        Returns a tractor.Image ("tim") object for this image.
        
        Options describing a subimage to return:

        - *slc*: y,x slice objects
        - *radecpoly*: numpy array, shape (N,2), RA,Dec polygon describing
            bounding box to select.

        Options determining the PSF model to use:

        - *gaussPsf*: single circular Gaussian PSF based on header FWHM value.
        - *pixPsf*: pixelized PsfEx model.
        - *hybridPsf*: combo pixelized PsfEx + Gaussian approx.

        Options determining the sky model to use:
        
        - *splinesky*: median filter chunks of the image, then spline those.

        Options determining the units of the image:

        - *nanomaggies*: convert the image to be in units of NanoMaggies;
          *tim.zpscale* contains the scale value the image was divided by.

        - *subsky*: instantiate and subtract the initial sky model,
          leaving a constant zero sky model?

        '''
        get_dq = dq
        get_invvar = invvar

        band = self.band
        wcs = self.get_wcs()

        x0,x1,y0,y1,slc = self.get_image_extent(wcs=wcs, slc=slc, radecpoly=radecpoly)
        if y1 - y0 < tiny or x1 - x0 < tiny:
            print('Skipping tiny subimage')
            return None

        # Read image pixels
        if pixels:
            print('Reading image slice:', slc)
            img,imghdr = self.read_image(header=True, slice=slc)
            self.check_image_header(imghdr)
        else:
            img = np.zeros((y1-y0, x1-x0), np.float32)
            imghdr = self.read_image_header()
        assert(np.all(np.isfinite(img)))
            
        # Read inverse-variance (weight) map
        if get_invvar:
            invvar = self.read_invvar(slice=slc, clipThresh=0.)
        else:
            invvar = np.ones_like(img)
        assert(np.all(np.isfinite(invvar)))
        if np.all(invvar == 0.):
            print('Skipping zero-invvar image')
            return None
        # Negative invvars (from, eg, fpack decompression noise) cause havoc
        if not np.all(invvar >= 0.):
            raise ValueError('not np.all(invvar >= 0.), hdu=%d fn=%s' % (self.hdu,self.wtfn))

        # Read data-quality (flags) map and zero out the invvars of masked pixels
        if get_dq:
            dq = self.read_dq(slice=slc)
            if dq is not None:
                invvar[dq != 0] = 0.
            if np.all(invvar == 0.):
                print('Skipping zero-invvar image (after DQ masking)')
                return None

        # header 'FWHM' is in pixels
        assert(self.fwhm > 0)
        psf_fwhm = self.fwhm 
        psf_sigma = psf_fwhm / 2.35
        primhdr = self.read_image_primary_header()

        #
        invvar = self.remap_invvar(invvar, primhdr, img, dq)


        sky = self.read_sky_model(splinesky=splinesky, slc=slc,
                                  primhdr=primhdr, imghdr=imghdr)
        skysig1 = getattr(sky, 'sig1', None)
        
        midsky = 0.
        if subsky:
            print('Instantiating and subtracting sky model')
            skymod = np.zeros_like(img)
            sky.addTo(skymod)
            img -= skymod
            midsky = np.median(skymod)
            zsky = ConstantSky(0.)
            zsky.version = getattr(sky, 'version', '')
            zsky.plver = getattr(sky, 'plver', '')
            del skymod
            sky = zsky
            del zsky

        orig_zpscale = zpscale = NanoMaggies.zeropointToScale(self.ccdzpt)
        if nanomaggies:
            # Scale images to Nanomaggies
            img /= zpscale
            invvar = invvar * zpscale**2
            if not subsky:
                sky.scale(1./zpscale)
            zpscale = 1.

        # Compute 'sig1', scalar typical per-pixel noise
        if get_invvar:
            sig1 = 1./np.sqrt(np.median(invvar[invvar > 0]))
        elif skysig1 is not None:
            sig1 = skysig1
            if nanomaggies:
                # skysig1 is in the native units
                sig1 /= orig_zpscale
        else:
            # Estimate per-pixel noise via Blanton's 5-pixel MAD
            slice1 = (slice(0,-5,10),slice(0,-5,10))
            slice2 = (slice(5,None,10),slice(5,None,10))
            mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
            sig1 = 1.4826 * mad / np.sqrt(2.)
            print('sig1 estimate:', sig1)
            invvar *= (1. / sig1**2)
        assert(np.isfinite(sig1))

        if constant_invvar:
            print('Setting constant invvar', 1./sig1**2)
            invvar[invvar > 0] = 1./sig1**2

        if subsky:
            # Warn if the subtracted sky doesn't seem to work well
            # (can happen, eg, if sky calibration product is inconsistent with
            #  the data)
            imgmed = np.median(img[invvar>0])
            if np.abs(imgmed) > sig1:
                print('WARNING: image median', imgmed, 'is more than 1 sigma',
                      'away from zero!')

        # tractor WCS object
        twcs = self.get_tractor_wcs(wcs, x0, y0, primhdr=primhdr, imghdr=imghdr)
                                    
        psf = self.read_psf_model(x0, y0, gaussPsf=gaussPsf, pixPsf=pixPsf,
                                  hybridPsf=hybridPsf,
                                  psf_sigma=psf_sigma,
                                  w=x1 - x0, h=y1 - y0)

        tim = Image(img, invvar=invvar, wcs=twcs, psf=psf,
                    photocal=LinearPhotoCal(zpscale, band=band),
                    sky=sky, name=self.name + ' ' + band)
        assert(np.all(np.isfinite(tim.getInvError())))

        # PSF norm
        tim.band = band
        #print('Computing PSF norm')

        # HACK -- create a local PSF model to instantiate the PsfEx
        # model, which handles non-unit pixel scaling.
        print('-- creating constant PSF model...')
        fullpsf = tim.psf
        th,tw = tim.shape
        tim.psf = fullpsf.constantPsfAt(tw//2, th//2)
        #print('-- created constant PSF model...')

        print('Computing PSF norm...')
        psfnorm = self.psf_norm(tim)
        print('Computed PSF norm:', psfnorm)

        # Galaxy-detection norm
        print('Computing galaxy norm')
        galnorm = self.galaxy_norm(tim)
        print('PSF norm', psfnorm, 'galaxy norm', galnorm)

        tim.psf = fullpsf

        # print('Computing galaxy norm with original PSF')
        # galnorm = self.galaxy_norm(tim)
        # print('PSF norm', psfnorm, 'galaxy norm w/orig PSF', galnorm)

        #assert(galnorm < psfnorm)

        # CP (DECam) images include DATE-OBS and MJD-OBS, in UTC.
        import astropy.time
        mjd_tai = astropy.time.Time(self.mjdobs,
                                    format='mjd', scale='utc').tai.mjd
        tim.time = TAITime(None, mjd=mjd_tai)
        tim.slice = slc
        tim.zpscale = orig_zpscale
        tim.midsky = midsky
        tim.sig1 = sig1
        tim.psf_fwhm = psf_fwhm
        tim.psf_sigma = psf_sigma
        tim.propid = self.propid
        tim.psfnorm = psfnorm
        tim.galnorm = galnorm
        tim.sip_wcs = wcs
        tim.x0,tim.y0 = int(x0),int(y0)
        tim.imobj = self
        tim.primhdr = primhdr
        tim.hdr = imghdr
        tim.plver = primhdr.get('PLVER','').strip()
        tim.skyver = (getattr(sky, 'version', ''), getattr(sky, 'plver', ''))
        tim.wcsver = (getattr(wcs, 'version', ''), getattr(wcs, 'plver', ''))
        tim.psfver = (getattr(psf, 'version', ''), getattr(psf, 'plver', ''))
        if get_dq:
            tim.dq = dq
        tim.dq_saturation_bits = self.dq_saturation_bits
        subh,subw = tim.shape
        tim.subwcs = tim.sip_wcs.get_subimage(tim.x0, tim.y0, subw, subh)
        return tim

    def get_image_extent(self, wcs=None, slc=None, radecpoly=None):
        '''
        Returns x0,x1,y0,y1,slc
        '''
        slc = None
        imh,imw = self.get_image_shape()
        x0,y0 = 0,0
        x1 = x0 + imw
        y1 = y0 + imh

        # Clip to RA,Dec polygon?
        if slc is None and radecpoly is not None:
            from astrometry.util.miscutils import clip_polygon
            imgpoly = [(1,1),(1,imh),(imw,imh),(imw,1)]
            ok,tx,ty = wcs.radec2pixelxy(radecpoly[:-1,0], radecpoly[:-1,1])
            tpoly = list(zip(tx,ty))
            clip = clip_polygon(imgpoly, tpoly)
            clip = np.array(clip)
            if len(clip) == 0:
                return 0,0,0,0,None
            x0,y0 = np.floor(clip.min(axis=0)).astype(int)
            x1,y1 = np.ceil (clip.max(axis=0)).astype(int)
            slc = slice(y0,y1+1), slice(x0,x1+1)
        # Slice?
        if slc is not None:
            sy,sx = slc
            y0,y1 = sy.start, sy.stop
            x0,x1 = sx.start, sx.stop

        # Is part of this image bad?
        old_extent = (x0,x1,y0,y1)
        new_extent = self.get_good_image_slice((x0,x1,y0,y1), get_extent=True)
        if new_extent != old_extent:
            x0,x1,y0,y1 = new_extent
            print('Applying good subregion of CCD: slice is', x0,x1,y0,y1)
            if x0 >= x1 or y0 >= y1:
                return 0,0,0,0,None
            slc = slice(y0,y1), slice(x0,x1)
        return x0,x1,y0,y1,slc

    def remap_invvar(self, invvar, primhdr, img, dq):
        return invvar

    def check_image_header(self, imghdr):
        pass
    
    def psf_norm(self, tim, x=None, y=None):
        # PSF norm
        psf = tim.psf
        h,w = tim.shape
        if x is None:
            x = w/2.
        if y is None:
            y = h/2.
        patch = psf.getPointSourcePatch(x, y).patch

        #print('Before clamping: PSF range', patch.min(), patch.max())

        # Clamp up to zero and normalize before taking the norm
        patch = np.maximum(0, patch)
        patch /= patch.sum()
        psfnorm = np.sqrt(np.sum(patch**2))

        # import pylab as plt
        # plt.clf()
        # plt.imshow(patch, interpolation='nearest', origin='lower',
        #            vmin=0, vmax=0.06)
        # # zoom in on 15x15 center
        # h,w = patch.shape
        # plt.axis([w//2-7, w//2+7, h//2-7, h//2+7])
        # plt.colorbar()
        # 
        # plt.title('psfmod: %s expnum %i, band %s, norm %.3f' % (self.camera, self.expnum, tim.band, psfnorm))
        # psgalnorm.savefig()

        return psfnorm

    def galaxy_norm(self, tim, x=None, y=None):
        # Galaxy-detection norm

        #import tractor.galaxy
        #tractor.galaxy.debug_ps = psgalnorm

        from tractor.galaxy import ExpGalaxy
        from tractor.ellipses import EllipseE
        from tractor.patch import ModelMask
        h,w = tim.shape
        band = tim.band


        if x is None:
            x = w//2
        if y is None:
            y = h//2
        pos = tim.wcs.pixelToPosition(x, y)
        gal = SimpleGalaxy(pos, NanoMaggies(**{band:1.}))
        S = 32
        mm = ModelMask(int(x-S), int(y-S), 2*S+1, 2*S+1)
        galmod = gal.getModelPatch(tim, modelMask=mm).patch

        #orig_galmod = galmod.copy()

        galmod = np.maximum(0, galmod)
        galmod /= galmod.sum()
        galnorm = np.sqrt(np.sum(galmod**2))

        #  h,w = galmod.shape
        #  import pylab as plt
        #  # plt.clf()
        #  # plt.imshow(galmod, interpolation='nearest', origin='lower',
        #  #            vmin=0, vmax=0.06)
        #  # # zoom in on 15x15 center
        #  # plt.axis([w//2-7, w//2+7, h//2-7, h//2+7])
        #  # plt.colorbar()
        #  # plt.title('galmod: %s expnum %i, band %s, galnorm %.3f' % (self.camera, self.expnum, tim.band, galnorm))
        #  # psgalnorm.savefig()
        #  
        #  from tractor import PointSource
        #  
        #  print('galaxy_norm: getting PointSource patch')
        #  psf = PointSource(pos, NanoMaggies(**{band:1.}))
        #  psfmod = psf.getModelPatch(tim, modelMask=mm).patch
        #  
        #  orig_psfmod = psfmod.copy()
        #  
        #  print('orig galmod range:', orig_galmod.min(), orig_galmod.max())
        #  print('orig psfmod range:', orig_psfmod.min(), orig_psfmod.max())
        #  
        #  print('Orig psfmod sum:', orig_psfmod.sum())
        #  print('Orig galmod sum:', orig_galmod.sum())
        #  
        #  mn = min(np.min(orig_galmod), np.min(orig_psfmod))
        #  mx = max(np.max(orig_galmod), np.max(orig_psfmod))
        #  
        #  psfmod = np.maximum(0, psfmod)
        #  print('PSF sum after clamping up to zero:', psfmod.sum())
        #  psfmod /= psfmod.sum()
        #  psfnorm = np.sqrt(np.sum(psfmod**2))
        #  
        #  slc = (slice(h//2-7, h//2+8), slice(w//2-7, w//2+8))
        #  print('Norm of central galaxy slice:', np.sqrt(np.sum((galmod[slc] / galmod[slc].sum())**2)))
        #  print('Norm of central PSF slice:', np.sqrt(np.sum((psfmod[slc] / psfmod[slc].sum())**2)))
        #  
        #  from scipy.ndimage.filters import gaussian_filter
        #  psfconv = gaussian_filter(orig_psfmod, 1.2)
        #  psfconv = np.maximum(0, psfconv)
        #  print('PSF sum after clamping up to zero:', psfmod.sum())
        #  psfconv /= psfconv.sum()
        #  nm = np.sqrt(np.sum(psfconv**2))
        #  print('Norm of PSF convolved by Gaussian:', nm)
        #  
        #  plt.clf()
        #  plt.subplot(2,2,1)
        #  plt.imshow(orig_galmod, interpolation='nearest', origin='lower',
        #             vmin=mn, vmax=mx)
        #  # zoom in on 15x15 center
        #  plt.axis([w//2-7, w//2+7, h//2-7, h//2+7])
        #  plt.title('orig gal: norm %.3f, pk %.3f' % (galnorm, np.max(galmod)))
        #  plt.subplot(2,2,2)
        #  plt.imshow(orig_psfmod, interpolation='nearest', origin='lower',
        #             vmin=mn, vmax=mx)
        #  # zoom in on 15x15 center
        #  plt.axis([w//2-7, w//2+7, h//2-7, h//2+7])
        #  plt.title('orig psf: norm %.3f, pk %.3f' % (psfnorm, np.max(psfmod)))
        #  plt.subplot(2,2,3)
        #  diff = orig_galmod - orig_psfmod
        #  dmx = np.max(np.abs(diff))
        #  plt.imshow(diff, interpolation='nearest', origin='lower',
        #             vmin=-dmx, vmax=dmx)
        #  # zoom in on 15x15 center
        #  plt.axis([w//2-7, w//2+7, h//2-7, h//2+7])
        #  plt.title('galmod - psfmod')
        #  plt.suptitle('%s expnum %i, band %s' % (self.camera, self.expnum, tim.band))
        #  psgalnorm.savefig()
        #  
        #  
        #  
        #  plt.clf()
        #  plt.subplot(2,2,1)
        #  plt.imshow(orig_galmod, interpolation='nearest', origin='lower',
        #             vmin=mn, vmax=mx)
        #  plt.title('orig gal: norm %.3f, pk %.3f' % (galnorm, np.max(galmod)))
        #  plt.subplot(2,2,2)
        #  plt.imshow(orig_psfmod, interpolation='nearest', origin='lower',
        #             vmin=mn, vmax=mx)
        #  plt.title('orig psf: norm %.3f, pk %.3f' % (psfnorm, np.max(psfmod)))
        #  plt.subplot(2,2,3)
        #  diff = orig_galmod - orig_psfmod
        #  dmx = np.max(np.abs(diff))
        #  plt.imshow(diff, interpolation='nearest', origin='lower',
        #             vmin=-dmx, vmax=dmx)
        #  plt.title('galmod - psfmod')
        #  plt.suptitle('%s expnum %i, band %s' % (self.camera, self.expnum, tim.band))
        #  psgalnorm.savefig()
        #  
        #  
        #  mx = max(np.max(galmod), np.max(psfmod))
        #  
        #  
        #  plt.clf()
        #  plt.subplot(2,2,1)
        #  plt.imshow(galmod, interpolation='nearest', origin='lower',
        #             vmin=mn, vmax=mx)
        #  # zoom in on 15x15 center
        #  plt.axis([w//2-7, w//2+7, h//2-7, h//2+7])
        #  plt.title('gal: norm %.3f, pk %.3f' % (galnorm, np.max(galmod)))
        #  
        #  plt.subplot(2,2,2)
        #  plt.imshow(psfmod, interpolation='nearest', origin='lower',
        #             vmin=mn, vmax=mx)
        #  # zoom in on 15x15 center
        #  plt.axis([w//2-7, w//2+7, h//2-7, h//2+7])
        #  plt.title('psf: norm %.3f, pk %.3f' % (psfnorm, np.max(psfmod)))
        #  
        #  plt.subplot(2,2,3)
        #  diff = galmod - psfmod
        #  mx = np.max(np.abs(diff))
        #  plt.imshow(diff, interpolation='nearest', origin='lower',
        #             vmin=-mx, vmax=mx)
        #  #plt.colorbar()
        #  # zoom in on 15x15 center
        #  plt.axis([w//2-7, w//2+7, h//2-7, h//2+7])
        #  plt.title('galmod - psfmod')
        #  
        #  plt.suptitle('%s expnum %i, band %s' % (self.camera, self.expnum, tim.band))
        #  
        #  psgalnorm.savefig()
        #  
        #  
        #  plt.clf()
        #  plt.subplot(2,2,1)
        #  plt.imshow(galmod, interpolation='nearest', origin='lower',
        #             vmin=mn, vmax=mx)
        #  plt.title('gal: norm %.3f, pk %.3f' % (galnorm, np.max(galmod)))
        #  plt.subplot(2,2,2)
        #  plt.imshow(psfmod, interpolation='nearest', origin='lower',
        #             vmin=mn, vmax=mx)
        #  plt.title('psf: norm %.3f, pk %.3f' % (psfnorm, np.max(psfmod)))
        #  plt.subplot(2,2,3)
        #  diff = galmod - psfmod
        #  mx = np.max(np.abs(diff))
        #  plt.imshow(diff, interpolation='nearest', origin='lower',
        #             vmin=-mx, vmax=mx)
        #  plt.title('galmod - psfmod')
        #  plt.suptitle('%s expnum %i, band %s' % (self.camera, self.expnum, tim.band))
        #  psgalnorm.savefig()
        #  
        #  plt.clf()
        #  import photutils
        #  apxy = np.array([w//2, h//2])
        #  galap = []
        #  psfap = []
        #  ogalap = []
        #  opsfap = []
        #  rads = np.arange(1, w//2)
        #  pnorm = []
        #  gnorm = []
        #  for rad in rads:
        #      aper = photutils.CircularAperture(apxy, rad)
        #      p = photutils.aperture_photometry(galmod, aper)
        #      galap.append(p.field('aperture_sum')[0])
        #      p = photutils.aperture_photometry(psfmod, aper)
        #      psfap.append(p.field('aperture_sum')[0])
        #      p = photutils.aperture_photometry(orig_galmod, aper)
        #      ogalap.append(p.field('aperture_sum')[0])
        #      p = photutils.aperture_photometry(orig_psfmod, aper)
        #      opsfap.append(p.field('aperture_sum')[0])
        #  
        #      subimg = galmod[h//2-rad : h//2+rad+1, w//2-rad : w//2+rad+1]
        #      subimg = subimg / subimg.sum()
        #      gnorm.append(np.sqrt(np.sum(subimg**2)))
        #      subimg = psfmod[h//2-rad : h//2+rad+1, w//2-rad : w//2+rad+1]
        #      subimg = subimg / subimg.sum()
        #      pnorm.append(np.sqrt(np.sum(subimg**2)))
        #  
        #  
        #  print('rads', rads, 'galap', galap)
        #  plt.subplot(2,1,1)
        #  plt.plot(rads, galap, 'r-', label='Galaxy')
        #  plt.plot(rads, ogalap, 'm-', label='Orig galaxy')
        #  plt.plot(rads, psfap, 'b-', label='PSF')
        #  plt.plot(rads, opsfap, 'c-', label='Orig PSF')
        #  plt.legend(loc='lower right')
        #  plt.title('%s expnum %i, band %s' % (self.camera, self.expnum, tim.band))
        #  plt.xlabel('aperture (pix)')
        #  plt.ylabel('Aperture Flux')
        #  
        #  plt.subplot(2,1,2)
        #  plt.plot(rads, pnorm, 'b-', label='PSF')
        #  plt.plot(rads, gnorm, 'r-', label='Galaxy')
        #  plt.legend(loc='upper right')
        #  plt.xlabel('aperture (pix)')
        #  plt.ylabel('Norm')
        #  
        #  psgalnorm.savefig()
        #  
        #  
        #  
        #  #ima = dict(interpolation='nearest', origin='lower', vmin=-0.001*mx,
        #  #           vmax=0.001*mx, cmap='RdBu')
        #  # ima = dict(interpolation='nearest', origin='lower', vmin=-1,
        #  #            vmax=1, cmap='RdBu')
        #  # 
        #  # plt.clf()
        #  # plt.subplot(1,2,1)
        #  # plt.imshow(np.sign(orig_galmod), **ima)
        #  # plt.title('gal: norm %.3f, pk %.3f' % (galnorm, np.max(galmod)))
        #  # 
        #  # plt.subplot(1,2,2)
        #  # plt.imshow(np.sign(orig_psfmod), **ima)
        #  # plt.title('psf: norm %.3f, pk %.3f' % (psfnorm, np.max(psfmod)))
        #  # 
        #  # plt.suptitle('%s expnum %i, band %s: sign' % (self.camera, self.expnum, tim.band))
        #  # 
        #  # psgalnorm.savefig()
        #  
        #  
        #  mx = max(np.max(orig_galmod), np.max(orig_psfmod))
        #  ima = dict(interpolation='nearest', origin='lower',
        #             vmin=-6 + np.log10(mx),
        #             vmax=np.log10(mx))
        #  
        #  plt.clf()
        #  plt.subplot(1,2,1)
        #  plt.imshow(np.log10(orig_galmod), **ima)
        #  plt.title('gal: norm %.3f, pk %.3f' % (galnorm, np.max(galmod)))
        #  
        #  plt.subplot(1,2,2)
        #  plt.imshow(np.log10(orig_psfmod), **ima)
        #  plt.title('psf: norm %.3f, pk %.3f' % (psfnorm, np.max(psfmod)))
        #  
        #  plt.suptitle('%s expnum %i, band %s' % (self.camera, self.expnum, tim.band))
        #  
        #  psgalnorm.savefig()



        return galnorm
    
    def _read_fits(self, fn, hdu, slice=None, header=None, **kwargs):
        if slice is not None:
            f = fitsio.FITS(fn)[hdu]
            img = f[slice]
            rtn = img
            if header:
                hdr = f.read_header()
                return (img,hdr)
            return img
        return fitsio.read(fn, ext=hdu, header=header, **kwargs)

    def read_image(self, **kwargs):
        '''
        Reads the image file from disk.

        The image is read from FITS file self.imgfn HDU self.hdu.

        Parameters
        ----------
        slice : slice, optional
            2-dimensional slice of the subimage to read.
        header : boolean, optional
            Return the image header also, as tuple (image, header) ?

        Returns
        -------
        image : numpy array
            The image pixels.
        (image, header) : (numpy array, fitsio header)
            If `header = True`.
        '''
        print('Reading image from', self.imgfn, 'hdu', self.hdu)
        return self._read_fits(self.imgfn, self.hdu, **kwargs)

    def get_image_info(self):
        '''
        Reads the FITS image header and returns some summary information
        as a dictionary (image size, type, etc).
        '''
        return fitsio.FITS(self.imgfn)[self.hdu].get_info()

    def get_image_shape(self):
        '''
        Returns image shape H,W.
        '''
        return self.height, self.width

    @property
    def shape(self):
        '''
        Returns the full shape of the image, (H,W).
        '''
        return self.get_image_shape()

    def read_image_primary_header(self, **kwargs):
        '''
        Reads the FITS primary (HDU 0) header from self.imgfn.

        Returns
        -------
        primary_header : fitsio header
            The FITS header
        '''
        return self.read_primary_header(self.imgfn)

    def read_primary_header(self, fn):
        '''
        Reads the FITS primary header (HDU 0) from the given filename.
        This is just a faster version of fitsio.read_header(fn).
        '''
        if fn.endswith('.gz'):
            return fitsio.read_header(self.fn)

        # Weirdly, this can be MUCH faster than letting fitsio do it...
        hdr = fitsio.FITSHDR()
        foundEnd = False
        ff = open(fn, 'rb')
        h = b''
        while True:
            h = h + ff.read(32768)
            while True:
                line = h[:80]
                h = h[80:]
                #print('Header line "%s"' % line)
                # HACK -- fitsio apparently can't handle CONTINUE.
                # It also has issues with slightly malformed cards, like
                # KEYWORD  =      / no value
                if line[:8] != b'CONTINUE':
                    try:
                        hdr.add_record(line.decode())
                    except:
                        print('Warning: failed to parse FITS header line: ' +
                              ('"%s"; skipped' % line.strip()))
                        import traceback
                        traceback.print_exc()
                              
                if line == (b'END' + b' '*77):
                    foundEnd = True
                    break
                if len(h) < 80:
                    break
            if foundEnd:
                break
        ff.close()
        return hdr

    def read_image_header(self, **kwargs):
        '''
        Reads the FITS image header from self.imgfn HDU self.hdu.

        Returns
        -------
        header : fitsio header
            The FITS header
        '''
        return fitsio.read_header(self.imgfn, ext=self.hdu)

    def read_dq(self, **kwargs):
        '''
        Reads the Data Quality (DQ) mask image.
        '''
        return None

    def read_invvar(self, clip=True, **kwargs):
        '''
        Reads the inverse-variance (weight) map image.
        '''
        return None

    def get_tractor_wcs(self, wcs, x0, y0,
                        primhdr=None, imghdr=None):
        twcs = ConstantFitsWcs(wcs)
        if x0 or y0:
            twcs.setX0Y0(x0,y0)
        return twcs

    def get_wcs(self):
        return None

    ### Yuck, this is not much better than just doing read_sky_model().sig1 ...
    def get_sky_sig1(self, splinesky=False):
        '''
        Returns the per-pixel noise estimate, which (for historical
        reasons) is stored in the sky model.  NOTE that this is in
        image pixel counts, NOT calibrated nanomaggies.
        '''
        if splinesky and getattr(self, 'merged_splineskyfn', None) is not None:
            if not os.path.exists(self.merged_splineskyfn):
                print('Merged spline sky model does not exist:', self.merged_splineskyfn)
        if (splinesky and getattr(self, 'merged_splineskyfn', None) is not None
            and os.path.exists(self.merged_splineskyfn)):
            try:
                print('Reading merged spline sky models from', self.merged_splineskyfn)
                T = fits_table(self.merged_splineskyfn)
                if 'sig1' in T.get_columns():
                    I, = np.nonzero((T.expnum == self.expnum) *
                                    np.array([c.strip() == self.ccdname
                                              for c in T.ccdname]))
                    print('Found', len(I), 'matching CCD')
                    if len(I) >= 1:
                        return T.sig1[I[0]]
            except:
                pass
        if splinesky:
            fn = self.splineskyfn
        else:
            fn = self.skyfn
        print('Reading sky model from', fn)
        hdr = fitsio.read_header(fn)
        sig1 = hdr.get('SIG1', None)
        return sig1

    def read_sky_model(self, splinesky=False, slc=None, **kwargs):
        '''
        Reads the sky model, returning a Tractor Sky object.
        '''
        sky = None
        if splinesky and getattr(self, 'merged_splineskyfn', None) is not None:
            if not os.path.exists(self.merged_splineskyfn):
                print('Merged spline sky model does not exist:', self.merged_splineskyfn)

        if (splinesky and getattr(self, 'merged_splineskyfn', None) is not None
            and os.path.exists(self.merged_splineskyfn)):
            try:
                print('Reading merged spline sky models from', self.merged_splineskyfn)
                T = fits_table(self.merged_splineskyfn)
                I, = np.nonzero((T.expnum == self.expnum) *
                                np.array([c.strip() == self.ccdname
                                          for c in T.ccdname]))
                print('Found', len(I), 'matching CCD')
                if len(I) == 1:
                    Ti = T[I[0]]
                    # Ti.about()
                    # print('Spline w,h', Ti.gridw, Ti.gridh)
                    # print('xgrid:', Ti.xgrid.shape)
                    # print('ygrid:', Ti.ygrid.shape)
                    # print('gridvals:', Ti.gridvals.shape)

                    # Remove any padding
                    h,w = Ti.gridh, Ti.gridw
                    Ti.gridvals = Ti.gridvals[:h, :w]
                    Ti.xgrid = Ti.xgrid[:w]
                    Ti.ygrid = Ti.ygrid[:h]

                    skyclass = Ti.skyclass.strip()
                    clazz = get_class_from_name(skyclass)
                    fromfits = getattr(clazz, 'from_fits_row')
                    sky = fromfits(Ti)
                    
                    if slc is not None:
                        sy,sx = slc
                        x0,y0 = sx.start,sy.start
                        sky.shift(x0, y0)
                    sky.version = Ti.legpipev
                    sky.plver = Ti.plver
                    if 'sig1' in Ti.get_columns():
                        sky.sig1 = Ti.sig1
                    return sky
            except:
                import traceback
                traceback.print_exc()

        fn = self.skyfn
        if splinesky:
            fn = self.splineskyfn
        print('Reading sky model from', fn)
        hdr = fitsio.read_header(fn)
        try:
            skyclass = hdr['SKY']
        except NameError:
            raise NameError('SKY not in header: skyfn=%s, imgfn=%s' % (fn,self.imgfn))
        clazz = get_class_from_name(skyclass)

        if getattr(clazz, 'from_fits', None) is not None:
            fromfits = getattr(clazz, 'from_fits')
            sky = fromfits(fn, hdr)
        else:
            fromfits = getattr(clazz, 'fromFitsHeader')
            sky = fromfits(hdr, prefix='SKY_')

        if slc is not None:
            sy,sx = slc
            x0,y0 = sx.start,sy.start
            sky.shift(x0, y0)

        sky.version = hdr.get('LEGPIPEV', '')
        if len(sky.version) == 0:
            sky.version = hdr.get('TRACTORV', '').strip()
            if len(sky.version) == 0:
                sky.version = str(os.stat(fn).st_mtime)
        sky.plver = hdr.get('PLVER', '').strip()
        sig1 = hdr.get('SIG1', None)
        if sig1 is not None:
            sky.sig1 = sig1
        return sky

    def read_psf_model(self, x0, y0,
                       gaussPsf=False, pixPsf=False, hybridPsf=False,
                       psf_sigma=1., w=0, h=0):
        assert(gaussPsf or pixPsf or hybridPsf)
        psffn = None
        if gaussPsf:
            from tractor import GaussianMixturePSF
            v = psf_sigma**2
            psf = GaussianMixturePSF(1., 0., 0., v, v, 0.)
            print('WARNING: using mock PSF:', psf)
            psf.version = '0'
            psf.plver = ''
            return psf

        # spatially varying pixelized PsfEx
        from tractor import PixelizedPsfEx, PsfExModel
        psf = None
        if getattr(self, 'merged_psffn', None) is not None:
            if not os.path.exists(self.merged_psffn):
                print('Merged PsfEx model does not exist:', self.merged_psffn)

        if (getattr(self, 'merged_psffn', None) is not None
            and os.path.exists(self.merged_psffn)):
            try:
                print('Reading merged PsfEx models from', self.merged_psffn)
                T = fits_table(self.merged_psffn)
                I, = np.nonzero((T.expnum == self.expnum) *
                                np.array([c.strip() == self.ccdname
                                          for c in T.ccdname]))
                print('Found', len(I), 'matching CCD')
                if len(I) == 1:
                    Ti = T[I[0]]

                    # Remove any padding
                    degree = Ti.poldeg1
                    # number of terms in polynomial
                    ne = (degree + 1) * (degree + 2) / 2
                    #print('PSF_mask shape', Ti.psf_mask.shape)
                    Ti.psf_mask = Ti.psf_mask[:ne, :Ti.psfaxis1, :Ti.psfaxis2]

                    psfex = PsfExModel(Ti=Ti)
                    psf = PixelizedPsfEx(None, psfex=psfex)
                    psf.version = Ti.legpipev.strip()
                    psf.plver = Ti.plver.strip()
            except:
                import traceback
                traceback.print_exc()

        if psf is None:
            print('Reading PsfEx model from', self.psffn)
            psf = PixelizedPsfEx(self.psffn)

            hdr = fitsio.read_header(self.psffn)
            psf.version = hdr.get('LEGSURV', None)
            if psf.version is None:
                psf.version = str(os.stat(self.psffn).st_mtime)
            psf.plver = hdr.get('PLVER', '').strip()

        psf.shift(x0, y0)
        if hybridPsf:
            from tractor.psf import HybridPixelizedPSF
            psf = HybridPixelizedPSF(psf, cx=w/2., cy=h/2.)

        print('Using PSF model', psf)
        return psf

    def run_calibs(self, **kwargs):
        '''
        Runs any required calibration processes for this image.
        '''
        print('run_calibs for', self)
        print('(not implemented)')
        pass

class CalibMixin(object):
    '''
    A class to hold common calibration tasks between the different
    surveys / image subclasses.
    '''

    def __init__(self):
        super(CalibMixin, self).__init__()
    
    def check_psf(self, psffn):
        '''
        Returns True if the PsfEx file is ok.
        '''
        # Sometimes SourceExtractor gets interrupted or something and
        # writes out 0 detections.  Then PsfEx fails but in a way that
        # an output file is still written.  Try to detect & fix this
        # case.
        # Check the PsfEx output file for POLNAME1
        try:
            hdr = fitsio.read_header(psffn, ext=1)
        except:
            print('Failed to read header from existing PSF model file', psffn)
            return False
        if hdr.get('POLNAME1', None) is None:
            print('Did not find POLNAME1 in PsfEx header',psffn,'-- deleting')
            os.unlink(psffn)
            return False
        return True

    def check_se_cat(self, fn):
        from astrometry.util.fits import fits_table
        from astrometry.util.file import file_size
        # Check SourceExtractor catalog for file size = 0 or FITS table
        # length = 0
        if os.path.exists(fn) and file_size(fn) == 0:
            try:
                os.unlink(fn)
            except:
                pass
            return False
        T = fits_table(fn, hdu=2)
        print('Read', len(T), 'sources from SE catalog', fn)
        if T is None or len(T) == 0:
            print('SourceExtractor catalog', fn, 'has no sources -- deleting')
            try:
                os.unlink(fn)
            except:
                pass
        return os.path.exists(fn)

    def funpack_files(self, imgfn, maskfn, hdu, todelete):
        from legacypipe.survey import create_temp

        tmpimgfn = None
        tmpmaskfn = None
        # For FITS files that are not actually fpack'ed, funpack -E
        # fails.  Check whether actually fpacked.
        fcopy = False
        hdr = fitsio.read_header(imgfn, ext=hdu)
        if not ((hdr['XTENSION'] == 'BINTABLE') and hdr.get('ZIMAGE', False)):
            print('Image %s, HDU %i is not fpacked; just imcopying.' %
                  (imgfn,  hdu))
            fcopy = True

        tmpimgfn  = create_temp(suffix='.fits')
        tmpmaskfn = create_temp(suffix='.fits')
        todelete.append(tmpimgfn)
        todelete.append(tmpmaskfn)
        
        if fcopy:
            cmd = 'imcopy %s"+%i" %s' % (imgfn, hdu, tmpimgfn)
        else:
            cmd = 'funpack -E %i -O %s %s' % (hdu, tmpimgfn, imgfn)
        print(cmd)
        if os.system(cmd):
            raise RuntimeError('Command failed: ' + cmd)
        
        if fcopy:
            cmd = 'imcopy %s"+%i" %s' % (maskfn, hdu, tmpmaskfn)
        else:
            cmd = 'funpack -E %i -O %s %s' % (hdu, tmpmaskfn, maskfn)
        print(cmd)
        if os.system(cmd):
            print('Command failed: ' + cmd)
            M,hdr = self._read_fits(maskfn, hdu, header=True)
            print('Read', M.dtype, M.shape)
            fitsio.write(tmpmaskfn, M, header=hdr, clobber=True)
            print('Wrote', tmpmaskfn, 'with fitsio')

        return tmpimgfn,tmpmaskfn

    def run_se(self, surveyname, imgfn, maskfn):
        from astrometry.util.file import trymakedirs
        # grab header values...
        primhdr = self.read_image_primary_header()
        try:
            magzp  = float(primhdr['MAGZERO'])
        except:
            magzp = 25.
        seeing = self.pixscale * self.fwhm
        print('FWHM', self.fwhm, 'pix')
        print('pixscale', self.pixscale, 'arcsec/pix')
        print('Seeing', seeing, 'arcsec')
        print('magzp', magzp)
        
        sedir = self.survey.get_se_dir()
        trymakedirs(self.sefn, dir=True)

        # We write the SE catalog to a temp file then rename, to avoid
        # partially-written outputs.
        tmpfn = os.path.join(os.path.dirname(self.sefn),
                             'tmp-' + os.path.basename(self.sefn))
        cmd = ' '.join([
            'sex',
            '-c', os.path.join(sedir, surveyname + '.se'),
            '-PARAMETERS_NAME', os.path.join(sedir, surveyname + '.param'),
            '-FILTER_NAME %s' % os.path.join(sedir, surveyname + '.conv'),
            '-FLAG_IMAGE %s' % maskfn,
            '-CATALOG_NAME %s' % tmpfn,
            '-SEEING_FWHM %f' % seeing,
            '-MAG_ZEROPOINT %f' % magzp,
            imgfn])
        print(cmd)
        rtn = os.system(cmd)
        if rtn:
            raise RuntimeError('Command failed: ' + cmd)
        os.rename(tmpfn, self.sefn)

    def run_psfex(self, surveyname):
        from astrometry.util.file import trymakedirs
        from legacypipe.survey import get_git_version
        sedir = self.survey.get_se_dir()
        trymakedirs(self.psffn, dir=True)
        primhdr = self.read_image_primary_header()
        plver = primhdr.get('PLVER', '')
        verstr = get_git_version()
        # We write the PSF model to a .fits.tmp file, then rename to .fits
        psfdir = os.path.dirname(self.psffn)
        psfoutfn = os.path.join(psfdir, os.path.basename(self.sefn).replace('.fits','') + '.fits')
        cmds = ['psfex -c %s -PSF_DIR %s -PSF_SUFFIX .fits.tmp %s' %
                (os.path.join(sedir, surveyname + '.psfex'),
                 psfdir, self.sefn),
                'mv %s %s' % (psfoutfn + '.tmp', psfoutfn),
                'modhead %s LEGPIPEV %s "legacypipe git version"' %
                (self.psffn, verstr),
                'modhead %s PLVER %s "CP ver of image file"' % (self.psffn, plver)]
        for cmd in cmds:
            print(cmd)
            rtn = os.system(cmd)
            if rtn:
                raise RuntimeError('Command failed: %s: return value: %i' %
                                   (cmd,rtn))

    def run_sky(self, surveyname, splinesky=False, git_version=None):
        from legacypipe.survey import get_version_header

        slc = self.get_good_image_slice(None)
        img = self.read_image(slice=slc)
        wt = self.read_invvar(slice=slc)
        hdr = get_version_header(None, self.survey.get_survey_dir(),
                                 git_version=git_version)
        primhdr = self.read_image_primary_header()
        plver = primhdr.get('PLVER', '')
        hdr.delete('PROCTYPE')
        hdr.add_record(dict(name='PROCTYPE', value='ccd',
                            comment='NOAO processing type'))
        hdr.add_record(dict(name='PRODTYPE', value='skymodel',
                            comment='NOAO product type'))
        hdr.add_record(dict(name='PLVER', value=plver,
                            comment='CP ver of image file'))

        if splinesky:
            from tractor.splinesky import SplineSky
            from scipy.ndimage.morphology import binary_dilation

            boxsize = self.splinesky_boxsize
            
            # Start by subtracting the overall median
            good = (wt > 0)
            if np.sum(good) == 0:
                raise RuntimeError('No pixels with weight > 0 in: ' + str(self))
            med = np.median(img[good])
            # Compute initial model...
            skyobj = SplineSky.BlantonMethod(img - med, good, boxsize)
            skymod = np.zeros_like(img)
            skyobj.addTo(skymod)
            # Now mask bright objects in (image - initial sky model)
            sig1 = 1./np.sqrt(np.median(wt[good]))
            masked = (img - med - skymod) > (5.*sig1)
            masked = binary_dilation(masked, iterations=3)
            masked[wt == 0] = True

            sig1b = 1./np.sqrt(np.median(wt[masked == False]))
            print('Sig1 vs sig1b:', sig1, sig1b)

            # Now find the final sky model using that more extensive mask
            skyobj = SplineSky.BlantonMethod(
                img - med, np.logical_not(masked), boxsize)
            # add the overall median back in
            skyobj.offset(med)

            if slc is not None:
                sy,sx = slc
                y0 = sy.start
                x0 = sx.start
                skyobj.shift(-x0, -y0)

            hdr.add_record(dict(name='SIG1', value=sig1,
                                comment='Median stdev of unmasked pixels'))
            hdr.add_record(dict(name='SIG1B', value=sig1,
                                comment='Median stdev of unmasked pixels+'))
                
            trymakedirs(self.splineskyfn, dir=True)
            skyobj.write_fits(self.splineskyfn, primhdr=hdr)
            print('Wrote sky model', self.splineskyfn)

        else:
            try:
                skyval = estimate_mode(img[wt > 0], raiseOnWarn=True)
                skymeth = 'mode'
            except:
                skyval = np.median(img[wt > 0])
                skymeth = 'median'
            tsky = ConstantSky(skyval)

            hdr.add_record(dict(name='SKYMETH', value=skymeth,
                                comment='estimate_mode, or fallback to median?'))

            sig1 = 1./np.sqrt(np.median(wt[wt>0]))
            masked = (img - skyval) > (5.*sig1)
            masked = binary_dilation(masked, iterations=3)
            masked[wt == 0] = True
            sig1b = 1./np.sqrt(np.median(wt[masked == False]))
            print('Sig1 vs sig1b:', sig1, sig1b)

            hdr.add_record(dict(name='SIG1', value=sig1,
                                comment='Median stdev of unmasked pixels'))
            hdr.add_record(dict(name='SIG1B', value=sig1,
                                comment='Median stdev of unmasked pixels+'))
            
            trymakedirs(self.skyfn, dir=True)
            tsky.write_fits(self.skyfn, hdr=hdr)
            print('Wrote sky model', self.skyfn)
