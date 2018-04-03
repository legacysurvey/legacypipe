from __future__ import print_function
import os
import numpy as np
import fitsio
from astrometry.util.fits import fits_table

'''
Base class for handling the images we process.  These are all
processed by variants of the NOAO Community Pipeline (CP), so this
base class is pretty specific.
'''

# From: http://www.noao.edu/noao/staff/fvaldes/CPDocPrelim/PL201_3.html
# 1   -- detector bad pixel           InstCal
# 1   -- detector bad pixel/no data   Resampled
# 1   -- No data                      Stacked
# 2   -- saturated                    InstCal/Resampled
# 4   -- interpolated                 InstCal/Resampled
# 16  -- single exposure cosmic ray   InstCal/Resampled
# 64  -- bleed trail                  InstCal/Resampled
# 128 -- multi-exposure transient     InstCal/Resampled 
CP_DQ_BITS = dict(badpix=1, satur=2, interp=4, cr=16, bleed=64,
                  trans=128,
                  edge = 256,
                  edge2 = 512,
                  ## masked by stage_mask_junk
                  longthin = 1024,
                  )



class LegacySurveyImage(object):
    '''A base class containing common code for the images we handle.

    You probably shouldn't need to directly instantiate this class,
    but rather use the recipe described in the __init__ method.

    Objects of this class represent the metadata we have on an image,
    and are used to handle some of the details of going from an entry
    in the CCDs table to a tractor Image object.

    '''

    # this is defined here for testing purposes (to handle the small
    # images used in unit tests): box size for SplineSky model
    splinesky_boxsize = 1024


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
        super(LegacySurveyImage, self).__init__()
        self.survey = survey

        imgfn = ccd.image_filename.strip()

        self.imgfn = os.path.join(self.survey.get_image_dir(), imgfn)
        self.compute_filenames()

        self.hdu     = ccd.image_hdu
        self.expnum  = ccd.expnum
        self.ccdname = ccd.ccdname.strip()
        self.band    = ccd.filter.strip()
        self.exptime = ccd.exptime
        self.camera  = ccd.camera.strip()
        self.fwhm    = ccd.fwhm
        self.propid  = ccd.propid
        self.mjdobs = ccd.mjd_obs
        self.width  = ccd.width
        self.height = ccd.height

        self.sig1 = getattr(ccd, 'sig1', None)

        # Which Data Quality bits mark saturation?
        self.dq_saturation_bits = CP_DQ_BITS['satur']

        # Photometric and astrometric zeropoints
        self.ccdzpt = ccd.ccdzpt
        self.dradec = (ccd.ccdraoff / 3600., ccd.ccddecoff / 3600.)
        
        # in arcsec/pixel
        self.pixscale = 3600. * np.sqrt(np.abs(ccd.cd1_1 * ccd.cd2_2 -
                                               ccd.cd1_2 * ccd.cd2_1))

        expstr = '%08i' % self.expnum
        self.name = '%s-%s-%s' % (self.camera, expstr, self.ccdname)
        calname = '%s/%s/%s-%s-%s' % (expstr[:5], expstr, self.camera, 
                                      expstr, self.ccdname)
        calibdir = os.path.join(self.survey.get_calib_dir(), self.camera)
        self.sefn  = os.path.join(calibdir, 'se',    calname + '.fits')
        self.psffn = os.path.join(calibdir, 'psfex', calname + '.fits')
        self.skyfn = os.path.join(calibdir, 'sky',   calname + '.fits')
        self.splineskyfn = os.path.join(calibdir, 'splinesky', calname + '.fits')
        self.merged_psffn = os.path.join(calibdir, 'psfex-merged', expstr[:5],
                                         '%s-%s.fits' % (self.camera, expstr))
        self.merged_splineskyfn = os.path.join(calibdir, 'splinesky-merged', expstr[:5],
                                               '%s-%s.fits' % (self.camera, expstr))

    def compute_filenames(self):
        # Compute data quality and weight-map filenames
        self.dqfn = self.imgfn.replace('_ooi_', '_ood_').replace('_oki_','_ood_')
        self.wtfn = self.imgfn.replace('_ooi_', '_oow_').replace('_oki_','_oow_')
        assert(self.dqfn != self.imgfn)
        assert(self.wtfn != self.imgfn)

        for attr in ['imgfn', 'dqfn', 'wtfn']:
            fn = getattr(self, attr)
            if os.path.exists(fn):
                continue
            if fn.endswith('.fz'):
                fun = fn[:-3]
                if os.path.exists(fun):
                    print('Using      ', fun)
                    print('rather than', fn)
                    setattr(self, attr, fun)
                    fn = fun

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def check_for_cached_files(self, survey):
        for key in self.get_cacheable_filename_variables():
            fn = getattr(self, key, None)
            if fn is None:
                continue
            cfn = survey.check_cache(fn)
            print('Checking for cached', key, ':', fn, '->', cfn)
            if cfn != fn:
                setattr(self, key, cfn)

    def get_cacheable_filename_variables(self):
        '''
        These are names of self.X variables that are filenames that
        could be cached.
        '''
        return ['imgfn', 'dqfn', 'wtfn', 'psffn', 'merged_psffn',
                'merged_splineskyfn', 'splineskyfn', 'skyfn']

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
                          normalizePsf=False,
                          splinesky=False,
                          apodize=False,
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
        import astropy.time
        from tractor.tractortime import TAITime
        from tractor.image import Image
        from tractor.basics import NanoMaggies, LinearPhotoCal

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
            invvar = self.read_invvar(slice=slc)
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
            dq,dqhdr = self.read_dq(slice=slc, header=True)
            if dq is not None:
                dq = self.remap_dq(dq, dqhdr)
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
            from tractor.sky import ConstantSky
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
            invvar *= (1. / sig1**2)
        assert(np.isfinite(sig1))

        if constant_invvar:
            print('Setting constant invvar', 1./sig1**2)
            invvar[invvar > 0] = 1./sig1**2

        if apodize and slc is not None:
            sy,sx = slc
            y0,y1 = sy.start, sy.stop
            x0,x1 = sx.start, sx.stop
            H,W = invvar.shape
            # Compute apodization ramps -- separately for x and y to
            # handle narrow images
            xx = np.linspace(-np.pi, np.pi, min(W,100))
            rampx = np.arctan(xx)
            rampx = (rampx - rampx.min()) / (rampx.max() - rampx.min())
            xx = np.linspace(-np.pi, np.pi, min(H,100))
            rampy = np.arctan(xx)
            rampy = (rampy - rampy.min()) / (rampy.max() - rampy.min())

            apo = False
            #if y0 == 0:
            if True:
                #print('Apodize bottom')
                invvar[:len(rampy),:] *= rampy[:,np.newaxis]
                apo = True
            #if x0 == 0:
            if True:
                #print('Apodize left')
                invvar[:,:len(rampx)] *= rampx[np.newaxis,:]
                apo = True
            #if y1 >= H:
            if True:
                #print('Apodize top')
                invvar[-len(rampy):,:] *= rampy[::-1][:,np.newaxis]
                apo = True
            #if x1 >= W:
            if True:
                #print('Apodize right')
                invvar[:,-len(rampx):] *= rampx[::-1][np.newaxis,:]
                apo = True

            if apo and False:
                import pylab as plt
                plt.clf()
                plt.imshow(invvar, interpolation='nearest', origin='lower')
                plt.savefig('apodized-%i-%s.png' % (self.expnum, self.ccdname))

        if subsky:
            # Warn if the subtracted sky doesn't seem to work well
            # (can happen, eg, if sky calibration product is inconsistent with
            #  the data)
            imgmed = np.median(img[invvar>0])
            if np.abs(imgmed) > sig1:
                print('WARNING: image median', imgmed, 'is more than 1 sigma',
                      'away from zero!')

        # Convert MJD-OBS, in UTC, into TAI
        mjd_tai = astropy.time.Time(self.mjdobs, format='mjd', scale='utc').tai.mjd
        tai = TAITime(None, mjd=mjd_tai)

        # tractor WCS object
        twcs = self.get_tractor_wcs(wcs, x0, y0, primhdr=primhdr, imghdr=imghdr,
                                    tai=tai)
                                    
        psf = self.read_psf_model(x0, y0, gaussPsf=gaussPsf, pixPsf=pixPsf,
                                  hybridPsf=hybridPsf, normalizePsf=normalizePsf,
                                  psf_sigma=psf_sigma,
                                  w=x1 - x0, h=y1 - y0)

        tim = Image(img, invvar=invvar, wcs=twcs, psf=psf,
                    photocal=LinearPhotoCal(zpscale, band=band),
                    sky=sky, name=self.name + ' ' + band)
        assert(np.all(np.isfinite(tim.getInvError())))
        tim.band = band

        # HACK -- create a local PSF model to instantiate the PsfEx
        # model, which handles non-unit pixel scaling.
        fullpsf = tim.psf
        th,tw = tim.shape
        tim.psf = fullpsf.constantPsfAt(tw//2, th//2)
        tim.psfnorm = self.psf_norm(tim)
        # Galaxy-detection norm
        tim.galnorm = self.galaxy_norm(tim)
        tim.psf = fullpsf

        tim.time = tai
        tim.slice = slc
        tim.zpscale = orig_zpscale
        tim.midsky = midsky
        tim.sig1 = sig1
        tim.psf_fwhm = psf_fwhm
        tim.psf_sigma = psf_sigma
        tim.propid = self.propid
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

    # A function that can be called by a subclasser's remap_invvar() method,
    # if desired, to include the contribution from the source Poisson fluctuations
    def remap_invvar_shotnoise(self, invvar, primhdr, img, dq):
        print('Remapping weight map for', self.name)
        const_sky = primhdr['SKYADU'] # e/s, Recommended sky level keyword from Frank 
        expt = primhdr['EXPTIME'] # s
        with np.errstate(divide='ignore'):
            var_SR = 1./invvar # e/s 
        var_Astro = np.abs(img - const_sky) / expt # e/s 
        wt = 1./(var_SR + var_Astro) # s/e

        # Zero out NaNs and masked pixels 
        wt[np.isfinite(wt) == False] = 0.
        wt[dq != 0] = 0.

        return wt

    def check_image_header(self, imghdr):
        # check consistency between the CCDs table and the image header
        e = imghdr['EXTNAME']
        if e.strip() != self.ccdname.strip():
            print('WARNING: Expected header EXTNAME="%s" to match self.ccdname="%s", self.imgfn=%s' % (e.strip(), self.ccdname,self.imgfn))

    def psf_norm(self, tim, x=None, y=None):
        # PSF norm
        psf = tim.psf
        h,w = tim.shape
        if x is None:
            x = w//2
        if y is None:
            y = h//2
        patch = psf.getPointSourcePatch(x, y).patch
        # Clamp up to zero and normalize before taking the norm
        # (decided that this is a poor idea - eg PSF normalization vs zeropoint)
        #patch = np.maximum(0, patch)
        #patch /= patch.sum()
        psfnorm = np.sqrt(np.sum(patch**2))
        return psfnorm

    def galaxy_norm(self, tim, x=None, y=None):
        # Galaxy-detection norm
        from tractor.patch import ModelMask
        from legacypipe.survey import SimpleGalaxy
        from tractor.basics import NanoMaggies

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
        #galmod = np.maximum(0, galmod)
        #galmod /= galmod.sum()
        galnorm = np.sqrt(np.sum(galmod**2))
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
            return fitsio.read_header(fn)

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
                    except OSError as err:
                        print('Warning: failed to parse FITS header line: ' +
                              ('"%s"; error "%s"; skipped' % (line.strip(), str(err))))
                              
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
        print('Reading data quality image', self.dqfn, 'ext', self.hdu)
        dq = self._read_fits(self.dqfn, self.hdu, **kwargs)
        return dq

    def remap_dq(self, dq, header):
        '''
        Called by get_tractor_image() to map the results from read_dq
        into a bitmask.
        '''
        return self.remap_dq_cp_codes(dq, header)

    def remap_dq_cp_codes(self, dq, header):
        '''
        Some versions of the CP use integer codes, not bit masks.
        This converts them.
        '''
        from legacypipe.image import CP_DQ_BITS
        dqbits = np.zeros(dq.shape, np.int16)
        '''
        1 = bad
        2 = no value (for remapped and stacked data)
        3 = saturated
        4 = bleed mask
        5 = cosmic ray
        6 = low weight
        7 = diff detect (multi-exposure difference detection from median)
        8 = long streak (e.g. satellite trail)
        '''
        dqbits[dq == 1] |= CP_DQ_BITS['badpix']
        dqbits[dq == 2] |= CP_DQ_BITS['badpix']
        dqbits[dq == 3] |= CP_DQ_BITS['satur']
        dqbits[dq == 4] |= CP_DQ_BITS['bleed']
        dqbits[dq == 5] |= CP_DQ_BITS['cr']
        dqbits[dq == 6] |= CP_DQ_BITS['badpix']
        dqbits[dq == 7] |= CP_DQ_BITS['trans']
        dqbits[dq == 8] |= CP_DQ_BITS['trans']
        return dqbits
    
    def read_invvar(self, **kwargs):
        '''
        Reads the inverse-variance (weight) map image.
        '''
        print('Reading weight map image', self.wtfn, 'ext', self.hdu)
        invvar = self._read_fits(self.wtfn, self.hdu, **kwargs)
        return invvar

    def read_invvar_clipped(self, clip=True, clipThresh=0.01, **kwargs):
        '''A function that can optionally be called by subclassers for read_invvar,
        clipping fpack artifacts to zero.'''
        invvar = self._read_fits(self.wtfn, self.hdu, **kwargs)
        if clip:
            # Clamp near-zero (incl negative!) invvars to zero.
            # These arise due to fpack.
            if clipThresh > 0.:
                med = np.median(invvar[invvar > 0])
                thresh = clipThresh * med
            else:
                thresh = 0.
            invvar[invvar < thresh] = 0
        return invvar

    def get_tractor_wcs(self, wcs, x0, y0, tai=None,
                        primhdr=None, imghdr=None):
        #from tractor.basics import ConstantFitsWcs
        #twcs = ConstantFitsWcs(wcs)
        from legacypipe.survey import LegacySurveyWcs
        twcs = LegacySurveyWcs(wcs, tai)
        if x0 or y0:
            twcs.setX0Y0(x0,y0)
        return twcs

    def get_wcs(self, hdr=None):
        from astrometry.util.util import wcs_pv2sip_hdr
        # Make sure the PV-to-SIP converter samples enough points for small
        # images
        stepsize = 0
        if min(self.width, self.height) < 600:
            stepsize = min(self.width, self.height) / 10.;
        if hdr is None:
            hdr = self.read_image_header()
        wcs = wcs_pv2sip_hdr(hdr, stepsize=stepsize)
        # Correction: ccd ra,dec offsets from zeropoints/CCDs file
        dra,ddec = self.dradec
        # print('Applying astrometric zeropoint:', (dra,ddec))
        r,d = wcs.get_crval()
        wcs.set_crval((r + dra / np.cos(np.deg2rad(d)), d + ddec))
        wcs.version = ''
        phdr = self.read_image_primary_header()
        wcs.plver = phdr.get('PLVER', '').strip()
        return wcs

    def get_sig1(self, **kwargs):
        if self.sig1 is not None:
            # CCDs table sig1 is in nanomaggies
            return self.sig1

        # these sig1 values are in image counts; scale to nanomaggies
        zpscale = NanoMaggies.zeropointToScale(self.ccdzpt)

        skysig1 = self.get_sky_sig1(**kwargs)
        if skysig1 is None:
            iv = im.read_invvar(**kwargs)
            dq = im.read_dq(**kwargs)
            skysig1 = 1./np.sqrt(np.median(iv[dq == 0]))
        return skysig1 / zpscale

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
        from tractor.utils import get_class_from_name

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
                print('Found', len(I), 'matching CCDs')
                if len(I) == 1:
                    Ti = T[I[0]]
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
                       normalizePsf=False,
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
        if self.merged_psffn is not None and not os.path.exists(self.merged_psffn):
            print('Merged PsfEx model does not exist:', self.merged_psffn)

        if self.merged_psffn is not None and os.path.exists(self.merged_psffn):
            try:
                print('Reading merged PsfEx models from', self.merged_psffn)
                T = fits_table(self.merged_psffn)
                I, = np.nonzero((T.expnum == self.expnum) *
                                np.array([c.strip() == self.ccdname
                                          for c in T.ccdname]))
                print('Found', len(I), 'matching CCDs')
                if len(I) == 1:
                    Ti = T[I[0]]
                    # Remove any padding
                    degree = Ti.poldeg1
                    # number of terms in polynomial
                    ne = (degree + 1) * (degree + 2) // 2
                    Ti.psf_mask = Ti.psf_mask[:ne, :Ti.psfaxis1, :Ti.psfaxis2]
                    # If degree 0, set polname* to avoid assertion error in tractor
                    if degree == 0:
                        Ti.polname1 = 'X_IMAGE'
                        Ti.polname2 = 'Y_IMAGE'
                        Ti.polgrp1 = 1
                        Ti.polgrp2 = 1
                        Ti.polngrp = 1

                    psfex = PsfExModel(Ti=Ti)

                    if normalizePsf:
                        print('NORMALIZING PSF!')
                        psf = NormalizedPixelizedPsfEx(None, psfex=psfex)
                    else:
                        psf = PixelizedPsfEx(None, psfex=psfex)

                    psf.version = Ti.legpipev.strip()
                    psf.plver = Ti.plver.strip()
                    psf.fwhm = Ti.psf_fwhm
            except:
                import traceback
                traceback.print_exc()

        if psf is None:
            print('Reading PsfEx model from', self.psffn)

            if normalizePsf:
                print('NORMALIZING PSF!')
                psf = NormalizedPixelizedPsfEx(self.psffn)
            else:
                psf = PixelizedPsfEx(self.psffn)

            hdr = fitsio.read_header(self.psffn)
            psf.version = hdr.get('LEGSURV', None)
            if psf.version is None:
                psf.version = str(os.stat(self.psffn).st_mtime)
            psf.plver = hdr.get('PLVER', '').strip()

            hdr = fitsio.read_header(self.psffn, ext=1)
            psf.fwhm = hdr['PSF_FWHM']

        psf.shift(x0, y0)
        if hybridPsf:
            from tractor.psf import HybridPixelizedPSF
            psf = HybridPixelizedPSF(psf, cx=w/2., cy=h/2.)

        print('Using PSF model', psf)
        return psf

    ######## Calibration tasks ###########


    def funpack_files(self, imgfn, maskfn, hdu, todelete):
        ''' Source Extractor can't handle .fz files, so unpack them.'''
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

    def run_se(self, imgfn, maskfn):
        from astrometry.util.file import trymakedirs
        sedir = self.survey.get_se_dir()
        trymakedirs(self.sefn, dir=True)
        # We write the SE catalog to a temp file then rename, to avoid
        # partially-written outputs.
        tmpfn = os.path.join(os.path.dirname(self.sefn),
                             'tmp-' + os.path.basename(self.sefn))
        cmd = ' '.join([
            'sex',
            '-c', os.path.join(sedir, self.camera + '.se'),
            '-PARAMETERS_NAME', os.path.join(sedir, self.camera + '.param'),
            '-FILTER_NAME %s' % os.path.join(sedir, self.camera + '.conv'),
            '-FLAG_IMAGE %s' % maskfn,
            '-CATALOG_NAME %s' % tmpfn,
            imgfn])
        print(cmd)
        rtn = os.system(cmd)
        if rtn:
            raise RuntimeError('Command failed: ' + cmd)
        os.rename(tmpfn, self.sefn)

    def run_psfex(self, git_version=None):
        from astrometry.util.file import trymakedirs
        from legacypipe.survey import get_git_version
        sedir = self.survey.get_se_dir()
        trymakedirs(self.psffn, dir=True)
        primhdr = self.read_image_primary_header()
        plver = primhdr.get('PLVER', 'V0.0')
        if git_version is None:
            git_version = get_git_version()
        # We write the PSF model to a .fits.tmp file, then rename to .fits
        psfdir = os.path.dirname(self.psffn)
        psfoutfn = os.path.join(psfdir, os.path.basename(self.sefn).replace('.fits','') + '.fits')
        cmds = ['psfex -c %s -PSF_DIR %s -PSF_SUFFIX .fits.tmp %s' %
                (os.path.join(sedir, self.camera + '.psfex'),
                 psfdir, self.sefn),
                'mv %s %s' % (psfoutfn + '.tmp', psfoutfn),
                'modhead %s LEGPIPEV "%s" "legacypipe git version"' %
                (self.psffn, git_version),
                'modhead %s PLVER "%s" "CP ver of image file"' % (self.psffn, plver)]
        for cmd in cmds:
            print(cmd)
            rtn = os.system(cmd)
            if rtn:
                raise RuntimeError('Command failed: %s: return value: %i' %
                                   (cmd,rtn))

    def run_sky(self, splinesky=False, git_version=None):
        from legacypipe.survey import get_version_header
        from scipy.ndimage.morphology import binary_dilation
        from astrometry.util.file import trymakedirs

        slc = self.get_good_image_slice(None)
        img = self.read_image(slice=slc)
        wt = self.read_invvar(slice=slc)
        hdr = get_version_header(None, self.survey.get_survey_dir(),
                                 git_version=git_version)
        primhdr = self.read_image_primary_header()
        plver = primhdr.get('PLVER', 'V0.0')
        hdr.delete('PROCTYPE')
        hdr.add_record(dict(name='PROCTYPE', value='ccd',
                            comment='NOAO processing type'))
        hdr.add_record(dict(name='PRODTYPE', value='skymodel',
                            comment='NOAO product type'))
        hdr.add_record(dict(name='PLVER', value=plver,
                            comment='CP ver of image file'))

        if splinesky:
            from tractor.splinesky import SplineSky
            from scipy.ndimage.filters import uniform_filter

            boxsize = self.splinesky_boxsize

            # Start by subtracting the overall median
            good = (wt > 0)
            if np.sum(good) == 0:
                raise RuntimeError('No pixels with weight > 0 in: ' + str(self))
            med = np.median(img[good])

            # For DECam chips where we drop half the chip, spline becomes underconstrained
            if min(img.shape) / boxsize < 4:
                boxsize /= 2

            # Compute initial model...
            skyobj = SplineSky.BlantonMethod(img - med, good, boxsize)
            skymod = np.zeros_like(img)
            skyobj.addTo(skymod)

            # Now mask bright objects in a boxcar-smoothed (image - initial sky model)
            sig1 = 1./np.sqrt(np.median(wt[good]))
            # Smooth by a boxcar filter before cutting pixels above threshold --
            boxcar = 5
            # Sigma of boxcar-smoothed image
            bsig1 = sig1 / boxcar
            masked = np.abs(uniform_filter(img-med-skymod, size=boxcar, mode='constant')
                            > (3.*bsig1))
            masked = binary_dilation(masked, iterations=3)
            good[masked] = False
            sig1b = 1./np.sqrt(np.median(wt[good]))

            # Now find the final sky model using that more extensive mask
            skyobj = SplineSky.BlantonMethod(img - med, good, boxsize)
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
            from tractor.sky import ConstantSky
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
            hdr.add_record(dict(name='SIG1', value=sig1,
                                comment='Median stdev of unmasked pixels'))
            hdr.add_record(dict(name='SIG1B', value=sig1,
                                comment='Median stdev of unmasked pixels+'))
            trymakedirs(self.skyfn, dir=True)
            tsky.write_fits(self.skyfn, hdr=hdr)
            print('Wrote sky model', self.skyfn)

    def run_calibs(self, psfex=True, sky=True, se=False,
                   fcopy=False, use_mask=True,
                   force=False, git_version=None,
                   splinesky=False):
        '''
        Run calibration pre-processing steps.
        '''
        if psfex and not force:
            # Check whether PSF model already exists
            try:
                self.read_psf_model(0, 0, pixPsf=True, hybridPsf=True)
                psfex = False
            except Exception as e:
                print('Did not find existing PsfEx model for', self, ':', e)
        if psfex:
            se = True

        # Don't need to run source extractor if the catalog file already exists
        if se and os.path.exists(self.sefn) and (not force):
            se = False

        if sky and not force:
            # Check whether sky model already exists
            try:
                self.read_sky_model(splinesky=splinesky)
                sky = False
            except Exception as e:
                print('Did not find existing sky model for', self, ':', e)

        if se:
            # The image & mask files to process (funpacked if necessary)
            todelete = []
            imgfn,maskfn = self.funpack_files(self.imgfn, self.dqfn,
                                              self.hdu, todelete)
            self.run_se(imgfn, maskfn)
            for fn in todelete:
                os.unlink(fn)
        if psfex:
            self.run_psfex(git_version=git_version)
        if sky:
            self.run_sky(splinesky=splinesky, git_version=git_version)



from tractor import PixelizedPsfEx, PixelizedPSF
class NormalizedPixelizedPsfEx(PixelizedPsfEx):
    def __str__(self):
        return 'NormalizedPixelizedPsfEx'

    def getFourierTransform(self, px, py, radius):
        fft, (cx,cy), shape, (v,w) = super(NormalizedPixelizedPsfEx, self).getFourierTransform(px, py, radius)
        #print('NormalizedPSF: getFourierTransform at', (px,py), ': sum', fft.sum(), 'zeroth element:', fft[0][0], 'max', np.max(np.abs(fft)))
        sum = np.abs(fft[0][0])
        fft /= sum
        #print('NormalizedPixelizedPsfEx: getFourierTransform at', (px,py), ': sum', sum)
        return fft, (cx,cy), shape, (v,w)

    def getImage(self, px, py):
        #print('NormalizedPixelizedPsfEx: getImage at', px,py)
        img = super(NormalizedPixelizedPsfEx, self).getImage(px, py)
        img /= np.sum(img)
        return img

    def constantPsfAt(self, x, y):
        #print('NormalizedPixelizedPsfEx: constantPsf at', x,y)
        pix = self.psfex.at(x, y)
        pix /= pix.sum()
        return PixelizedPSF(pix)

