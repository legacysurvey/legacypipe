from __future__ import print_function
import os
import numpy as np
import fitsio
from tractor.utils import get_class_from_name
from tractor.basics import NanoMaggies, ConstantFitsWcs, LinearPhotoCal
from tractor.image import Image
from tractor.tractortime import TAITime
from .common import CP_DQ_BITS

'''
Generic image handling code.
'''

class LegacySurveyImage(object):
    '''
    A base class containing common code for the images we handle.

    You shouldn't directly instantiate this class, but rather use the appropriate
    subclass:
     * DecamImage
     * BokImage
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
        #print('LegacySurveyImage __init__')
        
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
        if self.camera == '90prime': self.ccdzpt= ccd.ccdzpt #Bok specific, zp stored in ccd table for now
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

        Default is to return all CCDs.
        '''
        return np.arange(len(ccds))
    
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
                          gaussPsf=False, pixPsf=False,
                          splinesky=False,
                          nanomaggies=True, subsky=True, tiny=5,
                          dq=True, invvar=True, pixels=True):
        '''
        Returns a tractor.Image ("tim") object for this image.
        
        Options describing a subimage to return:

        - *slc*: y,x slice objects
        - *radecpoly*: numpy array, shape (N,2), RA,Dec polygon describing bounding box to select.

        Options determining the PSF model to use:

        - *gaussPsf*: single circular Gaussian PSF based on header FWHM value.
        - *const2Psf*: 2-component general Gaussian fit to PsfEx model at image center.
        - *pixPsf*: pixelized PsfEx model at image center.

        Options determining the sky model to use:
        
        - *splinesky*: median filter chunks of the image, then spline those.

        Options determining the units of the image:

        - *nanomaggies*: convert the image to be in units of NanoMaggies;
          *tim.zpscale* contains the scale value the image was divided by.

        - *subsky*: instantiate and subtract the initial sky model,
          leaving a constant zero sky model?

        '''
        from astrometry.util.miscutils import clip_polygon

        get_dq = dq
        get_invvar = invvar
        
        band = self.band
        imh,imw = self.get_image_shape()
        
        wcs = self.get_wcs()
        x0,y0 = 0,0
        x1 = x0 + imw
        y1 = y0 + imh
        if slc is None and radecpoly is not None:
            imgpoly = [(1,1),(1,imh),(imw,imh),(imw,1)]
            ok,tx,ty = wcs.radec2pixelxy(radecpoly[:-1,0], radecpoly[:-1,1])
            tpoly = zip(tx,ty)
            clip = clip_polygon(imgpoly, tpoly)
            clip = np.array(clip)
            if len(clip) == 0:
                return None
            x0,y0 = np.floor(clip.min(axis=0)).astype(int)
            x1,y1 = np.ceil (clip.max(axis=0)).astype(int)
            slc = slice(y0,y1+1), slice(x0,x1+1)
            if y1 - y0 < tiny or x1 - x0 < tiny:
                print('Skipping tiny subimage')
                return None
        if slc is not None:
            sy,sx = slc
            y0,y1 = sy.start, sy.stop
            x0,x1 = sx.start, sx.stop

        old_extent = (x0,x1,y0,y1)
        new_extent = self.get_good_image_slice((x0,x1,y0,y1), get_extent=True)
        if new_extent != old_extent:
            x0,x1,y0,y1 = new_extent
            print('Applying good subregion of CCD: slice is', x0,x1,y0,y1)
            if x0 >= x1 or y0 >= y1:
                return None
            slc = slice(y0,y1), slice(x0,x1)

        if pixels:
            print('Reading image slice:', slc)
            img,imghdr = self.read_image(header=True, slice=slc)
            #print('SATURATE is', imghdr.get('SATURATE', None))
            #print('Max value in image is', img.max())
            # check consistency... something of a DR1 hangover
            e = imghdr['EXTNAME']
            assert(e.strip() == self.ccdname.strip())
        else:
            img = np.zeros((imh, imw), np.float32)
            imghdr = dict()
            if slc is not None:
                img = img[slc]
            
        if get_invvar:
            invvar = self.read_invvar(slice=slc, clipThresh=0.)
        else:
            invvar = np.ones_like(img)
            
        if get_dq:
            dq = self.read_dq(slice=slc)
            invvar[dq != 0] = 0.
        if np.all(invvar == 0.):
            print('Skipping zero-invvar image')
            return None
        assert(np.all(np.isfinite(img)))
        assert(np.all(np.isfinite(invvar)))
        assert(not(np.all(invvar == 0.)))

        # header 'FWHM' is in pixels
        psf_fwhm = self.fwhm 
        psf_sigma = psf_fwhm / 2.35
        primhdr = self.read_image_primary_header()

        sky = self.read_sky_model(splinesky=splinesky, slc=slc,
                                  primhdr=primhdr, imghdr=imghdr)
        skysig1 = getattr(sky, 'sig1', None)
        
        midsky = 0.
        if subsky:
            print('Instantiating and subtracting sky model...')
            from tractor.sky import ConstantSky
            skymod = np.zeros_like(img)
            sky.addTo(skymod)
            img -= skymod
            midsky = np.median(skymod)
            zsky = ConstantSky(0.)
            zsky.version = sky.version
            zsky.plver = sky.plver
            del skymod
            del sky
            sky = zsky
            del zsky

        magzp = self.survey.get_zeropoint_for(self)
        orig_zpscale = zpscale = NanoMaggies.zeropointToScale(magzp)
        if nanomaggies:
            # Scale images to Nanomaggies
            img /= zpscale
            invvar = invvar * zpscale**2
            if not subsky:
                sky.scale(1./zpscale)
            zpscale = 1.

        assert(np.sum(invvar > 0) > 0)
        if get_invvar:
            sig1 = 1./np.sqrt(np.median(invvar[invvar > 0]))
        elif skysig1 is not None:
            sig1 = skysig1
        else:
            # Estimate from the image?
            # # Estimate per-pixel noise via Blanton's 5-pixel MAD
            slice1 = (slice(0,-5,10),slice(0,-5,10))
            slice2 = (slice(5,None,10),slice(5,None,10))
            mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
            sig1 = 1.4826 * mad / np.sqrt(2.)
            print('sig1 estimate:', sig1)
            invvar *= (1. / sig1**2)
            
        assert(np.all(np.isfinite(img)))
        assert(np.all(np.isfinite(invvar)))
        assert(np.isfinite(sig1))

        if subsky:
            ##
            imgmed = np.median(img[invvar>0])
            if np.abs(imgmed) > sig1:
                print('WARNING: image median', imgmed, 'is more than 1 sigma away from zero!')
                # Boom!
                #assert(False)

        twcs = ConstantFitsWcs(wcs)
        if x0 or y0:
            twcs.setX0Y0(x0,y0)

        psf = self.read_psf_model(x0, y0, gaussPsf=gaussPsf, pixPsf=pixPsf,
                                  psf_sigma=psf_sigma,
                                  cx=(x0+x1)/2., cy=(y0+y1)/2.)

        tim = Image(img, invvar=invvar, wcs=twcs, psf=psf,
                    photocal=LinearPhotoCal(zpscale, band=band),
                    sky=sky, name=self.name + ' ' + band)
        assert(np.all(np.isfinite(tim.getInvError())))

        # PSF norm
        psfnorm = self.psf_norm(tim)
        #print('PSF norm', psfnorm, 'vs Gaussian', 1./(2. * np.sqrt(np.pi) * psf_sigma))

        # Galaxy-detection norm
        tim.band = band
        galnorm = self.galaxy_norm(tim)
        #print('Galaxy norm:', galnorm)

        # CP (DECam) images include DATE-OBS and MJD-OBS, in UTC.
        import astropy.time
        #mjd_utc = mjd=primhdr.get('MJD-OBS', 0)
        #mjd_tai = astropy.time.Time(primhdr['DATE-OBS']).tai.mjd
        mjd_tai = astropy.time.Time(self.mjdobs, format='mjd', scale='utc').tai.mjd
        tim.time = TAITime(None, mjd=mjd_tai)
        tim.slice = slc
        tim.zr = [-3. * sig1, 10. * sig1]
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
        tim.plver = primhdr['PLVER'].strip()
        tim.skyver = (sky.version, sky.plver)
        tim.wcsver = (wcs.version, wcs.plver)
        tim.psfver = (psf.version, psf.plver)
        if get_dq:
            tim.dq = dq
        tim.dq_bits = CP_DQ_BITS
        tim.saturation = imghdr.get('SATURATE', None)
        tim.satval = tim.saturation or 0.
        if subsky:
            tim.satval -= midsky
        if nanomaggies:
            tim.satval /= orig_zpscale
        subh,subw = tim.shape
        tim.subwcs = tim.sip_wcs.get_subimage(tim.x0, tim.y0, subw, subh)
        mn,mx = tim.zr
        tim.ima = dict(interpolation='nearest', origin='lower', cmap='gray',
                       vmin=mn, vmax=mx)
        return tim

    def psf_norm(self, tim, x=None, y=None):
        # PSF norm
        psf = tim.psf
        h,w = tim.shape
        if x is None:
            x = w/2.
        if y is None:
            y = h/2.
        patch = psf.getPointSourcePatch(x, y).patch
        #print('PSF PointSourcePatch: sum', patch.sum())
        # Clamp up to zero and normalize before taking the norm
        patch = np.maximum(0, patch)
        patch /= patch.sum()
        psfnorm = np.sqrt(np.sum(patch**2))
        return psfnorm

    def galaxy_norm(self, tim, x=None, y=None):
        # Galaxy-detection norm
        from tractor.galaxy import ExpGalaxy
        from tractor.ellipses import EllipseE
        from tractor.patch import Patch
        h,w = tim.shape
        band = tim.band
        if x is None:
            x = w/2.
        if y is None:
            y = h/2.
        pos = tim.wcs.pixelToPosition(x, y)
        gal = ExpGalaxy(pos, NanoMaggies(**{band:1.}), EllipseE(0.45, 0., 0.))
        S = 32
        mm = Patch(int(x-S), int(y-S), np.ones((2*S+1, 2*S+1), bool))
        galmod = gal.getModelPatch(tim, modelMask=mm).patch
        galmod = np.maximum(0, galmod)
        galmod /= galmod.sum()
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
        # return self.get_image_info()['dims']
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
        if self.imgfn.endswith('.gz'):
            return fitsio.read_header(self.imgfn)
        # Crazily, this can be MUCH faster than letting fitsio do it...
        hdr = fitsio.FITSHDR()
        foundEnd = False
        ff = open(self.imgfn, 'r')
        h = ''
        while True:
            h = h + ff.read(32768)
            while True:
                line = h[:80]
                h = h[80:]
                # HACK -- fitsio apparently can't handle CONTINUE
                if line[:8] != 'CONTINUE':
                    hdr.add_record(line)
                if line == ('END' + ' '*77):
                    foundEnd = True
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

    def get_wcs(self):
        print('LegacySurveyImage.get_wcs(); self:', type(self), self)
        print('MRO:', type(self).__mro__)
        return None
    
    def read_sky_model(self, splinesky=False, slc=None, **kwargs):
        '''
        Reads the sky model, returning a Tractor Sky object.
        '''
        fn = self.skyfn
        if splinesky:
            fn = self.splineskyfn
        print('Reading sky model from', fn)
        hdr = fitsio.read_header(fn)
        skyclass = hdr['SKY']
        clazz = get_class_from_name(skyclass)

        if getattr(clazz, 'from_fits', None) is not None:
            fromfits = getattr(clazz, 'from_fits')
            skyobj = fromfits(fn, hdr)
        else:
            fromfits = getattr(clazz, 'fromFitsHeader')
            skyobj = fromfits(hdr, prefix='SKY_')

        if slc is not None:
            sy,sx = slc
            x0,y0 = sx.start,sy.start
            skyobj.shift(x0, y0)

        skyobj.version = hdr.get('LEGPIPEV', '')
        if len(skyobj.version) == 0:
            skyobj.version = hdr.get('TRACTORV', '').strip()
            if len(skyobj.version) == 0:
                skyobj.version = str(os.stat(fn).st_mtime)
        skyobj.plver = hdr.get('PLVER', '').strip()
        sig1 = hdr.get('SIG1', None)
        if sig1 is not None:
            skyobj.sig1 = sig1
        return skyobj

    def read_psf_model(self, x0, y0, gaussPsf=False, pixPsf=False,
                       psf_sigma=1., cx=0, cy=0):
        psffn = None
        if gaussPsf:
            from tractor.basics import GaussianMixturePSF
            v = psf_sigma**2
            psf = GaussianMixturePSF(1., 0., 0., v, v, 0.)
            print('WARNING: using mock PSF:', psf)
            psf.version = '0'
            psf.plver = ''
        elif pixPsf:
            # spatially varying pixelized PsfEx
            from tractor.psfex import PixelizedPsfEx
            print('Reading PsfEx model from', self.psffn)
            psf = PixelizedPsfEx(self.psffn)
            psf.shift(x0, y0)
            psffn = self.psffn
        else:
            assert(False)
        print('Using PSF model', psf)

        if psffn is not None:
            hdr = fitsio.read_header(psffn)
            psf.version = hdr.get('LEGSURV', None)
            if psf.version is None:
                psf.version = str(os.stat(psffn).st_mtime)
            psf.plver = hdr.get('PLVER', '').strip()
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
    A class to hold common calibration tasks between the different surveys / image
    subclasses.
    '''

    def __init__(self):
        #print('CalibMixin __init__')
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
        hdr = fitsio.read_header(psffn, ext=1)
        if hdr.get('POLNAME1', None) is None:
            print('Did not find POLNAME1 in PsfEx header', psffn, '-- deleting')
            os.unlink(psffn)
            return False
        return True

    def check_se_cat(self, fn):
        from astrometry.util.fits import fits_table
        from astrometry.util.file import file_size
        # Check SourceExtractor catalog for file size = 0 or FITS table length = 0
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
        from legacypipe.common import create_temp

        tmpimgfn = None
        tmpmaskfn = None
        # For FITS files that are not actually fpack'ed, funpack -E
        # fails.  Check whether actually fpacked.
        fcopy = False
        hdr = fitsio.read_header(imgfn, ext=hdu)
        if not ((hdr['XTENSION'] == 'BINTABLE') and hdr.get('ZIMAGE', False)):
            print('Image %s, HDU %i is not fpacked; just imcopying.' % (imgfn,  hdu))
            fcopy = True

        tmpimgfn  = create_temp(suffix='.fits')
        tmpmaskfn = create_temp(suffix='.fits')
        todelete.append(tmpimgfn)
        todelete.append(tmpmaskfn)
        
        if fcopy:
            cmd = 'imcopy %s"+%i" %s' % (imgfn, hdu, tmpimgfn)
        else:
            cmd = 'funpack -E %i -O %s %s' % (hdu, tmpimgfn, imgfn)
        #cmd = 'funpack -E %i -O %s %s' % (hdu, tmpimgfn, imgfn)
        print(cmd)
        if os.system(cmd):
            raise RuntimeError('Command failed: ' + cmd)
        
        #cmd = 'funpack -E %i -O %s %s' % (hdu, tmpmaskfn, maskfn)
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

        cmd = ' '.join([
            'sex',
            '-c', os.path.join(sedir, surveyname + '.se'),
            '-FLAG_IMAGE ' + maskfn,
            '-SEEING_FWHM %f' % seeing,
            '-PARAMETERS_NAME', os.path.join(sedir, surveyname + '.param'),
            '-FILTER_NAME', os.path.join(sedir, 'gauss_5.0_9x9.conv'),
            '-STARNNW_NAME', os.path.join(sedir, 'default.nnw'),
            '-PIXEL_SCALE 0',
            # SE has a *bizarre* notion of "sigma"
            '-DETECT_THRESH 1.0',
            '-ANALYSIS_THRESH 1.0',
            '-MAG_ZEROPOINT %f' % magzp,
            '-CATALOG_NAME', self.sefn,
            imgfn])
        print(cmd)
        if os.system(cmd):
            raise RuntimeError('Command failed: ' + cmd)

    def run_psfex(self, surveyname):
        from astrometry.util.file import trymakedirs
        from legacypipe.common import get_git_version
        sedir = self.survey.get_se_dir()
        trymakedirs(self.psffn, dir=True)
        primhdr = self.read_image_primary_header()
        plver = primhdr.get('PLVER', '')
        verstr = get_git_version()
        cmds = ['psfex -c %s -PSF_DIR %s %s' %
                (os.path.join(sedir, surveyname + '.psfex'),
                 os.path.dirname(self.psffn), self.sefn),
                'modhead %s LEGPIPEV %s "legacypipe git version"' %
                (self.psffn, verstr),
                'modhead %s PLVER %s "CP ver of image file"' % (self.psffn, plver)]
        for cmd in cmds:
            print(cmd)
            rtn = os.system(cmd)
            if rtn:
                raise RuntimeError('Command failed: %s: return value: %i' %
                                   (cmd,rtn))

