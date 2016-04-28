from __future__ import print_function
import sys
import os
import fitsio
import numpy as np

from image import LegacySurveyImage
from common import create_temp, CP_DQ_BITS
from astrometry.util.util import Tan, Sip, anwcs_t

from astrometry.util.util import wcs_pv2sip_hdr
from astrometry.util.file import trymakedirs

from tractor.sky import ConstantSky
from tractor.basics import NanoMaggies, ConstantFitsWcs, LinearPhotoCal
from tractor.image import Image
from tractor.tractortime import TAITime

from legacypipe.common import zeropoint_for_ptf

'''
Code specific to images from the 90prime camera on the Bok telescope,
processed by the NOAO pipeline.  This is currently just a sketch.
'''

class BokImage(LegacySurveyImage):
    '''
    A LegacySurveyImage subclass to handle images from the 90prime
    camera on the Bok telescope.

    Currently, there are several hacks and shortcuts in handling the
    calibration; this is a sketch, not a final working solution.

    '''
    def __init__(self, survey, t):
        super(BokImage, self).__init__(survey, t)
        self.pixscale= 0.455
        self.dqfn = 'junk' #self.imgfn.replace('_oi.fits', '_od.fits')
        self.whtfn= self.imgfn.replace('.fits','.wht.fits')

        #expstr = '%10i' % self.expnum
        #self.calname = '%s/%s/bok-%s-%s' % (expstr[:5], expstr, expstr, self.ccdname)
        #self.name = '%s-%s' % (expstr, self.ccdname)
        #print('self.ccdname=',self.ccdname,'type(self.ccdname)=',type(self.ccdname))
        
        expstr = '12345' #self.ccdname.strip()
        self.calname = os.path.basename(self.imgfn).replace('.fits','') 
        self.name = self.imgfn #'%s' % self.ccdname.strip() 

        calibdir = os.path.join(self.survey.get_calib_dir(), self.camera)
        self.pvwcsfn = os.path.join(calibdir, 'astrom-pv', self.calname + '.wcs.fits')
        self.sefn = os.path.join(calibdir, 'sextractor', self.calname + '.fits')
        self.psffn = os.path.join(calibdir, 'psfex', self.calname + '.fits')
        self.skyfn = os.path.join(calibdir, 'sky', self.calname + '.fits')
        print('in BokImage init, calibdir=%s,self.calname=%s,self.imgfn=%s, self.whtfn=%s,self.dqfn=%s, self.sefn=%s, self.psffn=%s' % (calibdir,self.calname,self.imgfn,self.whtfn,self.dqfn,self.sefn, self.psffn))
        
    def __str__(self):
        return 'Bok ' + self.name

    def read_sky_model(self, **kwargs):
        ## HACK -- create the sky model on the fly
        img = self.read_image()
        sky = np.median(img)
        print('Median "sky" model:', sky)
        sky = ConstantSky(sky)
        sky.version = '0'
        sky.plver = '0'
        return sky

    def read_dq(self, **kwargs):
        #already account for bad pixels in wht, so return array of 0s (good everywhere)
        img = self._read_fits(self.imgfn, self.hdu, **kwargs)
        dq=np.zeros(img.shape).astype(np.int)
        return dq

    def read_invvar(self, **kwargs):
        print('Reading inv=%s for image=%s, hdu=' % (self.whtfn,self.imgfn), self.hdu)
        X = self._read_fits(self.whtfn, self.hdu, **kwargs)
        return X
        #print('Reading inverse-variance for image', self.imgfn, 'hdu', self.hdu)
        #img = self.read_image(**kwargs)
        ## # Estimate per-pixel noise via Blanton's 5-pixel MAD
        #slice1 = (slice(0,-5,10),slice(0,-5,10))
        #slice2 = (slice(5,None,10),slice(5,None,10))
        #mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
        #sig1 = 1.4826 * mad / np.sqrt(2.)
        #print('sig1 estimate:', sig1)
        #invvar = np.ones_like(img) / sig1**2
        #return invvar

    #read the TPV header, convert it to SIP, and apply an offset from the CCDs tabl
    def get_wcs(self):
        print('---in bok.py get_wcs ----')
        hdr = fitsio.read_header(self.imgfn, self.hdu)
        wcs = wcs_pv2sip_hdr(hdr)
        dra,ddec = self.survey.get_astrometric_zeropoint_for(self)
        r,d = wcs.get_crval()
        print('Applying astrometric zeropoint:', (dra,ddec))
        wcs.set_crval((r + dra, d + ddec))
        wcs.version = ''
        phdr = fitsio.read_header(self.imgfn, 0)
        wcs.plver = phdr.get('PLVER', '').strip()
        return wcs



    #override funcs get_tractor_image calls
    #def get_wcs(self):
    #    return self.read_pv_wcs()

        
    def run_calibs(self, psfex=True, sky=True, funpack=False, git_version=None,
                       force=False,
                       **kwargs):
        print('run_calibs for', self.name, ': sky=', sky, 'kwargs', kwargs) 
        se = False
        if psfex and os.path.exists(self.psffn) and (not force):
            psfex = False
        if psfex:
            se = True

        if se and os.path.exists(self.sefn) and (not force):
            se = False
        if se:
            sedir = self.survey.get_se_dir()
            trymakedirs(self.sefn, dir=True)
            ####
            trymakedirs('junk',dir=True) #need temp dir for mask-2 and invvar map
            hdu=0
            maskfn= self.imgfn.replace('_scie_','_mask_')
            #invvar= self.read_invvar(self.imgfn,maskfn,hdu) #note, all post processing on image,mask done in read_invvar
            invvar= self.read_invvar()
            mask= self.read_dq()
            maskfn= os.path.join('junk',os.path.basename(maskfn))
            invvarfn= maskfn.replace('_mask_','_invvar_')
            fitsio.write(maskfn, mask)
            fitsio.write(invvarfn, invvar)
            print('wrote mask-2 to %s, invvar to %s' % (maskfn,invvarfn))
            #run se 
            hdr=fitsio.read_header(self.imgfn,ext=hdu)
            magzp  = zeropoint_for_ptf(hdr)
            seeing = hdr['PIXSCALE'] * hdr['MEDFWHM']
            gain= hdr['GAIN']
            cmd = ' '.join(['sex','-c', os.path.join(sedir,'ptf.se'),
                            '-WEIGHT_IMAGE %s' % invvarfn, '-WEIGHT_TYPE MAP_WEIGHT',
                            '-GAIN %f' % gain,
                            '-FLAG_IMAGE %s' % maskfn,
                            '-FLAG_TYPE OR',
                            '-SEEING_FWHM %f' % seeing,
                            '-DETECT_MINAREA 3',
                            '-PARAMETERS_NAME', os.path.join(sedir,'ptf.param'),
                            '-FILTER_NAME', os.path.join(sedir, 'ptf_gauss_3.0_5x5.conv'),
                            '-STARNNW_NAME', os.path.join(sedir, 'ptf_default.nnw'),
                            '-PIXEL_SCALE 0',
                            # SE has a *bizarre* notion of "sigma"
                            '-DETECT_THRESH 1.0',
                            '-ANALYSIS_THRESH 1.0',
                            '-MAG_ZEROPOINT %f' % magzp,
                            '-CATALOG_NAME', self.sefn,
                            self.imgfn])
            print(cmd)
            if os.system(cmd):
                raise RuntimeError('Command failed: ' + cmd)
        if psfex:
            sedir = self.survey.get_se_dir()
            trymakedirs(self.psffn, dir=True)
            # If we wrote *.psf instead of *.fits in a previous run...
            oldfn = self.psffn.replace('.fits', '.psf')
            if os.path.exists(oldfn):
                print('Moving', oldfn, 'to', self.psffn)
                os.rename(oldfn, self.psffn)
            else: 
                cmd= ' '.join(['psfex',self.sefn,'-c', os.path.join(sedir,'ptf.psfex'),
                    '-PSF_DIR',os.path.dirname(self.psffn)])
                print(cmd)
                if os.system(cmd):
                    raise RuntimeError('Command failed: ' + cmd)
    
    #cannot use image.py version of this function b/c 
    #don't need check that imghdr['EXTNAME'] matches ccdname
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
            #e = imghdr['EXTNAME']
            #assert(e.strip() == self.ccdname.strip())
        else:
            img = np.zeros((imh, imw))
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
        # imghdr['FWHM']
        psf_fwhm = self.fwhm 
        psf_sigma = psf_fwhm / 2.35
        primhdr = self.read_image_primary_header()

        sky = self.read_sky_model(splinesky=splinesky, slc=slc,
                                  primhdr=primhdr, imghdr=imghdr)
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
            #KJB print('img.dtype=',img.dtype,'invvar.dtype=',invvar.dtype,'zpscale.dtype=',zpscale.dtype)
            invvar= invvar.astype('float')* zpscale**2 #invvar *= zpscale**2
            if not subsky:
                sky.scale(1./zpscale)
            zpscale = 1.

        assert(np.sum(invvar > 0) > 0)
        if get_invvar:
            sig1 = 1./np.sqrt(np.median(invvar[invvar > 0]))
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
        print('PSF norm', psfnorm, 'vs Gaussian',
              1./(2. * np.sqrt(np.pi) * psf_sigma))

        # Galaxy-detection norm
        tim.band = band
        galnorm = self.galaxy_norm(tim)
        print('Galaxy norm:', galnorm)
        
        # CP (DECam) images include DATE-OBS and MJD-OBS, in UTC.
        import astropy.time
        #mjd_utc = mjd=primhdr.get('MJD-OBS', 0)
        mjd_tai = astropy.time.Time(primhdr['DATE-OBS']).tai.mjd
        tim.slice = slc
        tim.time = TAITime(None, mjd=mjd_tai)
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
        tim.plver = 'junk' #junk numer PLVER not in header
        tim.skyver = ('junk','junk') #(sky.version, sky.plver)
        tim.wcsver = ('junk','junk') #(wcs.version, wcs.plver)
        tim.psfver = ('junk','junk') #(psf.version, psf.plver)
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


