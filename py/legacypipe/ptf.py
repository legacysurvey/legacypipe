from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np

from legacypipe.image import LegacySurveyImage
from legacypipe.common import *

from astropy.io import fits as astro_fits
import fitsio
from astrometry.util.file import trymakedirs
from astrometry.util.fits import fits_table
from astrometry.util.util import Tan, Sip, anwcs_t
from tractor.tractortime import TAITime

#for SFDMap() including for PTF filters
#from astrometry.util.starutil_numpy import *
#from astrometry.util.util import *

'''
Code specific to images from the (intermediate) Palomar Transient Factory (iPTF/PTF), bands = g,R.
11 CCDs and 1.2m telescope at Palomar Observatory.
'''

#PTF special handling of zeropoint
def zeropoint_for_ptf(hdr):
    if isinstance(magzp,str):
        print('WARNING: no ZeroPoint in header for image: ',tractor_image.imgfn)
        raise ValueError #magzp= 23.
    return magzp

##key functions##
def read_image(imgfn,hdu):
    '''return gain*pixel DN as numpy array'''
    print('Reading image from', imgfn, 'hdu', hdu)
    img,hdr= fitsio.read(imgfn, ext=hdu, header=True) 
    return img,hdr 

def read_dq(dqfn,hdu):
    '''return bit mask which Tractor calls "data quality" image
    PTF DMASK BIT DEFINITIONS
    BIT00   =                    0 / AIRCRAFT/SATELLITE TRACK
    BIT01   =                    1 / OBJECT (detected by SExtractor)
    BIT02   =                    2 / HIGH DARK-CURRENT
    BIT03   =                    3 / RESERVED FOR FUTURE USE
    BIT04   =                    4 / NOISY
    BIT05   =                    5 / GHOST
    BIT06   =                    6 / CCD BLEED
    BIT07   =                    7 / RAD HIT
    BIT08   =                    8 / SATURATED
    BIT09   =                    9 / DEAD/BAD
    BIT10   =                   10 / NAN (not a number)
    BIT11   =                   11 / DIRTY (10-sigma below coarse local median)
    BIT12   =                   12 / HALO
    BIT13   =                   13 / RESERVED FOR FUTURE USE
    BIT14   =                   14 / RESERVED FOR FUTURE USE
    BIT15   =                   15 / RESERVED FOR FUTURE USE
    INFOBITS=                    0 / Database infobits (2^2 and 2^3 excluded)
    '''
    print('Reading data quality image from', dqfn, 'hdu', hdu)
    mask= fitsio.read(dqfn, ext=hdu, header=False)
    mask[mask > 0]= mask[mask > 0]-2 #pixels flagged as SEXtractor objects == 2 so are good
    return mask.astype(np.int16) 

def read_invvar(imgfn,dqfn,hdu, clip=False):
    img,hdr= read_image(imgfn,hdu)
    dq= read_dq(dqfn,hdu)
    assert(dq.shape == img.shape)
    invvar=np.zeros(img.shape)
    invvar[dq == 0]= hdr['GAIN']/img[dq == 0] #mask-2 already done, bit 2^1 for SExtractor ojbects
    if clip:
        # Clamp near-zero (incl negative!) invvars to zero.
        # These arise due to fpack.
        if clipThresh > 0.:
            med = np.median(invvar[invvar > 0])
            thresh = clipThresh * med
        else:
            thresh = 0.
        invvar[invvar < thresh] = 0
    if np.any(invvar < 0): 
        if invvar[invvar < 0].shape[0] <= 10:
            print('invvar < 0 at %d pixels setting to 0 there, image= %s' % (invvar[invvar < 0].shape[0],imgfn))
            invvar[invvar < 0]= 0.
        else: 
            print('---WARNING--- invvar < 0 at > 10 pixels, something bad could be happening, img=  %s' % imgfn)
            print('writing invvar and where invvar to ./ then crashing code') 
            hdu = astro_fits.PrimaryHDU(invvar)
            hdu.writeto('./bad_invvar_%s' % os.path.basename(imgfn))
            new= np.zeros(invvar.shape).astype('int')
            new[invvar < 0] = 1
            hdu = astro_fits.PrimaryHDU(new)
            hdu.writeto('./where_invvar_lt0_%s' % os.path.basename(imgfn))
            raise ValueError
    return invvar

def isPTF(bands):
    return 'g_PTF' in bands or 'R_PTF' in bands

class PtfImage(LegacySurveyImage):
    '''
   
    A LegacySurveyImage subclass to handle images from the Dark Energy
    Camera, DECam, on the Blanco telescope.

    '''
    def __init__(self, survey, t):
        super(PtfImage, self).__init__(survey, t)

        # FIXME -- this should happen in the CCD table creation step.
        self.imgfn= os.path.join(os.path.dirname(self.imgfn),
                                 'ptf', os.path.basename(self.imgfn))

        hdr= self.read_image_primary_header()
        self.ccdzpt = hdr['IMAGEZPT'] + 2.5 * np.log10(self.exptime)

        self.pixscale= 1.01
        #print("--------pixscale= ",self.pixscale)
        #print("--------changing pixscale to ",1.01)
        #bit-mask
        self.dqfn = self.imgfn.replace('_scie_', '_mask_')
        #psfex catalogues
        calibdir = os.path.join(self.survey.get_calib_dir(), self.camera)
        self.sefn = os.path.join(calibdir, 'sextractor',
                                 os.path.basename(self.imgfn))
        #self.psffn = os.path.join(calibdir, 'psfex', self.calname + '.fits')
        self.psffn= os.path.join(calibdir, 'psfex',
                                 os.path.basename(self.imgfn)) #.replace('.fits','.psf')))
        print('####### self.imgfn,dqfn,calibdir,psffn= ',self.imgfn,self.dqfn,calibdir,self.psffn)
        #self.wtfn = self.imgfn.replace('_ooi_', '_oow_')

        self.name= self.imgfn
        #for i in dir(self):
        #    if i.startswith('__'): continue
        #    else: print('self.%s= ' % i,getattr(self, i))

        #self.dqfn = self.imgfn.replace('_ooi_', '_ood_')
        #self.wtfn = self.imgfn.replace('_ooi_', '_oow_')

        #for attr in ['imgfn', 'dqfn', 'wtfn']:
        #    fn = getattr(self, attr)
        #    if os.path.exists(fn):
        #        continue
        #    if fn.endswith('.fz'):
        #        fun = fn[:-3]
        #        if os.path.exists(fun):
        #            print('Using      ', fun)
        #            print('rather than', fn)
        #            setattr(self, attr, fun)
        #calibdir = os.path.join(self.survey.get_calib_dir(), self.camera)
        #self.pvwcsfn = os.path.join(calibdir, 'astrom-pv', self.calname + '.wcs.fits')
        #self.sefn = os.path.join(calibdir, 'sextractor', self.calname + '.fits')
        #self.psffn = os.path.join(calibdir, 'psfex', self.calname + '.fits')
        #self.skyfn = os.path.join(calibdir, 'sky', self.calname + '.fits')

    #def get_image_shape(self):
    #    return self.height, self.width

    #def shape(self):
    #    return self.get_image_shape()

    #def get_tractor_image(self, **kwargs):
    #    tim = super(PtfImage, self).get_tractor_image(**kwargs)
    #    return tim

    def __str__(self):
        return 'PTF ' + self.name
    
    #override funcs get_tractor_image calls
    def get_wcs(self):
        return self.read_pv_wcs()

    def read_pv_wcs(self):
        '''extract wcs from fits header directly'''
        hdr = fitsio.read_header(self.imgfn, self.hdu)
        H,W = self.get_image_shape()
        wcs= Tan(hdr['CRVAL1'], hdr['CRVAL2'],hdr['CRPIX1'],hdr['CRPIX2'],\
                     hdr['CD1_1'],hdr['CD1_2'],hdr['CD2_1'],hdr['CD2_2'],\
                     float(W),float(H))
        return wcs
    #    wcs.version = '0' #done in bok.py 
    #    wcs.plver = '0'
    #    return wcs
        #from astrometry.util.util import Sip
        #print('Reading WCS from', self.pvwcsfn)
        #wcs = Sip(self.pvwcsfn)
        #dra,ddec = self.survey.get_astrometric_zeropoint_for(self)
        #r,d = wcs.get_crval()
        #print('Applying astrometric zeropoint:', (dra,ddec))
        #wcs.set_crval((r + dra, d + ddec))
        #hdr = fitsio.read_header(self.pvwcsfn)
        #wcs.version = hdr.get('LEGPIPEV', '')
        #if len(wcs.version) == 0:
        #    wcs.version = hdr.get('TRACTORV', '').strip()
        #    if len(wcs.version) == 0:
        #        wcs.version = str(os.stat(self.pvwcsfn).st_mtime)
        #wcs.plver = hdr.get('PLVER', '').strip()
        #return wcs

    def get_good_image_subregion(self):
        pass
       
    def read_image(self,**kwargs):
        '''returns tuple of img,hdr'''
        return read_image(self.imgfn,self.hdu) 

    def read_dq(self,**kwargs):
        return read_dq(self.dqfn,self.hdu)

    def read_invvar(self, clip=False, clipThresh=0.2, **kwargs):
        return read_invvar(self.imgfn,self.dqfn,self.hdu)
    
    def read_sky_model(self, **kwargs):
        print('Constant sky model, median of ', self.imgfn)
        img,hdr = self.read_image(header=True)
        sky = np.median(img)
        print('Median "sky" =', sky)
        sky = ConstantSky(sky)
        sky.version = '0'
        sky.plver = '0'
        return sky

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
            invvar= read_invvar(self.imgfn,maskfn,hdu) #note, all post processing on image,mask done in read_invvar
            mask= read_dq(maskfn,hdu)
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

    
    def get_tractor_image(self, slc=None, radecpoly=None,
                          gaussPsf=False, const2psf=False, pixPsf=False,
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
        #if don't comment out tim = NoneType b/c clips all pixels out
        #if slc is None and radecpoly is not None:
        #    imgpoly = [(1,1),(1,imh),(imw,imh),(imw,1)]
        #    ok,tx,ty = wcs.radec2pixelxy(radecpoly[:-1,0], radecpoly[:-1,1])
        #    tpoly = zip(tx,ty)
        #    clip = clip_polygon(imgpoly, tpoly)
        #    clip = np.array(clip)
        #    if len(clip) == 0:
        #        return None
        #    x0,y0 = np.floor(clip.min(axis=0)).astype(int)
        #    x1,y1 = np.ceil (clip.max(axis=0)).astype(int)
        #    slc = slice(y0,y1+1), slice(x0,x1+1)
        #    if y1 - y0 < tiny or x1 - x0 < tiny:
        #        print('Skipping tiny subimage')
        #        return None
        #if slc is not None:
        #    sy,sx = slc
        #    y0,y1 = sy.start, sy.stop
        #    x0,x1 = sx.start, sx.stop

        #old_extent = (x0,x1,y0,y1)
        #new_extent = self.get_good_image_slice((x0,x1,y0,y1), get_extent=True)
        #if new_extent != old_extent:
        #    x0,x1,y0,y1 = new_extent
        #    print('Applying good subregion of CCD: slice is', x0,x1,y0,y1)
        #    if x0 >= x1 or y0 >= y1:
        #        return None
        #    slc = slice(y0,y1), slice(x0,x1)
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

        sky = self.read_sky_model(splinesky=splinesky, slc=slc)
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
            invvar *= zpscale**2
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
                assert(False)

        twcs = ConstantFitsWcs(wcs)
        if x0 or y0:
            twcs.setX0Y0(x0,y0)

        #print('gaussPsf:', gaussPsf, 'pixPsf:', pixPsf, 'const2psf:', const2psf)
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
        tim.plver = str(primhdr['PTFVERSN']).strip()
        tim.skyver = (sky.version, sky.plver)
        tim.wcsver = ('-1','-1') #wcs.version, wcs.plver)
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



def make_dir(name):
    if not os.path.exists(name): os.makedirs(name)
    else: print('WARNING path exists: ',name)


