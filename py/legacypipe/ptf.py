from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np

from legacypipe.image import LegacySurveyImage
from legacypipe.common import *

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
    return invvar

def ptf_zeropoint(imgfn):
    print('WARNING: zeropoints from header of ',imgfn)
    hdr=fitsio.read_header(imgfn)
    return hdr['IMAGEZPT'] + 2.5 * np.log10(hdr['EXPTIME'])

def isPTF(bands):
    return 'g_PTF' in bands or 'R_PTF' in bands

class PtfImage(LegacySurveyImage):
    '''
   
    A LegacySurveyImage subclass to handle images from the Dark Energy
    Camera, DECam, on the Blanco telescope.

    '''
    def __init__(self, decals, t):
        super(PtfImage, self).__init__(decals, t)

        print("--------pixscale= ",self.pixscale)
        print("--------changing pixscale to ",1.01)
        #bit-mask
        self.dqfn = self.imgfn.replace('_scie_', '_mask_')
        #psfex catalogues
        calibdir = os.path.join(self.decals.get_calib_dir(), self.camera)
        self.psffn= os.path.join(calibdir,'psfex/',os.path.basename(self.imgfn))
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
        #calibdir = os.path.join(self.decals.get_calib_dir(), self.camera)
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
        #dra,ddec = self.decals.get_astrometric_zeropoint_for(self)
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

    def run_calibs(self, **kwargs):
        # def run_calibs(self, pvastrom=True, psfex=True, sky=True, se=False,
        #           funpack=False, fcopy=False, use_mask=True,
        #           force=False, just_check=False, git_version=None,
        #           splinesky=False):
        '''
        Run calibration pre-processing steps.
        '''
        print('doing Nothing for Calibrations')
        pass

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
        print('#### IN KBJs get tractor image code! ###')
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

        magzp = self.decals.get_zeropoint_for(self)
        if isinstance(magzp,str): 
            print('WARNING: no ZeroPoint in header for image: ',self.imgfn)
            magzp= 23.
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
                                  const2psf=const2psf, psf_sigma=psf_sigma)

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


class PtfDecals(Decals):
    def __init__(self, **kwargs):
        super(PtfDecals, self).__init__(**kwargs)
        self.image_typemap.update({'ptf' : PtfImage})

    def get_zeropoint_for(self,tractor_image):
        return ptf_zeropoint(tractor_image.imgfn)
   
    #def ccds_touching_wcs(self, wcs, **kwargs):
    #    '''PTF testing, continue even if no overlap with DECaLS bricks
    #    '''
    #    print('WARNING: ccds do not have to be overlapping with DECaLS bricks')
    #    T = self.get_ccds_readonly()
    #    I = ccds_touching_wcs(wcs, T, **kwargs)
    #    if len(I) == 0:
    #        return None
    #    T = T[I]
    #    return T

    def photometric_ccds(self, CCD):
        '''PTF testing process non-photometric ccds too'''
        print('WARNING: non-photometric ccds allowed')
        good = np.ones(len(CCD), bool)
        return np.flatnonzero(good)
    
    def get_ccds(self):
        '''
        Return SMALL CCD for testing: 2 rows
        '''
        fn = os.path.join(self.decals_dir, 'decals-ccds.fits')
        if not os.path.exists(fn):
            fn += '.gz'
        print('Reading CCDs from', fn)
        T = fits_table(fn)
        print('Got', len(T), 'CCDs')
        if 'ccdname' in T.columns():
            # "N4 " -> "N4"
            T.ccdname = np.array([s.strip() for s in T.ccdname])
        #T= T[ [np.where(T.filter == 'R')[0][0],np.where(T.filter == 'g')[0][0]] ] #1 R and 1 g band
        print('ccd ra= ',T.ra,'ccd dec= ',T.dec) 
        return T

#class SFDMap(object):
#    # These come from Schlafly & Finkbeiner, arxiv 1012.4804v2, Table 6, Rv=3.1
#    # but updated (and adding DES u) via email from Schlafly,
#    # decam-data thread from 11/13/2014, "New recommended SFD coefficients for DECam."
#    #
#    # The coefficients for the four WISE filters are derived from Fitzpatrick 1999,
#    # as recommended by Schafly & Finkbeiner, considered better than either the
#    # Cardelli et al 1989 curves or the newer Fitzpatrick & Massa 2009 NIR curve
#    # not vetted beyond 2 micron).
#    # These coefficients are A / E(B-V) = 0.184, 0.113, 0.0241, 0.00910. 
#    #
#    extinctions = {
#        'SDSS u': 4.239,
#        'DES u': 3.995,
#        'DES g': 3.214,
#        'DES r': 2.165,
#        'DES i': 1.592,
#        'DES z': 1.211,
#        'DES Y': 1.064,
#        'WISE W1': 0.184,
#        'WISE W2': 0.113,
#        'WISE W3': 0.0241,
#        'WISE W4': 0.00910,
#        'PTF g': 3.214,
#        'PTF R': 2.165,
#        }
#
#    def __init__(self, ngp_filename=None, sgp_filename=None, dustdir=None):
#        if dustdir is None:
#            dustdir = os.environ.get('DUST_DIR', None)
#        if dustdir is not None:
#            dustdir = os.path.join(dustdir, 'maps')
#        else:
#            dustdir = '.'
#            print 'Warning: $DUST_DIR not set; looking for SFD maps in current directory.'
#        if ngp_filename is None:
#            ngp_filename = os.path.join(dustdir, 'SFD_dust_4096_ngp.fits')
#        if sgp_filename is None:
#            sgp_filename = os.path.join(dustdir, 'SFD_dust_4096_sgp.fits')
#        if not os.path.exists(ngp_filename):
#            raise RuntimeError('Error: SFD map does not exist: %s' % ngp_filename)
#        if not os.path.exists(sgp_filename):
#            raise RuntimeError('Error: SFD map does not exist: %s' % sgp_filename)
#        self.north = fitsio.read(ngp_filename)
#        self.south = fitsio.read(sgp_filename)
#        self.northwcs = anwcs_t(ngp_filename, 0)
#        self.southwcs = anwcs_t(sgp_filename, 0)
#
#    @staticmethod
#    def bilinear_interp_nonzero(image, x, y):
#        H,W = image.shape
#        x0 = np.floor(x).astype(int)
#        y0 = np.floor(y).astype(int)
#        # Bilinear interpolate, but not outside the bounds (where ebv=0)
#        fx = np.clip(x - x0, 0., 1.)
#        ebvA = image[y0,x0]
#        ebvB = image[y0, np.clip(x0+1, 0, W-1)]
#        ebv1 = (1.-fx) * ebvA + fx * ebvB
#        ebv1[ebvA == 0] = ebvB[ebvA == 0]
#        ebv1[ebvB == 0] = ebvA[ebvB == 0]
#
#        ebvA = image[np.clip(y0+1, 0, H-1), x0]
#        ebvB = image[np.clip(y0+1, 0, H-1), np.clip(x0+1, 0, W-1)]
#        ebv2 = (1.-fx) * ebvA + fx * ebvB
#        ebv2[ebvA == 0] = ebvB[ebvA == 0]
#        ebv2[ebvB == 0] = ebvA[ebvB == 0]
#
#        fy = np.clip(y - y0, 0., 1.)
#        ebv = (1.-fy) * ebv1 + fy * ebv2
#        ebv[ebv1 == 0] = ebv2[ebv1 == 0]
#        ebv[ebv2 == 0] = ebv1[ebv2 == 0]
#        return ebv
#
#    def ebv(self, ra, dec):
#        l,b = radectolb(ra, dec)
#        ebv = np.zeros_like(l)
#        N = (b >= 0)
#        for wcs,image,cut in [(self.northwcs, self.north, N),
#                              (self.southwcs, self.south, np.logical_not(N))]:
#            # Our WCS routines are mis-named... the SFD WCSes convert 
#            #   X,Y <-> L,B.
#            if sum(cut) == 0:
#                continue
#            ok,x,y = wcs.radec2pixelxy(l[cut], b[cut])
#            assert(np.all(ok == 0))
#            H,W = image.shape
#            assert(np.all(x >= 0.5))
#            assert(np.all(x <= (W+0.5)))
#            assert(np.all(y >= 0.5))
#            assert(np.all(y <= (H+0.5)))
#            ebv[cut] = SFDMap.bilinear_interp_nonzero(image, x-1., y-1.)
#        return ebv
#
#    def extinction(self, filts, ra, dec, get_ebv=False):
#        ebv = self.ebv(ra, dec)
#        factors = np.array([SFDMap.extinctions[f] for f in filts])
#        rtn = factors[np.newaxis,:] * ebv[:,np.newaxis]
#        if get_ebv:
#            return ebv,rtn
#        return rtn

#def sed_matched_filters(bands):
#    '''
#    Determines which SED-matched filters to run based on the available
#    bands.
#
#    Returns
#    -------
#    SEDs : list of (name, sed) tuples
#    
#    '''
#    print('######## KJB sed_matched_filters ##########')
#    if len(bands) == 1:
#        return [(bands[0], (1.,))]
#
#    # single-band filters
#    SEDs = []
#    for i,band in enumerate(bands):
#        sed = np.zeros(len(bands))
#        sed[i] = 1.
#        SEDs.append((band, sed))
#
#    if len(bands) > 1:
#        flat = dict(g=1., r=1., z=1.,R=1.)
#        SEDs.append(('Flat', [flat[b] for b in bands]))
#        red = dict(g=2.5, r=1., z=0.4,R=1.)
#        SEDs.append(('Red', [red[b] for b in bands]))
#
#    return SEDs

#def get_rgb(imgs, bands, mnmx=None, arcsinh=None, scales=None):
#    '''
#    Given a list of images in the given bands, returns a scaled RGB
#    image.
#
#    *imgs*  a list of numpy arrays, all the same size, in nanomaggies
#    *bands* a list of strings, eg, ['g','r','z']
#    *mnmx*  = (min,max), values that will become black/white *after* scaling. Default is (-3,10)
#    *arcsinh* use nonlinear scaling as in SDSS
#    *scales*
#
#    Returns a (H,W,3) numpy array with values between 0 and 1.
#    '''
#    print('######## KJB get_rgb ##########')
#    bands = ''.join(bands)
#
#    grzscales = dict(g = (2, 0.0066),
#                      r = (1, 0.01),
#                      R = (1, 0.01),
#                      z = (0, 0.025),
#                      )
#
#    if scales is None:
#        if bands == 'grz':
#            scales = grzscales
#        elif bands == 'urz':
#            scales = dict(u = (2, 0.0066),
#                          r = (1, 0.01),
#                          z = (0, 0.025),
#                          )
#        elif bands == 'gri':
#            # scales = dict(g = (2, 0.004),
#            #               r = (1, 0.0066),
#            #               i = (0, 0.01),
#            #               )
#            scales = dict(g = (2, 0.002),
#                          r = (1, 0.004),
#                          i = (0, 0.005),
#                          )
#        else:
#            scales = grzscales
#        
#    h,w = imgs[0].shape
#    rgb = np.zeros((h,w,3), np.float32)
#    # Convert to ~ sigmas
#    for im,band in zip(imgs, bands):
#        plane,scale = scales[band]
#        rgb[:,:,plane] = (im / scale).astype(np.float32)
#
#    if mnmx is None:
#        mn,mx = -3, 10
#    else:
#        mn,mx = mnmx
#
#    if arcsinh is not None:
#        def nlmap(x):
#            return np.arcsinh(x * arcsinh) / np.sqrt(arcsinh)
#        rgb = nlmap(rgb)
#        mn = nlmap(mn)
#        mx = nlmap(mx)
#
#    rgb = (rgb - mn) / (mx - mn)
#    return np.clip(rgb, 0., 1.)
def make_dir(name):
    if not os.path.exists(name): os.makedirs(name)
    else: print('WARNING path exists: ',name)



def main():
    from runbrick import run_brick, get_parser, get_runbrick_kwargs
    
    parser = get_parser()
    opt = parser.parse_args()
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    print('Forcing --no-blacklist')
    opt.blacklist = False

    kwargs = get_runbrick_kwargs(opt)
    if kwargs in [-1,0]:
        return kwargs

    decals = PtfDecals(decals_dir=opt.decals_dir)
    kwargs['decals'] = decals
    
    #kwargs['bands'] = ['g','r','z','g_PTF','R_PTF']
    make_dir(opt.outdir)
    kwargs['pixscale'] = 1.01
    if kwargs['forceAll']: kwargs['writePickles']= False
    else: kwargs['writePickles']= True

    # runbrick...
    print("before call run_brick")
    run_brick(opt.brick, **kwargs)
    return 0
    
if __name__ == '__main__':
    import sys
    sys.exit(main())
