import os
import numpy as np
from legacypipe.hsc import HscImage
from legacypipe.image import LegacySurveyImage
import tractor

class LsstImage(HscImage):
    @classmethod
    def get_nominal_pixscale(cls):
        return 0.2

    def __init__(self, survey, ccd, image_fn=None, image_hdu=0,
                 camera_setup=False, **kwargs):
        super().__init__(survey, ccd, image_fn=image_fn, image_hdu=image_hdu, **kwargs)
        if camera_setup:
            return

        # Nominal zeropoints
        # nJy...
        #zpt = 31.4 # or 31.4 - 2.5*log10(30) ~= 27.707
        zpt = 27.707
        self.zp0 = dict(
            u = zpt,
            g = zpt,
            r = zpt,
            i = zpt,
            z = zpt,
            y = zpt,
        )

        self.k_ext.update(u=0.63) # from decam.py
        
        self.set_calib_filenames()

        # Try grabbing fwhm from PSFEx file, if it exists.
        # if hasattr(self, 'fwhm') and not np.isfinite(self.fwhm):
        #     try:
        #         # PSF model file may not have been created yet...
        #         self.fwhm = self.get_fwhm(None, None)
        #     except:
        #         pass

    def colorterm_ps1_to_observed(self, ps1stars, band):
        """ps1stars: ps1.median 2D array of median mag for each band"""
        from legacypipe.ps1cat import ps1_to_lsst_comcam
        return ps1_to_lsst_comcam(ps1stars, band)

    def set_calib_filenames(self):
        # Calib filenames
        calibdir = self.survey.get_calib_dir()
        imgdir = os.path.dirname(self.image_filename)
        basename = self.get_base_name()
        self.name = basename
        self.sefn         = os.path.join(calibdir, 'se',           imgdir, basename + '-se.fits')
        self.psffn        = os.path.join(calibdir, 'psfex-single', imgdir, basename + '-psfex.fits')
        self.skyfn        = None
        self.merged_psffn = None
        self.merged_skyfn = None
        self.old_merged_skyfns = []
        self.old_merged_psffns = []
        # not used by this code -- here for the sake of legacyzpts/merge_calibs.py
        self.old_single_psffn = None
        self.old_single_skyfn = None

    #def has_astrometric_calibration(self, ccd):
    #    return True

    def get_zeropoint(self, primhdr, hdr):
        # Have to calibrate!
        return None

    def get_ha_deg(self, primhdr):
        ## not sure what units this is in...
        ## HASTART =     2.10076597268751 / [HH:MM:SS] Telescope hour angle at start
        return (primhdr['HASTART'] + primhdr['HAEND'])/2.

    def get_mjd(self, primhdr):
        return primhdr['MJD-BEG']

    def get_band(self, primhdr):
        band = primhdr['FILTBAND']
        band = band.split()[0]
        return band

    def get_propid(self, primhdr):
        return primhdr.get('PROGRAM', '')

    def get_ccdname(self, primhdr, hdr):
        return primhdr['DETNAME'].strip().upper()

    def get_fwhm(self, primhdr, imghdr):
        psf = None
        try:
            # PSF model file may not have been created yet...
            psf = self.read_psf_model(0., 0., pixPsf=True)
        except:
            print('get_fwhm: Failed read PSF model:')
            import traceback
            traceback.print_exc()
            pass
        if psf is None:
            print("HACK - no FWHM readily available")
            return np.nan
        fwhm = psf.fwhm
        return fwhm

    def get_radec_bore(self, primhdr, hdr):
        # miracle of miracles, they just put in decimal degrees
        return primhdr['RA'], primhdr['DEC']

    # There is a CCDGAIN ("rough guess") = 1.0,
    # or HIERARCH LSST ISR GAIN C10 = 1.64786902096928
    # for C0..C15 (oh but with some missing..?)
    def get_gain(self, primhdr, hdr):
        vals = []
        for i in range(16):
            key = 'LSST ISR GAIN C%02i'%i
            if key in primhdr:
                vals.append(primhdr[key])
        if len(vals) == 0:
            return primhdr['CCDGAIN']
        return np.median(vals)

    def estimate_sky(self, img, invvar, dq, primhdr, imghdr):
        return LegacySurveyImage.estimate_sky(self, img, invvar, dq, primhdr, imghdr)

    # DP1 ComCam images: PiffPSF is stored as a pickle in a variable-length binary table.
    # Appears to be Piff version 1.5.0
    # but that doesn't build at NERSC, so just fall back to PsfEx.
    def read_psf_model(self, *args, **kwargs):
        return LegacySurveyImage.read_psf_model(self, *args, **kwargs)

# DP2 deepCoadd images
class LsstCoaddImage(LsstImage):
    def __init__(self, *args, **kwargs):
        self.psf = None
        super().__init__(*args, **kwargs)

        # Nominal zeropoints
        # nJy...
        zpt = 31.4
        self.zp0 = dict(
            u = zpt,
            g = zpt,
            r = zpt,
            i = zpt,
            z = zpt,
            y = zpt,
        )
    
    # Like HSC, we're going to use the calibrations built into the deepCoadd files.
    # hence no external calib filenames.
    def set_calib_filenames(self):
        self.sefn = None
        self.psffn = None
        basename = self.get_base_name()
        self.name = basename

    # LSST deepCoadds: defined on tracts & patches.  There is overlap between tracts
    # and also between patches.  The patches overlaps are simple: they're just 150 pix.
    # This conveniently matches the coadd cell size.  Here we're going to hardcode the
    # fact that the coadds are 22 x 22 cells = 3300 x 3300 pixels.
    def get_good_image_subregion(self):
        return 150, 3150, 150, 3150

    def get_band(self, primhdr):
        # HIERARCH LSST BUTLER DATAID BAND = 'g      '
        band = primhdr['LSST BUTLER DATAID BAND']
        band = band.strip().split()[0]
        return band

    def get_expnum(self, primhdr):
        # Tracts are bigger than Patches
        # 10x10 patches within each Tract
        # reserve 3 digits just to be safe
        tract = primhdr['LSST BUTLER DATAID TRACT'] # = 7032
        patch = primhdr['LSST BUTLER DATAID PATCH'] # = 80
        bandnum = dict(u=0, g=1, r=2, i=3, z=4, y=5)[self.get_band(primhdr)]
        return 10 * (tract * 1000 + patch) + bandnum

    def get_mjd(self, primhdr):
        from astrometry.util.starutil_numpy import datetomjd
        d = self.get_date(primhdr)
        return datetomjd(d)

    def get_date(self, primhdr):
        from datetime import datetime
        # HACK - this is just the date the COADD FILE was WRITTEN
        date = primhdr['DATE']
        # DATE    = '2026-06-18T13:13:43.519' / UTC date this HDU was written.
        return datetime.strptime(date[:19], "%Y-%m-%dT%H:%M:%S")

    def get_ha_deg(self, primhdr):
        # HACK
        return 0.0

    def get_camera(self, primhdr):
        # hack
        return 'lsstcoadd'

    def get_ccdname(self, primhdr, hdr):
        return ''

    def read_psf_model(self, x0, y0,
                       gaussPsf=False, pixPsf=False, hybridPsf=False,
                       normalizePsf=False, old_calibs_ok=False,
                       psf_sigma=1., w=0, h=0):
        if gaussPsf:
            return LegacySurveyImage.read_psf_model(self, x0,y0, gaussPsf=True, psf_sigma=psf_sigma)

        if self.psf is not None:
            return self.psf

        import tempfile
        import fitsio
        # piecewise constant pixelized PSF model

        # ugh, fitsio can't read 4-d compressed images...
        # but, funpack can handle them, and then fitsio can read an uncompressed 4-d image.
        F = self.read_image_fits()
        psf_hdu = -1
        for i,f in enumerate(F):
            if f.get_extname() == 'PSF':
                psf_hdu = i
                break
        assert(psf_hdu != -1)

        f,tmppsffn = tempfile.mkstemp(suffix='.fits')
        os.close(f)
        os.remove(tmppsffn)
        cmd = 'funpack -E %i -O %s %s' % (psf_hdu, tmppsffn, self.imgfn)
        print('Funpack command:', cmd)
        rtn = os.system(cmd)
        assert(rtn == 0)
        psf_cube = fitsio.read(tmppsffn)
        os.remove(tmppsffn)
        print('Read PSF cube:', psf_cube.shape)
        # MAGIC number 150 = LSST deepCoadd cell size, in pixels
        psf = PiecewiseConstantPixelizedPsf(psf_cube, 150, x0, y0)
        self.psf = psf
        return psf

    def get_radec_bore(self, primhdr, hdr):
        wcs = self.get_wcs(hdr=hdr)
        return wcs.radec_center()

    def get_airmass(self, primhdr, imghdr, ra, dec):
        # HACK
        #return None
        return 1.
    
    def get_cd_matrix(self, primhdr, hdr):
        # HACK - probably needs a * CDELT[12]?  But those are 1.0 in deepCoadds
        return hdr['PC1_1'], 0., 0., hdr['PC2_2']

    def get_exptime(self, primhdr):
        # HACK...
        return 1.

    def get_gain(self, primhdr, hdr):
        # HACK
        return 1.

    def remap_dq(self, dq, header, slc):
        from legacypipe.bits import DQ_BITS
        new_dq = np.zeros(dq.shape, np.int16)
        print('Remapping LSST bitmasks')
        masks = {}
        for i in range(32):
            key = 'MSKN%04i' % i
            if not key in header:
                break
            bitval = 'MSKM%04i' % i
            if not bitval in header:
                break
            masks[header[key]] = int(header[bitval])
        def val(name):
            return masks.get(name, 0)

        # The coadd bitmasks are pretty weird: some are ORs of the inputs

        # 'NO_DATA '
        # 'No data was available for this pixel.'

        # 'INTERPOLATED'
        # 'Pixel value is the result of interpolating nearby good pixels.'

        #  'COSMIC_RAY'
        # 'A cosmic ray affected this pixel on at least one input image (and &'
        # 'was interpolated).'

        # 'SATURATED'
        # 'More than 10% of the potential input visits had a saturated pixel &'
        # 'at this location (''potential'' because saturated pixel values are &'
        # 'not actually propagated to the coadd). SATURATED always implies &'
        # 'REJECTED, and is often a reason for NO_DATA.'

        # 'DETECTION_EDGE'
        # 'Pixel was too close to the edge of the patch to be considered for &'
        # 'detection, due to the finite size of the detection kernel.'

        # 'CLIPPED '
        # 'Region was identified as a probable artifact when comparing &'
        # 'multiple single-visit warps. CLIPPED always implies REJECTED.'

        # 'REJECTED'
        # 'At least one input visit was left out of the coadd for this pixel &'
        # 'due to masking. REJECTED always implies INEXACT_PSF.'

        # 'DETECTED'
        # 'Pixel was part of a detected source.'

        # 'INEXACT_PSF'
        # 'The set of visits contributing to this pixel differs from the set &'
        # 'of visits contributing to the PSF model for its cell.'

        no = val('NO_DATA') | val('INTERPOLATED')
        new_dq |= DQ_BITS['badpix'] * ((no & bits) != 0)

        # We only want to mark pixels SATUR if they end up having NO_DATA because _all_
        # the exposures are saturated.  This isn't exactly what the condition below does!
        sat = val('SATURATED')
        new_dq |= DQ_BITS['satur'] * np.logical_and((no & bits) != 0, (sat & bits) != 0)

        # bits = val('INTERPOLATED')
        # new_dq |= DQ_BITS['interp'] * ((dq & bits) != 0)
        # print('Interp:', np.sum((dq & bits) != 0), 'pixels have mask val 0x%x set' % bits)
        # 
        # bits = val('COSMIC_RAY')
        # new_dq |= DQ_BITS['cr'] * ((dq & bits) != 0)
        # print('CR:', np.sum((dq & bits) != 0), 'pixels have mask val 0x%x set' % bits)
        # 
        # bits = val('SATURATED')
        # new_dq |= DQ_BITS['satur' ] * ((dq & bits) != 0)
        # print('SATUR:', np.sum((dq & bits) != 0), 'pixels have mask val 0x%x set' % bits)
        # 
        # bits = val('DETECTION_EDGE')
        # new_dq |= DQ_BITS['edge'] * ((dq & bits) != 0)
        # print('Edge:', np.sum((dq & bits) != 0), 'pixels have mask val 0x%x set' % bits)

        return new_dq
    
from tractor.psf import PixelizedPSF, HybridPixelizedPSF, HybridPSF

class PiecewiseConstantPixelizedPsf(PixelizedPSF, HybridPSF):
#class PiecewiseConstantPixelizedPsf(HybridPixelizedPSF):
    '''
    A PSF class for the Rubin/LSST DeepCoadd PSF model, which has a
    constant PSF in each 150x150-pixel grid cell.
    '''
    def __init__(self, img_grid, grid_size, x0=0, y0=0, fwhm=None, fwhm_grid=None):
        '''
        img_grid: (grid_h,grid_w, psf_h,psf_w) data cube
        grid_size: integer, scalar: size in pixels of the cells where PSFs are defined,
            eg 150 for Rubin DP2
        '''
        # H x W x gridy x gridx
        assert(len(img_grid.shape) == 4)
        self.img_grid = img_grid
        self.grid_size = grid_size
        self.gh, self.gw, self.ph, self.pw = img_grid.shape
        self.x0 = x0
        self.y0 = y0

        # Here, I'm just using a single round Gaussian for the PSF approximation...
        # could do the general elliptical or multi-component instead...
        
        if fwhm is None:
            # compute FWHM from just averaging all the PSF images!
            # (the catch: some cells can be all NaNs!!)
            avgpsf = np.zeros((self.ph, self.pw))
            ngood = 0
            for i in range(self.gh):
                for j in range(self.gw):
                    if np.all(np.isfinite(img_grid[i,j,:,:])):
                        avgpsf += img_grid[i,j,:,:]
                        ngood += 1
            assert(ngood > 0)
            avgpsf /= ngood
            self.avgpsf = avgpsf
            #avgpsf = np.mean(img_grid, axis=(0,1))
            print('average psf img:', avgpsf.shape, 'sum', np.sum(avgpsf))
            fwhm_avg = fit_circular_gaussian(avgpsf)
            print('Gaussian-fit FWHM:', fwhm_avg)
            self.fwhm = fwhm_avg
        else:
            self.fwhm = fwhm

        if fwhm_grid is None:
            # Also precompute the Gaussian FWHM per cell.
            self.fwhm_grid = np.zeros((self.gh, self.gw))
            for i in range(self.gh):
                for j in range(self.gw):
                    if np.all(np.isfinite(img_grid[i,j,:,:])):
                        fwhm = fit_circular_gaussian(img_grid[i,j,:,:])
                    else:
                        fwhm = fwhm_avg
                    self.fwhm_grid[i,j] = fwhm
        else:
            self.fwhm_grid = fwhm_grid

        # call superclass constructor with the PSF in grid 0,0 to set parameters like .sampling
        super().__init__(self.img_grid[0, 0, :, :])

    def getMixtureOfGaussians(self, px=None, py=None):
        from tractor import mixture_profiles as mp
        if px is None or py is None:
            # image-wide average
            fwhm = self.fwhm
        else:
            cell_y, cell_x = self._getCell(px, py)
            fwhm = self.fwhm_grid[cell_y, cell_x]
        sigma = fwhm * 2.35
        gauss = tractor.NCircularGaussianPSF([sigma], [1.])
        variance = np.eye(2).reshape((1,2,2)) * sigma**2
        return mp.MixtureOfGaussians(np.ones(1), np.zeros((1,2)), variance)

    def __str__(self):
        return 'PiecewiseConstantPixelizedPsf'

    @property
    def shape(self):
        return (self.ph, self.pw)

    def copy(self):
        return self.__class__(self.img_grid.copy(), self.grid_size, x0=self.x0, y0=self.y0,
                              fwhm=self.fwhm, fwhm_grid=self.fwhm_grid.copy())

    def getShifted(self, x0, y0):
        return self.__class__(self.img_grid.copy(), self.grid_size, x0=self.x0 + x0, y0=self.y0 + y0,
                              fwhm=self.fwhm, fwhm_grid=self.fwhm_grid.copy())

    def constantPsfAt(self, x, y):
        #return PixelizedPSF(self.getImage(x, y))
        #psf = tractor.NCircularGaussianPSF([2.5], [1.])
        pix = self.getImage(x, y)
        pixpsf = PixelizedPSF(pix)
        cell_y, cell_x = self._getCell(x, y)
        fwhm = self.fwhm_grid[cell_y, cell_x]
        sigma = fwhm * 2.35
        gauss = tractor.NCircularGaussianPSF([sigma], [1.])
        return HybridPixelizedPSF(pixpsf, gauss=gauss)

    def _getCell(self, px, py):
        cell_x = int((px + self.x0) // self.grid_size)
        cell_y = int((py + self.y0) // self.grid_size)
        #assert(cell_x >= 0)
        #assert(cell_y >= 0)
        #assert(cell_y < self.gh)
        #assert(cell_x < self.gw)
        # clip
        cell_x = max(0, min(cell_x, self.gw-1))
        cell_y = max(0, min(cell_y, self.gh-1))
        return cell_y,cell_x

    def getImage(self, px, py):
        cell_y, cell_x = self._getCell(px, py)
        psf = self.img_grid[cell_y, cell_x, :, :]
        if np.all(np.isfinite(psf)):
            return psf
        print('PSF in cell [%i,%i] has NaNs - returning the image average PSF instead!' % (cell_y, cell_x))
        return self.avgpsf



# def estimate_fwhm_1d(y):
#     # Hackily, assume a spline interpolant and return the fit width of the half-max values
#     from scipy.interpolatio import CubicSpline
#     from scipy.optimize import minimize_scalar
#     x = np.arange(len(y))
#     spl = CubicSpline(x, y)
# 
#     # find peak value
#     deriv = spl.derivative()
#     sol = deriv.solve()
#     print('PSF derivative = 0: at', sol)
    
def fit_circular_gaussian(psfimg):
    # Assume centered PSF image in sky-subtracted image.
    import tractor
    h,w = psfimg.shape
    cx,cy = (w-1)/2, (h-1)/2
    flux = np.sum(psfimg)
    # sigmas, weights.  sigma=2.5 is fwhm=5.9 pix is 1.2 arcsec in Rubin
    psf = tractor.NCircularGaussianPSF([2.5], [1.])
    tim = tractor.Image(data=psfimg, inverr=np.ones_like(psfimg), psf=psf)
    src = tractor.PointSource(tractor.PixPos(cx, cy), tractor.Flux(flux))
    tr = tractor.Tractor([tim], [src])
    # Fit PSF width + Catalog entries
    tim.freezeAllBut('psf')
    psf.freezeAllBut('sigmas')
    optargs = dict(priors=False, shared_params=False)
    tr.optimize_loop(**optargs)
    #fwhms[i] = psf.sigmas[0] * 2.35 * pixsc
    fit_fwhm = psf.sigmas[0] * 2.35
    return fit_fwhm

    


    
