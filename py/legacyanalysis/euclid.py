from __future__ import print_function
import sys
import os

# YUCK!  scinet bonkers python setup
paths = os.environ['PYTHONPATH']
sys.path = paths.split(':') + sys.path
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import pylab as plt
from glob import glob

import fitsio
import astropy

from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.util import wcs_pv2sip_hdr
from astrometry.util.plotutils import PlotSequence, plothist, loghist
from astrometry.util.ttime import Time, MemMeas
from astrometry.util.multiproc import multiproc

from legacypipe.runbrick import run_brick, rgbkwargs, rgbkwargs_resid
from legacypipe.common import LegacySurveyData, imsave_jpeg, get_rgb
from legacypipe.image import LegacySurveyImage
from legacypipe.desi_common import read_fits_catalog

from tractor.sky import ConstantSky
from tractor.sfd import SFDMap
from tractor import Tractor, NanoMaggies
from tractor.galaxy import disable_galaxy_cache
from tractor.ellipses import EllipseE

rgbscales = dict(I=(0, 0.01))
rgbkwargs      .update(scales=rgbscales)
rgbkwargs_resid.update(scales=rgbscales)

SFDMap.extinctions.update({'DES I': 1.592})

allbands = 'I'


def make_zeropoints():
    C = fits_table()
    C.image_filename = []
    C.image_hdu = []
    C.camera = []
    C.expnum = []
    C.filter = []
    C.exptime = []
    C.crpix1 = []
    C.crpix2 = []
    C.crval1 = []
    C.crval2 = []
    C.cd1_1 = []
    C.cd1_2 = []
    C.cd2_1 = []
    C.cd2_2 = []
    C.width = []
    C.height = []
    C.ra = []
    C.dec = []
    C.ccdzpt = []
    C.ccdname = []
    C.ccdraoff = []
    C.ccddecoff = []
    C.fwhm = []
    C.propid = []
    C.mjd_obs = []
    
    base = 'euclid/images/'
    for iband,band in enumerate(['Y','J','H']):
        fn = base + 'vista/UVISTA_%s_DR1_CFHT.fits' % band
        hdu = 0
        hdr = fitsio.read_header(fn, ext=hdu)
        C.image_hdu.append(hdu)
        C.image_filename.append(fn.replace(base,''))
        C.camera.append('vista')
        expnum = iband + 1
        C.expnum.append(expnum)
        C.filter.append(band)
        C.exptime.append(hdr['EXPTIME'])
        C.ccdzpt.append(hdr['MAGZEROP'])
        for k in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2','CD1_1','CD1_2','CD2_1','CD2_2']:
            C.get(k.lower()).append(hdr[k])
        C.width.append(hdr['NAXIS1'])
        C.height.append(hdr['NAXIS2'])
            
        wcs = wcs_pv2sip_hdr(hdr)
        rc,dc = wcs.radec_center()
        C.ra.append(rc)
        C.dec.append(dc)

        psffn = fn.replace('.fits', '_psfex.psf')
        psfhdr = fitsio.read_header(psffn, ext=1)
        fwhm = psfhdr['PSF_FWHM']
        C.fwhm.append(fwhm)
        
        C.ccdname.append('0')
        C.ccdraoff.append(0.)
        C.ccddecoff.append(0.)
        C.propid.append('0')
        C.mjd_obs.append(0.)
        
    C.to_np_arrays()
    fn = 'euclid/survey-ccds-vista.fits.gz'
    C.writeto(fn)
    print('Wrote', fn)

    return

    C = fits_table()
    C.image_filename = []
    C.image_hdu = []
    C.camera = []
    C.expnum = []
    C.filter = []
    C.exptime = []
    C.crpix1 = []
    C.crpix2 = []
    C.crval1 = []
    C.crval2 = []
    C.cd1_1 = []
    C.cd1_2 = []
    C.cd2_1 = []
    C.cd2_2 = []
    C.width = []
    C.height = []
    C.ra = []
    C.dec = []
    C.ccdzpt = []
    C.ccdname = []
    C.ccdraoff = []
    C.ccddecoff = []
    C.fwhm = []
    C.propid = []
    C.mjd_obs = []
    
    base = 'euclid/images/'
    fns = glob(base + 'acs-vis/*_sci.VISRES.fits')
    fns.sort()
    for fn in fns:
        hdu = 0
        hdr = fitsio.read_header(fn, ext=hdu)
        C.image_hdu.append(hdu)
        C.image_filename.append(fn.replace(base,''))
        C.camera.append('acs-vis')
        words = fn.split('_')
        expnum = int(words[-2], 10)
        C.expnum.append(expnum)
        filt = words[-4]
        C.filter.append(filt)
        C.exptime.append(hdr['EXPTIME'])
        C.ccdzpt.append(hdr['PHOTZPT'])

        for k in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2','CD1_1','CD1_2','CD2_1','CD2_2']:
            C.get(k.lower()).append(hdr[k])
        C.width.append(hdr['NAXIS1'])
        C.height.append(hdr['NAXIS2'])
            
        wcs = wcs_pv2sip_hdr(hdr)
        rc,dc = wcs.radec_center()
        C.ra.append(rc)
        C.dec.append(dc)

        psffn = fn.replace('_sci.VISRES.fits', '_sci.VISRES_psfex.psf')
        psfhdr = fitsio.read_header(psffn, ext=1)
        fwhm = psfhdr['PSF_FWHM']
        
        C.ccdname.append('0')
        C.ccdraoff.append(0.)
        C.ccddecoff.append(0.)
        #C.fwhm.append(0.18 / 0.1)
        C.fwhm.append(fwhm)
        C.propid.append('0')
        C.mjd_obs.append(0.)
        
    C.to_np_arrays()
    fn = 'euclid/survey-ccds-acsvis.fits.gz'
    C.writeto(fn)
    print('Wrote', fn)

    C = fits_table()
    C.image_filename = []
    C.image_hdu = []
    C.camera = []
    C.expnum = []
    C.filter = []
    C.exptime = []
    C.crpix1 = []
    C.crpix2 = []
    C.crval1 = []
    C.crval2 = []
    C.cd1_1 = []
    C.cd1_2 = []
    C.cd2_1 = []
    C.cd2_2 = []
    C.width = []
    C.height = []
    C.ra = []
    C.dec = []
    C.ccdzpt = []
    C.ccdname = []
    C.ccdraoff = []
    C.ccddecoff = []
    C.fwhm = []
    C.propid = []
    C.mjd_obs = []
    fns = glob(base + 'megacam/*p.fits')
    fns.sort()

    for fn in fns:
        psffn = fn.replace('p.fits', 'p_psfex.psf')
        if not os.path.exists(psffn):
            print('Missing:', psffn)
            sys.exit(-1)
        wtfn = fn.replace('p.fits', 'p_weight.fits')
        if not os.path.exists(wtfn):
            print('Missing:', wtfn)
            sys.exit(-1)
        dqfn = fn.replace('p.fits', 'p_flag.fits')
        if not os.path.exists(dqfn):
            print('Missing:', dqfn)
            sys.exit(-1)


    for fn in fns:
        F = fitsio.FITS(fn)
        print(len(F), 'FITS extensions in', fn)
        print(F)
        phdr = F[0].read_header()
        expnum = phdr['EXPNUM']
        filt = phdr['FILTER']
        print('Filter', filt)
        filt = filt[0]
        exptime = phdr['EXPTIME']

        psffn = fn.replace('p.fits', 'p_psfex.psf')
        PF = fitsio.FITS(psffn)

        for hdu in range(1, len(F)):
            print('Reading header', fn, 'hdu', hdu)
            #hdr = fitsio.read_header(fn, ext=hdu)
            hdr = F[hdu].read_header()
            C.image_filename.append(fn.replace(base,''))
            C.image_hdu.append(hdu)
            C.camera.append('megacam')
            C.expnum.append(expnum)
            C.filter.append(filt)
            C.exptime.append(exptime)
            C.ccdzpt.append(hdr['PHOTZPT'])

            for k in ['CRPIX1','CRPIX2','CRVAL1','CRVAL2','CD1_1','CD1_2','CD2_1','CD2_2']:
                C.get(k.lower()).append(hdr[k])
            C.width.append(hdr['NAXIS1'])
            C.height.append(hdr['NAXIS2'])
            
            wcs = wcs_pv2sip_hdr(hdr)
            rc,dc = wcs.radec_center()
            C.ra.append(rc)
            C.dec.append(dc)

            print('Reading PSF from', psffn, 'hdu', hdu)
            #psfhdr = fitsio.read_header(psffn, ext=hdu)
            psfhdr = PF[hdu].read_header()
            fwhm = psfhdr['PSF_FWHM']
        
            C.ccdname.append(hdr['EXTNAME'])
            C.ccdraoff.append(0.)
            C.ccddecoff.append(0.)
            C.fwhm.append(fwhm)
            C.propid.append('0')
            C.mjd_obs.append(hdr['MJD-OBS'])
        
    C.to_np_arrays()
    fn = 'euclid/survey-ccds-megacam.fits.gz'
    C.writeto(fn)
    print('Wrote', fn)


class AcsVisImage(LegacySurveyImage):
    def __init__(self, *args, **kwargs):
        super(AcsVisImage, self).__init__(*args, **kwargs)

        self.psffn = self.imgfn.replace('_sci.VISRES.fits', '_sci.VISRES_psfex.psf')
        assert(self.psffn != self.imgfn)

        self.wtfn = self.imgfn.replace('_sci', '_wht')
        assert(self.wtfn != self.imgfn)

        self.dqfn = self.imgfn.replace('_sci', '_flg')
        assert(self.dqfn != self.imgfn)
        
        self.name = 'AcsVisImage: expnum %i' % self.expnum

        self.dq_saturation_bits = 0
        

    def get_wcs(self):
        # Make sure the PV-to-SIP converter samples enough points for small
        # images
        stepsize = 0
        if min(self.width, self.height) < 600:
            stepsize = min(self.width, self.height) / 10.;
        hdr = self.read_image_header()
        wcs = wcs_pv2sip_hdr(hdr, stepsize=stepsize)
        return wcs

    def read_invvar(self, clip=True, **kwargs):
        '''
        Reads the inverse-variance (weight) map image.
        '''
        #return self._read_fits(self.wtfn, self.hdu, **kwargs)

        img = self.read_image(**kwargs)
        # # Estimate per-pixel noise via Blanton's 5-pixel MAD
        slice1 = (slice(0,-5,10),slice(0,-5,10))
        slice2 = (slice(5,None,10),slice(5,None,10))
        mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
        sig1 = 1.4826 * mad / np.sqrt(2.)
        print('sig1 estimate:', sig1)
        invvar = np.ones_like(img) / sig1**2
        return invvar
        
    def read_dq(self, **kwargs):
        '''
        Reads the Data Quality (DQ) mask image.
        '''
        print('Reading data quality image', self.dqfn, 'ext', self.hdu)
        dq = self._read_fits(self.dqfn, self.hdu, **kwargs)
        return dq
        
    def read_sky_model(self, splinesky=False, slc=None, **kwargs):
        sky = ConstantSky(0.)
        return sky

class MegacamImage(LegacySurveyImage):
    def __init__(self, *args, **kwargs):
        super(MegacamImage, self).__init__(*args, **kwargs)

        #self.psffn = self.imgfn.replace('p.fits', 'p_psfex.psf')
        self.psffn = self.imgfn.replace('.fits', '_psfex.psf')
        assert(self.psffn != self.imgfn)

        #self.wtfn = self.imgfn.replace('p.fits', 'p_weight.fits')
        self.wtfn = self.imgfn.replace('.fits', '_weight.fits')
        assert(self.wtfn != self.imgfn)

        #self.dqfn = self.imgfn.replace('p.fits', 'p_flag.fits')
        self.dqfn = self.imgfn.replace('.fits', '_flag.fits')
        assert(self.dqfn != self.imgfn)
        
        self.name = 'MegacamImage: %i-%s' % (self.expnum, self.ccdname)
        self.dq_saturation_bits = 0
        
    def get_wcs(self):
        # Make sure the PV-to-SIP converter samples enough points for small
        # images
        stepsize = 0
        if min(self.width, self.height) < 600:
            stepsize = min(self.width, self.height) / 10.;
        hdr = self.read_image_header()
        wcs = wcs_pv2sip_hdr(hdr, stepsize=stepsize)
        return wcs

    def read_image(self, header=None, **kwargs):
        print('Reading image from', self.imgfn, 'hdu', self.hdu)
        R = self._read_fits(self.imgfn, self.hdu, header=header, **kwargs)
        if header:
            img,header = R
        else:
            img = R
        img = img.astype(np.float32)
        if header:
            return img,header
        return img
    
    def read_invvar(self, clip=True, **kwargs):
        '''
        Reads the inverse-variance (weight) map image.
        '''
        #return self._read_fits(self.wtfn, self.hdu, **kwargs)

        img = self.read_image(**kwargs)
        # # Estimate per-pixel noise via Blanton's 5-pixel MAD
        slice1 = (slice(0,-5,10),slice(0,-5,10))
        slice2 = (slice(5,None,10),slice(5,None,10))
        mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
        sig1 = 1.4826 * mad / np.sqrt(2.)
        print('sig1 estimate:', sig1)
        invvar = np.ones_like(img) / sig1**2
        return invvar
        
    def read_dq(self, **kwargs):
        '''
        Reads the Data Quality (DQ) mask image.
        '''
        print('Reading data quality image', self.dqfn, 'ext', self.hdu)
        dq = self._read_fits(self.dqfn, self.hdu, **kwargs)
        print('Got', dq.dtype, dq.shape)
        return dq
        
    def read_sky_model(self, splinesky=False, slc=None, **kwargs):
        sky = ConstantSky(0.)
        return sky


class VistaImage(MegacamImage):
    def __init__(self, *args, **kwargs):
        super(VistaImage, self).__init__(*args, **kwargs)
        self.name = 'VistaImage: %i-%s %s' % (self.expnum, self.ccdname, self.band)
        self.wtfn = self.wtfn.replace('_weight.fits', '.weight.fits')
        self.dqfn = self.dqfn.replace('_flag.fits', '.flg.fits')
        assert(self.dqfn != self.imgfn)
        assert(self.wtfn != self.imgfn)

    def read_invvar(self, clip=True, **kwargs):
        '''
        Reads the inverse-variance (weight) map image.
        '''
        print('Reading weight map', self.wtfn, 'ext', self.hdu)
        iv = self._read_fits(self.wtfn, self.hdu, **kwargs)
        print('Read', iv.shape, iv.dtype)
        print('Range', iv.min(), iv.max(), 'median', np.median(iv))
        return iv

    def read_dq(self, **kwargs):
        '''
        Reads the Data Quality (DQ) mask image.
        '''
        print('Reading data quality image', self.dqfn, 'ext', self.hdu)
        dq = self._read_fits(self.dqfn, self.hdu, **kwargs)
        print('Got', dq.dtype, dq.shape)
        print('Blanking out bit 1...')
        dq -= (dq & 1)
        return dq


# Hackily defined RA,Dec values that split the ACS 7x7 tiling into
# distinct 'bricks'
rasplits = np.array([ 149.72716429, 149.89394223,  150.06073352,
                      150.22752888,  150.39431559, 150.56110037])
                          
decsplits = np.array([ 1.79290318,  1.95956698,  2.12623253,
                       2.2929002,   2.45956215,  2.62621403])
    
def read_acs_catalogs():
    # Read all ACS catalogs
    mfn = 'euclid-out/merged-catalog.fits'
    if not os.path.exists(mfn):
        TT = []
        fns = glob('euclid-out/tractor/*/tractor-*.fits')
        for fn in fns:
            T = fits_table(fn)
            print(len(T), 'from', fn)

            mra  = np.median(T.ra)
            mdec = np.median(T.dec)

            print(np.sum(T.brick_primary), 'PRIMARY')
            I = np.flatnonzero(rasplits > mra)
            if len(I) > 0:
                T.brick_primary &= (T.ra < rasplits[I[0]])
                print(np.sum(T.brick_primary), 'PRIMARY after RA high cut')
            I = np.flatnonzero(rasplits < mra)
            if len(I) > 0:
                T.brick_primary &= (T.ra >= rasplits[I[-1]])
                print(np.sum(T.brick_primary), 'PRIMARY after RA low cut')
            I = np.flatnonzero(decsplits > mdec)
            if len(I) > 0:
                T.brick_primary &= (T.dec < decsplits[I[0]])
                print(np.sum(T.brick_primary), 'PRIMARY after DEC high cut')
            I = np.flatnonzero(decsplits < mdec)
            if len(I) > 0:
                T.brick_primary &= (T.dec >= decsplits[I[-1]])
                print(np.sum(T.brick_primary), 'PRIMARY after DEC low cut')

            TT.append(T)
        T = merge_tables(TT)
        del TT
        T.writeto(mfn)
    else:
        T = fits_table(mfn)
    return T

def get_survey():
    survey = LegacySurveyData(survey_dir='euclid', output_dir='euclid-out')
    survey.image_typemap['acs-vis'] = AcsVisImage
    survey.image_typemap['megacam'] = MegacamImage
    survey.image_typemap['vista'] = VistaImage
    return survey

def get_exposures_in_list(fn):
    expnums = []
    for line in open(fn,'r').readlines():
        words = line.split()
        e = int(words[0], 10)
        expnums.append(e)
    return expnums

def download():
    fns = glob('euclid/images/megacam/lists/*.lst')
    expnums = set()
    for fn in fns:
        expnums.update(get_exposures_in_list(fn))
    print(len(expnums), 'unique exposure numbers')
    for expnum in expnums:
        need = []
        fn = 'euclid/images/megacam/%ip.fits' % expnum
        if not os.path.exists(fn):
            print('Missing:', fn)
            need.append(fn)
        psffn = fn.replace('p.fits', 'p_psfex.psf')
        if not os.path.exists(psffn):
            print('Missing:', psffn)
            need.append(psffn)
        wtfn = fn.replace('p.fits', 'p_weight.fits')
        if not os.path.exists(wtfn):
            print('Missing:', wtfn)
            need.append(wtfn)
        dqfn = fn.replace('p.fits', 'p_flag.fits')
        if not os.path.exists(dqfn):
            print('Missing:', dqfn)
            need.append(dqfn)

        bands = ['u','g','r','i2','z']
        for band in bands:
            for fn in need:
                url = 'http://limu.cfht.hawaii.edu/COSMOS-Tractor/FITS/MegaCam/%s/%s' % (band, os.path.basename(fn))
                cmd = '(cd euclid/images/megacam && wget -c "%s")' % url
                print(cmd)
                os.system(cmd)

def queue_list(opt):
    survey = get_survey()
    ccds = survey.get_ccds_readonly()
    ccds.index = np.arange(len(ccds))

    if opt.name is None:
        opt.name = os.path.basename(opt.queue_list).replace('.lst','')

    allccds = []
    expnums = get_exposures_in_list(opt.queue_list)
    for e in expnums:
        ccds = survey.find_ccds(expnum=e)
        allccds.append(ccds)
        for ccd in ccds:
            print(ccd.index, '%i-%s' % (ccd.expnum, ccd.ccdname))

    allccds = merge_tables(allccds)
    # Now group by CCDname and print out sets of indices to run at once
    for name in np.unique(allccds.ccdname):
        I = np.flatnonzero(allccds.ccdname == name)
        print(','.join(['%i'%i for i in allccds.index[I]]),
              '%s-%s' % (opt.name, name))

def analyze(opt):
    fn = opt.analyze
    expnums = get_exposures_in_list(fn)
    print(len(expnums), 'exposures')
    name = os.path.basename(fn).replace('.lst','')
    print('Image list name:', name)
    #for ccd in range(36):
    for ccd in [0]:
        fn = os.path.join('euclid-out', 'forced',
                          'megacam-%s-ccd%02i.fits' % (name, ccd))
        print('Reading', fn)
        T = fits_table(fn)
        print(len(T), 'from', fn)

        fluxes = np.unique([c for c in T.columns()
                            if c.startswith('flux_') and
                            not c.startswith('flux_ivar_')])
        print('Fluxes:', fluxes)
        assert(len(fluxes) == 1)
        fluxcol = fluxes[0]
        print('Flux column:', fluxcol)
        T.rename(fluxcol, 'flux')
        T.rename(fluxcol.replace('flux_', 'flux_ivar_'), 'flux_ivar')
        
        T.allflux = np.zeros((len(T), len(expnums)), np.float32)
        T.allflux_ivar = np.zeros((len(T), len(expnums)), np.float32)

        T.allx = np.zeros((len(T), len(expnums)), np.float32)
        T.ally = np.zeros((len(T), len(expnums)), np.float32)
        
        TT = []
        for i,expnum in enumerate(expnums):
            fn = os.path.join('euclid-out', 'forced',
                              'megacam-%i-ccd%02i.fits' % (expnum, ccd))
            print('Reading', fn)
            t = fits_table(fn)
            t.iexp = np.zeros(len(t), np.uint8) + i
            t.rename(fluxcol, 'flux')
            t.rename(fluxcol.replace('flux_', 'flux_ivar_'), 'flux_ivar')
            print(len(t), 'from', fn)
            # Edge effects...
            W,H = 2112, 4644
            margin = 30
            t.cut((t.x > margin) * (t.x < W-margin) *
                  (t.y > margin) * (t.y < H-margin))
            print('Keeping', len(t), 'not close to the edges')
            
            TT.append(t)
        TT = merge_tables(TT)

        imap = dict([((n,o),i) for i,(n,o) in enumerate(zip(T.brickname,
                                                            T.objid))])
        TT.index = np.array([imap[(n,o)]
                             for n,o in zip(TT.brickname, TT.objid)])

        T.allflux     [TT.index, TT.iexp] = TT.flux
        T.allflux_ivar[TT.index, TT.iexp] = TT.flux_ivar
        T.allx        [TT.index, TT.iexp] = TT.x
        T.ally        [TT.index, TT.iexp] = TT.y

    T.cflux_ivar = np.sum(T.allflux_ivar, axis=1)
    T.cflux = np.sum(T.allflux * T.allflux_ivar, axis=1) / T.cflux_ivar

    ps = PlotSequence('euclid')

    plt.clf()
    ha = dict(range=(0, 1600), bins=100, histtype='step')
    plt.hist(T.flux_ivar, color='k', **ha)
    for i in range(len(expnums)):
        plt.hist(T.allflux_ivar[:,i], **ha)
    plt.xlabel('Invvar')
    ps.savefig()
        
    
    I = np.flatnonzero(T.flux_ivar > 0)

    tf  = np.repeat(T.flux     [:,np.newaxis], len(expnums), axis=1)
    tiv = np.repeat(T.flux_ivar[:,np.newaxis], len(expnums), axis=1)
    J = np.flatnonzero((T.allflux_ivar * tiv) > 0)
    
    plt.clf()
    plt.plot(tf.flat[J], T.allflux.flat[J], 'b.', alpha=0.1)
    plt.xlabel('Image set flux')
    plt.ylabel('Exposure fluxes')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.title(name)
    plt.axhline(0., color='k', alpha=0.1)
    plt.axvline(0., color='k', alpha=0.1)
    ax = plt.axis()
    plt.plot([-1e6, 1e6], [-1e6, 1e6], 'k-', alpha=0.1)
    plt.axis(ax)
    ps.savefig()

    # Outliers... turned out to be sources on the edges.
    # J = np.flatnonzero(((T.allflux_ivar * tiv) > 0) *
    #                    (np.abs(T.allflux) > 1.) *
    #                    (np.abs(tf) < 1.))
    # print('Plotting', len(J), 'with sig. diff')
    # plt.clf()
    # plt.scatter(T.allx.flat[J], T.ally.flat[J],
    #             c=(tf.flat[J] - T.allflux.flat[J]), alpha=0.1)
    # plt.title(name)
    # ps.savefig()
    
    plt.clf()
    plt.plot(T.flux[I], T.cflux[I], 'b.', alpha=0.1)
    plt.xlabel('Image set flux')
    plt.ylabel('Summed exposure fluxes')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.title(name)
    plt.axhline(0., color='k', alpha=0.1)
    plt.axvline(0., color='k', alpha=0.1)
    ax = plt.axis()
    plt.plot([-1e6, 1e6], [-1e6, 1e6], 'k-', alpha=0.1)
    plt.axis(ax)
    ps.savefig()

    plt.clf()
    plt.plot(T.flux_ivar[:,np.newaxis], T.allflux_ivar, 'b.', alpha=0.1)
    plt.xlabel('Image set flux invvar')
    plt.ylabel('Exposure flux invvars')
    plt.title(name)
    ps.savefig()

    plt.clf()
    plt.plot(T.flux_ivar, T.cflux_ivar, 'b.', alpha=0.1)
    plt.xlabel('Image set flux invvar')
    plt.ylabel('Summed exposure flux invvar')
    plt.title(name)
    ax = plt.axis()
    plt.plot([-1e6, 1e6], [-1e6, 1e6], 'k-', alpha=0.1)
    plt.axis(ax)
    ps.savefig()

    K = np.flatnonzero((T.flux_ivar * T.cflux_ivar) > 0)
    plt.clf()
    #plt.plot(T.flux_ivar, T.cflux_ivar, 'b.', alpha=0.1)
    loghist(T.flux_ivar[K], T.cflux_ivar[K], 200)
    plt.xlabel('Image set flux invvar')
    plt.ylabel('Summed exposure flux invvar')
    plt.title(name)
    ps.savefig()

    plt.clf()
    loghist(T.flux[K] * np.sqrt(T.flux_ivar[K]),
            T.cflux[K] * np.sqrt(T.cflux_ivar[K]), 200,
        range=[[-5,25]]*2)
    plt.xlabel('Image set S/N')
    plt.ylabel('Summed exposures S/N')
    plt.title(name)
    ps.savefig()

    ACS = read_acs_catalogs()

    imap = dict([((n,o),i) for i,(n,o) in enumerate(zip(ACS.brickname,
                                                        ACS.objid))])
    T.index = np.array([imap[(n,o)]
                        for n,o in zip(T.brickname, T.objid)])

    ACS.cut(T.index)
    
    plt.clf()
    plt.plot(ACS.decam_flux, T.cflux, 'b.', alpha=0.1)
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.xlabel('ACS flux (I)')
    plt.ylabel('CFHT flux (i)')
    plt.title(name)
    plt.plot([0,1e4],[0,1e4], 'k-', alpha=0.1)
    plt.ylim(-1, 1e4)
    ps.savefig()

    ACS.mag = -2.5 * (np.log10(ACS.decam_flux) - 9)
    T.cmag = -2.5 * (np.log10(T.cflux) - 9)
    
    plt.clf()
    plt.plot(ACS.mag, T.cmag, 'b.', alpha=0.1)
    plt.xlabel('ACS I (mag)')
    plt.ylabel('Summed CFHT i (mag)')
    plt.title(name)
    plt.plot([0,1e4],[0,1e4], 'k-', alpha=0.5)
    plt.axis([27, 16, 30, 15])
    ps.savefig()

def analyze2(opt):
    # Analyze forced photometry results
    ps = PlotSequence('euclid')

    forcedfn = 'forced-megacam.fits'

    if os.path.exists(forcedfn):
        F = fits_table(forcedfn)
    else:
        T = read_acs_catalogs()
        print(len(T), 'ACS catalog entries')
        #T.cut(T.brick_primary)
        #print(len(T), 'primary')
        # read Megacam forced-photometry catalogs
        fns = glob('euclid-out/forced/megacam-*.fits')
        fns.sort()
        F = []
        for fn in fns:
            f = fits_table(fn)
            print(len(f), 'from', fn)
            fn = os.path.basename(fn)
            fn = fn.replace('.fits','')
            words = fn.split('-')
            assert(words[0] == 'megacam')
            expnum = int(words[1], 10)
            ccdname = words[2]
            f.expnum = np.array([expnum]*len(f))
            f.ccdname = np.array([ccdname]*len(f))
            F.append(f)
        F = merge_tables(F)
        objmap = dict([((brickname,objid),i) for i,(brickname,objid) in
                       enumerate(zip(T.brickname, T.objid))])
        I = np.array([objmap[(brickname, objid)] for brickname,objid
                      in zip(F.brickname, F.objid)])
        F.type = T.type[I]
        F.acs_flux = T.decam_flux[I]
        F.ra  = T.ra[I]
        F.dec = T.dec[I]
        F.bx  = T.bx[I]
        F.by  = T.by[I]
        F.writeto(forcedfn)

    print(len(F), 'forced photometry measurements')
    # There's a great big dead zone around the outside of the image...
    # roughly X=[0 to 33] and X=[2080 to 2112] and Y=[4612..]
    J = np.flatnonzero((F.x > 40) * (F.x < 2075) * (F.y < 4600))
    F.cut(J)
    print('Cut out edges:', len(F))
    F.fluxsn = F.flux * np.sqrt(F.flux_ivar)
    F.mag = -2.5 * (np.log10(F.flux) - 9.)
    
    I = np.flatnonzero(F.type == 'PSF ')
    print(len(I), 'PSF')

    for t in ['SIMP', 'DEV ', 'EXP ', 'COMP']:
        J = np.flatnonzero(F.type == t)
        print(len(J), t)

    S = np.flatnonzero(F.type == 'SIMP')
    print(len(S), 'SIMP')
    S = F[S]
    
    plt.clf()
    plt.semilogx(F.fluxsn, F.mag, 'k.', alpha=0.1)
    plt.semilogx(F.fluxsn[I], F.mag[I], 'r.', alpha=0.1)
    plt.semilogx(S.fluxsn, S.mag, 'b.', alpha=0.1)
    plt.xlabel('Megacam forced-photometry Flux S/N')
    plt.ylabel('Megacam forced-photometry mag')

    #plt.xlim(1., 1e5)
    #plt.ylim(10, 26)
    plt.xlim(1., 1e4)
    plt.ylim(16, 26)

    J = np.flatnonzero((F.fluxsn[I] > 4.5) * (F.fluxsn[I] < 5.5))
    print(len(J), 'between flux S/N 4.5 and 5.5')
    J = I[J]
    medmag = np.median(F.mag[J])
    print('Median mag', medmag)
    plt.axvline(5., color='r', alpha=0.5, lw=2)
    plt.axvline(5., color='k', alpha=0.5)
    plt.axhline(medmag, color='r', alpha=0.5, lw=2)
    plt.axhline(medmag, color='k', alpha=0.5)

    J = np.flatnonzero((S.fluxsn > 4.5) * (S.fluxsn < 5.5))
    print(len(J), 'SIMP between flux S/N 4.5 and 5.5')
    medmag = np.median(S.mag[J])
    print('Median mag', medmag)
    plt.axhline(medmag, color='b', alpha=0.5, lw=2)
    plt.axhline(medmag, color='k', alpha=0.5)

    plt.title('Megacam forced-photometered from ACS-VIS')

    ax = plt.axis()
    p1 = plt.semilogx([0],[0], 'k.')
    p2 = plt.semilogx([0], [0], 'r.')
    p3 = plt.semilogx([0], [0], 'b.')
    plt.legend((p1[0], p2[0], p3[0]),
               ('All sources', 'Point sources',
                '"Simple" galaxies'))
    plt.axis(ax)
    ps.savefig()

def analyze3(opt):
    ps = PlotSequence('euclid')

    name1s = ['g.SNR12-MIQ', 'i.SNR10-MIQ', 'r.SNR10-HIQ',
              'u.SNR10-HIQ', 'z.SNR09-ALL', ]
    name2s = [
        ['g.SNR10-MIQ', 'g.SNR10-BIQ', 'g.SNR09-LIQ', 'g.PC.Shallow'],
        ['i.SNR12-LIQ', 'i.SNR10-LIQ', 'i.SNR09-HIQ', 'i.PC.Shallow'],
        ['r.SNR12-LIQ', 'r.SNR10-LIQ', 'r.SNR08-MIQ', 'r.PC.Shallow'],
        ['u.SNR10-MIQ', 'u.SNR08-PIQ', 'u.SNR07-HIQ'],
        ['z.SNR08-BIQ', 'z.SNR06-LIQ', 'z.SNR06-HIQ', 'z.PC.Shallow'],
        ]

    allnames = []
    for n1,n2 in zip(name1s, name2s):
        allnames.append(n1)
        allnames.extend(n2)

    for name in allnames:
        F = fits_table('euclid-out/forced-%s.fits' % name)

        F.fluxsn = F.flux * np.sqrt(F.flux_ivar)
        F.mag = -2.5 * (np.log10(F.flux) - 9.)

        I = np.flatnonzero(F.type == 'PSF ')
        print(len(I), 'PSF')
        P = F[I]

        I = np.flatnonzero(F.type == 'SIMP')
        print(len(I), 'SIMP')
        S = F[I]
    
        plt.clf()
        plt.semilogx(F.fluxsn, F.mag, 'k.', alpha=0.01)
        plt.semilogx(P.fluxsn, P.mag, 'r.', alpha=0.01)
        plt.semilogx(S.fluxsn, S.mag, 'b.', alpha=0.01)
        plt.xlabel('Megacam forced-photometry Flux S/N')
        plt.ylabel('Megacam forced-photometry mag')

        #plt.xlim(1., 1e4)
        #plt.ylim(16, 26)
        plt.xlim(1., 100.)
        plt.ylim(22., 28.)

        J = np.flatnonzero((P.fluxsn > 4.5) * (P.fluxsn < 5.5))
        print(len(J), 'between flux S/N 4.5 and 5.5')
        medmag_psf = np.median(P.mag[J])
        print('Median mag', medmag_psf)
        plt.axvline(5., color='r', alpha=0.5, lw=2)
        plt.axvline(5., color='k', alpha=0.5)
        plt.axhline(medmag_psf, color='r', alpha=0.5, lw=2)
        plt.axhline(medmag_psf, color='k', alpha=0.5)

        J = np.flatnonzero((S.fluxsn > 4.5) * (S.fluxsn < 5.5))
        print(len(J), 'SIMP between flux S/N 4.5 and 5.5')
        medmag_simp = np.median(S.mag[J])
        print('Median mag', medmag_simp)
        plt.axhline(medmag_simp, color='b', alpha=0.5, lw=2)
        plt.axhline(medmag_simp, color='k', alpha=0.5)

        plt.title('Megacam forced-photometered from ACS-VIS: %s' % name)

        ax = plt.axis()
        p1 = plt.semilogx([0],[0], 'k.')
        p2 = plt.semilogx([0], [0], 'r.')
        p3 = plt.semilogx([0], [0], 'b.')
        plt.legend((p1[0], p2[0], p3[0]),
                   ('All sources',
                    'Point sources (~%.2f @ 5 sigma)' % medmag_psf,
                    '"Simple" galaxies (~%.2f @ 5 sigma)' % medmag_simp))
                    
        plt.axis(ax)
        ps.savefig()

    for name1,N2 in zip(name1s, name2s):
        T1 = fits_table('euclid-out/forced-%s.fits' % name1)
        for name2 in N2:
            T2 = fits_table('euclid-out/forced-%s.fits' % name2)
            
            assert(len(T2) == len(T1))
            I = np.flatnonzero((T1.flux_ivar > 0) & (T2.flux_ivar > 0))

            plt.clf()
            plt.plot(T1.flux[I], T2.flux[I], 'k.')
            plt.xscale('symlog')
            plt.yscale('symlog')
            plt.xlabel(name1 + ' flux')
            plt.ylabel(name2 + ' flux')
            ps.savefig()


def analyze4(opt):
    ps = PlotSequence('euclid')

    names = ['r.SNR10-HIQ',
             'r.SNR12-LIQ', 'r.SNR10-LIQ', 'r.SNR08-MIQ', 'r.PC.Shallow']
    exps = [get_exposures_in_list('euclid/images/megacam/lists/%s.lst' % n)
            for n in names]

    name_to_exps = dict(zip(names, exps))

    uexps = np.unique(np.hstack(exps))
    print('Unique exposures:', uexps)

    e1 = uexps[0]
    e2 = uexps[1]

    for e1,e2 in zip(uexps[::2], uexps[1::2]):
        TT1 = []
        TT2 = []

        plt.clf()

        for ccd in range(36):
            fn1 = os.path.join('euclid-out', 'forced',
                               'megacam-%s-ccd%02i.fits' % (e1, ccd))
            fn2 = os.path.join('euclid-out', 'forced',
                               'megacam-%s-ccd%02i.fits' % (e2, ccd))
            T1 = fits_table(fn1)
            T2 = fits_table(fn2)
            print(len(T1), 'from', fn1)
            print(len(T2), 'from', fn2)
    
            imap = dict([((n,o),i) for i,(n,o) in enumerate(zip(T1.brickname,
                                                                T1.objid))])
            imatch = np.array([imap.get((n,o), -1) for n,o in zip(T2.brickname, T2.objid)])
            T2.cut(imatch >= 0)
            T1.cut(imatch[imatch >= 0])
            print(len(T1), 'matched')
    
            I = np.flatnonzero((T1.flux_ivar_r > 0) * (T2.flux_ivar_r > 0))
            T1.cut(I)
            T2.cut(I)
            print(len(T1), 'good')

            TT1.append(T1)
            TT2.append(T2)

            n,b,p = plt.hist((T1.flux_r - T2.flux_r) / np.sqrt(1./T1.flux_ivar_r + 1./T2.flux_ivar_r),
                             range=(-5, 5), bins=100,
                             histtype='step', color='k', label='All sources', normed=True)

        binc = (b[1:] + b[:-1])/2.
        yy = np.exp(-0.5 * binc**2)
        p1 = plt.plot(binc, yy / yy.sum() * np.sum(n), 'k-', alpha=0.5, lw=2,
                      label='N(0,1)')
        yy = np.exp(-0.5 * binc**2 / (1.2**2))
        p2 = plt.plot(binc, yy / yy.sum() * np.sum(n), 'k--', alpha=0.5, lw=2,
                      label='N(0,1.2)')
        plt.xlabel('Flux difference / Flux errors (sigma)')
        plt.xlim(-5,5)
        plt.title('Forced phot differences: Megacam r %s to %s' % (e1, e2))
        ps.savefig()


        T1 = merge_tables(TT1)
        T2 = merge_tables(TT2)
    
        plt.clf()
        n,b,p = plt.hist((T1.flux_r - T2.flux_r) / np.sqrt(1./T1.flux_ivar_r + 1./T2.flux_ivar_r),
                         range=(-5, 5), bins=100,
                         histtype='step', color='k', label='All sources', normed=True)

        binc = (b[1:] + b[:-1])/2.
        yy = np.exp(-0.5 * binc**2)
        plt.plot(binc, yy / yy.sum() * np.sum(n), 'k-', alpha=0.5, lw=2,
                      label='N(0,1)')
        yy = np.exp(-0.5 * binc**2 / (1.2**2))
        p2 = plt.plot(binc, yy / yy.sum() * np.sum(n), 'k--', alpha=0.5, lw=2,
                      label='N(0,1.2)')

        #for magcut in [21,23]:
        for magcut in [23]:
            meanmag = -2.5 * (np.log10((T1.flux_r + T2.flux_r)/2.) - 9)
            I = np.flatnonzero(meanmag < magcut)
            plt.hist((T1.flux_r[I] - T2.flux_r[I]) /
                     np.sqrt(1./T1.flux_ivar_r[I] + 1./T2.flux_ivar_r[I]),
                     range=(-5, 5), bins=100,
                     histtype='step', label='mag < %g' % magcut, normed=True)
        
        plt.xlabel('Flux difference / Flux errors (sigma)')
        plt.xlim(-5,5)
        #plt.title('Forced phot differences: Megacam r %s to %s, ccd%02i' % (e1, e2, ccd))
        plt.title('Forced phot differences: Megacam r %s to %s' % (e1, e2))
        plt.legend(loc='upper right')
        ps.savefig()

        T1.mag = -2.5 * (np.log10(T1.flux_r) - 9)
        T2.mag = -2.5 * (np.log10(T2.flux_r) - 9)
        plt.clf()
        ha = dict(range=(20,26), bins=100, histtype='step')
        plt.hist(T1.mag, color='b', **ha)
        plt.hist(T2.mag, color='r', **ha)
        plt.xlabel('Mag')
        plt.title('Forced phot: Megacam r %s, %s, ccd%02i' % (e1, e2, ccd))
        ps.savefig()


        break


    from collections import Counter

    #name1 = names[0]
    #names2 = names[1:]

    for iname1,name1 in enumerate(names):
        for name2 in names[iname1+1:]:
            #T1 = fits_table('euclid-out/forced-%s.fits' % name1)
            #T2 = fits_table('euclid-out/forced-%s.fits' % name2)
            T1 = fits_table('euclid-out/forced2-%s.fits' % name1)
            T2 = fits_table('euclid-out/forced2-%s.fits' % name2)

            #print('T1 nexp:', Counter(T1.nexp).most_common())
            #print('T2 nexp:', Counter(T2.nexp).most_common())
    
            mexp1 = np.median(T1.nexp[T1.nexp > 0])
            mexp2 = np.median(T2.nexp[T2.nexp > 0])
    
            I = np.flatnonzero((T1.nexp >= mexp1) * (T2.nexp >= mexp2))
    
            T1.cut(I)
            T2.cut(I)
            print(len(T1), 'sources kept')
    
            plt.clf()
            n,b,p = plt.hist((T1.flux - T2.flux) /
                             np.sqrt(1./T1.flux_ivar + 1./T2.flux_ivar),
                             range=(-5, 5), bins=100,
                             histtype='step', color='k', label='All sources', normed=True)
    
            # for magcut in [23, 24, 25]:
            #     meanmag = -2.5 * (np.log10((T1.flux + T2.flux)/2.) - 9)
            #     I = np.flatnonzero(meanmag < magcut)
            #     plt.hist((T1.flux[I] - T2.flux[I]) /
            #              np.sqrt(1./T1.flux_ivar[I] + 1./T2.flux_ivar[I]),
            #              range=(-5, 5), bins=100,
            #              histtype='step', label='mag < %g' % magcut, normed=True)

            for maglo, maghi in [(18,23), (23,24), (24,25)]:
                meanmag = -2.5 * (np.log10((T1.flux + T2.flux)/2.) - 9)
                I = np.flatnonzero((meanmag > maglo) * (meanmag <= maghi))
                plt.hist((T1.flux[I] - T2.flux[I]) /
                         np.sqrt(1./T1.flux_ivar[I] + 1./T2.flux_ivar[I]),
                         range=(-5, 5), bins=100,
                         histtype='step', label='mag %g to %g' % (maglo,maghi),
                         normed=True)
    
            binc = (b[1:] + b[:-1])/2.
            yy = np.exp(-0.5 * binc**2)
            plt.plot(binc, yy / yy.sum() * np.sum(n), 'k-', alpha=0.5, lw=2,
                      label='N(0,1)')
            yy = np.exp(-0.5 * binc**2 / (1.2**2))
            plt.plot(binc, yy / yy.sum() * np.sum(n), 'k--', alpha=0.5, lw=2,
                     label='N(0,1.2)')
            yy = np.exp(-0.5 * binc**2 / (1.25**2))
            plt.plot(binc, yy / yy.sum() * np.sum(n), 'k-',
                     label='N(0,1.25)')

            exps1 = set(name_to_exps[name1])
            exps2 = set(name_to_exps[name2])
            print(name1, 'has', len(exps1))
            print(name2, 'has', len(exps2))
            incommon = len(exps1.intersection(exps2))
            print(incommon, 'in common')
            
            plt.legend(loc='upper right')
            plt.xlabel('Flux difference / Flux errors (sigma)')
            plt.xlim(-5,5)
            #plt.title('Forced phot differences: Megacam %s to %s (%i exposures in common)' % (name1, name2, incommon))
            plt.title('Forced phot differences: Megacam %s to %s' % (name1, name2))
            ps.savefig()
    
            # plt.clf()
            # n,b,p = loghist(np.log(T1.flux),
            #                 np.log(T2.flux),
            #                 range=((-3,5),(-3,5)), bins=200)
            # plt.xlabel('log Flux 1')
            # plt.ylabel('log Flux 2')
            # plt.title('Forced phot differences: Megacam %s to %s' % (name1, name2))
            # ps.savefig()
    
            # plt.clf()
            # n,b,p = loghist(np.log((T1.flux + T2.flux)/2.),
            #                 (T1.flux - T2.flux) /
            #                 np.sqrt(1./T1.flux_ivar + 1./T2.flux_ivar),
            #                 range=((-3,5),(-5,5)), bins=200)
            # plt.axhline(0, color=(0.3,0.3,1.0))
            # plt.xlabel('log average Flux')
            # plt.ylabel('Flux difference / Flux errors (sigma)')
            # plt.title('Forced phot differences: Megacam %s to %s' % (name1, name2))
            # ps.savefig()
    
            # T1.mag = -2.5 * (np.log10(T1.flux) - 9)
            # T2.mag = -2.5 * (np.log10(T2.flux) - 9)
            # plt.clf()
            # ha = dict(range=(20,28), bins=100, histtype='step')
            # plt.hist(T1.mag, color='b', **ha)
            # plt.hist(T2.mag, color='r', **ha)
            # plt.xlabel('Mag')
            # plt.title('Forced phot: Megacam r %s, %s, ccd%02i' % (e1, e2, ccd))
            # ps.savefig()
        break


def geometry():
    # Regular tiling with small overlaps; RA,Dec aligned
    # In this dataset, the Megacam images are also very nearly all aligned.
    survey = get_survey()
    ccds = survey.get_ccds_readonly()
    T = ccds[ccds.camera == 'acs-vis']
    print(len(T), 'ACS CCDs')
    plt.clf()
    for ccd in T:
        wcs = survey.get_approx_wcs(ccd)
        h,w = wcs.shape
        rr,dd = wcs.pixelxy2radec([1,w,w,1,1], [1,1,h,h,1])
        plt.plot(rr, dd, 'b-')

    T = ccds[ccds.camera == 'megacam']
    print(len(T), 'Megacam CCDs')
    for ccd in T:
        wcs = survey.get_approx_wcs(ccd)
        h,w = wcs.shape
        rr,dd = wcs.pixelxy2radec([1,w,w,1,1], [1,1,h,h,1])
        plt.plot(rr, dd, 'r-')
    plt.savefig('ccd-outlines.png')

    TT = []
    fns = glob('euclid-out/tractor/*/tractor-*.fits')
    for fn in fns:
        T = fits_table(fn)
        print(len(T), 'from', fn)
        TT.append(T)
    T = merge_tables(TT)
    plt.clf()
    plothist(T.ra, T.dec, 200)
    plt.savefig('acs-sources1.png')
    T = fits_table('euclid/survey-ccds-acsvis.fits.gz')
    for ccd in T:
        wcs = survey.get_approx_wcs(ccd)
        h,w = wcs.shape
        rr,dd = wcs.pixelxy2radec([1,w,w,1,1], [1,1,h,h,1])
        plt.plot(rr, dd, 'b-')
    plt.savefig('acs-sources2.png')

    return 0

    # It's a 7x7 grid... hackily define RA,Dec boundaries.
    T = fits_table('euclid/survey-ccds-acsvis.fits.gz')
    ras = T.ra.copy()
    ras.sort()
    decs = T.dec.copy()
    decs.sort()
    print('RAs:', ras)
    print('Decs:', decs)
    ras  =  ras.reshape((-1, 7)).mean(axis=1)
    decs = decs.reshape((-1, 7)).mean(axis=1)
    print('RAs:', ras)
    print('Decs:', decs)
    rasplits = (ras[:-1] + ras[1:])/2.
    print('RA boundaries:', rasplits)
    decsplits = (decs[:-1] + decs[1:])/2.
    print('Dec boundaries:', decsplits)

def reduce_acs_image(opt, survey):
    ccds = survey.get_ccds_readonly()
    ccds.cut(ccds.camera == 'acs-vis')
    print('Cut to', len(ccds), 'from ACS-VIS')

    ccds.cut(ccds.expnum == opt.expnum)
    print('Cut to', len(ccds), 'with expnum', opt.expnum)
    allccds = ccds
    
    for iccd in range(len(allccds)):
        # Process just this single CCD.
        survey.ccds = allccds[np.array([iccd])]
        ccd = survey.ccds[0]
        brickname = 'acsvis-%03i' % ccd.expnum
        run_brick(brickname, survey, radec=(ccd.ra, ccd.dec), pixscale=0.1,
                  #width=200, height=200,
                  width=ccd.width, height=ccd.height, 
                  bands=['I'],
                  threads=opt.threads,
                  wise=False, do_calibs=False,
                  pixPsf=True, coadd_bw=True, ceres=False,
                  blob_image=True, allbands=allbands,
                  forceAll=True, writePickles=False)
    #plots=True, plotbase='euclid',
    return 0

def package(opt):

    print('HACK -- fixing up VISTA forced photometry...')
    # Match VISTA images to the others...
    ACS = fits_table('euclid-out/forced-r.SNR10-HIQ.fits')
    imap = dict([((n,o),i) for i,(n,o) in enumerate(zip(ACS.brickname,
                                                        ACS.objid))])
    for band in ['Y','J','H']:
        fn = 'euclid-out/forced-vista-%s.fits' % band
        T = fits_table(fn)
        print('Read', len(T), 'from', fn)
        T.rename('flux_%s' % band.lower(), 'flux')
        T.rename('flux_ivar_%s' % band.lower(), 'flux_ivar')
        T.acsindex = np.array([imap.get((n,o), -1)
                               for n,o in zip(T.brickname, T.objid)])
        T.cut(T.acsindex >= 0)
        print('Kept', len(T), 'with a match')
        
        ACS.flux = np.zeros(len(ACS), np.float32)
        ACS.flux_ivar = np.zeros(len(ACS), np.float32)
        ACS.nexp = np.zeros(len(ACS), np.uint8)
        ACS.flux[T.acsindex] = T.flux
        ACS.flux_ivar[T.acsindex] = T.flux_ivar
        ACS.nexp[T.acsindex] = 1

        fn = 'euclid-out/forced-vista-%s-2.fits' % band
        ACS.writeto(fn)
        print('Wrote', fn)

    return 0

    ACS = read_acs_catalogs()
    print('Read', len(ACS), 'ACS catalog entries')
    ACS.cut(ACS.brick_primary)
    print('Cut to', len(ACS), 'primary')

    imap = dict([((n,o),i) for i,(n,o) in enumerate(zip(ACS.brickname,
                                                        ACS.objid))])

    keepacs = np.zeros(len(ACS), bool)

    listfns = glob('euclid/images/megacam/lists/*.lst')

    listnames = [os.path.basename(fn).replace('.lst','') for fn in listfns]
    explists = [get_exposures_in_list(fn) for fn in listfns]
    ccds = list(range(36))

    results = []

    for expnums in explists:
        cflux      = np.zeros(len(ACS), np.float32)
        cflux_ivar = np.zeros(len(ACS), np.float32)
        nexp       = np.zeros(len(ACS), np.uint8)

        for i,expnum in enumerate(expnums):
            for ccd in ccds:
                fn = os.path.join('euclid-out', 'forced',
                                  'megacam-%i-ccd%02i.fits' % (expnum, ccd))
                print('Reading', fn)
                t = fits_table(fn)

                fluxes = np.unique([c for c in t.columns()
                                    if c.startswith('flux_') and
                                    not c.startswith('flux_ivar_')])
                print('Fluxes:', fluxes)
                assert(len(fluxes) == 1)
                fluxcol = fluxes[0]
                print('Flux column:', fluxcol)

                t.iexp = np.zeros(len(t), np.uint8) + i
                t.ccd  = np.zeros(len(t), np.uint8) + ccd
                t.rename(fluxcol, 'flux')
                t.rename(fluxcol.replace('flux_', 'flux_ivar_'), 'flux_ivar')
                print(len(t), 'from', fn)
                # Edge effects...
                W,H = 2112, 4644
                margin = 30
                t.cut((t.x > margin) * (t.x < W-margin) *
                      (t.y > margin) * (t.y < H-margin))
                print('Keeping', len(t), 'not close to the edges')
                acsindex = np.array([imap[(n,o)]
                                     for n,o in zip(t.brickname, t.objid)])

                # Find sources with existing measurements
                # I = np.flatnonzero((cflux_ivar[acsindex] > 0) * (cflux[acsindex] > 0) * (t.flux > 0))
                # if len(I) > 100:
                #     print(len(I), 'sources have previous measurements')
                #     ai = acsindex[I]
                #     cmag,cmagerr = NanoMaggies.fluxErrorsToMagErrors(
                #         cflux[ai] / np.maximum(cflux_ivar[ai], 1e-16), cflux_ivar[ai])
                #     mag,magerr = NanoMaggies.fluxErrorsToMagErrors(
                #         t.flux[I], t.flux_ivar[I])
                #     var = np.maximum(cmagerr, 0.02)**2 + np.maximum(magerr, 0.02)**2
                #     offset = np.sum((mag - cmag) * 1./var) / np.sum(1./var)
                #     print('Offset of', offset, 'mag')
                #     scale = 10.**(offset / 2.5)
                #     print('Applying scale of', scale, 'to fluxes and ivars')
                #     t.flux *= scale
                #     t.flux_ivar /= (scale**2)

                cflux     [acsindex] += t.flux * t.flux_ivar
                cflux_ivar[acsindex] += t.flux_ivar
                nexp      [acsindex] += 1

                keepacs[acsindex] = True

                filt = np.unique(t.filter)
                assert(len(filt) == 1)
                filt = filt[0]

        results.append((cflux / np.maximum(cflux_ivar, 1e-16), cflux_ivar, nexp, filt))

    ACS.rename('decam_flux', 'acs_flux')
    ACS.rename('decam_flux_ivar', 'acs_flux_ivar')

    for name,expnums,(cflux, cflux_ivar, nexp, filt) in zip(
        listnames, explists, results):
        ACS.flux = cflux
        ACS.flux_ivar = cflux_ivar
        ACS.nexp = nexp
        T = ACS[keepacs]

        primhdr = fitsio.FITSHDR()
        for i,expnum in enumerate(expnums):
            primhdr.add_record(dict(name='EXP%i'%i, value=expnum,
                                    comment='MegaCam exposure num'))
        primhdr.add_record(dict(name='FILTER', value=filt,
                                comment='MegaCam filter'))

        T.writeto('euclid-out/forced2-%s.fits' % name,
                  columns=['ra', 'dec', 'flux', 'flux_ivar', 'nexp',
                           'brickname', 'objid', 'type', 'ra_ivar', 'dec_ivar',
                           'dchisq', 'ebv', 'acs_flux', 'acs_flux_ivar',
                           'fracdev', 'fracdev_ivar',
                           'shapeexp_r', 'shapeexp_r_ivar', 'shapeexp_e1',
                           'shapeexp_e1_ivar', 'shapeexp_e2', 'shapeexp_e2_ivar',
                           'shapedev_r', 'shapedev_r_ivar', 'shapedev_e1',
                           'shapedev_e1_ivar', 'shapedev_e2', 'shapedev_e2_ivar'],
                  primheader=primhdr)




def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--zeropoints', action='store_true',
                        help='Read image files to produce CCDs tables.')
    parser.add_argument('--download', action='store_true',
                        help='Download images listed in image list files')

    parser.add_argument('--queue-list',
                        help='List the CCD indices of the images list in the given exposure list filename')
    parser.add_argument('--name',
                        help='Name for forced-phot output products, with --queue-list')

    parser.add_argument('--out',
                        help='Filename for forced-photometry output catalog')

    parser.add_argument('--analyze', 
                        help='Analyze forced photometry results for images listed in given image list file')

    parser.add_argument('--package', action='store_true',
                        help='Average and package up forced photometry results for different Megacam image lists')
    
    parser.add_argument('--expnum', type=int,
                        help='ACS exposure number to run')
    parser.add_argument('--threads', type=int, help='Run multi-threaded')

    parser.add_argument('--forced', help='Run forced photometry for given MegaCam CCD index (int) or comma-separated list of indices')

    parser.add_argument('--ceres', action='store_true', help='Use Ceres?')

    parser.add_argument(
        '--zoom', type=int, nargs=4,
        help='Set target image extent (default "0 3600 0 3600")')

    parser.add_argument('--grid', action='store_true',
                        help='Forced photometry on a grid of subimages?')

    parser.add_argument('--skip', action='store_true',
                        help='Skip forced-photometry if output file exists?')
    
    opt = parser.parse_args()

    Time.add_measurement(MemMeas)

    if opt.zeropoints:
        make_zeropoints()
        return 0

    if opt.download:
        download()
        return 0

    if opt.queue_list:
        queue_list(opt)
        return 0

    if opt.analyze:
        #analyze(opt)
        #analyze2(opt)
        #analyze3(opt)
        #geometry()
        analyze4(opt)
        return 0

    if opt.package:
        package(opt)
        return 0

    if opt.expnum is None and opt.forced is None:
        print('Need --expnum or --forced')
        return -1

    survey = get_survey()

    if opt.expnum is not None:
        # Run pipeline on a single ACS image.
        return reduce_acs_image(opt, survey)


    if opt.grid:
        
        assert(opt.out)

        i = int(opt.forced, 10)
        print('CCD index', i)
        ccds = survey.get_ccds_readonly()
        ccd = ccds[i]
        print('CCD', ccd)
        im = survey.get_image_object(ccd)
        print('Image', im)
        H,W = im.height, im.width
        print('Image size', W, 'x', H)

        overlap = 50
        ngrid = 20
        xsize = (W + overlap) / ngrid
        ysize = (H + overlap) / ngrid
        print('Size without overlap:', xsize, ysize)

        x0 = np.arange(ngrid) * xsize
        x1 = x0 + xsize + overlap
        x1[-1] = min(x1[-1], W)
        print('X ranges:')
        for xx0,xx1 in zip(x0,x1):
            print('  ', xx0,xx1)
        y0 = np.arange(ngrid) * ysize
        y1 = y0 + ysize + overlap
        y1[-1] = min(y1[-1], H)
        print('Y ranges:')
        for yy0,yy1 in zip(y0,y1):
            print('  ', yy0,yy1)

        print('Opt is a', opt)

        fns = []
        args = []
        for iy,(yy0,yy1) in enumerate(zip(y0,y1)):
            for ix,(xx0,xx1) in enumerate(zip(x0,x1)):
                newopt = argparse.Namespace()
                for k,v in opt.__dict__.items():
                    setattr(newopt, k, v)
                    print('Setting option', k, '=', v)
                outfn = opt.out.replace('.fits', '-x%02i-y%02i.fits' % (ix,iy))
                fns.append((outfn, ix, iy))
                newopt.out = outfn
                newopt.zoom = [xx0, xx1, yy0, yy1]
                print('New options:', newopt.__dict__)
                args.append((newopt, survey))

        xcuts = reduce(np.append, ([0], (x0[1:] + x1[:-1])/2., W))
        ycuts = reduce(np.append, ([0], (y0[1:] + y1[:-1])/2., H))
        print('X cuts:', xcuts)
        print('Y cuts:', ycuts)

        mp = multiproc(8)
        mp.map(_bounce_forced, args)

        TT = []
        for fn,ix,iy in fns:
            T = fits_table(fn)
            print('Read', len(T), 'from', fn)
            print('X range:', T.x.min(), T.x.max())
            xlo,xhi = xcuts[ix], xcuts[ix+1]
            print('Cut lo,hi', xlo, xhi)
            print('Y range:', T.y.min(), T.y.max())
            ylo,yhi = ycuts[iy], ycuts[iy+1]
            print('Cut lo,hi', ylo, yhi)
            T.cut((T.x >= xlo) * (T.x < xhi) * (T.y >= ylo) * (T.y < yhi))
            print('Cut to', len(T), 'sources')
            TT.append(T)
        TT = merge_tables(TT)
        TT.writeto(opt.out)
        print('Wrote', opt.out)
        return 0

    return forced_photometry(opt, survey)

def _bounce_forced(args):
    return forced_photometry(*args)

def forced_photometry(opt, survey):
    # Run forced photometry on a given image or set of Megacam images,
    # using the ACS catalog as the source.

    ps = PlotSequence('euclid')

    #if opt.name is None:
    #    opt.name = '%i-%s' % (im.expnum, im.ccdname)
    if opt.out is not None:
        outfn = opt.out
    else:
        outfn = 'euclid-out/forced/megacam-%s.fits' % opt.name

    if opt.skip and os.path.exists(outfn):
        print('Output file exists:', outfn)
        return 0
    
    t0 = Time()

    T = read_acs_catalogs()
    print('Read', len(T), 'ACS catalog entries')
    T.cut(T.brick_primary)
    print('Cut to', len(T), 'primary')

    # plt.clf()
    # I = T.brick_primary
    # plothist(T.ra[I], T.dec[I], 200)
    # plt.savefig('acs-sources3.png')

    opti = None
    forced_kwargs = {}
    if opt.ceres:
        from tractor.ceres_optimizer import CeresOptimizer
        B = 8
        opti = CeresOptimizer(BW=B, BH=B)
    
    T.shapeexp = np.vstack((T.shapeexp_r, T.shapeexp_e1, T.shapeexp_e2)).T
    T.shapedev = np.vstack((T.shapedev_r, T.shapedev_e1, T.shapedev_e2)).T

    ccds = survey.get_ccds_readonly()
    #I = np.flatnonzero(ccds.camera == 'megacam')
    #print(len(I), 'MegaCam CCDs')

    I = np.array([int(x,10) for x in opt.forced.split(',')])
    print(len(I), 'CCDs: indices', I)
    print('Exposure numbers:', ccds.expnum[I])
    print('CCD names:', ccds.ccdname[I])

    slc = None
    if opt.zoom:
        x0,x1,y0,y1 = opt.zoom
        zw = x1-x0
        zh = y1-y0
        slc = slice(y0,y1), slice(x0,x1)

    tims = []
    keep_sources = np.zeros(len(T), bool)

    for i in I:
        ccd = ccds[i]
        im = survey.get_image_object(ccd)
        print('CCD', im)

        wcs = im.get_wcs()
        if opt.zoom:
            wcs = wcs.get_subimage(x0, y0, zw, zh)

        ok,x,y = wcs.radec2pixelxy(T.ra, T.dec)
        x = (x-1).astype(np.float32)
        y = (y-1).astype(np.float32)
        h,w = wcs.shape
        J = np.flatnonzero((x >= 0) * (x < w) *
                           (y >= 0) * (y < h))
        if len(J) == 0:
            print('No sources within image.')
            continue

        print(len(J), 'sources are within this image')
        keep_sources[J] = True

        tim = im.get_tractor_image(pixPsf=True, slc=slc)
        print('Tim:', tim)
        tims.append(tim)

        # plt.clf()
        # plt.hist((tim.getImage() * tim.getInvError()).ravel(), range=(-5,5), bins=100,
        #          histtype='step', color='b')
        # plt.xlim(-5,5)
        # plt.xlabel('Pixel sigmas')
        # plt.title(tim.name)
        # ps.savefig()
        # 
        # plt.clf()
        # plt.imshow(tim.getImage(), interpolation='nearest', origin='lower',
        #            vmin=-2.*tim.sig1, vmax=5.*tim.sig1)
        # plt.title(tim.name)
        # ps.savefig()
        # 
        # ax = plt.axis()
        # plt.plot(x[keep_sources], y[keep_sources], 'b.')
        # plt.axis(ax)
        # ps.savefig()
        
    T.cut(keep_sources)
    print('Cut to', len(T), 'sources within at least one')

    bands = np.unique([tim.band for tim in tims])
    print('Bands:', bands)
    # convert to string
    bands = ''.join(bands)
    
    cat = read_fits_catalog(T, ellipseClass=EllipseE, allbands=bands,
                            bands=bands)
    for src in cat:
        src.brightness = NanoMaggies(**dict([(b, 1.) for b in bands]))

    for src in cat:
        # Limit sizes of huge models
        from tractor.galaxy import ProfileGalaxy
        if isinstance(src, ProfileGalaxy):
            tim = tims[0]
            px,py = tim.wcs.positionToPixel(src.getPosition())
            h = src._getUnitFluxPatchSize(tim, px, py, tim.modelMinval)
            MAXHALF = 128
            if h > MAXHALF:
                print('halfsize', h,'for',src,'-> setting to',MAXHALF)
                src.halfsize = MAXHALF
        
    tr = Tractor(tims, cat, optimizer=opti)
    tr.freezeParam('images')

    for src in cat:
        src.freezeAllBut('brightness')
        #src.getBrightness().freezeAllBut(tim.band)
    disable_galaxy_cache()
    # Reset fluxes
    nparams = tr.numberOfParams()
    tr.setParams(np.zeros(nparams, np.float32))
        
    F = fits_table()
    F.brickid   = T.brickid
    F.brickname = T.brickname
    F.objid     = T.objid
    F.mjd     = np.array([tim.primhdr.get('MJD-OBS', 0.)] * len(T)).astype(np.float32)
    F.exptime = np.array([tim.primhdr.get('EXPTIME', 0.)] * len(T)).astype(np.float32)

    if len(tims) == 1:
        tim = tims[0]
        F.filter  = np.array([tim.band] * len(T))
        ok,x,y = tim.sip_wcs.radec2pixelxy(T.ra, T.dec)
        F.x = (x-1).astype(np.float32)
        F.y = (y-1).astype(np.float32)


    t1 = Time()
    print('Prior to forced photometry:', t1-t0)

    # FIXME -- should we run one band at a time?

    R = tr.optimize_forced_photometry(variance=True, fitstats=False, #fitstats=True,
                                      shared_params=False, priors=False,
                                      **forced_kwargs)

    t2 = Time()
    print('Forced photometry:', t2-t1)

    units = {'exptime':'sec' }# 'flux':'nanomaggy', 'flux_ivar':'1/nanomaggy^2'}
    for band in bands:
        F.set('flux_%s' % band, np.array([src.getBrightness().getFlux(band)
                                          for src in cat]).astype(np.float32))
        units.update({'flux_%s' % band:' nanomaggy',
                      'flux_ivar_%s' % band:'1/nanomaggy^2'})

    # HACK -- use the parameter-setting machinery to set the source
    # brightnesses to the *inverse-variance* estimates, then read them
    # off...
    p0 = tr.getParams()
    tr.setParams(R.IV)
    for band in bands:
        F.set('flux_ivar_%s' % band,
              np.array([src.getBrightness().getFlux(band)
                        for src in cat]).astype(np.float32))
    tr.setParams(p0)
        
    hdr = fitsio.FITSHDR()
    columns = F.get_columns()
    for i,col in enumerate(columns):
        if col in units:
            hdr.add_record(dict(name='TUNIT%i' % (i+1), value=units[col]))

    primhdr = fitsio.FITSHDR()
    primhdr.add_record(dict(name='EXPNUM', value=im.expnum,
                            comment='Exposure number'))
    primhdr.add_record(dict(name='CCDNAME', value=im.ccdname,
                            comment='CCD name'))
    primhdr.add_record(dict(name='CAMERA', value=im.camera,
                            comment='Camera'))

    fitsio.write(outfn, None, header=primhdr, clobber=True)
    F.writeto(outfn, header=hdr, append=True)
    print('Wrote', outfn)
            
    t3 = Time()
    print('Wrap-up:', t3-t2)
    print('Total:', t3-t0)
    return 0
    
if __name__ == '__main__':
    sys.exit(main())

