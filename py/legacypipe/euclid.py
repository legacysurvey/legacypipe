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
from astrometry.util.plotutils import PlotSequence, plothist
from astrometry.util.ttime import Time, MemMeas

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

        self.psffn = self.imgfn.replace('p.fits', 'p_psfex.psf')
        assert(self.psffn != self.imgfn)

        self.wtfn = self.imgfn.replace('p.fits', 'p_weight.fits')
        assert(self.wtfn != self.imgfn)

        self.dqfn = self.imgfn.replace('p.fits', 'p_flag.fits')
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
    return survey

if __name__ == '__main__':

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

    parser.add_argument('--expnum', type=int,
                        help='ACS exposure number to run')
    parser.add_argument('--threads', type=int, help='Run multi-threaded')

    parser.add_argument('--forced', help='Run forced photometry for given MegaCam CCD index (int) or comma-separated list of indices')

    parser.add_argument('--ceres', action='store_true', help='Use Ceres?')

    parser.add_argument(
        '--zoom', type=int, nargs=4,
        help='Set target image extent (default "0 3600 0 3600")')

    opt = parser.parse_args()

    Time.add_measurement(MemMeas)

    if opt.zeropoints:
        make_zeropoints()
        sys.exit(0)

    if opt.download:
        fns = glob('euclid/images/megacam/lists/*.lst')
        expnums = set()
        for fn in fns:
            for line in open(fn,'r').readlines():
                words = line.split()
                e = int(words[0], 10)
                print('Exposure', e)
                expnums.add(e)
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
                    
        sys.exit(0)

    if opt.queue_list:
        survey = get_survey()
        ccds = survey.get_ccds_readonly()
        ccds.index = np.arange(len(ccds))

        if opt.name is None:
            opt.name = os.path.basename(opt.queue_list).replace('.lst','')

        allccds = []
        for line in open(opt.queue_list,'r').readlines():
            words = line.split()
            e = int(words[0], 10)
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

        sys.exit(0)

    if False:
        # Re-make jpeg images
        fns = glob('euclid-out/coadd/acs/acsvis-1??/*-image-I.fits')
        fns.sort()
        bands=['I']
        for fn in fns:
            I = fitsio.read(fn)
            print('Read', I.shape, 'from', fn)
            ims = [I]
            rgb = get_rgb(ims, bands, **rgbkwargs)
            print('RGB:', rgb.shape)
            # coadd_bw
            rgb = rgb.sum(axis=2)
            print('B/W:', rgb.shape)
            kwa = dict(cmap='gray')
            outfn = fn.replace('-I.fits', '.jpg')
            imsave_jpeg(outfn, rgb, origin='lower', **kwa)
            print('Wrote', outfn)
        sys.exit(0)
            
    if False:
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
                
        sys.exit(0)

    if False:
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

        sys.exit(0)

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

        sys.exit(0)
        
    if opt.expnum is None and opt.forced is None:
        print('Need --expnum or --forced')
        sys.exit(-1)

    survey = get_survey()

    if opt.expnum is not None:
        # Run pipeline on a single ACS image.
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
        sys.exit(0)
        
    # Run forced photometry on a given image or set of Megacam images,
    # using the ACS catalog as the source.

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
        tims.append(tim)
        
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
    F.mjd     = np.array([tim.primhdr['MJD-OBS']] * len(T))
    F.exptime = np.array([tim.primhdr['EXPTIME']] * len(T)).astype(np.float32)

    if len(tims) == 1:
        tim = tims[0]
        F.filter  = np.array([tim.band] * len(T))
        ok,x,y = tim.sip_wcs.radec2pixelxy(T.ra, T.dec)
        F.x = (x-1).astype(np.float32)
        F.y = (y-1).astype(np.float32)


    t1 = Time()
    print('Prior to forced photometry:', t1-t0)

    # FIXME -- should we run one band at a time?

    R = tr.optimize_forced_photometry(variance=True, fitstats=True,
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

    if opt.name is None:
        opt.name = '%i-%s' % (im.expnum, im.ccdname)
    outfn = 'euclid-out/forced/megacam-%s.fits' % opt.name
    fitsio.write(outfn, None, header=primhdr, clobber=True)
    F.writeto(outfn, header=hdr, append=True)
    print('Wrote', outfn)
            
    t3 = Time()
    print('Wrap-up:', t3-t2)
    print('Total:', t3-t0)

