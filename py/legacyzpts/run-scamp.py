import os
import tempfile
import shutil
import fitsio
import numpy as np
from astrometry.util.fits import fits_table
from astrometry.util.file import trymakedirs
from astrometry.util.util import Tan

def write_one_scamp_catalog(photom_fn, scamp_dir, survey_dir, photom_base_dir):
    fn = photom_fn
    
    if photom_base_dir is None:
        # Assume photom filenames are like SURVEY_DIR/zpt/DIRS/filename
        parts = fn.split('/')
        try:
            izpt = parts.index('zpt')
            relpath = '/'.join(parts[izpt+1:])
            if survey_dir is None:
                survey_dir = '/'.join(parts[:izpt])
        except:
            relpath = fn
    else:
        relpath = fn.replace(photom_base_dir, '')

    # Check for existing output file & skip
    tmpoutfn  = relpath.replace('-photom.fits', '-scamp.tmp.fits')
    realoutfn = relpath.replace('-photom.fits', '-scamp.fits')

    rtn = realoutfn
    
    tmpoutfn  = os.path.join(scamp_dir, tmpoutfn)
    realoutfn = os.path.join(scamp_dir, realoutfn)
    if os.path.exists(realoutfn):
        #print('Exists:', realoutfn)
        return rtn

    # Compute image filename
    print('Relative path', relpath)
    imgfn = os.path.join(survey_dir, 'images', relpath).replace('-photom.fits',
                                                                '.fits')
    if not os.path.exists(imgfn) and os.path.exists(imgfn + '.fz'):
        imgfn += '.fz'
    #print('Img filename', imgfn)
    P = fits_table(fn)
    P.sn = P.flux/P.dflux
    hdr = P.get_header()
    #print('Stars:', len(P), 'for', fn)
    newhdr = fitsio.FITSHDR()

    for k in ['AIRMASS', 'OBJECT', 'TELESCOP', 'INSTRUME', 'EXPTIME', 'DATE-OBS',
              'MJD-OBS', 'FILTER', 'EXPNUM',
              'RA_BORE', 'DEC_BORE', 'CCD_ZPT', 'FWHM', 'SEEING', 'FILENAME']:
        print('  ', k, '=', hdr[k])
        newhdr[k] = hdr[k]
    # HA doesn't exist in some CFHT image headers
    for k in ['HA']:
        v = hdr.get(k)
        if v is not None:
            print('  ', k, '=', hdr[k])
            newhdr[k] = v

    trymakedirs(tmpoutfn, dir=True)
    F = fitsio.FITS(tmpoutfn, 'rw', clobber=True)
    ccds = np.unique(P.ccdname)
    ngood = []
    for ccd in ccds:
        I1 = np.flatnonzero((P.ccdname == ccd))
        I2 = np.flatnonzero((P.ccdname == ccd) * (P.ra_gaia != 0.0))
        #print('  CCD', ccd, ':', len(I), 'stars, 10/50/90th pct S/N:', ', '.join(['%.1f' % p for p in np.percentile(P.sn[I], [10,50,90])]))
        I = np.flatnonzero((P.ccdname == ccd) * (P.ra_gaia != 0.0) * (P.sn > 5.))
        #print('CCD', ccd, ':', len(I1), 'in CCD,', len(I2), 'with Gaia RA,', len(I), 'with S/N > 5.  Median S/N %.1f' % np.median(P.sn[I2]))
        ngood.append(len(I))
        try:
            imghdr = fitsio.read_header(imgfn, ext=ccd)
            w,h = imghdr['ZNAXIS1'], imghdr['ZNAXIS2']
        except:
            # Older images, eg cfht-s82-u/931347p.fits.fz, suffer from
            # https://github.com/esheldon/fitsio/issues/324
            # Try reading with astropy.
            from astropy.io import fits as afits
            f = afits.open(imgfn)
            hdu = f[ccd]
            h,w = hdu.shape
            imghdr = hdu.header
        newhdr['EXTNAME'] = ccd
        for c in ['QRUNID']:
            newhdr[c] = imghdr[c]
        # Read Astrometry.net initial WCS header!
        imgid = os.path.basename(imgfn).replace('.fits','').replace('.fz', '')
        wcsfn = imgfn.replace('images', 'calib/wcs-initial').replace('.fits', '').replace('.fz','') + '/%s-%s.wcs' % (imgid, ccd)
        #print('WCS', wcsfn)

        # Reproject to a shared CRVAL (with large CRPIX values)
        # Primary header has target RA,Dec as CRVAL; later HDUs all have the same CRVAL, somewhat off.
        primhdr = fitsio.read_header(imgfn)
        cra,cdec = primhdr['CRVAL1'], primhdr['CRVAL2']
        wcs = Tan(wcsfn)
        ok,cx,cy = wcs.radec2pixelxy(cra, cdec)
        wcs.set_crval(cra, cdec)
        wcs.set_crpix(cx, cy)

        cd = wcs.get_cd()
        newhdr['EQUINOX'] = 2000.
        newhdr['CRPIX1'] = wcs.crpix[0]
        newhdr['CRPIX2'] = wcs.crpix[1]
        newhdr['CRVAL1'] = wcs.crval[0]
        newhdr['CRVAL2'] = wcs.crval[1]
        newhdr['CD1_1'] = cd[0]
        newhdr['CD1_2'] = cd[1]
        newhdr['CD2_1'] = cd[2]
        newhdr['CD2_2'] = cd[3]
        # wcshdr = fitsio.read_header(wcsfn)
        # for c in ['EQUINOX', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
        #           'CRVAL1', 'CRVAL2']:
        #     newhdr[c] = wcshdr[c]
        newhdr['CTYPE1'] = 'RA---TAN' # ... trim off the SIP
        newhdr['CTYPE2'] = 'DEC--TAN'
        newhdr['RADECSYS'] = 'FK5' # ... not really but it's what's in the CFHT headers, so scamp understands that, if it pays any attention

        # f,tmpfn = tempfile.mkstemp(suffix='.fits')
        # os.close(f)
        # fitsio.write(tmpfn, None, header=newhdr, extname=ccd, clobber=True)
        # hdrtxt = open(tmpfn, 'rb').read()
        # os.remove(tmpfn)

        fits = fitsio.FITS('mem://', 'rw')
        fits.write(None, header=newhdr, extname=ccd)
        hdrtxt = fits.read_raw()
        fits.close()

        # Ugh, it is so awkward to write out simple FITS headers!
        hdrtxt = hdrtxt.replace(
            b'NAXIS   =                    0 / number of data axes                            ',
            b'NAXIS   =                    2 / number of data axes                            '+
            b'NAXIS1  =                 %4i / number of data axes                            ' % w+
            b'NAXIS2  =                 %4i / number of data axes                            ' % h)
        hdr = np.zeros((1, len(hdrtxt)//80, 80), 'S1')
        hdr.data = hdrtxt
        F.write([hdr], names=['Field Header Card'],extname='LDAC_IMHEAD')

        # ignore fwhm
        err_pix = 1./P.sn[I]
        zero = np.zeros(len(I), np.float32)
        F.write([1.+P.x_fit[I], 1.+P.y_fit[I], err_pix, err_pix,
                 zero, P.flux[I], P.dflux[I], P.bitmask[I]],
                names=[c.upper() for c in ['x_image', 'y_image', 'err_a', 'err_b',
                                           'err_theta', 'flux', 'flux_err', 'flags']],
                extname='LDAC_OBJECTS')
    #print('Good stars per CCD for', fn, ': min %i, mean %.1f, median %.1f, max %.1f' % (np.min(ngood), np.mean(ngood), np.median(ngood), np.max(ngood)))
    #'[' + ', '.join([str(n) for n in ngood]) + ']')
    F.close()
    os.rename(tmpoutfn, realoutfn)
    #print('Wrote', realoutfn)
    return rtn

def _bounce_write_one(X):
    return write_one_scamp_catalog(*X)

def write_scamp_catalogs(scamp_dir, photom_fns, survey_dir, photom_base_dir, mp=None):
    if mp is None:
        relpaths = []
        for fn in photom_fns:
            outfn = write_one_scamp_catalog(fn, scamp_dir, survey_dir, photom_base_dir)
            relpaths.append(outfn)
        return relpaths
    return mp.map(_bounce_write_one, [(fn, scamp_dir, survey_dir, photom_base_dir)
                                      for fn in photom_fns])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('photom', metavar='photom-filename', nargs='+',
                        help='*-photom.fits files to process')
    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Base directory to move the Scamp outputs to')
    parser.add_argument('--scamp-config', type=str, default='scamp.conf',
                        help='Scamp configuration file')
    parser.add_argument('--scamp-dir', type=str, default=None,
                        help='Directory to write Scamp data to; default is a temp dir')
    parser.add_argument('--photom-base-dir', type=str, default=None,
                        help='Directory to trim from the *-photom.fits files to make relative paths')
    parser.add_argument('--threads', type=int, default=None,
                        help='Write scamp files in parallel')
    parser.add_argument('--scamp-command', type=str,
                        default='shifter --image docker:legacysurvey/legacypipe:DR10.3.1 scamp',
                        help='Set scamp command to run, default %(default)s')
    args = parser.parse_args()

    if args.scamp_dir is None:
        scamp_dir = tempfile.mkdtemp(prefix='scamp')
    else:
        scamp_dir = args.scamp_dir

    scamp_config = os.path.abspath(args.scamp_config)

    mp = None
    if args.threads:
        from astrometry.util.multiproc import multiproc
        mp = multiproc(nthreads=args.threads)

    scampfiles = write_scamp_catalogs(scamp_dir, args.photom, args.survey_dir,
                                      args.photom_base_dir, mp=mp)

    scamp_cmd = ('cd %s && %s -c %s %s' %
                 (scamp_dir, args.scamp_command, scamp_config, ' '.join(scampfiles)))
    print(scamp_cmd)
    r = os.system(scamp_cmd)
    assert(r == 0)

    survey_dir = args.survey_dir
    if survey_dir is None:
        survey_dir = '.'

    for fn in scampfiles:
        origfn = fn
        fn = fn.replace('-scamp.fits', '-scamp.head')
        infn = os.path.join(scamp_dir, fn)
        outfn = os.path.join(survey_dir, 'calib', 'wcs-scamp', fn)
        trymakedirs(outfn, dir=True)
        print('Scamp file:', origfn, 'copy header', infn, 'to', outfn)
        shutil.copyfile(infn, outfn)
        #os.rename(infn, outfn)
        print('Copied', outfn)

if __name__ == '__main__':
    main()
