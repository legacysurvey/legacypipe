import os
import tempfile
import shutil
import fitsio
import numpy as np
from astrometry.util.fits import fits_table
from astrometry.util.file import trymakedirs

def write_scamp_catalogs(scamp_dir, photom_fns, survey_dir):
    relpaths = []

    f,tmpfn = tempfile.mkstemp(suffix='.fits')
    os.close(f)

    for fn in photom_fns:

        # Assume photom filenames are like SURVEY_DIR/zpt/DIRS/filename
        parts = fn.split('/')
        try:
            izpt = parts.index('zpt')
            relpath = '/'.join(parts[izpt+1:])
            if survey_dir is None:
                survey_dir = '/'.join(parts[:izpt])
        except:
            relpath = fn

        # Check for existing output file & skip
        tmpoutfn  = relpath.replace('-photom.fits', '-scamp.tmp.fits')
        realoutfn = relpath.replace('-photom.fits', '-scamp.fits')
        relpaths.append(realoutfn)
        
        tmpoutfn  = os.path.join(scamp_dir, tmpoutfn)
        realoutfn = os.path.join(scamp_dir, realoutfn)
        if os.path.exists(realoutfn):
            print('Exists:', realoutfn)
            continue

        # Compute image filename
        imgfn = os.path.join(survey_dir, 'images', relpath).replace('-photom.fits',
                                                                    '.fits')
        if not os.path.exists(imgfn) and os.path.exists(imgfn + '.fz'):
            imgfn += '.fz'
            
        P = fits_table(fn)
        P.sn = P.flux/P.dflux
        hdr = P.get_header()
        print('Stars:', len(P), 'for', fn)

        newhdr = fitsio.FITSHDR()
        for k in ['AIRMASS', 'OBJECT', 'TELESCOP', 'INSTRUME', 'EXPTIME', 'DATE-OBS',
                  'MJD-OBS', 'FILTER', 'HA', 'EXPNUM',
                  'RA_BORE', 'DEC_BORE', 'CCD_ZPT', 'FWHM', 'SEEING', 'FILENAME']:
            newhdr[k] = hdr[k]

        trymakedirs(tmpoutfn, dir=True)
        F = fitsio.FITS(tmpoutfn, 'rw', clobber=True)

        ccds = np.unique(P.ccdname)
        ngood = []
        for ccd in ccds:
            I = np.flatnonzero((P.ccdname == ccd) * (P.ra_gaia != 0.0))
            #print('  CCD', ccd, ':', len(I), 'stars, 10/50/90th pct S/N:', ', '.join(['%.1f' % p for p in np.percentile(P.sn[I], [10,50,90])]))
            I = np.flatnonzero((P.ccdname == ccd) * (P.ra_gaia != 0.0) * (P.sn > 5.))
            #print(len(I), 'good stars for CCD', ccd)
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
            for c in ['EQUINOX', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                      'CRVAL1', 'CRVAL2', 'QRUNID', 'CTYPE1', 'CTYPE2', 'RADECSYS']:
                newhdr[c] = imghdr[c]

            fitsio.write(tmpfn, None, header=newhdr, extname=ccd, clobber=True)
            hdrtxt = open(tmpfn, 'rb').read()
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
        print('Good stars per CCD for', fn, ': min %i, mean %.1f' % (np.min(ngood), np.mean(ngood)))
        #'[' + ', '.join([str(n) for n in ngood]) + ']')
        F.close()
        os.rename(tmpoutfn, realoutfn)
        print('Wrote', realoutfn)
    return relpaths

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
    args = parser.parse_args()

    if args.scamp_dir is None:
        scamp_dir = tempfile.mkdtemp(prefix='scamp')
    else:
        scamp_dir = args.scamp_dir

    scamp_config = os.path.abspath(args.scamp_config)
        
    scampfiles = write_scamp_catalogs(scamp_dir, args.photom, args.survey_dir)

    scamp_exe = 'shifter --image docker:legacysurvey/legacypipe:DR10.2b scamp'
    
    scamp_cmd = ('cd %s && %s -c %s %s' %
                 (scamp_dir, scamp_exe, scamp_config, ' '.join(scampfiles)))
    print(scamp_cmd)
    r = os.system(scamp_cmd)
    assert(r == 0)

    survey_dir = args.survey_dir
    if survey_dir is None:
        survey_dir = '.'

    for fn in scampfiles:
        fn = fn.replace('-scamp.fits', '-scamp.head')
        infn = os.path.join(scamp_dir, fn)
        outfn = os.path.join(survey_dir, 'calib', 'wcs-scamp', fn)
        trymakedirs(outfn, dir=True)
        shutil.copyfile(infn, outfn)
        #os.rename(infn, outfn)
        print('Copied', outfn)

if __name__ == '__main__':
    main()
