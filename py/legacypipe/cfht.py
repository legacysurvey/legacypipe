from __future__ import print_function
import numpy as np
import os
from legacypipe.image import LegacySurveyImage, CP_DQ_BITS

'''
This is for the "pitcairn" reductions for CFIS-r data.

eg, search for data from here,
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/community/cfis/csky.html

and use "get data", then download a URL list, or grab a search box (here for u and r-band images)

http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/community/cfis/mcut.pl?&ra1=251&ra2=254.75&dec1=34.2&dec2=35.7&grid=true&images=true&tiles=false&fils=u2&fils=r2

and retrieve with:

wget -N --content-disposition -i ../urls.txt --http-user=dstn --http-password=$CADC_PASSWORD --auth-no-challenge

Zeropoints like:
python legacyzpts/legacy_zeropoints.py --psf --splinesky --calibdir cfis/calib --run-calibs --camera megaprime --image pitcairn/2106094p.fits.fz --not_on_proj --outdir cfis/zpts/ > 12.log 2>&1 &
ls cfis/zpts/2??????p-legacypipe.fits > zpts.txt
python legacyzpts/legacy_zeropoints_merge.py --nproc 0 --outname cfis/survey-ccds-cfis-pitcairn.fits --file_list zpts.txt

# Deep3 region:
fitscopy ~/cosmo/data/legacysurvey/dr6/survey-ccds-dr6plus.kd.fits+1"[(abs(ra-215)<2) && (abs(dec-52.75)<1) && ((filter=='g') || (filter=='z'))]" dr6-deep3.fits
fitsgetext -i dr6-deep3.fits -e 0 -e 1 -o cfis/survey-ccds-dr6-gz.fits
gzip cfis/survey-ccds-dr6-gz.fits

# Deep2-Field2 region:
fitscopy ~/cosmo/data/legacysurvey/dr6/survey-ccds-dr6plus.kd.fits+1"[(ra>251.0) && (ra<254.75) && (dec>34.2) && (dec<35.7) && ((filter=='g') || (filter=='z'))]" dr6-deep2f2.fits
fitsgetext -i dr6-deep2f2.fits -e 0 -e 1 -o cfis/survey-ccds-dr6-deep2f2-gz.fits
# CFIS search as above: RA 251.25 to 254.25 plus 0.5-deg margin, Dec 34.7 to 35.2 plus 0.5-deg margin
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/cadcbin/community/cfis/mcut.pl?&ra1=250.75&ra2=254.75&dec1=34.2&dec2=35.7&grid=true&images=true&tiles=false&fils=u2&fils=r2

http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2101680p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2101681p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2101682p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2101683p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2105431p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2105432p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2105433p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2105434p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2105660p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2105661p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2105662p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/2105663p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2114128p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2114135p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2116335p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2158878p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2158879p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2158880p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2158881p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2158882p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159064p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159065p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159066p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159067p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159068p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159073p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159215p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159216p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159217p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159354p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159355p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159356p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159357p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159578p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159579p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159580p.fits.fz
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/VOSPACE/cfis/pitcairn/2159581p.fits.fz

for u in 2101680 2101681 2101682 2101683 2105431 2105432 2105433 2105434 2105660 2105661 2105662 2105663; do
wget http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHTSG/${u}p.head;
vcp vos:cfis/weights/${u}p.weight.fits.fz .
vcp vos:cfis/flags/${u}p.flags.fits.fz .
done

for u in 2101680 2101681 2101682 2101683 2105431 2105432 2105433 2105434 2105660 2105661 2105662 2105663; do
funpack ${u}p.fits.fz;
done

for u in 2101680 2101681 2101682 2101683 2105431 2105432 2105433 2105434 2105660 2105661 2105662 2105663; do
muphead.pl ${u}p.fits ${u}p.head;
done

for r in 2114128 2114135 2116335 2158878 2158879 2158880 2158881 2158882 2159064 2159065 2159066 2159067 2159068 2159073 2159215 2159216 2159217 2159354 2159355 2159356 2159357 2159578 2159579 2159580 2159581; do
vcp vos:cfis/flags/${r}p.flag.fits.fz .
done

(not all these seem to exist in their processed locations yet)

For u-band, images are at URLs like
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHT/1234567p.fits.fz
with updated headers at URLs like
http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/CFHTSG/1234567p.head

Can update the files with the headers using:
  muphead.pl 1234567p.fits 1234567p.head

r-band images can be found:  
http://apps.canfar.net/storage/list/cfis/pitcairn
using http or vospace client.

The weight maps and flags maps are also being generated
(the r-band is done, the u-band should be done shortly)
They can be found in vos:cfis/weights and cfis/flags
respectively.

> vls vos:cfis/weights  # (takes a long time...)
...
2246825p.weight.fits.fz
2246826p.weight.fits.fz
2246827p.weight.fits.fz





# zpts like:
python legacyzpts/legacy_zeropoints.py --psf --splinesky --calibdir cfis/calib --run-calibs --camera megaprime --image pitcairn/$img --not_on_proj --outdir cfis/zpts/ --threads 8 > $log 2>&1

'''


class MegaPrimeImage(LegacySurveyImage):
    '''
    A LegacySurveyImage subclass to handle images from the MegaPrime
    camera on CFHT.
    '''
    def __init__(self, survey, t):
        super(MegaPrimeImage, self).__init__(survey, t)
        # Adjust zeropoint for exposure time
        self.ccdzpt += 2.5 * np.log10(self.exptime)
        # print('MegaPrimeImage: CCDs table entry', t)
        # for x in dir(t):
        #     if x.startswith('_'):
        #         continue
        #     print('  ', x, ':', getattr(t,x))

    def compute_filenames(self):
        self.dqfn = 'cfis/test.mask.0.40.01.fits'

    def remap_dq(self, dq, header):
        '''
        Called by get_tractor_image() to map the results from read_dq
        into a bitmask.  We only have a 0/1 bad pixel mask.
        '''
        dqbits = np.zeros(dq.shape, np.int16)
        dqbits[dq == 0] = CP_DQ_BITS['badpix']
        return dqbits

    def read_image(self, header=False, **kwargs):
        img = super(MegaPrimeImage, self).read_image(header=header, **kwargs)
        if header:
            img,hdr = img
        img = img.astype(np.float32)
        if header:
            return img,hdr
        return img

    def read_invvar(self, **kwargs):
        print('MegaPrimeImage.read_invvar')
        img = self.read_image(**kwargs)
        if self.sig1 is None:
            # Estimate per-pixel noise via Blanton's 5-pixel MAD
            slice1 = (slice(0,-5,10),slice(0,-5,10))
            slice2 = (slice(5,None,10),slice(5,None,10))
            mad = np.median(np.abs(img[slice1] - img[slice2]).ravel())
            sig1 = 1.4826 * mad / np.sqrt(2.)
            self.sig1 = sig1
            print('Computed sig1 by Blanton method:', self.sig1)
        else:
            print('sig1 from CCDs file:', self.sig1)

        iv = np.zeros_like(img) + (1./self.sig1**2)
        return iv



    # calibs



    def run_se(self, imgfn, maskfn):
        from astrometry.util.file import trymakedirs
        sedir = self.survey.get_se_dir()
        trymakedirs(self.sefn, dir=True)
        # We write the SE catalog to a temp file then rename, to avoid
        # partially-written outputs.

        from legacypipe.survey import create_temp
        import fitsio

        tmpmaskfn = create_temp(suffix='.fits')
        # The test.mask file has 1 for good pix, 0 for bad... invert this for SE
        goodpix = fitsio.read(maskfn)
        fitsio.write(tmpmaskfn, (1-goodpix).astype(np.uint8), clobber=True)

        tmpfn = os.path.join(os.path.dirname(self.sefn),
                             'tmp-' + os.path.basename(self.sefn))
        cmd = ' '.join([
            'sex',
            '-c', os.path.join(sedir, self.camera + '.se'),
            '-PARAMETERS_NAME', os.path.join(sedir, self.camera + '.param'),
            '-FILTER_NAME %s' % os.path.join(sedir, self.camera + '.conv'),
            '-FLAG_IMAGE %s' % tmpmaskfn,
            '-CATALOG_NAME %s' % tmpfn,
            '-SEEING_FWHM %f' % 0.8,
            '-FITS_UNSIGNED N',
            #'-VERBOSE_TYPE FULL',
            #'-PIXEL_SCALE 0.185',
            #'-SATUR_LEVEL 100000',
            imgfn])
        print(cmd)
        rtn = os.system(cmd)
        if rtn:
            raise RuntimeError('Command failed: ' + cmd)
        os.rename(tmpfn, self.sefn)

        os.unlink(tmpmaskfn)

    # def funpack_files(self, imgfn, dqfn, hdu, todelete):
    #     ''' Source Extractor can't handle .fz files, so unpack them.'''
    #     from legacypipe.survey import create_temp
    #     tmpimgfn = None
    #     # For FITS files that are not actually fpack'ed, funpack -E
    #     # fails.  Check whether actually fpacked.
    #     fcopy = False
    #     hdr = fitsio.read_header(imgfn, ext=hdu)
    #     if not ((hdr['XTENSION'] == 'BINTABLE') and hdr.get('ZIMAGE', False)):
    #         print('Image %s, HDU %i is not fpacked; just imcopying.' %
    #               (imgfn,  hdu))
    #         fcopy = True
    # 
    #     tmpimgfn  = create_temp(suffix='.fits')
    #     todelete.append(tmpimgfn)
    #     
    #     if fcopy:
    #         cmd = 'imcopy %s"+%i" %s' % (imgfn, hdu, tmpimgfn)
    #     else:
    #         cmd = 'funpack -E %i -O %s %s' % (hdu, tmpimgfn, imgfn)
    #     print(cmd)
    #     if os.system(cmd):
    #         raise RuntimeError('Command failed: ' + cmd)
    #     
    #     if fcopy:
    #         cmd = 'imcopy %s"+%i" %s' % (maskfn, hdu, tmpmaskfn)
    #     else:
    #         cmd = 'funpack -E %i -O %s %s' % (hdu, tmpmaskfn, maskfn)
    #     print(cmd)
    #     if os.system(cmd):
    #         print('Command failed: ' + cmd)
    #         M,hdr = self._read_fits(maskfn, hdu, header=True)
    #         print('Read', M.dtype, M.shape)
    #         fitsio.write(tmpmaskfn, M, header=hdr, clobber=True)
    #         print('Wrote', tmpmaskfn, 'with fitsio')
    # 
    #     return tmpimgfn,tmpmaskfn
    #     #imgfn,maskfn = self.
