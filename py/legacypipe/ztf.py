from __future__ import print_function
import numpy as np
import os
import glob
from collections import defaultdict

import astropy.time
from astropy.io import fits

from legacypipe.image import LegacySurveyImage, CP_DQ_BITS
from astrometry.util.fits import fits_table

'''
Code specific to images from the Zwicky Transient Facility (ZTF).
'''

def CreateCCDTable(image_list):

    filter_tbl = {1: 'g',
                  2: 'r',
                  3: 'i'}

    table = defaultdict(list)

    slashsplit = lambda x : x.split('/')[-1]
    slashdir = lambda x : '/'.join(x.split('/')[:-1])

    for image in image_list:

        if '/' in image:
            image_key = '_'.join(slashsplit(image).split('_')[:7])
            image_fname = slashsplit(image)
        else:
            image_key = '_'.join(image.split('_')[:7])
            image_fname = image
        
        table_name = 'survey-ccds-1.fits'

        with fits.open(image) as f:
            header = f[0].header

        table['image_filename'].append(image_fname)
        table['image_key'].append(image_key)
        table['image_hdu'].append(0)
        table['expnum'].append(header['EXPID'])
        table['ccdname'].append(header['EXTNAME'])
        table['filter'].append(filter_tbl[header['FILTERID']])
        table['exptime'].append(header['EXPTIME'])
        table['camera'].append('ztf')
        table['fwhm'].append(header['C3SEE'])
        table['propid'].append(header['PROGRMID'])
        table['mjd_obs'].append(header['OBSMJD'])
        table['width'].append(header['NAXIS1'])
        table['height'].append(header['NAXIS2'])
        table['sig1'].append(header['C3SKYSIG'])
        table['ccdzpt'].append(header['C3ZP'])
        table['ccdraoff'].append(0)
        table['ccddecoff'].append(0)
        table['cd1_1'].append(header['CD1_1'])
        table['cd1_2'].append(header['CD1_2'])
        table['cd2_1'].append(header['CD2_1'])
        table['cd2_2'].append(header['CD2_2'])
        table['ra'].append(float(header['CRVAL1']))
        table['dec'].append(float(header['CRVAL2']))
        table['crval1'].append(header['CRVAL1'])
        table['crval2'].append(header['CRVAL2'])
        table['crpix1'].append(header['CRPIX1'])
        table['crpix2'].append(header['CRPIX2'])

    f_table = fits_table()
    for key in table:
        f_table.set(key,table[key])

    f_table.write_to(table_name)

    if os.path.exists(table_name+'.gz'):
        os.remove(table_name+'.gz')

    os.system('gzip %s'%table_name)

class ZtfImage(LegacySurveyImage):
    '''
    A LegacySurveyImage subclass to handle images from the 
    Zwicky Transient Facility (ZTF) at Palomar Observatory.
    '''
    def __init__(self, survey, ccd):
        super(ZtfImage, self).__init__(survey, ccd)


    def compute_filenames(self):
        """Assume your image filename is here: self.imgfn"""
        self.dqfn = self.imgfn.replace('sciimg', 'mskimg')
        self.wtfn = self.imgfn.replace('sciimg', 'weight')
        assert(self.dqfn != self.imgfn)
        assert(self.wtfn != self.imgfn)

        for attr in ['imgfn', 'dqfn', 'wtfn']:
            fn = getattr(self, attr)
            if os.path.exists(fn):
                continue

    def read_dq(self, **kwargs):
        '''return bit mask which Tractor calls "data quality" image
        ZTF DMASK BIT DEFINITIONS
        BIT00 = 0 / AIRCRAFT/SATELLITE TRACK
        BIT01 = 1 / OBJECT (detected by SExtractor)
        BIT02 = 2 / LOW RESPONSIVITY
        BIT03 = 3 / HIGH RESPONSIVITY
        BIT04 = 4 / NOISY
        BIT05 = 5 / GHOST ** not yet implemented **
        BIT06 = 6 / CCD BLEED ** not yet implemented **
        BIT07 = 7 / PIXEL SPIKE (POSSIBLE RAD HIT)
        BIT08 = 8 / SATURATED
        BIT09 = 9 / DEAD (UNRESPONSIVE)
        BIT10 = 10 / NAN (not a number)
        BIT11 = 11 / CONTAINS PSF-EXTRACTED SOURCE POSITION
        BIT12 = 12 / HALO ** not yet implemented **
        BIT13 = 13 / RESERVED FOR FUTURE USE
        BIT14 = 14 / RESERVED FOR FUTURE USE
        BIT15 = 15 / RESERVED FOR FUTURE USE
        '''
        print('Reading data quality image from', dqfn, 'hdu', hdu)
        mask = fitsio.read(dqfn, ext=hdu, header=False)
        cond1 = mask & 6141 != 0
        mask[cond1] = 1
        mask[!cond1] = 0
        return mask.astype(np.int16)
