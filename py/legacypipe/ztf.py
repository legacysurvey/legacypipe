from __future__ import print_function
import numpy as np
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
            table_name = slashdir(image) + '/CCD_table.fits'
        else:
            image_key = '_'.join(image.split('_')[:7])
            table_name = 'CCD_table.fits'

        with fits.open(image) as f:
            header = f[0].header

        table['image_key'].append(image_key)
        table['image_hdu'].append(0)
        table['expnum'].append(header['EXPOSURE'])
        table['ccdname'].append(header['RCID'])
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

    f_table = fits_table()
    for image_key in table:
        f_table.set(image_key,table[image_key])

    f_table.write_to(table_name)

class ZtfImage(LegacySurveyImage):
    '''
    A LegacySurveyImage subclass to handle images from the 
    Zwicky Transient Facility (ZTF) at Palomar Observatory.
    '''
    def __init__(self, survey, ccd):
        super(ZtfImage, self).__init__(survey, ccd)


    def compute_filenames(self):
        """Assume your image filename is here: self.imgfn"""
        self.dqfn = self.imgfn.replace('_ooi_', '_ood_').replace('_oki_','_ood_')
        self.wtfn = self.imgfn.replace('_ooi_', '_oow_').replace('_oki_','_oow_')
        assert(self.dqfn != self.imgfn)
        assert(self.wtfn != self.imgfn)

        for attr in ['imgfn', 'dqfn', 'wtfn']:
            fn = getattr(self, attr)
            if os.path.exists(fn):
                continue
