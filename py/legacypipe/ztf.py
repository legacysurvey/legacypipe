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
        # Adjust zeropoint for exposure time
        self.ccdzpt += 2.5 * np.log10(self.exptime)

    def read_invvar(self, **kwargs):
        return self.read_invvar_clipped(**kwargs)

    glowmjd = astropy.time.Time('2014-08-01').utc.mjd

    def get_good_image_subregion(self):
        x0,x1,y0,y1 = None,None,None,None

        # Handle 'glowing' edges in DES r-band images
        # aww yeah
        if self.band == 'r' and (
                ('DES' in self.imgfn) or ('COSMOS' in self.imgfn) or
                (self.mjdobs < DecamImage.glowmjd)):
            # Northern chips: drop 100 pix off the bottom
            if 'N' in self.ccdname:
                print('Clipping bottom part of northern DES r-band chip')
                y0 = 100
            else:
                # Southern chips: drop 100 pix off the top
                print('Clipping top part of southern DES r-band chip')
                y1 = self.height - 100

        # Clip the bad half of chip S7.
        # The left half is OK.
        if self.ccdname == 'S7':
            print('Clipping the right half of chip S7')
            x1 = 1023

        return x0,x1,y0,y1

    def remap_dq(self, dq, hdr):
        '''
        Called by get_tractor_image() to map the results from read_dq
        into a bitmask.
        '''
        from distutils.version import StrictVersion
        # The format of the DQ maps changed as of version 3.5.0 of the
        # Community Pipeline.  Handle that here...
        primhdr = self.read_primary_header(self.dqfn)
        plver = primhdr['PLVER'].strip()
        plver = plver.replace('V','')
        plver = plver.replace('DES ', '')
        plver = plver.replace('+1', 'a1')
        if StrictVersion(plver) >= StrictVersion('3.5.0'):
            dq = self.remap_dq_cp_codes(dq, hdr)
        else:
            from legacypipe.image import CP_DQ_BITS
            dq = dq.astype(np.int16)
            # Un-set the SATUR flag for pixels that also have BADPIX set.
            bothbits = CP_DQ_BITS['badpix'] | CP_DQ_BITS['satur']
            I = np.flatnonzero((dq & bothbits) == bothbits)
            if len(I):
                print('Warning: un-setting SATUR for', len(I),
                      'pixels with SATUR and BADPIX set.')
                dq.flat[I] &= ~CP_DQ_BITS['satur']
                assert(np.all((dq & bothbits) != bothbits))
        return dq
