#!/usr/bin/env python

"""
Build CCD table for ZTF Coadd
"""

import sys
import numpy as np
import os
import glob
from collections import defaultdict
from astropy.io import fits
from astrometry.util.fits import fits_table

__whatami__ = 'ZTF CCD Table Maker'
__author__ = 'Michael Medford <MichaelMedford@berkeley.edu>'

def CreateCCDTable(image_list):

	image_list = glob.glob(folder+'/images/*sciimg.fits')

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

	table_name = folder+'/survey-ccds-1.fits'

	if os.path.exists(table_name+'.gz'):
	    os.remove(table_name+'.gz')

	os.system('gzip %s'%table_name)

if __name__ == '__main__':

	folder = sys.argv[1]
	CreateCCDTable(folder)
