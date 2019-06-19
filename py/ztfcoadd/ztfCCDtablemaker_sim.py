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

def CreateCCDTable(folder,outfolder):

	#image_list = glob.glob(folder+'/images/*sciimg.fits')
	image_list = np.loadtxt(folder+'/scie.list',dtype=str)
	filter_tbl = {1: 'g',
	              2: 'r',
	              3: 'i'}

	table = defaultdict(list)

	slashsplit = lambda x : x.split('/')[-1]
	slashdir = lambda x : '/'.join(x.split('/')[:-1])

	for image in image_list:
		print(image)
		if '/' in image:
			image_key = '_'.join(slashsplit(image).split('_')[:7])
			image_fname = slashsplit(image)
		else:
			image_key = '_'.join(image.split('_')[:7])
			image_fname = image

		with fits.open(image) as f:
			header = f[0].header
		try:
			table['fwhm'].append(header['C3SEE'])
		except KeyError:
			try:
				table['fwhm'].append(header['MEDFWHM'])
			except KeyError:
			#	try:
				table['fwhm'].append(header['SEEING'])
				#except KeyError:
			#		print('missing C3SEE')
			#		continue
		table['image_filename'].append(image_fname)
		#table['image_key'].append(image_key)
		table['image_hdu'].append(0)
		table['camera'].append('ztf')
		try:
			table['expnum'].append(header['EXPID'])
		except KeyError:
			table['expnum'].append(57119266)
		try:
			table['ccdname'].append(header['EXTNAME'])
		except KeyError:
			table['ccdname'].append('4')
		table['object'].append('None')
		try:
			table['propid'].append(header['PROGRMID'])
		except KeyError:
			table['propid'].append(2)
		try:
			table['filter'].append(filter_tbl[header['FILTERID']])
		except KeyError:
			table['filter'].append('1')
		try:
			table['exptime'].append(header['EXPTIME'])
		except KeyError:
			table['exptime'].append(header['TOTEXPT'])
		try:
			table['mjd_obs'].append(header['OBSMJD'])
		except KeyError:
			table['mjd_obs'].append(header['JDEND'])
		table['width'].append(header['NAXIS1'])
		table['height'].append(header['NAXIS2'])
		table['ra_bore'].append(0)
		table['dec_bore'].append(0)
		table['crpix1'].append(header['CRPIX1'])
		table['crpix2'].append(header['CRPIX2'])
		table['crval1'].append(header['CRVAL1'])
		table['crval2'].append(header['CRVAL2'])
		table['cd1_1'].append(header['CD1_1'])
		table['cd1_2'].append(header['CD1_2'])
		table['cd2_1'].append(header['CD2_1'])
		table['cd2_2'].append(header['CD2_2'])
		table['yshift'].append(0)
		table['ra'].append(float(header['CRVAL1']))
		table['dec'].append(float(header['CRVAL2']))
		table['skyrms'].append(0)
		try:
			table['sig1'].append(header['C3SKYSIG'])
		except KeyError:
			try:
				table['sig1'].append(header['GSTDDEV'])
			except KeyError:
				table['sig1'].append(header['ASTCHI2'])
		try:
			table['ccdzpt'].append(header['C3ZP'])
		except KeyError:
			table['ccdzpt'].append(header['MAGZP'])

		#table['zpt'].append(0)		
		table['ccdraoff'].append(0)
		table['ccddecoff'].append(0)
		#table['ccdskycounts'].append(0)
		#table['ccdrarms'].append(0)
		#table['ccddecrms'].append(0)
		#table['ccdphrms'].append(0)
		#table['ccdnmatch'].append(0)
		#table['ccd_cuts'].append(0)
		#print(header['C3SKYSIG'])
	f_table = fits_table()
	for key in table:
	    f_table.set(key,table[key])
	
	table_name = outfolder+'/survey-ccds-ztf.fits'
	
	f_table.write_to(table_name)
	os.system('gzip %s'%table_name)
	#if os.path.exists(table_name+'.gz'):
	#    os.remove(table_name+'.gz')

	

if __name__ == '__main__':
	
	folder = sys.argv[1]
	outfolder = sys.argv[2]
	CreateCCDTable(folder,outfolder)
