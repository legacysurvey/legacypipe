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
		#table['image_key'].append(image_key)
		table['image_hdu'].append(0)
		table['camera'].append('ztf')
		table['expnum'].append(header['EXPID'])
		table['ccdname'].append(header['EXTNAME'])
		table['object'].append('None')
		table['propid'].append(header['PROGRMID'])
		table['filter'].append(filter_tbl[header['FILTERID']])
		table['exptime'].append(header['EXPTIME'])
		table['mjd_obs'].append(header['OBSMJD'])
		table['fwhm'].append(header['C3SEE'])
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
		table['sig1'].append(header['C3SKYSIG'])
		table['ccdzpt'].append(header['C3ZP'])
		table['zpt'].append(0)		
		table['ccdraoff'].append(0)
		table['ccddecoff'].append(0)
		table['ccdskycounts'].append(0)
		table['ccdrarms'].append(0)
		table['ccddecrms'].append(0)
		table['ccdphrms'].append(0)
		table['ccdnmatch'].append(0)
		table['ccd_cuts'].append(0)

	#f_table = fits_table()
	nrows=len(image_list)	
	print(nrows)	
	f_table = fits.BinTableHDU.from_columns(
	fits.ColDefs([fits.Column( name = 'image_filename', format = '100A'), 
	fits.Column( name = 'image_hdu', format = 'I'), 
	fits.Column( name = 'camera', format = '5A'), 
	fits.Column( name = 'expnum', format = 'K'), 
	fits.Column( name = 'ccdname', format = '3A'), 
	fits.Column( name = 'object', format = '35A'), 
	fits.Column( name = 'propid', format = '10A'), 
	fits.Column( name = 'filter', format = '1A'), 
	fits.Column( name = 'exptime', format = 'E'), 
	fits.Column( name = 'mjd_obs', format = 'D'), 
	fits.Column( name = 'fwhm', format = 'E'), 
	fits.Column( name = 'width', format = 'I'), 
	fits.Column( name = 'height', format = 'I'), 
	fits.Column( name = 'ra_bore', format = 'D'), 
	fits.Column( name = 'dec_bore', format = 'D'), 
	fits.Column( name = 'crpix1', format = 'E'), 
	fits.Column( name = 'crpix2', format = 'E'), 
	fits.Column( name = 'crval1', format = 'D'), 
	fits.Column( name = 'crval2', format = 'D'), 
	fits.Column( name = 'cd1_1', format = 'E'), 
	fits.Column( name = 'cd1_2', format = 'E'), 
	fits.Column( name = 'cd2_1', format = 'E'), 
	fits.Column( name = 'cd2_2', format = 'E'), 
	fits.Column( name = 'yshift', format = 'L'), 
	fits.Column( name = 'ra', format = 'D'), 
	fits.Column( name = 'dec', format = 'D'), 
	fits.Column( name = 'skyrms', format = 'E'), 
	fits.Column( name = 'sig1', format = 'E'), 
	fits.Column( name = 'ccdzpt', format = 'E'), 
	fits.Column( name = 'zpt', format = 'E'), 
	fits.Column( name = 'ccdraoff', format = 'E'), 
	fits.Column( name = 'ccddecoff', format = 'E'), 
	fits.Column( name = 'ccdskycounts', format = 'E'), 
	fits.Column( name = 'ccdrarms', format = 'E'), 
	fits.Column( name = 'ccddecrms', format = 'E'), 
	fits.Column( name = 'ccdphrms', format = 'E'), 
	fits.Column( name = 'ccdnmatch', format = 'I'), 
	fits.Column( name = 'ccd_cuts', format = 'K')]),nrows=nrows) 


	for colname in f_table.columns.names:
		f_table.data[colname] = table[colname]

	
	prihdr = fits.Header()

	prihdu = fits.PrimaryHDU(header=prihdr)
	thdulist = fits.HDUList([prihdu, f_table])
	
	#for key in table:
	#    f_table.set(key,table[key])

	table_name = outfolder+'/survey-ccds-ztfmerge2.fits'
	thdulist.writeto(table_name)

	if os.path.exists(table_name+'.gz'):
	    os.remove(table_name+'.gz')

	os.system('gzip %s'%table_name)

if __name__ == '__main__':
	
	folder = sys.argv[1]
	outfolder = sys.argv[2]
	CreateCCDTable(folder,outfolder)
