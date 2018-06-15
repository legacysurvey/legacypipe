#!/usr/bin/env python

""" Prepares folders for coadd creation. """
__author__ = 'Michael Medford <MichaelMedford@berkeley.edu'

import glob
import os
import numpy as np
from astropy.io import fits
import psycopg2
import shutil
from ztfcoadd import zpsee
from ztfcoadd import utils

def delete_folder_contents(folder, debug):

	utils.print_d("CLEARING FOLDER: %s ..."%(folder),debug)

	keep_files = [s for s in glob.glob(folder+"/ztf*sciimg.fits") if not 'coadd' in s]
	keep_files += [s for s in glob.glob(folder+"/ztf*mskimg.fits") if not 'coadd' in s]
	keep_files.sort()

	all_files = glob.glob(folder+"/*")

	delete_files = [f for f in all_files if f not in keep_files]

	for j,f in enumerate(delete_files):
		if os.path.isfile(f):
			os.remove(f)

def create_lists(folder, debug):

	utils.print_d("Creating scie_list for %s ..."%(folder),debug)
	scie_list = np.sort(np.atleast_1d(glob.glob(folder+'/ztf*sciimg.fits')))
	utils.replace_list(scie_list,folder+"/scie.list")

def download_real_stars(folder, debug):

	utils.print_d("Downloading real_stars for %s ..."%(folder),debug)
	scie_list = np.atleast_1d(np.genfromtxt(folder+"/scie.list",dtype=str))

	filter_tbl = {1: 'g',
                  2: 'r',
                  3: 'i'}

	im = scie_list[0]
	with fits.open(im) as f:
		imfilter = filter_tbl[f[0].header['FILTERID']]
		imfilter += '_median'
		nax1 = f[0].header['NAXIS1']
		nax2 = f[0].header['NAXIS2']
	
	ra_ul, dec_ul = utils.xy2sky(im, 1, 1)
	ra_ur, dec_ur = utils.xy2sky(im, nax1, 1)
	ra_ll, dec_ll = utils.xy2sky(im, 1, nax2)
	ra_lr, dec_lr = utils.xy2sky(im, nax1, nax2)
	ra_c, dec_c = utils.xy2sky(im, nax1/2, nax2/2)
	
	query_dict = {'flt':imfilter.lower(), 'ra_ll':ra_ll, 'dec_ll':dec_ll,
	              'ra_lr':ra_lr, 'dec_lr':dec_lr, 'ra_ul':ra_ul,
	              'dec_ul':dec_ul, 'ra_ur':ra_ur, 'dec_ur':dec_ur,
	              'ra_c':ra_c, 'dec_c':dec_c}
	for key in query_dict:
		query_dict[key] = str(query_dict[key])
	
	username, password = utils.load_auth_info(utils.PROJECTPATH+'/.pass','DESI')
	with psycopg2.connect(dbname='desi', host='nerscdb03.nersc.gov', 
	                   port=5432, user=username, password=password) as desi_con:
		desi_stars = zpsee.query_desi_stars(query_dict, desi_con)

	username, password = utils.load_auth_info(utils.PROJECTPATH+'/.pass','IPTF')
	with psycopg2.connect(dbname='iptf', host='nerscdb03.nersc.gov', 
	 			port=5432, user=username, password=password) as iptf_con:
		iptf_stars = zpsee.query_iptf_stars(query_dict, iptf_con)
	
	real_stars_fname_master = folder +"/real_stars.npz"
	np.savez(real_stars_fname_master,desi_stars=desi_stars,iptf_stars=iptf_stars)

	for j,im in enumerate(scie_list):

		real_stars_fname = im.replace(".fits",".real_stars.npz")
		shutil.copy(real_stars_fname_master,real_stars_fname)

def edit_fits_headers(scie_list):

	for scie in scie_list:
	    with fits.open(scie, mode='update') as f:
	        data = f[0].data
	        header = f[0].header
	        header['EXTNAME'] = 'CCD0'
	        header['XTENSION'] = 'C3'
	        if 'FIXAPERS' in header:
	            try:
	                a = int(header['FIXAPERS'].split(',')[0])
	                header['FIXAPERS'] = ','.join(['a%s'%f for f in header['FIXAPERS'].split(',')])
	            except ValueError:
	                pass
	                