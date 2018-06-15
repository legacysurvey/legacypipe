#!/usr/bin/env python

"""
Tool for building simple coaads for ZTF Images.
Requires an input folder of ZTF science images.
"""

__whatami__ = 'ZTF Coass Maker'
__author__ = 'Michael Medford <MichaelMedford@berkeley.edu>'

import argparse
import glob
import shutil
import os
import sys
import psycopg2
import numpy as np
from astropy.io import fits
from ztfcoadd import utils
from ztfcoadd.zpsee import zpsee
from ztfcoadd.sex import initial_sex
from ztfcoadd.sex import final_sex
from ztfcoadd.makeweight import makeweight
from ztfcoadd.coadd import coadd

def delete_folder_contents(folder, debug):

	utils.print_d("CLEARING FOLDER: %s ..."%(folder),debug)

	keep_files = glob.glob(folder+"/ztf*sciimg.fits")
	keep_files += glob.glob(folder+"/ztf*mskimg.fits")
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
	        if 'FIXAPER' in header:
	            try:
	                a = int(header['FIXAPERS'].split(',')[0])
	                header['FIXAPERS'] = ','.join(['a%s'%f for f in header['FIXAPERS'].split(',')])
	            except ValueError:
	                pass

def main():

	parser = argparse.ArgumentParser(description=__doc__)

	required = parser.add_argument_group('required arguments')
	required.add_argument('--folder', type=str,
	                    help='Folder of ZTF observations in '
	                    'a single field, CCD and quadrant.')

	parser.add_argument('--images-coadd-num', type=int,
	                    help='Maximum number of images in a coadd. '
	                    'Default is 20.')
	parser.set_defaults(images_coadd_num=20)

	debuggroup = parser.add_mutually_exclusive_group()
	debuggroup.add_argument('--debug', dest='debug',
	                        action='store_true', help='Print statements. (default)')
	debuggroup.add_argument('--debug-off', dest='debug',
	                        action='store_false', help='Do not print statements.')
	parser.set_defaults(debug=True)

	args = parser.parse_args()

	os.chdir(args.folder)

	delete_folder_contents(args.folder, args.debug)
	create_lists(args.folder, args.debug)
	scie_list = np.genfromtxt(args.folder+"/scie.list",dtype=str)
	download_real_stars(args.folder, args.debug)

	# EDIT IMAGES FOR LEGACYPIPE CONSUMPTION
	edit_fits_headers(scie_list)
	

	# MAKE INITIAL SEX CATALOGS
	initial_sex.create_sex_cats(scie_list, args.debug)
	
	# MAKE WEIGHT FILES
	makeweight.make_weights(scie_list, args.debug)
	
	# CALCULATE IMAGE PROPERTIES AND UPDATE HEADERS	
	pre_cat_list = [scie.replace('ztf','prelim.ztf').replace('.fits','.cat') for scie in scie_list]
	zp, see, lmt_mag, skys, skysigs = zpsee.zpsee_info(scie_list, pre_cat_list, args.debug)
	zpsee.update_image_headers(zp, see, lmt_mag, skys, skysigs, scie_list, args.debug)
	
	# CREATION OF FINAL SEXTRACTOR CATALOGS FOR IMAGES
	final_sex.create_sex_cats(scie_list, args.debug)

	# CREATE HEADERS FOR ALL IMAGES WITH CORRECT ASTOROMETRY
	cat_list = [scie.replace('.fits','.cat') for scie in scie_list]
	utils.replace_list(cat_list,args.folder+'/cat.list')

	utils.print_d("Creating headers for all images ...",args.debug)
	cmd='scamp -c %s/legacypipe/py/coadd/coadd.conf @%s/cat.list'%(utils.PROJECTPATH,args.folder)
	stdout, stderr = utils.execute(cmd)

	utils.print_d("Putting header into images ...",args.debug)
	for scie in scie_list:
		cmd='%s/swarp -SUBTRACT_BACK N %s'%(utils.CODEPATH,scie)
		stdout, stderr = utils.execute(cmd)

	# MAKE A COADD
	utils.print_d("Selecting images for coadd ...",args.debug)
	coadd.select_best_images(scie_list, args.images_coadd_num, args.debug)
	coadd.make_coadd(scie_list, args.folder, args.debug)

if __name__ == '__main__':

	main()
