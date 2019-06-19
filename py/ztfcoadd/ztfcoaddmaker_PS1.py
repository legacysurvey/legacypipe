#!/usr/bin/env python

"""
Tool for building simple coaads for ZTF Images.
Requires an input folder of ZTF science images.
"""

__whatami__ = 'ZTF Coadd Maker'
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
from ztfcoadd import initialize
from ztfcoadd import zpsee
from ztfcoadd import initial_sex
from ztfcoadd import final_sex
from ztfcoadd import makeweight
from ztfcoadd import coadd

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
	if args.folder.endswith('/'):
		args.folder = args.folder[:-1]
	cwd=os.getcwd()
	os.chdir(args.folder)

	#initialize.delete_folder_contents(args.folder, args.debug)
	#initialize.create_lists(args.folder, args.debug)
	scie_list = np.genfromtxt(args.folder+"/scie.list",dtype=str)
	#print(len([scie_list]))

	#if len([scie_list])==1:
#		scie_list=list([str([scie_list][0])])
#	print(scie_list)
	initialize.download_real_stars(args.folder, args.debug)

	# EDIT IMAGES FOR LEGACYPIPE CONSUMPTION
	initialize.edit_fits_headers(scie_list)

	# MAKE INITIAL SEX CATALOGS
	initial_sex.create_sex_cats(scie_list, args.debug)
	
	# MAKE WEIGHT FILES
	makeweight.make_weights(scie_list, args.debug)
	
	# CALCULATE IMAGE PROPERTIES AND UPDATE HEADERS	
	pre_cat_list = [scie.replace('ztf','prelim.ztf').replace('.fits','.cat') for scie in scie_list]
	#zp, see, lmt_mag, skys, skysigs = zpsee.zpsee_info(scie_list, pre_cat_list, args.debug)
	#zpsee.update_image_headers(zp, see, lmt_mag, skys, skysigs, scie_list, args.debug)
	
	# CREATION OF FINAL SEXTRACTOR CATALOGS FOR IMAGES
	final_sex.create_sex_cats(scie_list, args.debug)

	# CREATE HEADERS FOR ALL IMAGES WITH CORRECT ASTOROMETRY
	cat_list = [scie.replace('.fits','.cat') for scie in scie_list]
	utils.replace_list(cat_list,args.folder+'/cat.list')

	utils.print_d("Creating headers for all images ...",args.debug)
	cmd='scamp -c %s/legacypipe/py/coadd/coadd.conf @%s/cat.list'%(utils.PROJECTPATH,args.folder)
	stdout, stderr = utils.execute(cmd)

	utils.print_d("Putting header into images ...",args.debug)
	cmd='%s/swarp -SUBTRACT_BACK N @%s/scie.list'%(utils.CODEPATH,args.folder)
	stdout, stderr = utils.execute(cmd)

	# MAKE A COADD
	utils.print_d("Selecting images for coadd ...",args.debug)
	coadd.select_best_images(scie_list, args.images_coadd_num, args.debug)
	coadd.make_coadd(scie_list, args.folder, args.debug)
	os.chdir(cwd)
if __name__ == '__main__':

	main()
