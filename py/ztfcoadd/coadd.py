#!/usr/bin/env python

""" Creates a coadd image."""
__author__ = 'Michael Medford <MichaelMedford@berkeley.edu'

import os
import shutil
import numpy as np
from astropy.io import fits
from ztfcoadd import initialize
from ztfcoadd import utils
from ztfcoadd import zpsee

def output_image_list(sub_image_list):
	np.savetxt('scie_coadd.list',sub_image_list,fmt='%s')
	cat_list = [image.replace('.fits','.cat') for image in sub_image_list]
	
def select_best_images(image_list, N_images_in_coadd, debug, output=True):

	if len(image_list) < N_images_in_coadd:
		utils.print_d('%i Images in Coadd : Minimum Selection'%len(image_list), debug)
		if output:
			output_image_list(image_list)
			return
		else:
			return image_list

	lmt_mag_images = []

	for image in image_list:
		with fits.open(image) as f:
			lmt_mag = f[0].header['C3lmtmag']
			if lmt_mag > 20:
				lmt_mag_images.append(image)

	if len(lmt_mag_images) < N_images_in_coadd:
		utils.print_d('%i Images in Coadd : lmt_mag Selection'%len(lmt_mag_images), debug)
		if output:
			output_image_list(lmt_mag_images)
			return
		else:
			return lmt_mag_images

	seeing_images_tuples = []
	for image in lmt_mag_images:
		with fits.open(image) as f:
			seeing = f[0].header['C3SEE']
			seeing_images_tuples.append((image,seeing))
	seeing_images_tuples = sorted(seeing_images_tuples, key=lambda x: x[1])[:N_images_in_coadd]

	seeing_images = [image[0] for image in seeing_images_tuples]

	utils.print_d('%i Images in Coadd : Seeing Selection'%len(seeing_images), debug)

	if output:
		output_image_list(seeing_images)
		return
	else:
		return seeing_images

def coadd_makemask(coadd_fname):

	mask_fname = coadd_fname.replace('sciimg','mskimg')

	with fits.open(coadd_fname) as scie:
		cond = np.abs(scie[0].data) > 0
		scie[0].data[cond] = 0
		scie[0].data[~cond] = 1
		scie.writeto(mask_fname, overwrite=True)

def edit_coadd_header(scie_list, coadd_fname):

	obsmjd_arr = []
	for scie in scie_list:
		with fits.open(scie) as f:
			obsmjd_arr.append(f[0].header['OBSMJD'])

	with fits.open(coadd_fname, mode='update'):
		f[0].header['OBSMJD'] = np.median(obsmjd_arr)

def make_coadd(scie_list, folder, debug):

	os.chdir(folder)
	
	# MAKE THAT COADD!
	coadd = "%s/coadd_sciimg.fits"%folder
	for scie in scie_list:
		weight = scie.replace('sciimg','weight')
		weight_link = scie.replace('.fits','.weight.fits')
		os.symlink(weight, weight_link)
	coaddweight = coadd.replace('sciimg','weight')
	cmd='%s/swarp @%s/scie_coadd.list -c %s/legacypipe/py/ztfcoadd/coadd/coadd.swarp -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -NTHREADS 1'%(utils.CODEPATH,folder,utils.PROJECTPATH,coadd,coaddweight)
	utils.print_d("Making the coadd ...",debug)
	stdout, stderr = utils.execute(cmd)
	for scie in scie_list:
		weight_link = scie.replace('.fits','.weight.fits')
		os.unlink(weight_link)

	# MAKE COADD MASK
	coadd_makemask(coadd)
	coaddmask=coadd.replace('sciimg','mskimg')
	
	# PRELIM SEXTRACTOR THE COADD
	coaddcat="%s/prelim.coadd.cat"%folder
	cmd="sex -c %s/legacypipe/py/ztfcoadd/zpsee/zpsee.sex -CHECKIMAGE_TYPE NONE -VERBOSE_TYPE QUIET -CATALOG_NAME %s %s"%(utils.PROJECTPATH,coaddcat,coadd)
	utils.print_d("Prelim sex the coadd ...",debug)
	stdout, stderr = utils.execute(cmd)
	
	utils.replace_list([coadd],folder+"/coadd.list")
	utils.replace_list([coaddcat],folder+"/coadd.cat.list")
	
	# ZEROPOINT THE COADD
	utils.print_d("Updating coadd header ...",debug)
	im_real_stars_fname = scie_list[0].replace(".fits",".real_stars.npz")
	coadd_real_stars_fname = coadd.replace(".fits",".real_stars.npz")
	shutil.copy(im_real_stars_fname,coadd_real_stars_fname)

	zp, see, lmt_mag, skys, skysigs = zpsee.zpsee_info([coadd], [coaddcat], debug, coaddFlag=True)
	zpsee.update_image_headers(zp, see, lmt_mag, skys, skysigs, [coadd], debug)
	
	# CREATE FINAL CATALOG WITH CORRECT MAGNITUDES
	with fits.open(coadd) as f:
		zp = f[0].header['C3ZP']
	coaddcat_final="%s/coadd_sciimg.cat"%folder
	cmd="sex -c %s/legacypipe/py/ztfcoadd/coadd/coadd.sex -WEIGHT_IMAGE %s -MAG_ZEROPOINT %s -VERBOSE_TYPE QUIET -CATALOG_NAME %s %s"%(utils.PROJECTPATH,coaddweight,zp,coaddcat_final,coadd)
	utils.print_d("Final sex the coadd ...",debug)
	stdout, stderr = utils.execute(cmd)

	# CREATE HEADERS FOR COADD
	cmd='scamp -c %s/legacypipe/py/ztfcoadd/coadd/coadd.conf %s'%(utils.PROJECTPATH,coaddcat_final)
	utils.print_d("Creating coadd header file ...",debug)
	stdout, stderr = utils.execute(cmd)

	# CREATE HEADERS FOR COADD WITH CORRECT ASTOROMETRY
	utils.print_d("Putting header into coadd ...",debug)
	cmd='%s/swarp -SUBTRACT_BACK N %s'%(utils.CODEPATH,coadd)
	stdout, stderr = utils.execute(cmd)

	initialize.edit_fits_headers([coadd])
	edit_coadd_header(scie_list,coadd)

	coadd_fname = '/'.join(scie_list[0].split('/')[:-1]) + '/ztf_coadd_' + '_'.join(scie_list[0].split('_')[3:])
	utils.print_d("Changing %s to %s ..."%(coadd,coadd_fname),debug)
	if os.path.exists(coadd_fname):
		os.remove(coadd_fname)
	shutil.move(coadd,coadd_fname)

	coadd_cat_fname = coadd_fname.replace('.fits','.cat')
	utils.print_d("Changing %s to %s ..."%(coaddcat_final,coadd_cat_fname),debug)
	if os.path.exists(coadd_cat_fname):
		os.remove(coadd_cat_fname)
	shutil.move(coaddcat_final,coadd_cat_fname)

	coadd_weight_fname = coadd_fname.replace('sciimg','weight')
	utils.print_d("Changing %s to %s ..."%(weight,coadd_weight_fname),debug)
	if os.path.exists(coadd_weight_fname):
		os.remove(coadd_weight_fname)
	shutil.move(coaddweight,coadd_weight_fname)

	coadd_mask_fname = coadd_fname.replace('sciimg','mskimg')
	utils.print_d("Changing %s to %s ..."%(coaddmask,coadd_mask_fname),debug)
	if os.path.exists(coadd_mask_fname):
		os.remove(coadd_mask_fname)
	shutil.move(coaddmask,coadd_mask_fname)

	utils.print_d("Coadd Making Complete!",debug)
