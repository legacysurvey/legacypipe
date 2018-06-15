#!/usr/bin/env python

""" Creates a coadd image."""
__author__ = 'Michael Medford <MichaelMedford@berkeley.edu'

import os
import shutil
import numpy as np
from astropy.io import fits
from ztfcoadd import utils
from ztfcoadd.zpsee import zpsee

def output_image_list(sub_image_list):
	np.savetxt('scie_coadd.list',sub_image_list,fmt='%s')
	cat_list = [image.replace('.fits','.cat') for image in sub_image_list]
	np.savetxt('tempcoadd.cat.list',cat_list,fmt='%s')
	
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

def make_coadd(scie_list, folder, debug):

	os.chdir(folder)
	
	# MAKE THAT COADD!
	coadd="%s/coadd.fits"%folder
	weight=coadd.replace('.fits','.weight.fits')
	cmd='%s/swarp @%s/scie_coadd.list -c %s/legacypipe/py/coadd/coadd.swarp -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -NTHREADS 1'%(utils.CODEPATH,folder,utils.PROJECTPATH,coadd,weight)
	utils.print_d("Making the coadd ...",debug)
	stdout, stderr = utils.execute(cmd)
	
	# PRELIM SEXTRACTOR THE COADD
	coadd="%s/coadd.fits"%folder
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
	weight = coadd.replace('.fits','.weight.fits')
	with fits.open(coadd) as f:
		zp = f[0].header['C3ZP']
	cat="%s/coadd.cat"%folder
	cmd="sex -c %s/legacypipe/py/ztfcoadd/coadd/coadd.sex -WEIGHT_IMAGE %s -MAG_ZEROPOINT %s -VERBOSE_TYPE QUIET -CATALOG_NAME %s %s"%(utils.PROJECTPATH,weight,zp,cat,coadd)
	utils.print_d("Final sex the coadd ...",debug)
	stdout, stderr = utils.execute(cmd)

	# CREATE HEADERS FOR COADD
	cmd='scamp -c %s/legacypipe/py/ztfcoadd/coadd/coadd.conf %s'%(utils.PROJECTPATH,cat)
	utils.print_d("Creating coadd header file ...",debug)
	stdout, stderr = utils.execute(cmd)

	# CREATE HEADERS FOR COADD WITH CORRECT ASTOROMETRY
	utils.print_d("Putting header into coadd ...",debug)
	cmd='%s/swarp -SUBTRACT_BACK N %s'%(utils.CODEPATH,coadd)
	stdout, stderr = utils.execute(cmd)

	coadd_fname = '/'.join(scie_list[0].split('/')[:-1])+'/ztf_'+'_'.join(scie_list[0].split('_')[3:7]) + '_coadd.fits'
	utils.print_d("Changing %sto %s ..."%(coadd,coadd_fname),debug)
	if os.path.exists(coadd_fname):
		os.remove(coadd_fname)
	shutil.move(coadd,coadd_fname)

	coadd_cat_fname = coadd_fname.replace('.fits','.cat')
	utils.print_d("Changing %sto %s ..."%(cat,coadd_cat_fname),debug)
	if os.path.exists(coadd_cat_fname):
		os.remove(coadd_cat_fname)
	shutil.move(cat,coadd_cat_fname)

	utils.print_d("Coadd Making Complete!",debug)
