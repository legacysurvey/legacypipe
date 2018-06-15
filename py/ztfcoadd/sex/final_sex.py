#!/usr/bin/env python

""" Creates final sextractor catalogs. """
__author__ = 'Michael Medford <MichaelMedford@berkeley.edu'

from ztfcoadd import utils
from astropy.io import fits

def create_sex_cats(my_scie_list, debug):

    for i,scie in enumerate(my_scie_list):
        
        noise = scie.replace('.fits','.noise.fits')
        weight = scie.replace('.fits','.weight.fits')
        newcat = scie.replace('.fits','.cat')
        var = scie.replace('.fits','.var.fits')
        
        with fits.open(scie) as f:
            zp = f[0].header['C3ZP']

        if zp == 0:
            continue
            
        utils.print_d('%i/%i Creating final_sex for %s'%(i+1,len(my_scie_list),utils.trim(newcat)),debug)
        
        command = 'sex -c %s/legacypipe/py/ztfcoadd/coadd/coadd.sex -WEIGHT_IMAGE %s -MAG_ZEROPOINT %s -VERBOSE_TYPE QUIET -CHECKIMAGE_NAME %s -CATALOG_NAME %s %s'%(utils.PROJECTPATH,weight,zp,noise,newcat,scie)
        utils.execute(command)
