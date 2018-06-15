#!/usr/bin/env python

""" Creates initial sextractor catalogs. """
__author__ = 'Michael Medford <MichaelMedford@berkeley.edu'

from ztfcoadd import utils

def create_sex_cats(my_scie_list, debug):
    
    for i,scie in enumerate(my_scie_list):

        cat = scie.replace('ztf','prelim.ztf').replace('.fits','.cat')
        var = scie.replace('.fits','.var.fits')
        utils.print_d('%i/%i) Creating initial_sex for %s'%(i+1,len(my_scie_list),utils.trim(cat)),debug)

        command = 'sex -c %s/legacypipe/py/ztfcoadd/zpsee/zpsee.sex -CHECKIMAGE_NAME %s -VERBOSE_TYPE QUIET -CATALOG_NAME %s %s'%(utils.PROJECTPATH,var,cat,scie)
        stdout, stderr = utils.execute(command)
