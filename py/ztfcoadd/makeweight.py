#!/usr/bin/env python

__author__ = 'Michael Medford <MichaelMedford@berkeley.edu>'

from ztfcoadd import utils
from astropy.io import fits
import shutil
import numpy as np
import numpy
def make_weights(my_scie_list, debug):

    for i,scie in enumerate(my_scie_list):

        print(scie)
        maskname = scie.replace('sciimg', 'mskimg')
        varname = scie.replace('.fits', '.var.fits')
        weightname = scie.replace('sciimg', 'weight')
	    #If a hot pixel or a dead pixel, give it a huge number
        with fits.open(varname) as img:
            with fits.open(maskname) as mask:
                print(mask[0].data)
                img[0].data = 1 / img[0].data
                img[0].data[mask[0].data & 6141 != 0] = 0 
                img.writeto(weightname, overwrite=True)

        utils.print_d('%i/%i) MakeWeight: %s'%(i+1,len(my_scie_list),utils.trim(scie)),debug)


def make_weights_PS1(my_scie_list, debug):

    for i,scie in enumerate(my_scie_list):

        print(scie)
        maskname = scie.replace('new_sciimg', 'new_mskimg')
        varname = scie.replace('new_sciimg.fits', 'weight.fits')
        weightname = scie.replace('new_sciimg', 'new_weight')
	    #If a hot pixel or a dead pixel, give it a huge number
        with fits.open(varname) as img:
            with fits.open(maskname) as mask:
                img[0].data = img[0].data-np.nanmin(img[0].data) 
                img[0].data = 1 / img[0].data 
                where_are_NaNs = np.isnan(img[0].data)
                img[0].data[where_are_NaNs] = 0.0
                where_are_inf = np.isinf(img[0].data)
                img[0].data[where_are_inf] = 0.0
                img.writeto(weightname, overwrite=True)

        utils.print_d('%i/%i) MakeWeight: %s'%(i+1,len(my_scie_list),utils.trim(scie)),debug)
