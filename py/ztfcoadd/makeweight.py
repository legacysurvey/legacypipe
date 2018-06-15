#!/usr/bin/env python

__author__ = 'Michael Medford <MichaelMedford@berkeley.edu>'

from ztfcoadd import utils
from astropy.io import fits
import shutil

def make_weights(my_scie_list, debug):

    for i,scie in enumerate(my_scie_list):

        maskname = scie.replace('sciimg', 'mskimg')
        varname = scie.replace('.fits', '.var.fits')
        weightname = scie.replace('sciimg', 'weight')

        with fits.open(varname) as img:
            with fits.open(maskname) as mask:
                img[0].data[mask[0].data & 6141 != 0] = 5e4
                img[0].data = 1 / img[0].data
                img.writeto(weightname, overwrite=True)

        utils.print_d('%i/%i) MakeWeight: %s'%(i+1,len(my_scie_list),utils.trim(scie)),debug)
