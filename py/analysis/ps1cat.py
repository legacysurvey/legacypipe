#! /usr/bin/env python

"""
Find all the PS1 stars in a given DECaLS CCD.
"""

import os
import numpy as np
from astrometry.util.fits import fits_table, merge_tables
from astrometry.util.util import radecdegtohealpix, healpix_xy_to_ring

class ps1cat():
    def __init__(self):
        self.ps1dir = os.getenv('PS1CAT_DIR')
        self.nside = 32

    def get_cat(self,ra=None,dec=None):
        print(ra, dec)
        nra = len(ra)

        ipring = np.array(nra)
        for iobj in range(nra):
            hpxy = radecdegtohealpix(ra[iobj],dec[iobj],self.nside)
            ipring.append(healpix_xy_to_ring(hpxy,self.nside))
        print(pix)
        pix = ipring.unique()
        print(pix)

        cat = None
        for ipix in pix:
            fname = os.path.join(self.ps1dir,'ps1-qa-'+'{:05d}'.format(ipix)+'-v3.fits')
            if cat is None:
                cat1 = fits_table(fname)
            else:
                cat = merge_tables(cat,cat1)

        return cat
