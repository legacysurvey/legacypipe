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

    def get_cat(self,ra,dec):
        """Read the stack of PS1 catalogs, given ra,dec coordinates"""
        nstar = len(ra)
        ipring = np.empty(nstar).astype(int)
        for iobj, (ra1, dec1) in enumerate(zip(ra,dec)):
            hpxy = radecdegtohealpix(ra1,dec1,self.nside)
            ipring[iobj] = healpix_xy_to_ring(hpxy,self.nside)
        pix = np.unique(ipring)

        cat = list()
        for ipix in pix:
            fname = os.path.join(self.ps1dir,'ps1-'+'{:05d}'.format(ipix)+'.fits')
            print(fname)
            cat.append(fits_table(fname))
        cat = merge_tables(cat)
        return cat

    def get_stars(self,ra,dec,ccdwcs):
        nstar = len(ra)
        onoffccd = np.empty(nstar)
        for iobj in range(nstar):
            onoffccd[iobj] = ccdwcs.inside(ra[iobj],dec[iobj])
        onccd = np.where((onccd*1)==1)
        return onccd
