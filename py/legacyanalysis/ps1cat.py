#! /usr/bin/env python

"""
Find all the PS1 stars in a given DECaLS CCD.
"""

import os
import numpy as np

class HealpixedCatalog(object):
    def __init__(self, fnpattern, nside=32):
        '''
        fnpattern: string formatter with key "hp", eg
        'dir/fn-%(hp)05i.fits'
        '''
        self.fnpattern = fnpattern
        self.nside = nside

    def healpix_for_radec(self, ra, dec):
        '''
        Returns the healpix number for a given single (scalar) RA,Dec.
        '''
        from astrometry.util.util import radecdegtohealpix, healpix_xy_to_ring
        hpxy = radecdegtohealpix(ra, dec, self.nside)
        ipring = healpix_xy_to_ring(hpxy, self.nside)
        return ipring

    def get_healpix_catalog(self, healpix):
        from astrometry.util.fits import fits_table
        fname = self.fnpattern % dict(hp=healpix)
        return fits_table(fname)
    
    def get_healpix_catalogs(self, healpixes):
        from astrometry.util.fits import merge_tables
        cats = []
        for hp in healpixes:
            cats.append(self.get_healpix_catalog(hp))
        return merge_tables(cats)

    def get_catalog_in_wcs(self, wcs, step=100., margin=10):
        # Grid the CCD in pixel space
        W,H = wcs.get_width(), wcs.get_height()
        xx,yy = np.meshgrid(
            np.linspace(1-margin, W+margin, 2+int((W+2*margin)/step)),
            np.linspace(1-margin, H+margin, 2+int((H+2*margin)/step)))
        # Convert to RA,Dec and then to unique healpixes
        ra,dec = wcs.pixelxy2radec(xx.ravel(), yy.ravel())
        healpixes = set()
        for r,d in zip(ra,dec):
            healpixes.add(self.healpix_for_radec(r, d))
        # Read catalog in those healpixes
        cat = self.get_healpix_catalogs(healpixes)
        # Cut to sources actually within the CCD.
        ok,xx,yy = wcs.radec2pixelxy(cat.ra, cat.dec)
        cat.x = x
        cat.y = y
        onccd = np.flatnonzero((xx >= 1.-margin) * (xx <= W+margin) *
                               (yy >= 1.-margin) * (yy <= H+margin))
        cat.cut(onccd)
        return cat
    
    
class ps1cat(HealpixedCatalog):
    ps1band = dict(g=0,r=1,i=2,z=3,Y=4)
    def __init__(self,expnum=None,ccdname=None,ccdwcs=None):
        """Initialize the class with either the exposure number *and* CCD name, or
        directly with the WCS of the CCD of interest.

        """
        self.ps1dir = os.getenv('PS1CAT_DIR')
        if self.ps1dir is None:
            raise ValueError('You must have the PS1CAT_DIR environment variable set to point to Pan-STARRS1 catalogs')
        fnpattern = os.path.join(self.ps1dir, 'ps1-%(hp)05d.fits')

        super(ps1cat, self).__init__(fnpattern)
        
        if ccdwcs is None:
            from legacypipe.common import LegacySurveyData
            survey = LegacySurveyData()
            ccd = survey.find_ccds(expnum=expnum,ccdname=ccdname)[0]
            im = survey.get_image_object(ccd)
            self.ccdwcs = im.get_wcs()
        else:
            self.ccdwcs = ccdwcs

    def get_cat(self,ra,dec):
        """Read the healpixed PS1 catalogs given input ra,dec coordinates."""
        # Convert RA,Decs to unique healpixes
        ra,dec = wcs.pixelxy2radec(xx.ravel(), yy.ravel())
        healpixes = set()
        for r,d in zip(ra,dec):
            healpixes.add(self.healpix_for_radec(r, d))
        # Read catalog in those healpixes
        cat = self.get_healpix_catalogs(healpixes)
        return cat

    def get_stars(self,magrange=None,band='r'):
        """Return the set of PS1 stars on a given CCD with well-measured grz
        magnitudes. Optionally trim the stars to a desired r-band magnitude
        range.
        """
        cat = self.get_catalog_in_wcs()
        print('Found {} good PS1 stars'.format(len(cat)))
        if magrange is not None:
            keep = np.where((cat.median[:,ps1cat.ps1band[band]]>magrange[0])*
                            (cat.median[:,ps1cat.ps1band[band]]<magrange[1]))[0]
            cat = cat[keep]
            print('Trimming to {} stars with {}=[{},{}]'.
                  format(len(cat),band,magrange[0],magrange[1]))
        return cat


def ps1_to_decam(psmags, band):
    '''
    psmags: 2-d array (Nstars, Nbands)
    band: [grz]
    '''
    g_index = ps1cat.ps1band['g']
    i_index = ps1cat.ps1band['i']
    gmag = psmags[:,g_index]
    imag = psmags[:,i_index]
    gi = gmag - imag
    coeffs = dict(
        g = [0.0, -0.04709, -0.00084, 0.00340],
        r = [0.0,  0.09939, -0.04509, 0.01488],
        z = [0.0,  0.13404, -0.06591, 0.01695])[band]

    colorterm = -(coeffs[0] + coeffs[1]*gi + coeffs[2]*gi**2 + coeffs[3]*gi**3)
    return colorterm
    
