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
        print('Reading', fname)
        return fits_table(fname)
    
    def get_healpix_catalogs(self, healpixes):
        from astrometry.util.fits import merge_tables
        cats = []
        for hp in healpixes:
            cats.append(self.get_healpix_catalog(hp))
        if len(cats) == 1:
            return cats[0]
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
        cat.x = xx
        cat.y = yy
        onccd = np.flatnonzero((xx >= 1.-margin) * (xx <= W+margin) *
                               (yy >= 1.-margin) * (yy <= H+margin))
        cat.cut(onccd)
        return cat
    
    
class ps1cat(HealpixedCatalog):
    ps1band = dict(g=0,r=1,i=2,z=3,Y=4)
    def __init__(self,expnum=None,ccdname=None,ccdwcs=None):
        """Read PS1 or gaia sources for an exposure number + CCD name or CCD WCS

        Args:
            expnum, ccdname: select catalogue with these
            ccdwcs: or select catalogue with this

        """
        self.ps1catdir = os.getenv('PS1CAT_DIR')
        if self.ps1catdir is None:
            raise ValueError('You must have the PS1CAT_DIR environment variable set to point to healpixed PS1 catalogs')
        fnpattern = os.path.join(self.ps1catdir, 'ps1-%(hp)05d.fits')
        super(ps1cat, self).__init__(fnpattern)
        
        if ccdwcs is None:
            from legacypipe.survey import LegacySurveyData
            survey = LegacySurveyData()
            ccd = survey.find_ccds(expnum=expnum,ccdname=ccdname)[0]
            im = survey.get_image_object(ccd)
            self.ccdwcs = im.get_wcs()
        else:
            self.ccdwcs = ccdwcs

    def get_stars(self,magrange=None,band='r'):
        """Return the set of PS1 or gaia-PS1 matched stars on a given CCD with well-measured grz
        magnitudes. Optionally trim the stars to a desired r-band magnitude
        range.
        """
        cat = self.get_catalog_in_wcs(self.ccdwcs)
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
    # https://desi.lbl.gov/trac/wiki/DecamLegacy/Reductions/Photometric
    g_index = ps1cat.ps1band['g']
    i_index = ps1cat.ps1band['i']
    gmag = psmags[:,g_index]
    imag = psmags[:,i_index]
    gi = gmag - imag

    # Eddie says (2017-Feb-14), in
    # [decam-chatter 4783] DECam grizY color transformations,
    # if ind EQ 0 then coeff = [ 0.00613, -0.01028, -0.03604, -0.00062] ; g
    # if ind EQ 1 then coeff = [ 0.01140, -0.03222,  0.08435, -0.00495] ; r
    # if ind EQ 2 then coeff = [ 0.00829, -0.00566,  0.04171, -0.00904] ; i
    # if ind EQ 3 then coeff = [ 0.00898, -0.02824,  0.07690, -0.02583] ; z
    # if ind EQ 4 then coeff = [ 0.00572, -0.02840,  0.05992, -0.02332] ; Y
    # coeffs = dict(
    #     g = [ 0.00613, -0.01028, -0.03604, -0.00062],
    #     r = [ 0.01140, -0.03222,  0.08435, -0.00495],
    #     i = [ 0.00829, -0.00566,  0.04171, -0.00904],
    #     z = [ 0.00898, -0.02824,  0.07690, -0.02583],
    #     Y = [ 0.00572, -0.02840,  0.05992, -0.02332],)[band]

    # But turns out the coeffs are in the opposite order!
    coeffs = dict(
        g = [ 0.00062,  0.03604, 0.01028, -0.00613 ],
        r = [ 0.00495, -0.08435, 0.03222, -0.01140 ],
        i = [ 0.00904, -0.04171, 0.00566, -0.00829 ],
        z = [ 0.02583, -0.07690, 0.02824, -0.00898 ],
        Y = [ 0.02332, -0.05992, 0.02840, -0.00572 ],)[band]

    # Previously, we used: (with an overall negative)
    # g = [0.0, -0.04709, -0.00084, 0.00340],
    # r = [0.0,  0.09939, -0.04509, 0.01488],
    # z = [0.0,  0.13404, -0.06591, 0.01695])[band]

    colorterm = coeffs[0] + coeffs[1]*gi + coeffs[2]*gi**2 + coeffs[3]*gi**3
    #print('Using DECam ColorTerm')
    return colorterm
    
def ps1_to_90prime(psmags, band):
    '''
    psmags: 2-d array (Nstars, Nbands)
    band: [gr]

    color terms are taken from:
      https://desi.lbl.gov/trac/wiki/BokLegacy/Photometric
    
    '''
    g_index = ps1cat.ps1band['g']
    i_index = ps1cat.ps1band['i']
    gmag = psmags[:, g_index]
    imag = psmags[:, i_index]
    gi = gmag - imag
    # 15 Nov 2018
    # https://desi.lbl.gov/trac/wiki/ImagingStandardBandpass
    coeffs = dict(
        g = [-0.00464, -0.08672,  0.00668,  0.00255],
        r = [-0.00110,  0.06875, -0.02480,  0.00855])[band]
        # Old version, from
        # July 22, 2016
        # https://desi.lbl.gov/trac/wiki/BokLegacy/Photometric
        #g = [0.0, +0.06630, +0.00958, -0.00672],
        #r = [0.0, -0.04836, +0.01100, -0.00563])[band]
        #Even older version.
        #g = [0.0, +0.08612, -0.00392, -0.00393],
        #r = [0.0, -0.07831, +0.03304, -0.01027])[band]
        # Note older versions used opposite sign.
    colorterm = -(coeffs[0] + coeffs[1]*gi + coeffs[2]*gi**2 + coeffs[3]*gi**3)
    print('Using 90prime ColorTerm')
    return colorterm
    
def ps1_to_mosaic(psmags, band):
    '''
    psmags: 2-d array (Nstars, Nbands)
    band: [gr]
    '''
    g_index = ps1cat.ps1band['g']
    i_index = ps1cat.ps1band['i']
    gmag = psmags[:, g_index]
    imag = psmags[:, i_index]
    gi = gmag - imag
    # Average color term for Mosaic3
    # 15 Nov 2018
    # https://desi.lbl.gov/trac/wiki/ImagingStandardBandpass
    coeffs = dict(z = [-0.03664,  0.11084, -0.04477,  0.01223])[band]
    # Old version from
    # https://desi.lbl.gov/trac/wiki/MayallZbandLegacy/CPReductions
    #coeffs = dict(z = [0.0, 0.121315, -0.046082623, 0.011642475])[band]

    colorterm = -(coeffs[0] + coeffs[1]*gi + coeffs[2]*gi**2 + coeffs[3]*gi**3)
    print('Using Mosaic3 ColorTerm')
    return colorterm
    
