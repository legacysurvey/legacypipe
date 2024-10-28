#! /usr/bin/env python
"""
Find all the PS1 stars in a given DECaLS CCD.
"""

import os
import numpy as np

class HealpixedCatalog(object):
    def __init__(self, fnpattern, nside=32, indexing='ring'):
        '''
        fnpattern: string formatter with key "hp", eg
        'dir/fn-%(hp)05i.fits'
        '''
        self.fnpattern = fnpattern
        self.nside = nside
        self.indexing = indexing

    def healpix_for_radec(self, ra, dec):
        '''
        Returns the healpix number for a given single (scalar) RA,Dec.
        '''
        from astrometry.util.util import radecdegtohealpix, healpix_xy_to_ring, healpix_xy_to_nested
        hpxy = radecdegtohealpix(ra, dec, self.nside)
        if self.indexing == 'ring':
            hp = healpix_xy_to_ring(hpxy, self.nside)
        elif self.indexing == 'nested':
            hp = healpix_xy_to_nested(hpxy, self.nside)
        else:
            # assume XY!
            hp = hpxy
        return hp

    def get_healpix_catalog(self, healpix):
        from astrometry.util.fits import fits_table
        fname = self.fnpattern % dict(hp=healpix)
        #print('Reading', fname)
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
        _,xx,yy = wcs.radec2pixelxy(cat.ra, cat.dec)
        cat.x = xx
        cat.y = yy
        onccd = np.flatnonzero((xx >= 1.-margin) * (xx <= W+margin) *
                               (yy >= 1.-margin) * (yy <= H+margin))
        cat.cut(onccd)
        return cat

class ps1cat(HealpixedCatalog):
    ps1band = dict(g=0,r=1,i=2,z=3,Y=4,
                   # ODIN
                   N419=0,
                   N501=0,
                   N673=1,
                   # Merian
                   N540=0,
                   # IBIS
                   M411=0,
                   M437=0,
                   M438=0,
                   M464=0,
                   M490=0,
                   M516=0,
                   M517=0,
    )
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

class sdsscat(HealpixedCatalog):
    sdssband = dict(u=0,
                    g=1,
                    r=2,
                    i=3,
                    z=4,
    )
    def __init__(self,expnum=None,ccdname=None,ccdwcs=None):
        """
        Read SDSS sources for an exposure number + CCD name or CCD WCS.

        Args:
            expnum, ccdname: select catalogue with these
            ccdwcs: or select catalogue with this

        """
        self.sdsscatdir = os.getenv('SDSSCAT_DIR')
        if self.sdsscatdir is None:
            raise ValueError('You must have the SDSSCAT_DIR environment variable set to point to healpixed SDSS catalogs')
        fnpattern = os.path.join(self.sdsscatdir, 'sdss-hp%(hp)05d.fits')
        super(sdsscat, self).__init__(fnpattern)

        if ccdwcs is None:
            from legacypipe.survey import LegacySurveyData
            survey = LegacySurveyData()
            ccd = survey.find_ccds(expnum=expnum,ccdname=ccdname)[0]
            im = survey.get_image_object(ccd)
            self.ccdwcs = im.get_wcs()
        else:
            self.ccdwcs = ccdwcs

    def get_catalog_in_wcs(self, wcs, **kwargs):
        cat = super().get_catalog_in_wcs(wcs, **kwargs)
        cat.psfmag = -2.5 * (np.log10(cat.psfflux) - 9.)
        return cat

    def get_stars(self,magrange=None,band='r'):
        """Return the set of SDSS matched stars on a given CCD.  Optionally
        trim the stars to a desired chosen-band magnitude range.
        """
        cat = self.get_catalog_in_wcs(self.ccdwcs)
        print('Found {} good SDSS stars'.format(len(cat)))
        if magrange is not None:
            band = sdsscat.sdssband[band]
            keep = np.where((self.psfmag[:,band] > magrange[0])*
                            (self.psfmag[:,band] < magrange[1]))[0]
            cat = cat[keep]
            print('Trimming to {} stars with {}=[{},{}]'.
                  format(len(cat),band,magrange[0],magrange[1]))
        return cat

def sdss_to_decam(psfmags, band):
    '''
    mags: 2-d array (Nstars, Nbands)
    '''
    g_index = sdsscat.sdssband['g']
    i_index = sdsscat.sdssband['i']
    gmag = psfmags[:,g_index]
    imag = psfmags[:,i_index]
    gi = gmag - imag
    coeffs = dict(
        # from Arjun 2021-03-17 based on DECosmos runs (2013-02-11,12,13)
        u = [-0.0643, 0.1787,-0.1552, 0.0259],
    ).get(band, [0.,0.,0.,0.])
    colorterm = coeffs[0] + coeffs[1]*gi + coeffs[2]*gi**2 + coeffs[3]*gi**3
    return colorterm

def ps1_to_decam(psmags, band):
    '''
    psmags: 2-d array (Nstars, Nbands)
    band: [grz]
    '''
    # https://desi.lbl.gov/trac/wiki/DecamLegacy/Reductions/Photometric
    g_index = ps1cat.ps1band['g']
    r_index = ps1cat.ps1band['r']
    i_index = ps1cat.ps1band['i']
    gmag = psmags[:,g_index]
    rmag = psmags[:,r_index]
    imag = psmags[:,i_index]
    gi = gmag - imag
    gr = gmag - rmag

    coeffs = dict(
        g = [ 0.00062,  0.03604, 0.01028, -0.00613 ],
        r = [ 0.00495, -0.08435, 0.03222, -0.01140 ],
        i = [ 0.00904, -0.04171, 0.00566, -0.00829 ],
        z = [ 0.02583, -0.07690, 0.02824, -0.00898 ],
        Y = [ 0.02332, -0.05992, 0.02840, -0.00572 ],
        # From Arjun 2022-11-08
        # c0: -0.8934
        N419 = [0., 0.2727, 0.9945,-0.6272, 0.1118],
        # From Arjun 2021-02-26
        # c0: 0.0059
        N501 = [ 0., -0.2784, 0.2915, -0.0686 ],
        # c0: 0.2324
        N673 = [ 0., -0.3456, 0.1334, -0.0146 ],
        # Merian
        N540 = [ 0.36270593, -0.40238762,  0.0551082 ],
        # CFHT
        CaHK = [0., 2.3],
        # IBIS -- from Arjun 2024-05-30
        # M411 = [0, 0.1164,  1.1036, -0.5814,  0.0817],
        # M464 = [0, 0.6712, -0.9042,  0.4621, -0.0737],
        # 2024-06-02
        # M411 = [0., -0.1761, 1.5862,-0.8962, 0.1526],
        # M464 = [0.,  0.3244,-0.5258, 0.3082,-0.0524],
        # 2024-06-04
        # g-r
        # M411 = [ 0.05546,-0.07920, 2.50180,-1.44430],
        # M464 = [ 0.12144,-0.53290, 1.25790,-0.93900],
        # 2024-06-05
        # g-i:
        M411 = [0.,  0.1416,  0.9785, -0.3954],
        M464 = [0.,  0.3244, -0.5258,  0.3082, -0.0524],
        # FAKE - equal to M464
        M437 = [0.,  0.3244, -0.5258,  0.3082, -0.0524],
        M438 = [0.,  0.3244, -0.5258,  0.3082, -0.0524],
        M490 = [0.,  0.3244, -0.5258,  0.3082, -0.0524],
        M516 = [0.,  0.3244, -0.5258,  0.3082, -0.0524],
        M517 = [0.,  0.3244, -0.5258,  0.3082, -0.0524],
        )[band]

    # Most are with respect to g-i, some are g-r...
    color = gi
    if band in ['CaHK']: #, 'M411', 'M464']:
        color = gr

    colorterm = np.zeros(len(color))
    for power,coeff in enumerate(coeffs):
        colorterm += coeff * color**power
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

def ps1_to_hsc(psmags, band):
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
    coeffs = dict(
        g = [ 0., 0., 0., 0.],
        r = [ 0., 0., 0., 0.],
        i = [ 0., 0., 0., 0.],
        z = [ 0., 0., 0., 0.],
        Y = [ 0., 0., 0., 0.],
    )[band]
    colorterm = coeffs[0] + coeffs[1]*gi + coeffs[2]*gi**2 + coeffs[3]*gi**3
    return colorterm
