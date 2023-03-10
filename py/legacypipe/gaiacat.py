import os
from legacypipe.ps1cat import HealpixedCatalog

class GaiaCatalog(HealpixedCatalog):
    def __init__(self, file_prefix=None, indexing=None, **kwargs):
        self.gaiadir = os.getenv('GAIA_CAT_DIR')
        if self.gaiadir is None:
            raise ValueError('You must have the GAIA_CAT_DIR environment variable set to point to healpixed Gaia catalogs')
        if indexing is None:
            indexing = os.getenv('GAIA_CAT_SCHEME', 'ring')
        if not indexing in ['nested', 'ring']:
            raise ValueError('Supported values for the GAIA_CAT_SCHEME environment variable or healpix indexing scheme are "nested" or "ring"')
        if file_prefix is None:
            file_prefix = os.getenv('GAIA_CAT_PREFIX', 'chunk')
        #
        fnpattern = os.path.join(self.gaiadir, file_prefix + '-%(hp)05d.fits')
        super(GaiaCatalog, self).__init__(fnpattern, indexing=indexing, **kwargs)

    def get_catalog_radec_box(self, ralo, rahi, declo, dechi):
        import numpy as np

        wrap = False
        if rahi < ralo:
            # wrap-around?
            rahi += 360.
            wrap = True

        # Prepare RA,Dec grid to pick up overlapping healpixes
        rr,dd = np.meshgrid(np.linspace(ralo,  rahi,  2+int(( rahi- ralo)/0.1)),
                            np.linspace(declo, dechi, 2+int((dechi-declo)/0.1)))
        healpixes = set()
        for r,d in zip(rr.ravel(), dd.ravel()):
            healpixes.add(self.healpix_for_radec(r, d))
        # Read catalog in those healpixes
        cat = self.get_healpix_catalogs(healpixes)
        #print('Read', len(cat), 'Gaia catalog entries.  RA range', cat.ra.min(), cat.ra.max(),
        #      'Dec range', cat.dec.min(), cat.dec.max())
        cat.cut((cat.dec >= declo) * (cat.dec <= dechi))
        if wrap:
            cat.cut(np.logical_or(cat.ra <= ralo, cat.ra >= (rahi - 360.)))
        else:
            cat.cut((cat.ra  >= ralo ) * (cat.ra  <= rahi))
        return cat

    @staticmethod
    def catalog_nantozero(gaia):
        gaia.pmra = nantozero(gaia.pmra)
        gaia.pmdec = nantozero(gaia.pmdec)
        gaia.parallax = nantozero(gaia.parallax)
        return gaia

def nantozero(x):
    import numpy as np
    x = x.copy()
    x[np.logical_not(np.isfinite(x))] = 0.
    return x
