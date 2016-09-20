import os
from ps1cat import HealpixedCatalog
    
class GaiaCatalog(HealpixedCatalog):
    def __init__(self):
        self.gaiadir = os.getenv('GAIA_CAT_DIR')
        if self.gaiadir is None:
            raise ValueError('You must have the GAIA_CAT_DIR environment variable set to point to healpixed Gaia catalogs')
        fnpattern = os.path.join(self.gaiadir, 'chunk-%(hp)05d.fits')
        super(GaiaCatalog, self).__init__(fnpattern)

    def get_catalog_radec_box(self, ralo, rahi, declo, dechi):
        import numpy as np
        # Prepare RA,Dec grid to pick up overlapping healpixes
        rr,dd = np.meshgrid(np.linspace(ralo,  rahi,  2+( rahi- ralo)/0.1),
                            np.linspace(declo, dechi, 2+(dechi-declo)/0.1))
        healpixes = set()
        for r,d in zip(rr,dd):
            healpixes.add(self.healpix_for_radec(r, d))
        # Read catalog in those healpixes
        cat = self.get_healpix_catalogs(healpixes)
        cat.cut((cat.ra  >= ralo ) * (cat.ra  <= rahi) *
                (cat.dec >= declo) * (cat.dec <= dechi))
        return cat
