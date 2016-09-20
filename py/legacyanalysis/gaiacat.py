from ps1cat import HealpixedCatalog
    
class GaiaCatalog(HealpixedCatalog):
    def __init__(self):
        self.gaiadir = os.getenv('GAIA_CAT_DIR')
        if self.gaiadir is None:
            raise ValueError('You must have the GAIA_CAT_DIR environment variable set to point to healpixed Gaia catalogs')
        fnpattern = os.path.join(self.gaiadir, 'chunk-%(hp)05d.fits')
        super(GaiaCatalog, self).__init__(fnpattern)
