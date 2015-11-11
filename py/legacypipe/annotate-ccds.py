from __future__ import print_function
import numpy as np

from astrometry.util.fits import fits_table, merge_tables
from legacypipe.common import Decals

def main():
    decals = Decals()
    ccds = decals.get_ccds()

    I = decals.photometric_ccds(ccds)
    ccds.photometric = np.zeros(len(ccds), bool)
    ccds.photometric[I] = True

    I = decals.apply_blacklist(ccds)
    ccds.blacklist_ok = np.zeros(len(ccds), bool)
    ccds.blacklist_ok[I] = True

    ccds.good_region = np.empty((len(ccds), 4), np.int16)
    ccds.good_region[:,:] = -1
    
    for iccd,ccd in enumerate(ccds):
        im = decals.get_image_object(ccd)

        X = im.get_good_image_subregion()
        for i,x in enumerate(X):
            if x is not None:
                ccds.good_region[iccd,i] = x

        psf = None
        try:
            psf = im.read_psf_model(pixPsf=True)
        except:
            import traceback
            traceback.print_exc()

        if psf is not None:
            print('Got PSF', psf)
            pass

        sky = None
        try:
            sky = im.read_sky_model(splinesky=True)
        except:
            import traceback
            traceback.print_exc()

        if sky is not None:
            print('Got sky', sky)
            pass

        wcs = None
        try:
            wcs = im.read_pv_wcs()
        except:
            import traceback
            traceback.print_exc()

        if wcs is not None:
            print('Got WCS', wcs)
            pass




if __name__ == '__main__':
    import sys
    sys.exit(main())
