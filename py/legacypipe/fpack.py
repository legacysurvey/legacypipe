from __future__ import print_function

import fitsio

from astrometry.util.fits import *

if __name__ == '__main__':
    #B = fits_table('cosmo/data/legacysurvey/dr2/decals-bricks-dr2.fits

    decals = Decals()
    bricks = decals.get_bricks_dr2()
    print(len(bricks), 'in DR2 bricks')
    bricks.cut(reduce(np.logical_or, [bricks.nobs_max_g > 0,
                                      bricks.nobs_max_r > 0,
                                      bricks.nobs_max_z > 0,]))
    print(len(bricks), 'with actual coverage')

    nobs = dict([(band, bricks.get('nobs_max_%s' % band)) for band in 'grz'])
    for ibrick,brick in enumerate(bricks):
        for band in 'grz':
            if nobs[band][ibrick] == 0:
                print('brick', brick.brickname, 'band', band, ': no coverage')
                continue

            # chi2
            # depth
            # galdepth
            # image
            # invvar
            # model
            # nexp

        # X ccds.fits
        # X depth.fits

'''
FZALGOR = 'RICE_1  ' # Default
FZQMETHD= 'SUBTRACTIVE_DITHER_2' # zero-valued pixels are not dithered
FZQVALUE=                    4  # Default
'''

'''
decals-0001m002-ccds.fits
decals-0001m002-chi2-g.fits
decals-0001m002-chi2-r.fits
decals-0001m002-chi2-z.fits
decals-0001m002-depth-g.fits.gz
decals-0001m002-depth-r.fits.gz
decals-0001m002-depth-z.fits.gz
decals-0001m002-depth.fits
decals-0001m002-galdepth-g.fits.gz
decals-0001m002-galdepth-r.fits.gz
decals-0001m002-galdepth-z.fits.gz
decals-0001m002-image-g.fits
decals-0001m002-image-r.fits
decals-0001m002-image-z.fits
decals-0001m002-image.jpg
decals-0001m002-invvar-g.fits
decals-0001m002-invvar-r.fits
decals-0001m002-invvar-z.fits
decals-0001m002-model-g.fits.gz
decals-0001m002-model-r.fits.gz
decals-0001m002-model-z.fits.gz
decals-0001m002-model.jpg
decals-0001m002-nexp-g.fits.gz
decals-0001m002-nexp-r.fits.gz
decals-0001m002-nexp-z.fits.gz
decals-0001m002-resid.jpg
legacysurvey_dr2_coadd_000_0001m002.sha1sum
'''


            
        
