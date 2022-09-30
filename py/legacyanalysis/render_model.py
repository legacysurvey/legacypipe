import os
import sys
import numpy as np
import fitsio
from astrometry.util.fits import fits_table
from astrometry.util.util import Tan
from tractor import Tractor
from legacypipe.survey import LegacySurveyData
from legacypipe.catalog import read_fits_catalog
from legacypipe.outliers import read_outlier_mask_file
from legacypipe.coadds import make_coadds

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', help='Catalog to render')
    parser.add_argument('--brickname', help='Optional, read CCDs from $LEGACY_SURVEY_DIR for the given brick name')
    parser.add_argument('--ccds', help='Use this table of CCDs')
    parser.add_argument('--wcs', help='File containing a WCS header describing the coadd WCS to render.')
    parser.add_argument('--wcs-ext', type=int, help='FITS file extension containing a WCS header describing the coadd WCS to render.', default=0)
    parser.add_argument('--outlier-mask-brick', help='Comma-separated list of bricknames from which outlier masks should be read.')
    parser.add_argument('--out', help='Filename pattern ("BAND" will be replaced by band name) of output images.')
    parser.add_argument('--resid', help='Filename pattern ("BAND" will be replaced by band name) of residual images.')
    parser.add_argument('--jpeg', help='Write RGB image to this filename')
    parser.add_argument('--resid-jpeg', help='Write RGB residual image to this filename')
    opt = parser.parse_args()

    if opt.catalog is None:
        print('Need catalog!')
        return -1
    cat = fits_table(opt.catalog)
    

    survey = LegacySurveyData()

    if opt.wcs is None:
        print('FIXME')
        return -1
    else:
        wcs = Tan(opt.wcs, opt.wcs_ext)
        
    tcat = read_fits_catalog(cat)

    ccds = None
    if opt.ccds:
        ccdfn = opt.ccds
    elif opt.brickname is not None:
        ccdfn = survey.find_file('ccds-table', brick=opt.brickname)
    else:
        ccds = survey.ccds_touching_wcs(wcs)
    if ccds is None:
        print('Reading', ccdfn)
        ccds = fits_table(ccdfn)

    H,W = wcs.shape
    targetrd = np.array([wcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])

    tims = []
    for ccd in ccds:
        im = survey.get_image_object(ccd)
        #slc = slice(ccd.ccd_y0, ccd.ccd_y1), slice(ccd.ccd_x0, ccd.ccd_x1)
        #tim = im.get_tractor_image(slc=slc)
        tim = im.get_tractor_image(radecpoly=targetrd)
        print('Read', tim)
        tims.append(tim)

    if opt.outlier_mask_brick is not None:
        bricks = opt.outlier_mask_brick.split(',')
        for b in bricks:
            print('Reading outlier mask for brick', b,
                  ':', survey.find_file('outliers_mask', brick=b, output=False))
            ok = read_outlier_mask_file(survey, tims, b,
                                        subimage=True, output=False)

        
    tr = Tractor(tims, tcat)
    mods = list(tr.getModelImages())

    bands = 'grz'

    def write_model(band, cowimg=None, cowmod=None, **kwargs):
        if cowmod is None:
            print('No model for', band)
            return
        outfn = opt.out.replace('BAND', band)
        fitsio.write(outfn, cowmod, clobber=True)
        print('Wrote model for', band, 'to', outfn)
        if opt.resid:
            outfn = opt.resid.replace('BAND', band)
            fitsio.write(outfn, cowimg - cowmod, clobber=True)
            print('Wrote resid for', band, 'to', outfn)

    C = make_coadds(tims, bands, wcs, mods=mods,
                    callback=write_model)
    if opt.jpeg:
        from legacypipe.survey import get_rgb
        import pylab as plt
        plt.imsave(opt.jpeg, get_rgb(C.comods, bands), origin='lower')
    if opt.resid_jpeg:
        from legacypipe.survey import get_rgb
        import pylab as plt
        plt.imsave(opt.resid_jpeg,
                   get_rgb([im-mod for im,mod in zip(C.coimgs, C.comods)], bands),
                   origin='lower')

        
if __name__ == '__main__':
    main()
    
