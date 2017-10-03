from legacypipe.survey import LegacySurveyData, wcs_for_brick, read_one_tim
import numpy as np

def main():
    survey = LegacySurveyData()

    brickname = '2351p137'
    # RA,Dec = 235.0442, 13.7125
    bx,by = 3300, 1285
    sz = 50
    bbox = [bx-sz, bx+sz, by-sz, by+sz]
    objid = 1394
    bands = ['g','r','z']

    from legacypipe.runbrick import stage_tims, _get_mod, rgbkwargs, rgbkwargs_resid
    from legacypipe.survey import get_rgb, imsave_jpeg
    from legacypipe.coadds import make_coadds
    from astrometry.util.multiproc import multiproc
    from astrometry.util.fits import fits_table
    from legacypipe.catalog import read_fits_catalog

    # brick = survey.get_brick_by_name(brickname)
    # # Get WCS object describing brick
    # targetwcs = wcs_for_brick(brick)
    # (x0,x1,y0,y1) = bbox
    # W = x1-x0
    # H = y1-y0
    # targetwcs = targetwcs.get_subimage(x0, y0, W, H)
    # H,W = targetwcs.shape

    mp = multiproc()
    P = stage_tims(brickname=brickname, survey=survey, target_extent=bbox,
                   pixPsf=True, hybridPsf=True, depth_cut=False, mp=mp)
    print('Got', P.keys())

    tims = P['tims']
    targetwcs = P['targetwcs']
    H,W = targetwcs.shape

    # Read Tractor catalog
    fn = survey.find_file('tractor', brick=brickname)
    print('Trying to read catalog', fn)
    cat = fits_table(fn)
    print('Read', len(cat), 'sources')
    ok,xx,yy = targetwcs.radec2pixelxy(cat.ra, cat.dec)
    I = np.flatnonzero((xx > 0) * (xx < W) * (yy > 0) * (yy < H))
    cat.cut(I)
    print('Cut to', len(cat), 'sources within box')

    I = np.flatnonzero(cat.objid != objid)
    cat.cut(I)
    print('Cut to', len(cat), 'sources with objid !=', objid)

    #cat.about()
    # Convert FITS catalog into tractor source objects
    print('Creating tractor sources...')

    srcs = read_fits_catalog(cat, fluxPrefix='')
    print('Sources:')
    for src in srcs:
        print(' ', src)

    print('Rendering model images...')
    mods = [_get_mod((tim,srcs)) for tim in tims]

    print('Producing coadds...')
    C = make_coadds(tims, bands, targetwcs, mods=mods, mp=mp)
    print('Coadds:', dir(C))

    coadd_list= [('image', C.coimgs,   rgbkwargs),
                 ('model', C.comods,   rgbkwargs),
                 ('resid', C.coresids, rgbkwargs_resid)]
    #C.coimgs, C.comods, C.coresids
    for name,ims,rgbkw in coadd_list:
        rgb = get_rgb(ims, bands, **rgbkw)
        kwa = {}
        #with survey.write_output(name + '-jpeg', brick=brickname) as out:
        #    imsave_jpeg(out.fn, rgb, origin='lower', **kwa)
        #    print('Wrote', out.fn)
        outfn = name + '.jpg'
        imsave_jpeg(outfn, rgb, origin='lower', **kwa)
        del rgb


    # from legacypipe.runbrick import run_brick
    # P = run_brick(brickname, survey, zoom=bbox,
    #               stages=['tims'], write_pickles=False,
    #               depth_cut=False, do_calibs=False, pixPsf=True, hybridPsf=True,
    #               rex=True, splinesky=True)
    # 
    # print('Got', P.keys())

    # brick = survey.get_brick_by_name(brickname)
    # # Get WCS object describing brick
    # targetwcs = wcs_for_brick(brick)
    # (x0,x1,y0,y1) = bbox
    # W = x1-x0
    # H = y1-y0
    # targetwcs = targetwcs.get_subimage(x0, y0, W, H)
    # pixscale = targetwcs.pixel_scale()
    # targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
    #                      [(1,1),(W,1),(W,H),(1,H),(1,1)]])
    # ccds = survey.ccds_touching_wcs(targetwcs, ccdrad=None)
    # print(len(ccds), 'CCDs touching target WCS')
    # print('Unique filters:', np.unique(ccds.filter))
    # ccds.cut(np.in1d(ccds.filter, bands))
    # print('Cut on filter:', len(ccds), 'CCDs remain.')
    # 
    # print('Cutting out non-photometric CCDs...')
    # I = survey.photometric_ccds(ccds)
    # if I is None:
    #     print('None cut')
    # else:
    #     print(len(I), 'of', len(ccds), 'CCDs are photometric')
    #     ccds.cut(I)
    # 
    # print('Applying CCD cuts...')
    # ccds.ccd_cuts = survey.ccd_cuts(ccds)
    # cutvals = ccds.ccd_cuts
    # print('CCD cut bitmask values:', cutvals)
    # ccds.cut(cutvals == 0)
    # print(len(ccds), 'CCDs survive cuts')
    # 
    # print('Cutting on CCDs to be used for fitting...')
    # I = survey.ccds_for_fitting(brick, ccds)
    # if I is not None:
    #     print('Cutting to', len(I), 'of', len(ccds), 'CCDs for fitting.')
    #     ccds.cut(I)
    # 
    # # Create Image objects for each CCD
    # ims = []
    # for ccd in ccds:
    #     im = survey.get_image_object(ccd)
    #     ims.append(im)
    #     print(im, im.band, 'exptime', im.exptime, 'propid', ccd.propid,
    #           'seeing %.2f' % (ccd.fwhm*im.pixscale),
    #           'object', getattr(ccd, 'object', None))
    # # Read Tractor images
    # tims = [read_one_tim((im, targetrd, dict(pixPsf=True, hybridPsf=True, splinesky=True)))
    #         for im in ims]
    # # Cut the table of CCDs to match the 'tims' list
    # I = np.array([i for i,tim in enumerate(tims) if tim is not None])
    # ccds.cut(I)
    # tims = [tim for tim in tims if tim is not None]
    # assert(len(ccds) == len(tims))
    # 
    # print('Read', len(tims), 'tims')

if __name__ == '__main__':
    main()


