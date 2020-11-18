'''
After running forced_photom.py on a set of CCDs, this script merges
the results back into a catalog.
'''
import sys
import logging
import numpy as np
from glob import glob
from collections import Counter

from astrometry.util.fits import fits_table, merge_tables

from legacypipe.survey import LegacySurveyData

def merge_forced(survey, brickname, cat, bands='grz', Nmax=0):
    ccdfn = survey.find_file('ccds-table', brick=brickname)
    CCDs = fits_table(ccdfn)
    print('Read', len(CCDs), 'CCDs')

    # objects in the catalog: (release,brickid,objid)
    catobjs = set([(r,b,o) for r,b,o in
                   zip(cat.release, cat.brickid, cat.objid)])

    # (release, brickid, objid, band) -> [ index in forced-phot table ]
    phot_index = {}

    camexp = set()
    FF = []
    for ccd in CCDs:
        cam = ccd.camera.strip()
        key = (cam, ccd.expnum)
        if key in camexp:
            # already read this camera-expnum
            continue
        camexp.add(key)
        ffn = survey.find_file('forced', camera=cam, expnum=ccd.expnum)
        print('Forced phot filename:', ffn)

        # DEBUG
        # if not os.path.exists(ffn):
        #     cmd = 'rsync -LRarv cori:/global/cscratch1/sd/dstn/dr9-forced/./%s dr9-north' % ffn.replace('dr9-north', '')
        #     print(cmd)
        #     os.system(cmd)

        F = fits_table(ffn)
        print('Read', len(F), 'forced-phot entries for CCD')
        ikeep = []
        for i,(r,b,o) in enumerate(zip(F.release, F.brickid, F.objid)):
            if (r,b,o) in catobjs:
                ikeep.append(i)
        if len(ikeep) == 0:
            print('No catalog objects found in this forced-phot table.')
            continue
        F.cut(np.array(ikeep))
        print('Cut to', len(F), 'phot entries matching catalog')
        FF.append(F)
    F = merge_tables(FF)
    del FF

    I = np.lexsort((F.ccdname, F.expnum, F.camera,
                    F.filter, F.objid, F.brickid, F.release))
    F.cut(I)

    for i,(r,brick,o,band) in enumerate(
            zip(F.release, F.brickid, F.objid, F.filter)):
        key = (r,brick,o,band)
        if not key in phot_index:
            phot_index[key] = []
        phot_index[key].append(i)

    if Nmax == 0:
        # find largest number of photometry measurements!
        Nmax = max([len(inds) for inds in phot_index.values()])
        print('Maximum number of photometry entries:', Nmax)

    for band in bands:
        nobs = np.zeros(len(cat), np.int16)
        indx = np.empty((len(cat), Nmax), np.int32)
        indx[:,:] = -1
        cat.set('nobs_%s' % band, nobs)
        cat.set('forced_index_%s' % band, indx)

        for i,(r,brick,o) in enumerate(zip(cat.release, cat.brickid, cat.objid)):
            key = (r,brick,o,band)
            try:
                inds = phot_index[key]
            except KeyError:
                continue
            nobs[i] = len(inds)
            indx[i, :min(Nmax, len(inds))] = inds
    return cat, F

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--brick',
        help='Brick name to run; required unless --radec is given')
    parser.add_argument('--survey-dir', type=str, default=None,
                        help='Override the $LEGACY_SURVEY_DIR environment variable')
    parser.add_argument('-d', '--outdir', dest='output_dir',
                        help='Set output base directory, default "."')
    parser.add_argument('-r', '--run', default=None,
                        help='Set the run type to execute (for images)')

    parser.add_argument('--catalog', help='Use the given FITS catalog file, rather than reading from a data release directory')
    parser.add_argument('--catalog-dir', help='Set LEGACY_SURVEY_DIR to use to read catalogs')
    parser.add_argument('--catalog-dir-north', help='Set LEGACY_SURVEY_DIR to use to read Northern catalogs')
    parser.add_argument('--catalog-dir-south', help='Set LEGACY_SURVEY_DIR to use to read Southern catalogs')
    parser.add_argument('--catalog-resolve-dec-ngc', type=float, help='Dec at which to switch from Northern to Southern catalogs (NGC only)', default=32.375)

    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')

    opt = parser.parse_args()
    if opt.brick is None:
        parser.print_help()
        return -1
    verbose = opt.verbose
    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    from legacypipe.runs import get_survey
    survey = get_survey(opt.run,
                        survey_dir=opt.survey_dir,
                        output_dir=opt.output_dir)

    cat = None
    catsurvey = survey
    if opt.catalog is not None:
        cat = fits_table(opt.catalog)
        print('Read', len(cat), 'sources from', opt.catalog)
    else:
        from astrometry.util.starutil_numpy import radectolb
        brick = survey.get_brick_by_name(opt.brick)
        l,b = radectolb(brick.ra, brick.dec)
        # NGC and above resolve line? -> north
        if b > 0 and brick.dec >= opt.catalog_resolve_dec_ngc:
            if opt.catalog_dir_north:
                catsurvey = LegacySurveyData(survey_dir=opt.catalog_dir_north)
        else:
            if opt.catalog_dir_south:
                catsurvey = LegacySurveyData(survey_dir=opt.catalog_dir_south)

        fn = catsurvey.find_file('tractor', brick=opt.brick)
        cat = fits_table(fn, columns=[
            'ra', 'dec', 'brick_primary', 'type', 'release',
            'brickid', 'brickname', 'objid', 'flux_r',
            'sersic', 'shape_r', 'shape_e1', 'shape_e2',
            'ref_epoch', 'pmra', 'pmdec', 'parallax'
            ])
        print('Read', len(cat), 'sources from', fn)

    cat,forced = merge_forced(survey, opt.brick, cat)
    outfn = 'merged.fits'
    cat.writeto(outfn)
    forced.writeto(outfn, append=True)

if __name__ == '__main__':
    main()



'''
fns = glob('forced/*/*/forced-*.fits')
F = merge_tables([fits_table(fn) for fn in fns])

dr6 = LegacySurveyData('/project/projectdirs/cosmo/data/legacysurvey/dr6')
B = dr6.get_bricks_readonly()

I = np.flatnonzero((B.ra1 < F.ra.max()) * (B.ra2 > F.ra.min()) * (B.dec1 < F.dec.max()) * (B.dec2 > F.dec.min()))
print(len(I), 'bricks')
T = merge_tables([fits_table(dr6.find_file('tractor', brick=B.brickname[i])) for i in I])
print(len(T), 'sources')
T.cut(T.brick_primary)
print(len(T), 'primary')

# map from F to T index
imap = dict([((b,o),i) for i,(b,o) in enumerate(zip(T.brickid, T.objid))])
F.tindex = np.array([imap[(b,o)] for b,o in zip(F.brickid, F.objid)])
assert(np.all(T.brickid[F.tindex] == F.brickid))
assert(np.all(T.objid[F.tindex] == F.objid))

fcols = 'apflux apflux_ivar camera expnum ccdname exptime flux flux_ivar fracflux mask mjd rchi2 x y brickid objid'.split()

bands = np.unique(F.filter)
for band in bands:
    Fb = F[F.filter == band]
    print(len(Fb), 'in band', band)
    c = Counter(zip(Fb.brickid, Fb.objid))
    NB = c.most_common()[0][1]
    print('Maximum of', NB, 'exposures per object')

    # we use uint8 below...
    assert(NB < 256)

    sourcearrays = []
    sourcearrays2 = []
    destarrays = []
    destarrays2 = []
    for c in fcols:
        src = Fb.get(c)
        if len(src.shape) == 2:
            narray = src.shape[1]
            dest = np.zeros((len(T), Nb, narray), src.dtype)
            T.set('forced_%s_%s' % (band, c), dest)
            sourcearrays2.append(src)
            destarrays2.append(dest)
        else:
            dest = np.zeros((len(T), Nb), src.dtype)
            T.set('forced_%s_%s' % (band, c), dest)
            sourcearrays.append(src)
            destarrays.append(dest)
    nf = np.zeros(len(T), np.uint8)
    T.set('forced_%s_n' % band, nf)
    for i,ti in enumerate(Fb.tindex):
        k = nf[ti]
        for src,dest in zip(sourcearrays, destarrays):
            dest[ti,k] = src[i]
        for src,dest in zip(sourcearrays2, destarrays2):
            dest[ti,k,:] = src[i,:]
        nf[ti] += 1

for band in bands:
    flux = T.get('forced_%s_flux' % band)
    ivar = T.get('forced_%s_flux_ivar' % band)
    miv = np.sum(ivar, axis=1)
    T.set('forced_%s_mean_flux' % band, np.sum(flux * ivar, axis=1) / np.maximum(1e-16, miv))
    T.set('forced_%s_mean_flux_ivar' % band, miv)

#K = np.flatnonzero(np.logical_or(T.forced_mean_u_flux_ivar > 0, T.forced_mean_r_flux_ivar > 0))
#T[K].writeto('forced/forced-cfis-deep2f2.fits')

T.writeto('forced-merged.fits')

'''
