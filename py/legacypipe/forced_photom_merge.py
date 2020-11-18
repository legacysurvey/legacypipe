'''
After running forced_photom.py on a set of CCDs, this script merges
the results back into a catalog.
'''
import sys
import logging
import numpy as np
from glob import glob
from collections import Counter

from astrometry.util.fits import *

from legacypipe.survey import LegacySurveyData

def merge_forced(survey, brickname):
    ccdfn = survey.find_file('ccds-table', brick=brickname)
    CCDs = fits_table(ccdfn)
    print('Read', len(CCDs), 'CCDs')
    camexp = set()
    for ccd in CCDs:
        cam = ccd.camera.strip()
        key = (cam, ccd.expnum)
        if key in camexp:
            # already read this camera-expnum
            continue
        camexp.add(key)
        ffn = survey.find_file('forced', camera=cam, expnum=ccd.expnum)
        print('Forced phot filename:', ffn)
        F = fits_table(ffn)
        


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
                        help='Set the run type to execute')
    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        default=0, help='Make more verbose')

    opt = parser.parse_args()
    if opt.brick is None:
        parser.print_help()
        return -1
    #optdict = vars(opt)
    #verbose = optdict.pop('verbose')
    verbose = opt.verbose

    from legacypipe.runs import get_survey
    survey = get_survey(opt.run,
                        survey_dir=opt.survey_dir,
                        output_dir=opt.output_dir)
    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)

    merge_forced(survey, opt.brick)
    
        
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
