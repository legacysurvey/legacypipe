'''
This script is used to produce lists of CCDs or bricks, for production
purposes (building qdo queue, eg).

python legacypipe/queue-calibs.py  | qdo load cal -


eg, DR3:

* Staging to $SCRATCH to images required to run the EDR region:

module switch legacysurvey/dr3
python legacypipe/queue-calibs.py --region edr --write-ccds edr-ccds.fits --calibs --touching --nper 100
for x in $(tablist edr-ccds.fits"[col image_filename]" | awk '{print $2}' | sort | uniq); do
  rsync -LRarv $LEGACY_SURVEY_DIR/images/./$(echo $x | sed s/o.i/o??/g) /global/cscratch1/sd/desiproc/legacypipe-dir/images/;
done

* Running calibration preprocessing for EDR:

module switch legacysurvey/dr3-cori-scratch
python legacypipe/queue-calibs.py --region edr --calibs --touching --nper 100
qdo load cal jobs

# qdo launch cal 1 --cores_per_worker 1 --batchqueue shared --script "python legacypipe/run-calib.py --splinesky" --walltime 4:00:00 --keep_env --batchopts "-a 0-15"

qdo launch cal 1 --cores_per_worker 1 --batchqueue shared --script $(pwd)/cal.sh --walltime 4:00:00 --keep_env --batchopts "-a 0-15"
cal.sh:
  cd /global/cscratch1/sd/desiproc/code/legacypipe/py
  python legacypipe/run-calib.py --splinesky $*

* Running bricks for EDR:

python legacypipe/queue-calibs.py --region edr --brickq 0 > bricks
grep '^....[pm]...$' bricks | qdo load edr0 -



dr1(d):
qdo launch bricks 16 --mpack 6 --batchopts "-A desi" --walltime=24:00:00 --script projects/desi/pipebrick.sh --batchqueue regular --verbose

qdo launch cal 1 --batchopts "-A cosmo -t 1-50" --walltime=24:00:00 --batchqueue serial --script projects/desi/run-calib.py
#qdo launch cal 1 --batchopts "-A cosmo -t 1-10" --walltime=24:00:00 --batchqueue serial

Or
qdo launch cal 8 --batchopts "-A cosmo -t 1-6" --pack --walltime=30:00 --batchqueue debug --script projects/desi/run-calib.py

Or lists of bricks to run in production:

python projects/desi/queue-calibs.py  | qdo load bricks -
qdo launch bricks 1 --batchopts "-A cosmo -t 1-10 -l walltime=24:00:00 -q serial -o pipebrick-logs -j oe -l pvmem=6GB" \
    --script projects/desi/pipebrick.sh
'''
from __future__ import print_function
import sys
import os
import numpy as np
from collections import OrderedDict

from astrometry.util.fits import fits_table
from astrometry.util.file import trymakedirs

from legacypipe.common import LegacySurveyData, wcs_for_brick, ccds_touching_wcs


from astrometry.libkd.spherematch import match_radec


import matplotlib
matplotlib.use('Agg')
import pylab as plt
from glob import glob

def log(*s):
    print(' '.join([str(ss) for ss in s]), file=sys.stderr)

def main():
    """Main program.
    """
    import argparse

    parser = argparse.ArgumentParser(description="This script is used to produce lists of CCDs or bricks, for production purposes (building qdo queue, eg).")
    parser.add_argument('--calibs', action='store_true',
                      help='Output CCDs that need to be calibrated.')

    parser.add_argument('--nper', type=int, default=None,
                      help='Batch N calibs per line')

    parser.add_argument('--forced', action='store_true',
                      help='Output forced-photometry commands')

    parser.add_argument('--lsb', action='store_true',
                      help='Output Low-Surface-Brightness commands')

    parser.add_argument('--touching', action='store_true',
                      help='Cut to only CCDs touching selected bricks')

    parser.add_argument('--check', action='store_true',
                      help='Check which calibrations actually need to run.')
    parser.add_argument('--check-coadd', action='store_true',
                      help='Check which caoadds actually need to run.')
    parser.add_argument('--out', help='Output filename for calibs, default %(default)s',
                      default='jobs')
    parser.add_argument('--command', action='store_true',
                      help='Write out full command-line to run calib')

    parser.add_argument('--maxdec', type=float, help='Maximum Dec to run')
    parser.add_argument('--mindec', type=float, help='Minimum Dec to run')

    parser.add_argument('--region', help='Region to select')

    parser.add_argument('--bricks', help='Set bricks.fits file to load')
    parser.add_argument('--ccds', help='Set ccds.fits file to load')

    parser.add_argument('--delete-sky', action='store_true',
                      help='Delete any existing sky calibration files')
    parser.add_argument('--delete-pvastrom', action='store_true',
                      help='Delete any existing PV WCS calibration files')

    parser.add_argument('--write-ccds', help='Write CCDs list as FITS table?')

    parser.add_argument('--brickq', type=int, default=None,
                        help='Queue only bricks with the given "brickq" value [0 to 3]')
    
    opt = parser.parse_args()

    survey = LegacySurveyData()
    if opt.bricks is not None:
        B = fits_table(opt.bricks)
        log('Read', len(B), 'from', opt.bricks)
    else:
        B = survey.get_bricks()

    if opt.ccds is not None:
        T = fits_table(opt.ccds)
        log('Read', len(T), 'from', opt.ccds)
    else:
        T = survey.get_ccds()
        log(len(T), 'CCDs')
    T.index = np.arange(len(T))

    # I,J,d,counts = match_radec(B.ra, B.dec, T.ra, T.dec, 0.2, nearest=True, count=True)
    # plt.clf()
    # plt.hist(counts, counts.max()+1)
    # plt.savefig('bricks.png')
    # B.cut(I[counts >= 9])
    # plt.clf()
    # plt.plot(B.ra, B.dec, 'b.')
    # #plt.scatter(B.ra[I], B.dec[I], c=counts)
    # plt.savefig('bricks2.png')


    # DES Stripe82
    #rlo,rhi = 350.,360.
    # rlo,rhi = 300., 10.
    # dlo,dhi = -6., 4.
    # TINY bit
    #rlo,rhi = 350.,351.1
    #dlo,dhi = 0., 1.1

    # EDR+
    # 860 bricks
    # ~10,000 CCDs
    #rlo,rhi = 239,246
    #dlo,dhi =   5, 13

    # DR1
    #rlo,rhi = 0, 360
    # part 1
    #dlo,dhi = 25, 40
    # part 2
    #dlo,dhi = 20,25
    # part 3
    #dlo,dhi = 15,20
    # part 4
    #dlo,dhi = 10,15
    # part 5
    #dlo,dhi = 5,10
    # the rest
    #dlo,dhi = -11, 5
    #dlo,dhi = 15,25.5

    dlo,dhi = -15, 40
    rlo,rhi = 0, 360

    # Arjun says 3x3 coverage area is roughly
    # RA=240-252 DEC=6-12 (but not completely rectangular)

    # COSMOS
    #rlo,rhi = 148.9, 151.2
    #dlo,dhi = 0.9, 3.5

    # A nice well-behaved region (EDR2/3)
    # rlo,rhi = 243.6, 244.6
    # dlo,dhi = 8.1, 8.6

    # 56 bricks, ~725 CCDs
    #B.cut((B.ra > 240) * (B.ra < 242) * (B.dec > 5) * (B.dec < 7))
    # 240 bricks, ~3000 CCDs
    #B.cut((B.ra > 240) * (B.ra < 244) * (B.dec > 5) * (B.dec < 9))
    # 535 bricks, ~7000 CCDs
    #B.cut((B.ra > 240) * (B.ra < 245) * (B.dec > 5) * (B.dec < 12))


    if opt.region in ['test1', 'test2', 'test3', 'test4']:
        nm = dict(test1='2446p115', # weird stuff around bright star
                  test2='1183p292', # faint sources around bright galaxy
                  test3='3503p005', # DES
                  test4='1163p277', # Pollux
                  )[opt.region]

        B.cut(np.flatnonzero(np.array([s == nm for s in B.brickname])))
        log('Cut to', len(B), 'bricks')
        log(B.ra, B.dec)
        dlo,dhi = -90,90
        rlo,rhi = 0, 360

    elif opt.region == 'badsky':
        # A handful of exposures in DR2 with inconsistent sky estimates.
        #C = decals.get_ccds()
        #log(len(C), 'CCDs')
        #C.cut((C.expnum >= 257400) * (C.expnum < 257500))
        T.cut(np.array([e in [257460, 257461, 257462, 257463, 257464,
                              257465, 257466, 257467, 257469, 257472,
                              257483, 257496]
                        for e in T.expnum]))
        log(len(T), 'CCDs with bad sky')
        # CCD radius
        radius = np.hypot(2048, 4096) / 2. * 0.262 / 3600.
        # Brick radius
        radius += np.hypot(survey.bricksize, survey.bricksize)/2.
        I,J,d = match_radec(B.ra, B.dec, T.ra, T.dec, radius * 1.05)
        keep = np.zeros(len(B), bool)
        keep[I] = True
        B.cut(keep)
        log('Cut to', len(B), 'bricks near CCDs with bad sky')

    elif opt.region == 'badsky2':
        # UGH, missed this one in original 'badsky' definition.
        T.cut(T.expnum == 257466)
        log(len(T), 'CCDs with bad sky')
        # CCD radius
        radius = np.hypot(2048, 4096) / 2. * 0.262 / 3600.
        # Brick radius
        radius += np.hypot(survey.bricksize, survey.bricksize)/2.
        I,J,d = match_radec(B.ra, B.dec, T.ra, T.dec, radius * 1.05)
        keep = np.zeros(len(B), bool)
        keep[I] = True
        B.cut(keep)
        log('Cut to', len(B), 'bricks near CCDs with bad sky')

    elif opt.region == 'edr':
        # EDR:
        # 535 bricks, ~7000 CCDs
        rlo,rhi = 240,245
        dlo,dhi =   5, 12

    elif opt.region == 'edr-south':
        rlo,rhi = 240,245
        dlo,dhi =   5, 10

    elif opt.region == 'cosmos1':
        # 16 bricks in the core of the COSMOS field.
        rlo,rhi = 149.75, 150.75
        dlo,dhi = 1.6, 2.6

    elif opt.region == 'pristine':
        # Stream?
        rlo,rhi = 240,250
        dlo,dhi = 10,15

    elif opt.region == 'des':
        dlo, dhi = -6., 4.
        rlo, rhi = 317., 7.

        T.cut(np.flatnonzero(np.array(['CPDES82' in fn for fn in T.cpimage])))
        log('Cut to', len(T), 'CCDs with "CPDES82" in filename')

    elif opt.region == 'subdes':
        rlo,rhi = 320., 360.
        dlo,dhi = -1.25, 1.25

    elif opt.region == 'northwest':
        rlo,rhi = 240,360
        dlo,dhi = 20,40
    elif opt.region == 'north':
        rlo,rhi = 120,240
        dlo,dhi = 20,40
    elif opt.region == 'northeast':
        rlo,rhi = 0,120
        dlo,dhi = 20,40
    elif opt.region == 'southwest':
        rlo,rhi = 240,360
        dlo,dhi = -20,0
    elif opt.region == 'south':
        rlo,rhi = 120,240
        dlo,dhi = -20,0
    elif opt.region == 'southeast':
        rlo,rhi = 0,120
        dlo,dhi = -20,0
    elif opt.region == 'midwest':
        rlo,rhi = 240,360
        dlo,dhi = 0,20
    elif opt.region == 'middle':
        rlo,rhi = 120,240
        dlo,dhi = 0,20
    elif opt.region == 'mideast':
        rlo,rhi = 0,120
        dlo,dhi = 0,20

    elif opt.region == 'grz':
        # Bricks with grz coverage.
        # Be sure to use  --bricks survey-bricks-in-dr1.fits
        # which has_[grz] columns.
        B.cut((B.has_g == 1) * (B.has_r == 1) * (B.has_z == 1))
        log('Cut to', len(B), 'bricks with grz coverage')

    elif opt.region == 'nogrz':
        # Bricks without grz coverage.
        # Be sure to use  --bricks survey-bricks-in-dr1.fits
        # which has_[grz] columns.
        B.cut(np.logical_not((B.has_g == 1) * (B.has_r == 1) * (B.has_z == 1)))
        log('Cut to', len(B), 'bricks withOUT grz coverage')
    elif opt.region == 'deep2':
        rlo,rhi = 250,260
        dlo,dhi = 30,35

    elif opt.region == 'virgo':
        rlo,rhi = 185,190
        dlo,dhi =  10, 15

    elif opt.region == 'virgo2':
        rlo,rhi = 182,192
        dlo,dhi =   8, 18

    elif opt.region == 'lsb':
        rlo,rhi = 147.2, 147.8
        dlo,dhi = -0.4, 0.4

    if opt.mindec is not None:
        dlo = opt.mindec
    if opt.maxdec is not None:
        dhi = opt.maxdec

    if rlo < rhi:
        B.cut((B.ra >= rlo) * (B.ra <= rhi) *
              (B.dec >= dlo) * (B.dec <= dhi))
    else: # RA wrap
        B.cut(np.logical_or(B.ra >= rlo, B.ra <= rhi) *
              (B.dec >= dlo) * (B.dec <= dhi))
    log(len(B), 'bricks in range')

    I,J,d = match_radec(B.ra, B.dec, T.ra, T.dec, survey.bricksize)
    keep = np.zeros(len(B), bool)
    for i in I:
        keep[i] = True
    B.cut(keep)
    log('Cut to', len(B), 'bricks near CCDs')

    if opt.brickq is not None:
        B.cut(B.brickq == opt.brickq)
        log('Cut to', len(B), 'with brickq =', opt.brickq)
    
    if opt.touching:
        keep = np.zeros(len(T), bool)
        for j in J:
            keep[j] = True
        T.cut(keep)
        log('Cut to', len(T), 'CCDs near bricks')

    # Aside -- how many near DR1=1 CCDs?
    if False:
        T2 = D.get_ccds()
        log(len(T2), 'CCDs')
        T2.cut(T2.dr1 == 1)
        log(len(T2), 'CCDs marked DR1=1')
        log(len(B), 'bricks in range')
        I,J,d = match_radec(B.ra, B.dec, T2.ra, T2.dec, survey.bricksize)
        keep = np.zeros(len(B), bool)
        for i in I:
            keep[i] = True
        B2 = B[keep]
        log('Total of', len(B2), 'bricks near CCDs with DR1=1')
        for band in 'grz':
            Tb = T2[T2.filter == band]
            log(len(Tb), 'in filter', band)
            I,J,d = match_radec(B2.ra, B2.dec, Tb.ra, Tb.dec, survey.bricksize)
            good = np.zeros(len(B2), np.uint8)
            for i in I:
                good[i] = 1
            B2.set('has_' + band, good)

        B2.writeto('survey-bricks-in-dr1.fits')
        sys.exit(0)

    # sort by dec decreasing
    B.cut(np.argsort(-B.dec))

    for b in B:
        if opt.check:
            fn = 'dr1n/tractor/%s/tractor-%s.fits' % (b.brickname[:3], b.brickname)
            if os.path.exists(fn):
                print('Exists:', fn, file=sys.stderr)
                continue
        if opt.check_coadd:
            fn = 'dr1b/coadd/%s/%s/decals-%s-image.jpg' % (b.brickname[:3], b.brickname, b.brickname)
            if os.path.exists(fn):
                print('Exists:', fn, file=sys.stderr)
                continue

        print(b.brickname)

    if not (opt.calibs or opt.forced or opt.lsb):
        sys.exit(0)

    bands = 'grz'
    log('Filters:', np.unique(T.filter))
    T.cut(np.flatnonzero(np.array([f in bands for f in T.filter])))
    log('Cut to', len(T), 'CCDs in filters', bands)

    if opt.touching:
        allI = set()
        for b in B:
            wcs = wcs_for_brick(b)
            I = ccds_touching_wcs(wcs, T)
            log(len(I), 'CCDs for brick', b.brickid, 'RA,Dec (%.2f, %.2f)' % (b.ra, b.dec))
            if len(I) == 0:
                continue
            allI.update(I)
        allI = list(allI)
        allI.sort()
    else:
        allI = np.arange(len(T))

    if opt.write_ccds:
        T[allI].writeto(opt.write_ccds)
        log('Wrote', opt.write_ccds)

    ## Be careful here -- T has been cut; we want to write out T.index.
    ## 'allI' contains indices into T.

    if opt.forced:
        log('Writing forced-photometry commands to', opt.out)
        f = open(opt.out,'w')
        log('Total of', len(allI), 'CCDs')
        for j,i in enumerate(allI):
            expstr = '%08i' % T.expnum[i]
            #outdir = os.path.join('forced', expstr[:5], expstr)
            #trymakedirs(outdir)
            outfn = os.path.join('forced', expstr[:5], expstr,
                                 'decam-%s-%s-forced.fits' %
                                 (expstr, T.ccdname[i]))
            imgfn = os.path.join(survey.survey_dir, 'images',
                                 T.image_filename[i].strip())
            if (not os.path.exists(imgfn) and
                imgfn.endswith('.fz') and
                os.path.exists(imgfn[:-3])):
                imgfn = imgfn[:-3]

            f.write('python legacypipe/forced-photom-decam.py %s %i DR1 %s\n' %
                    (imgfn, T.cpimage_hdu[i], outfn))

        f.close()
        log('Wrote', opt.out)
        sys.exit(0)

    if opt.lsb:
        log('Writing LSB commands to', opt.out)
        f = open(opt.out,'w')
        log('Total of', len(allI), 'CCDs')
        for j,i in enumerate(allI):
            exp = T.expnum[i]
            ext = T.ccdname[i].strip()
            outfn = 'lsb/lsb-%s-%s.fits' % (exp, ext)
            f.write('python projects/desi/lsb.py --expnum %i --extname %s --out %s -F -n > lsb/lsb-%s-%s.log 2>&1\n' % (exp, ext, outfn, exp, ext))
        f.close()
        log('Wrote', opt.out)
        sys.exit(0)


    log('Writing calibs to', opt.out)
    f = open(opt.out,'w')
    log('Total of', len(allI), 'CCDs')

    batch = []

    def write_batch(f, batch, cmd):
        if opt.command:
            s = '; '.join(batch)
        else:
            s = ' '.join(batch)
        f.write(s + '\n')


    for j,i in enumerate(allI):

        if opt.delete_sky or opt.delete_pvastrom:
            log(j+1, 'of', len(allI))
            im = survey.get_image_object(T[i])
            if opt.delete_sky and os.path.exists(im.skyfn):
                log('  deleting:', im.skyfn)
                os.unlink(im.skyfn)
            if opt.delete_pvastrom and os.path.exists(im.pvwcsfn):
                log('  deleting:', im.pvwcsfn)
                os.unlink(im.pvwcsfn)

        if opt.check:
            log(j+1, 'of', len(allI))
            im = survey.get_image_object(T[i])
            if not im.run_calibs(im, just_check=True):
                log('Calibs for', im.expnum, im.ccdname, im.calname, 'already done')
                continue

        if opt.command:
            s = ('python legacypipe/run-calib.py --expnum %i --ccdname %s' %
                 (T.expnum[i], T.ccdname[i]))
        else:
            s = '%i' % T.index[i]

        if j < 10:
            print('Index', T.index[i], 'expnum', T.expnum[i], 'ccdname', T.ccdname[i],
                  'filename', T.image_filename[i])
            
        if not opt.nper:
            f.write(s + '\n')
        else:
            batch.append(s)
            if len(batch) >= opt.nper:
                write_batch(f, batch, opt.command)
                batch = []

        if opt.check:
            f.flush()

    if len(batch):
        write_batch(f, batch, opt.command)

    f.close()
    log('Wrote', opt.out)
    return 0
#
#
#
if __name__ == '__main__':
    sys.exit(main())
