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

qdo launch edr0 16 --cores_per_worker 8 --batchqueue regular --walltime 12:00:00 --script ../bin/pipebrick.sh --keep_env

Brick stage 1:

python legacypipe/queue-calibs.py --region edr --brickq 1 > bricks1
grep '^....[pm]...$' bricks1 | awk '{print "/global/cscratch1/sd/desiproc/code/legacypipe/bin/pipebrick.sh",$1}' | qdo load jobs -


* Staging larger regions:

python legacypipe/queue-calibs.py --region northwest --write-ccds ccds-nw.fits --calibs --near --nper 100 --command --opt "--threads 8"

for x in $(tablist ccds-nw.fits"[col image_filename]" | awk '{print $2}' | sort | uniq); do   rsync -LRarv /project/projectdirs/desiproc/dr3/images/./$(echo $x | sed s/o.i/o??/g) /global/cscratch1/sd/desiproc/dr3/images/; done

qdo load --priority -10 jobs jobs


'''
from __future__ import print_function
import sys
import os
import numpy as np
from collections import OrderedDict
from glob import glob

import matplotlib
matplotlib.use('Agg')
import pylab as plt

from astrometry.util.fits import fits_table
from astrometry.util.file import trymakedirs

from legacypipe.survey import LegacySurveyData, wcs_for_brick, ccds_touching_wcs

from astrometry.libkd.spherematch import match_radec


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

    parser.add_argument('--stage', help='Stage image files to given directory')

    parser.add_argument('--touching', action='store_true',
                      help='Cut to only CCDs touching selected bricks')
    parser.add_argument('--near', action='store_true',
                      help='Quick cut to only CCDs near selected bricks')

    parser.add_argument('--check-coadd', action='store_true',
                      help='Check which caoadds actually need to run.')
    parser.add_argument('--out', help='Output filename for calibs, default %(default)s',
                      default='jobs')
    parser.add_argument('--command', action='store_true',
                      help='Write out full command-line to run calib')
    parser.add_argument('--opt', help='With --command, extra options to add')
    
    parser.add_argument('--maxra', type=float, help='Maximum RA to run')
    parser.add_argument('--minra', type=float, help='Minimum RA to run')
    parser.add_argument('--maxdec', type=float, help='Maximum Dec to run')
    parser.add_argument('--mindec', type=float, help='Minimum Dec to run')

    parser.add_argument('--region', help='Region to select')

    parser.add_argument('--bricks', help='Set bricks.fits file to load')
    parser.add_argument('--ccds', help='Set ccds.fits file to load')
    parser.add_argument('--ignore_cuts', action='store_true',default=False,help='no photometric cuts')
    parser.add_argument('--save_to_fits', action='store_true',default=False,help='save cut brick,ccd to fits table')
    parser.add_argument('--name', action='store',default='dr3',help='save with this suffix, e.g. refers to ccds table')

    parser.add_argument('--delete-sky', action='store_true',
                      help='Delete any existing sky calibration files')

    parser.add_argument('--write-ccds', help='Write CCDs list as FITS table?')

    parser.add_argument('--nccds', action='store_true', default=False, help='Prints number of CCDs per brick')

    parser.add_argument('--bands', default='g,r,z', help='Set bands to keep: comma-separated list.')


    opt = parser.parse_args()


    want_ccds = (opt.calibs or opt.forced or opt.lsb)
    want_bricks = not want_ccds



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

    if opt.ignore_cuts == False:
        print('Applying CCD cuts...')
        if 'ccd_cuts' in T.columns():
            T.cut(T.ccd_cuts == 0)
            print(len(T), 'CCDs survive cuts')

    bands = opt.bands.split(',')
    log('Filters:', np.unique(T.filter))
    T.cut(np.flatnonzero(np.array([f in bands for f in T.filter])))
    log('Cut to', len(T), 'CCDs in filters', bands)

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

    dlo,dhi = -25, 40
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

    elif opt.region == 'edr':
        # EDR:
        # 535 bricks, ~7000 CCDs
        rlo,rhi = 240,245
        dlo,dhi =   5, 12

    elif opt.region == 'edrplus':
        rlo,rhi = 235,248
        dlo,dhi =   5, 15

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
    elif opt.region == 'southsoutheast':
        rlo,rhi = 0,120
        dlo,dhi = -20,-10
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

    elif opt.region == 'deep2f2':
        rlo,rhi = 251.4, 254.4
        dlo,dhi =  34.6,  35.3

    elif opt.region == 'deep2f3':
        rlo,rhi = 351.25, 353.75
        dlo,dhi = 0, 0.5

    elif opt.region == 'deep3':
        rlo,rhi = 214,216
        dlo,dhi = 52.25,53.25

    elif opt.region == 'virgo':
        rlo,rhi = 185,190
        dlo,dhi =  10, 15

    elif opt.region == 'virgo2':
        rlo,rhi = 182,192
        dlo,dhi =   8, 18

    elif opt.region == 'coma':
        # van Dokkum et al Coma cluster ultra-diffuse galaxies: 3x3 field centered on Coma cluster
        rc,dc = 195., 28.
        dd = 1.5
        cosdec = np.cos(np.deg2rad(dc))
        rlo,rhi = rc - dd/cosdec, rc + dd/cosdec
        dlo,dhi = dc - dd, dc + dd

    elif opt.region == 'lsb':
        rlo,rhi = 147.2, 147.8
        dlo,dhi = -0.4, 0.4

    elif opt.region == 'eboss-sgc':
        # generous boundaries to make sure get all relevant images
        # RA -45 to +45
        # Dec -5 to +7
        rlo,rhi = 310., 50.
        dlo,dhi = -6., 6.

    elif opt.region == 'eboss-ngc':
        # generous boundaries to make sure get all relevant images
        # NGC ELGs
        # RA 115 to 175
        # Dec 15 to  30
        # rlo,rhi = 122., 177.
        # dlo,dhi =  12.,  32.
        rlo,rhi = 126., 168.
        dlo,dhi =  18.,  33.

    elif opt.region == 'mzls':
        dlo,dhi = -10., 90. # -10: pull in Stripe 82 data too

    elif opt.region == 'dr4-bootes':
        # https://desi.lbl.gov/trac/wiki/DecamLegacy/DR4sched 
        #dlo,dhi = 34., 35.
        #rlo,rhi = 209.5, 210.5
        dlo,dhi = 33., 36.
        rlo,rhi = 216.5, 219.5

    elif opt.region == 'des-sn-x3':
        #rlo,rhi = 36., 37.
        #dlo,dhi = -5., -4.
        rlo,rhi = 36., 36.5
        dlo,dhi = -4.5, -4.

    elif opt.region == 'ngc2632':
        # open cluster
        rlo,rhi = 129.0, 131.0
        dlo,dhi = 19.0, 20.5
        
    if opt.mindec is not None:
        dlo = opt.mindec
    if opt.maxdec is not None:
        dhi = opt.maxdec
    if opt.minra is not None:
        rlo = opt.minra
    if opt.maxra is not None:
        rhi = opt.maxra

    if rlo < rhi:
        B.cut((B.ra >= rlo) * (B.ra <= rhi) *
              (B.dec >= dlo) * (B.dec <= dhi))
    else: # RA wrap
        B.cut(np.logical_or(B.ra >= rlo, B.ra <= rhi) *
              (B.dec >= dlo) * (B.dec <= dhi))
    log(len(B), 'bricks in range')
    #for name in B.get('brickname'):
    #    print(name)
    #B.writeto('bricks-cut.fits')

    bricksize = 0.25
    # A bit more than 0.25-degree brick radius + Bok image radius ~ 0.57
    search_radius = 1.05 * np.sqrt(2.) * (bricksize +
                                          (0.455 * 4096 / 3600.))/2.

    log(len(T), 'CCDs')
    log(len(B), 'Bricks')
    I,J,d = match_radec(B.ra, B.dec, T.ra, T.dec, search_radius,
                        nearest=True)
    B.cut(I)
    log('Cut to', len(B), 'bricks near CCDs')

    # plt.clf()
    # plt.plot(B.ra, B.dec, 'b.')
    # plt.title('DR3 bricks')
    # plt.axis([360, 0, np.min(B.dec)-1, np.max(B.dec)+1])
    # plt.savefig('bricks.png')

    if opt.touching:
        I,J,d = match_radec(T.ra, T.dec, B.ra, B.dec, search_radius,
                            nearest=True)
        # list the ones that will be cut
        # drop = np.ones(len(T))
        # drop[I] = False
        # for i in np.flatnonzero(drop):
        #     from astrometry.util.starutil_numpy import degrees_between
        #     dists = degrees_between(B.ra, B.dec, T.ra[i], T.dec[i])
        #     mindist = min(dists)
        #     print('Dropping:', T.ra[i], T.dec[i], 'min dist', mindist, 'search_radius', search_radius)


        T.cut(I)
        log('Cut to', len(T), 'CCDs near bricks')

    # sort by RA increasing
    B.cut(np.argsort(B.ra))

    if opt.save_to_fits:
        assert(opt.touching)
        # Write cut tables to file
        for tab,typ in zip([B,T],['bricks','ccds']):
            fn='%s-%s-cut.fits' % (typ,opt.region)
            if os.path.exists(fn):
                os.remove(fn)
            tab.writeto(fn)
            log('Wrote %s' % fn)
        # Write text files listing ccd and filename names
        # nm1,nm2= 'ccds-%s.txt'% opt.region,'filenames-%s.txt' % opt.region
        # if os.path.exists(nm1):
        #     os.remove(nm1)
        # if os.path.exists(nm2):
        #     os.remove(nm2)
        # f1,f2=open(nm1,'w'),open(nm2,'w')
        # fns= list(set(T.get('image_filename')))
        # for fn in fns:
        #     f2.write('%s\n' % fn.strip())
        # for ti in T:
        #     f1.write('%s\n' % ti.get('image_filename').strip())
        # f1.close()
        # f2.close()
        # log('Wrote *-names.txt')
        
    if opt.touching:

        if want_bricks:
            # Shortcut the list of bricks that are definitely touching CCDs --
            # a brick-ccd pair within this radius must be touching.
            closest_radius = 0.95 * (bricksize + 0.262 * 2048 / 3600.) / 2.

            J1,nil,nil = match_radec(B.ra, B.dec, T.ra, T.dec, closest_radius, nearest=True)
            log(len(J1), 'of', len(B), 'bricks definitely touch CCDs')
            tocheck = np.ones(len(B), bool)
            tocheck[J1] = False
            J2 = []
            for j in np.flatnonzero(tocheck):
                b = B[j]
                wcs = wcs_for_brick(b)
                I = ccds_touching_wcs(wcs, T)
                log(len(I), 'CCDs for brick', b.brickname)
                if len(I) == 0:
                    continue
                J2.append(j)
            J = np.hstack((J1, J2))
            J = np.sort(J).astype(int)
            B.cut(J)
            log('Cut to', len(B), 'bricks touching CCDs')

        else:
            J = []
            allI = set()
            for j,b in enumerate(B):
                wcs = wcs_for_brick(b)
                I = ccds_touching_wcs(wcs, T)
                log(len(I), 'CCDs for brick', b.brickname)
                if len(I) == 0:
                    continue
                allI.update(I)
                J.append(j)
            allI = list(allI)
            allI.sort()
            B.cut(np.array(J))
            log('Cut to', len(B), 'bricks touching CCDs')

    elif opt.near:
        # Find CCDs near bricks
        allI,nil,nil = match_radec(T.ra, T.dec, B.ra, B.dec, search_radius, nearest=True)
        # Find bricks near CCDs
        J,nil,nil = match_radec(B.ra, B.dec, T.ra, T.dec, search_radius, nearest=True)
        B.cut(J)
        log('Cut to', len(B), 'bricks near CCDs')

    else:
        allI = np.arange(len(T))

    if opt.nccds:
        from queue import Queue
        from threading import Thread
        from time import sleep

        log('Checking number of CCDs per brick')

        def worker():
            while True:
                i = q.get()
                if i is None:
                    break
                b = B[i]
                wcs = wcs_for_brick(b)
                I = ccds_touching_wcs(wcs, T)
                log(b.brickname, len(I))
                q.task_done()

        q = Queue()
        num_threads = 24
        threads = []

        for i in range(num_threads):
            t = Thread(target=worker)
            t.start()
            threads.append(t)

        for i in range(len(B)):
            q.put(i)

        q.join()
        for i in range(num_threads):
            q.put(None)
        for t in threads:
            t.join()

    if opt.write_ccds:
        T[allI].writeto(opt.write_ccds)
        log('Wrote', opt.write_ccds)

    if want_bricks:
        # Print the list of bricks and exit.
        for b in B:
            print(b.brickname)
        if opt.save_to_fits:
            B.writeto('bricks-%s-touching.fits' % opt.region)
        if not want_ccds:
            sys.exit(0)

    ## Be careful here -- T has been cut; we want to write out T.index.
    ## 'allI' contains indices into T.

    if opt.stage is not None:
        cmd_pat = 'rsync -LRarv %s %s'
        fns = set()
        for iccd in allI:
            im = survey.get_image_object(T[iccd])
            fns.update([im.imgfn, im.wtfn, im.dqfn, im.psffn, im.merged_psffn,
                   im.merged_splineskyfn, im.splineskyfn])
        for i,fn in enumerate(fns):
            print('File', i+1, 'of', len(fns), ':', fn)
            if not os.path.exists(fn):
                print('No such file:', fn)
                continue
            base = survey.get_survey_dir()
            if base.endswith('/'):
                base = base[:-1]

            rel = os.path.relpath(fn, base)

            dest = os.path.join(opt.stage, rel)
            print('Dest:', dest)
            if os.path.exists(dest):
                print('Exists:', dest)
                continue

            cmd = cmd_pat % ('%s/./%s' % (base, rel), opt.stage)
            print(cmd)
            rtn = os.system(cmd)
            assert(rtn == 0)
        sys.exit(0)

    if opt.forced:
        log('Writing forced-photometry commands to', opt.out)
        f = open(opt.out,'w')
        log('Total of', len(allI), 'CCDs')
        for j,i in enumerate(allI):
            expstr = '%08i' % T.expnum[i]
            imgfn = os.path.join(survey.survey_dir, 'images',
                                 T.image_filename[i].strip())
            if (not os.path.exists(imgfn) and
                imgfn.endswith('.fz') and
                os.path.exists(imgfn[:-3])):
                imgfn = imgfn[:-3]

            outfn = os.path.join(expstr[:5], expstr,
                                 'forced-%s-%s-%s.fits' %
                                 (T.camera[i].strip(), expstr, T.ccdname[i]))

            f.write('python legacypipe/forced_photom.py --apphot --derivs --catalog-dir /project/projectdirs/cosmo/data/legacysurvey/dr7/ %i %s forced/%s\n' %
                    (T.expnum[i], T.ccdname[i], outfn))

        f.close()
        log('Wrote', opt.out)

        fn = 'forced-ccds.fits'
        T[allI].writeto(fn)
        print('Wrote', fn)

        sys.exit(0)

    if opt.lsb:
        log('Writing LSB commands to', opt.out)
        f = open(opt.out,'w')
        log('Total of', len(allI), 'CCDs')
        for j,i in enumerate(allI):
            exp = T.expnum[i]
            ext = T.ccdname[i].strip()
            outfn = 'lsb/lsb-%s-%s.fits' % (exp, ext)
            f.write('python legacyanalysis/lsb.py --expnum %i --extname %s --out %s -F -n > lsb/lsb-%s-%s.log 2>&1\n' % (exp, ext, outfn, exp, ext))
        f.close()
        log('Wrote', opt.out)
        sys.exit(0)

    log('Writing calibs to', opt.out)
    f = open(opt.out,'w')
    log('Total of', len(allI), 'CCDs')

    batch = []

    def write_batch(f, batch, cmd):
        if cmd is None:
            cmd = ''
        f.write(cmd + ' '.join(batch) + '\n')

    cmd = None
    if opt.command:
        cmd = 'python legacypipe/run-calib.py '
        if opt.opt is not None:
            cmd += opt.opt + ' '
        
    for j,i in enumerate(allI):

        if opt.delete_sky:
            log(j+1, 'of', len(allI))
            im = survey.get_image_object(T[i])
            if opt.delete_sky and os.path.exists(im.skyfn):
                log('  deleting:', im.skyfn)
                os.unlink(im.skyfn)

        if opt.command:
            s = '%i-%s' % (T.expnum[i], T.ccdname[i])
            prefix = 'python legacypipe/run-calib.py '
            if opt.opt is not None:
                prefix = prefix + opt.opt
            #('python legacypipe/run-calib.py --expnum %i --ccdname %s' %
            #     (T.expnum[i], T.ccdname[i]))
        else:
            s = '%i' % T.index[i]
            prefix = ''
            
        if j < 10:
            print('Index', T.index[i], 'expnum', T.expnum[i], 'ccdname', T.ccdname[i],
                  'filename', T.image_filename[i])
            
        if not opt.nper:
            f.write(prefix + s + '\n')
        else:
            batch.append(s)
            if len(batch) >= opt.nper:
                write_batch(f, batch, cmd)
                batch = []

    if len(batch):
        write_batch(f, batch, cmd)

    f.close()
    log('Wrote', opt.out)
    return 0
#
#
#
if __name__ == '__main__':
    sys.exit(main())
