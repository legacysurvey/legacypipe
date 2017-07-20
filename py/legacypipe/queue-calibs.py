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

    parser.add_argument('--touching', action='store_true',
                      help='Cut to only CCDs touching selected bricks')
    parser.add_argument('--near', action='store_true',
                      help='Quick cut to only CCDs near selected bricks')

    parser.add_argument('--check', action='store_true',
                      help='Check which calibrations actually need to run.')
    parser.add_argument('--check-coadd', action='store_true',
                      help='Check which caoadds actually need to run.')
    parser.add_argument('--out', help='Output filename for calibs, default %(default)s',
                      default='jobs')
    parser.add_argument('--command', action='store_true',
                      help='Write out full command-line to run calib')
    parser.add_argument('--opt', help='With --command, extra options to add')
    
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
    parser.add_argument('--delete-pvastrom', action='store_true',
                      help='Delete any existing PV WCS calibration files')

    parser.add_argument('--write-ccds', help='Write CCDs list as FITS table?')

    parser.add_argument('--brickq', type=int, default=None,
                        help='Queue only bricks with the given "brickq" value [0 to 3]')

    parser.add_argument('--brickq-deps', action='store_true', default=False,
                        help='Queue bricks directly using qdo API, setting brickq dependencies')
    parser.add_argument('--queue', default='bricks',
                        help='With --brickq-deps, the QDO queue name to use')
    
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

    if opt.ignore_cuts == False:
        I = survey.photometric_ccds(T)
        print(len(I), 'CCDs are photometric')
        T.cut(I)
        cuts = survey.ccd_cuts(T)
        print(len(cuts != 0), 'CCDs are subject to cuts')
        T.cut(cuts == 0)
    print(len(T), 'CCDs remain')

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

    elif opt.region == 'deep2f3':
        rlo,rhi = 351.25, 353.75
        dlo,dhi = 0, 0.5

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
        rlo,rhi = 122., 177.
        dlo,dhi =  12.,  32.

    elif opt.region == 'mzls':
        dlo,dhi = 30., 90.
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
    for name in B.get('brickname'):
        print(name)
    #B.writeto('bricks-cut.fits')

    print(len(T), 'CCDs')
    print(len(B), 'Bricks')
    I,J,d = match_radec(B.ra, B.dec, T.ra, T.dec, survey.bricksize)
    keep = np.zeros(len(B), bool)
    for i in I:
        keep[i] = True
    B.cut(keep)
    log('Cut to', len(B), 'bricks near CCDs')

    # plt.clf()
    # plt.plot(B.ra, B.dec, 'b.')
    # plt.title('DR3 bricks')
    # plt.axis([360, 0, np.min(B.dec)-1, np.max(B.dec)+1])
    # plt.savefig('bricks.png')

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
    #B.cut(np.argsort(-B.dec))
    # RA increasing
    B.cut(np.argsort(B.ra))

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

        #print(b.brickname)

    if opt.save_to_fits:
        assert(opt.touching)
        # Write cut tables to file
        for tab,typ in zip([B,T],['bricks','ccds']):
            fn='%s-%s-cut.fits' % (typ,opt.region)
            if os.path.exists(fn):
                os.remove(fn)
            tab.writeto(fn)
            print('Wrote %s' % fn)
        # Write text files listing ccd and filename names
        nm1,nm2= 'ccds-%s.txt'% opt.region,'filenames-%s.txt' % opt.region
        if os.path.exists(nm1):
            os.remove(nm1)
        if os.path.exists(nm2):
            os.remove(nm2)
        f1,f2=open(nm1,'w'),open(nm2,'w')
        fns= list(set(T.get('image_filename')))
        for fn in fns:
            f2.write('%s\n' % fn.strip())
        for ti in T:
            f1.write('%s\n' % ti.get('image_filename').strip())
        f1.close()
        f2.close()
        print('Wrote *-names.txt')
    

    if opt.brickq_deps:
        import qdo
        from legacypipe.survey import on_bricks_dependencies

        #... find Queue...
        q = qdo.connect(opt.queue, create_ok=True)
        print('Connected to QDO queue', opt.queue, q)
        brick_to_task = dict()

        I = survey.photometric_ccds(T)
        print(len(I), 'CCDs are photometric')
        T.cut(I)
        print(len(T), 'CCDs remaining')

        T.wra = T.ra + (T.ra > 180) * -360
        wra = rlo - 360
        plt.clf()
        plt.plot(T.wra, T.dec, 'b.')
        ax = [wra, rhi, dlo, dhi]
        plt.axis(ax)
        plt.title('CCDs')
        plt.savefig('q-ccds.png')

        B.wra = B.ra + (B.ra > 180) * -360

        # this slight overestimate (for DECam images) is fine
        radius = 0.3
        Iccds = match_radec(B.ra, B.dec, T.ra, T.dec, radius,
                            indexlist=True)
        ikeep = []
        for ib,(b,Iccd) in enumerate(zip(B, Iccds)):
            if Iccd is None or len(Iccd) == 0:
                print('No matched CCDs to brick', b.brickname)
                continue
            wcs = wcs_for_brick(b)
            cI = ccds_touching_wcs(wcs, T[np.array(Iccd)])
            print(len(cI), 'CCDs touching brick', b.brickname)
            if len(cI) == 0:
                continue
            ikeep.append(ib)
        B.cut(np.array(ikeep))
        print('Cut to', len(B), 'bricks touched by CCDs')
        
        for brickq in range(4):
            I = np.flatnonzero(B.brickq == brickq)
            print(len(I), 'bricks with brickq =', brickq)

            J = np.flatnonzero(B.brickq < brickq)
            preB = B[J]
            reqs = []
            if brickq > 0:
                for b in B[I]:
                    # find brick dependencies
                    brickdeps = on_bricks_dependencies(b, survey, bricks=preB)
                    # convert to task ids
                    taskdeps = [brick_to_task.get(b.brickname,None) for b in brickdeps]
                    # If we dropped a dependency brick from a previous brickq because
                    # of no overlapping CCDs, it won't appear in the brick_to_task map.
                    taskdeps = [t for t in taskdeps if t is not None]
                    reqs.append(taskdeps)

            plt.clf()
            plt.plot(B.wra, B.dec, '.', color='0.5')
            plt.plot(B.wra[I], B.dec[I], 'b.')
            plt.axis(ax)
            plt.title('Bricks: brickq=%i' % brickq)
            plt.savefig('q-bricks-%i.png' % brickq)
            
            # submit to qdo queue
            print('Queuing', len(B[I]), 'bricks')
            if brickq == 0:
                reqs = None
            else:
                assert(len(I) == len(reqs))
            taskids = q.add_multiple(B.brickname[I], requires=reqs)
            assert(len(taskids) == len(I))
            print('Queued', len(taskids), 'bricks')
            brick_to_task.update(dict(zip(B.brickname[I], taskids)))
        
    if not (opt.calibs or opt.forced or opt.lsb):

        for b in B:
            print(b.brickname)

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
    elif opt.near:
        # Roughly brick radius + DECam image radius
        radius = 0.35
        allI,nil,nil = match_radec(T.ra, T.dec, B.ra, B.dec, radius, nearest=True)
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
            imgfn = os.path.join(survey.survey_dir, 'images',
                                 T.image_filename[i].strip())
            if (not os.path.exists(imgfn) and
                imgfn.endswith('.fz') and
                os.path.exists(imgfn[:-3])):
                imgfn = imgfn[:-3]

            #f.write('python legacypipe/forced_photom_decam.py %s %i DR3 %s\n' %
            #        (imgfn, T.image_hdu[i], outfn))

            #outfn = os.path.join('forced', expstr[:5], expstr,
            outfn = os.path.join(expstr[:5], expstr,
                                 'forced-%s-%s-%s.fits' %
                                 (T.camera[i].strip(), expstr, T.ccdname[i]))

            f.write('python legacypipe/forced_photom_decam.py --hybrid-psf --apphot --constant-invvar %i %s DR3 forced/vanilla/%s\n' %
                    (T.expnum[i], T.ccdname[i], outfn))
            f.write('python legacypipe/forced_photom_decam.py --hybrid-psf --derivs --constant-invvar %i %s DR3 forced/derivs/%s\n' %
                    (T.expnum[i], T.ccdname[i], outfn))
            f.write('python legacypipe/forced_photom_decam.py --hybrid-psf --agn --constant-invvar %i %s DR3 forced/agn/%s\n' %
                    (T.expnum[i], T.ccdname[i], outfn))
            
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

        if opt.check:
            f.flush()

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
