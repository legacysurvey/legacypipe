from __future__ import print_function

import os
import numpy as np
import fitsio
from astrometry.libkd.spherematch import match_radec
from astrometry.util.multiproc import multiproc
from legacypipe.survey import LegacySurveyData, get_git_version, wcs_for_brick
from legacypipe.runbrick import make_depth_cut

def queue():
    if False:
        survey = LegacySurveyData()
        ccds = survey.get_ccds()
        bricks = survey.get_bricks()
        print(len(bricks), 'bricks')
        print(len(ccds), 'CCDs')
    
        bricks.cut((bricks.dec >= -30) * (bricks.dec <= 30))
        print(len(bricks), 'in Dec [-30, +30]')
    
        I = survey.photometric_ccds(ccds)
        ccds.cut(I)
        print(len(ccds), 'pass photometric cut')
    
        I,J,d = match_radec(bricks.ra, bricks.dec, ccds.ra, ccds.dec, 0.5, nearest=True)
        print(len(I), 'bricks with CCDs nearby')
        bricks.cut(I)
        bricknames = bricks.brickname

    else:
        # DR7: use Martin's list of bricks w/ CCD coverage
        f = open('nccds.dat')
        bricknames = []
        for line in f.readlines():
            words = line.strip().split(' ')
            brick = words[0]
            nccd = int(words[1])
            if nccd > 0:
                bricknames.append(brick)

    # qdo
    bb = bricknames
    while len(bb):
        print(' '.join(bb[:100]))
        bb = bb[100:]
    return

    mp = multiproc(16)
    N = len(bricks)
    args = [(brick, i, N, plots, {}) for i,brick in enumerate(bricks)]
    mp.map(run_one_brick, args)


def run_one_brick(X):
    brick, ibrick, nbricks, plots, kwargs = X

    survey = LegacySurveyData()

    print()
    print()
    print('Brick', (ibrick+1), 'of', nbricks, ':', brick.brickname)

    dirnm = os.path.join('depthcuts', brick.brickname[:3])
    outfn = os.path.join(dirnm, 'ccds-%s.fits' % brick.brickname)
    if os.path.exists(outfn):
        print('Exists:', outfn)
        return 0

    H,W = 3600,3600
    pixscale = 0.262
    bands = ['g','r','z']

    # Get WCS object describing brick
    targetwcs = wcs_for_brick(brick, W=W, H=H, pixscale=pixscale)
    targetrd = np.array([targetwcs.pixelxy2radec(x,y) for x,y in
                         [(1,1),(W,1),(W,H),(1,H),(1,1)]])
    gitver = get_git_version()

    ccds = survey.ccds_touching_wcs(targetwcs)
    if ccds is None:
        print('No CCDs actually touching brick')
        return 0
    print(len(ccds), 'CCDs actually touching brick')

    ccds.cut(np.in1d(ccds.filter, bands))
    print('Cut on filter:', len(ccds), 'CCDs remain.')

    if 'ccd_cuts' in ccds.get_columns():
        norig = len(ccds)
        ccds.cut(ccds.ccd_cuts == 0)
        print(len(ccds), 'of', norig, 'CCDs pass cuts')
    else:
        print('No CCD cuts')

    if len(ccds) == 0:
        print('No CCDs left')
        return 0

    ps = None
    if plots:
        from astrometry.util.plotutils import PlotSequence
        ps = PlotSequence('depth-%s' % brick.brickname)

    splinesky = True
    gaussPsf = False
    pixPsf = True
    do_calibs = False
    normalizePsf = True

    get_depth_maps = kwargs.pop('get_depth_maps', False)

    try:
        D = make_depth_cut(
            survey, ccds, bands, targetrd, brick, W, H, pixscale,
            plots, ps, splinesky, gaussPsf, pixPsf, normalizePsf, do_calibs,
            gitver, targetwcs, get_depth_maps=get_depth_maps, **kwargs)
        if get_depth_maps:
            keep,overlapping,depthmaps = D
        else:
            keep,overlapping = D
    except:
        print('Failed to make_depth_cut():')
        import traceback
        traceback.print_exc()
        return -1

    print(np.sum(overlapping), 'CCDs overlap the brick')
    print(np.sum(keep), 'CCDs passed depth cut')
    ccds.overlapping = overlapping
    ccds.passed_depth_cut = keep

    if not os.path.exists(dirnm):
        try:
            os.makedirs(dirnm)
        except:
            pass

    if get_depth_maps:
        for band,depthmap in depthmaps:
            doutfn = os.path.join(dirnm, 'depth-%s-%s.fits' % (brick.brickname, band))
            hdr = fitsio.FITSHDR()
            # Plug the WCS header cards into these images
            targetwcs.add_to_header(hdr)
            hdr.delete('IMAGEW')
            hdr.delete('IMAGEH')
            hdr.add_record(dict(name='EQUINOX', value=2000.))
            hdr.add_record(dict(name='FILTER', value=band))
            fitsio.write(doutfn, depthmap, header=hdr)
            print('Wrote', doutfn)

    tmpfn = os.path.join(os.path.dirname(outfn), 'tmp-' + os.path.basename(outfn))
    ccds.writeto(tmpfn)
    os.rename(tmpfn, outfn)
    print('Wrote', outfn)

    return 0

if __name__ == '__main__':
    print('Starting')
    import sys

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', type=float, default=None,
                        help='Set margin, in mags, above the DESI depth requirements.')
    parser.add_argument('--depth-maps', action='store_true', default=False,
                        help='Write sub-scale depth map images?')
    parser.add_argument('--plots', action='store_true', default=False)
    parser.add_argument('--threads', type=int, help='"qdo" mode: number of threads')
    parser.add_argument('--queue', help='"qdo" mode: queue to read', default='depth')
    parser.add_argument('bricks', nargs='*')
    args = parser.parse_args()
    plots = args.plots
    bricks = args.bricks

    kwargs = dict(get_depth_maps=args.depth_maps)
    if args.margin is not None:
        kwargs.update(margin=args.margin)

    print('args:', bricks)

    if len(bricks) == 1 and bricks[0] == 'qdo':
        import qdo
        #... find Queue...
        qname = args.queue
        q = qdo.connect(qname)
        print('Connected to QDO queue', qname, q)

        survey = LegacySurveyData()

        while True:
            task = q.get(timeout=10)
            if task is None:
                break
            try:
                print('Task:', task.task)

                brickname = task.task
                print('Checking for existing out file')
                # shortcut
                dirnm = os.path.join('depthcuts', brickname[:3])
                outfn = os.path.join(dirnm, 'ccds-%s.fits' % brickname)
                if os.path.exists(outfn):
                    print('Exists:', outfn)
                    task.set_state(qdo.Task.SUCCEEDED)
                    continue
                print('Getting brick', brickname)
                brick = survey.get_brick_by_name(brickname)
                print('Got brick, running depth cut')
                rtn = run_one_brick((brick, 0, 1, False, kwargs))
                if rtn != 0:
                    allgood = rtn
                print('Done, result', rtn)
                if rtn == 0:
                    task.set_state(qdo.Task.SUCCEEDED)
                else:
                    task.set_state(qdo.Task.FAILED, err=1)
            except:
                import traceback
                traceback.print_exc()
                task.set_state(qdo.Task.FAILED, err=1)
        sys.exit(0)


    if len(bricks):
        allgood = 0
        print('Creating survey object')
        bargs = []
        survey = LegacySurveyData()
        for brickname in bricks:

            print('Checking for existing out file')
            # shortcut
            dirnm = os.path.join('depthcuts', brickname[:3])
            outfn = os.path.join(dirnm, 'ccds-%s.fits' % brickname)
            if os.path.exists(outfn):
                print('Exists:', outfn)
                continue
            print('Getting brick', brickname)
            brick = survey.get_brick_by_name(brickname)
            bargs.append((brick, 0, 1, plots, kwargs))

        if args.threads is not None:
            mp = multiproc(args.threads)
            rtns = mp.map(run_one_brick, bargs)
            for rtn in rtns:
                if rtn != 0:
                    allgood = rtn
        else:
            for arg in bargs:
                rtn = run_one_brick(arg)
                if rtn != 0:
                    allgood = rtn
                print('Done, result', rtn)

        sys.exit(allgood)

    else:
        queue()
