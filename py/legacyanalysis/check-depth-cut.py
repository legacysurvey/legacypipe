from __future__ import print_function
import pylab as plt
from glob import glob
from astrometry.util.fits import *

import sys

### Update mistakes
if True:
    from astrometry.util.fits import *
    plots = False

    todo = open('redo.txt')
    todo = [t.strip() for t in todo.readlines()]
    print(len(todo), 'bricks to do:', todo)

    # for brickname in todo:
    #     # shortcut
    #     dirnm = os.path.join('depthcuts', brickname[:3])
    #     outfn = os.path.join(dirnm, 'ccds-%s.fits' % brickname)
    #     print('Brick', brickname, 'file', outfn)
    #     if os.path.exists(outfn):
    #         os.unlink(outfn)
    #         print('Deleted', outfn)
    # sys.exit(0)

    # (expnum,ccdname) pairs
    T = fits_table('depth-cut-kept-ccds.fits')
    ccds = set(zip(T.expnum, [c.strip() for c in T.ccdname]))
    print(len(ccds), 'unique expnum+ccdname pairs')

    T = fits_table('depth-cut-summary.fits')

    #survey = LegacySurveyData()
    Tnew = fits_table()
    Tnew.filename =  []
    Tnew.ra =        []
    Tnew.dec =       []
    Tnew.ntotal =    []
    Tnew.ntotal_g =  []
    Tnew.ntotal_r =  []
    Tnew.ntotal_z =  []
    Tnew.npassed =   []
    Tnew.npassed_g = []
    Tnew.npassed_r = []
    Tnew.npassed_z = []

    for brickname in todo:
        # shortcut
        dirnm = os.path.join('depthcuts', brickname[:3])
        outfn = os.path.join(dirnm, 'ccds-%s.fits' % brickname)
        print('Brick', brickname, 'file', outfn)
        # if os.path.exists(outfn):
        #     os.unlink(outfn)
        #     print('Deleted', outfn)

        base = brickname
        ra = int(base[:4], 10) * 0.1
        dec = int(base[5:], 10) * 0.1 * (-1 if base[4] == 'm' else 1)

        C = fits_table(outfn)
        print(np.sum(C.passed_depth_cut), 'of', len(C), 'CCDs passed depth cut')

        # brick = survey.get_brick_by_name(brickname)
        # print('Got brick, running depth cut')
        # rtn = run_one_brick((brick, 0, 1, plots))
        # assert(rtn == 0)

        I = np.flatnonzero(T.filename == outfn)
        if len(I) == 0:
            Tnew.filename.append(outfn)
            Tnew.ra.append(ra)
            Tnew.dec.append(dec)

            Tnew.ntotal.append(len(C))
            Tnew.ntotal_g.append(np.sum(C.filter == 'g'))
            Tnew.ntotal_r.append(np.sum(C.filter == 'r'))
            Tnew.ntotal_z.append(np.sum(C.filter == 'z'))
            C.cut(C.passed_depth_cut)
            Tnew.npassed.append(len(C))
            Tnew.npassed_g.append(np.sum(C.filter == 'g'))
            Tnew.npassed_r.append(np.sum(C.filter == 'r'))
            Tnew.npassed_z.append(np.sum(C.filter == 'z'))
        else:
            assert(len(I) == 1)
            it = I[0]

            # T.ra[it]  = ra
            # T.dec[it] = dec
            T.ntotal[it] = len(C)
            T.ntotal_g[it] = np.sum(C.filter == 'g')
            T.ntotal_r[it] = np.sum(C.filter == 'r')
            T.ntotal_z[it] = np.sum(C.filter == 'z')
            C.cut(C.passed_depth_cut)
            T.npassed[it] = len(C)
            T.npassed_g[it] = np.sum(C.filter == 'g')
            T.npassed_r[it] = np.sum(C.filter == 'r')
            T.npassed_z[it] = np.sum(C.filter == 'z')

        ccds.update(zip(C.expnum, [c.strip() for c in C.ccdname]))

    print(len(Tnew.filename))
    print(len(Tnew), 'new files,', len(T), 'old')
    Tnew.to_np_arrays()
    print(len(Tnew), 'new files,', len(T), 'old')
    T = merge_tables([T, Tnew])
    print('Total', len(T))

    T.writeto('depth-cut-summary-2.fits')

    T = fits_table()
    expnums,ccdnames = [],[]
    for (expnum,ccdname) in ccds:
        expnums.append(expnum)
        ccdnames.append(ccdname)
    T.expnum = np.array(expnums)
    T.ccdname = np.array(ccdnames)
    T.writeto('depth-cut-kept-ccds-2.fits')

    sys.exit(0)



fns = glob('depthcuts/*/ccds-*.fits')
fns.sort()
print('Found', len(fns), 'files')

# (expnum,ccdname) pairs
ccds = set()

ras = []
decs = []
ntotal = []
npassed = []

ntotal_g = []
ntotal_r = []
ntotal_z = []
npassed_g = []
npassed_r = []
npassed_z = []

thefns = []

redo = []

for ifn,fn in enumerate(fns):
    print()
    print(ifn+1, 'of', len(fns), ':', fn)
    try:
        T = fits_table(fn)
    except:
        import traceback
        traceback.print_exc()
        print('REDO:', fn)
        continue

    # no duplicate expnum,ccdnames, I hope!
    #assert(len(set(zip(T.expnum, T.ccdname))) == len(T))
    if len(set(zip(T.expnum, T.ccdname))) != len(T):
        print('REDO:', fn)
        redo.append(fn)
        continue

    thefns.append(fn)
    ntotal.append(len(T))
    ntotal_g.append(np.sum(T.filter == 'g'))
    ntotal_r.append(np.sum(T.filter == 'r'))
    ntotal_z.append(np.sum(T.filter == 'z'))

    print(np.sum(T.passed_depth_cut), 'of', len(T), 'CCDs passed depth cut')
    T.cut(T.passed_depth_cut)

    npassed.append(len(T))
    npassed_g.append(np.sum(T.filter == 'g'))
    npassed_r.append(np.sum(T.filter == 'r'))
    npassed_z.append(np.sum(T.filter == 'z'))

    ccds.update(zip(T.expnum, T.ccdname))
    print('Now total of', len(ccds), 'expnum/ccdname pairs')

    base = os.path.basename(fn)
    base = base.replace('ccds-', '').replace('.fits','')
    ra  = int(base[:4], 10) * 0.1
    dec = int(base[5:], 10) * 0.1 * (-1 if base[4] == 'm' else 1)
    ras.append(ra)
    decs.append(dec)


    # if (ifn+1) % 100 == 0:
    #     plt.clf()
    #     plt.scatter(ras, decs, c=np.array(ntotal).astype(float), vmin=0, vmax=np.max(ntotal), cmap='jet')
    #     plt.title('Total CCDs')
    #     plt.savefig('depthcut-total.png')
    # 
    #     plt.clf()
    #     plt.scatter(ras, decs, c=np.array(npassed).astype(float), vmin=0, vmax=np.max(npassed), cmap='jet')
    #     plt.title('Kept CCDs')
    #     plt.savefig('depthcut-passed.png')

    #if ifn > 200:
    #    break



T = fits_table()
T.filename = np.array(thefns)
T.ra = np.array(ras)
T.dec = np.array(decs)
T.ntotal = np.array(ntotal)
T.ntotal_g = np.array(ntotal_g)
T.ntotal_r = np.array(ntotal_r)
T.ntotal_z = np.array(ntotal_z)
T.npassed = np.array(npassed)
T.npassed_g = np.array(npassed_g)
T.npassed_r = np.array(npassed_r)
T.npassed_z = np.array(npassed_z)
T.writeto('depth-cut-summary.fits')

T = fits_table()
expnums,ccdnames = [],[]
for (expnum,ccdname) in ccds:
    expnums.append(expnum)
    ccdnames.append(ccdname)
T.expnum = np.array(expnums)
T.ccdname = np.array(ccdnames)
T.writeto('depth-cut-kept-ccds.fits')

print('To redo:')
for t in redo:
    print(t)
