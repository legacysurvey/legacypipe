#from astropy.table import Table
from astrometry.util.fits import fits_table
from astrometry.libkd.spherematch import match_radec
from astrometry.util.starutil_numpy import arcsec_between
from collections import Counter
import numpy as np
import fitsio
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dirA')
    parser.add_argument('dirB')
    parser.add_argument('brick')
    parser.add_argument('--plots', default=False, action='store_true')
    opt = parser.parse_args()

    from legacypipe.survey import LegacySurveyData

    surveyA = LegacySurveyData(opt.dirA)
    surveyB = LegacySurveyData(opt.dirB)
    brick = opt.brick

    fnA = surveyA.find_file('tractor', brick=brick)
    fnB = surveyB.find_file('tractor', brick=brick)

    print('Looking for tractor files:')
    print('  ', fnA)
    print('  ', fnB)

    trA = fits_table(fnA)
    trB = fits_table(fnB)

    blobA = None
    blobB = None
    try:
        blobfn = surveyA.find_file('blobmap', brick=brick)
        blobA = fitsio.read(blobfn)
    except:
        import traceback
        print('Failed to read blob map for catalog A:')
        traceback.print_exc()
    try:
        blobfn = surveyB.find_file('blobmap', brick=brick)
        blobB = fitsio.read(blobfn)
    except:
        import traceback
        print('Failed to read blob map for catalog B:')
        traceback.print_exc()
    
    print('Number of objects: %i vs %i' % (len(trA), len(trB)))

    # arcsec matching radius
    radius = 0.1
    I,J,D = match_radec(trA.ra, trA.dec, trB.ra, trB.dec, radius/3600.)
    print('Matched %i pairs with %i and %i unique objects' % (len(I), len(np.unique(I)), len(np.unique(J))))
    print('Matched within %.3f arcsec: median distance %.3f arcsec' % (radius, np.median(D * 3600.)))

    # Cut to only mutually closest matches
    # index in trB of the closest match to each trA.
    Iargnearest = np.empty(len(trA), int)
    Inearest = np.empty(len(trA), np.float32)
    Iargnearest[:] = -1
    Inearest[:] = 1e6
    Jargnearest = np.empty(len(trB), int)
    Jnearest = np.empty(len(trB), np.float32)
    Jargnearest[:] = -1
    Jnearest[:] = 1e6
    for i,j,d in zip(I, J, D):
        if d < Inearest[i]:
            Inearest[i] = d
            Iargnearest[i] = j
        if d < Jnearest[j]:
            Jnearest[j] = d
            Jargnearest[j] = i
    #J = Iargnearest[Iargnearest >= 0]
    #I = Jargnearest[Jargnearest >= 0]
    #print('Cut to', len(I), 'mutually closest pairs')
    I,J = [],[]
    for i,j in enumerate(Iargnearest):
        if j == -1:
            continue
        if Jargnearest[j] != i:
            continue
        I.append(i)
        J.append(j)
    I = np.array(I)
    J = np.array(J)
    print('Cut to', len(I), 'mutually closest pairs')
    # Cut to matched, row-aligned sources
    mA = trA[I]
    mB = trB[J]

    d = np.array([arcsec_between(r1,d1,r2,d2) for r1,d1,r2,d2 in
                  zip(mA.ra, mA.dec, mB.ra, mB.dec)])
    print('Mean distance: %.3f arcsec, median %.3f arcsec' % (np.mean(d), np.median(d)))
    
    def obj_string(trobj, blob):
        blobstr = ''
        if blob is not None:
            blobstr = ', Blob %4i' % get_blob(blob, trobj.bx, trobj.by)
        return ('X,Y (%6.1f, %6.1f), RA,Dec (%8.4f, %+8.4f), Objid %5i%s, Brick-primary? %s' %
                (trobj.bx, trobj.by, trobj.ra, trobj.dec, trobj.objid, blobstr, trobj.brick_primary))

    def get_blob(blob, bx, by):
        H,W = blob.shape
        ix = int(np.clip(int(bx), 0, W-1))
        iy = int(np.clip(int(by), 0, H-1))
        return blob[iy,ix]

    # Find & report the unmatched ones
    def report_unmatched(tr, survey, blob, I, letter):
        U = np.ones(len(tr))
        U[I] = False
        U = np.flatnonzero(U)
        print('In catalog %s, unmatched objects:' % letter)
        for u in U:
            tru = tr[u]
            print(obj_string(tru, blob))
        return U

    unmatchedA = None
    unmatchedB = None
    if len(I) < len(trA):
        U = report_unmatched(trA, surveyA, blobA, I, 'A')
        unmatchedA = trA[U]
    if len(J) < len(trB):
        U = report_unmatched(trB, surveyB, blobB, J, 'B')
        unmatchedB = trB[U]

    K = np.flatnonzero(mA.type == mB.type)
    if len(K) < len(mA):
        U = np.ones(len(mA))
        U[K] = False
        U = np.flatnonzero(U)
        print('Different types:')
        for u in U:
            ta = mA[u]
            tb = mB[u]
            print(obj_string(ta, blobA), 'Types %s, %s' % (ta.type, tb.type))

    worst = []
    bands = ['g','r','z']
    for band in bands:
        fA = mA.get('flux_%s' % band)
        fB = mB.get('flux_%s' % band)
        ivA = mA.get('flux_ivar_%s' % band)
        ivB = mB.get('flux_ivar_%s' % band)
        absdiff = np.abs(fA - fB)
        mad = np.median(absdiff)
        maxdiff = np.max(absdiff)
        good = (ivA > 0) * (ivB > 0)
        sigdiff = np.abs((fA - fB) * np.sqrt(ivA + ivB))
        madsig = np.median(sigdiff)
        maxdiffsig = np.max(sigdiff)
        ivdiff = np.median(np.abs(ivA - ivB)[good] / (ivA + ivB)[good])
        ivmaxdiff = np.max(np.abs(ivA - ivB)[good] / (ivA + ivB)[good])
        print('  %s band: abs flux difference: median %.2f, max %.1f nanomaggies' %
              (band, mad, maxdiff))
        print('          abs flux difference: median %.2f, max %.2f sigma' %
              (madsig, maxdiffsig))
        print('          number of differences above 1 sigma: %i' % np.sum(sigdiff > 1.))
        I = np.argsort(-sigdiff)
        for i in I[:10]:
            if sigdiff[i] < 1:
                break
            print('               ', obj_string(trA[i], blobA), 'fluxes %8.3f, %8.3f, sigma %5.3f, nsigma %5.2f' % (fA[i], fB[i], 1./np.sqrt(ivA[i]), np.abs(fA[i]-fB[i])*np.sqrt(ivA[i])))

        print('          inverse-variances fraction diffs: median %.1f %%, max %.2f %%' %
              (ivdiff, ivmaxdiff))
        print('          fraction of good inverse-variances: %.1f %%' % (100.*np.sum(good)/len(ivA)))
        I = I[sigdiff[I] > 1][:10]
        if len(I):
            worst.append(('flux %s' % band, sigdiff[I], I))

    print()
    for typ in ['REX', 'EXP', 'DEV', 'SER']:
        Iboth = np.flatnonzero((mA.type == typ) * (mB.type == typ))
        print(len(Iboth), 'are type', typ, 'in both catalogs.')
        if len(Iboth) == 0:
            continue
        rA = mA.shape_r[Iboth]
        rB = mB.shape_r[Iboth]
        ivA = mA.shape_r_ivar[Iboth]
        ivB = mB.shape_r_ivar[Iboth]
        absdiff = np.abs(rA - rB)
        mad = np.median(absdiff)
        maxdiff = np.max(absdiff)
        good = (ivA > 0) * (ivB > 0)
        sigdiff = np.abs((rA - rB) * np.sqrt(ivA + ivB))
        madsig = np.median(sigdiff)
        maxdiffsig = np.max(sigdiff)
        ivdiff = np.median(np.abs(ivA - ivB)[good] / (ivA + ivB)[good])
        ivmaxdiff = np.max(np.abs(ivA - ivB)[good] / (ivA + ivB)[good])

        print('  radius: abs difference: median %.2f, max %.1f arcsec' % (mad, maxdiff))
        print('          abs difference: median %.2f, max %.2f sigma' % (madsig, maxdiffsig))
        print('          number of differences above 1 sigma: %i' % np.sum(sigdiff > 1.))
        I = np.argsort(-sigdiff)
        print('          largest sigma differences:')
        for i in I[:10]:
            print('               ', obj_string(trA[i], blobA), 'radii %8.3f, %8.3f, sigma %5.3f, nsigma %5.2f' % (rA[i], rB[i], 1./np.sqrt(ivA[i]), np.abs(rA[i]-rB[i])*np.sqrt(ivA[i])))

        print('          inverse-variances fraction diffs: median %.1f %%, max %.2f %%' % (ivdiff, ivmaxdiff))
        print('          fraction of good inverse-variances: %.1f %%' % (100.*np.sum(good)/len(ivA)))
        I = I[sigdiff[I] > 1][:10]
        if len(I):
            worst.append(('radius: %s' % typ, sigdiff[I], I))

    if opt.plots:
        import pylab as plt
        from astrometry.util.plotutils import PlotSequence

        ps = PlotSequence('compare')
        jpgA = plt.imread(surveyA.find_file('image-jpeg', brick=brick))
        jpgA = np.flipud(jpgA)
        jpgB = plt.imread(surveyB.find_file('image-jpeg', brick=brick))
        jpgB = np.flipud(jpgB)
        modA = plt.imread(surveyA.find_file('model-jpeg', brick=brick))
        modA = np.flipud(modA)
        modB = plt.imread(surveyB.find_file('model-jpeg', brick=brick))
        modB= np.flipud(modB)

        plt.clf()
        plt.imshow(jpgA, interpolation='nearest', origin='lower')
        ps.savefig()

        plt.clf()
        plt.imshow(jpgA, interpolation='nearest', origin='lower')
        ax = plt.axis()
        plt.plot(mA.bx, mA.by, '.', color='1', label='Matched')
        if unmatchedA is not None:
            plt.plot(unmatchedA.bx, unmatchedA.by, 'r.', label='Unmatched A')
        if unmatchedB is not None:
            plt.plot(unmatchedB.bx, unmatchedB.by, 'g.', label='Unmatched B')
        plt.axis(ax)
        plt.figlegend()
        ps.savefig()

        plt.clf()
        plt.imshow(jpgA, interpolation='nearest', origin='lower')
        ax = plt.axis()
        if unmatchedA is not None:
            plt.plot(unmatchedA.bx, unmatchedA.by, 'r.', label='Unmatched A')
        if unmatchedB is not None:
            plt.plot(unmatchedB.bx, unmatchedB.by, 'g.', label='Unmatched B')
        plt.axis(ax)
        plt.figlegend()
        ps.savefig()

        for name, sigmas, I in worst:
            plt.clf()
            C = min(len(sigmas), 5)
            R = 3
            for k,(s,i) in enumerate(zip(sigmas, I)):
                if k >= C:
                    break
                plt.subplot(R, C, 1+k)
                x,y = int(np.round(mA.bx[i])), int(np.round(mA.by[i]))
                h,w,_ = jpgA.shape
                x0 = max(x-10, 0)
                y0 = max(y-10, 0)
                x0 = min(w-20, x0)
                y0 = min(h-20, y0)
                #slc = slice(y0, y0+20), slice(x0, x0+20)
                x1 = x0+20
                y1 = y0+20
                plt.imshow(jpgA[y0:y1, x0:x1, :], extent=[x0,x1,y0,y1],
                           interpolation='nearest', origin='lower')
                #ax = plt.axis()
                plt.subplot(R, C, 1+k + C)
                plt.imshow(modA[y0:y1, x0:x1, :], extent=[x0,x1,y0,y1],
                           interpolation='nearest', origin='lower')
                plt.subplot(R, C, 1+k + C*2)
                plt.imshow(modB[y0:y1, x0:x1, :], extent=[x0,x1,y0,y1],
                           interpolation='nearest', origin='lower')
            plt.suptitle(name)
            ps.savefig()
    
if __name__ == '__main__':
    sys.exit(main())
