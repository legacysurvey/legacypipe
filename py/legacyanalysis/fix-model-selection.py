from glob import glob
import fitsio
import sys
from astrometry.util.fits import *
from astrometry.util.file import *
from astrometry.util.starutil_numpy import *
from astrometry.libkd.spherematch import *
from collections import Counter
from legacypipe.oneblob import _select_model
from legacypipe.survey import wcs_for_brick
from astrometry.util.multiproc import multiproc

B = fits_table('/global/project/projectdirs/cosmo/work/legacysurvey/dr8/survey-bricks.fits.gz')

def patch_one(X):
    (ifn, Nfns, fn) = X

    T8 = fits_table(fn)
    phdr = fitsio.read_header(fn)
    hdr = T8.get_header()

    amfn = fn.replace('/tractor-', '/all-models-').replace('/tractor/', '/metrics/')
    A = fits_table(amfn)

    Ahdr = fitsio.read_header(amfn)
    abands = Ahdr['BANDS'].strip()

    nparams = dict(ptsrc=2, simple=2, rex=3, exp=5, dev=5, comp=9)
    galaxy_margin = 3.**2 + (nparams['exp'] - nparams['ptsrc'])
    rex = True

    brick = B[B.brickname == T8.brickname[0]]
    brick = brick[0]
    brickwcs = wcs_for_brick(brick)
    
    assert(len(A) == len(np.flatnonzero(T8.type != 'DUP ')))
    typemap = dict(ptsrc='PSF', rex='REX', dev='DEV', exp='EXP', comp='COMP')
    Tnew = T8.copy()
    #II = []
    npatched = 0
    for i,(d,ttype) in enumerate(zip(A.dchisq, T8.type)):
        dchisqs = dict(zip(['ptsrc','rex','dev','exp','comp'], d))
        mod = _select_model(dchisqs, nparams, galaxy_margin, rex)
        ttype = ttype.strip()
        # The DUP elements appear at the end, and we *zip* A and T8; A does not contain the DUPs
        # so is shorter by the number of DUP elements.
        assert(ttype != 'DUP')
        newtype = typemap[mod]

        #if ttype in ['PSF', 'REX']:
        #    #assert(newtype == ttype)
        #    #if newtyp
        #    continue
        # Can have PSF->EXP, eg, if the REX model fit fails

        # exp->exp, dev->dev, comp->comp
        if ttype == newtype:
            continue
    
        # exp,dev,comp -> ptsrc
        #if (ttype in ['DEV', 'EXP', 'COMP']) and newtype == 'PSF':
        #    II.append(i)
        #    continue

        # Rarer cases.
        #print(ttype, '->', newtype)

        # Copy fit values from the "newtype" entries in all-models
        cols = ['ra', 'dec', 'ra_ivar', 'dec_ivar']
        nt = newtype.lower()
        for c in cols:
            Tnew.get(c)[i] = A.get('%s_%s' % (nt,c))[i]
        # expand flux, flux_ivar
        for c in ['flux', 'flux_ivar']:
            flux = A.get('%s_%s' % (nt,c))[i]
            if len(abands) == 1:
                Tnew.get('%s_%s' % (c,abands[0]))[i] = flux
            else:
                for ib,band in enumerate(abands):
                    Tnew.get('%s_%s' % (c,band))[i] = flux[ib]
        cc = []
        if newtype in ['EXP', 'COMP']:
            cc.append('exp')
        if newtype in ['DEV', 'COMP']:
            cc.append('dev')
        for c1 in cc:
            for c2 in ['e1','e2','r']:
                for c3 in ['', '_ivar']:
                    c = 'shape%s_%s%s' % (c1, c2, c3)
                    ac = '%s_shape%s_%s%s' % (nt, c1, c2, c3)
                    Tnew.get(c)[i] = A.get(ac)[i]
        if newtype == 'COMP':
            Tnew.fracdev[i] = A.comp_fracdev[i]
            Tnew.fracdev_ivar[i] = A.comp_fracdev_ivar[i]

        if newtype == 'PSF':
            # Zero out
            for c1 in ['dev','exp']:
                for c2 in ['e1','e2','r']:
                    for c3 in ['', '_ivar']:
                        c = 'shape%s_%s%s' % (c1, c2, c3)
                        Tnew.get(c)[i] = 0.
            Tnew.fracdev[i] = 0.
            Tnew.fracdev_ivar[i] = 0.

        # recompute bx,by, brick_primary
        ok,x,y = brickwcs.radec2pixelxy(Tnew.ra[i], Tnew.dec[i])
        Tnew.bx[i] = x-1.
        Tnew.by[i] = y-1.
        Tnew.brick_primary[i] = ((Tnew.ra[i] >= brick.ra1 ) * (Tnew.ra[i] < brick.ra2) *
                                  (Tnew.dec[i] >= brick.dec1) * (Tnew.dec[i] < brick.dec2))
        npatched += 1


    print('%i of %i: %s patching %i sources' % (ifn+1, Nfns, os.path.basename(fn), npatched))
    # if len(II) > 0:
    #     II = np.array(II)
    #     print('%i of %i: %s patching %i sources' % (ifn+1, len(fns), os.path.basename(fn), len(II)))
    #     Tnew.type[II] = 'PSF '
    #     # Copy fit values from the "psf" entries in all-models
    #     cols = ['ra', 'dec', 'ra_ivar', 'dec_ivar']
    #     for c in cols:
    #         Tnew.get(c)[II] = A.get('psf_%s' % c)[II]
    #     # expand flux, flux_ivar
    #     for c in ['flux', 'flux_ivar']:
    #         flux = A.get('psf_%s' % (c))[II]
    #         if len(abands) == 1:
    #             Tnew.get('%s_%s' % (c,abands[0]))[i] = flux
    #         else:
    #             for ib,band in enumerate(abands):
    #                 Tnew.get('%s_%s' % (c,band))[i] = flux[ib]
    #     # zero out shape{dev,exp}_{e1,e2,r}{,_ivar} (recall that the sources are all PSFs)
    #     for c1 in ['dev','exp']:
    #         for c2 in ['e1','e2','r']:
    #             for c3 in ['', '_ivar']:
    #                 c = 'shape%s_%s%s' % (c1, c2, c3)
    #                 Tnew.get(c)[II] = 0.
    #     Tnew.fracdev[II] = 0.
    #     Tnew.fracdev_ivar[II] = 0.
    # 
    #     # recompute bx,by, brick_primary
    #     ok,x,y = brickwcs.radec2pixelxy(Tnew.ra[II], Tnew.dec[II])
    #     Tnew.bx[II] = x-1.
    #     Tnew.by[II] = y-1.
    #     Tnew.brick_primary[II] = ((Tnew.ra[II] >= brick.ra1 ) * (Tnew.ra[II] < brick.ra2) *
    #                               (Tnew.dec[II] >= brick.dec1) * (Tnew.dec[II] < brick.dec2))

    outfn = fn.replace('/global/project/projectdirs/cosmo/work/legacysurvey/dr8/decam/tractor/',
                       'patched/')
    outdir = os.path.dirname(outfn)
    try:
        os.makedirs(outdir)
    except:
        pass
    Tnew.writeto(outfn, header=hdr, primheader=phdr)


def main():
    #fns = glob('/global/project/projectdirs/cosmo/work/legacysurvey/dr8/decam/tractor/000/tractor-000??00?.fits')
    fns = glob('/global/project/projectdirs/cosmo/work/legacysurvey/dr8/decam/tractor/*/tractor-*.fits')
    fns.sort()

    N = len(fns)
    args = [(i,N,fn) for i,fn in enumerate(fns)]
    mp = multiproc(8)
    mp.map(patch_one, args)

    # vers = Counter()
    # for fn in fns:
    #     hdr = fitsio.read_header(fn)
    #     vers[hdr['LEGPIPEV']] += 1
    # print('Header versions:', vers.most_common())
    
    #for ifn,fn in enumerate(fns):


if __name__ == '__main__':
    main()
