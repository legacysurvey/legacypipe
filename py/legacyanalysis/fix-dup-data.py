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

def patch_one(X):
    (ifn, Nfns, fn) = X

    T8 = fits_table(fn)
    phdr = fitsio.read_header(fn)
    hdr = T8.get_header()

    I = np.flatnonzero(T8.type == 'DUP ')
    print(ifn, 'of', Nfns, ':', fn, ':', len(I), 'DUP', 'ver:', phdr['LEGPIPEV'])
    if len(I) == 0:
        return

    T8.objid[I] = I
    assert(len(np.unique(T8.objid)) == len(T8))
    T8.brickname[I] = T8.brickname[0]
    T8.brickid[I] = T8.brickid[0]

    outfn = fn.replace('/global/project/projectdirs/cosmo/work/legacysurvey/dr8/90prime-mosaic/tractor/',
                       'patched-dup/')
    outdir = os.path.dirname(outfn)
    try:
        os.makedirs(outdir)
    except:
        pass
    T8.writeto(outfn, header=hdr, primheader=phdr)


def main():
    fns = glob('/global/project/projectdirs/cosmo/work/legacysurvey/dr8/90prime-mosaic/tractor/000/tractor-*.fits')
    fns.sort()
    print(len(fns), 'Tractor catalogs')

    # vers = Counter()
    # keepfns = []
    # for fn in fns:
    #     hdr = fitsio.read_header(fn)
    #     ver = hdr['LEGPIPEV']
    #     ver = ver.strip()
    #     vers[ver] += 1
    #     if ver == 'DR8.2.1':
    #         keepfns.append(fn)
    # 
    # print('Header versions:', vers.most_common())
    # 
    # fns = keepfns
    # print('Keeping', len(fns), 'with bad version')

    N = len(fns)
    args = [(i,N,fn) for i,fn in enumerate(fns)]
    mp = multiproc(8)
    mp.map(patch_one, args)

if __name__ == '__main__':
    main()
