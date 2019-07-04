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
    (ifn, Nfns, fn, outfn, fix_dup) = X

    if os.path.exists(outfn):
        print(ifn, 'of', Nfns, ':', fn, ': output', outfn, 'exists')
        return

    T8 = fits_table(fn)
    phdr = fitsio.read_header(fn)
    hdr = T8.get_header()

    utypes = np.unique(T8.type)
    T8.type = T8.type.astype('S4')

    dupstring = np.array('DUP ').astype('S4')
    I = np.flatnonzero(T8.type == dupstring)
    #print(ifn, 'of', Nfns, ':', fn, ':', len(I), 'DUP', 'ver:', phdr['LEGPIPEV'])
    print(ifn, 'of', Nfns, ':', fn, ':', 'ver:', phdr['LEGPIPEV'], 'types:', list(utypes))

    if fix_dup and len(I) > 0:
        T8.objid[I] = I
        T8.brickname[I] = T8.brickname[0]
        T8.brickid[I] = T8.brickid[0]
    assert(len(np.unique(T8.objid)) == len(T8))

    # Add mask bit definitions to headers
    phdr.add_record(dict(name='COMMENT', value='WISEMASK bit values:'))
    wisebits = [
        (0, 'BRIGHT' , 'Bright star core and wings'),
        (1, 'SPIKE'  , 'PSF-based diffraction spike'),
        (2, 'GHOST'  , ''),
        (3, 'LATENT' , 'First latent'),
        (4, 'LATENT2', 'Second latent image'),
        (5, 'HALO'   , 'AllWISE-like circular halo'),
        (6, 'SATUR'  , 'Bright star saturation'),
        (7, 'SPIKE2' , 'Geometric diffraction spike'),
    ]
    #name_map = {'LATENT2': 'LATEN2'}

    for bit,name,comm in wisebits:
        phdr.add_record(dict(name='WBITN%i' % bit, value=name, comment=comm + ' (0x%x)' % (1<<bit)))
    #for bit,name,comm in wisebits:
    #    phdr.add_record(dict(name='W%s' % name_map.get(name, name), value=1<<bit, comment=comm))

    phdr.add_record(dict(name='COMMENT', value='MASKBITS bit values:'))
    maskbits = [
        (0 , 'NPRIMARY', 'Not-brick-primary'),
        (1 , 'BRIGHT',   'Bright star in blob'),
        (2 , 'SATUR_G',  'g saturated + margin'),
        (3 , 'SATUR_R',  'r saturated + margin'),
        (4 , 'SATUR_Z',  'z saturated + margin'),
        (5 , 'ALLMASK_G', 'Any ALLMASK_G bit set'),
        (6 , 'ALLMASK_R', 'Any ALLMASK_R bit set'),
        (7 , 'ALLMASK_Z', 'Any ALLMASK_Z bit set'),
        (8 , 'WISEM1',   'WISE W1 bright star mask'),
        (9 , 'WISEM2',   'WISE W2 bright star mask'),
        (10, 'BAILOUT',  'Bailed out of processing'),
        (11, 'MEDIUM',   'Medium-bright star'),
        (12, 'GALAXY',   'LSLGA large galaxy'),
        (13, 'CLUSTER',  'Cluster'),
    ]
    # name_map = {
    #     'NPRIMARY':  'NPRIMRY',
    #     'ALLMASK_G': 'ALLM_G',
    #     'ALLMASK_R': 'ALLM_R',
    #     'ALLMASK_Z': 'ALLM_Z',
    #     }

    for bit,name,comm in maskbits:
        phdr.add_record(dict(name='MBITN%i' % bit, value=name, comment=comm + ' (0x%x)' % (1<<bit)))
    #for bit,name,comm in maskbits:
    #    phdr.add_record(dict(name='M%s' % name_map.get(name, name), value=1<<bit, comment=comm))

    phdr.add_record(dict(name='COMMENT', value='ANYMASK/ALLMASK bit values:'))
    anybits = [
        (0, 'BADPIX', 'Bad columns, hot pixels, etc'),
        (1, 'SATUR',  'Saturated'),
        (2, 'INTERP', 'Interpolated'),
        (4, 'CR',     'Cosmic ray'),
        (6, 'BLEED',  'Bleed trail'),
        (7, 'TRANS',  'Transient'),
        (8, 'EDGE',   'Edge pixel'),
        (9, 'EDGE2',  'Edge pixel, jr'),
        (11,'OUTLIER', 'Outlier from stack'),
    ]
    #name_map = {}

    for bit,name,comm in anybits:
        phdr.add_record(dict(name='ABITN%i' % bit, value=name, comment=comm + ' (0x%x)' % (1<<bit)))
    #for bit,name,comm in anybits:
    #    phdr.add_record(dict(name='A%s' % name_map.get(name, name), value=1<<bit, comment=comm))

    phdr.add_record(dict(name='COMMENT', value='BRIGHTBLOB bit values:'))
    brightbits = [
        (0, 'BRIGHT', 'Bright star'),
        (1, 'MEDIUM', 'Medium-bright star'),
        (2, 'CLUSTER', 'Globular cluster'),
        (3, 'GALAXY',  'Large LSLGA galaxy'),
    ]
    #name_map = {}

    for bit,name,comm in brightbits:
        phdr.add_record(dict(name='BBITN%i' % bit, value=name, comment=comm + ' (0x%x)' % (1<<bit)))
    #for bit,name,comm in brightbits:
    #    phdr.add_record(dict(name='B%s' % name_map.get(name, name), value=1<<bit, comment=comm))

    # Ugh, need to copy units
    columns = T8.get_columns()

    # Add in missing units
    extraunits = dict(bx='pix', by='pix',
                      pmra='mas/yr', pmdec='mas/yr', parallax='mas',
                      pmra_ivar='1/(mas/yr)^2', pmdec_ivar='1/(mas/yr)^2', parallax_ivar='1/mas^2',
                      ref_epoch='yr',
                      gaia_phot_g_mean_mag='mag',
                      gaia_phot_bp_mean_mag='mag',
                      gaia_phot_rp_mean_mag='mag',
                      psfdepth_w1='1/nanomaggy^2',
                      psfdepth_w2='1/nanomaggy^2',
                      psfdepth_w3='1/nanomaggy^2',
                      psfdepth_w4='1/nanomaggy^2')

    units = []
    for i,col in enumerate(columns):
        typekey = 'TTYPE%i' % (i+1)
        assert(hdr[typekey].strip() == col)
        unitkey = 'TUNIT%i' % (i+1)
        if unitkey in hdr:
            unit = hdr[unitkey]
        else:
            unit = extraunits.get(col, '')
        units.append(unit)

    outdir = os.path.dirname(outfn)
    try:
        os.makedirs(outdir)
    except:
        pass

    try:
        T8.writeto(outfn, header=hdr, primheader=phdr, units=units)
    except:
        print('Failed to write', outfn)
        if os.path.exists(outfn):
            os.remove(outfn)

def main():
    #fns = glob('/global/project/projectdirs/cosmo/work/legacysurvey/dr8/90prime-mosaic/tractor/*/tractor-*.fits')

    # DR8-south
    #prefix = '/global/project/projectdirs/cosmo/work/legacysurvey/dr8/south/tractor/'
    prefix = '/global/project/projectdirs/cosmo/work/legacysurvey/dr8-garage/south/original-tractor/'
    out_prefix = 'patched-dr8-south-2/'
    pat = prefix + '*/tractor-*.fits'
    #pat = prefix + '013/tractor-0132m265.fits'
    fix_dup = True

    fns = glob(pat)
    fns.sort()
    print(len(fns), 'Tractor catalogs')

    outfns = [fn.replace(prefix, out_prefix) for fn in fns]

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
    args = [(i,N,fn,outfn,fix_dup) for i,(fn,outfn) in enumerate(zip(fns, outfns))]
    mp = multiproc(32)
    #mp = multiproc()
    mp.map(patch_one, args)

if __name__ == '__main__':
    main()
