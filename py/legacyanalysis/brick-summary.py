from __future__ import print_function
import os
import fitsio
import numpy as np
from glob import glob
from collections import Counter
from astrometry.util.fits import fits_table
from astrometry.util.util import Tan
#from tractor.sfd import SFDMap

def ring_unique(wcs, W, i, unique, ra1,ra2,dec1,dec2):
    lo,hi = i, W-i-1
    # one slice per side; we double-count the last pix of each side.
    side = slice(lo,hi+1)
    top = (lo, side)
    bot = (hi, side)
    left  = (side, lo)
    right = (side, hi)
    xx = yy = np.arange(W)
    nu,ntot = 0,0
    for slc in [top, bot, left, right]:
        #print('xx,yy', xx[slc], yy[slc])
        (yslc,xslc) = slc
        rr,dd = wcs.pixelxy2radec(xx[xslc]+1, yy[yslc]+1)
        U = (rr >= ra1 ) * (rr < ra2 ) * (dd >= dec1) * (dd < dec2)
        #print('Pixel', i, ':', np.sum(U), 'of', len(U), 'pixels are unique')
        #allin *= np.all(U)
        unique[slc] = U
        nu += np.sum(U)
        ntot += len(U)
    #if allin:
    #    print('Scanned to pixel', i)
    #    break
    return nu,ntot

def find_unique_pixels(wcs, W, unique, ra1,ra2,dec1,dec2):
    step = 10
    for i in range(0, W/2, step):
        nu,ntot = ring_unique(wcs, W, i, unique, ra1,ra2,dec1,dec2)
        #print('Pixel', i, ': nu/ntot', nu, ntot)
        if nu > 0:
            i -= step
            break
        unique[:i,:] = False
        unique[W-1-i:,:] = False
        unique[:,:i] = False
        unique[:,W-1-i:] = False

    for j in range(i+1, W/2):
        nu,ntot = ring_unique(wcs, W, j, unique, ra1,ra2,dec1,dec2)
        #print('Pixel', j, ': nu/ntot', nu, ntot)
        if nu == ntot:
            break


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', dest='outfn', help='Output filename',
                      default='TMP/nexp.fits')
    parser.add_argument('--merge', action='store_true', help='Merge sub-tables')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('files', metavar='nexp-file.fits.gz', nargs='+',
                        help='List of nexp files to process')

    opt = parser.parse_args()

    fns = opt.files

    if opt.merge:
        from astrometry.util.fits import merge_tables
        TT = []
        for fn in fns:
            T = fits_table(fn)
            print(fn, '->', len(T))
            TT.append(T)
        T = merge_tables(TT)
        T.writeto(opt.outfn)
        print('Wrote', opt.outfn)

    if opt.plot:
        T = fits_table(opt.files[0])
        import pylab as plt
        import matplotlib
        
        cm = matplotlib.cm.get_cmap('jet', 6)

        ax = [360, 0, -21, 36]

        plt.figure(figsize=(8,5))
        for band in 'grz':
            plt.clf()
            plt.scatter(T.ra, T.dec, c=T.get('nexp_%s' % band), s=2,
                        edgecolors='none',
                        vmin=-0.5, vmax=5.5, cmap=cm)
            plt.axis(ax)
            plt.xlabel('RA (deg)')
            plt.xticks(np.arange(0, 361, 45))
            plt.ylabel('Dec (deg)')
            plt.colorbar(ticks=range(6))
            plt.title('DECaLS DR3: Number of exposures in %s' % band)
            plt.savefig('nexp-%s.png' % band)

        for col in ['nobjs', 'npsf', 'nsimp', 'nexp', 'ndev', 'ncomp']:
            plt.clf()
            N = T.get(col)
            mx = np.percentile(N, 99.5)
            plt.scatter(T.ra, T.dec, c=N, s=2,
                        edgecolors='none', vmin=0, vmax=mx)
            plt.axis(ax)
            plt.xlabel('RA (deg)')
            plt.xticks(np.arange(0, 361, 45))
            plt.ylabel('Dec (deg)')
            plt.colorbar()
            plt.title('DECaLS DR3: Number of objects of type %s' % col[1:])
            plt.savefig('nobjs-%s.png' % col[1:])

        Ntot = T.nobjs
        for col in ['npsf', 'nsimp', 'nexp', 'ndev', 'ncomp']:
            plt.clf()
            N = T.get(col) / Ntot.astype(np.float32)
            mx = np.percentile(N, 99.5)
            plt.scatter(T.ra, T.dec, c=N, s=2,
                        edgecolors='none', vmin=0, vmax=mx)
            plt.axis(ax)
            plt.xlabel('RA (deg)')
            plt.xticks(np.arange(0, 361, 45))
            plt.ylabel('Dec (deg)')
            plt.colorbar()
            plt.title('DECaLS DR3: Fraction of objects of type %s' % col[1:])
            plt.savefig('fobjs-%s.png' % col[1:])

            
        return 0

    # fnpats = opt.files
    # fns = []
    # for pat in fnpats:
    #     pfns = glob(pat)
    #     fns.extend(pfns)
    #     print('Pattern', pat, '->', len(pfns), 'files')
    #fns = glob('coadd/*/*/*-nexp*')
    #fns = glob('coadd/000/*/*-nexp*')
    #fns = glob('coadd/000/0001*/*-nexp*')
    fns.sort()
    print(len(fns), 'nexp files')
    
    brickset = set()
    bricklist = []
    gn = []
    rn = []
    zn = []
    
    gnhist = []
    rnhist = []
    znhist = []
    
    nnhist = 6
    
    gdepth = []
    rdepth = []
    zdepth = []
    
    ibricks = []
    nsrcs = []
    npsf  = []
    nsimp = []
    nexp  = []
    ndev  = []
    ncomp = []
    
    gpsfsize = []
    rpsfsize = []
    zpsfsize = []
    ebv = []
    gtrans = []
    rtrans = []
    ztrans = []
    
    bricks = fits_table('survey-bricks.fits.gz')
    
    #sfd = SFDMap()
    
    W = H = 3600
    # H=3600
    # xx,yy = np.meshgrid(np.arange(W), np.arange(H))
    unique = np.ones((H,W), bool)
    tlast = 0
    
    for fn in fns:
        print('File', fn)
        words = fn.split('/')
        dirprefix = '/'.join(words[:-4])
        print('Directory prefix:', dirprefix)
        words = words[-4:]
        brick = words[2]
        print('Brick', brick)
        if not brick in brickset:
            brickset.add(brick)
            bricklist.append(brick)
            gn.append(0)
            rn.append(0)
            zn.append(0)
    
            gnhist.append([0 for i in range(nnhist)])
            rnhist.append([0 for i in range(nnhist)])
            znhist.append([0 for i in range(nnhist)])
    
            index = -1
            ibrick = np.nonzero(bricks.brickname == brick)[0][0]
            ibricks.append(ibrick)
            tfn = os.path.join(dirprefix, 'tractor', brick[:3], 'tractor-%s.fits'%brick)
            print('Tractor filename', tfn)
            T = fits_table(tfn, columns=['brick_primary', 'type', 'decam_psfsize',
                                         'ebv', 'decam_mw_transmission'])
            T.cut(T.brick_primary)
            nsrcs.append(len(T))
            types = Counter([t.strip() for t in T.type])
            npsf.append(types['PSF'])
            nsimp.append(types['SIMP'])
            nexp.append(types['EXP'])
            ndev.append(types['DEV'])
            ncomp.append(types['COMP'])
            print('N sources', nsrcs[-1])
    
            gpsfsize.append(np.median(T.decam_psfsize[:,1]))
            rpsfsize.append(np.median(T.decam_psfsize[:,2]))
            zpsfsize.append(np.median(T.decam_psfsize[:,4]))
    
            ebv.append(np.median(T.ebv))
            gtrans.append(np.median(T.decam_mw_transmission[:,1]))
            rtrans.append(np.median(T.decam_mw_transmission[:,2]))
            ztrans.append(np.median(T.decam_mw_transmission[:,4]))
    
            br = bricks[ibrick]
    
            print('Computing unique brick pixels...')
            #wcs = Tan(fn, 0)
            #W,H = int(wcs.get_width()), int(wcs.get_height())
    
            pixscale = 0.262/3600.
            wcs = Tan(br.ra, br.dec, W/2.+0.5, H/2.+0.5,
                      -pixscale, 0., 0., pixscale,
                      float(W), float(H))
            import time
    
            t0 = time.clock()
    
            # scan the outer annulus of pixels, and shrink in until all pixels are unique.
            unique[:,:] = True
    
            find_unique_pixels(wcs, W, unique, br.ra1, br.ra2, br.dec1, br.dec2)
    
            # for i in range(W/2):
            #     allin = True
            #     lo,hi = i, W-i-1
            #     # one slice per side
            #     side = slice(lo,hi+1)
            #     top = (lo, side)
            #     bot = (hi, side)
            #     left  = (side, lo)
            #     right = (side, hi)
            #     for slc in [top, bot, left, right]:
            #         #print('xx,yy', xx[slc], yy[slc])
            #         rr,dd = wcs.pixelxy2radec(xx[slc]+1, yy[slc]+1)
            #         U = (rr >= br.ra1 ) * (rr < br.ra2 ) * (dd >= br.dec1) * (dd < br.dec2)
            #         #print('Pixel', i, ':', np.sum(U), 'of', len(U), 'pixels are unique')
            #         allin *= np.all(U)
            #         unique[slc] = U
            #     if allin:
            #         print('Scanned to pixel', i)
            #         break
    
            t1 = time.clock()
            U = np.flatnonzero(unique)
            t2 = time.clock()
            print(len(U), 'of', W*H, 'pixels are unique to this brick')
    
            # #t3 = time.clock()
            #rr,dd = wcs.pixelxy2radec(xx+1, yy+1)
            # #t4 = time.clock()
            # #u = (rr >= br.ra1 ) * (rr < br.ra2 ) * (dd >= br.dec1) * (dd < br.dec2)
            # #t5 = time.clock()
            # #U2 = np.flatnonzero(u)
            #U2 = np.flatnonzero((rr >= br.ra1 ) * (rr < br.ra2 ) *
            #                    (dd >= br.dec1) * (dd < br.dec2))
            #assert(np.all(U == U2))
            #assert(len(U) == len(U2))
            # #t6 = time.clock()
            # print(len(U2), 'of', W*H, 'pixels are unique to this brick')
            # 
    
            #print(t0-tlast, 'other time')
            #tlast = time.clock() #t2
            #print('t1:', t1-t0, 't2', t2-t1)
    
            # #print('t4:', t4-t3, 't5', t5-t4, 't6', t6-t5)
            # 
    
        else:
            index = bricklist.index(brick)
            assert(index == len(bricklist)-1)
    
        index = bricklist.index(brick)
        assert(index == len(bricklist)-1)
    
        filepart = words[-1]
        filepart = filepart.replace('.fits.gz', '')
        print('File:', filepart)
        band = filepart[-1]
        assert(band in 'grz')
    
        nlist,nhist = dict(g=(gn,gnhist), r=(rn,rnhist), z=(zn,znhist))[band]
    
        upix = fitsio.read(fn).flat[U]
        med = np.median(upix)
        print('Band', band, ': Median', med)
        nlist[index] = med
    
        hist = nhist[index]
        for i in range(nnhist):
            if i < nnhist-1:
                hist[i] = np.sum(upix == i)
            else:
                hist[i] = np.sum(upix >= i)
        assert(sum(hist) == len(upix))
        print('Number of exposures histogram:', hist)
    
    ibricks = np.array(ibricks)
    
    print('Maximum number of sources:', max(nsrcs))
    
    T = fits_table()
    T.brickname = np.array(bricklist)
    T.ra  = bricks.ra [ibricks]
    T.dec = bricks.dec[ibricks]
    T.nexp_g = np.array(gn).astype(np.int16)
    T.nexp_r = np.array(rn).astype(np.int16)
    T.nexp_z = np.array(zn).astype(np.int16)
    T.nexphist_g = np.array(gnhist).astype(np.int32)
    T.nexphist_r = np.array(rnhist).astype(np.int32)
    T.nexphist_z = np.array(znhist).astype(np.int32)
    T.nobjs  = np.array(nsrcs).astype(np.int16)
    T.npsf   = np.array(npsf ).astype(np.int16)
    T.nsimp  = np.array(nsimp).astype(np.int16)
    T.nexp   = np.array(nexp ).astype(np.int16)
    T.ndev   = np.array(ndev ).astype(np.int16)
    T.ncomp  = np.array(ncomp).astype(np.int16)
    T.psfsize_g = np.array(gpsfsize).astype(np.float32)
    T.psfsize_r = np.array(rpsfsize).astype(np.float32)
    T.psfsize_z = np.array(zpsfsize).astype(np.float32)
    T.ebv = np.array(ebv).astype(np.float32)
    T.trans_g = np.array(gtrans).astype(np.float32)
    T.trans_r = np.array(rtrans).astype(np.float32)
    T.trans_z = np.array(ztrans).astype(np.float32)
    T.writeto(opt.outfn)

if __name__ == '__main__':
    main()
