from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import pylab as plt
import numpy as np
import fitsio

from astrometry.util.fits import *
from astrometry.util.util import wcs_pv2sip_hdr, Tan
from astrometry.util.resample import *
from astrometry.libkd.spherematch import match_radec
from astrometry.util.plotutils import *
from astrometry.blind.plotstuff import *
from astrometry.util.util import anwcs_new_sip

def mosaic():
    '''
    > cp ~/cosmo/staging/mosaicz/MZLS_CP/CP20180102/k4m_180103_040423_ooi_zd_v1.fits.fz /tmp
    > funpack /tmp/k4m_180103_040423_ooi_zd_v1.fits.fz
    > fitsgetext -i /tmp/k4m_180103_040423_ooi_zd_v1.fits -o mosaic-%02i.wcs -a -H
    > cat mosaic-??.wcs > mosaic.wcs
    > for ((i=1; i<=4; i++)); do modhead mosaic.wcs+$i NAXIS2; modhead mosaic.wcs+$i NAXIS2 0; done
    NAXIS2  =                 4079 / Axis length

    NAXIS2  =                 4079 / Axis length

    NAXIS2  =                 4079 / Axis length

    NAXIS2  =                 4061 / Axis length
    '''
    
    plt.figure(figsize=(4,3))
    plt.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.15)
    
    T = fits_table('obstatus/mosaic-tiles_obstatus.fits')
    print(len(T), 'tiles')
    T.rename('pass', 'passnum')
    T.cut(T.passnum <= 3)
    print(len(T), 'tiles with passnum <= 3')
    ra,dec = 180.216, 40.191
    #ra,dec = 180., 40.
    I,J,d = match_radec(T.ra, T.dec, ra, dec, 2.)
    print(len(I), 'tiles near', ra,dec)
    T.cut(I)
    T.dist = d
    print('dists:', d)
    print('Passes:', T.passnum)
    
    F = fitsio.FITS(os.path.join(os.path.dirname(__file__), 'mosaic.wcs'))
    wcs = []

    heights = [ 4079, 4079, 4079, 4061 ]

    for i in range(1, len(F)):
        hdr = F[i].read_header()
        W = hdr['NAXIS1']
        wcs.append(wcs_pv2sip_hdr(hdr, H=heights[i-1], W=W))
        print('WCS:', wcs[-1])

    # Rendering canvas
    W,H = 2200, 2200
    pixsc = 4./3600.
    targetwcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5, -pixsc, 0., 0., pixsc,
                    float(W), float(H))
    II = np.lexsort((T.dist, T.passnum))

    print('First tile center:', T.ra[II[0]], T.dec[II[0]])
    
    # This is for making the (vector) PDF format tiling images.
    for maxit in [0, 8, 36, 37, 44, 73, 74, 82, 112]:
        plot = Plotstuff(outformat='pdf', ra=ra, dec=dec, width=W*pixsc,
                         size=(W,H), outfn='tile-mosaic-%02i.pdf' % maxit)
        plot.color = 'white'
        plot.alpha = 1.
        plot.plot('fill')
    
        out = plot.outline
        out.fill = True
        out.stepsize = 1024.
        plot.color = 'black'
        plot.alpha = 0.4
        plot.apply_settings()
    
        for it,t in enumerate(T[II]):
            print('Tile', it, 'pass', t.passnum)
            for w in wcs:
                w.set_crval((t.ra, t.dec))
                out.wcs = anwcs_new_sip(w)
                plot.plot('outline')
            if it == maxit:
                print('Writing', it)
                plot.write()
                break
    
    # # And this is for PNG-format tiling images and histograms.
    cov = np.zeros((H,W), np.uint8)
    for it,t in enumerate(T[II]):
        print('Tile', it, 'pass', t.passnum)
        for w in wcs:
            w.set_crval((t.ra, t.dec))
            try:
                Yo,Xo,Yi,Xi,nil = resample_with_wcs(targetwcs, w)
            except:
                continue
            cov[Yo,Xo] += 1
    
        if it in [0, 8, 36, 37, 44, 73, 74, 82, 112]:
            mx = { 1: 2, 2: 4, 3: 6 }[t.passnum]
            # plt.clf()
            # plt.imshow(cov, interpolation='nearest', origin='lower', vmin=0, vmax=mx)
            # plt.colorbar()
            # plt.savefig('tile-%02i.png' % it) 
            plt.imsave('tile-mosaic-%02i.png' % it, cov, origin='lower', vmin=0, vmax=mx, cmap=antigray)
    
        if it in [36, 73, 112]:
            from collections import Counter
            print('Coverage counts:', Counter(cov.ravel()).most_common())
            bins = -0.5 + np.arange(8)
            plt.clf()
            n,b,p = plt.hist(cov.ravel(), bins=bins, normed=True)
            #plt.hist(cov.ravel(), bins=bins, normed=True, cumulative=True, histtype='step')
            # Cumulative histogram from the right...
            xx,yy = [],[]
            for blo,bhi,ni in reversed(list(zip(bins, bins[1:], n))):
                nc = float(np.sum(cov.ravel() > blo)) / len(cov.ravel())
                yy.extend([nc,nc])
                xx.extend([bhi,blo])
                if ni > 0:
                    if nc != ni:
                        if nc > ni+0.03:
                            # If there's room, label the histogram bin above, else below
                            plt.text((blo+bhi)/2., ni, '%.1f \%%' % (100.*ni), ha='center', va='bottom', color='k')
                        else:
                            plt.text((blo+bhi)/2., ni-0.01, '%.1f \%%' % (100.*ni), ha='center', va='top', color='k')
                    plt.text((blo+bhi)/2., nc, '%.1f \%%' % (100.*nc), ha='center', va='bottom', color='k')
    
            plt.plot(xx, yy, 'k-')
    
            plt.xlim(bins.min(), bins.max())
            plt.ylim(0., 1.1)
            plt.xlabel('Number of exposures')
            plt.ylabel('Fraction of sky')
            plt.savefig('hist-mosaic-%02i.pdf' % it)


def decam():
    '''
    cp ~/cosmo/staging/decam/DECam_CP/CP20170731/c4d_170801_080516_oki_g_v1.fits.fz /tmp
    funpack /tmp/c4d_170801_080516_oki_g_v1.fits.fz
    fitsgetext -i /tmp/c4d_170801_080516_oki_g_v1.fits -o decam-%02i.wcs -a -H
    cat decam-??.wcs > decam.wcs
    for ((i=1; i<=61; i++)); do modhead decam.wcs+$i NAXIS2 0; done
    '''
    
    plt.figure(figsize=(4,3))
    plt.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.15)
    
    T = fits_table('obstatus/decam-tiles_obstatus.fits')
    T.rename('pass', 'passnum')
    ra,dec = 0.933, 0.
    I,J,d = match_radec(T.ra, T.dec, ra, dec, 5.) #2.8)
    print(len(I), 'tiles near 0,0')
    T.cut(I)
    T.dist = d
    print('dists:', d)
    print('Passes:', T.passnum)
    
    F = fitsio.FITS(os.path.join(os.path.dirname(__file__), 'decam.wcs'))
    wcs = []
    for i in range(1, len(F)):
        hdr = F[i].read_header()
        wcs.append(wcs_pv2sip_hdr(hdr, W=2046, H=4094))
    
    W,H = 5000, 5000
    pixsc = 4./3600.
    targetwcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5, -pixsc, 0., 0., pixsc,
                    float(W), float(H))
    II = np.lexsort((T.dist, T.passnum))
    
    # This is for making the (vector) PDF format tiling images.
    for maxit in [0, 6, 30, 31, 37, 61, 62, 68, 90]:
        #mx = { 1: 2, 2: 4, 3: 6 }[t.passnum]
        plot = Plotstuff(outformat='pdf', ra=ra, dec=dec, width=W*pixsc,
                         size=(W,H), outfn='tile-%02i.pdf' % maxit)
        plot.color = 'white'
        plot.alpha = 1.
        plot.plot('fill')
    
        out = plot.outline
        out.fill = True
        out.stepsize = 1024.
        plot.color = 'black'
        plot.alpha = 0.4
        plot.apply_settings()
    
        for it,t in enumerate(T[II]):
            print('Tile', it, 'pass', t.passnum)
            for w in wcs:
                w.set_crval((t.ra, t.dec))
                out.wcs = anwcs_new_sip(w)
                plot.plot('outline')
            if it == maxit:
                print('Writing', it)
                plot.write()
                break
    
    # And this is for PNG-format tiling images and histograms.
    cov = np.zeros((H,W), np.uint8)
    for it,t in enumerate(T[II]):
        print('Tile', it, 'pass', t.passnum)
        for w in wcs:
            w.set_crval((t.ra, t.dec))
            #print('WCS:', w)
            try:
                Yo,Xo,Yi,Xi,nil = resample_with_wcs(targetwcs, w)
            except:
                #import traceback
                #traceback.print_exc()
                continue
            cov[Yo,Xo] += 1
    
        if it in [0, 6, 30, 31, 37, 61, 62, 68, 90]:
            mx = { 1: 2, 2: 4, 3: 6 }[t.passnum]
            # plt.clf()
            # plt.imshow(cov, interpolation='nearest', origin='lower', vmin=0, vmax=mx)
            # plt.colorbar()
            # plt.savefig('tile-%02i.png' % it) 
    
            plt.imsave('tile-%02i.png' % it, cov, origin='lower', vmin=0, vmax=mx, cmap=antigray)
            #plt.imsave('tile-%02i.pdf' % it, cov, origin='lower', vmin=0, vmax=mx, cmap=antigray, format='pdf')
    
        if it in [30, 61, 90]:
            from collections import Counter
            print('Coverage counts:', Counter(cov.ravel()).most_common())
            bins = -0.5 + np.arange(8)
            plt.clf()
            n,b,p = plt.hist(cov.ravel(), bins=bins, normed=True)
            #plt.hist(cov.ravel(), bins=bins, normed=True, cumulative=True, histtype='step')
            # Cumulative histogram from the right...
            xx,yy = [],[]
            for blo,bhi,ni in reversed(zip(bins, bins[1:], n)):
                nc = float(np.sum(cov.ravel() > blo)) / len(cov.ravel())
                yy.extend([nc,nc])
                xx.extend([bhi,blo])
                if ni > 0:
                    if nc != ni:
                        if nc > ni+0.03:
                            # If there's room, label the histogram bin above, else below
                            plt.text((blo+bhi)/2., ni, '%.1f \%%' % (100.*ni), ha='center', va='bottom', color='k')
                        else:
                            plt.text((blo+bhi)/2., ni-0.01, '%.1f \%%' % (100.*ni), ha='center', va='top', color='k')
                    plt.text((blo+bhi)/2., nc, '%.1f \%%' % (100.*nc), ha='center', va='bottom', color='k')
    
            plt.plot(xx, yy, 'k-')
    
            plt.xlim(bins.min(), bins.max())
            plt.ylim(0., 1.1)
            plt.xlabel('Number of exposures')
            plt.ylabel('Fraction of sky')
            #plt.title('DECaLS tiling, %i pass%s' % (t.passnum, t.passnum > 1 and 'es' or ''))
            #plt.savefig('hist-%02i.png' % it)
            plt.savefig('hist-%02i.pdf' % it)
        
        


mosaic()
