from __future__ import print_function
import os
import pylab as plt
import matplotlib
import numpy as np
from legacypipe.survey import LegacySurveyData
from astrometry.util.plotutils import PlotSequence
from astrometry.util.fits import fits_table
from astrometry.util.util import Tan

'''
A little script to spot-check the number of exposures in each part of the
sky achieved by the DECaLS tiling.
'''

def main():
    ps = PlotSequence('cov')
    
    survey = LegacySurveyData()

    ra,dec = 242.0, 10.2
    
    fn = 'coverage-ccds.fits'
    if not os.path.exists(fn):
        ccds = survey.get_ccds()
        ccds.cut(ccds.filter == 'r')
        ccds.cut(ccds.propid == '2014B-0404')
        ccds.cut(np.hypot(ccds.ra_bore - ra, ccds.dec_bore - dec) < 2.5)
        print(np.unique(ccds.expnum), 'unique exposures')
        print('propids', np.unique(ccds.propid))
        ccds.writeto(fn)
    else:
        ccds = fits_table(fn)

    plt.clf()
    for e in np.unique(ccds.expnum):
        I = np.flatnonzero(ccds.expnum == e)
        plt.plot(ccds.ra[I], ccds.dec[I], '.')
    ps.savefig()

    degw = 3.0
    pixscale = 10.

    W = degw * 3600 / 10.
    H = W

    hi = 6
    cmap = cmap_discretize('jet', hi+1)

    wcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
              -pixscale/3600., 0., 0., pixscale/3600., float(W), float(H))

    r0,d0 = wcs.pixelxy2radec(1,1)
    r1,d1 = wcs.pixelxy2radec(W,H)
    extent = [min(r0,r1),max(r0,r1), min(d0,d1),max(d0,d1)]
    
    for expnums in [ [348666], [348666,348710, 348686], 
                     [348659, 348667, 348658, 348666, 348665, 348669, 348668],
                     None,
                     [348683, 348687, 347333, 348686, 348685, 348692, 348694,
                      348659, 348667, 348658, 348666, 348665, 348669, 348668,
                      348707, 348709, 348708, 348710, 348711, 348716, 348717],
                      ]:

        nexp = np.zeros((H,W), np.uint8)

        for ccd in ccds:
            if expnums is not None and not ccd.expnum in expnums:
                continue

            ccdwcs = survey.get_approx_wcs(ccd)
            r,d = ccdwcs.pixelxy2radec(1, 1)
            ok,x0,y0 = wcs.radec2pixelxy(r, d)
            r,d = ccdwcs.pixelxy2radec(ccd.width, ccd.height)
            ok,x1,y1 = wcs.radec2pixelxy(r, d)
            xlo = np.clip(int(np.round(min(x0,x1))) - 1, 0, W-1)
            xhi = np.clip(int(np.round(max(x0,x1))) - 1, 0, W-1)
            ylo = np.clip(int(np.round(min(y0,y1))) - 1, 0, H-1)
            yhi = np.clip(int(np.round(max(y0,y1))) - 1, 0, H-1)
            nexp[ylo:yhi+1, xlo:xhi+1] += 1

        plt.clf()
        plt.imshow(nexp, interpolation='nearest', origin='lower',
                   vmin=-0.5, vmax=hi+0.5, cmap=cmap, extent=extent)
        plt.colorbar(ticks=np.arange(hi+1))
        ps.savefig()
    

    O = fits_table('obstatus/decam-tiles_obstatus.fits')
    O.cut(np.hypot(O.ra - ra, O.dec - dec) < 2.5)

    for p in [1,2,3]:
        print('Pass', p, 'exposures:', O.r_expnum[O.get('pass') == p])

    O.cut(O.get('pass') == 2)
    print(len(O), 'pass 2 nearby')

    d = np.hypot(O.ra - ra, O.dec - dec)
    print('Dists:', d)

    I = np.flatnonzero(d < 0.5)
    assert(len(I) == 1)
    ocenter = O[I[0]]
    print('Center expnum', ocenter.r_expnum)
    
    I = np.flatnonzero(d >= 0.5)
    O.cut(I)

    #center = ccds[ccds.expnum == ocenter.r_expnum]
    #p2 = ccds[ccds.

    ok,xc,yc = wcs.radec2pixelxy(ocenter.ra, ocenter.dec)
    
    xx,yy = np.meshgrid(np.arange(W)+1, np.arange(H)+1)
    c_d2 = (xc - xx)**2 + (yc - yy)**2

    best = np.ones((H,W), bool)

    for o in O:
        ok,x,y = wcs.radec2pixelxy(o.ra, o.dec)
        d2 = (x - xx)**2 + (y - yy)**2
        best[d2 < c_d2] = False
        del d2
        
    del c_d2,xx,yy
        
    # plt.clf()
    # plt.imshow(best, interpolation='nearest', origin='lower', cmap='gray',
    #            vmin=0, vmax=1)
    # ps.savefig()

    plt.clf()
    plt.imshow(nexp * best, interpolation='nearest', origin='lower',
               vmin=-0.5, vmax=hi+0.5, cmap=cmap, extent=extent)
    plt.colorbar(ticks=np.arange(hi+1))
    ps.savefig()

    plt.clf()
    n,b,p = plt.hist(np.clip(nexp[best], 0, hi), range=(-0.5,hi+0.5), bins=hi+1)
    plt.xlim(-0.5, hi+0.5)
    ps.savefig()

    print('b', b)
    print('n', n)
    print('fracs', np.array(n) / np.sum(n))

    print('pcts', ', '.join(['%.1f' % f for f in 100. * np.array(n)/np.sum(n)]))
    
    #rr,dd = wcs.pixelxy2radec(xx,yy)
    


    
# From http://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html
def cmap_discretize(cmap, N):
    from matplotlib.cm import get_cmap
    from numpy import concatenate, linspace
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = concatenate((linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in xrange(N+1) ]
    # Return colormap object.
    return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

if __name__ == '__main__':
    main()
    
