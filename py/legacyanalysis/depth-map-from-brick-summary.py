import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = (12,8)
matplotlib.rcParams['figure.dpi'] = 200
import pylab as plt

from astrometry.util.fits import *
import numpy as np
from astrometry.util.starutil_numpy import *
from astrometry.util.util import *

zoom = 1
W,H = 1000,500
#W,H = 2000,1000
ra_center = 265.
wcs = anwcs_create_hammer_aitoff(ra_center, 0., zoom, W, H, False)

#Bs = fits_table('~/legacypipe/py/survey-bricks-dr5.fits.gz')
#Bs = fits_table('~/legacypipe/py/survey-bricks-dr7.fits.gz')
#Bn = fits_table('~/legacypipe/py/survey-bricks-dr6.fits.gz')
Bs = fits_table('/global/project/projectdirs/cosmo/data/legacysurvey/dr8/south/survey-bricks-dr8-south.fits.gz')
Bn = fits_table('/global/project/projectdirs/cosmo/data/legacysurvey/dr8/north/survey-bricks-dr8-north.fits.gz')


Bs.l,Bs.b = radectolb(Bs.ra, Bs.dec)
Bn.l,Bn.b = radectolb(Bn.ra, Bn.dec)

ok,Bs.x,Bs.y = wcs.radec2pixelxy(Bs.ra, Bs.dec)
ok,Bn.x,Bn.y = wcs.radec2pixelxy(Bn.ra, Bn.dec)

decsplit = 32.375

Bn.cut((Bn.b > 0) * (Bn.dec > decsplit))

Bs.cut(np.logical_or(Bs.b <= 0, (Bs.b > 0) * (Bs.dec <= decsplit)))

# Daniel
Bn.cut(Bn.dec >= -10)
#Bs.cut(Bs.dec >= -20)

### subsample
#Bs.cut(np.random.permutation(len(Bs))[:int(0.1*len(Bs))])
#Bn.cut(np.random.permutation(len(Bn))[:int(0.1*len(Bn))])

for band in 'zrg':
    plt.clf()
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    lo,hi = { 'g':(23.0,25.0), 'r':(22.4,24.4), 'z':(21.5,23.5) }[band]

    if False:
        plt.plot(Bn.x, Bn.y, 'o', ms=1, color='0.5')
        plt.plot(Bs.x, Bs.y, 'o', ms=1, color='0.5')
    else:
        kw = dict(s=1, vmin=lo, vmax=hi, cmap='RdYlBu')
        plt.scatter(Bn.x, Bn.y, c=Bn.get('galdepth_'+band) - Bn.get('ext_'+band), **kw)
        plt.scatter(Bs.x, Bs.y, c=Bs.get('galdepth_'+band) - Bs.get('ext_'+band), **kw)
        c = plt.colorbar(orientation='horizontal')
        c.set_label('%s-band depth (mag)' % band)

    dec_lo = -70
        
    dec_gridlines = list(range(dec_lo, 90, 10))
    dec_gridlines_ras = np.arange(ra_center-180, ra_center+180, 1)
    ra_gridlines = range(0, 360, 30)
    ra_gridlines_decs = np.arange(dec_lo, 90, 1.)
    for d in dec_gridlines:
        rr = dec_gridlines_ras
        dd = np.zeros_like(rr) + d
        ok,xx,yy = wcs.radec2pixelxy(rr, dd)
        plt.plot(xx, yy, 'k-', alpha=0.1)
    for r in ra_gridlines:
        dd = ra_gridlines_decs
        rr = np.zeros_like(dd) + r
        ok,xx,yy = wcs.radec2pixelxy(rr, dd)
        plt.plot(xx, yy, 'k-', alpha=0.1)
    
    ra_gridlines2 = [ra_center-180, ra_center+180]
    ra_gridlines2_decs = np.arange(dec_lo, 91, 1.)
    for r in ra_gridlines2:
        dd = ra_gridlines2_decs
        rr = np.zeros_like(dd) + r
        ok,xx,yy = wcs.radec2pixelxy(rr, dd)
        plt.plot(xx, yy, 'k-', alpha=0.5)
    
    ra_labels = ra_gridlines
    dec_labels = dec_gridlines
    ra_labels_dec = -30
    dec_labels_ra = ra_center+180
    
    ok,xx,yy = wcs.radec2pixelxy(ra_labels, ra_labels_dec)
    for x,y,v in zip(xx, yy, ra_labels):
        plt.text(x, y, '%i'%(v%360), ha='center', va='top', alpha=0.5)
    ok,xx,yy = wcs.radec2pixelxy(dec_labels_ra, dec_labels)
    for x,y,v in zip(xx, yy, dec_labels):
        plt.text(x-20, y, '%+i'%v, ha='right', va='center', alpha=0.5)

    # Galactic plane
    ll = np.linspace(0., 360., 720)
    bb = np.zeros_like(ll)
    rr,dd = lbtoradec(ll, bb)
    ok,xx,yy = wcs.radec2pixelxy(rr, dd)
    # Plot segments that are above Dec=-30 and not discontinuous
    d = np.append([0], np.hypot(np.diff(xx), np.diff(yy)))
    ok = (d < 100)# * (dd > -30)
    istart = 0
    while istart < len(ok):
        while istart < len(ok) and ok[istart] == False:
            istart += 1
        iend = istart
        while iend < len(ok) and ok[iend] == True:
            iend += 1
        if iend != istart:
            #print('Plotting from', istart, 'to', iend, 'ok', ok[istart:iend])
            plt.plot(xx[istart:iend], yy[istart:iend], '-', color='0.6', lw=2)
        istart = iend

    # Label regions
    for r,d,n in [(30, 0, 'DES'),
                  (0, 20, 'DECaLS'),
                  (180, 10, 'DECaLS'),
                  (180, 50, 'MzLS+BASS')]:
        ok,x,y = wcs.radec2pixelxy(r, d)
        plt.text(x, y, n, fontsize=16, ha='center', va='center')
    
    plt.xticks([])
    plt.yticks([])
    #plt.axis('equal');
    ax = [0,W, 0.1*H, H]
    plt.axis(ax)
    plt.axis('equal')
    plt.axis(ax)
    plt.gca().set_frame_on(False)

    plt.savefig('depth-%s.png' % band)
    print('Wrote', band, 'png')
    plt.savefig('depth-%s.pdf' % band)

