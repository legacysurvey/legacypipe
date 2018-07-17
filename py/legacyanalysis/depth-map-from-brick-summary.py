import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = (12,8)
import pylab as plt

from astrometry.util.fits import *
import numpy as np
from astrometry.util.starutil_numpy import *
from astrometry.util.util import *

zoom = 1
W,H = 1000,500
ra_center = 265.
wcs = anwcs_create_hammer_aitoff(ra_center, 0., zoom, W, H, False)

B5 = fits_table('~/legacypipe/py/survey-bricks-dr5.fits.gz')
B6 = fits_table('~/legacypipe/py/survey-bricks-dr6.fits.gz')

B6.l,B6.b = radectolb(B6.ra, B6.dec)

ok,B5.x,B5.y = wcs.radec2pixelxy(B5.ra, B5.dec)
ok,B6.x,B6.y = wcs.radec2pixelxy(B6.ra, B6.dec)

I = np.flatnonzero((B6.b > 0) * (B6.dec > 30))

for band in 'grz':
    plt.clf()
    #plt.plot(B5.x, B5.y, 'k.')
    #plt.plot(B6.x[I], B6.y[I], 'k.')

    lo,hi = { 'g':(23,25), 'r':(22.5,24.5), 'z':(21.5,23.5) }[band]

    kw = dict(s=1, vmin=lo, vmax=hi, cmap='summer')    
    plt.scatter(B5.x, B5.y, c=B5.get('galdepth_'+band), **kw)
    plt.scatter(B6.x[I], B6.y[I], c=B6.get('galdepth_'+band)[I], **kw)
    c = plt.colorbar(orientation='horizontal')
    c.set_label('%s-band depth (mag)' % band)

    dec_gridlines = list(range(-30, 90, 10))
    dec_gridlines_ras = np.arange(ra_center-180, ra_center+180, 1)
    ra_gridlines = range(0, 360, 30)
    ra_gridlines_decs = np.arange(-30, 90, 1.)
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
    ra_gridlines2_decs = np.arange(-30, 91, 1.)
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
    
    plt.xticks([])
    plt.yticks([])
    #plt.axis('equal');
    plt.axis([0,1000,100,500]);
    plt.axis('equal')
    plt.axis([0,1000,100,500]);
    plt.gca().set_frame_on(False)

    plt.savefig('depth-%s.png' % band)

