import matplotlib
matplotlib.use('Agg')
import pylab as plt

from astrometry.util.fits import *
from astrometry.libkd.spherematch import *

fns = ['dr2i/tractor/149/tractor-1498p017.fits',
       'dr2i/tractor/149/tractor-1498p020.fits',
       'dr2i/tractor/149/tractor-1498p022.fits',
       'dr2i/tractor/149/tractor-1498p025.fits',
       'dr2i/tractor/150/tractor-1501p017.fits',
       'dr2i/tractor/150/tractor-1501p020.fits',
       'dr2i/tractor/150/tractor-1501p022.fits',
       'dr2i/tractor/150/tractor-1501p025.fits',
       'dr2i/tractor/150/tractor-1503p017.fits',
       'dr2i/tractor/150/tractor-1503p020.fits',
       'dr2i/tractor/150/tractor-1503p022.fits',
       'dr2i/tractor/150/tractor-1503p025.fits',
       'dr2i/tractor/150/tractor-1506p017.fits',
       'dr2i/tractor/150/tractor-1506p020.fits',
       'dr2i/tractor/150/tractor-1506p022.fits',
       'dr2i/tractor/150/tractor-1506p025.fits',
       ]

T = merge_tables([fits_table(fn) for fn in fns])

print len(T), 'total'
T.cut(T.brick_primary)
print len(T), 'primary'

T.t0 = np.array([t[0] for t in T.type])

P = T[T.t0 == 'P']
S = T[T.t0 == 'S']
E = T[T.t0 == 'E']
D = T[T.t0 == 'D']
C = T[T.t0 == 'C']

print len(P), 'PSF'
print len(S), 'Simple'
print len(E), 'Exp'
print len(D), 'Dev'
print len(C), 'Comp'

plt.clf()
plt.hist(np.clip(C.fracdev, -0.2, 1.2), range=(-0.1,1.1), bins=50)
plt.xlabel('FracDev')
plt.title('Composite galaxies')
plt.savefig('fracdev.png')

print 'Checking finite-ness of shapes'
assert(np.all(np.isfinite(E.shapeexp_r)))
assert(np.all(np.isfinite(E.shapeexp_e1)))
assert(np.all(np.isfinite(E.shapeexp_e2)))

assert(np.all(np.isfinite(D.shapedev_r)))
assert(np.all(np.isfinite(D.shapedev_e1)))
assert(np.all(np.isfinite(D.shapedev_e2)))

assert(np.all(np.isfinite(C.shapeexp_r)))
assert(np.all(np.isfinite(C.shapeexp_e1)))
assert(np.all(np.isfinite(C.shapeexp_e2)))
assert(np.all(np.isfinite(C.shapedev_r)))
assert(np.all(np.isfinite(C.shapedev_e1)))
assert(np.all(np.isfinite(C.shapedev_e2)))

assert(np.all(E.shapeexp_r > 0.))
assert(np.all(D.shapedev_r > 0.))
assert(np.all(C.shapeexp_r > 0.))
assert(np.all(C.shapedev_r > 0.))

print 'Checking flux distributions'
plt.clf()
lo,hi = -1, 20
plt.hist(np.clip(T.decam_flux[:,1], lo, hi), range=(lo,hi), bins=50,
         histtype='step', color='g')
plt.hist(np.clip(T.decam_flux[:,2], lo, hi), range=(lo,hi), bins=50,
         histtype='step', color='r')
plt.hist(np.clip(T.decam_flux[:,4], lo, hi), range=(lo,hi), bins=50,
         histtype='step', color='m')
plt.savefig('flux.png')

# At least one flux should be positive...
I = np.flatnonzero(np.all(T.decam_flux <= 0, axis=1))
print 'Sources with all fluxes not positive:', len(I)
T[I].writeto('neg.fits')
assert(np.all(np.any(T.decam_flux > 0, axis=1)))

# Match distances
I,J,d = match_radec(T.ra, T.dec, T.ra, T.dec, 10./3600., notself=True)
K = np.flatnonzero(I < J)
I,J,d = I[K],J[K],d[K]

plt.clf()
plt.hist(d*3600., range=(0,10), bins=50)
plt.xlabel('Arcsec between pairs')
plt.title('COSMOS dr2i -- close pairs')
plt.savefig('dists.png')

I,J,d = match_radec(T.ra, T.dec, T.ra, T.dec, 1./3600., notself=True)
K = np.flatnonzero(I < J)
I,J,d = I[K],J[K],d[K]

print 'Fracflux for nearby pairs:'
B = np.array([1,2,4])
for i,j in zip(I,J):
    print T.decam_fracflux[i,B], T.decam_fracflux[j,B]
    

print 'r-band fluxes and fracfluxes:'
b = 2
for i,j in zip(I,J):
    print 'fluxes', T.decam_flux[i,b], T.decam_flux[j, b], 'fracs', T.decam_fracflux[i,b], T.decam_fracflux[j,b]
