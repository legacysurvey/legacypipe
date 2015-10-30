from astrometry.util.fits import *
import numpy as np

from legacypipe.common import *

decals = Decals()
C = decals.get_ccds()
print len(C), 'CCDs'

C.cut(np.hypot(C.ra_bore - 150, C.dec_bore - 2.2) < 1.)
print len(C), 'CCDs on COSMOS'

bands = 'grz'
C.cut(np.array([f in bands for f in C.filter]))
print len(C), 'in', bands

C.cut(C.exptime >= 50.)
print len(C), 'with exptime >= 50 sec'

C.cut(decals.photometric_ccds(C))
print len(C), 'photometric'

nil,I = np.unique(C.expnum, return_index=True)
E = C[I]
print len(E), 'exposures'

print 'Exptime', E.exptime.min(), E.exptime.max()

# Target depths (90th percentile)
target = dict(g=24.0, r=23.4, z=22.5)

E.sig1 = []
E.psfnorm = []
E.galnorm = []

for i in range(len(E)):
    im = decals.get_image_object(E[i])
    tim = im.get_tractor_image(pixPsf=True, splinesky=True)
    E.sig1.append(tim.sig1)
    E.psfnorm.append(tim.psfnorm)
    E.galnorm.append(tim.galnorm)

E.writeto('cosmos-exposures.fits')
    

for band in bands:
    B = E[E.filter == band]
    print len(B), 'exposures in', band, 'band'

    Nsigma = 5.
    sig = NanoMaggies.magToNanomaggies(target[band]) / Nsigma
    targetiv = 1./sig**2

    detiv = 0.
    for i in range(len(B)):
        im = decals.get_image_object(B[i])
        tim = im.get_tractor_image(pixPsf=True, splinesky=True)
        thisdetiv = 1. / (tim.sig1 / tim.galnorm)**2
        detiv += thisdetiv
        print 'this detiv', thisdetiv, 'vs target', targetiv, '(fraction %.2f)' % (thisdetiv / targetiv)
        if detiv > targetiv:
            break


    
