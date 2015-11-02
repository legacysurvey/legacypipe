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

C.cut(np.lexsort((C.expnum, C.filter)))

efn = 'cosmos-exposures.fits'
if not os.path.exists(efn):
    nil,I = np.unique(C.expnum, return_index=True)
    E = C[I]
    print len(E), 'exposures'

    E.sig1    = np.zeros(len(E), np.float32)
    E.psfnorm = np.zeros(len(E), np.float32)
    E.galnorm = np.zeros(len(E), np.float32)
    for i in range(len(E)):
        im = decals.get_image_object(E[i])
        tim = im.get_tractor_image(pixPsf=True, splinesky=True)
        E.sig1[i] = tim.sig1
        E.psfnorm[i] = tim.psfnorm
        E.galnorm[i] = tim.galnorm
    E.writeto(efn)

else:
    E = fits_table(efn)

# Target depths (90th percentile), for 5-sigma galaxy profile
target = dict(g=24.0, r=23.4, z=22.5)

isdecals = np.array([p == '2014B-0404' for p in E.propid])
print 'Is DECaLS:', np.unique(isdecals), len(isdecals), len(E)
# Lexsort doesn't seem to work with only a single boolean column; add dumb arange
E.cut(np.lexsort((np.arange(len(E)), E.filter, np.logical_not(isdecals))))

sets = []
for iset in xrange(100):
    print '-------------------------------------------'
    thisset = []
    addnoise = []
    for band in bands:
        gotband = False

        B = E[E.filter == band]
        print len(B), 'exposures in', band, 'band'

        Nsigma = 5.
        sig = NanoMaggies.magToNanomaggies(target[band]) / Nsigma
        targetiv = 1./sig**2

        maxfrac = 0.34
        detiv = 0.
        for i in range(len(B)):
            exp = B[i]
            print 'Image', exp.expnum, 'propid', exp.propid, 'exptime', exp.exptime, 'seeing', exp.seeing
            thisdetiv = 1. / (exp.sig1 / exp.galnorm)**2
            print '  detiv', thisdetiv, '= fraction of target: %.2f' % (thisdetiv / targetiv)

            maxiv = maxfrac * targetiv
            if thisdetiv > maxiv:
                #addnoise = 
                targetsig1 = exp.sig1 * np.sqrt(thisdetiv / maxiv)
                addsig1 = np.sqrt(max(0, targetsig1**2 - exp.sig1**2))
                addnoise.append(addsig1)
                newiv = 1. / (np.hypot(exp.sig1, addsig1) / exp.galnorm)**2
                #print 'Adding', addsig1, 'we will get detiv', newiv, 'vs', maxiv
                print '  adding factor %.2f more noise' % (addsig1/exp.sig1)
                thisdetiv = maxiv
            else:
                addnoise.append(0.)

            #thisdetiv = min(thisdetiv, maxfrac * targetiv)
            detiv += thisdetiv
            thisset.append(exp.expnum)
            if detiv > targetiv:
                gotband = True
                break
        if not gotband:
            break
    if not gotband:
        break

    Eset = E[np.array([expnum in thisset for expnum in E.expnum])]
    print 'Cut to', len(E), 'exposures in set'

    Eset.addnoise = np.array(addnoise)
    #sets.append(thisset)
    #sets.append(Eset)

    expnoise = dict(zip(Eset.expnum, Eset.addnoise))
    Cset = C[np.array([expnum in thisset for expnum in C.expnum])]
    Cset.addnoise = np.array([expnoise[e] for e in Cset.expnum]).astype(np.float32)
    Cset.camera = np.array([c + '+noise' for c in Cset.camera])
    sets.append(Cset)
    
    E.cut(np.array([expnum not in thisset for expnum in E.expnum]))
    print 'Cut to', len(E), 'remaining exposures'


print 'Got', len(sets), 'sets of exposures'

for i,C in enumerate(sets):
    C.writeto('cosmos-ccds-sub%i.fits' % i)
    C.subset = np.array([i] * len(C)).astype(np.uint8)
    
C = merge_tables(sets)
C.writeto('cosmos-ccds.fits')

#for i,E in enumerate(sets):
#    E.writeto('cosmos-subset-%i.fits' % i)

